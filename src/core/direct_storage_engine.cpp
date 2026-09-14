#if defined(_WIN32)

#include "core/direct_storage_engine.h"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::core {

namespace {

// Largest single DirectStorage request; matches the queue staging buffer size below.
constexpr std::uint64_t kMaxRequestBytes = 32ULL * 1024 * 1024;

} // namespace

DirectStorageEngine& DirectStorageEngine::instance() {
    static DirectStorageEngine engine;
    return engine;
}

DirectStorageEngine::DirectStorageEngine() {
    if (!init_d3d12_and_fence()) {
        cleanup();
        return;
    }
    if (!init_direct_storage()) {
        cleanup();
        return;
    }
    initialized_ = true;
    std::cout << "[info] ninfer: [directstorage] initialized DirectStorage 1.3.0\n" << std::flush;
}

DirectStorageEngine::~DirectStorageEngine() {
    cleanup();
}

bool DirectStorageEngine::init_d3d12_and_fence() {
    int current_device = 0;
    cudaError_t cuda_err = cudaGetDevice(&current_device);
    if (cuda_err != cudaSuccess) { return false; }

    cudaDeviceProp prop{};
    cuda_err = cudaGetDeviceProperties(&prop, current_device);
    if (cuda_err != cudaSuccess) { return false; }

    HRESULT hr = CreateDXGIFactory2(0, IID_PPV_ARGS(&dxgi_factory_));
    if (FAILED(hr)) { return false; }

    ComPtr<IDXGIAdapter1> matched_adapter;
    ComPtr<IDXGIAdapter1> adapter;
    for (UINT i = 0; dxgi_factory_->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 desc{};
        adapter->GetDesc1(&desc);
        if (std::memcmp(&desc.AdapterLuid, prop.luid, sizeof(LUID)) == 0) {
            matched_adapter = adapter;
            break;
        }
    }
    if (!matched_adapter) {
        dxgi_factory_->EnumAdapters1(0, &matched_adapter);
    }
    if (!matched_adapter) { return false; }

    hr = D3D12CreateDevice(matched_adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&d3d12_device_));
    if (FAILED(hr)) { return false; }

    // Create D3D12 shared fence for DirectStorage to CUDA stream synchronization
    hr = d3d12_device_->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&d3d12_fence_));
    if (FAILED(hr)) { return false; }
    fence_value_ = 0;

    HANDLE fence_shared_handle = nullptr;
    hr = d3d12_device_->CreateSharedHandle(d3d12_fence_.Get(), nullptr, GENERIC_ALL, nullptr, &fence_shared_handle);
    if (FAILED(hr) || !fence_shared_handle) { return false; }

    cudaExternalSemaphoreHandleDesc sem_desc{};
    sem_desc.type                = cudaExternalSemaphoreHandleTypeD3D12Fence;
    sem_desc.handle.win32.handle = fence_shared_handle;
    sem_desc.flags               = 0;
    cuda_err = cudaImportExternalSemaphore(&cuda_fence_sem_, &sem_desc);
    CloseHandle(fence_shared_handle);
    if (cuda_err != cudaSuccess) { return false; }

    return true;
}

bool DirectStorageEngine::init_direct_storage() {
    HRESULT hr = DStorageGetFactory(IID_PPV_ARGS(&ds_factory_));
    if (FAILED(hr) || !ds_factory_) { return false; }

    ds_factory_->SetDebugFlags(DSTORAGE_DEBUG_SHOW_ERRORS);
    ds_factory_->SetStagingBufferSize(DSTORAGE_STAGING_BUFFER_SIZE_32MB);

    DSTORAGE_QUEUE_DESC queue_desc{};
    queue_desc.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
    queue_desc.Capacity   = DSTORAGE_MAX_QUEUE_CAPACITY;
    queue_desc.Priority   = DSTORAGE_PRIORITY_NORMAL;
    queue_desc.Name       = "NInferPromptCacheQueue";
    queue_desc.Device     = d3d12_device_.Get();

    hr = ds_factory_->CreateQueue(&queue_desc, IID_PPV_ARGS(&ds_queue_));
    if (FAILED(hr) || !ds_queue_) { return false; }

    return true;
}

bool DirectStorageEngine::ensure_device_alive() {
    if (initialized_ && d3d12_device_) {
        const HRESULT remove_reason = d3d12_device_->GetDeviceRemovedReason();
        if (SUCCEEDED(remove_reason)) {
            return true;
        }
        std::cerr << "[warn] ninfer: [directstorage] D3D12 device was removed (reason=0x"
                  << std::hex << remove_reason << std::dec << "), re-initializing\n" << std::flush;
    }
    cleanup();
    if (!init_d3d12_and_fence() || !init_direct_storage()) {
        std::cerr << "[warn] ninfer: [directstorage] re-initialization failed, DirectStorage disabled\n"
                  << std::flush;
        cleanup();
        return false;
    }
    initialized_ = true;
    return true;
}

void DirectStorageEngine::sync_last_consumer() noexcept {
    if (last_consumer_stream_ == nullptr) {
        return;
    }
    if (!consumer_event_) {
        cudaEventCreateWithFlags(&consumer_event_, cudaEventDisableTiming);
    }
    cudaEventRecord(consumer_event_, last_consumer_stream_);
    cudaEventSynchronize(consumer_event_);
    last_consumer_stream_ = nullptr;
}

void DirectStorageEngine::wait_fence_on_cpu() noexcept {
    if (!d3d12_fence_ || fence_value_ == 0 || d3d12_fence_->GetCompletedValue() >= fence_value_) {
        return;
    }
    HANDLE event = CreateEventEx(nullptr, nullptr, 0, EVENT_ALL_ACCESS);
    if (event) {
        if (SUCCEEDED(d3d12_fence_->SetEventOnCompletion(fence_value_, event))) {
            WaitForSingleObject(event, 5000); // 5s safety timeout
        }
        CloseHandle(event);
    }
}

bool DirectStorageEngine::ensure_staging_vram() {
    if (staging_resource_) {
        return true;
    }

    // Round up to 64KB placement alignment
    const std::size_t aligned_bytes = (kStagingCapacityBytes + 65535ULL) & ~65535ULL;

    D3D12_HEAP_PROPERTIES heap_props{};
    heap_props.Type                 = D3D12_HEAP_TYPE_DEFAULT;
    heap_props.CPUPageProperty      = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heap_props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;

    D3D12_RESOURCE_DESC res_desc{};
    res_desc.Dimension          = D3D12_RESOURCE_DIMENSION_BUFFER;
    res_desc.Alignment          = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT; // 64KB
    res_desc.Width              = aligned_bytes;
    res_desc.Height             = 1;
    res_desc.DepthOrArraySize   = 1;
    res_desc.MipLevels          = 1;
    res_desc.Format             = DXGI_FORMAT_UNKNOWN;
    res_desc.SampleDesc.Count   = 1;
    res_desc.SampleDesc.Quality = 0;
    res_desc.Layout             = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    res_desc.Flags              = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    HRESULT hr = d3d12_device_->CreateCommittedResource(
        &heap_props,
        D3D12_HEAP_FLAG_SHARED,
        &res_desc,
        D3D12_RESOURCE_STATE_COMMON,
        nullptr,
        IID_PPV_ARGS(&staging_resource_)
    );
    if (FAILED(hr) || !staging_resource_) {
        std::cerr << "[error] ninfer: [directstorage] failed to allocate D3D12 staging buffer ("
                  << (aligned_bytes / (1024 * 1024)) << " MB, HRESULT=0x" << std::hex << hr << std::dec << ")\n";
        return false;
    }

    // Export shared NT handle and import into CUDA external memory
    HANDLE shared_resource_handle = nullptr;
    hr = d3d12_device_->CreateSharedHandle(staging_resource_.Get(), nullptr, GENERIC_ALL, nullptr, &shared_resource_handle);
    if (FAILED(hr) || !shared_resource_handle) {
        release_staging();
        return false;
    }

    cudaExternalMemoryHandleDesc mem_desc{};
    mem_desc.type                = cudaExternalMemoryHandleTypeD3D12Resource;
    mem_desc.handle.win32.handle = shared_resource_handle;
    mem_desc.size                = aligned_bytes;
    mem_desc.flags               = cudaExternalMemoryDedicated;
    cudaError_t cuda_err = cudaImportExternalMemory(&cuda_ext_mem_, &mem_desc);
    CloseHandle(shared_resource_handle);
    if (cuda_err != cudaSuccess) {
        release_staging();
        return false;
    }

    cudaExternalMemoryBufferDesc buf_desc{};
    buf_desc.offset = 0;
    buf_desc.size   = aligned_bytes;
    buf_desc.flags  = 0;
    cuda_err = cudaExternalMemoryGetMappedBuffer(&d_staging_ptr_, cuda_ext_mem_, &buf_desc);
    if (cuda_err != cudaSuccess || !d_staging_ptr_) {
        release_staging();
        return false;
    }

    current_capacity_ = aligned_bytes;
    return true;
}

// Only for teardown (process exit, device removal, failed allocation). Never call it
// between restores: see kStagingCapacityBytes.
void DirectStorageEngine::release_staging() noexcept {
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    sync_last_consumer();
    wait_fence_on_cpu();

    if (cuda_ext_mem_) {
        cudaDestroyExternalMemory(cuda_ext_mem_);
        cuda_ext_mem_ = nullptr;
    }
    d_staging_ptr_ = nullptr;

    staging_resource_.Reset();
    current_capacity_ = 0;
}

bool DirectStorageEngine::restore_segments(const std::filesystem::path& pool_path,
                                          const std::filesystem::path& manifest_path,
                                          std::span<const DirectStorageSegment> segments,
                                          cudaStream_t stream,
                                          const DirectStorageBatchConsumer& consumer) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_ || segments.empty()) {
        return false;
    }

    // The previous restore's consumer may still be reading the staging buffer.
    sync_last_consumer();

    if (!ensure_device_alive() || !ensure_staging_vram()) {
        return false;
    }

    bool need_pool = false;
    bool need_manifest = false;
    for (const auto& seg : segments) {
        if (seg.bytes == 0 || seg.bytes > current_capacity_) {
            return false;
        }
        (seg.source == DirectStorageSource::Pool ? need_pool : need_manifest) = true;
    }

    ComPtr<IDStorageFile> pool_file;
    ComPtr<IDStorageFile> manifest_file;
    if (need_pool) {
        HRESULT hr = ds_factory_->OpenFile(pool_path.wstring().c_str(), IID_PPV_ARGS(&pool_file));
        if (FAILED(hr) || !pool_file) { return false; }
    }
    if (need_manifest) {
        HRESULT hr = ds_factory_->OpenFile(manifest_path.wstring().c_str(), IID_PPV_ARGS(&manifest_file));
        if (FAILED(hr) || !manifest_file) { return false; }
    }

    // Drop failures recorded before this restore; the count resets on each retrieval.
    DSTORAGE_ERROR_RECORD stale_record{};
    ds_queue_->RetrieveErrorRecord(&stale_record);

    const std::byte* staging = static_cast<const std::byte*>(d_staging_ptr_);
    std::vector<std::size_t> offsets;
    offsets.reserve(std::min<std::size_t>(segments.size(), 4096));

    bool ok = true;
    std::size_t first = 0;
    while (first < segments.size()) {
        // Pack consecutive whole segments into one staging batch.
        offsets.clear();
        std::size_t batch_bytes = 0;
        std::size_t end = first;
        while (end < segments.size() && batch_bytes + segments[end].bytes <= current_capacity_) {
            offsets.push_back(batch_bytes);
            batch_bytes += static_cast<std::size_t>(segments[end].bytes);
            ++end;
        }

        // Earlier batch consumers must be done before this batch overwrites staging.
        sync_last_consumer();

        for (std::size_t i = first; i < end; ++i) {
            const auto& seg = segments[i];
            IDStorageFile* file = seg.source == DirectStorageSource::Pool ? pool_file.Get() : manifest_file.Get();
            std::uint64_t done = 0;
            while (done < seg.bytes) {
                const UINT32 chunk = static_cast<UINT32>(std::min<std::uint64_t>(seg.bytes - done, kMaxRequestBytes));

                DSTORAGE_REQUEST request{};
                request.Options.SourceType        = DSTORAGE_REQUEST_SOURCE_FILE;
                request.Options.DestinationType   = DSTORAGE_REQUEST_DESTINATION_BUFFER;
                request.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT_NONE;

                request.Source.File.Source = file;
                request.Source.File.Offset = seg.file_offset + done;
                request.Source.File.Size   = chunk;
                request.UncompressedSize   = chunk;

                request.Destination.Buffer.Resource = staging_resource_.Get();
                request.Destination.Buffer.Offset   = offsets[i - first] + done;
                request.Destination.Buffer.Size     = chunk;

                ds_queue_->EnqueueRequest(&request);
                done += chunk;
            }
        }

        const std::uint64_t signal_val = ++fence_value_;
        ds_queue_->EnqueueSignal(d3d12_fence_.Get(), signal_val);
        ds_queue_->Submit();

        // Non-blocking fence wait queued on the consumer's CUDA stream
        cudaExternalSemaphoreWaitParams wait_params{};
        wait_params.flags              = 0;
        wait_params.params.fence.value = signal_val;
        if (cudaWaitExternalSemaphoresAsync(&cuda_fence_sem_, &wait_params, 1, stream) != cudaSuccess) {
            ok = false;
            break;
        }

        last_consumer_stream_ = stream;
        if (!consumer(staging, first, std::span<const std::size_t>(offsets.data(), offsets.size()))) {
            ok = false;
            break;
        }
        first = end;
    }

    // The files must outlive every request that references them.
    wait_fence_on_cpu();

    DSTORAGE_ERROR_RECORD record{};
    ds_queue_->RetrieveErrorRecord(&record);
    if (record.FailureCount > 0) {
        std::cerr << "[warn] ninfer: [directstorage] " << record.FailureCount
                  << " read request(s) failed (HRESULT=0x" << std::hex << record.FirstFailure.HResult
                  << std::dec << ")\n" << std::flush;
        ok = false;
    }
    return ok;
}

void DirectStorageEngine::cleanup() {
    release_staging();
    last_consumer_stream_ = nullptr;
    if (cuda_fence_sem_) {
        cudaDestroyExternalSemaphore(cuda_fence_sem_);
        cuda_fence_sem_ = nullptr;
    }
    if (consumer_event_) {
        cudaEventDestroy(consumer_event_);
        consumer_event_ = nullptr;
    }
    ds_queue_.Reset();
    ds_factory_.Reset();
    d3d12_fence_.Reset();
    fence_value_ = 0;
    d3d12_device_.Reset();
    dxgi_factory_.Reset();
    initialized_ = false;
}

} // namespace ninfer::core

#endif // _WIN32
