#if defined(_WIN32)

#include "core/direct_storage_engine.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>

namespace ninfer::core {

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

void DirectStorageEngine::recreate_queue() {
    if (!ds_factory_ || !d3d12_device_) { return; }
    ds_queue_.Reset();
    DSTORAGE_QUEUE_DESC queue_desc{};
    queue_desc.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
    queue_desc.Capacity   = DSTORAGE_MAX_QUEUE_CAPACITY;
    queue_desc.Priority   = DSTORAGE_PRIORITY_NORMAL;
    queue_desc.Name       = "NInferPromptCacheQueue";
    queue_desc.Device     = d3d12_device_.Get();
    ds_factory_->CreateQueue(&queue_desc, IID_PPV_ARGS(&ds_queue_));
}

bool DirectStorageEngine::allocate_staging_vram(std::size_t required_bytes) {
    if (staging_resource_ && current_capacity_ >= required_bytes) {
        return true;
    }

    release_staging();

    // Check if D3D12 device was removed by driver and re-initialize if needed
    if (d3d12_device_) {
        const HRESULT remove_reason = d3d12_device_->GetDeviceRemovedReason();
        if (FAILED(remove_reason)) {
            std::cerr << "[warn] ninfer: [directstorage] D3D12 device was removed (reason=0x"
                      << std::hex << remove_reason << std::dec << "), re-initializing...\n";
            cleanup();
            if (!init_d3d12_and_fence() || !init_direct_storage()) {
                cleanup();
                return false;
            }
        }
    }

    // Round up to 64KB placement alignment
    const std::size_t aligned_bytes = (required_bytes + 65535ULL) & ~65535ULL;

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
        std::cerr << "[error] ninfer: [directstorage] failed to allocate dynamic D3D12 staging buffer ("
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

void DirectStorageEngine::release_staging() noexcept {
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    // 1. Ensure D3D12 / DirectStorage queue has 100% finished retiring all DMA requests
    if (d3d12_fence_ && fence_value_ > 0) {
        if (d3d12_fence_->GetCompletedValue() < fence_value_) {
            HANDLE event = CreateEventEx(nullptr, nullptr, 0, EVENT_ALL_ACCESS);
            if (event) {
                if (SUCCEEDED(d3d12_fence_->SetEventOnCompletion(fence_value_, event))) {
                    WaitForSingleObject(event, 5000); // 5s safety timeout
                }
                CloseHandle(event);
            }
        }
    }

    // 2. Safely tear down CUDA external memory mappings
    if (cuda_ext_mem_) {
        cudaDestroyExternalMemory(cuda_ext_mem_);
        cuda_ext_mem_ = nullptr;
    }
    d_staging_ptr_ = nullptr;

    // 3. Safely reset D3D12 COM resources
    staging_resource_.Reset();
    pool_file_.Reset();
    manifest_file_.Reset();
    current_capacity_ = 0;
}

bool DirectStorageEngine::restore_snapshot_payload(const std::filesystem::path& path,
                                                  std::uint64_t file_offset_4k_aligned,
                                                  std::uint64_t payload_bytes,
                                                  cudaStream_t stream,
                                                  void*& out_d_staging_ptr) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_ || payload_bytes == 0) {
        return false;
    }

    // Dynamically allocate the exact staging capacity in VRAM with zero permanent reservation
    if (!allocate_staging_vram(payload_bytes)) {
        return false;
    }

    manifest_file_.Reset();
    HRESULT hr = ds_factory_->OpenFile(path.wstring().c_str(), IID_PPV_ARGS(&manifest_file_));
    if (FAILED(hr) || !manifest_file_) {
        release_staging();
        return false;
    }

    constexpr UINT32 kChunkSize = 32 * 1024 * 1024; // 32 MiB staging chunk
    UINT64 bytes_remaining = payload_bytes;
    UINT64 curr_file_offset = file_offset_4k_aligned;
    UINT64 curr_dst_offset = 0;

    while (bytes_remaining > 0) {
        const UINT32 chunk = static_cast<UINT32>(std::min<UINT64>(bytes_remaining, kChunkSize));

        DSTORAGE_REQUEST request{};
        request.Options.SourceType        = DSTORAGE_REQUEST_SOURCE_FILE;
        request.Options.DestinationType   = DSTORAGE_REQUEST_DESTINATION_BUFFER;
        request.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT_NONE;

        request.Source.File.Source        = manifest_file_.Get();
        request.Source.File.Offset        = curr_file_offset;
        request.Source.File.Size          = chunk;
        request.UncompressedSize          = chunk;

        request.Destination.Buffer.Resource = staging_resource_.Get();
        request.Destination.Buffer.Offset   = curr_dst_offset;
        request.Destination.Buffer.Size     = chunk;

        ds_queue_->EnqueueRequest(&request);

        curr_file_offset += chunk;
        curr_dst_offset  += chunk;
        bytes_remaining  -= chunk;
    }

    const std::uint64_t signal_val = ++fence_value_;
    ds_queue_->EnqueueSignal(d3d12_fence_.Get(), signal_val);
    ds_queue_->Submit();

    // Enqueue non-blocking hardware fence wait directly onto caller's CUDA stream
    cudaExternalSemaphoreWaitParams wait_params{};
    wait_params.flags              = 0;
    wait_params.params.fence.value = signal_val;
    cudaError_t cuda_err = cudaWaitExternalSemaphoresAsync(&cuda_fence_sem_, &wait_params, 1, stream);
    if (cuda_err != cudaSuccess) {
        release_staging();
        return false;
    }

    out_d_staging_ptr = d_staging_ptr_;
    return true;
}

bool DirectStorageEngine::restore_snapshot_cow(const std::filesystem::path& pool_path,
                                              std::span<const PageRestoreEntry> text_pages,
                                              const std::filesystem::path& manifest_path,
                                              std::uint64_t manifest_payload_offset,
                                              std::uint64_t manifest_payload_bytes,
                                              cudaStream_t stream,
                                              void*& out_d_staging_ptr,
                                              std::size_t& out_text_bytes) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_) {
        return false;
    }

    std::size_t total_text_bytes = 0;
    for (const auto& p : text_pages) {
        total_text_bytes += p.page_bytes;
    }

    const std::size_t total_staging_bytes = total_text_bytes + static_cast<std::size_t>(manifest_payload_bytes);
    if (total_staging_bytes == 0) {
        return false;
    }

    if (!allocate_staging_vram(total_staging_bytes)) {
        return false;
    }

    // 1. Enqueue Text KV pages from shared pool_data.ninfer_pages
    if (!text_pages.empty()) {
        pool_file_.Reset();
        HRESULT hr = ds_factory_->OpenFile(pool_path.wstring().c_str(), IID_PPV_ARGS(&pool_file_));
        if (FAILED(hr) || !pool_file_) {
            release_staging();
            return false;
        }

        std::uint64_t curr_dst_offset = 0;
        for (const auto& p : text_pages) {
            DSTORAGE_REQUEST request{};
            request.Options.SourceType        = DSTORAGE_REQUEST_SOURCE_FILE;
            request.Options.DestinationType   = DSTORAGE_REQUEST_DESTINATION_BUFFER;
            request.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT_NONE;

            request.Source.File.Source        = pool_file_.Get();
            request.Source.File.Offset        = p.file_offset;
            request.Source.File.Size          = p.page_bytes;
            request.UncompressedSize          = p.page_bytes;

            request.Destination.Buffer.Resource = staging_resource_.Get();
            request.Destination.Buffer.Offset   = curr_dst_offset;
            request.Destination.Buffer.Size     = p.page_bytes;

            ds_queue_->EnqueueRequest(&request);
            curr_dst_offset += p.page_bytes;
        }
    }

    // 2. Enqueue Manifest Payload (GDN state, MTP KV, tail hidden) from .ninfer_manifest
    if (manifest_payload_bytes > 0) {
        manifest_file_.Reset();
        HRESULT hr = ds_factory_->OpenFile(manifest_path.wstring().c_str(), IID_PPV_ARGS(&manifest_file_));
        if (FAILED(hr) || !manifest_file_) {
            release_staging();
            return false;
        }

        constexpr UINT32 kChunkSize = 32 * 1024 * 1024; // 32 MiB chunk
        UINT64 bytes_remaining = manifest_payload_bytes;
        UINT64 curr_file_offset = manifest_payload_offset;
        UINT64 curr_dst_offset = total_text_bytes;

        while (bytes_remaining > 0) {
            const UINT32 chunk = static_cast<UINT32>(std::min<UINT64>(bytes_remaining, kChunkSize));

            DSTORAGE_REQUEST request{};
            request.Options.SourceType        = DSTORAGE_REQUEST_SOURCE_FILE;
            request.Options.DestinationType   = DSTORAGE_REQUEST_DESTINATION_BUFFER;
            request.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT_NONE;

            request.Source.File.Source        = manifest_file_.Get();
            request.Source.File.Offset        = curr_file_offset;
            request.Source.File.Size          = chunk;
            request.UncompressedSize          = chunk;

            request.Destination.Buffer.Resource = staging_resource_.Get();
            request.Destination.Buffer.Offset   = curr_dst_offset;
            request.Destination.Buffer.Size     = chunk;

            ds_queue_->EnqueueRequest(&request);

            curr_file_offset += chunk;
            curr_dst_offset  += chunk;
            bytes_remaining  -= chunk;
        }
    }

    const std::uint64_t signal_val = ++fence_value_;
    ds_queue_->EnqueueSignal(d3d12_fence_.Get(), signal_val);
    ds_queue_->Submit();

    // Enqueue non-blocking hardware fence wait directly onto caller's CUDA stream
    cudaExternalSemaphoreWaitParams wait_params{};
    wait_params.flags              = 0;
    wait_params.params.fence.value = signal_val;
    cudaError_t cuda_err = cudaWaitExternalSemaphoresAsync(&cuda_fence_sem_, &wait_params, 1, stream);
    if (cuda_err != cudaSuccess) {
        release_staging();
        return false;
    }

    out_d_staging_ptr = d_staging_ptr_;
    out_text_bytes    = total_text_bytes;
    return true;
}

void DirectStorageEngine::cleanup() {
    release_staging();
    if (cuda_fence_sem_) {
        cudaDestroyExternalSemaphore(cuda_fence_sem_);
        cuda_fence_sem_ = nullptr;
    }
    ds_queue_.Reset();
    ds_factory_.Reset();
    d3d12_fence_.Reset();
    d3d12_device_.Reset();
    dxgi_factory_.Reset();
    initialized_ = false;
}

} // namespace ninfer::core

#endif // _WIN32
