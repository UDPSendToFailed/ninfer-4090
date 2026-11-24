#include "core/arena.h"

#include <cuda_runtime.h>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#endif

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace ninfer {

namespace detail {

// Owns one D3D12 residency-locked heap and its CUDA import. The instance is released exactly once,
// with the arena that owns it; releasing the import before the heap keeps CUDA's mapping valid for
// the arena's whole lifetime.
struct D3D12Residency {
#if defined(_WIN32)
    Microsoft::WRL::ComPtr<IDXGIFactory4> factory;
    Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
    Microsoft::WRL::ComPtr<ID3D12Device3> device;
    Microsoft::WRL::ComPtr<ID3D12Resource> resource;
    Microsoft::WRL::ComPtr<ID3D12Fence> fence;
    cudaExternalMemory_t ext_mem = nullptr;
    std::uint64_t fence_value    = 1;
    std::mutex mutex;

    ~D3D12Residency();
    [[nodiscard]] bool ensure_resident(DWORD timeout_ms);
#else
    ~D3D12Residency() = default;
#endif
};

// Owns one pinned device-memory pool. The allocation itself is a cudaMallocFromPoolAsync
// allocation with cudaMemAllocationTypePinned, which WDDM counts exactly once and which the
// runtime does not place in the pageable cudaMalloc heap.
struct PinnedPool {
    cudaMemPool_t pool                 = nullptr;
    std::size_t bytes                  = 0;
    unsigned over_budget_streak        = 0;

    ~PinnedPool();
};

} // namespace detail

namespace {

__device__ unsigned g_residency_touch_sink = 0;

// Reads one byte per 128-byte sector across the allocation, forcing the driver to fault every
// page back into VRAM. The almost-never-taken branch keeps the loads observable to the compiler.
__global__ void residency_touch_kernel(const unsigned char* data, std::size_t bytes) {
    const std::size_t stride =
        static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x) * 128ULL;
    std::size_t offset =
        (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 128ULL;
    for (; offset < bytes; offset += stride) {
        const unsigned value = data[offset];
        if (value == 0xDEADBEEFu) { atomicAdd(&g_residency_touch_sink, 1u); }
    }
}

std::string cuda_error_message(const char* prefix, cudaError_t err) {
    return std::string(prefix) + ": " + cudaGetErrorName(err) + ": " + cudaGetErrorString(err);
}

void log_cuda_error(const char* op, cudaError_t err) noexcept {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA cleanup failed during %s: %s: %s\n", op, cudaGetErrorName(err),
                     cudaGetErrorString(err));
    }
}

bool is_power_of_two(std::size_t value) { return value != 0 && (value & (value - 1)) == 0; }

std::uintptr_t checked_add_uintptr(std::uintptr_t a, std::size_t b) {
    if (b > std::numeric_limits<std::uintptr_t>::max() - a) {
        throw std::overflow_error("arena address arithmetic overflow");
    }
    return a + b;
}

std::uintptr_t align_up_addr(std::uintptr_t addr, std::size_t align) {
    const std::size_t mask         = align - 1;
    const std::uintptr_t with_mask = checked_add_uintptr(addr, mask);
    return with_mask & ~static_cast<std::uintptr_t>(mask);
}

void free_device(void*& ptr) noexcept {
    if (ptr != nullptr) {
        log_cuda_error("cudaFree", cudaFree(ptr));
        ptr = nullptr;
    }
}

void free_pinned(void*& ptr) noexcept {
    if (ptr != nullptr) {
        log_cuda_error("cudaFreeHost", cudaFreeHost(ptr));
        ptr = nullptr;
    }
}

#if defined(_WIN32)
// Upper bound on how long we wait for WDDM to make the heap resident.
constexpr DWORD kResidencyWaitTimeoutMs = 10000;

// How often ensure_resident() may re-assert residency (pinned budget check or D3D12 re-assert).
constexpr auto kResidencyCheckInterval = std::chrono::seconds(1);

// Consecutive one-second over-budget samples before the watchdog warns, and before it gives up.
// The fatal threshold is deliberately far longer than a desktop application's VRAM spike: pinned
// arenas cannot be evicted by another process, so a short spike is pressure, not residency loss.
constexpr unsigned kOverBudgetWarningStreak = 5;
constexpr unsigned kOverBudgetFatalStreak   = 30;

static std::atomic<bool> g_wddm_residency_lock_enabled{false};

enum class D3D12AllocationOutcome {
    Success,
    InsufficientVram,
    Unavailable,
};

std::string hresult_message(const char* operation, HRESULT result) {
    return std::string(operation) + " failed (HRESULT " +
           std::to_string(static_cast<long>(result)) + ")";
}

D3D12AllocationOutcome allocate_d3d12_residency(std::size_t capacity_bytes, void*& out_ptr,
                                                std::shared_ptr<detail::D3D12Residency>& out,
                                                std::string& reason) {
    auto res = std::make_shared<detail::D3D12Residency>();

    if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&res->factory)))) {
        reason = "CreateDXGIFactory2 failed";
        return D3D12AllocationOutcome::Unavailable;
    }

    // Match DXGI Adapter with current CUDA Device LUID
    int device_id = 0;
    if (cudaGetDevice(&device_id) != cudaSuccess) {
        reason = "cudaGetDevice failed";
        return D3D12AllocationOutcome::Unavailable;
    }
    cudaDeviceProp props{};
    if (cudaGetDeviceProperties(&props, device_id) != cudaSuccess) {
        reason = "cudaGetDeviceProperties failed";
        return D3D12AllocationOutcome::Unavailable;
    }

    LUID cuda_luid{};
    std::memcpy(&cuda_luid, props.luid, sizeof(LUID));

    Microsoft::WRL::ComPtr<IDXGIAdapter1> cur_adapter;
    for (UINT i = 0; res->factory->EnumAdapters1(i, &cur_adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 desc{};
        cur_adapter->GetDesc1(&desc);
        if (desc.AdapterLuid.LowPart == cuda_luid.LowPart &&
            desc.AdapterLuid.HighPart == cuda_luid.HighPart) {
            res->adapter = cur_adapter;
            break;
        }
    }
    if (!res->adapter) {
        reason = "no DXGI adapter matches the CUDA device LUID";
        return D3D12AllocationOutcome::Unavailable;
    }

    Microsoft::WRL::ComPtr<ID3D12Device> base_device;
    if (FAILED(D3D12CreateDevice(res->adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                 IID_PPV_ARGS(&base_device)))) {
        reason = "D3D12CreateDevice failed";
        return D3D12AllocationOutcome::Unavailable;
    }
    if (FAILED(base_device.As(&res->device))) {
        reason = "the D3D12 device does not expose ID3D12Device3";
        return D3D12AllocationOutcome::Unavailable;
    }

    constexpr std::size_t kD3D12Alignment = 64 * 1024;
    const std::size_t aligned_size =
        ((capacity_bytes + kD3D12Alignment - 1) / kD3D12Alignment) * kD3D12Alignment;

    DXGI_ADAPTER_DESC1 desc1{};
    if (SUCCEEDED(res->adapter->GetDesc1(&desc1))) {
        // Enforce a minimum 512 MiB floor for DWM display scanout
        constexpr UINT64 kDwmHeadroom = 512ULL * 1024ULL * 1024ULL;
        const UINT64 max_usable = desc1.DedicatedVideoMemory > kDwmHeadroom
                                      ? desc1.DedicatedVideoMemory - kDwmHeadroom
                                      : 0;

        Microsoft::WRL::ComPtr<IDXGIAdapter3> adapter3;
        if (SUCCEEDED(res->adapter.As(&adapter3))) {
            DXGI_QUERY_VIDEO_MEMORY_INFO mem_info{};
            if (SUCCEEDED(
                    adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mem_info))) {
                if (mem_info.CurrentUsage + aligned_size > max_usable) {
                    reason = "the dedicated-VRAM budget cannot hold " +
                             std::to_string(aligned_size) + " bytes (current usage " +
                             std::to_string(mem_info.CurrentUsage) + ", usable " +
                             std::to_string(max_usable) + ", budget " +
                             std::to_string(mem_info.Budget) + ")";
                    return D3D12AllocationOutcome::InsufficientVram;
                }
            }
        }
    }

    D3D12_HEAP_PROPERTIES heap_properties = {};
    heap_properties.Type                  = D3D12_HEAP_TYPE_DEFAULT;
    heap_properties.CreationNodeMask      = 1;

    D3D12_RESOURCE_DESC resource_desc = {};
    resource_desc.Dimension          = D3D12_RESOURCE_DIMENSION_BUFFER;
    resource_desc.Width              = aligned_size;
    resource_desc.Height             = 1;
    resource_desc.DepthOrArraySize   = 1;
    resource_desc.MipLevels          = 1;
    resource_desc.Format             = DXGI_FORMAT_UNKNOWN;
    resource_desc.SampleDesc.Count   = 1;
    resource_desc.Layout             = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    resource_desc.Flags              = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    const HRESULT heap_result = res->device->CreateCommittedResource(
        &heap_properties, D3D12_HEAP_FLAG_SHARED, &resource_desc, D3D12_RESOURCE_STATE_COMMON,
        nullptr, IID_PPV_ARGS(&res->resource));
    if (FAILED(heap_result)) {
        reason = hresult_message("CreateCommittedResource", heap_result);
        return D3D12AllocationOutcome::InsufficientVram;
    }

    // 1. Set priority to MAXIMUM/CRITICAL (places NInfer above normal desktop apps)
    ID3D12Pageable* pageable = res->resource.Get();
    const D3D12_RESIDENCY_PRIORITY priority =
        static_cast<D3D12_RESIDENCY_PRIORITY>(0xc8000000); // D3D12_RESIDENCY_PRIORITY_MAXIMUM
    const HRESULT priority_result = res->device->SetResidencyPriority(1, &pageable, &priority);
    if (FAILED(priority_result)) {
        // Without MAXIMUM priority the heap is not protected from WDDM paging, so a strict arena
        // must not pretend it is residency-locked.
        reason = hresult_message("SetResidencyPriority(MAXIMUM)", priority_result);
        return D3D12AllocationOutcome::Unavailable;
    }

    // 2. Deny overbudget paging (forbid WDDM from paging this heap to GART / System RAM)
    const HRESULT fence_result =
        res->device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&res->fence));
    if (FAILED(fence_result)) {
        reason = hresult_message("CreateFence", fence_result);
        return D3D12AllocationOutcome::Unavailable;
    }
    const HRESULT resident_result = res->device->EnqueueMakeResident(
        D3D12_RESIDENCY_FLAG_DENY_OVERBUDGET, 1, &pageable, res->fence.Get(), 1);
    if (FAILED(resident_result)) {
        reason = hresult_message("EnqueueMakeResident(DENY_OVERBUDGET)", resident_result) +
                 ": the heap does not fit the current budget";
        return D3D12AllocationOutcome::InsufficientVram;
    }

    // EnqueueMakeResident completes asynchronously and signals the fence once the heap is
    // actually resident in device memory. Touching the heap before that point is undefined
    // behaviour: under desktop VRAM contention the pages are not yet backed, so weights
    // written into the mapping are silently lost and the model emits noise.
    if (res->fence->GetCompletedValue() < 1) {
        HANDLE residency_event = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (residency_event == nullptr) {
            reason = "CreateEventW failed while waiting for residency";
            return D3D12AllocationOutcome::Unavailable;
        }
        if (FAILED(res->fence->SetEventOnCompletion(1, residency_event))) {
            CloseHandle(residency_event);
            reason = "SetEventOnCompletion failed while waiting for residency";
            return D3D12AllocationOutcome::Unavailable;
        }
        const DWORD wait_result = WaitForSingleObject(residency_event, kResidencyWaitTimeoutMs);
        CloseHandle(residency_event);
        // A timeout means WDDM could not make the heap resident without going overbudget.
        if (wait_result != WAIT_OBJECT_0) {
            reason = "residency was not granted within " +
                     std::to_string(kResidencyWaitTimeoutMs) + " ms";
            return D3D12AllocationOutcome::InsufficientVram;
        }
    }

    // 3. Create shared handle for CUDA import
    HANDLE shared_handle = nullptr;
    const HRESULT handle_result = res->device->CreateSharedHandle(
        res->resource.Get(), nullptr, GENERIC_ALL, nullptr, &shared_handle);
    if (FAILED(handle_result)) {
        reason = hresult_message("CreateSharedHandle", handle_result);
        return D3D12AllocationOutcome::Unavailable;
    }

    // 4. Import into CUDA
    cudaExternalMemoryHandleDesc ext_desc = {};
    ext_desc.type = cudaExternalMemoryHandleTypeD3D12Resource;
    ext_desc.handle.win32.handle = shared_handle;
    ext_desc.size = aligned_size;
    ext_desc.flags = cudaExternalMemoryDedicated;

    cudaExternalMemory_t ext_mem = nullptr;
    const cudaError_t import_err = cudaImportExternalMemory(&ext_mem, &ext_desc);
    CloseHandle(shared_handle); // Safe to close immediately after import
    if (import_err != cudaSuccess) {
        reason = cuda_error_message("cudaImportExternalMemory failed", import_err);
        return D3D12AllocationOutcome::Unavailable;
    }

    res->ext_mem = ext_mem;

    cudaExternalMemoryBufferDesc buf_desc = {};
    buf_desc.offset = 0;
    buf_desc.size = capacity_bytes;
    buf_desc.flags = 0;

    void* dev_ptr = nullptr;
    const cudaError_t map_err = cudaExternalMemoryGetMappedBuffer(&dev_ptr, ext_mem, &buf_desc);
    if (map_err != cudaSuccess) {
        reason = cuda_error_message("cudaExternalMemoryGetMappedBuffer failed", map_err);
        return D3D12AllocationOutcome::Unavailable;
    }

    // Eagerly touch physical pages to commit them
    const cudaError_t fill_err = cudaMemset(dev_ptr, 0, capacity_bytes);
    if (fill_err != cudaSuccess) {
        reason = cuda_error_message("cudaMemset failed on the residency heap", fill_err);
        return D3D12AllocationOutcome::Unavailable;
    }

    // Verify the mapping actually round-trips before trusting 16 GiB of weights to it.
    // A D3D12 heap that imported cleanly can still fail to read back correctly, and the
    // failure mode downstream is silent output corruption rather than an error.
    {
        constexpr std::size_t kProbeBytes = 4096;
        const std::size_t probe = capacity_bytes < kProbeBytes ? capacity_bytes : kProbeBytes;
        std::vector<unsigned char> pattern(probe);
        for (std::size_t i = 0; i < probe; ++i) {
            pattern[i] = static_cast<unsigned char>((i * 31 + 7) & 0xFF);
        }
        std::vector<unsigned char> readback(probe, 0);

        bool coherent = false;
        const std::size_t tail_offset = capacity_bytes > probe ? capacity_bytes - probe : 0;
        if (cudaMemcpy(dev_ptr, pattern.data(), probe, cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(static_cast<unsigned char*>(dev_ptr) + tail_offset, pattern.data(), probe,
                       cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaDeviceSynchronize() == cudaSuccess &&
            cudaMemcpy(readback.data(), static_cast<unsigned char*>(dev_ptr) + tail_offset, probe,
                       cudaMemcpyDeviceToHost) == cudaSuccess) {
            coherent = std::memcmp(pattern.data(), readback.data(), probe) == 0;
        }

        if (!coherent) {
            reason = "the imported D3D12 heap failed read-back verification";
            cudaGetLastError();
            return D3D12AllocationOutcome::Unavailable;
        }
        cudaMemset(dev_ptr, 0, capacity_bytes);
    }

    // The final zero-fill above is enqueued on the legacy default stream, which does not order
    // against non-blocking streams such as DeviceContext::load_stream. Wait for it to land so
    // stream-ordered weight copies cannot be wiped by the fill (silent all-zero weights).
    const cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        reason = cuda_error_message("cudaDeviceSynchronize failed after the residency fill",
                                    sync_err);
        return D3D12AllocationOutcome::Unavailable;
    }

    out_ptr = dev_ptr;
    out     = std::move(res);
    return D3D12AllocationOutcome::Success;
}

bool find_cuda_adapter(Microsoft::WRL::ComPtr<IDXGIAdapter3>& out) {
    Microsoft::WRL::ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)))) { return false; }

    int device_id = 0;
    if (cudaGetDevice(&device_id) != cudaSuccess) { return false; }
    cudaDeviceProp props{};
    if (cudaGetDeviceProperties(&props, device_id) != cudaSuccess) { return false; }

    LUID cuda_luid{};
    std::memcpy(&cuda_luid, props.luid, sizeof(LUID));

    Microsoft::WRL::ComPtr<IDXGIAdapter1> current;
    for (UINT i = 0; factory->EnumAdapters1(i, &current) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 desc{};
        current->GetDesc1(&desc);
        if (desc.AdapterLuid.LowPart == cuda_luid.LowPart &&
            desc.AdapterLuid.HighPart == cuda_luid.HighPart) {
            return SUCCEEDED(current.As(&out));
        }
    }
    return false;
}

bool query_local_video_memory(std::uint64_t& usage, std::uint64_t& budget) {
    Microsoft::WRL::ComPtr<IDXGIAdapter3> adapter3;
    if (!find_cuda_adapter(adapter3)) { return false; }
    DXGI_QUERY_VIDEO_MEMORY_INFO info{};
    if (FAILED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info))) {
        return false;
    }
    usage  = info.CurrentUsage;
    budget = info.Budget;
    return true;
}

bool reserve_local_video_memory(std::uint64_t bytes) {
    Microsoft::WRL::ComPtr<IDXGIAdapter3> adapter3;
    if (!find_cuda_adapter(adapter3)) { return false; }
    return SUCCEEDED(
        adapter3->SetVideoMemoryReservation(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, bytes));
}
#endif

// Bytes this process currently holds in device-local memory. On Windows DXGI reports the figure
// per-process, so unrelated desktop applications cannot perturb it; elsewhere the device-wide free
// counter is the only available source.
bool process_device_bytes(std::uint64_t& out) {
#if defined(_WIN32)
    std::uint64_t budget = 0;
    if (query_local_video_memory(out, budget)) { return true; }
#endif
    std::size_t free_bytes = 0;
    std::size_t total      = 0;
    if (cudaMemGetInfo(&free_bytes, &total) != cudaSuccess) { return false; }
    out = static_cast<std::uint64_t>(total) - static_cast<std::uint64_t>(free_bytes);
    return true;
}

enum class PinnedAllocationOutcome {
    Success,
    Unsupported,
    InsufficientVram,
    NotResident,
};

bool touch_resident_pages(void* data, std::size_t bytes) {
    if (data == nullptr || bytes == 0) { return true; }
    constexpr int kBlocks  = 512;
    constexpr int kThreads = 256;
    residency_touch_kernel<<<kBlocks, kThreads>>>(static_cast<const unsigned char*>(data), bytes);
    const cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    return true;
}

PinnedAllocationOutcome allocate_pinned_residency(std::size_t capacity_bytes, void*& out_ptr,
                                                  std::shared_ptr<detail::PinnedPool>& out,
                                                  std::string& reason) {
    std::uint64_t usage_before = 0;
    const bool have_usage      = process_device_bytes(usage_before);

    int device_id = 0;
    if (cudaGetDevice(&device_id) != cudaSuccess) {
        reason = "cudaGetDevice failed";
        return PinnedAllocationOutcome::Unsupported;
    }

    cudaMemPoolProps props{};
    props.allocType     = cudaMemAllocationTypePinned;
    props.location.type = cudaMemLocationTypeDevice;
    props.location.id   = device_id;

    auto pinned     = std::make_shared<detail::PinnedPool>();
    cudaError_t err = cudaMemPoolCreate(&pinned->pool, &props);
    if (err != cudaSuccess) {
        reason = cuda_error_message("cudaMemPoolCreate(pinned device) failed", err);
        return PinnedAllocationOutcome::Unsupported;
    }

    void* ptr = nullptr;
    err       = cudaMallocFromPoolAsync(&ptr, capacity_bytes, pinned->pool, nullptr);
    if (err != cudaSuccess) {
        reason = cuda_error_message("cudaMallocFromPoolAsync failed", err);
        return PinnedAllocationOutcome::InsufficientVram;
    }
    err = cudaMemsetAsync(ptr, 0, capacity_bytes, nullptr);
    if (err == cudaSuccess) { err = cudaStreamSynchronize(nullptr); }
    if (err != cudaSuccess) {
        reason = cuda_error_message("committing the pinned allocation failed", err);
        (void)cudaFreeAsync(ptr, nullptr);
        (void)cudaStreamSynchronize(nullptr);
        return PinnedAllocationOutcome::InsufficientVram;
    }

    // The runtime may silently back a "pinned" allocation with system memory when the driver cannot
    // satisfy it from device-local memory. Compare this process's device-resident bytes before and
    // after the commit; a shortfall means part of the arena would live in system memory, which is
    // exactly the failure mode this arena exists to prevent.
    std::uint64_t usage_after = 0;
    if (have_usage && process_device_bytes(usage_after)) {
        const std::uint64_t consumed =
            usage_after > usage_before ? usage_after - usage_before : 0;
        // DXGI rounds to allocation granularity, so allow one page-multiple of slack.
        constexpr std::uint64_t kResidencyTolerance = 64ULL * 1024 * 1024;
        if (consumed + kResidencyTolerance < capacity_bytes) {
            reason = "only " + std::to_string(consumed) + " of the requested " +
                     std::to_string(capacity_bytes) + " bytes are device-resident (" +
                     std::to_string(capacity_bytes - consumed) +
                     " bytes would be backed by system memory)";
            (void)cudaFreeAsync(ptr, nullptr);
            (void)cudaStreamSynchronize(nullptr);
            return PinnedAllocationOutcome::NotResident;
        }
    }

    out_ptr       = ptr;
    pinned->bytes = capacity_bytes;
    out           = std::move(pinned);
    return PinnedAllocationOutcome::Success;
}

} // namespace

#if defined(_WIN32)
detail::D3D12Residency::~D3D12Residency() {
    if (ext_mem != nullptr) {
        log_cuda_error("cudaDestroyExternalMemory", cudaDestroyExternalMemory(ext_mem));
        ext_mem = nullptr;
    }
}

bool detail::D3D12Residency::ensure_resident(DWORD timeout_ms) {
    std::lock_guard<std::mutex> lock(mutex);
    if (!device || !resource || !fence) { return false; }

    // WDDM only pages under budget pressure. While this process is inside its budget, the object is
    // resident by construction and a re-assert would only bump the D3D12 residency refcount.
    Microsoft::WRL::ComPtr<IDXGIAdapter3> adapter3;
    if (SUCCEEDED(adapter.As(&adapter3))) {
        DXGI_QUERY_VIDEO_MEMORY_INFO info{};
        if (SUCCEEDED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info)) &&
            info.CurrentUsage <= info.Budget) {
            return true;
        }
    }

    // MakeResident is ref-counted and idempotent: on an already resident object this is a legal
    // no-op that signals the fence, and on an evicted object it reclaims residency or fails with
    // E_OUTOFMEMORY under DENY_OVERBUDGET. That makes it the authoritative re-assert call.
    ID3D12Pageable* pageable   = resource.Get();
    const std::uint64_t signal = ++fence_value;
    if (FAILED(device->EnqueueMakeResident(D3D12_RESIDENCY_FLAG_DENY_OVERBUDGET, 1, &pageable,
                                           fence.Get(), signal))) {
        return false;
    }
    if (fence->GetCompletedValue() >= signal) { return true; }

    HANDLE completion = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (completion == nullptr) { return false; }
    if (FAILED(fence->SetEventOnCompletion(signal, completion))) {
        CloseHandle(completion);
        return false;
    }
    const DWORD wait_result = WaitForSingleObject(completion, timeout_ms);
    CloseHandle(completion);
    return wait_result == WAIT_OBJECT_0;
}

namespace core {
void set_wddm_residency_lock_enabled(bool enabled) noexcept {
    g_wddm_residency_lock_enabled.store(enabled, std::memory_order_relaxed);
}

bool wddm_residency_lock_enabled() noexcept {
    return g_wddm_residency_lock_enabled.load(std::memory_order_relaxed);
}

bool device_budget_headroom_bytes(std::size_t& out) noexcept {
    std::uint64_t usage = 0, budget = 0;
    if (!query_local_video_memory(usage, budget)) { return false; }
    out = budget > usage ? static_cast<std::size_t>(budget - usage) : 0;
    return true;
}
} // namespace core
#endif

#if !defined(_WIN32)
namespace core {
bool device_budget_headroom_bytes(std::size_t& out) noexcept {
    (void)out;
    return false;
}
} // namespace core
#endif

detail::PinnedPool::~PinnedPool() {
    if (pool != nullptr) {
        log_cuda_error("cudaMemPoolDestroy", cudaMemPoolDestroy(pool));
        pool = nullptr;
    }
}

DeviceBuffer::DeviceBuffer(std::size_t size_bytes) : bytes(size_bytes) {
    if (bytes == 0) { return; }

    void* ptr             = nullptr;
    const cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaMalloc failed", err));
    }
    p = ptr;
}

DeviceBuffer::~DeviceBuffer() { free_device(p); }

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept : p(other.p), bytes(other.bytes) {
    other.p     = nullptr;
    other.bytes = 0;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
    if (this == &other) { return *this; }

    free_device(p);
    p     = other.p;
    bytes = other.bytes;

    other.p     = nullptr;
    other.bytes = 0;
    return *this;
}

void DeviceBuffer::fill(int byte_value) {
    if (bytes == 0) { return; }
    const cudaError_t err = cudaMemset(p, byte_value, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaMemset failed", err));
    }
}

void DeviceBuffer::copy_from_host(const void* source, std::size_t count, std::size_t byte_offset) {
    require_range(byte_offset, count, "host-to-device copy");
    if (count == 0) { return; }
    void* destination     = static_cast<std::uint8_t*>(p) + byte_offset;
    const cudaError_t err = cudaMemcpy(destination, source, count, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaMemcpy host-to-device failed", err));
    }
    // Pageable H2D copies may return before the device transfer completes. Settle the
    // default stream before callers submit consumers on non-blocking streams.
    const cudaError_t settled = cudaStreamSynchronize(nullptr);
    if (settled != cudaSuccess) {
        throw std::runtime_error(
            cuda_error_message("host-to-device copy synchronization failed", settled));
    }
}

void DeviceBuffer::copy_to_host(void* destination, std::size_t count,
                                std::size_t byte_offset) const {
    require_range(byte_offset, count, "device-to-host copy");
    if (count == 0) { return; }
    const void* source    = static_cast<const std::uint8_t*>(p) + byte_offset;
    const cudaError_t err = cudaMemcpy(destination, source, count, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaMemcpy device-to-host failed", err));
    }
}

void DeviceBuffer::require_range(std::size_t byte_offset, std::size_t count,
                                 const char* operation) const {
    if (byte_offset > bytes || count > bytes - byte_offset) {
        throw std::out_of_range(std::string(operation) + " exceeds device buffer");
    }
}

DeviceArena::Scope::Scope(DeviceArena& arena) noexcept
    : arena_(&arena), saved_offset_(arena.off_) {}

DeviceArena::Scope::~Scope() noexcept {
    if (arena_ != nullptr && saved_offset_ <= arena_->off_) { arena_->off_ = saved_offset_; }
}

DeviceArena::Scope::Scope(Scope&& other) noexcept
    : arena_(other.arena_), saved_offset_(other.saved_offset_) {
    other.arena_ = nullptr;
}

DeviceArena::DeviceArena(std::size_t capacity_bytes, ResidencyRequirement requirement) {
    if (capacity_bytes == 0) {
        throw std::invalid_argument("DeviceArena capacity must be nonzero");
    }

    std::size_t free_bytes             = 0;
    std::size_t total_bytes            = 0;
    const bool have_memory_information = cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess;
    if (cudaGetDevice(&device_) != cudaSuccess) { device_ = 0; }
    if (have_memory_information && total_bytes > 0 && capacity_bytes > total_bytes) {
        throw std::runtime_error(
            "DeviceArena requested capacity (" + std::to_string(capacity_bytes) +
            " bytes) exceeds total physical device VRAM (" + std::to_string(total_bytes) +
            " bytes)");
    }

    void* ptr = nullptr;
    if (requirement == ResidencyRequirement::Required) {
        std::string reason;
        std::shared_ptr<detail::PinnedPool> pinned;
        PinnedAllocationOutcome outcome = PinnedAllocationOutcome::NotResident;
        // One retry after a short settle: a transient desktop burst can deny residency for a
        // moment without meaning the arena genuinely does not fit (AGENTS.md: reclaim before
        // declaring failure). A genuine overcommit fails both attempts.
        for (int attempt = 0; attempt < 2; ++attempt) {
            if (attempt != 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(300));
                reason.clear();
                pinned.reset();
                ptr = nullptr;
            }
            outcome = allocate_pinned_residency(capacity_bytes, ptr, pinned, reason);
            if (outcome == PinnedAllocationOutcome::Success ||
                outcome == PinnedAllocationOutcome::Unsupported) {
                break;
            }
        }
        if (outcome == PinnedAllocationOutcome::Success) {
            pinned_pool_ = std::move(pinned);
        } else {
            std::string message = "device arena of " + std::to_string(capacity_bytes) +
                                  " bytes cannot be placed in dedicated VRAM";
            if (!reason.empty()) { message += ": " + reason; }
            if (have_memory_information) {
                message += " (CUDA reports " + std::to_string(free_bytes) + " of " +
                           std::to_string(total_bytes) + " bytes free)";
            }
            if (outcome == PinnedAllocationOutcome::Unsupported) {
                throw ResidencyUnavailableError(message);
            }
            throw std::runtime_error(message);
        }
    }
#if defined(_WIN32)
    else if (core::wddm_residency_lock_enabled()) {
        std::string reason;
        std::shared_ptr<detail::D3D12Residency> residency;
        const D3D12AllocationOutcome outcome =
            allocate_d3d12_residency(capacity_bytes, ptr, residency, reason);
        if (outcome == D3D12AllocationOutcome::Success) { residency_ = std::move(residency); }
    }
#endif

    if (ptr == nullptr) {
        const cudaError_t err = cudaMalloc(&ptr, capacity_bytes);
        if (err != cudaSuccess) {
            (void)cudaMemGetInfo(&free_bytes, &total_bytes);
            throw std::runtime_error(
                "cudaMalloc failed for a " + std::to_string(capacity_bytes) +
                "-byte device arena: " + cudaGetErrorName(err) + ": " + cudaGetErrorString(err) +
                " (free " + std::to_string(free_bytes) + " of " + std::to_string(total_bytes) +
                " bytes)");
        }

#if defined(_WIN32)
        // Eagerly touch allocated pages to force WDDM to fault physical VRAM and evict
        // background apps. Only safe when this GPU is not also driving the desktop.
        if (core::wddm_residency_lock_enabled()) {
            const cudaError_t set_err = cudaMemset(ptr, 0, capacity_bytes);
            if (set_err != cudaSuccess) {
                free_device(ptr);
                throw std::runtime_error(cuda_error_message("cudaMemset failed", set_err));
            }
            // The touch is enqueued on the legacy default stream, which does not order against
            // non-blocking streams such as DeviceContext::load_stream. Wait for it to land so
            // stream-ordered weight copies cannot be wiped by the fill.
            const cudaError_t sync_err = cudaDeviceSynchronize();
            if (sync_err != cudaSuccess) {
                free_device(ptr);
                throw std::runtime_error(cuda_error_message("cudaDeviceSynchronize failed", sync_err));
            }
        }
#endif
    }

    base_ = ptr;
    cap_  = capacity_bytes;
    off_  = 0;
}

DeviceArena::DeviceArena(DeviceSpan storage)
    : base_(storage.data), cap_(storage.bytes), owns_(false) {
    if (base_ == nullptr || cap_ == 0) {
        throw std::invalid_argument("borrowed DeviceArena storage must be non-empty");
    }
}

DeviceArena::~DeviceArena() { release_storage(); }

void DeviceArena::release_storage() noexcept {
    if (pinned_pool_ != nullptr) {
        if (base_ != nullptr) {
            log_cuda_error("cudaFreeAsync(pinned arena)", cudaFreeAsync(base_, nullptr));
            log_cuda_error("cudaStreamSynchronize(pinned arena)",
                           cudaStreamSynchronize(nullptr));
        }
        pinned_pool_.reset();
        base_                 = nullptr;
        cap_                  = 0;
        off_                  = 0;
        peak_                 = 0;
        owns_                 = true;
        last_residency_check_ = {};
        return;
    }

    // An external-memory mapping is invalidated by cudaDestroyExternalMemory (run by the D3D12
    // lifetime's destructor); it must never be passed to cudaFree.
    const bool external = residency_ != nullptr;
    residency_.reset();
    if (owns_ && base_ != nullptr && !external) { free_device(base_); }

    base_                 = nullptr;
    cap_                  = 0;
    off_                  = 0;
    peak_                 = 0;
    owns_                 = true;
    last_residency_check_ = {};
}

DeviceArena::DeviceArena(DeviceArena&& other) noexcept
    : base_(other.base_), cap_(other.cap_), off_(other.off_), peak_(other.peak_),
      owns_(other.owns_), device_(other.device_), residency_(std::move(other.residency_)),
      pinned_pool_(std::move(other.pinned_pool_)),
      last_residency_check_(other.last_residency_check_) {
    other.base_                 = nullptr;
    other.cap_                  = 0;
    other.off_                  = 0;
    other.peak_                 = 0;
    other.owns_                 = true;
    other.last_residency_check_ = {};
}

DeviceArena& DeviceArena::operator=(DeviceArena&& other) noexcept {
    if (this == &other) { return *this; }

    release_storage();
    base_                 = other.base_;
    cap_                  = other.cap_;
    off_                  = other.off_;
    peak_                 = other.peak_;
    owns_                 = other.owns_;
    device_               = other.device_;
    residency_            = std::move(other.residency_);
    pinned_pool_          = std::move(other.pinned_pool_);
    last_residency_check_ = other.last_residency_check_;

    other.base_                 = nullptr;
    other.cap_                  = 0;
    other.off_                  = 0;
    other.peak_                 = 0;
    other.owns_                 = true;
    other.last_residency_check_ = {};
    return *this;
}

DeviceSpan DeviceArena::alloc_bytes(std::size_t bytes, std::size_t align) {
    if (base_ == nullptr) { throw std::runtime_error("DeviceArena has no backing allocation"); }
    if (!is_power_of_two(align)) {
        throw std::invalid_argument("arena alignment must be a nonzero power of two");
    }
    if (bytes == 0) { throw std::invalid_argument("arena allocation must be nonzero"); }

    const std::uintptr_t base_addr           = reinterpret_cast<std::uintptr_t>(base_);
    const std::uintptr_t current_addr        = checked_add_uintptr(base_addr, off_);
    const std::uintptr_t aligned_addr        = align_up_addr(current_addr, align);
    const std::uintptr_t aligned_offset_addr = aligned_addr - base_addr;
    if (aligned_offset_addr > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("arena aligned offset overflows size_t");
    }
    const auto aligned_offset = static_cast<std::size_t>(aligned_offset_addr);
    if (bytes > std::numeric_limits<std::size_t>::max() - aligned_offset) {
        throw std::overflow_error("arena allocation end offset overflows size_t");
    }
    const std::size_t end = aligned_offset + bytes;
    if (end > cap_) { throw std::bad_alloc(); }

    auto* ptr = static_cast<unsigned char*>(base_) + aligned_offset;
    off_      = end;
    if (off_ > peak_) { peak_ = off_; }
    return DeviceSpan{ptr, bytes};
}

Tensor DeviceArena::alloc(DType dtype, std::initializer_list<std::int32_t> shape,
                          std::size_t align) {
    Tensor view(nullptr, dtype, shape);
    const DeviceSpan storage = alloc_bytes(view.bytes(), align);
    return Tensor(storage.data, dtype, shape);
}

DeviceArena::Scope DeviceArena::scope() noexcept { return Scope(*this); }

void DeviceArena::reset() noexcept { off_ = 0; }

void* DeviceArena::base() const noexcept { return base_; }

std::size_t DeviceArena::used() const noexcept { return off_; }

std::size_t DeviceArena::capacity() const noexcept { return cap_; }

std::size_t DeviceArena::peak_used() const noexcept { return peak_; }

void DeviceArena::reset_peak() noexcept { peak_ = off_; }

bool DeviceArena::residency_enforced() const noexcept {
#if defined(_WIN32)
    return pinned_pool_ != nullptr || residency_ != nullptr;
#else
    // Without WDDM, a pinned device allocation is not pageable, so owned storage is device-resident
    // for its whole lifetime. Borrowed storage is outside this arena's control.
    return pinned_pool_ != nullptr || owns_;
#endif
}

bool DeviceArena::ensure_resident(std::chrono::milliseconds timeout) {
    if (pinned_pool_ != nullptr) {
#if defined(_WIN32)
        (void)timeout;
        const auto now = std::chrono::steady_clock::now();
        if (last_residency_check_ != std::chrono::steady_clock::time_point{} &&
            now - last_residency_check_ < kResidencyCheckInterval) {
            return true;
        }
        last_residency_check_ = now;

        // The watchdog runs on the engine worker thread, whose current device is not inherited
        // from the loading thread. Select the arena's device before querying or touching.
        if (cudaSetDevice(device_) != cudaSuccess) { return true; }
        std::uint64_t usage = 0, budget = 0;
        if (!query_local_video_memory(usage, budget)) { return true; }
        if (usage <= budget) {
            pinned_pool_->over_budget_streak = 0;
            return true;
        }

        // Over budget: ask the OS to keep this working set, then read every page back so a driver
        // that did page part of it faults the bytes in again. A CUDA failure here is a hard
        // residency loss; budget pressure alone is not.
        (void)reserve_local_video_memory(usage);
        if (!touch_resident_pages(base_, cap_)) { return false; }
        if (query_local_video_memory(usage, budget) && usage <= budget) {
            pinned_pool_->over_budget_streak = 0;
            return true;
        }

        // Still over budget after reclaiming. This is usually another application competing for
        // VRAM, so it is reported but tolerated for a while: killing the engine on a desktop burst
        // would be worse than a few slow steps. Startup headroom (see construct_registered) is what
        // keeps this path rare. Only a sustained failure is treated as unrecoverable.
        const auto streak = ++pinned_pool_->over_budget_streak;
        if (streak == 1 || streak == kOverBudgetWarningStreak) {
            std::fprintf(stderr,
                         "[ninfer] warning: process holds %llu bytes of device memory against a "
                         "%llu-byte budget; reclaimed %zu bytes of arena pages and is still over "
                         "budget (%u consecutive samples)\n",
                         static_cast<unsigned long long>(usage),
                         static_cast<unsigned long long>(budget), cap_, streak);
            std::fflush(stderr);
        }
        return streak < kOverBudgetFatalStreak;
#else
        (void)timeout;
        return true;
#endif
    }

#if defined(_WIN32)
    if (residency_ == nullptr) { return true; }

    const auto now = std::chrono::steady_clock::now();
    if (last_residency_check_ != std::chrono::steady_clock::time_point{} &&
        now - last_residency_check_ < kResidencyCheckInterval) {
        return true;
    }

    const long long requested = timeout.count() > 0 ? timeout.count() : 1;
    const DWORD timeout_ms =
        requested > static_cast<long long>(std::numeric_limits<DWORD>::max())
            ? std::numeric_limits<DWORD>::max()
            : static_cast<DWORD>(requested);
    if (!residency_->ensure_resident(timeout_ms)) { return false; }

    last_residency_check_ = now;
    return true;
#else
    (void)timeout;
    return true;
#endif
}

PinnedHostBuffer::PinnedHostBuffer(std::size_t size_bytes) {
    if (size_bytes == 0) { throw std::invalid_argument("PinnedHostBuffer size must be nonzero"); }

    void* ptr             = nullptr;
    const cudaError_t err = cudaMallocHost(&ptr, size_bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaMallocHost failed", err));
    }

    data_ = ptr;
    size_ = size_bytes;
}

PinnedHostBuffer::~PinnedHostBuffer() { free_pinned(data_); }

PinnedHostBuffer::PinnedHostBuffer(PinnedHostBuffer&& other) noexcept
    : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
}

PinnedHostBuffer& PinnedHostBuffer::operator=(PinnedHostBuffer&& other) noexcept {
    if (this == &other) { return *this; }

    free_pinned(data_);
    data_ = other.data_;
    size_ = other.size_;

    other.data_ = nullptr;
    other.size_ = 0;
    return *this;
}

void* PinnedHostBuffer::data() const noexcept { return data_; }

std::size_t PinnedHostBuffer::size() const noexcept { return size_; }

} // namespace ninfer
