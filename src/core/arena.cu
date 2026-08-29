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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer {
namespace {

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
struct D3D12AllocationLifetime {
    Microsoft::WRL::ComPtr<IDXGIFactory4> factory;
    Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
    Microsoft::WRL::ComPtr<ID3D12Device3> device;
    Microsoft::WRL::ComPtr<ID3D12Heap> heap;
    Microsoft::WRL::ComPtr<ID3D12Fence> fence;
    cudaExternalMemory_t ext_mem = nullptr;
};

// Keep lifetime references alive so WDDM residency priority is never dismantled
static std::vector<D3D12AllocationLifetime> g_d3d12_allocations;

// Upper bound on how long we wait for WDDM to make the heap resident before falling back.
constexpr DWORD kResidencyWaitTimeoutMs = 10000;

#if defined(_WIN32)
static std::atomic<bool> g_wddm_residency_lock_enabled{false};
#endif

bool try_allocate_d3d12_residency_locked(std::size_t capacity_bytes, void*& out_ptr) {
    D3D12AllocationLifetime res;

    if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&res.factory)))) { return false; }

    // Match DXGI Adapter with current CUDA Device LUID
    int device_id = 0;
    if (cudaGetDevice(&device_id) != cudaSuccess) { return false; }
    cudaDeviceProp props{};
    if (cudaGetDeviceProperties(&props, device_id) != cudaSuccess) { return false; }

    LUID cuda_luid{};
    std::memcpy(&cuda_luid, props.luid, sizeof(LUID));

    Microsoft::WRL::ComPtr<IDXGIAdapter1> cur_adapter;
    for (UINT i = 0; res.factory->EnumAdapters1(i, &cur_adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 desc{};
        cur_adapter->GetDesc1(&desc);
        if (desc.AdapterLuid.LowPart == cuda_luid.LowPart &&
            desc.AdapterLuid.HighPart == cuda_luid.HighPart) {
            res.adapter = cur_adapter;
            break;
        }
    }
    if (!res.adapter) { return false; }

    Microsoft::WRL::ComPtr<ID3D12Device> base_device;
    if (FAILED(D3D12CreateDevice(res.adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&base_device)))) {
        return false;
    }
    if (FAILED(base_device.As(&res.device))) { return false; }

    constexpr std::size_t kD3D12Alignment = 64 * 1024;
    const std::size_t aligned_size =
        ((capacity_bytes + kD3D12Alignment - 1) / kD3D12Alignment) * kD3D12Alignment;

    D3D12_HEAP_DESC desc = {};
    desc.SizeInBytes = aligned_size;
    desc.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    desc.Properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    desc.Properties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    desc.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
    desc.Flags = D3D12_HEAP_FLAG_SHARED;

    if (FAILED(res.device->CreateHeap(&desc, IID_PPV_ARGS(&res.heap)))) { return false; }

    // 1. Set priority to MAXIMUM/CRITICAL (places NInfer above normal desktop apps)
    ID3D12Pageable* pageable = res.heap.Get();
    const D3D12_RESIDENCY_PRIORITY priority =
        static_cast<D3D12_RESIDENCY_PRIORITY>(0xc8000000); // D3D12_RESIDENCY_PRIORITY_MAXIMUM
    res.device->SetResidencyPriority(1, &pageable, &priority);

    // 2. Deny overbudget paging (forbid WDDM from paging this heap to GART / System RAM)
    if (FAILED(res.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&res.fence)))) { return false; }
    if (FAILED(res.device->EnqueueMakeResident(D3D12_RESIDENCY_FLAG_DENY_OVERBUDGET, 1, &pageable, res.fence.Get(), 1))) {
        return false;
    }

    // EnqueueMakeResident completes asynchronously and signals the fence once the heap is
    // actually resident in device memory. Touching the heap before that point is undefined
    // behaviour: under desktop VRAM contention the pages are not yet backed, so weights
    // written into the mapping are silently lost and the model emits noise.
    if (res.fence->GetCompletedValue() < 1) {
        HANDLE residency_event = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (residency_event == nullptr) { return false; }
        if (FAILED(res.fence->SetEventOnCompletion(1, residency_event))) {
            CloseHandle(residency_event);
            return false;
        }
        const DWORD wait_result = WaitForSingleObject(residency_event, kResidencyWaitTimeoutMs);
        CloseHandle(residency_event);
        // A timeout means WDDM could not make the heap resident without going overbudget.
        // Fall back to cudaMalloc rather than running against non-resident memory.
        if (wait_result != WAIT_OBJECT_0) { return false; }
    }

    // 3. Create shared handle for CUDA import
    HANDLE shared_handle = nullptr;
    if (FAILED(res.device->CreateSharedHandle(res.heap.Get(), nullptr, GENERIC_ALL, nullptr, &shared_handle))) {
        return false;
    }

    // 4. Import into CUDA
    cudaExternalMemoryHandleDesc ext_desc = {};
    ext_desc.type = cudaExternalMemoryHandleTypeD3D12Heap;
    ext_desc.handle.win32.handle = shared_handle;
    ext_desc.size = aligned_size;
    // cudaExternalMemoryDedicated applies to dedicated allocations (D3D12 committed resources /
    // Vulkan dedicated allocations). A D3D12 heap is a suballocatable range, so the flag must
    // not be set here.
    ext_desc.flags = 0;

    cudaExternalMemory_t ext_mem = nullptr;
    const cudaError_t import_err = cudaImportExternalMemory(&ext_mem, &ext_desc);
    CloseHandle(shared_handle); // Safe to close immediately after import
    if (import_err != cudaSuccess) { return false; }

    res.ext_mem = ext_mem;

    cudaExternalMemoryBufferDesc buf_desc = {};
    buf_desc.offset = 0;
    buf_desc.size = capacity_bytes;
    buf_desc.flags = 0;

    void* dev_ptr = nullptr;
    if (cudaExternalMemoryGetMappedBuffer(&dev_ptr, ext_mem, &buf_desc) != cudaSuccess) {
        cudaDestroyExternalMemory(ext_mem);
        return false;
    }

    // Eagerly touch physical pages to commit them
    if (cudaMemset(dev_ptr, 0, capacity_bytes) != cudaSuccess) {
        cudaFree(dev_ptr);
        cudaDestroyExternalMemory(ext_mem);
        return false;
    }

    // Verify the mapping actually round-trips before trusting 16 GiB of weights to it.
    // A D3D12 heap that imported cleanly can still fail to read back correctly, and the
    // failure mode downstream is silent output corruption rather than an error.
    {
        constexpr std::size_t kProbeBytes = 4096;
        const std::size_t probe = capacity_bytes < kProbeBytes ? capacity_bytes : kProbeBytes;
        std::vector<unsigned char> pattern(probe);
        for (std::size_t i = 0; i < probe; ++i) { pattern[i] = static_cast<unsigned char>((i * 31 + 7) & 0xFF); }
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
            std::fprintf(stderr,
                         "[ninfer] D3D12 residency arena failed read-back verification "
                         "(%zu bytes); falling back to cudaMalloc\n",
                         capacity_bytes);
            cudaDestroyExternalMemory(ext_mem);
            return false;
        }
        cudaMemset(dev_ptr, 0, capacity_bytes);
    }

    out_ptr = dev_ptr;
    g_d3d12_allocations.push_back(std::move(res));
    return true;
}
#endif

} // namespace

#if defined(_WIN32)
namespace core {
void set_wddm_residency_lock_enabled(bool enabled) noexcept {
    g_wddm_residency_lock_enabled.store(enabled, std::memory_order_relaxed);
}

bool wddm_residency_lock_enabled() noexcept {
    return g_wddm_residency_lock_enabled.load(std::memory_order_relaxed);
}
} // namespace core
#endif

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

DeviceArena::DeviceArena(std::size_t capacity_bytes) {
    if (capacity_bytes == 0) {
        throw std::invalid_argument("DeviceArena capacity must be nonzero");
    }

    void* ptr = nullptr;
#if defined(_WIN32)
    if (core::wddm_residency_lock_enabled()) {
        if (!try_allocate_d3d12_residency_locked(capacity_bytes, ptr)) { ptr = nullptr; }
    }
#endif

    if (ptr == nullptr) {
        const cudaError_t err = cudaMalloc(&ptr, capacity_bytes);
        if (err != cudaSuccess) {
            throw std::runtime_error(cuda_error_message("cudaMalloc failed", err));
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

DeviceArena::~DeviceArena() {
    if (owns_) { free_device(base_); }
}

DeviceArena::DeviceArena(DeviceArena&& other) noexcept
    : base_(other.base_), cap_(other.cap_), off_(other.off_), peak_(other.peak_),
      owns_(other.owns_) {
    other.base_ = nullptr;
    other.cap_  = 0;
    other.off_  = 0;
    other.peak_ = 0;
    other.owns_ = true;
}

DeviceArena& DeviceArena::operator=(DeviceArena&& other) noexcept {
    if (this == &other) { return *this; }

    if (owns_) { free_device(base_); }
    base_ = other.base_;
    cap_  = other.cap_;
    off_  = other.off_;
    peak_ = other.peak_;
    owns_ = other.owns_;

    other.base_ = nullptr;
    other.cap_  = 0;
    other.off_  = 0;
    other.peak_ = 0;
    other.owns_ = true;
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

PinnedHostBuffer::PinnedHostBuffer(std::size_t size_bytes) {
    if (size_bytes == 0) { throw std::invalid_argument("PinnedHostBuffer size must be nonzero"); }

    void* ptr             = nullptr;
    const cudaError_t err = cudaHostAlloc(&ptr, size_bytes, cudaHostAllocWriteCombined);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaHostAlloc failed", err));
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
