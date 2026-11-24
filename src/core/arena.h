#pragma once

#include "core/dtype.h"
#include "core/tensor.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <stdexcept>

namespace ninfer {

struct DeviceSpan {
    void* data        = nullptr;
    std::size_t bytes = 0;
};

// Raised when a residency-required allocation cannot be established (startup) or has left
// dedicated VRAM and cannot be reclaimed (runtime).
class ResidencyError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// The host has no usable D3D12/WDDM residency mechanism, so dedicated-VRAM residency cannot be
// guaranteed at all. Distinct from a capacity failure so callers can tell the two apart.
class ResidencyUnavailableError : public ResidencyError {
public:
    using ResidencyError::ResidencyError;
};

enum class ResidencyRequirement : std::uint8_t {
    // Best effort: a D3D12 residency-locked heap when the process opted in, otherwise cudaMalloc.
    None,
    // Dedicated VRAM is mandatory. The arena is a pinned CUDA device-memory allocation; startup
    // verifies that the allocation actually became VRAM-resident and fails otherwise. There is no
    // cudaMalloc or system-memory fallback.
    Required,
};

namespace detail {
// Owns the D3D12 device/heap/fence/import lifetime of one residency-locked arena.
struct D3D12Residency;
// Owns one pinned device-memory pool used by a residency-required arena.
struct PinnedPool;
} // namespace detail

// Owning device allocation for long-lived buffers. DeviceArena remains the
// suballocation primitive for workspaces; this type owns exactly one cudaMalloc.
class DeviceBuffer {
public:
    DeviceBuffer() noexcept = default;
    explicit DeviceBuffer(std::size_t size_bytes);
    ~DeviceBuffer();

    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

    void fill(int byte_value = 0);
    // Completes the upload before returning. Callers must first order any prior device
    // accesses to the destination range.
    void copy_from_host(const void* source, std::size_t count, std::size_t byte_offset = 0);
    void copy_to_host(void* destination, std::size_t count, std::size_t byte_offset = 0) const;

    // Raw access is intentional: Tensor and Weight are non-owning views.
    void* p           = nullptr;
    std::size_t bytes = 0;

private:
    void require_range(std::size_t byte_offset, std::size_t count, const char* operation) const;
};

class DeviceArena {
public:
    class Scope {
    public:
        ~Scope() noexcept;

        Scope(const Scope&)            = delete;
        Scope& operator=(const Scope&) = delete;
        Scope(Scope&& other) noexcept;
        Scope& operator=(Scope&&) = delete;

    private:
        friend class DeviceArena;

        explicit Scope(DeviceArena& arena) noexcept;

        DeviceArena* arena_       = nullptr;
        std::size_t saved_offset_ = 0;
    };

    explicit DeviceArena(std::size_t capacity_bytes,
                         ResidencyRequirement requirement = ResidencyRequirement::None);
    // Non-owning arena over an already allocated device region.
    explicit DeviceArena(DeviceSpan storage);
    ~DeviceArena();

    DeviceArena(const DeviceArena&)            = delete;
    DeviceArena& operator=(const DeviceArena&) = delete;
    DeviceArena(DeviceArena&& other) noexcept;
    DeviceArena& operator=(DeviceArena&& other) noexcept;

    DeviceSpan alloc_bytes(std::size_t bytes, std::size_t align = 256);
    Tensor alloc(DType dtype, std::initializer_list<std::int32_t> shape, std::size_t align = 256);
    [[nodiscard]] Scope scope() noexcept;
    void reset() noexcept;

    void* base() const noexcept;
    std::size_t used() const noexcept;
    std::size_t capacity() const noexcept;
    std::size_t peak_used() const noexcept;
    void reset_peak() noexcept;

    // True when the backing storage is guaranteed to stay in dedicated VRAM: a pinned CUDA
    // device-memory allocation for residency-required arenas, or a D3D12 residency-locked heap
    // for opt-in best-effort arenas.
    [[nodiscard]] bool residency_enforced() const noexcept;

    // Re-asserts dedicated-VRAM residency. Returns false when the storage was evicted and could
    // not be made resident again within `timeout`. Rate-limited to one re-assert per second, so it
    // is safe to call at request-step frequency.
    [[nodiscard]] bool ensure_resident(std::chrono::milliseconds timeout);

private:
    void release_storage() noexcept;

    void* base_       = nullptr;
    std::size_t cap_  = 0;
    std::size_t off_  = 0;
    std::size_t peak_ = 0;
    bool owns_        = true;
    int device_       = 0;
    std::shared_ptr<detail::D3D12Residency> residency_;
    std::shared_ptr<detail::PinnedPool> pinned_pool_;
    std::chrono::steady_clock::time_point last_residency_check_{};
};

class PinnedHostBuffer {
public:
    explicit PinnedHostBuffer(std::size_t size_bytes);
    ~PinnedHostBuffer();

    PinnedHostBuffer(const PinnedHostBuffer&)            = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer(PinnedHostBuffer&& other) noexcept;
    PinnedHostBuffer& operator=(PinnedHostBuffer&& other) noexcept;

    void* data() const noexcept;
    std::size_t size() const noexcept;

private:
    void* data_       = nullptr;
    std::size_t size_ = 0;
};

using WorkspaceArena = DeviceArena;

namespace core {
#if defined(_WIN32)
void set_wddm_residency_lock_enabled(bool enabled) noexcept;
[[nodiscard]] bool wddm_residency_lock_enabled() noexcept;
#endif

// Bytes this process may still allocate before the OS starts paging its device memory. This is the
// figure that decides whether paging happens, and it is not distorted by D3D12/CUDA interop the way
// cudaMemGetInfo is (an imported allocation is charged more than once against the device-wide free
// counter). Returns false on hosts without a WDDM budget, where callers fall back to free memory.
[[nodiscard]] bool device_budget_headroom_bytes(std::size_t& out) noexcept;
} // namespace core

} // namespace ninfer
