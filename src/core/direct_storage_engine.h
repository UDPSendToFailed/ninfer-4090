#pragma once

#if defined(_WIN32)

#include "core/device.h"

#include <d3d12.h>
#include <dxgi1_6.h>
#include <dstorage.h>
#include <dstorageerr.h>
#include <wrl/client.h>

#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <mutex>
#include <span>
#include <string>

namespace ninfer::core {

using Microsoft::WRL::ComPtr;

enum class DirectStorageSource : std::uint8_t {
    Pool,
    Manifest,
};

// One contiguous byte range of a source file. A segment is never split across
// staging batches, so its bytes must not exceed kStagingCapacityBytes.
struct DirectStorageSegment {
    DirectStorageSource source      = DirectStorageSource::Pool;
    std::uint64_t       file_offset = 0;
    std::uint64_t       bytes       = 0;
};

// Called once per staging batch after the batch's read has been enqueued. The fence
// wait is already queued on `stream`, so work the consumer enqueues on that stream sees
// the batch's bytes. segment_offsets[i] is the staging offset of segment first_segment + i.
// Return false to abort the restore.
using DirectStorageBatchConsumer =
    std::function<bool(const std::byte* staging, std::size_t first_segment,
                       std::span<const std::size_t> segment_offsets)>;

class DirectStorageEngine {
public:
    // The staging resource is allocated once and kept for the life of the process.
    // Releasing it right after a restore races the DirectStorage worker's own
    // post-completion GPU work on the resource and ends in a GPU context reset
    // (nvlddmkm 153, DXGI_ERROR_DEVICE_HUNG) followed by an access violation on the
    // worker thread. A fixed capacity bounds VRAM and commit charge regardless of
    // snapshot size.
    static constexpr std::size_t kStagingCapacityBytes = 256ULL * 1024 * 1024;

    static DirectStorageEngine& instance();

    DirectStorageEngine(const DirectStorageEngine&)            = delete;
    DirectStorageEngine& operator=(const DirectStorageEngine&) = delete;
    DirectStorageEngine(DirectStorageEngine&&)                 = delete;
    DirectStorageEngine& operator=(DirectStorageEngine&&)      = delete;

    [[nodiscard]] bool available() const noexcept { return initialized_; }

    // Allocates the staging buffer now instead of on the first restore. Call it before
    // device memory is budgeted (KV capacity auto) so the budget accounts for the buffer;
    // allocating it later on a nearly full device leaves D3D12 contending for residency.
    bool prepare();

    // Reads `segments` in order into the persistent staging buffer, packing consecutive
    // segments into batches of at most kStagingCapacityBytes, and hands each batch to
    // `consumer`. Before the next batch overwrites the staging buffer, the consumer's
    // work on `stream` is synchronized. On false, part of the destination may already
    // have been written; the caller must fall back to a path that rewrites all of it.
    bool restore_segments(const std::filesystem::path& pool_path,
                          const std::filesystem::path& manifest_path,
                          std::span<const DirectStorageSegment> segments,
                          cudaStream_t stream,
                          const DirectStorageBatchConsumer& consumer);

private:
    DirectStorageEngine();
    ~DirectStorageEngine();

    bool init_d3d12_and_fence();
    bool init_direct_storage();
    bool ensure_device_alive();
    bool ensure_staging_vram();
    void release_staging() noexcept;
    void cleanup();
    void sync_last_consumer() noexcept;
    bool wait_fence_on_cpu() noexcept;

    std::atomic<bool> initialized_ = false;
    mutable std::recursive_mutex mutex_;

    // Direct3D 12 & DXGI
    ComPtr<IDXGIFactory6> dxgi_factory_;
    ComPtr<ID3D12Device> d3d12_device_;
    ComPtr<ID3D12Resource> staging_resource_;
    ComPtr<ID3D12Fence> d3d12_fence_;
    std::uint64_t fence_value_ = 0;
    std::size_t current_capacity_ = 0;

    // DirectStorage
    ComPtr<IDStorageFactory> ds_factory_;
    ComPtr<IDStorageQueue> ds_queue_;

    // CUDA Interop
    cudaExternalMemory_t cuda_ext_mem_ = nullptr;
    cudaExternalSemaphore_t cuda_fence_sem_ = nullptr;
    void* d_staging_ptr_ = nullptr;
    cudaStream_t last_consumer_stream_ = nullptr;
    cudaEvent_t consumer_event_ = nullptr;
};

} // namespace ninfer::core

#endif // _WIN32
