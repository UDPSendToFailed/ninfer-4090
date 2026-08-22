#pragma once

#if defined(_WIN32)

#include "core/device.h"

#include <d3d12.h>
#include <dxgi1_6.h>
#include <dstorage.h>
#include <dstorageerr.h>
#include <wrl/client.h>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <span>
#include <string>

namespace ninfer::core {

using Microsoft::WRL::ComPtr;

struct PageRestoreEntry {
    std::uint64_t file_offset = 0;
    std::uint32_t page_bytes  = 0;
};

class DirectStorageEngine {
public:
    static DirectStorageEngine& instance();

    DirectStorageEngine(const DirectStorageEngine&)            = delete;
    DirectStorageEngine& operator=(const DirectStorageEngine&) = delete;
    DirectStorageEngine(DirectStorageEngine&&)                 = delete;
    DirectStorageEngine& operator=(DirectStorageEngine&&)      = delete;

    [[nodiscard]] bool available() const noexcept { return initialized_; }

    [[nodiscard]] void* staging_ptr() const noexcept { return d_staging_ptr_; }

    [[nodiscard]] std::size_t staging_capacity() const noexcept { return current_capacity_; }

    // Restores snapshot payload into VRAM staging buffer via DirectStorage.
    // file_offset_4k_aligned must be 4096-byte aligned.
    bool restore_snapshot_payload(const std::filesystem::path& path,
                                  std::uint64_t file_offset_4k_aligned,
                                  std::uint64_t payload_bytes,
                                  cudaStream_t stream,
                                  void*& out_d_staging_ptr);

    // Restores CoW paged KV snapshot payload:
    // - text_pages are read from pool_path and placed in staging buffer at [0 .. total_text_bytes)
    // - manifest payload (GDN state, MTP KV, tail hidden) is read from manifest_path and placed at [total_text_bytes .. total_staging_bytes)
    bool restore_snapshot_cow(const std::filesystem::path& pool_path,
                              std::span<const PageRestoreEntry> text_pages,
                              const std::filesystem::path& manifest_path,
                              std::uint64_t manifest_payload_offset,
                              std::uint64_t manifest_payload_bytes,
                              cudaStream_t stream,
                              void*& out_d_staging_ptr,
                              std::size_t& out_text_bytes);

    // Releases the D3D12 staging buffer and unmaps CUDA external memory.
    void release_staging() noexcept;

private:
    DirectStorageEngine();
    ~DirectStorageEngine();

    bool init_d3d12_and_fence();
    bool init_direct_storage();
    void recreate_queue();
    bool allocate_staging_vram(std::size_t required_bytes);
    void cleanup();

    bool initialized_ = false;
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
    ComPtr<IDStorageFile> pool_file_;
    ComPtr<IDStorageFile> manifest_file_;

    // CUDA Interop
    cudaExternalMemory_t cuda_ext_mem_ = nullptr;
    cudaExternalSemaphore_t cuda_fence_sem_ = nullptr;
    void* d_staging_ptr_ = nullptr;
};

} // namespace ninfer::core

#endif // _WIN32
