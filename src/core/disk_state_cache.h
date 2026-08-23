#pragma once

#include "core/device.h"
#include "core/linear_attention_state.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"
#include "ninfer/types.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer {

inline constexpr std::uint64_t kDiskStateCacheMagic   = 0x4E494E465F4D414EULL; // "NINF_MAN"
inline constexpr std::uint32_t kDiskStateCacheVersion = 4;

#pragma pack(push, 1)
struct DiskStateHeader {
    std::uint64_t magic                = kDiskStateCacheMagic;
    std::uint32_t version              = kDiskStateCacheVersion;
    std::uint32_t header_bytes         = sizeof(DiskStateHeader);
    std::uint64_t model_hash           = 0;
    std::uint64_t prompt_hash          = 0;
    std::uint64_t created_at_utc       = 0;
    std::uint32_t token_count          = 0;
    std::uint32_t turn_index           = 0;
    std::int32_t  rope_delta           = 0;
    std::uint32_t text_page_count      = 0; // Number of 64-token Text KV pages
    std::uint32_t text_page_bytes      = 0; // Bytes per single page across all planes
    std::uint32_t payload_offset_bytes = 0; // 4096-byte aligned offset to manifest payload (GDN, MTP, Tail)
    std::uint64_t gdn_state_bytes      = 0;
    std::uint64_t mtp_kv_bytes         = 0;
    std::uint32_t mtp_page_count       = 0;
    std::uint64_t tail_hidden_bytes    = 0;
    std::uint64_t checksum             = 0;
};

struct PageIndexRecord {
    std::uint64_t page_hash   = 0;
    std::uint64_t file_offset = 0; // 4096-byte sector aligned offset in pool_data.ninfer_pages
    std::uint32_t page_bytes  = 0; // Raw page size (pool.total_page_bytes())
    std::uint32_t reserved    = 0; // Alignment padding
};
#pragma pack(pop)

struct DiskStateMatch {
    std::uint32_t matched_tokens = 0;
    std::uint32_t turn_index     = 0;
    std::filesystem::path file_path;
    DiskStateHeader header;
};

inline std::string current_utc_timestamp_compact() {
    auto now = std::chrono::system_clock::now();
    std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf{};
#if defined(_WIN32)
    gmtime_s(&tm_buf, &tt);
#else
    gmtime_r(&tt, &tm_buf);
#endif
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%04d%02d%02d_%02d%02d%02d",
                  tm_buf.tm_year + 1900, tm_buf.tm_mon + 1, tm_buf.tm_mday,
                  tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec);
    return std::string(buf);
}

inline std::filesystem::path default_prompt_cache_dir() {
    const char* local_app_data = std::getenv("LOCALAPPDATA");
    if (local_app_data && *local_app_data) {
        return std::filesystem::path(local_app_data) / "ninfer" / "cache";
    }
    const char* user_profile = std::getenv("USERPROFILE");
    if (user_profile && *user_profile) {
        return std::filesystem::path(user_profile) / "AppData" / "Local" / "ninfer" / "cache";
    }
    return "./.ninfer_cache";
}

struct DiskStateCacheConfig {
    std::filesystem::path cache_dir = default_prompt_cache_dir();
    std::size_t max_cache_bytes     = 30ULL << 30; // 30 GiB default
    bool enabled                    = false;       // Opt-in by default
};

class DiskStateCache {
public:
    explicit DiskStateCache(DiskStateCacheConfig config = {});
    ~DiskStateCache();

    DiskStateCache(const DiskStateCache&)            = delete;
    DiskStateCache& operator=(const DiskStateCache&) = delete;
    DiskStateCache(DiskStateCache&&)                 = delete;
    DiskStateCache& operator=(DiskStateCache&&)      = delete;

    [[nodiscard]] bool enabled() const noexcept { return config_.enabled; }
    [[nodiscard]] const std::filesystem::path& cache_dir() const noexcept { return config_.cache_dir; }

    // Fast 64-bit prompt prefix hash computation
    [[nodiscard]] static std::uint64_t hash_prompt_prefix(std::span<const TokenId> tokens);

    // Queries whether a physical page is already stored in the journal
    [[nodiscard]] bool has_page(std::uint64_t page_hash) const;

    // Queries the disk index for the longest common prefix match matching the given tokens
    [[nodiscard]] std::optional<DiskStateMatch>
    find_longest_matching_prefix(std::uint64_t model_hash, std::span<const TokenId> tokens) const;

    // Restores state from disk into host pinned buffers
    [[nodiscard]] bool load_snapshot(const std::filesystem::path& path,
                                     DiskStateHeader& out_header,
                                     std::vector<TokenId>& out_tokens,
                                     std::vector<std::byte>& out_gdn_state,
                                     std::vector<std::byte>& out_text_kv_payload,
                                     std::vector<std::byte>& out_mtp_kv_payload,
                                     std::vector<std::byte>& out_tail_hidden);

#if defined(_WIN32)
    // DirectStorage kernel-bypass DMA restore directly into VRAM staging buffer
    [[nodiscard]] bool load_snapshot_direct_storage(const std::filesystem::path& path,
                                                    DiskStateHeader& out_header,
                                                    std::vector<TokenId>& out_tokens,
                                                    cudaStream_t stream,
                                                    void*& out_d_staging_ptr,
                                                    std::size_t& out_text_bytes);
#endif

    // True CoW enqueue with explicit delta physical pages
    void enqueue_save_cow(std::uint64_t model_hash,
                          std::vector<TokenId> tokens,
                          std::uint32_t turn_index,
                          std::int32_t rope_delta,
                          std::vector<std::byte> gdn_state,
                          std::vector<std::uint64_t> all_page_hashes,
                          std::vector<std::uint64_t> missing_page_hashes,
                          std::vector<std::byte> missing_pages_data,
                          std::uint32_t single_page_bytes,
                          std::vector<std::byte> mtp_kv_payload,
                          std::uint32_t mtp_page_count,
                          std::vector<std::byte> tail_hidden);

    // Move enqueue overload (automatically hashes and extracts missing pages)
    void enqueue_save(std::uint64_t model_hash,
                      std::vector<TokenId> tokens,
                      std::uint32_t turn_index,
                      std::int32_t rope_delta,
                      std::vector<std::byte> gdn_state,
                      std::vector<std::byte> text_kv_payload,
                      std::uint32_t text_page_count,
                      std::vector<std::byte> mtp_kv_payload,
                      std::uint32_t mtp_page_count,
                      std::vector<std::byte> tail_hidden);

    // Span overload for test suites
    void enqueue_save(std::uint64_t model_hash,
                      std::span<const TokenId> tokens,
                      std::uint32_t turn_index,
                      std::int32_t rope_delta,
                      std::span<const std::byte> gdn_state,
                      std::span<const std::byte> text_kv_payload,
                      std::uint32_t text_page_count,
                      std::span<const std::byte> mtp_kv_payload,
                      std::uint32_t mtp_page_count,
                      std::span<const std::byte> tail_hidden);

    // Evicts oldest manifests if total cache size exceeds max_cache_bytes
    void prune_lru();

    // Performs mark-and-sweep compaction on the physical page pool to reclaim dead pages
    void compact_pool(bool force = true);

    // Immediately cancels any currently running or queued background save tasks
    void cancel_in_flight() noexcept;

    [[nodiscard]] std::size_t device_memory_bytes() const noexcept { return 0; }

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    DiskStateCacheConfig config_;
};

} // namespace ninfer
