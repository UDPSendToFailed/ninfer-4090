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
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer {

inline constexpr std::uint64_t kDiskStateCacheMagic   = 0x4E494E465F4D414EULL; // "NINF_MAN"
inline constexpr std::uint32_t kDiskStateCacheVersion = 5; // 5: model_hash no longer mixes kv_capacity/max_context

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

// Pure, GPU-free geometry predicate: rejects a snapshot whose header page geometry
// disagrees with the live pool's page geometry. 0 for live_text_page_bytes means
// unchecked (always accepted).
[[nodiscard]] bool disk_state_layout_matches(const DiskStateHeader& h,
                                             std::uint32_t live_text_page_bytes) noexcept;

enum class DirectStorageRestoreKind : std::uint8_t {
    TextPage,
    Gdn,
    MtpPage,
    Tail,
};

// A device-side view of one restored unit inside the DirectStorage staging buffer.
struct DirectStorageRestoreSlice {
    DirectStorageRestoreKind kind  = DirectStorageRestoreKind::TextPage;
    std::uint32_t            index = 0; // page or unit index within its kind
    const std::byte*         data  = nullptr;
    std::size_t              bytes = 0;
};

using DirectStorageRestoreConsumer = std::function<bool(std::span<const DirectStorageRestoreSlice>)>;

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
    class Impl;

public:
    // Keeps pool pages from being compacted away between the check that finds them stored and
    // the save that reuses them. Take it before deciding which pages a save must write, pass it
    // to enqueue_save_cow, and do not let it outlive the cache.
    class PageLease {
    public:
        PageLease() = default;
        PageLease(PageLease&& other) noexcept;
        PageLease& operator=(PageLease&& other) noexcept;
        PageLease(const PageLease&)            = delete;
        PageLease& operator=(const PageLease&) = delete;
        ~PageLease();

        // Whether the i-th leased page was stored in the pool when the lease was taken.
        [[nodiscard]] bool stored(std::size_t i) const { return stored_[i]; }

    private:
        friend class DiskStateCache::Impl;
        void release() noexcept;

        Impl* owner_ = nullptr;
        std::vector<std::uint64_t> hashes_;
        std::vector<bool> stored_;
    };

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

    // Leases `page_hashes` and reports which are stored. A save that writes only the missing
    // pages must decide which are missing from this lease, not from has_page.
    [[nodiscard]] PageLease lease_pages(std::span<const std::uint64_t> page_hashes);

    // Queries the disk index for the longest common prefix match matching the given tokens
    [[nodiscard]] std::optional<DiskStateMatch>
    find_longest_matching_prefix(std::uint64_t model_hash, std::span<const TokenId> tokens) const;

    // A failed load that proves the on-disk data bad (unreadable or short pool, missing page,
    // invalid manifest, checksum or layout mismatch) removes the manifest from the index and
    // the disk, and retires the unreadable pool page records, so the next save of the same
    // prompt writes a fresh manifest and fresh pages. Transient failures (DirectStorage or the
    // device unavailable, a file held open, a pool compaction during the load) keep both.

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
    // Restores a snapshot through DirectStorage in bounded staging batches. Each batch
    // is handed to `consumer` as device-side slices that stay valid only until the
    // consumer's work on `stream` completes. The manifest payload is sliced as
    // [GDN units][MTP pages][tail]; `gdn_unit_bytes` gives the GDN unit sizes in file
    // order (empty means one unit) and must sum to header.gdn_state_bytes.
    // `out_header` and `out_tokens` are filled before the first consumer call.
    // On false, part of the destination may already be written.
    [[nodiscard]] bool load_snapshot_direct_storage(const std::filesystem::path& path,
                                                    DiskStateHeader& out_header,
                                                    std::vector<TokenId>& out_tokens,
                                                    cudaStream_t stream,
                                                    std::span<const std::uint64_t> gdn_unit_bytes,
                                                    const DirectStorageRestoreConsumer& consumer,
                                                    std::uint32_t expected_text_page_bytes = 0);
#endif

    // Removes a loaded snapshot that the caller rejected (for example a live layout mismatch
    // found after load_snapshot). Its pool pages are kept.
    void discard_snapshot(const std::filesystem::path& path, const char* reason);

    // True CoW enqueue with explicit delta physical pages. `lease` should come from
    // lease_pages(all_page_hashes) taken before `missing_page_hashes` was chosen. Without one,
    // pages compacted away in between make the writer skip the snapshot rather than index a
    // manifest whose pages are gone.
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
                          std::vector<std::byte> tail_hidden,
                          PageLease lease = {});

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
