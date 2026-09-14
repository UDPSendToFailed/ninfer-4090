#include "core/disk_state_cache.h"
#if defined(_WIN32)
#include "core/direct_storage_engine.h"
#endif

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <queue>
#include <set>
#include <span>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace ninfer {

namespace {

// 64-bit FNV-1a streaming hasher with final avalanche mixer
class Fnv1aHasher {
public:
    void update(const void* data, std::size_t size) noexcept {
        if (!data || size == 0) return;
        const auto* ptr = static_cast<const std::uint8_t*>(data);
        for (std::size_t i = 0; i < size; ++i) {
            hash_ ^= static_cast<std::uint64_t>(ptr[i]);
            hash_ *= 0x100000001b3ULL;
        }
    }

    [[nodiscard]] std::uint64_t finalize() const noexcept {
        std::uint64_t h = hash_;
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33;
        h *= 0xc4ceb9fe1a85ec53ULL;
        h ^= h >> 33;
        return h;
    }

private:
    std::uint64_t hash_ = 0xcbf29ce484222325ULL;
};

inline std::uint64_t fnv1a_64(const void* data, std::size_t size) {
    Fnv1aHasher hasher;
    hasher.update(data, size);
    return hasher.finalize();
}

std::string format_hex64(std::uint64_t value) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0') << std::setw(16) << value;
    return oss.str();
}

bool is_valid_header(const DiskStateHeader& h, std::uint64_t file_size) {
    if (h.magic != kDiskStateCacheMagic) return false;
    if (h.version != kDiskStateCacheVersion) return false;
    if (h.header_bytes != sizeof(DiskStateHeader)) return false;
    if (h.token_count > 2'000'000) return false;
    if (h.gdn_state_bytes > (1ULL << 30)) return false;   // 1 GiB max GDN
    if (h.mtp_kv_bytes > (16ULL << 30)) return false;     // 16 GiB max MTP KV
    if (h.tail_hidden_bytes > (1ULL << 30)) return false; // 1 GiB max Tail hidden

    const std::uint64_t raw_header_bytes = sizeof(DiskStateHeader) +
                                         static_cast<std::uint64_t>(h.token_count) * sizeof(TokenId) +
                                         static_cast<std::uint64_t>(h.text_page_count) * sizeof(std::uint64_t);
    const std::uint64_t payload_offset = h.payload_offset_bytes != 0
        ? static_cast<std::uint64_t>(h.payload_offset_bytes)
        : ((raw_header_bytes + 4095ULL) & ~4095ULL);

    const std::uint64_t expected_total = payload_offset +
                                         h.gdn_state_bytes +
                                         h.mtp_kv_bytes +
                                         h.tail_hidden_bytes;
    return file_size == expected_total;
}

void read_manifest_page_hashes(const std::filesystem::path& path,
                               const DiskStateHeader& header,
                               std::unordered_set<std::uint64_t>& out_hashes) {
    if (header.text_page_count == 0) { return; }
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) { return; }
    const std::uint64_t page_hashes_offset =
        sizeof(DiskStateHeader) + static_cast<std::uint64_t>(header.token_count) * sizeof(TokenId);
    file.seekg(static_cast<std::streamoff>(page_hashes_offset));
    std::vector<std::uint64_t> hashes(header.text_page_count);
    if (file.read(reinterpret_cast<char*>(hashes.data()),
                  header.text_page_count * sizeof(std::uint64_t))) {
        for (std::uint64_t h : hashes) {
            out_hashes.insert(h);
        }
    }
}

// PageIndexRecord::reserved value of a record that retires its page_hash. The pool index is
// append-only; the last record for a hash wins when the index is loaded.
constexpr std::uint32_t kPageRecordTombstone = 0x42544F4DU; // "MOTB"

} // namespace

bool disk_state_layout_matches(const DiskStateHeader& h, std::uint32_t live_text_page_bytes) noexcept {
    if (h.text_page_count > 0 && live_text_page_bytes != 0 && h.text_page_bytes != live_text_page_bytes) {
        return false;
    }
    return true;
}

std::uint64_t DiskStateCache::hash_prompt_prefix(std::span<const TokenId> tokens) {
    return fnv1a_64(tokens.data(), tokens.size_bytes());
}

// Manages the shared, content-addressed append-only physical page pool (pool_data.ninfer_pages)
// and its persistent binary index (pool_index.ninfer_idx).
class PageStoreJournal {
public:
    PageStoreJournal(std::filesystem::path pool_data_path, std::filesystem::path pool_index_path)
        : pool_data_path_(std::move(pool_data_path)), pool_index_path_(std::move(pool_index_path)) {
        load_index();
    }

    [[nodiscard]] bool has_page(std::uint64_t hash) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return page_index_.find(hash) != page_index_.end();
    }

    [[nodiscard]] std::optional<PageIndexRecord> get_page(std::uint64_t hash) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = page_index_.find(hash);
        if (it != page_index_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    [[nodiscard]] const std::filesystem::path& pool_data_path() const noexcept { return pool_data_path_; }

    // Changes whenever the pool file is rewritten or removed, which moves or invalidates offsets.
    [[nodiscard]] std::uint64_t generation() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return generation_;
    }

    // Leases `hashes` against compaction and reports which were stored at that moment. A page
    // stored then stays stored until unpin_pages, unless it is retired as unreadable.
    [[nodiscard]] std::vector<bool> pin_pages(std::span<const std::uint64_t> hashes) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<bool> stored(hashes.size());
        for (std::size_t i = 0; i < hashes.size(); ++i) {
            ++pins_[hashes[i]];
            if (compacting_) {
                referenced_while_compacting_.insert(hashes[i]);
            }
            stored[i] = page_index_.contains(hashes[i]);
        }
        return stored;
    }

    void unpin_pages(std::span<const std::uint64_t> hashes) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const std::uint64_t hash : hashes) {
            const auto it = pins_.find(hash);
            if (it != pins_.end() && --it->second == 0) {
                pins_.erase(it);
            }
        }
    }

    // Retires `hashes` and every record whose bytes lie past the end of the pool. Returns
    // nullopt and retires nothing when the pool was rewritten since `expected_generation`: the
    // failed read then used offsets that no longer apply.
    [[nodiscard]] std::optional<std::size_t>
    drop_unreadable_pages(std::span<const std::uint64_t> hashes, std::uint64_t expected_generation) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (generation_ != expected_generation) {
            return std::nullopt;
        }
        std::vector<std::uint64_t> retired;
        for (const std::uint64_t hash : hashes) {
            if (page_index_.contains(hash)) {
                retired.push_back(hash);
            }
        }
        retire_locked(retired);
        return retired.size() + drop_records_past_end_locked();
    }

    // `referenced_hashes` are all pages the snapshot being saved uses. Returns whether every one
    // of them is stored once the missing ones are appended; its manifest must not be written
    // otherwise.
    [[nodiscard]] bool append_missing_pages(std::span<const std::uint64_t> referenced_hashes,
                                            std::span<const std::uint64_t> missing_hashes,
                                            std::span<const std::byte> missing_data,
                                            std::uint32_t single_page_bytes) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (missing_hashes.empty() || single_page_bytes == 0) {
            return pages_stored_locked(referenced_hashes);
        }

        std::error_code ec;
        std::uint64_t curr_file_size = 0;
        if (std::filesystem::exists(pool_data_path_, ec)) {
            curr_file_size = std::filesystem::file_size(pool_data_path_, ec);
        }

        // A pool that shrank behind the journal's back would take the new pages at offsets that
        // older records still claim, and those records would then read another page's bytes.
        if (curr_file_size < known_pool_bytes_) {
            const std::size_t retired = drop_records_past_end_locked();
            std::cerr << "[warn] ninfer: [prompt-cache] " << pool_data_path_.filename().string()
                      << " shrank outside the cache, retired " << retired << " pool page record(s)\n"
                      << std::flush;
        }

        std::ofstream data_file(pool_data_path_, std::ios::binary | std::ios::app);
        std::ofstream index_file(pool_index_path_, std::ios::binary | std::ios::app);
        if (!data_file.is_open() || !index_file.is_open()) {
            return pages_stored_locked(referenced_hashes);
        }

        const std::uint64_t aligned_page_bytes = (static_cast<std::uint64_t>(single_page_bytes) + 4095ULL) & ~4095ULL;
        const std::size_t pad_size = static_cast<std::size_t>(aligned_page_bytes - single_page_bytes);
        static const std::vector<char> zeros(4096, 0);

        for (std::size_t i = 0; i < missing_hashes.size(); ++i) {
            const std::uint64_t hash = missing_hashes[i];
            if (page_index_.find(hash) != page_index_.end()) {
                continue; // Already stored
            }
            if ((i + 1) * static_cast<std::size_t>(single_page_bytes) > missing_data.size()) {
                break; // No bytes for the rest; the check below reports them missing.
            }

            const std::uint64_t page_offset = curr_file_size;
            const std::byte* src_ptr = missing_data.data() + i * single_page_bytes;

            data_file.write(reinterpret_cast<const char*>(src_ptr), single_page_bytes);
            if (pad_size > 0) {
                data_file.write(zeros.data(), pad_size);
            }

            curr_file_size += aligned_page_bytes;

            PageIndexRecord rec;
            rec.page_hash   = hash;
            rec.file_offset = page_offset;
            rec.page_bytes  = single_page_bytes;
            rec.reserved    = 0;

            index_file.write(reinterpret_cast<const char*>(&rec), sizeof(rec));
            page_index_[hash] = rec;
            if (compacting_) {
                // Stored again after a retirement: the new record, not the retirement, is the latest,
                // and the compaction must copy it even if it copied the retired bytes already.
                retired_while_compacting_.erase(hash);
                referenced_while_compacting_.insert(hash);
            }
        }

        data_file.flush();
        index_file.flush();
        known_pool_bytes_ = curr_file_size;
        return pages_stored_locked(referenced_hashes);
    }

    struct CompactStats {
        std::size_t live_pages_retained = 0;
        std::size_t dead_pages_purged   = 0;
        std::uint64_t bytes_freed       = 0;
        std::int64_t elapsed_ms         = 0;
    };

    [[nodiscard]] std::size_t total_pages() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return page_index_.size();
    }

    // Keeps the pages in `collect_live_hashes` and every page leased while the compaction runs.
    // The collection runs after the leases are recorded, and a save holds its lease until its
    // manifest is indexed, so each save is either already in the collection or still leased.
    CompactStats compact(const std::function<std::unordered_set<std::uint64_t>()>& collect_live_hashes,
                         std::size_t min_dead_pages = 256,
                         std::uint64_t min_dead_bytes = 2ULL << 30, // 2 GiB minimum dead space
                         double min_dead_ratio = 0.25,              // 25% minimum fragmentation ratio
                         bool force = false,
                         const std::atomic<bool>* cancel_token = nullptr) {
        CompactStats stats{};
        std::error_code ec;

        std::lock_guard<std::mutex> compact_lock(compact_mutex_);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            compacting_ = true;
            retired_while_compacting_.clear();
            referenced_while_compacting_.clear();
            for (const auto& [hash, count] : pins_) {
                referenced_while_compacting_.insert(hash);
            }
        }
        struct CompactingReset {
            PageStoreJournal& journal;
            ~CompactingReset() {
                std::lock_guard<std::mutex> lock(journal.mutex_);
                journal.compacting_ = false;
                journal.retired_while_compacting_.clear();
                journal.referenced_while_compacting_.clear();
            }
        } compacting_reset{*this};

        std::unordered_set<std::uint64_t> live_hashes = collect_live_hashes();

        std::unordered_map<std::uint64_t, PageIndexRecord> local_index;
        std::uint64_t old_file_size = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            live_hashes.insert(referenced_while_compacting_.begin(), referenced_while_compacting_.end());
            if (!std::filesystem::exists(pool_data_path_, ec)) {
                page_index_.clear();
                std::filesystem::remove(pool_index_path_, ec);
                known_pool_bytes_ = 0;
                ++generation_;
                return stats;
            }
            old_file_size = std::filesystem::file_size(pool_data_path_, ec);
            if (live_hashes.empty()) {
                stats.dead_pages_purged = page_index_.size();
                stats.bytes_freed       = old_file_size;
                page_index_.clear();
                std::filesystem::remove(pool_data_path_, ec);
                std::filesystem::remove(pool_index_path_, ec);
                known_pool_bytes_ = 0;
                ++generation_;
                return stats;
            }
            local_index = page_index_;
        }

        std::size_t dead_count = 0;
        std::uint64_t dead_bytes = 0;
        for (const auto& [hash, old_rec] : local_index) {
            if (!live_hashes.contains(hash)) {
                ++dead_count;
                const std::uint64_t aligned_page_bytes =
                    (static_cast<std::uint64_t>(old_rec.page_bytes) + 4095ULL) & ~4095ULL;
                dead_bytes += aligned_page_bytes;
            }
        }

        if (!force) {
            const double dead_ratio = old_file_size > 0
                ? (static_cast<double>(dead_bytes) / static_cast<double>(old_file_size))
                : 0.0;
            if (dead_count < min_dead_pages || dead_bytes < min_dead_bytes || dead_ratio < min_dead_ratio) {
                // Below threshold: avoid heavy full-pool rewrites for low fragmentation or small dead space
                stats.live_pages_retained = local_index.size();
                return stats;
            }
        } else if (dead_count < min_dead_pages) {
            stats.live_pages_retained = local_index.size();
            return stats;
        }

        const auto t_compact_0 = std::chrono::steady_clock::now();

        const auto temp_data_path  = pool_data_path_.string() + ".tmp";
        const auto temp_index_path = pool_index_path_.string() + ".tmp";

        // Clean up any stale leftover temporary files
        std::filesystem::remove(temp_data_path, ec);
        std::filesystem::remove(temp_index_path, ec);

        std::ifstream src_data(pool_data_path_, std::ios::binary);
        std::ofstream dst_data(temp_data_path, std::ios::binary | std::ios::trunc);
        std::ofstream dst_index(temp_index_path, std::ios::binary | std::ios::trunc);

        if (!src_data.is_open() || !dst_data.is_open() || !dst_index.is_open()) {
            std::filesystem::remove(temp_data_path, ec);
            std::filesystem::remove(temp_index_path, ec);
            return stats;
        }

        std::unordered_map<std::uint64_t, PageIndexRecord> new_page_index;
        new_page_index.reserve(live_hashes.size());
        std::uint64_t new_file_size = 0;
        std::vector<char> buffer;
        // Old pool offset each copied page came from; a page stored again since has a new one.
        std::unordered_map<std::uint64_t, std::uint64_t> copied_from;

        const auto cancelled = [cancel_token] {
            return cancel_token && cancel_token->load(std::memory_order_acquire);
        };
        // Leaves the old pool and index untouched. Does not take mutex_, so the swap may call it.
        const auto abandon = [&] {
            src_data.close();
            dst_data.close();
            dst_index.close();
            std::filesystem::remove(temp_data_path, ec);
            std::filesystem::remove(temp_index_path, ec);
            stats.dead_pages_purged   = 0;
            stats.bytes_freed         = 0;
            stats.live_pages_retained = local_index.size();
            return stats;
        };
        // Appends the page `old_rec` points to in the old pool; false when its bytes cannot be read.
        const auto copy_page = [&](const PageIndexRecord& old_rec) {
            const std::uint64_t aligned_page_bytes =
                (static_cast<std::uint64_t>(old_rec.page_bytes) + 4095ULL) & ~4095ULL;
            if (buffer.size() < aligned_page_bytes) {
                buffer.resize(aligned_page_bytes);
            }

            src_data.clear(); // A failed read or EOF would otherwise fail every later read.
            src_data.seekg(static_cast<std::streamoff>(old_rec.file_offset));
            if (!src_data.read(buffer.data(), static_cast<std::streamsize>(aligned_page_bytes))) {
                return false;
            }

            dst_data.write(buffer.data(), static_cast<std::streamsize>(aligned_page_bytes));

            PageIndexRecord new_rec;
            new_rec.page_hash   = old_rec.page_hash;
            new_rec.file_offset = new_file_size;
            new_rec.page_bytes  = old_rec.page_bytes;
            new_rec.reserved    = 0;

            dst_index.write(reinterpret_cast<const char*>(&new_rec), sizeof(new_rec));
            new_page_index[old_rec.page_hash] = new_rec;
            copied_from[old_rec.page_hash] = old_rec.file_offset;
            new_file_size += aligned_page_bytes;
            ++stats.live_pages_retained;
            return true;
        };
        // Records leased or stored since the snapshot that the new pool lacks in their current
        // form: pages appended to the old pool after the snapshot, snapshot pages that looked
        // dead but a save now depends on, and pages stored again after a retirement.
        const auto uncopied_references_locked = [&] {
            std::vector<PageIndexRecord> records;
            for (const std::uint64_t hash : referenced_while_compacting_) {
                const auto it = page_index_.find(hash);
                if (it == page_index_.end()) {
                    continue;
                }
                const auto copied = copied_from.find(hash);
                if (copied == copied_from.end() || copied->second != it->second.file_offset) {
                    records.push_back(it->second);
                }
            }
            return records;
        };

        for (const auto& [hash, old_rec] : local_index) {
            if (cancelled()) {
                return abandon();
            }

            if (!live_hashes.contains(hash)) {
                ++stats.dead_pages_purged;
                continue;
            }

            copy_page(old_rec);
        }

        if (cancelled()) {
            return abandon();
        }

        // Most pages saved during the copy are carried over here without the lock; the swap below
        // copies the few saved after this pass, so no append can land in the old files unseen.
        std::vector<PageIndexRecord> carried_over;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            carried_over = uncopied_references_locked();
        }
        for (const auto& rec : carried_over) {
            if (!copy_page(rec)) {
                return abandon();
            }
        }

        if (cancelled()) {
            return abandon();
        }

        // Atomic swap under journal lock
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (const auto& rec : uncopied_references_locked()) {
                if (!copy_page(rec)) {
                    return abandon();
                }
            }
            dst_data.flush();
            dst_index.flush();
            const bool written = dst_data.good() && dst_index.good();
            src_data.close();
            dst_data.close();
            dst_index.close();
            if (!written) {
                return abandon();
            }

            std::filesystem::remove(pool_data_path_, ec);
            std::filesystem::rename(temp_data_path, pool_data_path_, ec);
            std::filesystem::remove(pool_index_path_, ec);
            std::filesystem::rename(temp_index_path, pool_index_path_, ec);

            page_index_ = std::move(new_page_index);
            known_pool_bytes_ = new_file_size;
            ++generation_;

            // Pages retired while the copy ran came from the old index snapshot and may have
            // been copied into the new files; retire them again against the new index.
            compacting_ = false;
            const std::vector<std::uint64_t> retired(retired_while_compacting_.begin(),
                                                     retired_while_compacting_.end());
            retired_while_compacting_.clear();
            for (const std::uint64_t hash : retired) {
                page_index_.erase(hash);
            }
            append_tombstones_locked(retired);
        }

        stats.bytes_freed = (old_file_size > new_file_size) ? (old_file_size - new_file_size) : 0;
        stats.elapsed_ms  = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_compact_0).count();
        return stats;
    }

private:
    void load_index() {
        std::lock_guard<std::mutex> lock(mutex_);
        page_index_.clear();
        {
            std::ifstream index_file(pool_index_path_, std::ios::binary);
            if (!index_file.is_open()) {
                return;
            }

            PageIndexRecord rec;
            while (index_file.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
                if (rec.reserved == kPageRecordTombstone) {
                    page_index_.erase(rec.page_hash);
                } else {
                    page_index_[rec.page_hash] = rec;
                }
            }
        }

        // A pool shorter than its index (truncated or replaced while the process was down) leaves
        // records whose bytes are gone. The manifests referencing them are then removed by
        // scan_cache_dir.
        const std::size_t retired = drop_records_past_end_locked();
        if (retired > 0) {
            std::cout << "[info] ninfer: [prompt-cache] retired " << retired
                      << " pool page record(s) past the end of " << pool_data_path_.filename().string() << "\n"
                      << std::flush;
        }
    }

    // Later appends reuse the offsets past the end of a shortened pool, so these records are
    // retired on disk, not only in memory.
    std::size_t drop_records_past_end_locked() {
        std::error_code ec;
        std::uint64_t pool_bytes = std::filesystem::file_size(pool_data_path_, ec);
        if (ec) {
            if (std::filesystem::exists(pool_data_path_, ec) || ec) {
                return 0; // Present but not measurable right now.
            }
            pool_bytes = 0;
        }
        known_pool_bytes_ = pool_bytes;
        std::vector<std::uint64_t> past_end;
        for (const auto& [hash, rec] : page_index_) {
            if (rec.file_offset + rec.page_bytes > pool_bytes) {
                past_end.push_back(hash);
            }
        }
        retire_locked(past_end);
        return past_end.size();
    }

    [[nodiscard]] bool pages_stored_locked(std::span<const std::uint64_t> hashes) const {
        return std::all_of(hashes.begin(), hashes.end(),
                           [this](std::uint64_t hash) { return page_index_.contains(hash); });
    }

    void retire_locked(std::span<const std::uint64_t> hashes) {
        for (const std::uint64_t hash : hashes) {
            page_index_.erase(hash);
            if (compacting_) {
                retired_while_compacting_.insert(hash);
                referenced_while_compacting_.erase(hash);
            }
        }
        append_tombstones_locked(hashes);
    }

    void append_tombstones_locked(std::span<const std::uint64_t> hashes) {
        if (hashes.empty()) {
            return;
        }
        std::ofstream index_file(pool_index_path_, std::ios::binary | std::ios::app);
        if (!index_file.is_open()) {
            return;
        }
        for (const std::uint64_t hash : hashes) {
            PageIndexRecord rec;
            rec.page_hash = hash;
            rec.reserved  = kPageRecordTombstone;
            index_file.write(reinterpret_cast<const char*>(&rec), sizeof(rec));
        }
        index_file.flush();
    }

    std::filesystem::path pool_data_path_;
    std::filesystem::path pool_index_path_;
    mutable std::mutex mutex_;
    // Held for a whole compaction: two would share the temporary files and swap each other's pools.
    std::mutex compact_mutex_;
    std::unordered_map<std::uint64_t, PageIndexRecord> page_index_;
    std::uint64_t generation_ = 0;
    // Pool size as of the journal's last write or measurement; a smaller file means truncation.
    std::uint64_t known_pool_bytes_ = 0;
    bool compacting_          = false;
    std::unordered_set<std::uint64_t> retired_while_compacting_;
    // Pages leased or stored while a compaction ran; disjoint from retired_while_compacting_, the
    // later event for a hash wins.
    std::unordered_set<std::uint64_t> referenced_while_compacting_;
    // Lease count per page hash, see pin_pages.
    std::unordered_map<std::uint64_t, std::uint32_t> pins_;
};

struct QueuedSaveTask {
    std::filesystem::path manifest_path;
    DiskStateHeader header;
    std::vector<TokenId> tokens;
    std::vector<std::uint64_t> all_page_hashes;
    std::vector<std::uint64_t> missing_page_hashes;
    std::vector<std::byte> missing_pages_data;
    std::uint32_t single_page_bytes = 0;
    std::vector<std::byte> gdn_state;
    std::vector<std::byte> mtp_kv_payload;
    std::vector<std::byte> tail_hidden;
    // Held until the manifest is indexed or the task is dropped.
    DiskStateCache::PageLease lease;
};

class DiskStateCache::Impl {
public:
    explicit Impl(DiskStateCacheConfig config)
        : config_(std::move(config)),
          journal_(config_.cache_dir / "pool_data.ninfer_pages",
                   config_.cache_dir / "pool_index.ninfer_idx"),
          running_(true) {
        if (config_.enabled) {
            std::error_code ec;
            std::filesystem::create_directories(config_.cache_dir / "manifests", ec);
            scan_cache_dir();
            worker_ = std::jthread([this](std::stop_token stop) { background_writer_loop(stop); });
        }
    }

    ~Impl() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            running_ = false;
        }
        cv_.notify_all();
        if (worker_.joinable()) {
            worker_.request_stop();
            worker_.join();
        }
    }

    [[nodiscard]] bool has_page(std::uint64_t page_hash) const {
        return journal_.has_page(page_hash);
    }

    [[nodiscard]] PageLease lease_pages(std::span<const std::uint64_t> page_hashes) {
        PageLease lease;
        lease.stored_ = journal_.pin_pages(page_hashes);
        lease.hashes_.assign(page_hashes.begin(), page_hashes.end());
        lease.owner_ = this;
        return lease;
    }

    void unpin_pages(std::span<const std::uint64_t> page_hashes) {
        journal_.unpin_pages(page_hashes);
    }

    [[nodiscard]] std::optional<DiskStateMatch>
    find_longest_matching_prefix(std::uint64_t model_hash, std::span<const TokenId> tokens) const {
        if (!config_.enabled || tokens.empty()) { return std::nullopt; }

        std::lock_guard<std::mutex> lock(mutex_);
        std::optional<DiskStateMatch> best_match;

        for (const auto& [path, header] : index_) {
            if (header.model_hash != model_hash) { continue; }
            if (header.token_count > tokens.size()) { continue; }

            const std::uint64_t sub_hash = hash_prompt_prefix(tokens.subspan(0, header.token_count));
            if (sub_hash == header.prompt_hash) {
                if (!best_match || header.token_count > best_match->matched_tokens) {
                    DiskStateMatch match;
                    match.matched_tokens = header.token_count;
                    match.turn_index     = header.turn_index;
                    match.file_path      = path;
                    match.header         = header;
                    best_match           = match;
                }
            }
        }

        return best_match;
    }

    [[nodiscard]] bool load_snapshot(const std::filesystem::path& path,
                                     DiskStateHeader& out_header,
                                     std::vector<TokenId>& out_tokens,
                                     std::vector<std::byte>& out_gdn_state,
                                     std::vector<std::byte>& out_text_kv_payload,
                                     std::vector<std::byte>& out_mtp_kv_payload,
                                     std::vector<std::byte>& out_tail_hidden) {
        const std::uint64_t journal_generation = journal_.generation();
        LoadFailure failure;
        const bool ok = load_snapshot_host(path, out_header, out_tokens, out_gdn_state, out_text_kv_payload,
                                           out_mtp_kv_payload, out_tail_hidden, failure);
        if (!ok) { discard_after_failed_load(path, failure, journal_generation); }
        return ok;
    }

#if defined(_WIN32)
    [[nodiscard]] bool load_snapshot_direct_storage(const std::filesystem::path& path,
                                                    DiskStateHeader& out_header,
                                                    std::vector<TokenId>& out_tokens,
                                                    cudaStream_t stream,
                                                    std::span<const std::uint64_t> gdn_unit_bytes,
                                                    const DirectStorageRestoreConsumer& consumer,
                                                    std::uint32_t expected_text_page_bytes) {
        const std::uint64_t journal_generation = journal_.generation();
        LoadFailure failure;
        const bool ok = load_snapshot_ds(path, out_header, out_tokens, stream, gdn_unit_bytes, consumer,
                                         expected_text_page_bytes, failure);
        if (!ok) { discard_after_failed_load(path, failure, journal_generation); }
        return ok;
    }
#endif

    void discard_snapshot(const std::filesystem::path& path, const char* reason) {
        LoadFailure failure;
        failure.kind   = LoadFault::Manifest;
        failure.reason = reason;
        discard_after_failed_load(path, failure, journal_.generation());
    }

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
                          PageLease lease) {
        if (!config_.enabled || tokens.empty()) { return; }

        // Pages lost before this point make the writer skip the save; a lease over other pages
        // protects nothing, so the save leases its own from here on.
        if (lease.owner_ != this || !std::ranges::equal(lease.hashes_, all_page_hashes)) {
            lease = lease_pages(all_page_hashes);
        }

        const std::uint64_t prompt_hash = hash_prompt_prefix(tokens);
        const std::string timestamp_str = current_utc_timestamp_compact();
        const std::string filename = "state_" + timestamp_str + "_p" +
                                     format_hex64(prompt_hash) + "_t" +
                                     std::to_string(tokens.size()) + ".ninfer_manifest";
        const std::filesystem::path target_path = config_.cache_dir / "manifests" / filename;

        QueuedSaveTask task;
        task.manifest_path                   = target_path;
        task.header.magic                    = kDiskStateCacheMagic;
        task.header.version                  = kDiskStateCacheVersion;
        task.header.header_bytes             = sizeof(DiskStateHeader);
        task.header.model_hash               = model_hash;
        task.header.prompt_hash              = prompt_hash;
        task.header.created_at_utc           = static_cast<std::uint64_t>(std::time(nullptr));
        task.header.token_count              = static_cast<std::uint32_t>(tokens.size());
        task.header.turn_index               = turn_index;
        task.header.rope_delta               = rope_delta;
        task.header.text_page_count          = static_cast<std::uint32_t>(all_page_hashes.size());
        task.header.text_page_bytes          = single_page_bytes;
        task.header.mtp_page_count           = mtp_page_count;
        task.header.gdn_state_bytes          = static_cast<std::uint64_t>(gdn_state.size());
        task.header.mtp_kv_bytes             = static_cast<std::uint64_t>(mtp_kv_payload.size());
        task.header.tail_hidden_bytes        = static_cast<std::uint64_t>(tail_hidden.size());

        // Manifest payload offset aligned to 4096 bytes
        const std::uint64_t raw_header_bytes = sizeof(DiskStateHeader) +
                                             static_cast<std::uint64_t>(tokens.size()) * sizeof(TokenId) +
                                             static_cast<std::uint64_t>(all_page_hashes.size()) * sizeof(std::uint64_t);
        const std::uint64_t payload_offset   = (raw_header_bytes + 4095ULL) & ~4095ULL;
        task.header.payload_offset_bytes     = static_cast<std::uint32_t>(payload_offset);

        Fnv1aHasher hasher;
        hasher.update(gdn_state.data(), gdn_state.size());
        hasher.update(mtp_kv_payload.data(), mtp_kv_payload.size());
        hasher.update(tail_hidden.data(), tail_hidden.size());
        task.header.checksum                 = hasher.finalize();

        task.tokens                          = std::move(tokens);
        task.all_page_hashes                 = std::move(all_page_hashes);
        task.missing_page_hashes             = std::move(missing_page_hashes);
        task.missing_pages_data              = std::move(missing_pages_data);
        task.single_page_bytes               = single_page_bytes;
        task.gdn_state                       = std::move(gdn_state);
        task.mtp_kv_payload                  = std::move(mtp_kv_payload);
        task.tail_hidden                     = std::move(tail_hidden);
        task.lease                           = std::move(lease);

        constexpr std::size_t kMaxQueuedTasks = 8;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (auto it = index_.begin(); it != index_.end(); ++it) {
                const auto& [path, header] = *it;
                if (header.model_hash != task.header.model_hash ||
                    header.prompt_hash != task.header.prompt_hash ||
                    header.token_count != task.header.token_count) {
                    continue;
                }
                if (manifest_pages_stored(path, header)) {
                    return;
                }
                // The duplicate lost pages it references (a retired or compacted-away record) and
                // would only fail its next load; the new save replaces it instead of being skipped.
                std::error_code ec;
                std::filesystem::remove(path, ec);
                std::cerr << "[warn] ninfer: [prompt-cache] replacing snapshot " << path.filename().string()
                          << " whose pool pages are gone\n" << std::flush;
                index_.erase(it);
                break;
            }

            if (pending_hashes_.contains({task.header.prompt_hash, task.header.token_count})) {
                return;
            }

            if (queue_.size() >= kMaxQueuedTasks) {
                const auto& old = queue_.front();
                pending_hashes_.erase({old.header.prompt_hash, old.header.token_count});
                queue_.pop();
            }
            pending_hashes_.insert({task.header.prompt_hash, task.header.token_count});
            queue_.push(std::move(task));
        }
        cv_.notify_one();
    }

    void enqueue_save(std::uint64_t model_hash,
                      std::vector<TokenId> tokens,
                      std::uint32_t turn_index,
                      std::int32_t rope_delta,
                      std::vector<std::byte> gdn_state,
                      std::vector<std::byte> text_kv_payload,
                      std::uint32_t text_page_count,
                      std::vector<std::byte> mtp_kv_payload,
                      std::uint32_t mtp_page_count,
                      std::vector<std::byte> tail_hidden) {
        if (!config_.enabled || tokens.empty()) { return; }

        const std::uint32_t single_page_bytes = (text_page_count > 0 && !text_kv_payload.empty())
            ? static_cast<std::uint32_t>(text_kv_payload.size() / text_page_count)
            : 0;

        std::vector<std::uint64_t> all_page_hashes;
        all_page_hashes.reserve(text_page_count);
        for (std::uint32_t i = 0; i < text_page_count; ++i) {
            const std::size_t span_end = std::min<std::size_t>(tokens.size(), static_cast<std::size_t>(i + 1) * 64);
            all_page_hashes.push_back(hash_prompt_prefix(std::span<const TokenId>(tokens.data(), span_end)));
        }

        auto lease = lease_pages(all_page_hashes);
        std::vector<std::uint64_t> missing_page_hashes;
        std::vector<std::byte> missing_pages_data;
        for (std::uint32_t i = 0; i < text_page_count; ++i) {
            if (!lease.stored(i) && single_page_bytes > 0) {
                missing_page_hashes.push_back(all_page_hashes[i]);
                const std::size_t offset = i * single_page_bytes;
                missing_pages_data.insert(missing_pages_data.end(),
                                          text_kv_payload.begin() + offset,
                                          text_kv_payload.begin() + offset + single_page_bytes);
            }
        }

        enqueue_save_cow(model_hash, std::move(tokens), turn_index, rope_delta,
                         std::move(gdn_state), std::move(all_page_hashes),
                         std::move(missing_page_hashes), std::move(missing_pages_data),
                         single_page_bytes, std::move(mtp_kv_payload),
                         mtp_page_count, std::move(tail_hidden), std::move(lease));
    }

    void enqueue_save(std::uint64_t model_hash,
                      std::span<const TokenId> tokens,
                      std::uint32_t turn_index,
                      std::int32_t rope_delta,
                      std::span<const std::byte> gdn_state,
                      std::span<const std::byte> text_kv_payload,
                      std::uint32_t text_page_count,
                      std::span<const std::byte> mtp_kv_payload,
                      std::uint32_t mtp_page_count,
                      std::span<const std::byte> tail_hidden) {
        std::vector<TokenId> toks(tokens.begin(), tokens.end());
        std::vector<std::byte> gdn(gdn_state.begin(), gdn_state.end());
        std::vector<std::byte> kv(text_kv_payload.begin(), text_kv_payload.end());
        std::vector<std::byte> mtp(mtp_kv_payload.begin(), mtp_kv_payload.end());
        std::vector<std::byte> tail(tail_hidden.begin(), tail_hidden.end());
        enqueue_save(model_hash, std::move(toks), turn_index, rope_delta,
                     std::move(gdn), std::move(kv), text_page_count,
                     std::move(mtp), mtp_page_count, std::move(tail));
    }

    void prune_lru() {
        std::size_t evicted_count = 0;
        std::uint64_t evicted_bytes = 0;
        std::uint64_t pool_bytes = 0;
        bool should_compact = false;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            std::uint64_t total_bytes = 0;
            std::vector<std::pair<std::filesystem::file_time_type, std::filesystem::path>> entries;

            std::error_code ec;
            const auto manifest_dir = config_.cache_dir / "manifests";
            for (const auto& entry : std::filesystem::directory_iterator(manifest_dir, ec)) {
                if (entry.is_regular_file() && entry.path().extension() == ".ninfer_manifest") {
                    const auto sz = entry.file_size(ec);
                    total_bytes += sz;
                    entries.emplace_back(entry.last_write_time(ec), entry.path());
                }
            }

            if (std::filesystem::exists(journal_.pool_data_path(), ec)) {
                pool_bytes = std::filesystem::file_size(journal_.pool_data_path(), ec);
                total_bytes += pool_bytes;
            }

            // High Watermark: 100% of max_cache_bytes
            // Low Watermark: 75% of max_cache_bytes (frees 25% headroom in one pass to prevent turn-by-turn thrashing)
            const std::uint64_t high_watermark = config_.max_cache_bytes;
            const std::uint64_t low_watermark  = config_.max_cache_bytes * 3 / 4;

            if (total_bytes > high_watermark) {
                std::sort(entries.begin(), entries.end()); // Oldest first
                for (const auto& [time, path] : entries) {
                    if (total_bytes <= low_watermark) { break; }
                    const auto sz = std::filesystem::file_size(path, ec);
                    std::filesystem::remove(path, ec);
                    total_bytes -= sz;
                    evicted_bytes += sz;
                    ++evicted_count;
                    index_.erase(path);
                }
            }

            should_compact = evicted_count > 0 || pool_bytes > (config_.max_cache_bytes * 4 / 10);
        } // mutex_ released here: <1ms hold time

        if (evicted_count > 0) {
            const double evicted_mb = static_cast<double>(evicted_bytes) / (1024.0 * 1024.0);
            const double limit_gb = static_cast<double>(config_.max_cache_bytes) / (1024.0 * 1024.0 * 1024.0);
            std::cout << "[info] ninfer: [prompt-cache] LRU bulk evicted " << evicted_count
                      << " manifest(s) (" << std::fixed << std::setprecision(1) << evicted_mb
                      << "MB freed, limit=" << limit_gb << "GB)\n" << std::flush;
        }

        // Perform Mark-and-Sweep compaction only if dead space >= 2.0 GiB AND fragmentation >= 25%
        if (should_compact) {
            constexpr std::size_t kMinDeadPages = 256;
            constexpr std::uint64_t kMinDeadBytes = 2ULL << 30; // 2 GiB
            constexpr double kMinDeadRatio = 0.25;               // 25% fragmentation
            const auto stats = journal_.compact([this] { return collect_live_page_hashes(); },
                                               kMinDeadPages, kMinDeadBytes, kMinDeadRatio,
                                               /*force=*/false, &cancel_requested_);
            if (stats.dead_pages_purged > 0) {
                const double freed_mb = static_cast<double>(stats.bytes_freed) / (1024.0 * 1024.0);
                std::cout << "[info] ninfer: [prompt-cache] compacted page pool: purged "
                          << stats.dead_pages_purged << " dead page(s), retained "
                          << stats.live_pages_retained << " live page(s) ("
                          << std::fixed << std::setprecision(1) << freed_mb << "MB freed in "
                          << stats.elapsed_ms << "ms)\n"
                          << std::flush;
            }
        }
    }

    // Pages of indexed manifests. Saves not indexed yet keep theirs through their leases.
    [[nodiscard]] std::unordered_set<std::uint64_t> collect_live_page_hashes() const {
        std::unordered_set<std::uint64_t> live_hashes;
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& [manifest_path, header] : index_) {
            read_manifest_page_hashes(manifest_path, header, live_hashes);
        }
        return live_hashes;
    }

    void compact_pool(bool force = true) {
        const std::size_t min_dead_pages   = force ? 1 : 256;
        const std::uint64_t min_dead_bytes = force ? 0 : (2ULL << 30);
        const double min_dead_ratio        = force ? 0.0 : 0.25;
        const auto stats = journal_.compact([this] { return collect_live_page_hashes(); },
                                           min_dead_pages, min_dead_bytes, min_dead_ratio,
                                           force, &cancel_requested_);
        if (stats.dead_pages_purged > 0) {
            const double freed_mb = static_cast<double>(stats.bytes_freed) / (1024.0 * 1024.0);
            std::cout << "[info] ninfer: [prompt-cache] compacted page pool: purged "
                      << stats.dead_pages_purged << " dead page(s), retained "
                      << stats.live_pages_retained << " live page(s) ("
                      << std::fixed << std::setprecision(1) << freed_mb << "MB freed in "
                      << stats.elapsed_ms << "ms)\n"
                      << std::flush;
        }
    }

    void cancel_in_flight() noexcept {
        const bool was_canceling = cancel_requested_.exchange(true, std::memory_order_acq_rel);
        std::size_t cleared = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            cleared = queue_.size();
            while (!queue_.empty()) {
                queue_.pop();
            }
            pending_hashes_.clear();
        }
        if (!was_canceling || cleared > 0) {
            std::cout << "[info] ninfer: [prompt-cache] incoming request arrived: prioritized execution (cleared "
                      << cleared << " queued snapshot save(s))\n" << std::flush;
        }
    }

private:
    enum class LoadFault : std::uint8_t {
        None,
        Transient, // The files are not implicated; keep the snapshot.
        Manifest,  // The manifest is unusable; its pool pages are intact.
        Pages,     // Pool bytes the manifest references could not be read.
    };

    struct LoadFailure {
        LoadFault kind     = LoadFault::None;
        const char* reason = "";
        // Page records to retire for LoadFault::Pages, on top of every record past the pool's end.
        std::vector<std::uint64_t> pages;
    };

    [[nodiscard]] bool manifest_pages_stored(const std::filesystem::path& path, const DiskStateHeader& header) const {
        if (header.text_page_count == 0) { return true; }
        std::unordered_set<std::uint64_t> hashes;
        read_manifest_page_hashes(path, header, hashes);
        if (hashes.empty()) { return false; } // Page list unreadable
        return std::all_of(hashes.begin(), hashes.end(),
                           [this](std::uint64_t hash) { return journal_.has_page(hash); });
    }

    static bool fail(LoadFailure& failure, LoadFault kind, const char* reason) {
        failure.kind   = kind;
        failure.reason = reason;
        return false;
    }

    // Distinguishes a file that is gone from one that is present but not accessible right now.
    [[nodiscard]] static bool file_missing(const std::filesystem::path& path) {
        std::error_code ec;
        return !std::filesystem::exists(path, ec) && !ec;
    }

    // Resolves the manifest's page records and checks that their bytes lie inside the pool
    // file, so a truncated pool is diagnosed before any read is issued. Records are fetched
    // before the pool is measured: a record becomes visible only after its bytes are flushed.
    [[nodiscard]] bool resolve_pages(std::span<const std::uint64_t> page_hashes, std::uint32_t page_bytes,
                                     std::vector<PageIndexRecord>& out_records, LoadFailure& failure) const {
        out_records.clear();
        out_records.reserve(page_hashes.size());
        for (const std::uint64_t hash : page_hashes) {
            const auto rec = journal_.get_page(hash);
            if (!rec) { return fail(failure, LoadFault::Manifest, "page missing from pool index"); }
            if (rec->page_bytes != page_bytes) {
                return fail(failure, LoadFault::Manifest, "pool page geometry disagrees with the header");
            }
            out_records.push_back(*rec);
        }
        if (out_records.empty()) { return true; }

        std::error_code ec;
        const std::uint64_t pool_bytes = std::filesystem::file_size(journal_.pool_data_path(), ec);
        if (ec) {
            return file_missing(journal_.pool_data_path())
                ? fail(failure, LoadFault::Pages, "page pool file missing")
                : fail(failure, LoadFault::Transient, "page pool file not accessible");
        }
        for (const auto& rec : out_records) {
            if (rec.file_offset + rec.page_bytes > pool_bytes) {
                return fail(failure, LoadFault::Pages, "page pool shorter than its index");
            }
        }
        return true;
    }

    [[nodiscard]] bool load_snapshot_host(const std::filesystem::path& path,
                                          DiskStateHeader& out_header,
                                          std::vector<TokenId>& out_tokens,
                                          std::vector<std::byte>& out_gdn_state,
                                          std::vector<std::byte>& out_text_kv_payload,
                                          std::vector<std::byte>& out_mtp_kv_payload,
                                          std::vector<std::byte>& out_tail_hidden,
                                          LoadFailure& failure) {
        std::error_code ec;
        const auto file_size = std::filesystem::file_size(path, ec);
        if (ec) {
            return file_missing(path) ? fail(failure, LoadFault::Manifest, "manifest missing")
                                      : fail(failure, LoadFault::Transient, "manifest not accessible");
        }

        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) { return fail(failure, LoadFault::Transient, "manifest could not be opened"); }

        DiskStateHeader header;
        file.read(reinterpret_cast<char*>(&header), sizeof(header));
        if (!file || !is_valid_header(header, file_size)) {
            return fail(failure, LoadFault::Manifest, "invalid or short manifest");
        }

        out_header = header;
        out_tokens.resize(header.token_count);
        if (header.token_count > 0) {
            file.read(reinterpret_cast<char*>(out_tokens.data()), header.token_count * sizeof(TokenId));
        }

        std::vector<std::uint64_t> page_hashes(header.text_page_count);
        if (header.text_page_count > 0) {
            file.read(reinterpret_cast<char*>(page_hashes.data()), header.text_page_count * sizeof(std::uint64_t));
        }
        if (!file) { return fail(failure, LoadFault::Manifest, "manifest token or page list unreadable"); }

        const auto t_load_0 = std::chrono::steady_clock::now();

        // 1. Read Text KV pages from shared pool_data.ninfer_pages
        out_text_kv_payload.resize(header.text_page_count * header.text_page_bytes);
        if (header.text_page_count > 0 && header.text_page_bytes > 0) {
            std::vector<PageIndexRecord> page_records;
            if (!resolve_pages(page_hashes, header.text_page_bytes, page_records, failure)) { return false; }

            std::ifstream pool_file(journal_.pool_data_path(), std::ios::binary);
            if (!pool_file.is_open()) {
                return fail(failure, LoadFault::Transient, "page pool file could not be opened");
            }

            for (std::uint32_t i = 0; i < header.text_page_count; ++i) {
                pool_file.seekg(static_cast<std::streamoff>(page_records[i].file_offset));
                pool_file.read(reinterpret_cast<char*>(out_text_kv_payload.data() + i * header.text_page_bytes),
                               header.text_page_bytes);
                if (!pool_file) {
                    failure.pages = page_hashes;
                    return fail(failure, LoadFault::Pages, "page pool read failed");
                }
            }
        }

        // 2. Read GDN state, MTP KV, tail hidden from manifest file
        const std::uint64_t raw_header_bytes = sizeof(DiskStateHeader) +
                                             static_cast<std::uint64_t>(header.token_count) * sizeof(TokenId) +
                                             static_cast<std::uint64_t>(header.text_page_count) * sizeof(std::uint64_t);
        const std::uint64_t payload_offset = header.payload_offset_bytes != 0
            ? static_cast<std::uint64_t>(header.payload_offset_bytes)
            : ((raw_header_bytes + 4095ULL) & ~4095ULL);
        file.seekg(static_cast<std::streamoff>(payload_offset));

        out_gdn_state.resize(header.gdn_state_bytes);
        if (header.gdn_state_bytes > 0) {
            file.read(reinterpret_cast<char*>(out_gdn_state.data()), header.gdn_state_bytes);
        }

        out_mtp_kv_payload.resize(header.mtp_kv_bytes);
        if (header.mtp_kv_bytes > 0) {
            file.read(reinterpret_cast<char*>(out_mtp_kv_payload.data()), header.mtp_kv_bytes);
        }

        out_tail_hidden.resize(header.tail_hidden_bytes);
        if (header.tail_hidden_bytes > 0) {
            file.read(reinterpret_cast<char*>(out_tail_hidden.data()), header.tail_hidden_bytes);
        }

        if (!file) { return fail(failure, LoadFault::Manifest, "manifest payload unreadable"); }

        if (header.checksum != 0) {
            Fnv1aHasher hasher;
            hasher.update(out_gdn_state.data(), out_gdn_state.size());
            hasher.update(out_mtp_kv_payload.data(), out_mtp_kv_payload.size());
            hasher.update(out_tail_hidden.data(), out_tail_hidden.size());
            const std::uint64_t actual_checksum = hasher.finalize();
            if (actual_checksum != header.checksum) {
                std::cerr << "[warn] ninfer: [prompt-cache] checksum mismatch on " << path.filename().string()
                          << " (expected 0x" << std::hex << header.checksum
                          << ", actual 0x" << actual_checksum << std::dec << ")\n";
                return fail(failure, LoadFault::Manifest, "checksum mismatch");
            }
        }

        const auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t_load_0).count();
        const std::size_t total_payload_bytes = out_text_kv_payload.size() + out_gdn_state.size() +
                                              out_mtp_kv_payload.size() + out_tail_hidden.size();
        const double raw_mb = static_cast<double>(total_payload_bytes) / (1024.0 * 1024.0);
        std::cout << "[info] ninfer: [prompt-cache] restored " << header.token_count
                  << " tokens from disk (" << std::fixed << std::setprecision(1) << raw_mb
                  << "MB in " << load_ms << "ms)\n" << std::flush;

        return true;
    }

#if defined(_WIN32)
    [[nodiscard]] bool load_snapshot_ds(const std::filesystem::path& path,
                                        DiskStateHeader& out_header,
                                        std::vector<TokenId>& out_tokens,
                                        cudaStream_t stream,
                                        std::span<const std::uint64_t> gdn_unit_bytes,
                                        const DirectStorageRestoreConsumer& consumer,
                                        std::uint32_t expected_text_page_bytes,
                                        LoadFailure& failure) {
        std::error_code ec;
        const auto file_size = std::filesystem::file_size(path, ec);
        if (ec) {
            return file_missing(path) ? fail(failure, LoadFault::Manifest, "manifest missing")
                                      : fail(failure, LoadFault::Transient, "manifest not accessible");
        }

        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) { return fail(failure, LoadFault::Transient, "manifest could not be opened"); }

        DiskStateHeader header;
        file.read(reinterpret_cast<char*>(&header), sizeof(header));
        if (!file || !is_valid_header(header, file_size)) {
            return fail(failure, LoadFault::Manifest, "invalid or short manifest");
        }

        if (!disk_state_layout_matches(header, expected_text_page_bytes)) {
            std::cerr << "[warn] ninfer: [prompt-cache] layout mismatch: header.text_page_bytes="
                      << header.text_page_bytes << " live_text_page_bytes=" << expected_text_page_bytes
                      << "\n" << std::flush;
            return fail(failure, LoadFault::Manifest, "page layout mismatch");
        }

        out_header = header;
        out_tokens.resize(header.token_count);
        if (header.token_count > 0) {
            file.read(reinterpret_cast<char*>(out_tokens.data()), header.token_count * sizeof(TokenId));
        }

        std::vector<std::uint64_t> page_hashes(header.text_page_count);
        if (header.text_page_count > 0) {
            file.read(reinterpret_cast<char*>(page_hashes.data()), header.text_page_count * sizeof(std::uint64_t));
        }
        if (!file) { return fail(failure, LoadFault::Manifest, "manifest token or page list unreadable"); }
        file.close();

        std::vector<PageIndexRecord> page_records;
        if (!resolve_pages(page_hashes, header.text_page_bytes, page_records, failure)) { return false; }

        // Segment list in file order, with the kind and per-kind index of each segment.
        std::vector<core::DirectStorageSegment> segments;
        std::vector<DirectStorageRestoreSlice> slice_meta;
        segments.reserve(header.text_page_count + gdn_unit_bytes.size() + header.mtp_page_count + 2);
        slice_meta.reserve(segments.capacity());
        const auto add_segment = [&](core::DirectStorageSource source, std::uint64_t file_offset,
                                     std::uint64_t bytes, DirectStorageRestoreKind kind, std::uint32_t index) {
            segments.push_back({source, file_offset, bytes});
            slice_meta.push_back({kind, index, nullptr, static_cast<std::size_t>(bytes)});
        };

        for (std::uint32_t i = 0; i < header.text_page_count; ++i) {
            add_segment(core::DirectStorageSource::Pool, page_records[i].file_offset, page_records[i].page_bytes,
                        DirectStorageRestoreKind::TextPage, i);
        }

        std::uint64_t manifest_offset = header.payload_offset_bytes;
        if (header.gdn_state_bytes > 0) {
            if (gdn_unit_bytes.empty()) {
                add_segment(core::DirectStorageSource::Manifest, manifest_offset, header.gdn_state_bytes,
                            DirectStorageRestoreKind::Gdn, 0);
            } else {
                std::uint64_t gdn_sum = 0;
                for (std::size_t i = 0; i < gdn_unit_bytes.size(); ++i) {
                    if (gdn_unit_bytes[i] == 0) {
                        return fail(failure, LoadFault::Transient, "live GDN unit has no bytes");
                    }
                    add_segment(core::DirectStorageSource::Manifest, manifest_offset + gdn_sum, gdn_unit_bytes[i],
                                DirectStorageRestoreKind::Gdn, static_cast<std::uint32_t>(i));
                    gdn_sum += gdn_unit_bytes[i];
                }
                if (gdn_sum != header.gdn_state_bytes) {
                    return fail(failure, LoadFault::Manifest, "GDN layout disagrees with the live state");
                }
            }
            manifest_offset += header.gdn_state_bytes;
        }
        if (header.mtp_kv_bytes > 0) {
            const std::uint32_t mtp_pages = std::max<std::uint32_t>(header.mtp_page_count, 1);
            if (header.mtp_kv_bytes % mtp_pages != 0) {
                return fail(failure, LoadFault::Manifest, "MTP payload not divisible into pages");
            }
            const std::uint64_t mtp_page_bytes = header.mtp_kv_bytes / mtp_pages;
            for (std::uint32_t i = 0; i < mtp_pages; ++i) {
                add_segment(core::DirectStorageSource::Manifest, manifest_offset + i * mtp_page_bytes, mtp_page_bytes,
                            DirectStorageRestoreKind::MtpPage, i);
            }
            manifest_offset += header.mtp_kv_bytes;
        }
        if (header.tail_hidden_bytes > 0) {
            add_segment(core::DirectStorageSource::Manifest, manifest_offset, header.tail_hidden_bytes,
                        DirectStorageRestoreKind::Tail, 0);
        }

        if (segments.empty()) { return fail(failure, LoadFault::Transient, "snapshot has no payload"); }
        if (!core::DirectStorageEngine::instance().available()) {
            return fail(failure, LoadFault::Transient, "DirectStorage unavailable");
        }

        const std::uint64_t manifest_payload_bytes = header.gdn_state_bytes + header.mtp_kv_bytes + header.tail_hidden_bytes;
        const auto t_load_0 = std::chrono::steady_clock::now();

        std::vector<DirectStorageRestoreSlice> batch;
        const auto status = core::DirectStorageEngine::instance().restore_segments(
            journal_.pool_data_path(), path, segments, stream,
            [&](const std::byte* staging, std::size_t first, std::span<const std::size_t> offsets) {
                batch.clear();
                for (std::size_t i = 0; i < offsets.size(); ++i) {
                    DirectStorageRestoreSlice slice = slice_meta[first + i];
                    slice.data = staging + offsets[i];
                    batch.push_back(slice);
                }
                return consumer(std::span<const DirectStorageRestoreSlice>(batch.data(), batch.size()));
            });
        switch (status) {
        case core::DirectStorageRestoreStatus::Ok:
            break;
        case core::DirectStorageRestoreStatus::Unavailable:
            return fail(failure, LoadFault::Transient, "DirectStorage unavailable");
        case core::DirectStorageRestoreStatus::OpenFailed:
            // Both files were present a moment ago, so this is a sharing or access failure.
            return fail(failure, LoadFault::Transient, "DirectStorage could not open a snapshot file");
        case core::DirectStorageRestoreStatus::ReadFailed:
            // The error record names only the first failed request, not which page or manifest
            // range it covered, so every page of the snapshot is retired with the manifest.
            failure.pages = page_hashes;
            return fail(failure, LoadFault::Pages, "DirectStorage read failed");
        case core::DirectStorageRestoreStatus::ConsumerRejected:
            return fail(failure, LoadFault::Manifest, "restore slices disagree with the live layout");
        }

        const auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t_load_0).count();
        const std::size_t total_payload_bytes = header.text_page_count * header.text_page_bytes + manifest_payload_bytes;
        const double raw_mb = static_cast<double>(total_payload_bytes) / (1024.0 * 1024.0);
        std::cout << "[info] ninfer: [prompt-cache] directstorage restored " << header.token_count
                  << " tokens into VRAM (" << std::fixed << std::setprecision(1) << raw_mb
                  << "MB in " << load_ms << "ms)\n" << std::flush;

        return true;
    }
#endif

    // Without this, a bad snapshot stays in the index: enqueue_save_cow skips the fallback's
    // save as a duplicate and has_page keeps reporting the unreadable pages as stored, so every
    // matching request repeats the failed load and a full prefill.
    void discard_after_failed_load(const std::filesystem::path& path, const LoadFailure& failure,
                                   std::uint64_t journal_generation) {
        if (failure.kind == LoadFault::None) { return; }
        if (failure.kind == LoadFault::Transient) {
            std::cerr << "[warn] ninfer: [prompt-cache] kept snapshot " << path.filename().string()
                      << " after a transient load failure (" << failure.reason << ")\n" << std::flush;
            return;
        }

        std::size_t retired_pages = 0;
        if (failure.kind == LoadFault::Pages) {
            const auto retired = journal_.drop_unreadable_pages(failure.pages, journal_generation);
            if (!retired) {
                std::cerr << "[warn] ninfer: [prompt-cache] kept snapshot " << path.filename().string()
                          << " (" << failure.reason << "): the page pool was compacted during the load\n"
                          << std::flush;
                return;
            }
            retired_pages = *retired;
        }

        bool removed = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            removed = index_.erase(path) > 0;
            std::error_code ec;
            removed = std::filesystem::remove(path, ec) || removed;
        }
        if (removed || retired_pages > 0) {
            std::cerr << "[warn] ninfer: [prompt-cache] dropped snapshot " << path.filename().string()
                      << " (" << failure.reason << "), retired " << retired_pages
                      << " pool page record(s)\n" << std::flush;
        }
    }

    void scan_cache_dir() {
        std::error_code ec;
        const auto manifest_dir = config_.cache_dir / "manifests";
        std::size_t removed_invalid_manifests = 0;
        for (const auto& entry : std::filesystem::directory_iterator(manifest_dir, ec)) {
            if (entry.is_regular_file() && entry.path().extension() == ".tmp") {
                std::filesystem::remove(entry.path(), ec);
                continue;
            }
            if (entry.is_regular_file() && entry.path().extension() == ".ninfer_manifest") {
                const auto file_size = entry.file_size(ec);
                std::ifstream f(entry.path(), std::ios::binary);
                bool valid = false;
                if (!ec && f.is_open()) {
                    DiskStateHeader h;
                    f.read(reinterpret_cast<char*>(&h), sizeof(h));
                    if (f && is_valid_header(h, file_size)) {
                        std::unordered_set<std::uint64_t> hashes;
                        read_manifest_page_hashes(entry.path(), h, hashes);
                        bool all_pages_exist = true;
                        for (std::uint64_t ph : hashes) {
                            if (!journal_.has_page(ph)) {
                                all_pages_exist = false;
                                break;
                            }
                        }
                        if (all_pages_exist) {
                            index_[entry.path()] = h;
                            valid                = true;
                        }
                    }
                }
                if (!valid) {
                    std::filesystem::remove(entry.path(), ec);
                    ++removed_invalid_manifests;
                }
            }
        }

        if (removed_invalid_manifests > 0) {
            std::cout << "[info] ninfer: [prompt-cache] removed " << removed_invalid_manifests
                      << " invalid/incomplete manifest(s) on startup\n" << std::flush;
        }

        // On startup: index existing valid manifests. Startup is zero-I/O and finishes in <5ms.
    }

    void background_writer_loop(std::stop_token stop) {
        while (!stop.stop_requested()) {
            QueuedSaveTask task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this, &stop] { return !running_ || stop.stop_requested() || !queue_.empty(); });
                if ((!running_ || stop.stop_requested()) && queue_.empty()) { break; }
                if (queue_.empty()) { continue; }
                task = std::move(queue_.front());
                pending_hashes_.erase({task.header.prompt_hash, task.header.token_count});
                queue_.pop();
            }

            cancel_requested_.store(false, std::memory_order_relaxed);
            const auto t_save_0 = std::chrono::steady_clock::now();

            if (cancel_requested_.load(std::memory_order_acquire)) {
                std::cout << "[info] ninfer: [prompt-cache] aborted in-flight snapshot save tokens="
                          << task.header.token_count << " (discarded from memory)\n" << std::flush;
                continue;
            }

            // 1. Append missing Text KV pages to shared PageStoreJournal
            if (!journal_.append_missing_pages(task.all_page_hashes, task.missing_page_hashes,
                                               task.missing_pages_data, task.single_page_bytes)) {
                std::cerr << "[warn] ninfer: [prompt-cache] skipped snapshot save tokens="
                          << task.header.token_count << ": pool pages it reuses are no longer stored\n"
                          << std::flush;
                task.lease.release();
                prune_lru();
                continue;
            }

            // 2. Write snapshot manifest file
            const std::filesystem::path temp_path = task.manifest_path.string() + ".tmp";
            std::ofstream out(temp_path, std::ios::binary | std::ios::trunc);
            if (out.is_open()) {
                out.write(reinterpret_cast<const char*>(&task.header), sizeof(task.header));
                if (!task.tokens.empty()) {
                    out.write(reinterpret_cast<const char*>(task.tokens.data()), task.tokens.size() * sizeof(TokenId));
                }
                if (!task.all_page_hashes.empty()) {
                    out.write(reinterpret_cast<const char*>(task.all_page_hashes.data()),
                              task.all_page_hashes.size() * sizeof(std::uint64_t));
                }

                // 4 KiB sector padding for DirectStorage DMA
                const std::uint64_t written_so_far = sizeof(DiskStateHeader) +
                                                     task.tokens.size() * sizeof(TokenId) +
                                                     task.all_page_hashes.size() * sizeof(std::uint64_t);
                if (task.header.payload_offset_bytes > written_so_far) {
                    const std::size_t pad_size = task.header.payload_offset_bytes - written_so_far;
                    static const std::vector<char> zeros(4096, 0);
                    out.write(zeros.data(), pad_size);
                }

                if (!task.gdn_state.empty()) {
                    out.write(reinterpret_cast<const char*>(task.gdn_state.data()), task.gdn_state.size());
                }
                if (!task.mtp_kv_payload.empty()) {
                    out.write(reinterpret_cast<const char*>(task.mtp_kv_payload.data()), task.mtp_kv_payload.size());
                }
                if (!task.tail_hidden.empty()) {
                    out.write(reinterpret_cast<const char*>(task.tail_hidden.data()), task.tail_hidden.size());
                }
                out.flush();
                out.close();

                // Deliberately not cancelled here. cancel_requested_ is a latency-priority signal
                // raised when a request arrives - it does not invalidate the snapshot, whose data was
                // captured before the task was queued. By this point the expensive work is already
                // spent: the missing pages are committed to the shared journal and the manifest is
                // written and flushed. All that remains is a remove and a rename, microseconds of
                // filesystem metadata. Discarding here forfeited 100-160 ms of completed I/O per save
                // and, worse, orphaned the pages just appended - they stayed in the pool with no
                // manifest referencing them. Under sustained load that starved the index of recent
                // checkpoints, so restores matched stale early turns and re-prefilled the remainder.
                // The pre-work check above is the one that genuinely avoids cost; once past it, finish.
                std::error_code ec;
                std::filesystem::remove(task.manifest_path, ec);
                std::filesystem::rename(temp_path, task.manifest_path, ec);

                if (!ec) {
                    std::lock_guard<std::mutex> lock(mutex_);
                    index_[task.manifest_path] = task.header;
                }

                const auto save_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t_save_0).count();
                const std::size_t written_bytes = task.missing_pages_data.size() + task.gdn_state.size() +
                                                  task.mtp_kv_payload.size() + task.tail_hidden.size();
                const double written_mb = static_cast<double>(written_bytes) / (1024.0 * 1024.0);
                std::cout << "[info] ninfer: [prompt-cache] saved CoW snapshot tokens=" << task.header.token_count
                          << " (new pages=" << task.missing_page_hashes.size() << ", "
                          << std::fixed << std::setprecision(1) << written_mb << "MB) in "
                          << save_ms << "ms\n" << std::flush;
            }

            // The manifest is indexed or abandoned, so the pages need no lease; holding it would
            // make the compaction below keep them even when the manifest is evicted.
            task.lease.release();
            prune_lru();
        }
    }

    DiskStateCacheConfig config_;
    PageStoreJournal journal_;
    bool running_ = true;
    std::atomic<bool> cancel_requested_{false};
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<QueuedSaveTask> queue_;
    std::set<std::pair<std::uint64_t, std::uint32_t>> pending_hashes_;
    std::map<std::filesystem::path, DiskStateHeader> index_;
    std::jthread worker_;
};

DiskStateCache::DiskStateCache(DiskStateCacheConfig config)
    : impl_(std::make_unique<Impl>(config)), config_(std::move(config)) {}

DiskStateCache::~DiskStateCache() = default;

DiskStateCache::PageLease::PageLease(PageLease&& other) noexcept
    : owner_(std::exchange(other.owner_, nullptr)),
      hashes_(std::move(other.hashes_)),
      stored_(std::move(other.stored_)) {}

DiskStateCache::PageLease& DiskStateCache::PageLease::operator=(PageLease&& other) noexcept {
    if (this != &other) {
        release();
        owner_  = std::exchange(other.owner_, nullptr);
        hashes_ = std::move(other.hashes_);
        stored_ = std::move(other.stored_);
    }
    return *this;
}

DiskStateCache::PageLease::~PageLease() {
    release();
}

void DiskStateCache::PageLease::release() noexcept {
    if (owner_ != nullptr) {
        owner_->unpin_pages(hashes_);
        owner_ = nullptr;
    }
}

bool DiskStateCache::has_page(std::uint64_t page_hash) const {
    return impl_->has_page(page_hash);
}

DiskStateCache::PageLease DiskStateCache::lease_pages(std::span<const std::uint64_t> page_hashes) {
    return impl_->lease_pages(page_hashes);
}

std::optional<DiskStateMatch>
DiskStateCache::find_longest_matching_prefix(std::uint64_t model_hash, std::span<const TokenId> tokens) const {
    return impl_->find_longest_matching_prefix(model_hash, tokens);
}

bool DiskStateCache::load_snapshot(const std::filesystem::path& path,
                                   DiskStateHeader& out_header,
                                   std::vector<TokenId>& out_tokens,
                                   std::vector<std::byte>& out_gdn_state,
                                   std::vector<std::byte>& out_text_kv_payload,
                                   std::vector<std::byte>& out_mtp_kv_payload,
                                   std::vector<std::byte>& out_tail_hidden) {
    return impl_->load_snapshot(path, out_header, out_tokens, out_gdn_state,
                                out_text_kv_payload, out_mtp_kv_payload, out_tail_hidden);
}

#if defined(_WIN32)
bool DiskStateCache::load_snapshot_direct_storage(const std::filesystem::path& path,
                                                  DiskStateHeader& out_header,
                                                  std::vector<TokenId>& out_tokens,
                                                  cudaStream_t stream,
                                                  std::span<const std::uint64_t> gdn_unit_bytes,
                                                  const DirectStorageRestoreConsumer& consumer,
                                                  std::uint32_t expected_text_page_bytes) {
    return impl_->load_snapshot_direct_storage(path, out_header, out_tokens, stream, gdn_unit_bytes, consumer, expected_text_page_bytes);
}
#endif

void DiskStateCache::discard_snapshot(const std::filesystem::path& path, const char* reason) {
    impl_->discard_snapshot(path, reason);
}

void DiskStateCache::enqueue_save_cow(std::uint64_t model_hash,
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
                                      PageLease lease) {
    impl_->enqueue_save_cow(model_hash, std::move(tokens), turn_index, rope_delta,
                            std::move(gdn_state), std::move(all_page_hashes),
                            std::move(missing_page_hashes), std::move(missing_pages_data),
                            single_page_bytes, std::move(mtp_kv_payload),
                            mtp_page_count, std::move(tail_hidden), std::move(lease));
}

void DiskStateCache::enqueue_save(std::uint64_t model_hash,
                                  std::vector<TokenId> tokens,
                                  std::uint32_t turn_index,
                                  std::int32_t rope_delta,
                                  std::vector<std::byte> gdn_state,
                                  std::vector<std::byte> text_kv_payload,
                                  std::uint32_t text_page_count,
                                  std::vector<std::byte> mtp_kv_payload,
                                  std::uint32_t mtp_page_count,
                                  std::vector<std::byte> tail_hidden) {
    impl_->enqueue_save(model_hash, std::move(tokens), turn_index, rope_delta,
                        std::move(gdn_state), std::move(text_kv_payload), text_page_count,
                        std::move(mtp_kv_payload), mtp_page_count, std::move(tail_hidden));
}

void DiskStateCache::enqueue_save(std::uint64_t model_hash,
                                  std::span<const TokenId> tokens,
                                  std::uint32_t turn_index,
                                  std::int32_t rope_delta,
                                  std::span<const std::byte> gdn_state,
                                  std::span<const std::byte> text_kv_payload,
                                  std::uint32_t text_page_count,
                                  std::span<const std::byte> mtp_kv_payload,
                                  std::uint32_t mtp_page_count,
                                  std::span<const std::byte> tail_hidden) {
    impl_->enqueue_save(model_hash, tokens, turn_index, rope_delta, gdn_state,
                        text_kv_payload, text_page_count, mtp_kv_payload, mtp_page_count,
                        tail_hidden);
}

void DiskStateCache::prune_lru() {
    impl_->prune_lru();
}

void DiskStateCache::compact_pool(bool force) {
    impl_->compact_pool(force);
}

void DiskStateCache::cancel_in_flight() noexcept {
    impl_->cancel_in_flight();
}

} // namespace ninfer
