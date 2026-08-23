#include "core/disk_state_cache.h"
#if defined(_WIN32)
#include "core/direct_storage_engine.h"
#endif

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <fstream>
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

} // namespace

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

    void append_missing_pages(std::span<const std::uint64_t> missing_hashes,
                              std::span<const std::byte> missing_data,
                              std::uint32_t single_page_bytes) {
        if (missing_hashes.empty() || single_page_bytes == 0) {
            return;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        std::error_code ec;
        std::uint64_t curr_file_size = 0;
        if (std::filesystem::exists(pool_data_path_, ec)) {
            curr_file_size = std::filesystem::file_size(pool_data_path_, ec);
        }

        std::ofstream data_file(pool_data_path_, std::ios::binary | std::ios::app);
        std::ofstream index_file(pool_index_path_, std::ios::binary | std::ios::app);
        if (!data_file.is_open() || !index_file.is_open()) {
            return;
        }

        const std::uint64_t aligned_page_bytes = (static_cast<std::uint64_t>(single_page_bytes) + 4095ULL) & ~4095ULL;
        const std::size_t pad_size = static_cast<std::size_t>(aligned_page_bytes - single_page_bytes);
        static const std::vector<char> zeros(4096, 0);

        for (std::size_t i = 0; i < missing_hashes.size(); ++i) {
            const std::uint64_t hash = missing_hashes[i];
            if (page_index_.find(hash) != page_index_.end()) {
                continue; // Already stored
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
        }

        data_file.flush();
        index_file.flush();
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

    CompactStats compact(const std::unordered_set<std::uint64_t>& live_hashes,
                         std::size_t min_dead_pages = 256,
                         std::uint64_t min_dead_bytes = 2ULL << 30, // 2 GiB minimum dead space
                         double min_dead_ratio = 0.25,              // 25% minimum fragmentation ratio
                         bool force = false,
                         const std::atomic<bool>* cancel_token = nullptr) {
        CompactStats stats{};
        std::error_code ec;

        std::unordered_map<std::uint64_t, PageIndexRecord> local_index;
        std::uint64_t old_file_size = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!std::filesystem::exists(pool_data_path_, ec)) {
                page_index_.clear();
                std::filesystem::remove(pool_index_path_, ec);
                return stats;
            }
            old_file_size = std::filesystem::file_size(pool_data_path_, ec);
            if (live_hashes.empty()) {
                stats.dead_pages_purged = page_index_.size();
                stats.bytes_freed       = old_file_size;
                page_index_.clear();
                std::filesystem::remove(pool_data_path_, ec);
                std::filesystem::remove(pool_index_path_, ec);
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

        for (const auto& [hash, old_rec] : local_index) {
            if (cancel_token && cancel_token->load(std::memory_order_acquire)) {
                src_data.close();
                dst_data.close();
                dst_index.close();
                std::filesystem::remove(temp_data_path, ec);
                std::filesystem::remove(temp_index_path, ec);
                stats.dead_pages_purged = 0;
                stats.bytes_freed       = 0;
                stats.live_pages_retained = local_index.size();
                return stats;
            }

            if (!live_hashes.contains(hash)) {
                ++stats.dead_pages_purged;
                continue;
            }

            const std::uint64_t aligned_page_bytes =
                (static_cast<std::uint64_t>(old_rec.page_bytes) + 4095ULL) & ~4095ULL;
            if (buffer.size() < aligned_page_bytes) {
                buffer.resize(aligned_page_bytes);
            }

            src_data.seekg(static_cast<std::streamoff>(old_rec.file_offset));
            if (!src_data.read(buffer.data(), static_cast<std::streamsize>(aligned_page_bytes))) {
                continue;
            }

            dst_data.write(buffer.data(), static_cast<std::streamsize>(aligned_page_bytes));

            PageIndexRecord new_rec;
            new_rec.page_hash   = hash;
            new_rec.file_offset = new_file_size;
            new_rec.page_bytes  = old_rec.page_bytes;
            new_rec.reserved    = 0;

            dst_index.write(reinterpret_cast<const char*>(&new_rec), sizeof(new_rec));
            new_page_index[hash] = new_rec;
            new_file_size += aligned_page_bytes;
            ++stats.live_pages_retained;
        }

        dst_data.flush();
        dst_index.flush();
        src_data.close();
        dst_data.close();
        dst_index.close();

        if (cancel_token && cancel_token->load(std::memory_order_acquire)) {
            std::filesystem::remove(temp_data_path, ec);
            std::filesystem::remove(temp_index_path, ec);
            stats.dead_pages_purged = 0;
            stats.bytes_freed       = 0;
            stats.live_pages_retained = local_index.size();
            return stats;
        }

        // Atomic swap under journal lock
        {
            std::lock_guard<std::mutex> lock(mutex_);
            std::filesystem::remove(pool_data_path_, ec);
            std::filesystem::rename(temp_data_path, pool_data_path_, ec);
            std::filesystem::remove(pool_index_path_, ec);
            std::filesystem::rename(temp_index_path, pool_index_path_, ec);

            page_index_ = std::move(new_page_index);
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
        std::ifstream index_file(pool_index_path_, std::ios::binary);
        if (!index_file.is_open()) {
            return;
        }

        PageIndexRecord rec;
        while (index_file.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
            page_index_[rec.page_hash] = rec;
        }
    }

    std::filesystem::path pool_data_path_;
    std::filesystem::path pool_index_path_;
    mutable std::mutex mutex_;
    std::unordered_map<std::uint64_t, PageIndexRecord> page_index_;
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
        std::error_code ec;
        const auto file_size = std::filesystem::file_size(path, ec);
        if (ec) { return false; }

        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) { return false; }

        DiskStateHeader header;
        file.read(reinterpret_cast<char*>(&header), sizeof(header));
        if (!file || !is_valid_header(header, file_size)) {
            return false;
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

        const auto t_load_0 = std::chrono::steady_clock::now();

        // 1. Read Text KV pages from shared pool_data.ninfer_pages
        out_text_kv_payload.resize(header.text_page_count * header.text_page_bytes);
        if (header.text_page_count > 0 && header.text_page_bytes > 0) {
            std::ifstream pool_file(journal_.pool_data_path(), std::ios::binary);
            if (!pool_file.is_open()) { return false; }

            for (std::uint32_t i = 0; i < header.text_page_count; ++i) {
                auto page_rec = journal_.get_page(page_hashes[i]);
                if (!page_rec) { return false; }

                pool_file.seekg(static_cast<std::streamoff>(page_rec->file_offset));
                pool_file.read(reinterpret_cast<char*>(out_text_kv_payload.data() + i * header.text_page_bytes),
                               header.text_page_bytes);
                if (!pool_file) { return false; }
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

        if (!file) { return false; }

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
                return false;
            }
        }

        const auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t_load_0).count();
        const std::size_t total_payload_bytes = out_text_kv_payload.size() + out_gdn_state.size() +
                                              out_mtp_kv_payload.size() + out_tail_hidden.size();
        const double raw_mb = static_cast<double>(total_payload_bytes) / (1024.0 * 1024.0);
        std::cout << "[info] ninfer: [prompt-cache] restored " << header.token_count
                  << " tokens from disk (" << std::fixed << std::setprecision(1) << raw_mb
                  << "MB in " << load_ms << "ms)\n" << std::flush;

        return file.good() || file.eof();
    }

#if defined(_WIN32)
    [[nodiscard]] bool load_snapshot_direct_storage(const std::filesystem::path& path,
                                                    DiskStateHeader& out_header,
                                                    std::vector<TokenId>& out_tokens,
                                                    cudaStream_t stream,
                                                    void*& out_d_staging_ptr,
                                                    std::size_t& out_text_bytes) {
        std::error_code ec;
        const auto file_size = std::filesystem::file_size(path, ec);
        if (ec) { return false; }

        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) { return false; }

        DiskStateHeader header;
        file.read(reinterpret_cast<char*>(&header), sizeof(header));
        if (!file || !is_valid_header(header, file_size)) {
            return false;
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
        if (!file) { return false; }

        std::vector<core::PageRestoreEntry> text_pages;
        text_pages.reserve(header.text_page_count);
        for (std::uint64_t phash : page_hashes) {
            auto page_rec = journal_.get_page(phash);
            if (!page_rec) {
                return false; // Incomplete journal
            }
            text_pages.push_back({page_rec->file_offset, page_rec->page_bytes});
        }

        if (!core::DirectStorageEngine::instance().available()) {
            return false;
        }

        const std::uint64_t manifest_payload_bytes = header.gdn_state_bytes + header.mtp_kv_bytes + header.tail_hidden_bytes;
        const auto t_load_0 = std::chrono::steady_clock::now();

        const bool ok = core::DirectStorageEngine::instance().restore_snapshot_cow(
            journal_.pool_data_path(), text_pages, path, header.payload_offset_bytes,
            manifest_payload_bytes, stream, out_d_staging_ptr, out_text_bytes
        );
        if (!ok) { return false; }

        const auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t_load_0).count();
        const std::size_t total_payload_bytes = header.text_page_count * header.text_page_bytes + manifest_payload_bytes;
        const double raw_mb = static_cast<double>(total_payload_bytes) / (1024.0 * 1024.0);
        std::cout << "[info] ninfer: [prompt-cache] directstorage restored " << header.token_count
                  << " tokens into VRAM (" << std::fixed << std::setprecision(1) << raw_mb
                  << "MB in " << load_ms << "ms)\n" << std::flush;

        return true;
    }
#endif

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
                          std::vector<std::byte> tail_hidden) {
        if (!config_.enabled || tokens.empty()) { return; }

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

        constexpr std::size_t kMaxQueuedTasks = 8;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (const auto& [path, header] : index_) {
                if (header.model_hash == task.header.model_hash &&
                    header.prompt_hash == task.header.prompt_hash &&
                    header.token_count == task.header.token_count) {
                    return;
                }
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
        std::vector<std::uint64_t> missing_page_hashes;
        std::vector<std::byte> missing_pages_data;

        for (std::uint32_t i = 0; i < text_page_count; ++i) {
            const std::size_t span_end = std::min<std::size_t>(tokens.size(), static_cast<std::size_t>(i + 1) * 64);
            const std::uint64_t phash = hash_prompt_prefix(std::span<const TokenId>(tokens.data(), span_end));
            all_page_hashes.push_back(phash);

            if (!journal_.has_page(phash) && single_page_bytes > 0) {
                missing_page_hashes.push_back(phash);
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
                         mtp_page_count, std::move(tail_hidden));
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
        std::unordered_set<std::uint64_t> live_hashes;
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

            if (evicted_count > 0 || pool_bytes > (config_.max_cache_bytes * 4 / 10)) {
                should_compact = true;
                for (const auto& [manifest_path, header] : index_) {
                    read_manifest_page_hashes(manifest_path, header, live_hashes);
                }
            }
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
            const auto stats = journal_.compact(live_hashes, kMinDeadPages, kMinDeadBytes, kMinDeadRatio,
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

    void compact_pool(bool force = true) {
        std::unordered_set<std::uint64_t> live_hashes;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (const auto& [manifest_path, header] : index_) {
                read_manifest_page_hashes(manifest_path, header, live_hashes);
            }
        }
        const std::size_t min_dead_pages   = force ? 1 : 256;
        const std::uint64_t min_dead_bytes = force ? 0 : (2ULL << 30);
        const double min_dead_ratio        = force ? 0.0 : 0.25;
        const auto stats = journal_.compact(live_hashes, min_dead_pages, min_dead_bytes,
                                           min_dead_ratio, force, &cancel_requested_);
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
            if (!task.missing_page_hashes.empty()) {
                journal_.append_missing_pages(task.missing_page_hashes, task.missing_pages_data, task.single_page_bytes);
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

                if (cancel_requested_.load(std::memory_order_acquire)) {
                    std::error_code ec;
                    std::filesystem::remove(temp_path, ec);
                    std::cout << "[info] ninfer: [prompt-cache] aborted in-flight snapshot save tokens="
                              << task.header.token_count << " (discarded temp file)\n" << std::flush;
                    continue;
                }

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

bool DiskStateCache::has_page(std::uint64_t page_hash) const {
    return impl_->has_page(page_hash);
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
                                                  void*& out_d_staging_ptr,
                                                  std::size_t& out_text_bytes) {
    return impl_->load_snapshot_direct_storage(path, out_header, out_tokens, stream, out_d_staging_ptr, out_text_bytes);
}
#endif

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
                                      std::vector<std::byte> tail_hidden) {
    impl_->enqueue_save_cow(model_hash, std::move(tokens), turn_index, rope_delta,
                            std::move(gdn_state), std::move(all_page_hashes),
                            std::move(missing_page_hashes), std::move(missing_pages_data),
                            single_page_bytes, std::move(mtp_kv_payload),
                            mtp_page_count, std::move(tail_hidden));
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
