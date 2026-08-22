#include "core/disk_state_cache.h"
#include "core/device.h"
#include "core/linear_attention_state.h"
#include "core/paged_kv_cache.h"
#if defined(_WIN32)
#include "core/direct_storage_engine.h"
#endif

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <span>
#include <thread>
#include <vector>

using namespace ninfer;

namespace {

int failures = 0;

void expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

std::vector<std::byte> generate_deterministic_bytes(std::size_t size, std::uint32_t seed) {
    std::vector<std::byte> buffer(size);
    std::mt19937 rng(seed);
    for (std::size_t i = 0; i < size; ++i) {
        buffer[i] = static_cast<std::byte>(rng() & 0xFF);
    }
    return buffer;
}

std::vector<TokenId> generate_tokens(std::size_t count, std::uint32_t seed) {
    std::vector<TokenId> tokens(count);
    std::mt19937 rng(seed);
    for (std::size_t i = 0; i < count; ++i) {
        tokens[i] = static_cast<TokenId>(100 + (rng() % 50000));
    }
    return tokens;
}

std::optional<DiskStateMatch> wait_for_match(const DiskStateCache& cache,
                                             uint64_t model_hash,
                                             std::span<const TokenId> tokens,
                                             int max_ms = 3000) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(max_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        auto match = cache.find_longest_matching_prefix(model_hash, tokens);
        if (match.has_value() && match->matched_tokens == tokens.size()) {
            return match;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return cache.find_longest_matching_prefix(model_hash, tokens);
}

} // namespace

void test_prompt_hashing() {
    std::vector<TokenId> tokens1 = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<TokenId> tokens2 = {1, 2, 3, 4, 5, 6, 7, 9};
    std::vector<TokenId> prefix  = {1, 2, 3, 4};

    const uint64_t h1 = DiskStateCache::hash_prompt_prefix(tokens1);
    const uint64_t h2 = DiskStateCache::hash_prompt_prefix(tokens2);
    const uint64_t hp = DiskStateCache::hash_prompt_prefix(prefix);

    expect(h1 != h2, "Different prompts must produce distinct hashes");
    expect(h1 != hp, "Prefix must produce distinct hash from full prompt");
    expect(DiskStateCache::hash_prompt_prefix(tokens1) == h1, "Hash must be deterministic");
}

void test_disabled_by_default() {
    const std::filesystem::path test_dir = "./.test_ninfer_cache_disabled";
    std::error_code ec;
    std::filesystem::remove_all(test_dir, ec);

    DiskStateCacheConfig config;
    config.cache_dir = test_dir;
    expect(!config.enabled, "DiskStateCacheConfig must be disabled by default");

    {
        DiskStateCache cache(config);
        expect(!cache.enabled(), "DiskStateCache must report enabled() == false");

        const auto tokens = generate_tokens(100, 1);
        const auto dummy  = generate_deterministic_bytes(1024, 2);

        cache.enqueue_save(0x1234, tokens, 1, 0, dummy, dummy, 1, {}, 0, dummy);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        auto match = cache.find_longest_matching_prefix(0x1234, tokens);
        expect(!match.has_value(), "Disabled cache must never return matches");
        expect(!std::filesystem::exists(test_dir), "Disabled cache must not create cache directory on disk");
    }

    std::filesystem::remove_all(test_dir, ec);
}

void test_timestamped_filename_and_datetime() {
    const std::filesystem::path test_dir = "./.test_ninfer_cache_time";
    std::error_code ec;
    std::filesystem::remove_all(test_dir, ec);

    DiskStateCacheConfig config;
    config.cache_dir = test_dir;
    config.enabled   = true;

    {
        DiskStateCache cache(config);
        const auto tokens = generate_tokens(256, 42);
        const auto dummy  = generate_deterministic_bytes(1024, 1);

        cache.enqueue_save(0x9999ULL, tokens, 1, 0, dummy, dummy, 1, {}, 0, dummy);
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        auto match = cache.find_longest_matching_prefix(0x9999ULL, tokens);
        expect(match.has_value(), "Must find snapshot with timestamped filename");
        expect(match->header.created_at_utc > 0, "Header created_at_utc must be populated");

        const std::string filename = match->file_path.filename().string();
        expect(filename.rfind("state_", 0) == 0, "Filename must start with state_");
        expect(filename.find("_t256.ninfer_manifest") != std::string::npos, "Filename must contain token suffix");
    }

    std::filesystem::remove_all(test_dir, ec);
}

void test_corruption_and_version_invalidation() {
    const std::filesystem::path test_dir = "./.test_ninfer_cache_corrupt";
    std::error_code ec;
    std::filesystem::remove_all(test_dir, ec);
    std::filesystem::create_directories(test_dir / "manifests", ec);

    // 1. File shorter than header
    {
        std::ofstream out(test_dir / "manifests" / "state_01_t10.ninfer_manifest", std::ios::binary);
        const char garbage[] = "SHORT_GARBAGE";
        out.write(garbage, sizeof(garbage));
    }

    // 2. Old/mismatched version file
    {
        std::ofstream out(test_dir / "manifests" / "state_02_t10.ninfer_manifest", std::ios::binary);
        DiskStateHeader h;
        h.version = 0; // Mismatched version
        out.write(reinterpret_cast<const char*>(&h), sizeof(h));
    }

    // 3. Astronomical/malicious gdn_state_bytes that would cause bad_alloc
    {
        std::ofstream out(test_dir / "manifests" / "state_03_t10.ninfer_manifest", std::ios::binary);
        DiskStateHeader h;
        h.gdn_state_bytes = 0xFFFFFFFFFFFFFFFFULL;
        out.write(reinterpret_cast<const char*>(&h), sizeof(h));
    }

    DiskStateCacheConfig config;
    config.cache_dir       = test_dir;
    config.max_cache_bytes = 100ULL << 20;
    config.enabled         = true;

    {
        DiskStateCache cache(config);
        auto match = cache.find_longest_matching_prefix(0x0123, std::vector<TokenId>(100, 1));
        expect(!match.has_value(), "Corrupted and outdated files must be rejected and not matched");

        DiskStateHeader out_h;
        std::vector<TokenId> out_toks;
        std::vector<std::byte> out_gdn, out_kv, out_mtp, out_tail;
        const bool load_ok = cache.load_snapshot(test_dir / "manifests" / "state_03_t10.ninfer_manifest", out_h,
                                                 out_toks, out_gdn, out_kv, out_mtp, out_tail);
        expect(!load_ok, "load_snapshot on corrupted payload must fail safely without bad_alloc");
    }

    std::filesystem::remove_all(test_dir, ec);
}

void test_multi_config_isolation_and_cross_contamination() {
    const std::filesystem::path root_dir = "./.test_ninfer_cache_isolation";
    const std::filesystem::path config_a_dir = root_dir / "qwen3_8_27b_rk2v4-e8_mtp_d3_ctx450k";
    const std::filesystem::path config_b_dir = root_dir / "qwen3_8_27b_int8_none_ctx450k";
    std::error_code ec;
    std::filesystem::remove_all(root_dir, ec);

    const uint64_t model_hash_a = 0xAAAAAAAAAAAAAAAAULL;
    const uint64_t model_hash_b = 0xBBBBBBBBBBBBBBBBULL;
    const auto shared_prompt_tokens = generate_tokens(512, 100);

    {
        DiskStateCacheConfig config_a;
        config_a.cache_dir = config_a_dir;
        config_a.enabled   = true;
        DiskStateCache cache_a(config_a);

        const auto gdn_a = generate_deterministic_bytes(4096, 111);
        const auto kv_a  = generate_deterministic_bytes(8192, 222);
        const auto mtp_a = generate_deterministic_bytes(2048, 333);

        cache_a.enqueue_save(model_hash_a, shared_prompt_tokens, 1, 0,
                             gdn_a, kv_a, 4, mtp_a, 2, {});
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        auto match_a = cache_a.find_longest_matching_prefix(model_hash_a, shared_prompt_tokens);
        expect(match_a.has_value(), "Config A must hit its own snapshot");
    }

    {
        DiskStateCacheConfig config_b;
        config_b.cache_dir = config_b_dir;
        config_b.enabled   = true;
        DiskStateCache cache_b(config_b);

        auto match_b_pre = cache_b.find_longest_matching_prefix(model_hash_b, shared_prompt_tokens);
        expect(!match_b_pre.has_value(), "Config B must NOT hit Config A's cache (isolated subdirectories)");

        const auto gdn_b = generate_deterministic_bytes(4096, 111);
        const auto kv_b  = generate_deterministic_bytes(16384, 444);
        cache_b.enqueue_save(model_hash_b, shared_prompt_tokens, 1, 0,
                             gdn_b, kv_b, 4, {}, 0, {});
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        auto match_b = cache_b.find_longest_matching_prefix(model_hash_b, shared_prompt_tokens);
        expect(match_b.has_value(), "Config B must hit its own INT8 snapshot");
    }

    std::filesystem::remove_all(root_dir, ec);
}

void test_cancel_in_flight() {
    const std::filesystem::path test_dir = "./.test_ninfer_cache_cancellation";
    std::error_code ec;
    std::filesystem::remove_all(test_dir, ec);

    DiskStateCacheConfig config;
    config.cache_dir = test_dir;
    config.enabled = true;

    const std::uint64_t model_hash = 0x1234567890abcdefULL;
    std::vector<std::byte> large_dummy(8 * 1024 * 1024, std::byte{0x42});

    {
        DiskStateCache cache(config);
        for (int i = 0; i < 5; ++i) {
            std::vector<TokenId> t = {1, 2, static_cast<TokenId>(10 + i)};
            cache.enqueue_save(model_hash, t, i, 0, large_dummy, {}, 0, {}, 0, {});
        }
        cache.cancel_in_flight();
    }

    std::filesystem::remove_all(test_dir, ec);
}

void test_paged_kv_gather_scatter_page_major_layout_verification() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::cout << "SKIP: No CUDA device available for gather/scatter page-major layout test\n";
        return;
    }

    DeviceContext ctx(0);

    // Build a 4-plane heterogeneous quantized PagedKVPool
    LayoutBuilder kv_builder;
    PagedKVPoolSpec kv_spec{
        .page_group_count      = 16,
        .logical_page_capacity = 16,
        .table_rows            = 2,
        .plane_order           = PagedKVPlaneOrder::HeadMajor,
        .planes                = {
            {DType::U8, 64, 8, 256},   // Plane 0: K (64 dim, 8 heads, 64 tokens = 32768 bytes/page)
            {DType::U8, 128, 8, 256},  // Plane 1: V (128 dim, 8 heads, 64 tokens = 65536 bytes/page)
            {DType::FP16, 4, 8, 256},  // Plane 2: K-scale (4 dim, 8 heads, 64 tokens = 4096 bytes/page)
            {DType::FP16, 4, 8, 256},  // Plane 3: V-scale (4 dim, 8 heads, 64 tokens = 4096 bytes/page)
        },
    };
    auto kv_layout = plan_paged_kv_pool(kv_builder, kv_spec);
    const std::size_t kv_arena_bytes = kv_builder.finish(256);
    DeviceArena kv_arena(kv_arena_bytes);
    CUDA_CHECK(cudaDeviceSynchronize());
    PagedKVPool kv_pool({kv_arena.base(), kv_arena.capacity()}, kv_layout);

    const std::size_t single_page_bytes = kv_pool.total_page_bytes();
    expect(single_page_bytes == (32768 + 65536 + 4096 + 4096), "Single page bytes must match sum of planes");

    const std::vector<std::int32_t> physical_pages = {3, 7, 11, 15};
    const std::size_t num_pages = physical_pages.size();
    const std::size_t total_staging_bytes = single_page_bytes * num_pages;

    // Populate each physical page with distinct deterministic per-page bytes
    std::vector<std::vector<std::byte>> ground_truth_page_planes(num_pages);
    for (std::size_t i = 0; i < num_pages; ++i) {
        const std::int32_t page_id = physical_pages[i];
        for (std::size_t p = 0; p < kv_pool.plane_count(); ++p) {
            const std::size_t p_bytes = kv_pool.page_bytes(p);
            auto page_plane_data = generate_deterministic_bytes(p_bytes, static_cast<std::uint32_t>(1000 * i + 10 * p + 7));
            kv_pool.copy_page_from_host(p, page_id, page_plane_data.data(), ctx.stream);
            ground_truth_page_planes[i].insert(ground_truth_page_planes[i].end(),
                                               page_plane_data.begin(), page_plane_data.end());
        }
        expect(ground_truth_page_planes[i].size() == single_page_bytes,
               "Ground truth page size matches total_page_bytes");
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

    // 1. Gather all pages to GPU contiguous staging
    void* d_staging = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_staging, total_staging_bytes, ctx.stream));
    kv_pool.gather_to_contiguous_device(physical_pages, d_staging, ctx.stream);

    // 2. Read back staging buffer and strictly verify Page-Major layout:
    // Slice [i * single_page_bytes .. (i+1) * single_page_bytes) MUST EXACTLY MATCH Page i across all planes!
    std::vector<std::byte> h_staging(total_staging_bytes);
    CUDA_CHECK(cudaMemcpyAsync(h_staging.data(), d_staging, total_staging_bytes, cudaMemcpyDeviceToHost, ctx.stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

    for (std::size_t i = 0; i < num_pages; ++i) {
        const std::byte* slice_ptr = h_staging.data() + i * single_page_bytes;
        expect(std::memcmp(slice_ptr, ground_truth_page_planes[i].data(), single_page_bytes) == 0,
               "GPU staging slice " + std::to_string(i) + " must be 100% bit-exact Page-Major (all planes for page " + std::to_string(i) + ")");
    }

    // 3. Scatter back from staging into a different set of physical pages {0, 1, 2, 4}
    const std::vector<std::int32_t> target_pages = {0, 1, 2, 4};
    kv_pool.scatter_from_contiguous_device(target_pages, d_staging, ctx.stream);
    CUDA_CHECK(cudaFreeAsync(d_staging, ctx.stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

    // 4. Read back target physical pages and verify 100% bit-exact match across all heads and planes
    for (std::size_t i = 0; i < num_pages; ++i) {
        const std::int32_t target_page_id = target_pages[i];
        std::vector<std::byte> readback_page;
        for (std::size_t p = 0; p < kv_pool.plane_count(); ++p) {
            const std::size_t p_bytes = kv_pool.page_bytes(p);
            std::vector<std::byte> readback_plane(p_bytes);
            kv_pool.copy_page_to_host(p, target_page_id, readback_plane.data(), ctx.stream);
            CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
            readback_page.insert(readback_page.end(), readback_plane.begin(), readback_plane.end());
        }
        expect(readback_page == ground_truth_page_planes[i],
               "Scattered target page " + std::to_string(target_page_id) + " must be bit-exact to original page " + std::to_string(i));
    }
}

#if defined(_WIN32)
void test_cow_multi_turn_delta_splicing_live_gpu() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::cout << "SKIP: No CUDA device available for CoW multi-turn delta splicing test\n";
        return;
    }

    DeviceContext ctx(0);
    const std::filesystem::path test_dir = "./.test_ninfer_cache_cow_splicing";
    std::error_code ec;
    std::filesystem::remove_all(test_dir, ec);

    DiskStateCacheConfig config;
    config.cache_dir = test_dir;
    config.enabled   = true;

    // Build 4-plane quantized GPU PagedKVPool
    LayoutBuilder kv_builder;
    PagedKVPoolSpec kv_spec{
        .page_group_count      = 32,
        .logical_page_capacity = 32,
        .table_rows            = 2,
        .plane_order           = PagedKVPlaneOrder::HeadMajor,
        .planes                = {
            {DType::U8, 64, 8, 256},   // Plane 0: 32768 bytes/page
            {DType::U8, 128, 8, 256},  // Plane 1: 65536 bytes/page
            {DType::FP16, 4, 8, 256},  // Plane 2: 4096 bytes/page
            {DType::FP16, 4, 8, 256},  // Plane 3: 4096 bytes/page
        },
    };
    auto kv_layout = plan_paged_kv_pool(kv_builder, kv_spec);
    const std::size_t kv_arena_bytes = kv_builder.finish(256);
    DeviceArena kv_arena(kv_arena_bytes);
    CUDA_CHECK(cudaDeviceSynchronize());
    PagedKVPool kv_pool({kv_arena.base(), kv_arena.capacity()}, kv_layout);

    const std::size_t single_page_bytes = kv_pool.total_page_bytes();
    const uint64_t model_hash = 0x1122334455667788ULL;

    // Generate 3 multi-turn token progressions:
    // Turn 1: 512 tokens (8 pages: 0..7)
    // Turn 2: 1024 tokens (16 pages: 0..15) -> extends Turn 1 by 8 delta pages (8..15)
    // Turn 3: 1536 tokens (24 pages: 0..23) -> extends Turn 2 by 8 delta pages (16..23)
    const auto full_tokens = generate_tokens(1536, 12345);
    std::vector<TokenId> t1_tokens(full_tokens.begin(), full_tokens.begin() + 512);
    std::vector<TokenId> t2_tokens(full_tokens.begin(), full_tokens.begin() + 1024);
    std::vector<TokenId> t3_tokens(full_tokens.begin(), full_tokens.begin() + 1536);

    // Ground truth data for all 24 pages
    std::vector<std::vector<std::byte>> all_ground_truth_pages(24);
    for (std::size_t i = 0; i < 24; ++i) {
        const std::int32_t page_id = static_cast<std::int32_t>(i);
        for (std::size_t p = 0; p < kv_pool.plane_count(); ++p) {
            const std::size_t p_bytes = kv_pool.page_bytes(p);
            auto page_plane_data = generate_deterministic_bytes(p_bytes, static_cast<std::uint32_t>(5000 + i * 50 + p));
            kv_pool.copy_page_from_host(p, page_id, page_plane_data.data(), ctx.stream);
            all_ground_truth_pages[i].insert(all_ground_truth_pages[i].end(),
                                             page_plane_data.begin(), page_plane_data.end());
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

    // Session 1: Perform saves using true incremental CoW delta appending
    {
        DiskStateCache cache(config);

        // Turn 1: Gather pages 0..7 on GPU and save
        std::vector<std::int32_t> t1_phys_pages;
        for (int i = 0; i < 8; ++i) t1_phys_pages.push_back(i);
        void* d_t1_staging = nullptr;
        CUDA_CHECK(cudaMallocAsync(&d_t1_staging, single_page_bytes * 8, ctx.stream));
        kv_pool.gather_to_contiguous_device(t1_phys_pages, d_t1_staging, ctx.stream);
        std::vector<std::byte> h_t1_data(single_page_bytes * 8);
        CUDA_CHECK(cudaMemcpyAsync(h_t1_data.data(), d_t1_staging, single_page_bytes * 8, cudaMemcpyDeviceToHost, ctx.stream));
        CUDA_CHECK(cudaFreeAsync(d_t1_staging, ctx.stream));
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

        // Compute page hashes for Turn 1
        std::vector<std::uint64_t> t1_all_hashes;
        for (std::size_t i = 0; i < 8; ++i) {
            const std::size_t span_end = std::min<std::size_t>(512, (i + 1) * 64);
            t1_all_hashes.push_back(DiskStateCache::hash_prompt_prefix(std::span<const TokenId>(t1_tokens.data(), span_end)));
        }
        cache.enqueue_save_cow(model_hash, t1_tokens, 1, 0, {}, t1_all_hashes, t1_all_hashes,
                               h_t1_data, static_cast<std::uint32_t>(single_page_bytes), {}, 0, {});
        wait_for_match(cache, model_hash, t1_tokens);

        // Turn 2: Gather ONLY delta pages 8..15 on GPU and save
        std::vector<std::int32_t> t2_delta_pages;
        for (int i = 8; i < 16; ++i) t2_delta_pages.push_back(i);
        void* d_t2_staging = nullptr;
        CUDA_CHECK(cudaMallocAsync(&d_t2_staging, single_page_bytes * 8, ctx.stream));
        kv_pool.gather_to_contiguous_device(t2_delta_pages, d_t2_staging, ctx.stream);
        std::vector<std::byte> h_t2_delta_data(single_page_bytes * 8);
        CUDA_CHECK(cudaMemcpyAsync(h_t2_delta_data.data(), d_t2_staging, single_page_bytes * 8, cudaMemcpyDeviceToHost, ctx.stream));
        CUDA_CHECK(cudaFreeAsync(d_t2_staging, ctx.stream));
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

        std::vector<std::uint64_t> t2_all_hashes;
        std::vector<std::uint64_t> t2_delta_hashes;
        for (std::size_t i = 0; i < 16; ++i) {
            const std::size_t span_end = std::min<std::size_t>(1024, (i + 1) * 64);
            const uint64_t phash = DiskStateCache::hash_prompt_prefix(std::span<const TokenId>(t2_tokens.data(), span_end));
            t2_all_hashes.push_back(phash);
            if (i >= 8) t2_delta_hashes.push_back(phash);
        }
        cache.enqueue_save_cow(model_hash, t2_tokens, 2, 0, {}, t2_all_hashes, t2_delta_hashes,
                               h_t2_delta_data, static_cast<std::uint32_t>(single_page_bytes), {}, 0, {});
        wait_for_match(cache, model_hash, t2_tokens);

        // Turn 3: Gather ONLY delta pages 16..23 on GPU and save
        std::vector<std::int32_t> t3_delta_pages;
        for (int i = 16; i < 24; ++i) t3_delta_pages.push_back(i);
        void* d_t3_staging = nullptr;
        CUDA_CHECK(cudaMallocAsync(&d_t3_staging, single_page_bytes * 8, ctx.stream));
        kv_pool.gather_to_contiguous_device(t3_delta_pages, d_t3_staging, ctx.stream);
        std::vector<std::byte> h_t3_delta_data(single_page_bytes * 8);
        CUDA_CHECK(cudaMemcpyAsync(h_t3_delta_data.data(), d_t3_staging, single_page_bytes * 8, cudaMemcpyDeviceToHost, ctx.stream));
        CUDA_CHECK(cudaFreeAsync(d_t3_staging, ctx.stream));
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

        std::vector<std::uint64_t> t3_all_hashes;
        std::vector<std::uint64_t> t3_delta_hashes;
        for (std::size_t i = 0; i < 24; ++i) {
            const std::size_t span_end = std::min<std::size_t>(1536, (i + 1) * 64);
            const uint64_t phash = DiskStateCache::hash_prompt_prefix(std::span<const TokenId>(t3_tokens.data(), span_end));
            t3_all_hashes.push_back(phash);
            if (i >= 16) t3_delta_hashes.push_back(phash);
        }
        cache.enqueue_save_cow(model_hash, t3_tokens, 3, 0, {}, t3_all_hashes, t3_delta_hashes,
                               h_t3_delta_data, static_cast<std::uint32_t>(single_page_bytes), {}, 0, {});
        wait_for_match(cache, model_hash, t3_tokens);
    }

    // Zero out all physical GPU memory to guarantee no resident cache leaks
    for (std::size_t p = 0; p < kv_pool.plane_count(); ++p) {
        CUDA_CHECK(cudaMemsetAsync(kv_pool.plane(p).data, 0x55, kv_pool.plane(p).bytes(), ctx.stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

    // Session 2: Cold restart simulation and full Turn 3 DirectStorage restore
    {
        DiskStateCache restarted_cache(config);
        auto match_t3 = restarted_cache.find_longest_matching_prefix(model_hash, t3_tokens);
        expect(match_t3.has_value() && match_t3->matched_tokens == 1536, "Turn 3 match must succeed on cold restart");

        DiskStateHeader loaded_header;
        std::vector<TokenId> loaded_tokens;
        void* d_ds_staging = nullptr;
        std::size_t text_staging_bytes = 0;

        const bool ok = restarted_cache.load_snapshot_direct_storage(
            match_t3->file_path, loaded_header, loaded_tokens, ctx.stream, d_ds_staging, text_staging_bytes);
        expect(ok, "DirectStorage multi-turn delta restore must succeed");
        expect(loaded_header.text_page_count == 24, "Restored text page count must be 24");
        expect(text_staging_bytes == single_page_bytes * 24, "Staging bytes must match 24 full pages");

        // Scatter the restored 24 pages into physical pages 0..23
        std::vector<std::int32_t> restore_phys_pages;
        for (int i = 0; i < 24; ++i) restore_phys_pages.push_back(i);
        kv_pool.scatter_from_contiguous_device(restore_phys_pages, d_ds_staging, ctx.stream);
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

        // Read back each physical page and assert 100% bit-exact parity across all 24 pages and all 4 planes
        for (std::size_t i = 0; i < 24; ++i) {
            std::vector<std::byte> readback_page;
            for (std::size_t p = 0; p < kv_pool.plane_count(); ++p) {
                const std::size_t p_bytes = kv_pool.page_bytes(p);
                std::vector<std::byte> readback_plane(p_bytes);
                kv_pool.copy_page_to_host(p, static_cast<std::int32_t>(i), readback_plane.data(), ctx.stream);
                CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
                readback_page.insert(readback_page.end(), readback_plane.begin(), readback_plane.end());
            }
            expect(readback_page == all_ground_truth_pages[i],
                   "DirectStorage spliced multi-turn page " + std::to_string(i) + " must be 100% bit-exact across all 4 planes");
        }

        ninfer::core::DirectStorageEngine::instance().release_staging();
    }

    std::filesystem::remove_all(test_dir, ec);
}

void test_full_state_direct_storage_restore_live_gpu() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::cout << "SKIP: No CUDA device available for full state DirectStorage test\n";
        return;
    }

    DeviceContext ctx(0);
    const std::filesystem::path test_dir = "./.test_ninfer_cache_full_directstorage";
    std::error_code ec;
    std::filesystem::remove_all(test_dir, ec);

    DiskStateCacheConfig config;
    config.cache_dir = test_dir;
    config.enabled   = true;

    const uint64_t model_hash = 0xFEEDFACE01234567ULL;
    const auto tokens = generate_tokens(512, 777);

    // 1. Text KV Pool (HeadMajor 4 planes)
    LayoutBuilder kv_builder;
    PagedKVPoolSpec kv_spec{
        .page_group_count      = 8,
        .logical_page_capacity = 8,
        .table_rows            = 2,
        .plane_order           = PagedKVPlaneOrder::HeadMajor,
        .planes                = {
            {DType::U8, 64, 8, 256},
            {DType::U8, 128, 8, 256},
            {DType::FP16, 4, 8, 256},
            {DType::FP16, 4, 8, 256},
        },
    };
    auto kv_layout = plan_paged_kv_pool(kv_builder, kv_spec);
    DeviceArena kv_arena(kv_builder.finish(256));
    PagedKVPool kv_pool({kv_arena.base(), kv_arena.capacity()}, kv_layout);

    // 2. GDN Pool (Recurrent & Conv states)
    LayoutBuilder gdn_builder;
    LinearAttentionStatePoolSpec gdn_spec{
        .layers         = 2,
        .conv_channels  = 128,
        .conv_width     = 4,
        .value_heads    = 8,
        .value_head_dim = 64,
        .key_head_dim   = 64,
        .slot_count     = 2,
        .conv_dtype     = DType::BF16,
    };
    auto gdn_layout = plan_linear_attention_state_pool(gdn_builder, gdn_spec);
    DeviceArena gdn_arena(gdn_builder.finish(256));
    LinearAttentionStatePool gdn_pool({gdn_arena.base(), gdn_arena.capacity()}, gdn_layout);

    // 3. Tail hidden tensor
    DeviceArena tail_arena(2048 * sizeof(uint16_t));
    Tensor tail_hidden(tail_arena.base(), DType::BF16, {2048, 1});

    // Populate live GPU state
    const std::vector<std::int32_t> valid_page_ids = {2, 5};
    const std::int32_t active_slot = 0;

    std::vector<std::byte> original_gdn;
    for (const auto& t : gdn_pool.recurrent) {
        Tensor slot_t = t.slice(3, active_slot, 1);
        auto data = generate_deterministic_bytes(slot_t.bytes(), 101 + static_cast<std::uint32_t>(original_gdn.size()));
        CUDA_CHECK(cudaMemcpyAsync(slot_t.data, data.data(), slot_t.bytes(), cudaMemcpyHostToDevice, ctx.stream));
        original_gdn.insert(original_gdn.end(), data.begin(), data.end());
    }
    for (const auto& t : gdn_pool.conv) {
        Tensor slot_t = t.slice(2, active_slot, 1);
        auto data = generate_deterministic_bytes(slot_t.bytes(), 202 + static_cast<std::uint32_t>(original_gdn.size()));
        CUDA_CHECK(cudaMemcpyAsync(slot_t.data, data.data(), slot_t.bytes(), cudaMemcpyHostToDevice, ctx.stream));
        original_gdn.insert(original_gdn.end(), data.begin(), data.end());
    }

    std::vector<std::vector<std::byte>> original_kv_pages(valid_page_ids.size());
    for (std::size_t i = 0; i < valid_page_ids.size(); ++i) {
        const std::int32_t page_id = valid_page_ids[i];
        for (std::size_t p = 0; p < kv_pool.plane_count(); ++p) {
            const std::size_t p_bytes = kv_pool.page_bytes(p);
            auto page_plane_data = generate_deterministic_bytes(p_bytes, 303 + static_cast<std::uint32_t>(i * 10 + p));
            kv_pool.copy_page_from_host(p, page_id, page_plane_data.data(), ctx.stream);
            original_kv_pages[i].insert(original_kv_pages[i].end(), page_plane_data.begin(), page_plane_data.end());
        }
    }

    auto original_tail = generate_deterministic_bytes(tail_hidden.bytes(), 404);
    CUDA_CHECK(cudaMemcpyAsync(tail_hidden.data, original_tail.data(), tail_hidden.bytes(), cudaMemcpyHostToDevice, ctx.stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

    // Save full state via DiskStateCache
    {
        DiskStateCache cache(config);

        void* d_staging = nullptr;
        const std::size_t kv_staging_bytes = kv_pool.total_page_bytes() * valid_page_ids.size();
        CUDA_CHECK(cudaMallocAsync(&d_staging, kv_staging_bytes, ctx.stream));
        kv_pool.gather_to_contiguous_device(valid_page_ids, d_staging, ctx.stream);
        std::vector<std::byte> h_kv_data(kv_staging_bytes);
        CUDA_CHECK(cudaMemcpyAsync(h_kv_data.data(), d_staging, kv_staging_bytes, cudaMemcpyDeviceToHost, ctx.stream));
        CUDA_CHECK(cudaFreeAsync(d_staging, ctx.stream));
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

        std::vector<std::uint64_t> page_hashes;
        for (std::size_t i = 0; i < valid_page_ids.size(); ++i) {
            const std::size_t span_end = std::min<std::size_t>(tokens.size(), (i + 1) * 64);
            page_hashes.push_back(DiskStateCache::hash_prompt_prefix(std::span<const TokenId>(tokens.data(), span_end)));
        }

        cache.enqueue_save_cow(model_hash, tokens, 1, 42, original_gdn, page_hashes, page_hashes,
                               h_kv_data, static_cast<std::uint32_t>(kv_pool.total_page_bytes()), {}, 0, original_tail);
        auto match = wait_for_match(cache, model_hash, tokens);
        expect(match.has_value(), "Snapshot must be saved and matched");

        // Clear GPU memory
        for (const auto& t : gdn_pool.recurrent) CUDA_CHECK(cudaMemsetAsync(t.data, 0xCC, t.bytes(), ctx.stream));
        for (const auto& t : gdn_pool.conv) CUDA_CHECK(cudaMemsetAsync(t.data, 0xCC, t.bytes(), ctx.stream));
        for (std::size_t p = 0; p < kv_pool.plane_count(); ++p) CUDA_CHECK(cudaMemsetAsync(kv_pool.plane(p).data, 0xCC, kv_pool.plane(p).bytes(), ctx.stream));
        CUDA_CHECK(cudaMemsetAsync(tail_hidden.data, 0xCC, tail_hidden.bytes(), ctx.stream));
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

        // Restore entire state via DirectStorage DMA
        DiskStateHeader loaded_header;
        std::vector<TokenId> loaded_tokens;
        void* d_ds_staging = nullptr;
        std::size_t text_staging_bytes = 0;
        const bool load_ok = cache.load_snapshot_direct_storage(match->file_path, loaded_header,
                                                               loaded_tokens, ctx.stream, d_ds_staging,
                                                               text_staging_bytes);
        expect(load_ok, "load_snapshot_direct_storage must succeed");
        expect(loaded_header.rope_delta == 42, "RoPE delta matches");

        const std::byte* d_staging_bytes = static_cast<const std::byte*>(d_ds_staging);

        // 1. Scatter Text KV
        kv_pool.scatter_from_contiguous_device(valid_page_ids, d_staging_bytes, ctx.stream);

        // 2. Restore GDN state
        std::size_t offset = text_staging_bytes;
        for (const auto& t : gdn_pool.recurrent) {
            Tensor slot_t = t.slice(3, active_slot, 1);
            CUDA_CHECK(cudaMemcpyAsync(slot_t.data, d_staging_bytes + offset, slot_t.bytes(), cudaMemcpyDeviceToDevice, ctx.stream));
            offset += slot_t.bytes();
        }
        for (const auto& t : gdn_pool.conv) {
            Tensor slot_t = t.slice(2, active_slot, 1);
            CUDA_CHECK(cudaMemcpyAsync(slot_t.data, d_staging_bytes + offset, slot_t.bytes(), cudaMemcpyDeviceToDevice, ctx.stream));
            offset += slot_t.bytes();
        }

        // 3. Restore Tail Hidden
        CUDA_CHECK(cudaMemcpyAsync(tail_hidden.data, d_staging_bytes + offset, tail_hidden.bytes(), cudaMemcpyDeviceToDevice, ctx.stream));
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

        // Verify GDN parity
        std::vector<std::byte> readback_gdn;
        for (const auto& t : gdn_pool.recurrent) {
            Tensor slot_t = t.slice(3, active_slot, 1);
            const std::size_t old_sz = readback_gdn.size();
            readback_gdn.resize(old_sz + slot_t.bytes());
            CUDA_CHECK(cudaMemcpyAsync(readback_gdn.data() + old_sz, slot_t.data, slot_t.bytes(), cudaMemcpyDeviceToHost, ctx.stream));
        }
        for (const auto& t : gdn_pool.conv) {
            Tensor slot_t = t.slice(2, active_slot, 1);
            const std::size_t old_sz = readback_gdn.size();
            readback_gdn.resize(old_sz + slot_t.bytes());
            CUDA_CHECK(cudaMemcpyAsync(readback_gdn.data() + old_sz, slot_t.data, slot_t.bytes(), cudaMemcpyDeviceToHost, ctx.stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
        expect(readback_gdn == original_gdn, "DirectStorage restored GDN state must match original bytes");

        // Verify KV parity across all planes
        for (std::size_t i = 0; i < valid_page_ids.size(); ++i) {
            std::vector<std::byte> readback_page;
            for (std::size_t p = 0; p < kv_pool.plane_count(); ++p) {
                const std::size_t p_bytes = kv_pool.page_bytes(p);
                std::vector<std::byte> readback_plane(p_bytes);
                kv_pool.copy_page_to_host(p, valid_page_ids[i], readback_plane.data(), ctx.stream);
                CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
                readback_page.insert(readback_page.end(), readback_plane.begin(), readback_plane.end());
            }
            expect(readback_page == original_kv_pages[i],
                   "DirectStorage restored Text KV page " + std::to_string(valid_page_ids[i]) + " must be bit-exact");
        }

        // Verify Tail hidden parity
        std::vector<std::byte> readback_tail(tail_hidden.bytes());
        CUDA_CHECK(cudaMemcpyAsync(readback_tail.data(), tail_hidden.data, tail_hidden.bytes(), cudaMemcpyDeviceToHost, ctx.stream));
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
        expect(readback_tail == original_tail, "DirectStorage restored Tail hidden state must match original bytes");

        ninfer::core::DirectStorageEngine::instance().release_staging();
    }

    std::filesystem::remove_all(test_dir, ec);
}
#endif

int main() {
    std::cout << "Starting Comprehensive DiskStateCache & DirectStorage test suite...\n";
    test_prompt_hashing();
    test_disabled_by_default();
    test_timestamped_filename_and_datetime();
    test_corruption_and_version_invalidation();
    test_multi_config_isolation_and_cross_contamination();
    test_cancel_in_flight();
    test_paged_kv_gather_scatter_page_major_layout_verification();
#if defined(_WIN32)
    test_cow_multi_turn_delta_splicing_live_gpu();
    test_full_state_direct_storage_restore_live_gpu();
#endif

    if (failures != 0) {
        std::cerr << failures << " DiskStateCache checks FAILED\n";
        return 1;
    }
    std::cout << "All DiskStateCache tests PASSED.\n";
    return 0;
}
