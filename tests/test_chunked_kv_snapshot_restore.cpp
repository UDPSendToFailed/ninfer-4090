#include "core/device.h"
#include "core/paged_kv_cache.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <span>
#include <string>
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

// ---------------------------------------------------------------------------
// Test 1: Multi-Batch Chunked Gather & Save (>32 pages across batch boundaries)
// ---------------------------------------------------------------------------
void test_multi_batch_chunked_gather_save(DeviceContext& ctx) {
    std::cout << "Running test_multi_batch_chunked_gather_save...\n";

    constexpr std::uint32_t kTotalPages = 80; // 80 pages = 32 + 32 + 16 (3 batches)
    LayoutBuilder builder;
    PagedKVPoolSpec spec{
        .page_group_count      = kTotalPages,
        .logical_page_capacity = kTotalPages,
        .table_rows            = 1,
        .plane_order           = PagedKVPlaneOrder::HeadMajor,
        .planes                = {
            {DType::U8, 64, 8, 256},   // Plane 0: 32768 bytes/page
            {DType::U8, 128, 8, 256},  // Plane 1: 65536 bytes/page
            {DType::FP16, 4, 8, 256},  // Plane 2: 4096 bytes/page
            {DType::FP16, 4, 8, 256},  // Plane 3: 4096 bytes/page
        },
    };
    auto layout = plan_paged_kv_pool(builder, spec);
    const std::size_t arena_bytes = builder.finish(256);
    DeviceArena arena(arena_bytes);
    CUDA_CHECK(cudaDeviceSynchronize());
    PagedKVPool pool({arena.base(), arena.capacity()}, layout);

    const std::size_t single_page_bytes = pool.total_page_bytes();
    std::vector<std::int32_t> physical_page_ids(kTotalPages);
    std::iota(physical_page_ids.begin(), physical_page_ids.end(), 0);

    // Populate ground-truth data in all 80 pages
    std::vector<std::vector<std::byte>> ground_truth_pages(kTotalPages);
    for (std::uint32_t i = 0; i < kTotalPages; ++i) {
        for (std::size_t p = 0; p < pool.plane_count(); ++p) {
            const std::size_t p_bytes = pool.page_bytes(p);
            auto pdata = generate_deterministic_bytes(p_bytes, 10000 + i * 100 + static_cast<std::uint32_t>(p));
            pool.copy_page_from_host(p, static_cast<std::int32_t>(i), pdata.data(), ctx.stream);
            ground_truth_pages[i].insert(ground_truth_pages[i].end(), pdata.begin(), pdata.end());
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

    // Execute chunked gather in 32-page batches
    constexpr std::uint32_t kSaveBatchPages = 32;
    const std::size_t total_bytes = single_page_bytes * kTotalPages;
    std::vector<std::byte> gathered_data(total_bytes);

    for (std::size_t b = 0; b < physical_page_ids.size(); b += kSaveBatchPages) {
        const std::size_t n = std::min<std::size_t>(kSaveBatchPages, physical_page_ids.size() - b);
        const std::size_t batch_bytes = n * single_page_bytes;

        void* d_batch = nullptr;
        CUDA_CHECK(cudaMallocAsync(&d_batch, batch_bytes, ctx.stream));
        pool.gather_to_contiguous_device(
            std::span<const std::int32_t>(physical_page_ids.data() + b, n),
            d_batch, ctx.stream);
        CUDA_CHECK(cudaMemcpyAsync(gathered_data.data() + b * single_page_bytes, d_batch,
                                   batch_bytes, cudaMemcpyDeviceToHost, ctx.stream));
        CUDA_CHECK(cudaFreeAsync(d_batch, ctx.stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

    // Verify all 80 pages bit-for-bit in page-major order
    for (std::size_t i = 0; i < kTotalPages; ++i) {
        const std::byte* page_ptr = gathered_data.data() + i * single_page_bytes;
        expect(std::memcmp(page_ptr, ground_truth_pages[i].data(), single_page_bytes) == 0,
               "Gathered chunked page " + std::to_string(i) + " must match ground truth exactly");
    }
}

// ---------------------------------------------------------------------------
// Test 2: Multi-Batch Chunked Scatter & Restore (>32 pages across batch boundaries)
// ---------------------------------------------------------------------------
void test_multi_batch_chunked_scatter_restore(DeviceContext& ctx) {
    std::cout << "Running test_multi_batch_chunked_scatter_restore...\n";

    constexpr std::uint32_t kTotalPages = 80;
    LayoutBuilder builder;
    PagedKVPoolSpec spec{
        .page_group_count      = kTotalPages,
        .logical_page_capacity = kTotalPages,
        .table_rows            = 1,
        .plane_order           = PagedKVPlaneOrder::HeadMajor,
        .planes                = {
            {DType::U8, 64, 8, 256},
            {DType::U8, 128, 8, 256},
            {DType::FP16, 4, 8, 256},
            {DType::FP16, 4, 8, 256},
        },
    };
    auto layout = plan_paged_kv_pool(builder, spec);
    const std::size_t arena_bytes = builder.finish(256);
    DeviceArena arena(arena_bytes);
    CUDA_CHECK(cudaDeviceSynchronize());
    PagedKVPool pool({arena.base(), arena.capacity()}, layout);

    const std::size_t single_page_bytes = pool.total_page_bytes();

    // Create contiguous host snapshot data for 80 pages
    std::vector<std::byte> host_snapshot_data;
    host_snapshot_data.reserve(kTotalPages * single_page_bytes);
    std::vector<std::vector<std::byte>> ground_truth_pages(kTotalPages);

    for (std::uint32_t i = 0; i < kTotalPages; ++i) {
        auto page_data = generate_deterministic_bytes(single_page_bytes, 20000 + i);
        ground_truth_pages[i] = page_data;
        host_snapshot_data.insert(host_snapshot_data.end(), page_data.begin(), page_data.end());
    }

    // Target physical page mapping: reversed {79, 78, ..., 0}
    std::vector<std::int32_t> target_physical_pages(kTotalPages);
    for (std::uint32_t i = 0; i < kTotalPages; ++i) {
        target_physical_pages[i] = static_cast<std::int32_t>(kTotalPages - 1 - i);
    }

    // Execute chunked restore (scatter) in 32-page batches
    constexpr std::uint32_t kRestoreBatchPages = 32;
    for (std::size_t b = 0; b < target_physical_pages.size(); b += kRestoreBatchPages) {
        const std::size_t n = std::min<std::size_t>(kRestoreBatchPages, target_physical_pages.size() - b);
        const std::size_t batch_bytes = n * single_page_bytes;

        void* d_text_staging = nullptr;
        CUDA_CHECK(cudaMallocAsync(&d_text_staging, batch_bytes, ctx.stream));
        CUDA_CHECK(cudaMemcpyAsync(d_text_staging,
                                   host_snapshot_data.data() + b * single_page_bytes,
                                   batch_bytes, cudaMemcpyHostToDevice, ctx.stream));
        pool.scatter_from_contiguous_device(
            std::span<const std::int32_t>(target_physical_pages.data() + b, n),
            d_text_staging, ctx.stream);
        CUDA_CHECK(cudaFreeAsync(d_text_staging, ctx.stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

    // Read back and verify each scattered target page matches the corresponding ground-truth page
    for (std::size_t i = 0; i < kTotalPages; ++i) {
        const std::int32_t phys_id = target_physical_pages[i];
        std::vector<std::byte> readback_page;
        for (std::size_t p = 0; p < pool.plane_count(); ++p) {
            const std::size_t p_bytes = pool.page_bytes(p);
            std::vector<std::byte> readback_plane(p_bytes);
            pool.copy_page_to_host(p, phys_id, readback_plane.data(), ctx.stream);
            CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
            readback_page.insert(readback_page.end(), readback_plane.begin(), readback_plane.end());
        }
        expect(readback_page == ground_truth_pages[i],
               "Scattered physical page " + std::to_string(phys_id) + " must match ground truth page " + std::to_string(i));
    }
}

// ---------------------------------------------------------------------------
// Test 3: MTP KV Partial-Sequence Restore Page Stride Regression Test
// ---------------------------------------------------------------------------
void test_mtp_kv_partial_restore_stride_bug(DeviceContext& ctx) {
    std::cout << "Running test_mtp_kv_partial_restore_stride_bug...\n";

    // Scenario: Snapshot saved 64 MTP pages, but target sequence only restores 40 pages
    constexpr std::uint32_t kSavedMtpPages = 64;
    constexpr std::uint32_t kRestoredMtpPages = 40;

    LayoutBuilder builder;
    PagedKVPoolSpec spec{
        .page_group_count      = kSavedMtpPages,
        .logical_page_capacity = kSavedMtpPages,
        .table_rows            = 1,
        .plane_order           = PagedKVPlaneOrder::HeadMajor,
        .planes                = {
            {DType::FP16, 128, 4, 256}, // MTP KV plane: 128 * 4 * 2 * 64 = 65536 bytes/page
        },
    };
    auto layout = plan_paged_kv_pool(builder, spec);
    const std::size_t arena_bytes = builder.finish(256);
    DeviceArena arena(arena_bytes);
    CUDA_CHECK(cudaDeviceSynchronize());
    PagedKVPool mtp_pool({arena.base(), arena.capacity()}, layout);

    const std::size_t real_page_bytes = mtp_pool.total_page_bytes();
    expect(real_page_bytes > 0, "MTP pool page bytes must be non-zero");

    // Create loaded_mtp_kv payload representing the full 64-page snapshot
    std::vector<std::byte> loaded_mtp_kv(kSavedMtpPages * real_page_bytes);
    std::vector<std::vector<std::byte>> ground_truth_mtp_pages(kSavedMtpPages);
    for (std::uint32_t i = 0; i < kSavedMtpPages; ++i) {
        auto pdata = generate_deterministic_bytes(real_page_bytes, 30000 + i);
        ground_truth_mtp_pages[i] = pdata;
        std::memcpy(loaded_mtp_kv.data() + i * real_page_bytes, pdata.data(), real_page_bytes);
    }

    // Target valid page IDs (only 40 pages)
    std::vector<std::int32_t> valid_mtp_page_ids(kRestoredMtpPages);
    std::iota(valid_mtp_page_ids.begin(), valid_mtp_page_ids.end(), 0);

    // Execute MTP restore using the fixed runtime formula (program_impl.h line 676):
    constexpr std::uint32_t kRestoreBatchPages = 32;
    const std::size_t mtp_page_bytes = mtp_pool.total_page_bytes();

    for (std::size_t b = 0; b < valid_mtp_page_ids.size(); b += kRestoreBatchPages) {
        const std::size_t n = std::min<std::size_t>(kRestoreBatchPages, valid_mtp_page_ids.size() - b);
        const std::size_t batch_bytes = n * mtp_page_bytes;

        void* d_mtp_staging = nullptr;
        CUDA_CHECK(cudaMallocAsync(&d_mtp_staging, batch_bytes, ctx.stream));
        CUDA_CHECK(cudaMemcpyAsync(d_mtp_staging,
                                   loaded_mtp_kv.data() + b * mtp_page_bytes,
                                   batch_bytes,
                                   cudaMemcpyHostToDevice, ctx.stream));
        mtp_pool.scatter_from_contiguous_device(
            std::span<const std::int32_t>(valid_mtp_page_ids.data() + b, n),
            d_mtp_staging, ctx.stream);
        CUDA_CHECK(cudaFreeAsync(d_mtp_staging, ctx.stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

    // Verify all 40 restored pages against ground truth
    for (std::size_t i = 0; i < kRestoredMtpPages; ++i) {
        std::vector<std::byte> readback_page(real_page_bytes);
        mtp_pool.copy_page_to_host(0, valid_mtp_page_ids[i], readback_page.data(), ctx.stream);
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
        expect(readback_page == ground_truth_mtp_pages[i],
               "Restored MTP page " + std::to_string(i) + " must be bit-exact with ground truth");
    }
}

// ---------------------------------------------------------------------------
// Test 4: Streamed Chunked Save without Per-Batch Stalls (CUDA Best Practices)
// ---------------------------------------------------------------------------
void test_streamed_chunked_save_no_sync_in_loop(DeviceContext& ctx) {
    std::cout << "Running test_streamed_chunked_save_no_sync_in_loop...\n";

    constexpr std::uint32_t kTotalPages = 64;
    LayoutBuilder builder;
    PagedKVPoolSpec spec{
        .page_group_count      = kTotalPages,
        .logical_page_capacity = kTotalPages,
        .table_rows            = 1,
        .plane_order           = PagedKVPlaneOrder::HeadMajor,
        .planes                = {
            {DType::FP16, 64, 8, 256},
        },
    };
    auto layout = plan_paged_kv_pool(builder, spec);
    const std::size_t arena_bytes = builder.finish(256);
    DeviceArena arena(arena_bytes);
    CUDA_CHECK(cudaDeviceSynchronize());
    PagedKVPool pool({arena.base(), arena.capacity()}, layout);

    const std::size_t single_page_bytes = pool.total_page_bytes();
    std::vector<std::int32_t> physical_page_ids(kTotalPages);
    std::iota(physical_page_ids.begin(), physical_page_ids.end(), 0);

    std::vector<std::vector<std::byte>> ground_truth(kTotalPages);
    for (std::uint32_t i = 0; i < kTotalPages; ++i) {
        auto pdata = generate_deterministic_bytes(single_page_bytes, 40000 + i);
        ground_truth[i] = pdata;
        pool.copy_page_from_host(0, static_cast<std::int32_t>(i), pdata.data(), ctx.stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

    // Stream-ordered chunked save into pre-allocated buffer WITHOUT calling cudaStreamSynchronize
    // or cudaMallocHost inside the loop (as recommended by Section 10.1.2 / 10.3)
    constexpr std::uint32_t kSaveBatchPages = 32;
    std::vector<std::byte> missing_pages_data(kTotalPages * single_page_bytes);

    for (std::size_t b = 0; b < physical_page_ids.size(); b += kSaveBatchPages) {
        const std::size_t n = std::min<std::size_t>(kSaveBatchPages, physical_page_ids.size() - b);
        const std::size_t batch_bytes = n * single_page_bytes;

        void* d_batch = nullptr;
        CUDA_CHECK(cudaMallocAsync(&d_batch, batch_bytes, ctx.stream));
        pool.gather_to_contiguous_device(
            std::span<const std::int32_t>(physical_page_ids.data() + b, n),
            d_batch, ctx.stream);
        CUDA_CHECK(cudaMemcpyAsync(missing_pages_data.data() + b * single_page_bytes, d_batch,
                                   batch_bytes, cudaMemcpyDeviceToHost, ctx.stream));
        CUDA_CHECK(cudaFreeAsync(d_batch, ctx.stream));
    }
    // Single stream sync at the end
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

    for (std::size_t i = 0; i < kTotalPages; ++i) {
        const std::byte* ptr = missing_pages_data.data() + i * single_page_bytes;
        expect(std::memcmp(ptr, ground_truth[i].data(), single_page_bytes) == 0,
               "Streamed saved page " + std::to_string(i) + " must match ground truth exactly");
    }
}

} // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::cout << "SKIP: No CUDA device available for chunked KV snapshot restore test\n";
        return 77;
    }

    try {
        DeviceContext ctx(0);
        std::cout << "Starting Chunked KV Snapshot Restore & MTP Parity test suite...\n";

        test_multi_batch_chunked_gather_save(ctx);
        test_multi_batch_chunked_scatter_restore(ctx);
        test_mtp_kv_partial_restore_stride_bug(ctx);
        test_streamed_chunked_save_no_sync_in_loop(ctx);

        if (failures != 0) {
            std::cerr << failures << " chunked KV snapshot restore test(s) FAILED\n";
            return 1;
        }

        std::cout << "All Chunked KV Snapshot Restore & MTP Parity tests PASSED.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Unhandled exception: " << e.what() << '\n';
        return 1;
    }
}
