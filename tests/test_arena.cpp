#include "core/arena.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <new>
#include <stdexcept>
#include <utility>

namespace {

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

template <typename Exception, typename Fn>
int expect_throws(Fn&& fn, const char* label) {
    try {
        fn();
    } catch (const Exception&) { return 0; }
    std::cerr << label << " did not throw expected exception\n";
    return 1;
}

int expect_size(std::size_t actual, std::size_t expected, const char* label) {
    if (actual == expected) { return 0; }
    std::cerr << label << " expected " << expected << ", got " << actual << '\n';
    return 1;
}

int expect_ptr(void* actual, void* expected, const char* label) {
    if (actual == expected) { return 0; }
    std::cerr << label << " expected " << expected << ", got " << actual << '\n';
    return 1;
}

int expect(bool condition, const char* label) {
    if (condition) { return 0; }
    std::cerr << label << " condition failed\n";
    return 1;
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }
    if (count == 0) {
        std::cout << "SKIP: no CUDA devices\n";
        return 77;
    }

    int failures = 0;
    CUDA_CHECK(cudaSetDevice(0));

    ninfer::DeviceBuffer empty;
    failures += expect_ptr(empty.p, nullptr, "empty device buffer pointer");
    failures += expect_size(empty.bytes, 0, "empty device buffer size");

    const std::array<std::uint8_t, 8> host_source{0, 1, 2, 3, 4, 5, 6, 7};
    std::array<std::uint8_t, 8> host_destination{};
    ninfer::DeviceBuffer buffer(host_source.size());
    buffer.copy_from_host(host_source.data(), host_source.size());
    buffer.copy_to_host(host_destination.data(), host_destination.size());
    if (host_destination != host_source) {
        ++failures;
        std::cerr << "device buffer round trip changed payload\n";
    }
    failures += expect_throws<std::out_of_range>(
        [&] { buffer.copy_from_host(host_source.data(), 2, buffer.bytes - 1); },
        "device buffer upload range");
    ninfer::DeviceBuffer moved_buffer(std::move(buffer));
    failures += expect_ptr(buffer.p, nullptr, "moved-from device buffer pointer");
    failures += expect_size(buffer.bytes, 0, "moved-from device buffer size");
    failures += expect_size(moved_buffer.bytes, host_source.size(), "moved device buffer size");

    ninfer::DeviceArena arena(1024);
    failures += expect_size(arena.capacity(), 1024, "arena.capacity");
    failures += expect_size(arena.used(), 0, "arena.used initial");
    failures += expect_size(arena.peak_used(), 0, "arena.peak initial");
    if (arena.base() == nullptr) {
        ++failures;
        std::cerr << "arena base is null\n";
    }

    auto* base       = static_cast<unsigned char*>(arena.base());
    ninfer::Tensor a = arena.alloc(ninfer::DType::BF16, {3, 5});
    failures += expect_ptr(a.data, base, "first allocation pointer");
    failures += expect_size(a.bytes(), 30, "first allocation bytes");
    failures += expect_size(arena.used(), 30, "arena.used after first allocation");
    failures += expect_size(arena.peak_used(), 30, "arena.peak after first allocation");

    ninfer::Tensor b = arena.alloc(ninfer::DType::U8, {17}, 64);
    failures += expect_ptr(b.data, base + 64, "second allocation pointer");
    if (reinterpret_cast<std::uintptr_t>(b.data) % 64 != 0) {
        ++failures;
        std::cerr << "second allocation is not 64-byte aligned\n";
    }
    failures += expect_size(arena.used(), 81, "arena.used after second allocation");
    failures += expect_size(arena.peak_used(), 81, "arena.peak after second allocation");

    const std::size_t used_before_scope = arena.used();
    void* transient_ptr                 = nullptr;
    std::size_t peak_after_transient    = 0;
    {
        auto outer_scope         = arena.scope();
        ninfer::Tensor transient = arena.alloc(ninfer::DType::U8, {11}, 128);
        transient_ptr            = transient.data;
        const std::size_t used   = arena.used();
        if (used <= used_before_scope) {
            ++failures;
            std::cerr << "transient allocation did not advance arena cursor\n";
        }
        {
            auto inner_scope = arena.scope();
            (void)arena.alloc(ninfer::DType::U8, {7}, 64);
        }
        failures += expect_size(arena.used(), used, "arena.used after nested scope");
        peak_after_transient = arena.peak_used();
    }
    failures += expect_size(arena.peak_used(), peak_after_transient, "arena.peak after scope exit");
    failures += expect_size(arena.used(), used_before_scope, "arena.used after scope exit");
    ninfer::Tensor reused = arena.alloc(ninfer::DType::U8, {5}, 128);
    failures += expect_ptr(reused.data, transient_ptr, "allocation after scope pointer");

    const std::size_t used_before_exception_scope = arena.used();
    failures += expect_throws<std::runtime_error>(
        [&] {
            auto exception_scope = arena.scope();
            (void)arena.alloc(ninfer::DType::U8, {9}, 64);
            throw std::runtime_error("scope unwind");
        },
        "arena scope exception");
    failures +=
        expect_size(arena.used(), used_before_exception_scope, "arena.used after exception scope");

    const std::size_t used_before_failures = arena.used();
    const std::size_t peak_before_failures = arena.peak_used();
    failures += expect_throws<std::bad_alloc>(
        [&] { (void)arena.alloc(ninfer::DType::FP32, {300}, 256); }, "arena oom");
    failures += expect_size(arena.used(), used_before_failures, "arena.used after oom");
    failures += expect_size(arena.peak_used(), peak_before_failures, "arena.peak after oom");

    arena.reset();
    failures += expect_size(arena.used(), 0, "arena.used after reset");
    failures += expect_size(arena.peak_used(), peak_before_failures, "arena.peak after reset");
    arena.reset_peak();
    failures += expect_size(arena.peak_used(), 0, "arena.peak after reset_peak on empty arena");
    ninfer::Tensor c = arena.alloc(ninfer::DType::U8, {4});
    failures += expect_ptr(c.data, base, "allocation after reset pointer");
    failures += expect_size(arena.peak_used(), 4, "arena.peak after reset allocation");

    ninfer::DeviceArena moved(std::move(arena));
    if (arena.base() != nullptr || arena.capacity() != 0 || arena.used() != 0) {
        ++failures;
        std::cerr << "move construction did not clear source arena\n";
    }
    failures += expect_size(arena.peak_used(), 0, "moved-from arena peak");
    failures += expect_size(moved.capacity(), 1024, "moved arena capacity");
    failures += expect_size(moved.peak_used(), 4, "moved arena peak");

    void* external = nullptr;
    CUDA_CHECK(cudaMalloc(&external, 512));
    {
        ninfer::DeviceArena borrowed(ninfer::DeviceSpan{external, 512});
        failures += expect_ptr(borrowed.base(), external, "borrowed arena base");
        failures += expect_size(borrowed.capacity(), 512, "borrowed arena capacity");
        const ninfer::Tensor item = borrowed.alloc(ninfer::DType::U8, {17}, 64);
        failures += expect_ptr(item.data, external, "borrowed arena allocation");
        failures += expect_size(borrowed.used(), 17, "borrowed arena used");
    }
    CUDA_CHECK(cudaMemset(external, 0, 512));
    CUDA_CHECK(cudaFree(external));

    ninfer::PinnedHostBuffer pinned(128);
    if (pinned.data() == nullptr) {
        ++failures;
        std::cerr << "pinned data is null\n";
    }
    failures += expect_size(pinned.size(), 128, "pinned.size");
    std::memset(pinned.data(), 0x5a, pinned.size());

#if defined(_WIN32)
    const bool initial_residency = ninfer::core::wddm_residency_lock_enabled();
    failures += expect(!initial_residency, "wddm_residency_lock_enabled should default to false");
    ninfer::core::set_wddm_residency_lock_enabled(true);
    failures += expect(ninfer::core::wddm_residency_lock_enabled(),
                       "wddm_residency_lock_enabled should be true after set(true)");

    // =========================================================================
    // Realistic Multi-Arena Startup Sequence Under Active Residency Lock
    // Represents NInfer's real startup chain:
    //   1. Model Weights Arena (materialized first)
    //   2. Persistent State / KV Cache Arena (allocated concurrently with weights)
    //   3. Workspace Scratchpad Arena (allocated concurrently with weights + KV)
    // =========================================================================
    {
        // 1. Model Weights Arena (16 MiB)
        constexpr std::size_t kWeightsBytes = 16 * 1024 * 1024;
        ninfer::DeviceArena weights_arena(kWeightsBytes);
        failures += expect(weights_arena.base() != nullptr, "weights arena base should be non-null");
        failures += expect_size(weights_arena.capacity(), kWeightsBytes, "weights arena capacity");

        const ninfer::Tensor embed_weights = weights_arena.alloc(ninfer::DType::U8, {128}, 64);
        const ninfer::Tensor attn_weights  = weights_arena.alloc(ninfer::DType::U8, {256}, 64);
        CUDA_CHECK(cudaMemset(embed_weights.data, 0x11, 128));
        CUDA_CHECK(cudaMemset(attn_weights.data, 0x22, 256));
        CUDA_CHECK(cudaDeviceSynchronize());

        // 2. Persistent State / KV Cache Arena (8 MiB) concurrently resident with weights
        constexpr std::size_t kKvBytes = 8 * 1024 * 1024;
        ninfer::DeviceArena kv_arena(kKvBytes);
        failures += expect(kv_arena.base() != nullptr, "kv arena base should be non-null");
        failures += expect_size(kv_arena.capacity(), kKvBytes, "kv arena capacity");
        failures += expect(kv_arena.base() != weights_arena.base(), "kv arena overlaps weights arena base");

        const std::uintptr_t w_start  = reinterpret_cast<std::uintptr_t>(weights_arena.base());
        const std::uintptr_t w_end    = w_start + weights_arena.capacity();
        const std::uintptr_t kv_start = reinterpret_cast<std::uintptr_t>(kv_arena.base());
        const std::uintptr_t kv_end   = kv_start + kv_arena.capacity();
        failures += expect(kv_end <= w_start || kv_start >= w_end, "kv arena overlaps weights arena range");

        const ninfer::Tensor kv_chunk = kv_arena.alloc(ninfer::DType::U8, {128}, 64);
        CUDA_CHECK(cudaMemset(kv_chunk.data, 0x33, 128));
        CUDA_CHECK(cudaDeviceSynchronize());

        // 3. Workspace Scratchpad Arena (4 MiB) concurrently resident with weights + KV
        constexpr std::size_t kWorkspaceBytes = 4 * 1024 * 1024;
        ninfer::DeviceArena workspace_arena(kWorkspaceBytes);
        failures += expect(workspace_arena.base() != nullptr, "workspace arena base should be non-null");
        failures += expect_size(workspace_arena.capacity(), kWorkspaceBytes, "workspace arena capacity");

        const std::uintptr_t ws_start = reinterpret_cast<std::uintptr_t>(workspace_arena.base());
        const std::uintptr_t ws_end   = ws_start + workspace_arena.capacity();
        failures += expect((ws_end <= w_start || ws_start >= w_end) &&
                           (ws_end <= kv_start || ws_start >= kv_end),
                           "workspace arena overlaps existing active arenas");

        const ninfer::Tensor scratch = workspace_arena.alloc(ninfer::DType::U8, {128}, 64);
        CUDA_CHECK(cudaMemset(scratch.data, 0x44, 128));
        CUDA_CHECK(cudaDeviceSynchronize());

        // 4. Verify simultaneous integrity across all 3 active resident arenas
        std::uint8_t probe_embed = 0, probe_attn = 0, probe_kv = 0, probe_ws = 0;
        CUDA_CHECK(cudaMemcpy(&probe_embed, embed_weights.data, 1, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&probe_attn, attn_weights.data, 1, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&probe_kv, kv_chunk.data, 1, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&probe_ws, scratch.data, 1, cudaMemcpyDeviceToHost));

        failures += expect(probe_embed == 0x11, "concurrent weights embed readback corrupted");
        failures += expect(probe_attn == 0x22, "concurrent weights attn readback corrupted");
        failures += expect(probe_kv == 0x33, "concurrent KV cache readback corrupted");
        failures += expect(probe_ws == 0x44, "concurrent workspace readback corrupted");

        // 5. Overbudget while weights arena is resident: must reject cleanly without corrupting live weights
        std::size_t free_b = 0;
        std::size_t total_b = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
        const std::size_t impossible_bytes = (total_b > 0 ? total_b : (24ULL << 30)) * 2ULL;

        failures += expect_throws<std::runtime_error>(
            [&] { ninfer::DeviceArena impossible_arena(impossible_bytes); },
            "overbudget residency allocation while weights live should fail with std::runtime_error");

        // CRITICAL: Live weights must remain fully intact and readable after rejection
        std::uint8_t probe_embed_post = 0;
        CUDA_CHECK(cudaMemcpy(&probe_embed_post, embed_weights.data, 1, cudaMemcpyDeviceToHost));
        failures += expect(probe_embed_post == 0x11, "live weights corrupted after overbudget rejection");

        // Verify CUDA context remains pristine (no sticky error)
        const cudaError_t pending_err = cudaGetLastError();
        failures += expect(pending_err == cudaSuccess,
                           "CUDA context has lingering error after overbudget rejection");

        // 6. Dynamic Recovery: Allocate a valid arena after overbudget rejection
        constexpr std::size_t kRecoveryBytes = 2 * 1024 * 1024;
        ninfer::DeviceArena recovery_arena(kRecoveryBytes);
        failures += expect(recovery_arena.base() != nullptr, "recovery arena base should be non-null");
        CUDA_CHECK(cudaMemset(recovery_arena.base(), 0x77, 64));
        CUDA_CHECK(cudaDeviceSynchronize());
        std::uint8_t recovery_probe = 0;
        CUDA_CHECK(cudaMemcpy(&recovery_probe, recovery_arena.base(), 1, cudaMemcpyDeviceToHost));
        failures += expect(recovery_probe == 0x77, "recovery arena readback pattern mismatch");
    }

    // =========================================================================
    // Standard Mode (Residency Lock Disabled) Startup Validation
    // =========================================================================
    ninfer::core::set_wddm_residency_lock_enabled(false);
    failures += expect(!ninfer::core::wddm_residency_lock_enabled(),
                       "wddm_residency_lock_enabled should be false after set(false)");

    {
        std::size_t free_b = 0;
        std::size_t total_b = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
        const std::size_t impossible_bytes = (total_b > 0 ? total_b : (24ULL << 30)) * 2ULL;

        // In standard mode, physical VRAM bound must also reject impossible sizes cleanly
        failures += expect_throws<std::runtime_error>(
            [&] { ninfer::DeviceArena impossible_arena(impossible_bytes); },
            "impossible allocation in standard mode should fail with std::runtime_error");

        const cudaError_t pending_err = cudaGetLastError();
        failures += expect(pending_err == cudaSuccess,
                           "CUDA context has lingering error after standard impossible rejection");
    }

    // Restore initial state
    if (initial_residency) { ninfer::core::set_wddm_residency_lock_enabled(true); }
#endif

    return failures == 0 ? 0 : fail("arena test failed");
}
