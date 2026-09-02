#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/q4/q4_rowsplit_gemm_simt.cuh"
#include "ops/linear/q4/q4_rowsplit_gemv.cuh"
#include "ops/linear/q4/q4_small_t_mma.cuh"
#include "ops/linear/q5/q5_rowsplit_gemm_simt.cuh"
#include "ops/linear/q5/q5_rowsplit_gemv.cuh"
#include "ops/linear/q5/q5_small_t_mma.cuh"

#include <cuda_bf16.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kParentRows = 7168;
constexpr std::int32_t kSplitRow   = 6144;
constexpr std::int32_t kHidden     = 5120;

using Q4AttnSimtR8C4Schedule = Q4RowSplitSimtGemmSchedule<8, 4, 16, 2, Cache::ca, 1>;
using Q4AttnSimtR8C8Schedule = Q4RowSplitSimtGemmSchedule<8, 8, 16, 2, Cache::ca, 1>;

void launch_q4_gemv(const Tensor& x, const Weight& weight, Tensor& q, Tensor& key,
                    cudaStream_t stream) {
    using Schedule = Q4GemvR1W8DirectSchedule;
    const dim3 grid(static_cast<unsigned>(div_up(kParentRows, Schedule::kRowsPerCta)), 1u, 1u);
    constexpr dim3 block(static_cast<unsigned>(Schedule::kThreads), 1u, 1u);
    q4_rowsplit_gemv_kernel<Schedule, true, kSplitRow><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(key.data), kParentRows, kHidden);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule, bool Full>
void launch_q4_simt(const Tensor& x, const Weight& weight, Tensor& q, Tensor& key,
                    cudaStream_t stream) {
    const std::int32_t cols = x.ne[1];
    const dim3 grid(static_cast<unsigned>(div_up(kParentRows, Schedule::kRowsPerCta)),
                    static_cast<unsigned>(div_up(cols, Schedule::kColsPerTile)), 1u);
    q4_rowsplit_gemm_simt_kernel<Schedule, Full, true, kSplitRow>
        <<<grid, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(q.data),
            static_cast<__nv_bfloat16*>(key.data), q.ne[0], key.ne[0], kParentRows, kHidden, cols,
            weight.padded_shape[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule>
void launch_q4_simt_route(const Tensor& x, const Weight& weight, Tensor& q, Tensor& key,
                          cudaStream_t stream) {
    const bool full = (kParentRows % Schedule::kRowsPerCta) == 0 &&
                      ((kHidden / Q4RowSplitStorage::kGroupK) % Schedule::kGroupsPerStage) == 0 &&
                      (x.ne[1] % Schedule::kColsPerTile) == 0;
    if (full) {
        launch_q4_simt<Schedule, true>(x, weight, q, key, stream);
    } else {
        launch_q4_simt<Schedule, false>(x, weight, q, key, stream);
    }
}

template <int ActiveCols>
void launch_q4_attn_small_t_mma_active(const Tensor& x, const Weight& weight, Tensor& q,
                                       Tensor& key, cudaStream_t stream) {
    constexpr int TileCols = ActiveCols <= 8 ? 8 : 16;
    using Geometry         = Q4SmallTGeometry<kParentRows, kHidden>;
    using Epilogue         = Q4SmallTSplitEpilogue<kSplitRow>;
    constexpr int kBlocks  = kParentRows / Q4DraftSmallTSchedule::kRowsPerCta; // 7168 / 16 = 448
    const auto q_ld        = static_cast<std::int32_t>(q.nb[1] / sizeof(__nv_bfloat16));
    const auto key_ld      = static_cast<std::int32_t>(key.nb[1] / sizeof(__nv_bfloat16));
    const Epilogue epilogue{static_cast<__nv_bfloat16*>(q.data),
                            static_cast<__nv_bfloat16*>(key.data), q_ld, key_ld};

    q4_small_t_mma_kernel<Geometry, TileCols, ActiveCols, Epilogue>
        <<<kBlocks, Q4DraftSmallTSchedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales),
            static_cast<__nv_bfloat16*>(q.data), epilogue);
    CUDA_CHECK(cudaGetLastError());
}

using Q4AttnSmallTLauncher = void (*)(const Tensor&, const Weight&, Tensor&, Tensor&, cudaStream_t);

template <std::size_t... Offsets>
constexpr auto make_q4_attn_small_t_launchers(std::index_sequence<Offsets...>) {
    return std::array<Q4AttnSmallTLauncher, sizeof...(Offsets)>{
        &launch_q4_attn_small_t_mma_active<2 + static_cast<int>(Offsets)>...};
}

constexpr auto kQ4AttnSmallTLaunchers =
    make_q4_attn_small_t_launchers(std::make_index_sequence<15>{}); // 2..16

void launch_q4(const Tensor& x, const Weight& weight, Tensor& q, Tensor& key, cudaStream_t stream) {
    if (x.ne[1] == 1) {
        launch_q4_gemv(x, weight, q, key, stream);
        return;
    }
    if (x.ne[1] <= 16) {
        kQ4AttnSmallTLaunchers[static_cast<std::size_t>(x.ne[1] - 2)](x, weight, q, key, stream);
        return;
    }
    throw std::invalid_argument("attention Q4 split-output requires T in [1,16]");
}

void launch_q5_gemv(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
                    cudaStream_t stream) {
    constexpr int kRowsPerBlock = 16;
    constexpr int kBlockThreads = kRowsPerBlock * 32;
    constexpr int kGrid         = kParentRows / kRowsPerBlock;
    q5_rowsplit_gemv_kernel<kParentRows, kHidden, kRowsPerBlock, 2, true, false, true, kSplitRow>
        <<<kGrid, kBlockThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                              static_cast<const std::uint8_t*>(weight.qdata),
                                              static_cast<const std::uint8_t*>(weight.qhigh),
                                              static_cast<const std::uint8_t*>(weight.scales),
                                              static_cast<__nv_bfloat16*>(gate.data),
                                              static_cast<__nv_bfloat16*>(value.data));
    CUDA_CHECK(cudaGetLastError());
}

template <int Cols>
void launch_q5_split4_rows(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
                           cudaStream_t stream) {
    constexpr int kThreads = 4 * 32;
    constexpr int kRows    = 2;
    const dim3 grid(static_cast<unsigned>(div_up(kParentRows, kRows)), 1u, 1u);
    q5_rowsplit_gemm_simt_split4_rows_kernel<Q5RowSplitSimtSchedule, kRows, Cols, 5, kHidden, true,
                                             kSplitRow><<<grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.qhigh),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(gate.data),
        static_cast<__nv_bfloat16*>(value.data), kParentRows, gate.ne[0], kHidden, Cols,
        weight.padded_shape[1], 5);
    CUDA_CHECK(cudaGetLastError());
}

template <int Cols>
void launch_q5_split4(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
                      cudaStream_t stream) {
    // Same eight-column crossover as the GDN value_z projection.
    if constexpr (Cols >= 8) {
        launch_q5_split4_rows<Cols>(x, weight, gate, value, stream);
        return;
    }
    constexpr int kThreads = 4 * 32;
    const dim3 grid(static_cast<unsigned>(kParentRows), 1u, 1u);
    q5_rowsplit_gemm_simt_split4_kernel<Q5RowSplitSimtSchedule, Cols, 5, kHidden, true, kSplitRow>
        <<<grid, kThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                        static_cast<const std::uint8_t*>(weight.qdata),
                                        static_cast<const std::uint8_t*>(weight.qhigh),
                                        static_cast<const std::uint8_t*>(weight.scales),
                                        static_cast<__nv_bfloat16*>(gate.data),
                                        static_cast<__nv_bfloat16*>(value.data), kParentRows,
                                        gate.ne[0], kHidden, Cols, weight.padded_shape[1], 5);
    CUDA_CHECK(cudaGetLastError());
}

void launch_q5_split4_exact(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
                            cudaStream_t stream) {
    switch (x.ne[1]) {
    case 2:
        launch_q5_split4<2>(x, weight, gate, value, stream);
        return;
    case 3:
        launch_q5_split4<3>(x, weight, gate, value, stream);
        return;
    case 4:
        launch_q5_split4<4>(x, weight, gate, value, stream);
        return;
    case 5:
        launch_q5_split4<5>(x, weight, gate, value, stream);
        return;
    case 6:
        launch_q5_split4<6>(x, weight, gate, value, stream);
        return;
    case 7:
        launch_q5_split4<7>(x, weight, gate, value, stream);
        return;
    case 8:
        launch_q5_split4<8>(x, weight, gate, value, stream);
        return;
    case 9:
        launch_q5_split4<9>(x, weight, gate, value, stream);
        return;
    case 10:
        launch_q5_split4<10>(x, weight, gate, value, stream);
        return;
    case 11:
        launch_q5_split4<11>(x, weight, gate, value, stream);
        return;
    case 12:
        launch_q5_split4<12>(x, weight, gate, value, stream);
        return;
    case 13:
        launch_q5_split4<13>(x, weight, gate, value, stream);
        return;
    case 14:
        launch_q5_split4<14>(x, weight, gate, value, stream);
        return;
    case 15:
        launch_q5_split4<15>(x, weight, gate, value, stream);
        return;
    case 16:
        launch_q5_split4<16>(x, weight, gate, value, stream);
        return;
    default:
        throw std::invalid_argument("attention Q5 split4 requires T in [2,16]");
    }
}

template <int ColsPerTile>
void launch_q5_simt(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
                    cudaStream_t stream) {
    constexpr int kRowsPerBlock = 8;
    constexpr int kStages       = 2;
    constexpr int kThreads      = kRowsPerBlock * 32;
    const std::int32_t cols     = x.ne[1];
    const dim3 grid(static_cast<unsigned>(div_up(kParentRows, kRowsPerBlock)),
                    static_cast<unsigned>(div_up(cols, ColsPerTile)), 1u);
    q5_rowsplit_gemm_simt_kernel<Q5RowSplitSimtSchedule, ColsPerTile, kRowsPerBlock, kStages, true,
                                 kSplitRow><<<grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.qhigh),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(gate.data),
        static_cast<__nv_bfloat16*>(value.data), kParentRows, gate.ne[0], kHidden, cols,
        weight.padded_shape[1], 5);
    CUDA_CHECK(cudaGetLastError());
}

template <int ActiveCols>
void launch_q5_attn_small_t_mma_active(const Tensor& x, const Weight& weight, Tensor& gate,
                                       Tensor& value, cudaStream_t stream) {
    constexpr int TileCols = ActiveCols <= 8 ? 8 : 16;
    using Geometry         = Q5SmallTGeometry<kParentRows, kHidden>;
    using Epilogue         = Q5SmallTSplitEpilogue<kSplitRow>;
    constexpr int kBlocks  = kParentRows / Q5SmallTSchedule::kRowsPerCta; // 7168 / 16 = 448
    const auto in_ld       = static_cast<std::int32_t>(x.nb[1] / sizeof(__nv_bfloat16));
    const auto gate_ld     = static_cast<std::int32_t>(gate.nb[1] / sizeof(__nv_bfloat16));
    const auto value_ld    = static_cast<std::int32_t>(value.nb[1] / sizeof(__nv_bfloat16));
    const Epilogue epilogue{static_cast<__nv_bfloat16*>(value.data), value_ld};

    q5_small_t_mma_kernel<Geometry, TileCols, ActiveCols, Epilogue>
        <<<kBlocks, Q5SmallTSchedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.qhigh),
            static_cast<const std::uint8_t*>(weight.scales),
            static_cast<__nv_bfloat16*>(gate.data), in_ld, gate_ld, epilogue);
    CUDA_CHECK(cudaGetLastError());
}

using Q5AttnSmallTLauncher = void (*)(const Tensor&, const Weight&, Tensor&, Tensor&, cudaStream_t);

template <std::size_t... Offsets>
constexpr auto make_q5_attn_small_t_launchers(std::index_sequence<Offsets...>) {
    return std::array<Q5AttnSmallTLauncher, sizeof...(Offsets)>{
        &launch_q5_attn_small_t_mma_active<2 + static_cast<int>(Offsets)>...};
}

constexpr auto kQ5AttnSmallTLaunchers =
    make_q5_attn_small_t_launchers(std::make_index_sequence<15>{}); // 2..16

void launch_q5(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
               cudaStream_t stream) {
    if (x.ne[1] == 1) {
        launch_q5_gemv(x, weight, gate, value, stream);
        return;
    }
    if (x.ne[1] <= 16) {
        kQ5AttnSmallTLaunchers[static_cast<std::size_t>(x.ne[1] - 2)](x, weight, gate, value, stream);
        return;
    }
    throw std::invalid_argument("attention Q5 split-output requires T in [1,16]");
}

struct StreamForkJoinContext {
    cudaStream_t side_stream = nullptr;
    cudaEvent_t fork_event   = nullptr;
    cudaEvent_t join_event   = nullptr;

    StreamForkJoinContext() {
        CUDA_CHECK(cudaStreamCreateWithFlags(&side_stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&fork_event, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&join_event, cudaEventDisableTiming));
    }

    ~StreamForkJoinContext() {
        if (join_event != nullptr) {
            cudaEventDestroy(join_event);
            join_event = nullptr;
        }
        if (fork_event != nullptr) {
            cudaEventDestroy(fork_event);
            fork_event = nullptr;
        }
        if (side_stream != nullptr) {
            cudaStreamDestroy(side_stream);
            side_stream = nullptr;
        }
    }
};

StreamForkJoinContext& get_fork_join_context() {
    static thread_local StreamForkJoinContext ctx;
    return ctx;
}

} // namespace

void q4_q5_attn_input_small_t_launch(const Tensor& x, const Weight& query_key_weight,
                                     const Weight& gate_value_weight, Tensor& q, Tensor& gate,
                                     Tensor& k, Tensor& v, cudaStream_t stream) {
    auto& ctx = get_fork_join_context();

    // 1. Fork: record event on origin stream, have side_stream wait on it
    CUDA_CHECK(cudaEventRecord(ctx.fork_event, stream));
    CUDA_CHECK(cudaStreamWaitEvent(ctx.side_stream, ctx.fork_event, 0));

    // 2. Co-schedule concurrent launches across 128 SMs:
    //    launch_q4 (448 blocks) + launch_q5 (448 blocks) = 896 blocks = 7.0 exact integer waves
    launch_q4(x, query_key_weight, q, k, stream);
    launch_q5(x, gate_value_weight, gate, v, ctx.side_stream);

    // 3. Join: record event on side_stream, have origin stream wait on it
    CUDA_CHECK(cudaEventRecord(ctx.join_event, ctx.side_stream));
    CUDA_CHECK(cudaStreamWaitEvent(stream, ctx.join_event, 0));
}

} // namespace ninfer::ops::detail
