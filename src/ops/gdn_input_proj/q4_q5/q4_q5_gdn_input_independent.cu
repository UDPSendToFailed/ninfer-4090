#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"

#include "core/device.h"
#include "core/pdl.cuh"
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

constexpr std::int32_t kQkRows     = 4096;
constexpr std::int32_t kValueRows  = 6144;
constexpr std::int32_t kZRows      = 6144;
constexpr std::int32_t kValueZRows = kValueRows + kZRows;
constexpr std::int32_t kHidden     = 5120;

using Q4GdnSimtR8C4Schedule = Q4RowSplitSimtGemmSchedule<8, 4, 16, 2, Cache::ca, 1>;
using Q4GdnSimtR8C8Schedule = Q4RowSplitSimtGemmSchedule<8, 8, 16, 2, Cache::ca, 1>;

void launch_q4_gemv(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = Q4GemvR1W8DirectSchedule;
    const dim3 grid(static_cast<unsigned>(div_up(kQkRows, Schedule::kRowsPerCta)), 1u, 1u);
    constexpr dim3 block(static_cast<unsigned>(Schedule::kThreads), 1u, 1u);
    q4_rowsplit_gemv_kernel<Schedule><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data),
        nullptr, kQkRows, kHidden);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule, bool Full>
void launch_q4_simt(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const std::int32_t cols   = x.ne[1];
    const std::int32_t out_ld = static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));
    const dim3 grid(static_cast<unsigned>(div_up(kQkRows, Schedule::kRowsPerCta)),
                    static_cast<unsigned>(div_up(cols, Schedule::kColsPerTile)), 1u);
    q4_rowsplit_gemm_simt_kernel<Schedule, Full><<<grid, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data),
        nullptr, out_ld, 0, kQkRows, kHidden, cols, weight.padded_shape[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule>
void launch_q4_simt_route(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const bool full = (kQkRows % Schedule::kRowsPerCta) == 0 &&
                      ((kHidden / Q4RowSplitStorage::kGroupK) % Schedule::kGroupsPerStage) == 0 &&
                      (x.ne[1] % Schedule::kColsPerTile) == 0;
    if (full) {
        launch_q4_simt<Schedule, true>(x, weight, out, stream);
    } else {
        launch_q4_simt<Schedule, false>(x, weight, out, stream);
    }
}

template <int ActiveCols>
void launch_q4_gdn_small_t_mma_active(const Tensor& x, const Weight& weight, Tensor& out,
                                      cudaStream_t stream) {
    constexpr int TileCols = ActiveCols <= 8 ? 8 : 16;
    using Geometry         = Q4SmallTGeometry<kQkRows, kHidden>;
    using Epilogue         = Q4SmallTStrideEpilogue;
    constexpr int kBlocks  = kQkRows / Q4DraftSmallTSchedule::kRowsPerCta; // 4096 / 16 = 256
    const auto out_ld      = static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));
    const Epilogue epilogue{static_cast<__nv_bfloat16*>(out.data), out_ld};

    q4_small_t_mma_kernel<Geometry, TileCols, ActiveCols, Epilogue>
        <<<kBlocks, Q4DraftSmallTSchedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales),
            static_cast<__nv_bfloat16*>(out.data), epilogue);
    CUDA_CHECK(cudaGetLastError());
}

using Q4GdnSmallTLauncher = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <std::size_t... Offsets>
constexpr auto make_q4_gdn_small_t_launchers(std::index_sequence<Offsets...>) {
    return std::array<Q4GdnSmallTLauncher, sizeof...(Offsets)>{
        &launch_q4_gdn_small_t_mma_active<2 + static_cast<int>(Offsets)>...};
}

constexpr auto kQ4GdnSmallTLaunchers =
    make_q4_gdn_small_t_launchers(std::make_index_sequence<15>{}); // 2..16

void launch_q4(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    if (x.ne[1] == 1) {
        launch_q4_gemv(x, weight, out, stream);
        return;
    }
    if (x.ne[1] <= 16) {
        kQ4GdnSmallTLaunchers[static_cast<std::size_t>(x.ne[1] - 2)](x, weight, out, stream);
        return;
    }
    throw std::invalid_argument("Q4/Q5 GDN independent launch requires T in [1,16]");
}

void launch_q5_gemv(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
                    cudaStream_t stream) {
    constexpr int kRowsPerBlock = 16;
    constexpr int kThreads      = kRowsPerBlock * 32;
    q5_rowsplit_gemv_kernel<kValueZRows, kHidden, kRowsPerBlock, 2, true, false, true, kValueRows>
        <<<kValueZRows / kRowsPerBlock, kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.qhigh),
            static_cast<const std::uint8_t*>(weight.scales),
            static_cast<__nv_bfloat16*>(value.data), static_cast<__nv_bfloat16*>(z.data));
    CUDA_CHECK(cudaGetLastError());
}

template <int Cols>
void launch_q5_split4_rows(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
                           cudaStream_t stream) {
    constexpr int kThreads    = 4 * 32;
    constexpr int kRows       = 2;
    const std::int32_t out_ld = static_cast<std::int32_t>(value.nb[1] / sizeof(__nv_bfloat16));
    const dim3 grid(static_cast<unsigned>(div_up(kValueZRows, kRows)), 1u, 1u);
    q5_rowsplit_gemm_simt_split4_rows_kernel<Q5RowSplitSimtSchedule, kRows, Cols, 5, kHidden, true,
                                             kValueRows><<<grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.qhigh),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(value.data),
        static_cast<__nv_bfloat16*>(z.data), kValueZRows, out_ld, kHidden, Cols,
        weight.padded_shape[1], 5);
    CUDA_CHECK(cudaGetLastError());
}

template <int Cols>
void launch_q5_split4(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
                      cudaStream_t stream) {
    // Two rows per CTA share one widened activation. Below eight columns the extra accumulators
    // cost more than the shared conversion saves, so the single-row kernel stays.
    if constexpr (Cols >= 8) {
        launch_q5_split4_rows<Cols>(x, weight, value, z, stream);
        return;
    }
    constexpr int kThreads    = 4 * 32;
    const std::int32_t out_ld = static_cast<std::int32_t>(value.nb[1] / sizeof(__nv_bfloat16));
    const dim3 grid(static_cast<unsigned>(kValueZRows), 1u, 1u);
    q5_rowsplit_gemm_simt_split4_kernel<Q5RowSplitSimtSchedule, Cols, 5, kHidden, true, kValueRows>
        <<<grid, kThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                        static_cast<const std::uint8_t*>(weight.qdata),
                                        static_cast<const std::uint8_t*>(weight.qhigh),
                                        static_cast<const std::uint8_t*>(weight.scales),
                                        static_cast<__nv_bfloat16*>(value.data),
                                        static_cast<__nv_bfloat16*>(z.data), kValueZRows, out_ld,
                                        kHidden, Cols, weight.padded_shape[1], 5);
    CUDA_CHECK(cudaGetLastError());
}

void launch_q5_split4_exact(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
                            cudaStream_t stream) {
    switch (x.ne[1]) {
    case 2:
        launch_q5_split4<2>(x, weight, value, z, stream);
        return;
    case 3:
        launch_q5_split4<3>(x, weight, value, z, stream);
        return;
    case 4:
        launch_q5_split4<4>(x, weight, value, z, stream);
        return;
    case 5:
        launch_q5_split4<5>(x, weight, value, z, stream);
        return;
    case 6:
        launch_q5_split4<6>(x, weight, value, z, stream);
        return;
    case 7:
        launch_q5_split4<7>(x, weight, value, z, stream);
        return;
    case 8:
        launch_q5_split4<8>(x, weight, value, z, stream);
        return;
    case 9:
        launch_q5_split4<9>(x, weight, value, z, stream);
        return;
    case 10:
        launch_q5_split4<10>(x, weight, value, z, stream);
        return;
    case 11:
        launch_q5_split4<11>(x, weight, value, z, stream);
        return;
    case 12:
        launch_q5_split4<12>(x, weight, value, z, stream);
        return;
    case 13:
        launch_q5_split4<13>(x, weight, value, z, stream);
        return;
    case 14:
        launch_q5_split4<14>(x, weight, value, z, stream);
        return;
    case 15:
        launch_q5_split4<15>(x, weight, value, z, stream);
        return;
    case 16:
        launch_q5_split4<16>(x, weight, value, z, stream);
        return;
    default:
        throw std::invalid_argument("GDN Q5 split4 requires T in [2,16]");
    }
}

void launch_q5_simt_r8_c8(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
                          cudaStream_t stream) {
    constexpr int kColsPerTile  = 8;
    constexpr int kRowsPerBlock = 8;
    constexpr int kStages       = 2;
    constexpr int kThreads      = kRowsPerBlock * 32;
    const std::int32_t cols     = x.ne[1];
    const std::int32_t out_ld   = static_cast<std::int32_t>(value.nb[1] / sizeof(__nv_bfloat16));
    const dim3 grid(static_cast<unsigned>(div_up(kValueZRows, kRowsPerBlock)),
                    static_cast<unsigned>(div_up(cols, kColsPerTile)), 1u);
    q5_rowsplit_gemm_simt_kernel<Q5RowSplitSimtSchedule, kColsPerTile, kRowsPerBlock, kStages, true,
                                 kValueRows><<<grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.qhigh),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(value.data),
        static_cast<__nv_bfloat16*>(z.data), kValueZRows, out_ld, kHidden, cols,
        weight.padded_shape[1], 5);
    CUDA_CHECK(cudaGetLastError());
}

template <int ActiveCols>
void launch_q5_gdn_small_t_mma_active(const Tensor& x, const Weight& weight, Tensor& value,
                                      Tensor& z, cudaStream_t stream) {
    constexpr int TileCols = ActiveCols <= 8 ? 8 : 16;
    using Geometry         = Q5SmallTGeometry<kValueZRows, kHidden>;
    using Epilogue         = Q5SmallTSplitEpilogue<kValueRows>;
    constexpr int kBlocks  = kValueZRows / Q5SmallTSchedule::kRowsPerCta; // 12288 / 16 = 768
    const auto in_ld       = static_cast<std::int32_t>(x.nb[1] / sizeof(__nv_bfloat16));
    const auto value_ld    = static_cast<std::int32_t>(value.nb[1] / sizeof(__nv_bfloat16));
    const auto z_ld        = static_cast<std::int32_t>(z.nb[1] / sizeof(__nv_bfloat16));
    const Epilogue epilogue{static_cast<__nv_bfloat16*>(z.data), z_ld};

    q5_small_t_mma_kernel<Geometry, TileCols, ActiveCols, Epilogue>
        <<<kBlocks, Q5SmallTSchedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.qhigh),
            static_cast<const std::uint8_t*>(weight.scales),
            static_cast<__nv_bfloat16*>(value.data), in_ld, value_ld, epilogue);
    CUDA_CHECK(cudaGetLastError());
}

using Q5GdnSmallTLauncher = void (*)(const Tensor&, const Weight&, Tensor&, Tensor&, cudaStream_t);

template <std::size_t... Offsets>
constexpr auto make_q5_gdn_small_t_launchers(std::index_sequence<Offsets...>) {
    return std::array<Q5GdnSmallTLauncher, sizeof...(Offsets)>{
        &launch_q5_gdn_small_t_mma_active<2 + static_cast<int>(Offsets)>...};
}

constexpr auto kQ5GdnSmallTLaunchers =
    make_q5_gdn_small_t_launchers(std::make_index_sequence<15>{}); // 2..16

void launch_q5(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
               cudaStream_t stream) {
    if (x.ne[1] == 1) {
        launch_q5_gemv(x, weight, value, z, stream);
        return;
    }
    if (x.ne[1] <= 16) {
        kQ5GdnSmallTLaunchers[static_cast<std::size_t>(x.ne[1] - 2)](x, weight, value, z, stream);
        return;
    }
    throw std::invalid_argument("Q4/Q5 GDN independent launch requires T in [1,16]");
}

} // namespace

void q4_q5_gdn_input_independent_launch(const Tensor& x, const Weight& qk_weight,
                                        const Weight& value_z_weight, Tensor& qk, Tensor& value,
                                        Tensor& z, cudaStream_t stream) {
    launch_q4(x, qk_weight, qk, stream);
    launch_q5(x, value_z_weight, value, z, stream);
}

} // namespace ninfer::ops::detail
