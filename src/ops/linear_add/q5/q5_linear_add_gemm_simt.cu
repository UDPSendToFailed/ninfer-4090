#include "ops/linear_add/q5/q5_linear_add_kernels.h"

#include "ops/linear/q5/q5_launch.h"

namespace ninfer::ops::detail {

void q5_linear_add_split2_exact_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream) {
    launch_q5_linear_add_small_t_mma(x, w, residual_out, stream);
}

} // namespace ninfer::ops::detail
