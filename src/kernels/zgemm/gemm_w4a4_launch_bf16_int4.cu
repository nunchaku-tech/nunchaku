#include "gemm_w4a4_launch_impl.cuh"

namespace nunchaku::kernels {
template class GEMM_W4A4_Launch<GEMMConfig_W4A4_BF16, false>;
};
