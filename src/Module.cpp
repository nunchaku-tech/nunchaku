#include "common.h"
#include "Module.h"
#include "kernels/misc_kernels.h"

void Module::copyWithCast(Tensor dst, Tensor src) {
    assert(dst.is_contiguous());
    assert(dst.device().type == Device::CUDA);

    if (src.device().type == Device::CUDA && src.device().idx == dst.device().idx) {
        nunchaku::kernels::cast(src, dst);
    } else {
        Tensor tmp;
        tmp.buffer     = dst.buffer;
        tmp.shape      = dst.shape;
        tmp.scalarType = src.scalarType;
        tmp.copy_(src);
        nunchaku::kernels::cast(tmp, dst);
    }
}
