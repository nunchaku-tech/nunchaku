#include "activation.h"
#include "kernels/activation_kernels.h"

Tensor Silu::forward(Tensor x) {
    Tensor out = Tensor::empty_like(x);
    silu(out, x);
    return out;
}

Tensor GELU::forward(Tensor x) {
    Tensor out = Tensor::empty_like(x);
    gelu_new(out, x);
    return out;
}

// Tensor SiluAndMul::forward(Tensor x) {
//     int d = x.shape[-1] / 2;
//     auto output_shape = x.shape;
//     output_shape[-1] = d;
//     Tensor out = Tensor::empty(output_shape, x.scalar_type(), x.device());
//     silu_and_mul(out, x);
//     return out;
// }

// Tensor SiluAndMulQuant::forward_with_act_sum(Tensor x, Tensor quantized_mlp_act_buffer, Tensor
// quantized_scale_buffer, Tensor quantized_sum_buffer) {
//     Tensor out = SiluAndMul::forward(x);
//     invoke_quant_fuse_sum(quantized_mlp_act_buffer, out, quantized_sum_buffer, quantized_scale_buffer);
//     return out;
// }

// Tensor SiluAndMulQuant::forward_wo_act_sum(Tensor x, Tensor quantized_mlp_act_buffer, Tensor quantized_scale_buffer,
// Tensor quantized_sum_buffer) {
//     Tensor out = SiluAndMul::forward(x);
//     invoke_quant(quantized_mlp_act_buffer, out, quantized_scale_buffer, {});
//     return out;
// }
