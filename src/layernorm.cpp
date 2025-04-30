#include "layernorm.h"
#include "kernels/layernorm_kernels.h"

LayerNorm::LayerNorm(int hidden_size, float eps, bool elementwise_affine, Tensor::ScalarType dtype, Device device)
    : hidden_size(hidden_size), eps(eps) {
    if (elementwise_affine) {
        weight = Tensor::allocate({hidden_size}, dtype, device);
        bias   = Tensor::allocate({hidden_size}, dtype, device);
    }

    registerParams(weight, "weight")(bias, "bias");
}

Tensor LayerNorm::forward(Tensor x) {
    Tensor out = Tensor::empty(x.shape, x.scalar_type(), x.device());
    layernorm_general(out, x, this->weight, this->bias, this->eps);
    return out;
}

Tensor RMSNorm::forward(Tensor x) {
    Tensor out = Tensor::empty(x.shape, use_quant ? Tensor::INT8 : x.scalar_type(), x.device());
    rms_norm(out, x, this->weight, this->variance_epsilon, this->use_quant);
    return out;
}

void RMSNormGeneral::forward_with_act_sum(Tensor x,
                                          Tensor quantized_hidden_states_buffer,
                                          Tensor quantized_scale_buffer,
                                          Tensor quantized_sum_buffer) {
    rms_norm_general_fuse_sum(quantized_hidden_states_buffer,
                              x,
                              this->weight,
                              quantized_sum_buffer,
                              quantized_scale_buffer,
                              variance_epsilon,
                              use_per_token_quant);
}

void RMSNormGeneral::forward_wo_act_sum(Tensor x,
                                        Tensor quantized_hidden_states_buffer,
                                        Tensor quantized_scale_buffer,
                                        Tensor quantized_sum_buffer) {
    rms_norm_general(
        quantized_hidden_states_buffer, x, this->weight, quantized_scale_buffer, variance_epsilon, use_per_token_quant);
}
