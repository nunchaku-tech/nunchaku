#pragma once

#include "common.h"
#include "Tensor.h"

namespace nunchaku::kernels {

Tensor add(Tensor a, Tensor b);
void mul_add(Tensor x, Tensor scale, Tensor bias);
void mul_add_batch(Tensor x, Tensor scale, bool batch_scale, double scale_shift, Tensor bias, bool batch_bias);

Tensor embedding(Tensor input_id, Tensor lookup);
Tensor argmax_sample(Tensor logits);
void splitqkv(Tensor qkv, Tensor q, Tensor k, Tensor v);
Tensor quant_static(Tensor x, float scale);
Tensor quant_static_fuse_gelu(Tensor x, float scale);

void cast(Tensor input, Tensor output);

Tensor topk(Tensor x, int k);

template<size_t N>
std::array<Tensor, N> split_mod(Tensor input);

}; // namespace nunchaku::kernels
