#pragma once

#include "common.h"
#include "Tensor.h"

class Silu {
public:
    static Tensor forward(Tensor x);
};

class GELU {
public:
    static Tensor forward(Tensor x);
};

// class SiluAndMul {
// public:
//     static Tensor forward(Tensor x);
// };

// class SiluAndMulQuant {
// public:
//     static Tensor forward(Tensor x, Tensor quantized_mlp_act_buffer, Tensor quantized_scale_buffer, Tensor
//     quantized_sum_buffer, bool act_sum) {
//         if (act_sum) {
//             return forward_with_act_sum(x, quantized_mlp_act_buffer, quantized_scale_buffer, quantized_sum_buffer);
//         } else {
//             return forward_wo_act_sum(x, quantized_mlp_act_buffer, quantized_scale_buffer, quantized_sum_buffer);
//         }
//     }
// private:
//     static Tensor forward_with_act_sum(Tensor x, Tensor quantized_mlp_act_buffer, Tensor quantized_scale_buffer,
//     Tensor quantized_sum_buffer); static Tensor forward_wo_act_sum(Tensor x, Tensor quantized_mlp_act_buffer, Tensor
//     quantized_scale_buffer, Tensor quantized_sum_buffer);
// };
