#pragma once

#include "common.h"
#include "Tensor.h"

void silu(Tensor &out, // [..., d]
          Tensor &input);

void silu_and_mul(Tensor &out,    // [..., d]
                  Tensor &input); // [..., 2 * d]

void gelu_new(Tensor &out, Tensor &input);

void gelu_fast(Tensor &out, Tensor &input);

void invoke_dequant_silu_and_mul_quant(Tensor &out,   // [..., d]
                                       Tensor &input, // [..., 2 * d]
                                       const float scale_gate,
                                       const float scale_up,
                                       const float scale_out);

void invoke_dequant_silu_and_mul_quant(Tensor &out,   // [..., d]
                                       Tensor &input, // [..., 2 * d]
                                       const float scale_gate,
                                       const float scale_up,
                                       Tensor &scale_out, // [num_tokens]
                                       Tensor &tmp        // [num_tokens, d]
);
