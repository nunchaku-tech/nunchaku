#pragma once

#include "common.h"
#include "Tensor.h"
#include <cuda_fp16.h>

void rms_norm(Tensor &out,    // [num_tokens, hidden_size]
              Tensor &input,  // [num_tokens, hidden_size]
              Tensor &weight, // [hidden_size]
              float epsilon,
              bool use_quant);

void layernorm_general(Tensor out, Tensor input, Tensor weight, Tensor bias, float epsilon);

void rms_norm_general(Tensor &out,     // [..., hidden_size]
                      Tensor &input,   // [..., hidden_size]
                      Tensor &weight,  // [hidden_size]
                      Tensor &scaling, // [tokens] or [1]
                      float epsilon,
                      bool use_per_token_quant);

void rms_norm_general_fuse_sum(Tensor &out,       // [..., hidden_size]
                               Tensor &input,     // [..., hidden_size]
                               Tensor &weight,    // [hidden_size]
                               Tensor &input_sum, // [tokens] or [1]
                               Tensor &scaling,   // [tokens] or [1]
                               float epsilon,
                               bool use_per_token_quant);

void invoke_dequant_add_residual_rms_norm_quant(Tensor &out,      // [..., hidden_size]
                                                Tensor &input,    // [..., hidden_size]
                                                Tensor &residual, // [..., hidden_size]
                                                Tensor &gamma,    // [hidden_size]
                                                half scale,
                                                float epsilon);

void invoke_dequant_add_residual_rms_norm_quant(Tensor &out,      // [..., hidden_size]
                                                Tensor &input,    // [..., hidden_size]
                                                Tensor &residual, // [..., hidden_size]
                                                Tensor &gamma,    // [hidden_size]
                                                Tensor &scale,    // [num_tokens]
                                                float epsilon);
