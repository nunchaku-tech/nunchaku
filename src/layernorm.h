#pragma once

#include "common.h"
#include "Tensor.h"
#include "Module.h"

class LayerNorm : public Module {
public:
    LayerNorm(int hidden_size, float eps, bool elementwise_affine, Tensor::ScalarType dtype, Device device);
    Tensor forward(Tensor x);

public:
    const int hidden_size;
    const float eps;

private:
    Tensor weight;
    Tensor bias;
};

class RMSNorm : public Module {
public:
    RMSNorm(int hidden_size, float eps, bool use_quant, Tensor::ScalarType dtype, Device device)
        : use_quant(use_quant), variance_epsilon(eps) {
        weight = Tensor::allocate({hidden_size}, dtype, device);
        registerParams(weight, "weight");
    }
    Tensor forward(Tensor x);

public:
    const bool use_quant;
    const float variance_epsilon;
    Tensor weight;
};

class RMSNormGeneral {
    friend class LlamaDecoderLayer;

public:
    RMSNormGeneral(int hidden_size, bool act_sum, float eps, bool use_per_token_quant, Device device)
        : act_sum(act_sum), use_per_token_quant(use_per_token_quant), variance_epsilon(eps) {
        this->weight = Tensor::ones({hidden_size}, Tensor::FP32, device);
    }
    void forward(Tensor x,
                 Tensor quantized_hidden_states_buffer,
                 Tensor quantized_scale_buffer,
                 Tensor quantized_sum_buffer) {
        if (act_sum) {
            forward_with_act_sum(x, quantized_hidden_states_buffer, quantized_scale_buffer, quantized_sum_buffer);
        } else {
            forward_wo_act_sum(x, quantized_hidden_states_buffer, quantized_scale_buffer, quantized_sum_buffer);
        }
    }

private:
    void forward_with_act_sum(Tensor x,
                              Tensor quantized_hidden_states_buffer,
                              Tensor quantized_scale_buffer,
                              Tensor quantized_sum_buffer);
    void forward_wo_act_sum(Tensor x,
                            Tensor quantized_hidden_states_buffer,
                            Tensor quantized_scale_buffer,
                            Tensor quantized_sum_buffer);

private:
    const bool act_sum;
    const bool use_per_token_quant;
    const float variance_epsilon;
    Tensor weight;
};
