#pragma once

#include "common.h"
#include "Tensor.h"
#include "Module.h"

class GEMM_F16 : public Module {
public:
    GEMM_F16(int in_features, int out_features, bool use_bias, Tensor::ScalarType dtype, Device device);

    Tensor forward(Tensor x);

public:
    const int in_features;
    const int out_features;

public:
    Tensor weight;
    Tensor bias;
};

class GEMV_AWQ : public Module {
public:
    GEMV_AWQ(int in_features, int out_features, bool use_bias, Tensor::ScalarType dtype, Device device);

    Tensor forward(Tensor x);

protected:
    virtual void loadParam(std::string key, Tensor &dst, Tensor src) override;

public:
    const int in_features;
    const int out_features;
    const int group_size;

    int lora_rank;
    float lora_scale;

    const Device device;

public:
    Tensor qweight;
    Tensor wscales;
    Tensor wzeros;
    Tensor bias;

    Tensor lora_down;
    Tensor lora_up;

    // std::shared_ptr<CUBLASWrapper> cublas;
};

class GEMM_W4A4 : public Module {
public:
    enum class FuseOptions {
        EMPTY = 0,
        GELU_QUANT,
        SILU,
    };
    struct QuantizedActivation {
        Tensor act;
        Tensor ascales;
        Tensor lora_act;
        bool is_unsigned = false;
        TensorShape actShape;
    };

public:
    GEMM_W4A4(int in_features, int out_features, bool bias, bool use_fp4, Tensor::ScalarType dtype, Device device);
    Tensor forward(Tensor x);
    Tensor forward_silu(Tensor x);
    std::variant<Tensor, QuantizedActivation> forward(Tensor x, FuseOptions fuse, GEMM_W4A4 *nextGEMM = nullptr);
    void forward(Tensor x,
                 Tensor out,
                 Tensor pool       = {},
                 Tensor norm_q     = {},
                 Tensor norm_k     = {},
                 Tensor rotary_emb = {},
                 Tensor out_q      = {},
                 Tensor out_k      = {},
                 Tensor out_v      = {},
                 int numTokens     = 0);
    std::variant<Tensor, QuantizedActivation>
    forward_quant(QuantizedActivation qact, FuseOptions fuse, GEMM_W4A4 *nextGEMM = nullptr);
    Tensor forward_quant(QuantizedActivation qact);

public:
    QuantizedActivation quantize(Tensor x, bool fuse_glu);

public:
    const int in_features;
    const int out_features;
    const int in_features_pad;
    const int out_features_pad;
    const bool use_fp4;

    int lora_rank;
    std::vector<float> lora_scales; // every 16 ranks share a scale

    const Tensor::ScalarType dtype;
    const Device device;

protected:
    virtual void loadParam(std::string key, Tensor &dst, Tensor src) override;

public:
    Tensor qweight;
    Tensor wscales;
    Tensor bias;

    Tensor lora_down;
    Tensor lora_up;

    Tensor smooth;

    Tensor wtscale;
    Tensor wcscales;

    cublasHandle_t handle;
};

class GEMM_W8A8 : public Module {
public:
    struct QuantizedActivation {
        Tensor act;
        Tensor ascales;
    };

public:
    GEMM_W8A8(int in_features, int out_features, bool bias, Tensor::ScalarType dtype, Device device);

public:
    QuantizedActivation quantize(Tensor x, bool fuse_glu);
    Tensor forward_quant(QuantizedActivation qact);
    Tensor forward(Tensor x) {
        return forward_quant(quantize(x, false));
    }

public:
    const int in_features;
    const int out_features;
    const Tensor::ScalarType dtype;

public:
    Tensor qweight;
    Tensor wscales;
    Tensor bias;
};

class DWCONV : public Module {
public:
    DWCONV(int in_features, bool bias, Tensor::ScalarType dtype, Device device);

    Tensor forward(Tensor x);

public:
    const int in_features;

public:
    Tensor weight;
    Tensor bias;
};
