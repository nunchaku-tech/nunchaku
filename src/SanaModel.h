#pragma once

#include "common.h"
#include "Tensor.h"
#include "Linear.h"
#include "layernorm.h"

class SanaLinearAttention : public Module {
public:
    SanaLinearAttention(int dim, bool bias, bool pag, bool use_fp4, Tensor::ScalarType dtype, Device device);

    Tensor forward(Tensor x, Tensor out = {});
    Tensor forward_pag(Tensor x, bool cfg);

public:
    const int dim;
    const int dim_pad;

private:
    GEMM_W4A4 qkv_proj;
    GEMM_W4A4 out_proj;

    std::optional<GEMM_W4A4> pag_to_v;
};

class MultiHeadCrossAttention : public Module {
public:
    MultiHeadCrossAttention(int num_heads, int head_dim, bool use_fp4, Tensor::ScalarType dtype, Device device);

    Tensor forward(Tensor x, Tensor cond, Tensor cu_seqlens_img, Tensor cu_seqlens_txt);

public:
    const int num_heads;
    const int head_dim;

private:
    GEMM_W4A4 q_linear;
    GEMM_F16 kv_linear;
    GEMM_W4A4 out_proj;
};

class SanaGLUMBConv : public Module {
public:
    SanaGLUMBConv(int in_features, int hidden_features, bool use_fp4, Tensor::ScalarType dtype, Device device);

    Tensor forward(Tensor x, int H, int W);

public:
    const int in_features;
    const int hidden_features;

private:
    GEMM_W4A4 inverted_conv;
    DWCONV depth_conv;
    GEMM_W4A4 point_conv;
};

class SanaLinearTransformerBlock : public Module {
public:
    SanaLinearTransformerBlock(int hidden_size,
                               int intermediate_size,
                               int num_cross_attention_heads,
                               bool pag,
                               bool use_fp4,
                               Tensor::ScalarType dtype,
                               Device device);

    Tensor forward(Tensor hidden_states,
                   Tensor encoder_hidden_states,
                   Tensor timestep,
                   Tensor cu_seqlens_img,
                   Tensor cu_seqlens_txt,
                   int H,
                   int W,
                   bool pag,
                   bool cfg);

public:
    const int hidden_size;
    const int num_cross_attention_heads;

private:
    Tensor scale_shift_table;
    // Tensor ones;

    SanaLinearAttention attn;
    MultiHeadCrossAttention cross_attn;
    SanaGLUMBConv ff;

    LayerNorm norm1, norm2;
};

struct SanaConfig {
    int num_layers;
    int num_attention_heads;
    int attention_head_dim;
    int num_cross_attention_heads;
    double expand_ratio;
    std::vector<int> pag_layers;
    bool use_fp4;
};

class SanaModel : public Module {
public:
    SanaModel(SanaConfig config, Tensor::ScalarType dtype, Device device);
    Tensor forward(Tensor hidden_states,
                   Tensor encoder_hidden_states,
                   Tensor timestep,
                   Tensor cu_seqlens_img,
                   Tensor cu_seqlens_txt,
                   int H,
                   int W,
                   bool pag,
                   bool cfg,
                   bool skip_first_layer);

public:
    const SanaConfig config;

public:
    std::vector<std::unique_ptr<SanaLinearTransformerBlock>> transformer_blocks;
};
