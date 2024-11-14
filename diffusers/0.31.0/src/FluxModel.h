#pragma once

#include "common.h"
#include "Tensor.h"
#include "Module.h"
#include "Linear.h"
#include "layernorm.h"

class AdaLayerNormZeroSingle : public Module {
public:
    static constexpr bool USE_4BIT = true;
    using GEMM = std::conditional_t<USE_4BIT, GEMV_AWQ, GEMM_W8A8>;

    struct Output {
        Tensor x;
        Tensor gate_msa;
    };

public:
    AdaLayerNormZeroSingle(int dim, Tensor::ScalarType dtype, Device device);
    Output forward(Tensor x, Tensor emb);

public:
    const int dim;

private:
    GEMM linear;
    LayerNorm norm;
};

class AdaLayerNormZero : public Module {
public:
    static constexpr bool USE_4BIT = true;
    using GEMM = std::conditional_t<USE_4BIT, GEMV_AWQ, GEMM_W8A8>;

    struct Output {
        Tensor x;
        Tensor gate_msa;
        Tensor shift_mlp;
        Tensor scale_mlp;
        Tensor gate_mlp;
    };
public:
    AdaLayerNormZero(int dim, bool pre_only, Tensor::ScalarType dtype, Device device);
    Output forward(Tensor x, Tensor emb);

public:
    const int dim;
    const bool pre_only;

private:
    GEMM linear;
    LayerNorm norm;
};

class Attention : public Module {
public:
    static constexpr int POOL_SIZE = 128;
    
    Attention(int num_heads, int dim_head, Device device);
    Tensor forward(Tensor qkv, Tensor pool_qkv, float sparsityRatio);

    static void setForceFP16(Module *module, bool value);

public:
    const int num_heads;
    const int dim_head;
    bool force_fp16;

private:
    Tensor cu_seqlens_cpu;
    Tensor headmask_type;
};

class FluxSingleTransformerBlock : public Module {
public:
    static constexpr bool USE_4BIT = true;
    using GEMM = std::conditional_t<USE_4BIT, GEMM_W4A4, GEMM_W8A8>;

    FluxSingleTransformerBlock(int dim, int num_attention_heads, int attention_head_dim, int mlp_ratio, Tensor::ScalarType dtype, Device device);
    Tensor forward(Tensor hidden_states, Tensor temb, Tensor rotary_emb);

public:
    const int dim;
    const int dim_head;
    const int num_heads;
    const int mlp_hidden_dim;

private:
    AdaLayerNormZeroSingle norm;
    GEMM mlp_fc1;
    GEMM mlp_fc2;
    GEMM qkv_proj;
    RMSNorm norm_q, norm_k;
    Attention attn;
    GEMM out_proj;
};

class JointTransformerBlock : public Module {
public:
    static constexpr bool USE_4BIT = true;
    using GEMM = std::conditional_t<USE_4BIT, GEMM_W4A4, GEMM_W8A8>;

    JointTransformerBlock(int dim, int num_attention_heads, int attention_head_dim, bool context_pre_only, Tensor::ScalarType dtype, Device device);
    std::tuple<Tensor, Tensor> forward(Tensor hidden_states, Tensor encoder_hidden_states, Tensor temb, Tensor rotary_emb, Tensor rotary_emb_context, float sparsityRatio);

public:
    const int dim;
    const int dim_head;
    const int num_heads;
    const bool context_pre_only;

private:
    AdaLayerNormZero norm1;
    AdaLayerNormZero norm1_context;
    GEMM qkv_proj;
    GEMM qkv_proj_context;
    RMSNorm norm_q, norm_k;
    RMSNorm norm_added_q, norm_added_k;
    Attention attn;
    GEMM out_proj;
    GEMM out_proj_context;
    LayerNorm norm2;
    LayerNorm norm2_context;
    GEMM mlp_fc1, mlp_fc2;
    GEMM mlp_context_fc1, mlp_context_fc2;
};

class FluxModel : public Module {
public:
    FluxModel(Tensor::ScalarType dtype, Device device);
    
    Tensor forward(
        Tensor hidden_states,
        Tensor encoder_hidden_states,
        Tensor temb,
        Tensor rotary_emb_img,
        Tensor rotary_emb_context,
        Tensor rotary_emb_single,
        float guidance_scale = 0.0f
    );

    void setGuidanceScale(float scale);
    float getGuidanceScale() const;

public:
    std::vector<std::unique_ptr<JointTransformerBlock>> transformer_blocks;
    std::vector<std::unique_ptr<FluxSingleTransformerBlock>> single_transformer_blocks;
    
private:
    float guidance_scale_;
};

// 添加辅助函数声明
std::tuple<Tensor, Tensor> get_1d_rotary_pos_embed(
    int dim,
    Tensor pos,
    bool repeat_interleave_real = true,
    bool use_real = true,
    Tensor::ScalarType freqs_dtype = Tensor::FP32
);

class FluxPosEmbed : public Module {
public:
    FluxPosEmbed(int theta, const std::vector<int>& axes_dim);
    
    // 返回 cos 和 sin 的元组
    std::tuple<Tensor, Tensor> forward(Tensor ids);

private:
    int theta_;
    std::vector<int> axes_dim_;
};