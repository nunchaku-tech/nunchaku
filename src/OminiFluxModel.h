#pragma once

#include "common.h"
#include "Tensor.h"
#include "Module.h"
#include "Linear.h"
#include "layernorm.h"
#include <pybind11/functional.h>
namespace pybind11 {
class function;
}

// Enum to specify the attention implementation to be used.
enum class OminiAttentionImpl {
    FlashAttention2 = 0, // Uses FlashAttention-2 kernel
    NunchakuFP16,        // Uses Nunchaku's custom FP16 attention kernel
};

// AdaLayerNormZero for single conditioning input.
// Applies layer normalization and then scales and shifts the output
// based on a learned linear transformation of an embedding vector.
// Also produces a gate value.
class OminiAdaLayerNormZeroSingle : public Module {
public:
    static constexpr bool USE_4BIT = true; // Flag to indicate if 4-bit quantization is used for GEMM
    using GEMM                     = std::conditional_t<USE_4BIT, GEMV_AWQ, GEMM_W8A8>;

    // Output structure for the forward pass
    struct Output {
        Tensor x;        // The normalized and conditioned tensor
        Tensor gate_msa; // Gate tensor for Multi-Head Self-Attention
    };

public:
    OminiAdaLayerNormZeroSingle(int dim, Tensor::ScalarType dtype, Device device);
    Output forward(Tensor x, Tensor emb);

public:
    const int dim; // Dimensionality of the input and output tensors

private:
    GEMM linear;    // Linear layer to process the embedding
    LayerNorm norm; // Layer normalization module
};

// AdaLayerNormZero for potentially multiple conditioning inputs (MSA and MLP).
// Applies layer normalization and then scales and shifts the output
// based on a learned linear transformation of an embedding vector.
// Produces gate, shift, and scale values for both MSA and MLP parts of a transformer block.
// If `pre_only` is true, only computes parameters for MSA (pre-attention conditioning).
class OminiAdaLayerNormZero : public Module {
public:
    static constexpr bool USE_4BIT = true; // Flag to indicate if 4-bit quantization is used for GEMM
    using GEMM                     = std::conditional_t<USE_4BIT, GEMV_AWQ, GEMM_W8A8>;

    // Output structure for the forward pass
    struct Output {
        Tensor x;         // The normalized and conditioned tensor
        Tensor gate_msa;  // Gate tensor for Multi-Head Self-Attention
        Tensor shift_mlp; // Shift tensor for the MLP block
        Tensor scale_mlp; // Scale tensor for the MLP block
        Tensor gate_mlp;  // Gate tensor for the MLP block
    };

public:
    OminiAdaLayerNormZero(int dim, bool pre_only, Tensor::ScalarType dtype, Device device);
    Output forward(Tensor x, Tensor emb);

public:
    const int dim;      // Dimensionality of the input and output tensors
    const bool pre_only; // Flag to indicate if only pre-attention conditioning parameters are computed

private:
    GEMM linear;    // Linear layer to process the embedding
    LayerNorm norm; // Layer normalization module
};

// Multi-Head Attention module for OminiFlux models.
// Supports both standard dense attention and block-sparse attention.
class OminiAttention : public Module {
public:
    static constexpr int POOL_SIZE = 128; // Block size for pooling in sparse attention

    OminiAttention(int num_heads, int dim_head, Device device);
    // Standard dense attention forward pass
    Tensor forward(Tensor qkv);
    // Block-sparse attention forward pass with a given sparsity ratio
    Tensor forward(Tensor qkv, Tensor pool_qkv, float sparsityRatio);

    // Utility to recursively set the force_fp16 flag on OminiAttention modules
    static void setForceFP16(Module *module, bool value);

public:
    const int num_heads; // Number of attention heads
    const int dim_head;  // Dimension of each attention head
    bool force_fp16;     // Flag to force FP16 computation for attention

private:
    Tensor cu_seqlens_cpu; // CPU tensor for cumulative sequence lengths (used in variable-length attention)
    Tensor headmask_type;  // Tensor defining the type/group of each head (used in some attention variants)
};

// A single transformer block tailored for the OminiFlux architecture.
// This block is typically used for processing modalities separately after an initial joint phase.
// It includes AdaLayerNorm, QKV projection, multi-head attention, output projection, and an MLP.
class OminiFluxSingleTransformerBlock : public Module {
public:
    static constexpr bool USE_4BIT = true; // Flag to indicate if 4-bit quantization is used for GEMM
    using GEMM                     = std::conditional_t<USE_4BIT, GEMM_W4A4, GEMM_W8A8>;

    OminiFluxSingleTransformerBlock(int dim,
                               int num_attention_heads,
                               int attention_head_dim,
                               int mlp_ratio, // Ratio to determine MLP hidden dimension (dim * mlp_ratio)
                               bool use_fp4, // Whether to use 4-bit quantization for GEMM layers
                               Tensor::ScalarType dtype, // Data type for computations
                               Device device);          // Device for tensor allocation

   // Forward pass for the single transformer block
   // Takes hidden_states, conditional_hidden_states, time embeddings (temb, cond_temb),
   // and rotary embeddings (rotary_emb, cond_rotary_emb).
   // Returns a tuple of processed hidden_states and cond_hidden_states.
   std::tuple<Tensor, Tensor> forward(Tensor hidden_states, Tensor cond_hidden_states, Tensor temb, Tensor cond_temb, Tensor rotary_emb, Tensor cond_rotary_emb);

public:
    const int dim;            // Input/output dimension of the block
    const int dim_head;       // Dimension of each attention head
    const int num_heads;      // Number of attention heads
    const int mlp_hidden_dim; // Hidden dimension of the MLP

    OminiAttentionImpl attnImpl = OminiAttentionImpl::FlashAttention2; // Attention implementation to use

private:
    OminiAdaLayerNormZeroSingle norm; // AdaLayerNorm for this block
    GEMM mlp_fc1;                   // First linear layer of the MLP
    GEMM mlp_fc2;                   // Second linear layer of the MLP
    GEMM qkv_proj;                 // Linear layer for Q, K, V projection
    RMSNorm norm_q, norm_k;         // RMSNorm for Q and K before attention
    OminiAttention attn;            // Multi-head attention module
    GEMM out_proj;                  // Output projection layer after attention
};

// A joint transformer block for the OminiFlux architecture.
// This block processes multiple input modalities (e.g., image, text, conditioning) together.
// It includes AdaLayerNorm for image and context, QKV projections for image and context,
// multi-head attention, output projections, and MLP layers for image and context.
class OminiJointTransformerBlock : public Module {
public:
    static constexpr bool USE_4BIT = true; // Flag to indicate if 4-bit quantization is used for GEMM
    using GEMM                     = std::conditional_t<USE_4BIT, GEMM_W4A4, GEMM_W8A8>;

    OminiJointTransformerBlock(int dim,
                          int num_attention_heads,
                          int attention_head_dim,
                          bool context_pre_only, // If true, context conditioning is applied only before attention
                          bool use_fp4,          // Whether to use 4-bit quantization for GEMM layers
                          Tensor::ScalarType dtype,      // Data type for computations
                          Device device);               // Device for tensor allocation

    // Forward pass for the joint transformer block
    // Takes hidden_states (image), cond_hidden_states (conditional), encoder_hidden_states (text/context),
    // time embeddings (temb, cond_temb), rotary embeddings for image, context, and conditional inputs,
    // and an optional sparsity ratio for attention.
    // Returns a tuple of processed hidden_states, cond_hidden_states, and encoder_hidden_states.
    std::tuple<Tensor, Tensor, Tensor> forward(Tensor hidden_states,
                                       Tensor cond_hidden_states,
                                       Tensor encoder_hidden_states,
                                       Tensor temb,
                                       Tensor cond_temb,
                                       Tensor rotary_emb,         // Rotary embedding for image
                                       Tensor rotary_emb_context, // Rotary embedding for context/text
                                       Tensor cond_rotary_emb,    // Rotary embedding for conditional input
                                       float sparsityRatio);

public:
    const int dim;              // Input/output dimension of the block
    const int dim_head;         // Dimension of each attention head
    const int num_heads;        // Number of attention heads
    const bool context_pre_only; // True if context conditioning is only pre-attention
    OminiAdaLayerNormZero norm1; // AdaLayerNorm for image features

    OminiAttentionImpl attnImpl = OminiAttentionImpl::FlashAttention2; // Attention implementation to use

private:
    OminiAdaLayerNormZero norm1_context; // AdaLayerNorm for context/text features
    GEMM qkv_proj;                     // QKV projection for image features
    GEMM qkv_proj_context;            // QKV projection for context/text features
    RMSNorm norm_q, norm_k;            // RMSNorm for image Q and K
    RMSNorm norm_added_q, norm_added_k; // RMSNorm for context Q and K (often termed 'added' or cross-attention QK)
    OminiAttention attn;               // Multi-head attention module
    GEMM out_proj;                     // Output projection for image features
    GEMM out_proj_context;             // Output projection for context/text features
    LayerNorm norm2;                   // LayerNorm before MLP for image features
    LayerNorm norm2_context;           // LayerNorm before MLP for context/text features
    GEMM mlp_fc1, mlp_fc2;             // MLP layers for image features
    GEMM mlp_context_fc1, mlp_context_fc2; // MLP layers for context/text features
};

// The main OminiFlux model class.
// Comprises a sequence of OminiJointTransformerBlocks followed by OminiFluxSingleTransformerBlocks.
// Supports features like 4-bit quantization (`use_fp4`) and layer offloading (`offload`).
class OminiFluxModel : public Module {
public:
    OminiFluxModel(bool use_fp4, bool offload, Tensor::ScalarType dtype, Device device);

    // Main forward pass for the entire model.
    // Processes hidden_states, cond_hidden_states, and encoder_hidden_states through all blocks.
    // `temb`, `cond_temb`: Time embeddings.
    // `rotary_emb_*`: Various rotary position embeddings.
    // `controlnet_*_samples`: Optional inputs from ControlNet.
    // `skip_first_layer`: If true, the first joint transformer block is skipped.
    Tensor forward(Tensor hidden_states,
                   Tensor cond_hidden_states,
                   Tensor encoder_hidden_states,
                   Tensor temb,
                   Tensor cond_temb,
                   Tensor rotary_emb_img,     // Rotary embeddings for image features
                   Tensor rotary_emb_context, // Rotary embeddings for context/text features
                   Tensor rotary_emb_single,  // Rotary embeddings for single (concatenated) features
                   Tensor rotary_emb_cond,    // Rotary embeddings for conditional features
                   Tensor controlnet_block_samples,        // ControlNet features for joint blocks
                   Tensor controlnet_single_block_samples, // ControlNet features for single blocks
                   bool skip_first_layer = false);

    // Forward pass for a specific layer `idx`.
    // Allows for inspecting or modifying intermediate representations.
    // Returns a tuple of (image_hidden_states, text_encoder_hidden_states, conditional_hidden_states).
    std::tuple<Tensor, Tensor, Tensor> forward_layer(size_t layer,
                                             Tensor hidden_states,
                                             Tensor cond_hidden_states,
                                             Tensor encoder_hidden_states,
                                             Tensor temb,
                                             Tensor cond_temb,
                                             Tensor rotary_emb_img,
                                             Tensor rotary_emb_context,
                                             Tensor rotary_emb_cond,
                                             Tensor controlnet_block_samples,
                                             Tensor controlnet_single_block_samples);

    // Sets the attention implementation (e.g., FlashAttention2) for all attention modules.
    void setAttentionImpl(OminiAttentionImpl impl);

    // Sets a Python callback function to be invoked at specific points, often for residual injection.
    void set_residual_callback(std::function<Tensor(const Tensor &)> cb);

public:
    const Tensor::ScalarType dtype; // Data type used throughout the model

    // Storage for the transformer blocks
    std::vector<std::unique_ptr<OminiJointTransformerBlock>> transformer_blocks;
    std::vector<std::unique_ptr<OminiFluxSingleTransformerBlock>> single_transformer_blocks;

    std::function<Tensor(const Tensor &)> residual_callback; // Stores the Python callback

private:
    bool offload; // Flag indicating if layer offloading is enabled
};
