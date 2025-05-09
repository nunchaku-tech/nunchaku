#include "OminiFluxModel.h"
#include "kernels/misc_kernels.h"
#include "kernels/gemm_batched.h"
#include "kernels/zgemm/zgemm.h"
#include "flash_api.h"
#include "activation.h"
#include <nvtx3/nvToolsExt.h>

#include <pybind11/functional.h>

#include <iostream>

using spdlog::fmt_lib::format;
using namespace nunchaku;

// Performs a forward pass through two fully connected layers (fc1 and fc2)
// with a GELU activation function and quantization applied.
Tensor omini_forward_mlp(GEMM_W4A4 &fc1, GEMM_W4A4 &fc2, Tensor norm_hidden_states) {
    Tensor ff_output = fc2.forward_quant(std::get<GEMM_W4A4::QuantizedActivation>(
        fc1.forward(norm_hidden_states, GEMM_W4A4::FuseOptions::GELU_QUANT, &fc2)));
    return ff_output;
}

// Tensor omini_forward_mlp(GEMM_W8A8 &fc1, GEMM_W8A8 &fc2, Tensor norm_hidden_states) {
//     Tensor ff_output = fc2.forward(fc1.forward(norm_hidden_states), GEMM_W8A8::FuseOptions::GELU);
//     return ff_output;
// }

// Performs a forward pass through a single fully connected layer.
Tensor omini_forward_fc(GEMM_W4A4 &fc, Tensor x) {
    return fc.forward(x);
    // return std::get<Tensor>(fc.forward(x));
}

// Tensor omini_forward_fc(GEMM_W8A8 &fc, Tensor x) {
//     return fc.forward(x);
// }

// Implements an AdaLayerNormZero module with a single set of shift, scale, and gate parameters.
// This is typically used in transformer blocks where conditioning is applied once.
OminiAdaLayerNormZeroSingle::OminiAdaLayerNormZeroSingle(int dim, Tensor::ScalarType dtype, Device device)
    : dim(dim), linear(dim, 3 * dim, true, dtype, device), norm(dim, 1e-6, false, dtype, device) {
    registerChildren(linear, "linear")(norm, "norm");
}

// Forward pass for OminiAdaLayerNormZeroSingle.
// Applies layer normalization to the input tensor 'x' and then modulates it
// using shift and scale parameters derived from the embedding tensor 'emb'.
// It also computes a gate parameter from 'emb'.
OminiAdaLayerNormZeroSingle::Output OminiAdaLayerNormZeroSingle::forward(Tensor x, Tensor emb) {
    debug("emb_input", emb);
    emb = linear.forward(Silu::forward(emb));
    debug("emb_linear", emb);
    auto &&[shift_msa, scale_msa, gate_msa] = kernels::split_mod<3>(emb);
    debug("scale_msa", scale_msa);
    debug("shift_msa", shift_msa);

    debug("x", x);
    Tensor norm_x = norm.forward(x);
    debug("norm_x", norm_x);

    // kernels::mul_add(norm_x, scale_msa, shift_msa);
    kernels::mul_add_batch(norm_x, scale_msa, true, 0.0, shift_msa, true);
    return Output{norm_x, gate_msa};
}

// Implements an AdaLayerNormZero module.
// If `pre_only` is true, it computes only shift and scale for pre-attention conditioning.
// Otherwise, it computes shift, scale, and gate for both MSA (Multi-Head Self-Attention)
// and MLP (Multi-Layer Perceptron) conditioning.
OminiAdaLayerNormZero::OminiAdaLayerNormZero(int dim, bool pre_only, Tensor::ScalarType dtype, Device device)
    : dim(dim), pre_only(pre_only), linear(dim, pre_only ? 2 * dim : 6 * dim, true, dtype, device),
      norm(dim, 1e-6, false, dtype, device) {
    registerChildren(linear, "linear")(norm, "norm");
}

// Forward pass for OminiAdaLayerNormZero.
// Applies layer normalization to the input tensor 'x'.
// Derives conditioning parameters (shift, scale, gate) from the embedding tensor 'emb'.
// If `pre_only` is true, only MSA shift and scale are computed and applied.
// Otherwise, MSA and MLP shift, scale, and gate parameters are computed and applied.
OminiAdaLayerNormZero::Output OminiAdaLayerNormZero::forward(Tensor x, Tensor emb) {
    debug("x", x);

    debug("emb_input", emb);
    emb = linear.forward(Silu::forward(emb));
    debug("emb_linear", emb);

    if (pre_only) {
        auto &&[shift_msa, scale_msa] = kernels::split_mod<2>(emb);
        debug("shift_msa", shift_msa);

        Tensor norm_x = norm.forward(x);
        debug("norm_x", norm_x);

        // kernels::mul_add(norm_x, scale_msa, shift_msa);
        kernels::mul_add_batch(norm_x, scale_msa, true, 0.0, shift_msa, true);
        debug("norm_x_scaled", norm_x);

        return Output{norm_x};
    } else {
        auto &&[shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp] = kernels::split_mod<6>(emb);
        debug("shift_msa", shift_msa);

        Tensor norm_x = norm.forward(x);
        debug("norm_x", norm_x);

        // kernels::mul_add(norm_x, scale_msa, shift_msa);
        kernels::mul_add_batch(norm_x, scale_msa, true, 0.0, shift_msa, true);
        debug("norm_x_scaled", norm_x);

        return Output{norm_x, gate_msa, shift_mlp, scale_mlp, gate_mlp};
    }
}

// Implements a multi-head attention mechanism.
// It can operate in a standard dense mode or a block-sparse mode.
OminiAttention::OminiAttention(int num_heads, int dim_head, Device device)
    : num_heads(num_heads), dim_head(dim_head), force_fp16(false) {
    headmask_type = Tensor::allocate({num_heads}, Tensor::INT32, Device::cpu());
    for (int i = 0; i < num_heads; i++) {
        headmask_type.data_ptr<int32_t>()[i] = i + 1;
    }
    headmask_type = headmask_type.copy(device);
}

// Forward pass for OminiAttention in standard dense mode.
// Takes a QKV tensor, reshapes it, and performs multi-head attention.
Tensor OminiAttention::forward(Tensor qkv) {
    assert(qkv.ndims() == 3);

    const Device device  = qkv.device();
    const int batch_size = qkv.shape[0];
    const int num_tokens = qkv.shape[1];
    assert(qkv.shape[2] == num_heads * dim_head * 3);

    Tensor reshaped = qkv.view({batch_size, num_tokens, num_heads * 3, dim_head});
    Tensor q        = reshaped.slice(2, 0, num_heads);
    Tensor k        = reshaped.slice(2, num_heads, num_heads * 2);
    Tensor v        = reshaped.slice(2, num_heads * 2, num_heads * 3);

    Tensor raw_attn_output = mha_fwd(q, k, v, 0.0f, pow(q.shape[-1], (-0.5)), false, -1, -1, false).front();

    assert(raw_attn_output.shape[0] == batch_size);
    assert(raw_attn_output.shape[1] == num_tokens);
    assert(raw_attn_output.shape[2] == num_heads);
    assert(raw_attn_output.shape[3] == dim_head);

    return raw_attn_output.view({batch_size * num_tokens, num_heads, dim_head});
}

// Forward pass for OminiAttention in block-sparse mode.
// Takes QKV and pooled QKV tensors, computes block scores, and performs sparse multi-head attention.
// `sparsityRatio` controls the fraction of blocks to prune.
Tensor OminiAttention::forward(Tensor qkv, Tensor pool_qkv, float sparsityRatio) {
    const bool cast_fp16 = this->force_fp16 && qkv.scalar_type() != Tensor::FP16;

    assert(qkv.ndims() == 3);

    const Device device  = qkv.device();
    const int batch_size = qkv.shape[0];
    const int num_tokens = qkv.shape[1];
    assert(qkv.shape[2] == num_heads * dim_head * 3);

    constexpr int POOL_SIZE = 128;
    const int pool_tokens   = ceilDiv(num_tokens, POOL_SIZE);

    Tensor blockmask;

    if (pool_qkv.valid()) {
        assert(pool_qkv.shape[0] == batch_size);
        assert(pool_qkv.shape[1] == pool_tokens);
        assert(pool_qkv.shape[2] == num_heads * dim_head * 3);
    }

    Tensor pool_score = Tensor::allocate({batch_size, num_heads, pool_tokens, pool_tokens}, Tensor::FP32, device);

    if (pool_qkv.valid() && sparsityRatio > 0) {
        pool_qkv = pool_qkv.view({batch_size, pool_tokens, 3, num_heads, dim_head});
        pool_qkv = pool_qkv.transpose(1, 2).transpose(2, 3); // [batch_size, 3, num_heads, poolTokens, dim_head]
        for (int i = 0; i < batch_size; i++) {
            Tensor pool_q = pool_qkv.slice(0, i, i + 1).slice(1, 0, 1);
            Tensor pool_k = pool_qkv.slice(0, i, i + 1).slice(1, 1, 2);
            Tensor pool_s = pool_score.slice(0, i, i + 1);
            gemm_batched_fp16(pool_q, pool_k, pool_s);
        }
    }

    blockmask = kernels::topk(pool_score, pool_tokens * (1 - sparsityRatio));

    if (cu_seqlens_cpu.valid()) {
        if (cu_seqlens_cpu.shape[0] != batch_size + 1) {
            cu_seqlens_cpu = Tensor{};
        } else {
            for (int i = 0; i <= batch_size; i++) {
                if (cu_seqlens_cpu.data_ptr<int32_t>()[i] != num_tokens * i) {
                    cu_seqlens_cpu = Tensor{};
                    break;
                }
            }
        }
    }
    if (!cu_seqlens_cpu.valid()) {
        cu_seqlens_cpu                        = Tensor::allocate({batch_size + 1}, Tensor::INT32, Device::cpu());
        cu_seqlens_cpu.data_ptr<int32_t>()[0] = 0;
        for (int i = 1; i <= batch_size; i++) {
            cu_seqlens_cpu.data_ptr<int32_t>()[i] = cu_seqlens_cpu.data_ptr<int32_t>()[i - 1] + num_tokens;
        }
    }

    if (cast_fp16) {
        Tensor tmp = Tensor::empty(qkv.shape.dataExtent, Tensor::FP16, qkv.device());
        kernels::cast(qkv, tmp);
        qkv = tmp;
    }

    debug("qkv", qkv);

    Tensor cu_seqlens = cu_seqlens_cpu.copy(device);

    Tensor reshaped = qkv.view({batch_size * num_tokens, num_heads * 3, dim_head});
    Tensor q        = reshaped.slice(1, 0, num_heads);
    Tensor k        = reshaped.slice(1, num_heads, num_heads * 2);
    Tensor v        = reshaped.slice(1, num_heads * 2, num_heads * 3);

    spdlog::debug("q,k,v={}", q.shape.str());

    Tensor raw_attn_output = mha_fwd_block(q,
                                           k,
                                           v,
                                           cu_seqlens,
                                           cu_seqlens,
                                           POOL_SIZE,
                                           POOL_SIZE,
                                           headmask_type,
                                           {},
                                           blockmask,
                                           num_tokens,
                                           num_tokens,
                                           0.0f,
                                           pow(q.shape[-1], (-0.5)),
                                           false,
                                           false,
                                           false,
                                           -1,
                                           -1)
                                 .front();

    debug("raw_attn_output", raw_attn_output);

    if (cast_fp16) {
        Tensor tmp = Tensor::empty(raw_attn_output.shape.dataExtent, Tensor::BF16, raw_attn_output.device());
        kernels::cast(raw_attn_output, tmp);
        raw_attn_output = tmp;
    }

    /**
    Tensor raw_attn_output = mha_varlen_fwd(q, k, v,
        cu_seqlens,
        cu_seqlens,
        concat.shape[1],
        concat.shape[1],
        0.0f,
        pow(q.shape[-1], (-0.5)),
        false,
        true,
        -1, -1,
        false
    ).front();

    Tensor raw_attn_output = mha_fwd(q, k, v,
        0.0f,
        pow(q.shape[-1], (-0.5)),
        false, -1, -1, false
    ).front();

    Tensor raw_attn_output = mha_varlen_fwd(
        q, k, v,
        cu_seqlens, cu_seqlens,
        num_tokens_img + num_tokens_txt, num_tokens_img + num_tokens_txt,
        0.0f,
        pow(q.shape[-1], (-0.5)),
        false, false, -1, -1, false
    ).front();
    **/

    assert(raw_attn_output.shape[0] == batch_size * num_tokens);
    assert(raw_attn_output.shape[1] == num_heads);
    assert(raw_attn_output.shape[2] == dim_head);

    return raw_attn_output;
}

// Static method to enable or disable forcing FP16 precision for attention operations
// within a given module and its children.
void OminiAttention::setForceFP16(Module *module, bool value) {
    spdlog::info("{} force fp16 attention", value ? "Enable" : "Disable");

    module->traverse([&](Module *m) {
        if (OminiAttention *attn = dynamic_cast<OminiAttention *>(m)) {
            attn->force_fp16 = value;
        }
    });
}

// Implements a single transformer block for the OminiFlux model.
// This block is used when processing image and conditional tokens separately
// after the initial joint processing phase.
OminiFluxSingleTransformerBlock::OminiFluxSingleTransformerBlock(int dim,
                                                       int num_attention_heads,
                                                       int attention_head_dim,
                                                       int mlp_ratio,
                                                       bool use_fp4,
                                                       Tensor::ScalarType dtype,
                                                       Device device)
    : dim(dim), dim_head(attention_head_dim / num_attention_heads), num_heads(num_attention_heads),
      mlp_hidden_dim(dim * mlp_ratio), norm(dim, dtype, device),
      mlp_fc1(dim, mlp_hidden_dim, true, use_fp4, dtype, device),
      mlp_fc2(mlp_hidden_dim, dim, true, use_fp4, dtype, device), qkv_proj(dim, dim * 3, true, use_fp4, dtype, device),
      norm_q(dim_head, 1e-6, false, dtype, device), norm_k(dim_head, 1e-6, false, dtype, device),
      attn(num_attention_heads, attention_head_dim / num_attention_heads, device),
      out_proj(dim, dim, true, use_fp4, dtype, device) {
    registerChildren(norm, "norm")(mlp_fc1, "mlp_fc1")(mlp_fc2, "mlp_fc2")(qkv_proj, "qkv_proj")(norm_q, "norm_q")(
        norm_k, "norm_k")(attn, "attn")(out_proj, "out_proj");
}

// Forward pass for the OminiFluxSingleTransformerBlock.
// Processes `hidden_states` and `cond_hidden_states` (conditional information) separately.
// Applies AdaLayerNorm, self-attention, and MLP layers to both.
// `temb` and `cond_temb` are time embeddings for conditioning.
// `rotary_emb` and `cond_rotary_emb` are rotary position embeddings.
std::tuple<Tensor, Tensor> OminiFluxSingleTransformerBlock::forward(Tensor hidden_states, 
                                           Tensor cond_hidden_states, 
                                           Tensor temb, 
                                           Tensor cond_temb,
                                           Tensor rotary_emb,
                                           Tensor cond_rotary_emb) {

    nvtxRangePushA("OminiFluxSingleTransformerBlock");

    const int batch_size = hidden_states.shape[0];
    const int num_tokens = hidden_states.shape[1];
    const int num_cond_tokens = cond_hidden_states.shape[1];

    auto &&[norm_hidden_states, gate] = this->norm.forward(hidden_states, temb);
    auto &&[norm_cond_hidden_states, cond_gate] = this->norm.forward(cond_hidden_states, cond_temb);
    debug("norm_hidden_states", norm_hidden_states);
    debug("gate", gate);
    debug("norm_cond_hidden_states", norm_cond_hidden_states);
    debug("cond_gate", cond_gate);

    Tensor residual = hidden_states;
    Tensor residual_cond = cond_hidden_states;

    Tensor attn_output;


    debug("rotary_emb", rotary_emb);

    if (attnImpl == OminiAttentionImpl::FlashAttention2) {
        Tensor concat_qkv = Tensor::allocate(
            {batch_size, num_tokens + num_cond_tokens, dim * 3}, 
            norm_hidden_states.scalar_type(), 
            norm_hidden_states.device());

        for (int i = 0; i < batch_size; i++) {
            // Process image tokens
            Tensor qkv = concat_qkv.slice(0, i, i + 1).slice(1, 0, num_tokens);
            // Process condition tokens
            Tensor qkv_cond = concat_qkv.slice(0, i, i + 1).slice(1, num_tokens, num_tokens + num_cond_tokens);

            qkv_proj.forward(
                norm_hidden_states.slice(0, i, i + 1),
                qkv,
                {},
                norm_q.weight,
                norm_k.weight,
                rotary_emb);


            qkv_proj.forward(
                norm_cond_hidden_states.slice(0, i, i + 1),
                qkv_cond,
                {},
                norm_q.weight,
                norm_k.weight,
                cond_rotary_emb);
        }

        attn_output = attn.forward(concat_qkv);
        attn_output = attn_output.reshape({batch_size, num_tokens + num_cond_tokens, num_heads * dim_head});

        // Split outputs
        if (batch_size == 1) {
            hidden_states = attn_output.slice(1, 0, num_tokens);
            cond_hidden_states = attn_output.slice(1, num_tokens, num_tokens + num_cond_tokens);
        } else {
            // Allocate output tensors
            hidden_states = Tensor::allocate({batch_size, num_tokens, num_heads * dim_head}, attn_output.scalar_type(), attn_output.device());
            cond_hidden_states = Tensor::allocate({batch_size, num_cond_tokens, num_heads * dim_head}, attn_output.scalar_type(), attn_output.device());
            // Copy for each batch
            checkCUDA(cudaMemcpy2DAsync(
                hidden_states.data_ptr(),
                num_tokens * num_heads * dim_head * hidden_states.scalar_size(),
                attn_output.data_ptr(),
                (num_tokens + num_cond_tokens) * num_heads * dim_head * attn_output.scalar_size(),
                num_tokens * num_heads * dim_head * hidden_states.scalar_size(),
                batch_size,
                cudaMemcpyDeviceToDevice,
                getCurrentCUDAStream()
            ));
            checkCUDA(cudaMemcpy2DAsync(
                cond_hidden_states.data_ptr(),
                num_cond_tokens * num_heads * dim_head * cond_hidden_states.scalar_size(),
                attn_output.data_ptr<char>() + num_tokens * num_heads * dim_head * attn_output.scalar_size(),
                (num_tokens + num_cond_tokens) * num_heads * dim_head * attn_output.scalar_size(),
                num_cond_tokens * num_heads * dim_head * cond_hidden_states.scalar_size(),
                batch_size,
                cudaMemcpyDeviceToDevice,
                getCurrentCUDAStream()
            ));
        }
    } else if (attnImpl == OminiAttentionImpl::NunchakuFP16) {
        const int num_tokens_pad = ceilDiv((num_tokens), 256) * 256;
        const int num_tokens_cond_pad = ceilDiv(num_cond_tokens, 256) * 256;

        Tensor concat_q, concat_k, concat_v;
        {
            nvtxRangePushA("qkv_proj");
            concat_q = Tensor::allocate({batch_size, num_heads, num_tokens_pad + num_tokens_cond_pad, dim_head},
                            Tensor::FP16,
                            norm_hidden_states.device());
            concat_k = Tensor::empty_like(concat_q);
            concat_v = Tensor::empty_like(concat_q);

            for (int i = 0; i < batch_size; i++) {
                // Define slice functions
                auto sliceImg = [&](Tensor x) { return x.slice(0, i, i + 1).slice(2, 0, num_tokens_pad); };
                auto sliceCond = [&](Tensor x) { 
                    return x.slice(0, i, i + 1).slice(2, num_tokens_pad, num_tokens_pad + num_tokens_cond_pad); 
                };

                        // Process image tokens QKV
            qkv_proj.forward(
                norm_hidden_states.slice(0, i, i + 1),
                {},
                {},
                norm_q.weight,
                norm_k.weight,
                rotary_emb,
                sliceImg(concat_q),
                sliceImg(concat_k),
                sliceImg(concat_v),
                num_tokens);

            // Process condition tokens QKV
            qkv_proj.forward(
                norm_cond_hidden_states.slice(0, i, i + 1),
                {},
                {},
                norm_q.weight,
                norm_k.weight,
                cond_rotary_emb,
                sliceCond(concat_q),
                sliceCond(concat_k),
                sliceCond(concat_v),
                num_cond_tokens);

            }

            nvtxRangePop();

        }

        Tensor o = Tensor::allocate(
            {batch_size, num_tokens_pad + num_tokens_cond_pad, num_heads * dim_head},
            norm_hidden_states.scalar_type(),
            norm_hidden_states.device());

        nvtxRangePushA("OminiAttention");

        kernels::attention_fp16(concat_q, concat_k, concat_v, o, pow(dim_head, (-0.5)));

        nvtxRangePop();


        // Split outputs
        if (batch_size == 1 || num_tokens_pad == num_tokens) {
            hidden_states = o.slice(1, 0, num_tokens);
            cond_hidden_states = o.slice(1, num_tokens_pad, num_tokens_pad + num_cond_tokens);
        } else {
            // Allocate output tensors
            hidden_states = Tensor::allocate({batch_size, num_tokens, num_heads * dim_head}, o.scalar_type(), o.device());
            cond_hidden_states = Tensor::allocate({batch_size, num_cond_tokens, num_heads * dim_head}, o.scalar_type(), o.device());
            // Copy for each batch
            checkCUDA(cudaMemcpy2DAsync(
                hidden_states.data_ptr(),
                num_tokens * num_heads * dim_head * hidden_states.scalar_size(),
                o.data_ptr(),
                (num_tokens_pad + num_tokens_cond_pad) * num_heads * dim_head * o.scalar_size(),
                num_tokens * num_heads * dim_head * hidden_states.scalar_size(),
                batch_size,
                cudaMemcpyDeviceToDevice,
                getCurrentCUDAStream()
            ));
            checkCUDA(cudaMemcpy2DAsync(
                cond_hidden_states.data_ptr(),
                num_cond_tokens * num_heads * dim_head * cond_hidden_states.scalar_size(),
                o.data_ptr<char>() + num_tokens_pad * num_heads * dim_head * o.scalar_size(),
                (num_tokens_pad + num_tokens_cond_pad) * num_heads * dim_head * o.scalar_size(),
                num_cond_tokens * num_heads * dim_head * cond_hidden_states.scalar_size(),
                batch_size,
                cudaMemcpyDeviceToDevice,
                getCurrentCUDAStream()
            ));
        }
    } else {
        assert(false);
    }

    debug("raw_attn_output_img", hidden_states);

    hidden_states = omini_forward_fc(out_proj, hidden_states);
    debug("attn_output_img_processed", hidden_states);

    Tensor ff_output_img = omini_forward_mlp(mlp_fc1, mlp_fc2, norm_hidden_states);
    debug("ff_output_img", ff_output_img);

    // Combine the processed attention outputs
    hidden_states = kernels::add(hidden_states, ff_output_img);
    debug("attn_ff_output", hidden_states);

    // Apply gates
    kernels::mul_add_batch(hidden_states, gate, true, 0.0, residual, true);

    cond_hidden_states = omini_forward_fc(out_proj, cond_hidden_states);

    Tensor ff_output_cond = omini_forward_mlp(mlp_fc1, mlp_fc2, norm_cond_hidden_states);

    cond_hidden_states = kernels::add(cond_hidden_states, ff_output_cond);

    kernels::mul_add_batch(cond_hidden_states, cond_gate, true, 0.0, residual_cond, true);

    nvtxRangePop();

    return {hidden_states, cond_hidden_states};
}

// Implements a joint transformer block for the OminiFlux model.
// This block is used in the initial layers to jointly process image, conditional, and text (encoder) tokens.
OminiJointTransformerBlock::OminiJointTransformerBlock(int dim,
                                             int num_attention_heads,
                                             int attention_head_dim,
                                             bool context_pre_only,
                                             bool use_fp4,
                                             Tensor::ScalarType dtype,
                                             Device device)
    : dim(dim), dim_head(attention_head_dim / num_attention_heads), num_heads(num_attention_heads),
      context_pre_only(context_pre_only), norm1(dim, false, dtype, device),
      norm1_context(dim, context_pre_only, dtype, device), qkv_proj(dim, dim * 3, true, use_fp4, dtype, device),
      qkv_proj_context(dim, dim * 3, true, use_fp4, dtype, device), norm_q(dim_head, 1e-6, false, dtype, device),
      norm_k(dim_head, 1e-6, false, dtype, device), norm_added_q(dim_head, 1e-6, false, dtype, device),
      norm_added_k(dim_head, 1e-6, false, dtype, device),
      attn(num_attention_heads, attention_head_dim / num_attention_heads, device),
      out_proj(dim, dim, true, use_fp4, dtype, device), out_proj_context(dim, dim, true, use_fp4, dtype, device),
      norm2(dim, 1e-6, false, dtype, device), norm2_context(dim, 1e-6, false, dtype, device),
      mlp_fc1(dim, dim * 4, true, use_fp4, dtype, device), mlp_fc2(dim * 4, dim, true, use_fp4, dtype, device),
      mlp_context_fc1(dim, dim * 4, true, use_fp4, dtype, device),
      mlp_context_fc2(dim * 4, dim, true, use_fp4, dtype, device) {
    registerChildren(norm1, "norm1")(norm1_context, "norm1_context")(qkv_proj, "qkv_proj")(qkv_proj_context,
                                                                                           "qkv_proj_context")(
        norm_q, "norm_q")(norm_k, "norm_k")(norm_added_q, "norm_added_q")(norm_added_k, "norm_added_k")(attn, "attn")(
        out_proj, "out_proj")(out_proj_context, "out_proj_context")(norm2, "norm2")(norm2_context, "norm2_context")(
        mlp_fc1, "mlp_fc1")(mlp_fc2, "mlp_fc2")(mlp_context_fc1, "mlp_context_fc1")(mlp_context_fc2, "mlp_context_fc2");
}

// Forward pass for the OminiJointTransformerBlock.
// Processes `hidden_states` (image), `cond_hidden_states` (conditional), and `encoder_hidden_states` (text) jointly.
// Applies AdaLayerNorm, performs cross-attention between these states, and then applies MLP layers.
// `temb` and `cond_temb` are time embeddings.
// `rotary_emb*` are rotary position embeddings for different modalities.
// `sparsityRatio` controls block sparsity if FlashAttention2 with sparsity is used.
// hidden_states: [Batch, Width * Height, dim]
// encoder_hidden_states: [Batch, Token, dim]
std::tuple<Tensor, Tensor, Tensor> OminiJointTransformerBlock::forward(Tensor hidden_states,
                                                          Tensor cond_hidden_states,
                                                          Tensor encoder_hidden_states,
                                                          Tensor temb,
                                                          Tensor cond_temb,
                                                          Tensor rotary_emb,
                                                          Tensor rotary_emb_context,
                                                          Tensor cond_rotary_emb,
                                                          float sparsityRatio) {
    int batch_size = hidden_states.shape[0];
    assert(encoder_hidden_states.shape[0] == batch_size);

    nvtxRangePushA("OminiJointTransformerBlock");

    nvtxRangePushA("OminiAdaNorm");

    int num_tokens_img = hidden_states.shape[1];
    int num_tokens_txt = encoder_hidden_states.shape[1];
    int num_tokens_cond = cond_hidden_states.shape[1];

    assert(hidden_states.shape[2] == dim);
    assert(encoder_hidden_states.shape[2] == dim);
    assert(cond_hidden_states.shape[2] == dim);

    spdlog::debug("hidden_states={} encoder_hidden_states={} temb={}",
                  hidden_states.shape.str(),
                  encoder_hidden_states.shape.str(),
                  temb.shape.str());
    spdlog::debug("batch_size={} num_tokens_img={} num_tokens_txt={}", batch_size, num_tokens_img, num_tokens_txt);

    auto norm1_output         = norm1.forward(hidden_states, temb);
    auto norm1_cond_output    = norm1.forward(cond_hidden_states, cond_temb);
    auto norm1_context_output = norm1_context.forward(encoder_hidden_states, temb);

#if 0
    norm1_output.x = hidden_states;
    norm1_cond_output.x = cond_hidden_states;
    norm1_context_output.x = encoder_hidden_states;
#endif

    debug("norm_hidden_states", norm1_output.x);
    debug("norm_cond_hidden_states", norm1_cond_output.x);
    debug("norm_encoder_hidden_states", norm1_context_output.x);

    constexpr int POOL_SIZE = OminiAttention::POOL_SIZE;

    nvtxRangePop();

    auto stream = getCurrentCUDAStream();
    
    int num_tokens_img_pad = 0;
    int num_tokens_txt_pad = 0;
    int num_tokens_cond_pad = 0;

    Tensor raw_attn_output;

    if (attnImpl == OminiAttentionImpl::FlashAttention2) {
        num_tokens_img_pad = num_tokens_img;
        num_tokens_txt_pad = num_tokens_txt;
        num_tokens_cond_pad = num_tokens_cond;

        Tensor concat;
        Tensor pool;

        {
            nvtxRangePushA("qkv_proj");

            const bool blockSparse = sparsityRatio > 0;

            const int poolTokens = num_tokens_img / POOL_SIZE + num_tokens_txt / POOL_SIZE + num_tokens_cond / POOL_SIZE;
            concat               = Tensor::allocate({batch_size, num_tokens_img + num_tokens_cond + num_tokens_txt, dim * 3},
                                      norm1_output.x.scalar_type(),
                                      norm1_output.x.device());

            pool = blockSparse ? Tensor::allocate({batch_size, poolTokens, dim * 3},
                                                  norm1_output.x.scalar_type(),
                                                  norm1_output.x.device())
                               : Tensor{};

            for (int i = 0; i < batch_size; i++) {
                // txt first
                Tensor qkv_context =
                    concat.slice(0, i, i + 1).slice(1, 0, num_tokens_txt);

                Tensor qkv = concat.slice(0, i, i + 1).slice(1, num_tokens_txt, num_tokens_txt + num_tokens_img);
                    
                Tensor qkv_cond =
                    concat.slice(0, i, i + 1).slice(1, num_tokens_txt + num_tokens_img, num_tokens_img + (num_tokens_cond + num_tokens_txt));

                Tensor pool_qkv_context =
                    pool.valid() ? pool.slice(0, i, i + 1).slice(1, 0, num_tokens_txt / POOL_SIZE) : Tensor{};
                    
                Tensor pool_qkv =
                    pool.valid() ? pool.slice(0, i, i + 1).slice(1, num_tokens_txt / POOL_SIZE,
                                                                    num_tokens_txt / POOL_SIZE + num_tokens_img / POOL_SIZE): Tensor{};

                Tensor pool_qkv_cond = pool.valid()
                                              ? pool.slice(0, i, i + 1)
                                                    .slice(1,
                                                           num_tokens_img / POOL_SIZE + num_tokens_txt / POOL_SIZE,
                                                           (num_tokens_img / POOL_SIZE + num_tokens_cond / POOL_SIZE) + num_tokens_txt / POOL_SIZE)
                                              : Tensor{};

                // qkv_proj.forward(norm1_output.x.slice(0, i, i + 1), qkv);
                // debug("qkv_raw", qkv);

                debug("rotary_emb", rotary_emb);

                qkv_proj.forward(
                    norm1_output.x.slice(0, i, i + 1), qkv, pool_qkv, norm_q.weight, norm_k.weight, rotary_emb);
                debug("qkv", qkv);

                debug("cond_rotary_emb", cond_rotary_emb);
                qkv_proj.forward(
                    norm1_cond_output.x.slice(0, i, i + 1), qkv_cond, pool_qkv_cond, norm_q.weight, norm_k.weight, cond_rotary_emb);
                

                // qkv_proj_context.forward(norm1_context_output.x.slice(0, i, i + 1), qkv_context);
                // debug("qkv_context_raw", qkv_context);

                debug("rotary_emb_context", rotary_emb_context);
                qkv_proj_context.forward(norm1_context_output.x.slice(0, i, i + 1),
                                         qkv_context,
                                         pool_qkv_context,
                                         norm_added_q.weight,
                                         norm_added_k.weight,
                                         rotary_emb_context);
                debug("qkv_context", qkv_context);
            }

            nvtxRangePop();
        }

        spdlog::debug("concat={}", concat.shape.str());
        debug("concat", concat);

        assert(concat.shape[2] == num_heads * dim_head * 3);

        nvtxRangePushA("Attention");

        if (pool.valid()) {
            raw_attn_output = attn.forward(concat, pool, sparsityRatio);
        } else {
            raw_attn_output = attn.forward(concat);
        }

        nvtxRangePop();

        spdlog::debug("raw_attn_output={}", raw_attn_output.shape.str());

        raw_attn_output = raw_attn_output.view(TensorShape{batch_size, num_tokens_img + num_tokens_cond + num_tokens_txt, num_heads, dim_head});

    } else if (attnImpl == OminiAttentionImpl::NunchakuFP16) {
        num_tokens_img_pad = ceilDiv(num_tokens_img, 256) * 256;
        num_tokens_txt_pad = ceilDiv(num_tokens_txt, 256) * 256;
        num_tokens_cond_pad = ceilDiv(num_tokens_cond, 256) * 256;

        Tensor concat_q, concat_k, concat_v;

        {
            nvtxRangePushA("qkv_proj");

            concat_q = Tensor::allocate({batch_size, num_heads, (num_tokens_img_pad + num_tokens_cond_pad) + num_tokens_txt_pad, dim_head},
                                        Tensor::FP16,
                                        norm1_output.x.device());
            concat_k = Tensor::empty_like(concat_q);
            concat_v = Tensor::empty_like(concat_q);

            for (int i = 0; i < batch_size; i++) {
                // txt first
                auto sliceTxt = [&](Tensor x) { return x.slice(0, i, i + 1).slice(2, 0, num_tokens_txt_pad); };
                auto sliceImg = [&](Tensor x) { return x.slice(0, i, i + 1).slice(2, num_tokens_txt_pad, num_tokens_txt_pad + num_tokens_img_pad); };
                auto sliceCond = [&](Tensor x) {
                    return x.slice(0, i, i + 1).slice(2, num_tokens_txt_pad+num_tokens_img_pad, num_tokens_img_pad + num_tokens_txt_pad + num_tokens_cond_pad);
                };

                qkv_proj.forward(norm1_output.x.slice(0, i, i + 1),
                                 {},
                                 {},
                                 norm_q.weight,
                                 norm_k.weight,
                                 rotary_emb,
                                 sliceImg(concat_q),
                                 sliceImg(concat_k),
                                 sliceImg(concat_v),
                                 num_tokens_img);

                qkv_proj.forward(norm1_cond_output.x.slice(0, i, i + 1),
                                {},
                                {},
                                norm_q.weight,
                                norm_k.weight,
                                cond_rotary_emb,
                                sliceCond(concat_q),
                                sliceCond(concat_k),
                                sliceCond(concat_v),
                                num_tokens_cond);

                qkv_proj_context.forward(norm1_context_output.x.slice(0, i, i + 1),
                                         {},
                                         {},
                                         norm_added_q.weight,
                                         norm_added_k.weight,
                                         rotary_emb_context,
                                         sliceTxt(concat_q),
                                         sliceTxt(concat_k),
                                         sliceTxt(concat_v),
                                         num_tokens_txt);
            }

            debug("concat_q", concat_q);
            debug("concat_k", concat_k);
            debug("concat_v", concat_v);

            nvtxRangePop();
        }

        raw_attn_output = Tensor::allocate({batch_size, (num_tokens_img_pad + num_tokens_cond_pad) + num_tokens_txt_pad, num_heads * dim_head},
                                           norm1_output.x.scalar_type(),
                                           norm1_output.x.device());

        nvtxRangePushA("Attention");

        kernels::attention_fp16(concat_q, concat_k, concat_v, raw_attn_output, pow(dim_head, (-0.5)));

        nvtxRangePop();

        raw_attn_output =
            raw_attn_output.view(TensorShape{batch_size, (num_tokens_img_pad + num_tokens_cond_pad) + num_tokens_txt_pad, num_heads, dim_head});
    } else {
        assert(false);
    }

    debug("raw_attn_output", raw_attn_output);

    {
        nvtxRangePushA("o_proj");

        auto &&[_, gate_msa, shift_mlp, scale_mlp, gate_mlp] = norm1_output;

        // raw_attn_output: [batch_size, num_tokens_img + num_tokens_txt, num_heads * dim_head]

        Tensor raw_attn_output_split;
        if (batch_size == 1) {
            raw_attn_output_split =
                raw_attn_output.slice(1, num_tokens_txt_pad, num_tokens_txt_pad + num_tokens_img).reshape({batch_size, num_tokens_img, num_heads * dim_head});
        } else {
            raw_attn_output_split = Tensor::allocate({batch_size, num_tokens_img, num_heads * dim_head},
                                                     raw_attn_output.scalar_type(),
                                                     raw_attn_output.device());
            
            checkCUDA(cudaMemcpy2DAsync(raw_attn_output_split.data_ptr(),
                                        num_tokens_img * num_heads * dim_head * raw_attn_output_split.scalar_size(),
                                        raw_attn_output.data_ptr<char>() + num_tokens_txt_pad * num_heads * dim_head *
                                                                               raw_attn_output_split.scalar_size(),
                                        ((num_tokens_img_pad + num_tokens_cond_pad) + num_tokens_txt_pad) * num_heads * dim_head *
                                            raw_attn_output.scalar_size(),
                                        num_tokens_img * num_heads * dim_head * raw_attn_output_split.scalar_size(),
                                        batch_size,
                                        cudaMemcpyDeviceToDevice,
                                        stream));
        }

        spdlog::debug("raw_attn_output_split={}", raw_attn_output_split.shape.str());
        debug("img.raw_attn_output_split", raw_attn_output_split);

        Tensor attn_output =
            omini_forward_fc(out_proj, raw_attn_output_split); // std::get<Tensor>(out_proj.forward(raw_attn_output_split));
        debug("img.attn_output", attn_output);

#if 1
        // kernels::mul_add(attn_output, gate_msa, hidden_states);
        kernels::mul_add_batch(attn_output, gate_msa, true, 0.0, hidden_states, true);
        hidden_states = std::move(attn_output);

        nvtxRangePop();
        nvtxRangePushA("MLP");

        spdlog::debug("attn_output={}", hidden_states.shape.str());

        Tensor norm_hidden_states = norm2.forward(hidden_states);
        debug("scale_mlp", scale_mlp);
        debug("shift_mlp", shift_mlp);
        // kernels::mul_add(norm_hidden_states, scale_mlp, shift_mlp);
        kernels::mul_add_batch(norm_hidden_states, scale_mlp, true, 0.0, shift_mlp, true);

        spdlog::debug("norm_hidden_states={}", norm_hidden_states.shape.str());
#else
        Tensor norm_hidden_states = hidden_states;
#endif

        // Tensor ff_output = mlp_fc2.forward(GELU::forward(mlp_fc1.forward(norm_hidden_states)));
        debug("img.ff_input", norm_hidden_states);
        Tensor ff_output = omini_forward_mlp(mlp_fc1, mlp_fc2, norm_hidden_states);
        debug("img.ff_output", ff_output);

        debug("gate_mlp", gate_mlp);
        // kernels::mul_add(ff_output, gate_mlp, hidden_states);
        kernels::mul_add_batch(ff_output, gate_mlp, true, 0.0, hidden_states, true);
        hidden_states = std::move(ff_output);

        nvtxRangePop();

        spdlog::debug("ff_output={}", hidden_states.shape.str());
    }

    {
        nvtxRangePushA("o_proj_cond");

        auto &&[_, cond_gate_msa, cond_shift_mlp, cond_scale_mlp, cond_gate_mlp] = norm1_cond_output;

        Tensor raw_attn_output_split_cond;
        if (batch_size == 1) {
            raw_attn_output_split_cond = raw_attn_output.slice(1, num_tokens_txt_pad + num_tokens_img_pad, num_tokens_txt_pad + num_tokens_img_pad + num_tokens_cond).reshape({batch_size, num_tokens_cond, num_heads * dim_head});
        } else {
            raw_attn_output_split_cond = Tensor::allocate({batch_size, num_tokens_cond, num_heads * dim_head},
                                                     raw_attn_output.scalar_type(),
                                                     raw_attn_output.device());
            checkCUDA(cudaMemcpy2DAsync(raw_attn_output_split_cond.data_ptr(),
                                        num_tokens_cond * num_heads * dim_head * raw_attn_output_split_cond.scalar_size(),
                                        raw_attn_output.data_ptr<char>() + (num_tokens_txt_pad + num_tokens_img_pad) * num_heads * dim_head *
                                            raw_attn_output_split_cond.scalar_size(),
                                        ((num_tokens_img_pad + num_tokens_cond_pad) + num_tokens_txt_pad) * num_heads * dim_head *raw_attn_output_split_cond.scalar_size(),
                                        num_tokens_cond * num_heads * dim_head * raw_attn_output_split_cond.scalar_size(),
                                        batch_size, 
                                        cudaMemcpyDeviceToDevice, 
                                        stream));
        }

        Tensor attn_output_cond = 
            omini_forward_fc(out_proj, raw_attn_output_split_cond);
        debug("cond.attn_output_cond", attn_output_cond);
#if 1
        kernels::mul_add_batch(attn_output_cond, cond_gate_msa, true, 0.0, cond_hidden_states, true);
        cond_hidden_states = std::move(attn_output_cond);

        nvtxRangePop();
        nvtxRangePushA("MLP");

        Tensor norm_cond_hidden_states = norm2.forward(cond_hidden_states);
        kernels::mul_add_batch(norm_cond_hidden_states, cond_scale_mlp, true, 0.0, cond_shift_mlp, true);

        spdlog::debug("norm_cond_hidden_states={}", norm_cond_hidden_states.shape.str());
#else
        Tensor norm_cond_hidden_states = cond_hidden_states;
#endif

        // Tensor ff_output_cond = mlp_context_fc2.forward(GELU::forward(mlp_context_fc1.forward(norm_cond_hidden_states)));
        debug("cond.ff_input", norm_cond_hidden_states);
        Tensor ff_output_cond = omini_forward_mlp(mlp_fc1, mlp_fc2, norm_cond_hidden_states);
        debug("cond.ff_output", ff_output_cond);

        kernels::mul_add_batch(ff_output_cond, cond_gate_mlp, true, 0.0, cond_hidden_states, true);
        cond_hidden_states = std::move(ff_output_cond);

        nvtxRangePop();

        spdlog::debug("ff_output_cond={}", cond_hidden_states.shape.str());
        
    }

    if (context_pre_only) {
        return {hidden_states, cond_hidden_states, encoder_hidden_states};
    }

    {
        nvtxRangePushA("o_proj_context");

        auto &&[_, gate_msa, shift_mlp, scale_mlp, gate_mlp] = norm1_context_output;

        Tensor raw_attn_output_split;
        if (batch_size == 1) {
            raw_attn_output_split = raw_attn_output.slice(1, 0, num_tokens_txt)
                                        .reshape({batch_size, num_tokens_txt, num_heads * dim_head});
        } else {
            raw_attn_output_split = Tensor::allocate({batch_size, num_tokens_txt, num_heads * dim_head},
                                                     raw_attn_output.scalar_type(),
                                                     raw_attn_output.device());

            checkCUDA(cudaMemcpy2DAsync(raw_attn_output_split.data_ptr(),
                                        num_tokens_txt * num_heads * dim_head * raw_attn_output_split.scalar_size(),
                                        raw_attn_output.data_ptr(),
                                        ((num_tokens_img_pad + num_tokens_cond_pad) + num_tokens_txt_pad) * num_heads * dim_head * raw_attn_output.scalar_size(),
                                        num_tokens_txt * num_heads * dim_head * raw_attn_output_split.scalar_size(),
                                        batch_size,
                                        cudaMemcpyDeviceToDevice,
                                        stream));
        }

        spdlog::debug("raw_attn_output_split={}", raw_attn_output_split.shape.str());
        debug("context.raw_attn_output_split", raw_attn_output_split);

        Tensor attn_output =
            omini_forward_fc(out_proj_context,
                       raw_attn_output_split); // std::get<Tensor>(out_proj_context.forward(raw_attn_output_split));
        debug("context.attn_output", attn_output);

#if 1
        // kernels::mul_add(attn_output, gate_msa, encoder_hidden_states);
        kernels::mul_add_batch(attn_output, gate_msa, true, 0.0, encoder_hidden_states, true);
        encoder_hidden_states = std::move(attn_output);

        nvtxRangePop();
        nvtxRangePushA("MLP");

        spdlog::debug("attn_output={}", encoder_hidden_states.shape.str());

        Tensor norm_hidden_states = norm2_context.forward(encoder_hidden_states);
        debug("c_scale_mlp", scale_mlp);
        debug("c_shift_mlp", shift_mlp);
        // kernels::mul_add(norm_hidden_states, scale_mlp, shift_mlp);
        kernels::mul_add_batch(norm_hidden_states, scale_mlp, true, 0.0, shift_mlp, true);

        spdlog::debug("norm_hidden_states={}", norm_hidden_states.shape.str());
#else
        auto norm_hidden_states = encoder_hidden_states;
#endif

        // Tensor ff_output = mlp_context_fc2.forward(GELU::forward(mlp_context_fc1.forward(norm_hidden_states)));
        // Tensor ff_output =
        // mlp_context_fc2.forward_quant(quant_static_fuse_gelu(mlp_context_fc1.forward(norm_hidden_states), 1.0));
        debug("context.ff_input", norm_hidden_states);
        Tensor ff_output = omini_forward_mlp(mlp_context_fc1, mlp_context_fc2, norm_hidden_states);
        debug("context.ff_output", ff_output);

        debug("c_gate_mlp", gate_mlp);
        // kernels::mul_add(ff_output, gate_mlp, encoder_hidden_states);
        kernels::mul_add_batch(ff_output, gate_mlp, true, 0.0, encoder_hidden_states, true);
        encoder_hidden_states = std::move(ff_output);

        nvtxRangePop();

        spdlog::debug("ff_output={}", encoder_hidden_states.shape.str());
    }

    nvtxRangePop();

    return {hidden_states, cond_hidden_states, encoder_hidden_states};
}

// Main class for the OminiFlux model, composed of OminiJointTransformerBlocks and OminiFluxSingleTransformerBlocks.
// `use_fp4` enables 4-bit quantization.
// `offload` enables layer-wise parameter offloading to CPU to save GPU memory.
OminiFluxModel::OminiFluxModel(bool use_fp4, bool offload, Tensor::ScalarType dtype, Device device)
    : dtype(dtype), offload(offload) {
    // Initialize joint transformer blocks (typically the first set of layers)
    for (int i = 0; i < 19; i++) {
        transformer_blocks.push_back(
            std::make_unique<OminiJointTransformerBlock>(3072, 24, 3072, false, use_fp4, dtype, device));
        registerChildren(*transformer_blocks.back(), format("transformer_blocks.{}", i));
        if (offload && i > 0) { // don't offload first block
            transformer_blocks.back()->setLazyLoad(true);
            transformer_blocks.back()->releaseLazyParams();
        }
    }
    for (int i = 0; i < 38; i++) {
        single_transformer_blocks.push_back(
            std::make_unique<OminiFluxSingleTransformerBlock>(3072, 24, 3072, 4, use_fp4, dtype, device));
        registerChildren(*single_transformer_blocks.back(), format("single_transformer_blocks.{}", i));
        if (offload) {
            single_transformer_blocks.back()->setLazyLoad(true);
            single_transformer_blocks.back()->releaseLazyParams();
        }
    }
}

// Full forward pass for the OminiFluxModel.
// Iterates through all transformer blocks (joint then single).
// `hidden_states`: Image latents.
// `cond_hidden_states`: Conditional latents.
// `encoder_hidden_states`: Text latents.
// `temb`, `cond_temb`: Time embeddings.
// `rotary_emb_*`: Rotary position embeddings for different modalities.
// `controlnet_block_samples`, `controlnet_single_block_samples`: Optional ControlNet inputs.
// `skip_first_layer`: Option to skip the very first joint transformer block.
Tensor OminiFluxModel::forward(Tensor hidden_states,
                          Tensor cond_hidden_states,
                          Tensor encoder_hidden_states,
                          Tensor temb,
                          Tensor cond_temb,
                          Tensor rotary_emb_img,
                          Tensor rotary_emb_context,
                          Tensor rotary_emb_single,
                          Tensor rotary_emb_cond,
                          Tensor controlnet_block_samples,
                          Tensor controlnet_single_block_samples,
                          bool skip_first_layer) {
    // Add validation at the start
    assert(hidden_states.shape[0] == cond_hidden_states.shape[0] && "Batch sizes must match");
    assert(hidden_states.shape[2] == cond_hidden_states.shape[2] && "Hidden dimensions must match");
    assert(encoder_hidden_states.shape[0] == hidden_states.shape[0] && "Encoder batch size must match");
    assert(encoder_hidden_states.shape[2] == hidden_states.shape[2] && "Encoder hidden dimension must match");

    const int batch_size           = hidden_states.shape[0];
    const Tensor::ScalarType dtype = hidden_states.dtype();
    const Device device            = hidden_states.device();

    const int txt_tokens = encoder_hidden_states.shape[1];
    const int img_tokens = hidden_states.shape[1];
    const int num_cond_tokens = cond_hidden_states.shape[1];


    const int numLayers = transformer_blocks.size() + single_transformer_blocks.size();

    Tensor concat;

    auto compute = [&](int layer) {
        if (skip_first_layer && size_t(layer) == 0)
            return;
        if (size_t(layer) < transformer_blocks.size()) {
            auto &block = transformer_blocks.at(layer);
            std::tie(hidden_states, cond_hidden_states, encoder_hidden_states) =
                block->forward(hidden_states, cond_hidden_states, encoder_hidden_states, temb,  cond_temb, rotary_emb_img, rotary_emb_context, rotary_emb_cond, 0.0f);
            if (controlnet_block_samples.valid()) {
                const int num_controlnet_block_samples = controlnet_block_samples.shape[0];

                int interval_control =
                    ceilDiv(transformer_blocks.size(), static_cast<size_t>(num_controlnet_block_samples));
                int block_index = layer / interval_control;
                // Xlabs ControlNet
                // block_index = layer % num_controlnet_block_samples;

                hidden_states = kernels::add(hidden_states, controlnet_block_samples[block_index]);
            }
            if (residual_callback && layer % 2 == 0) {
                Tensor cpu_input = hidden_states.copy(Device::cpu());
                pybind11::gil_scoped_acquire gil;
                Tensor cpu_output = residual_callback(cpu_input);
                Tensor residual   = cpu_output.copy(Device::cuda());
                hidden_states     = kernels::add(hidden_states, residual);
            }
        } else {
            if (size_t(layer) == transformer_blocks.size()) {
                // txt first, same as diffusers
                concat = Tensor::allocate(TensorShape{batch_size, txt_tokens + img_tokens, 3072}, dtype, device);
                for (int i = 0; i < batch_size; i++) {
                    // Copy text tokens first
                    concat.slice(0, i, i + 1)
                        .slice(1, 0, txt_tokens)
                        .copy_(encoder_hidden_states.slice(0, i, i + 1));
                    
                    // Copy image tokens second
                    concat.slice(0, i, i + 1)
                        .slice(1, txt_tokens, txt_tokens + img_tokens)
                        .copy_(hidden_states.slice(0, i, i + 1));
                    
                }
                hidden_states = concat;
                cond_hidden_states = cond_hidden_states;
                encoder_hidden_states = {};
                // concat = {};  // Clear the concat tensor to free memory
            }

            auto &block   = single_transformer_blocks.at(layer - transformer_blocks.size());
            std::tie(hidden_states, cond_hidden_states) = block->forward(hidden_states, cond_hidden_states, temb, cond_temb, rotary_emb_single, rotary_emb_cond);
            if (controlnet_single_block_samples.valid()) {
                const int num_controlnet_single_block_samples = controlnet_single_block_samples.shape[0];

                int interval_control =
                    ceilDiv(single_transformer_blocks.size(), static_cast<size_t>(num_controlnet_single_block_samples));
                int block_index = (layer - transformer_blocks.size()) / interval_control;
                // Xlabs ControlNet
                // block_index = layer % num_controlnet_single_block_samples

                auto slice = hidden_states.slice(1, txt_tokens, txt_tokens + img_tokens);
                slice      = kernels::add(slice, controlnet_single_block_samples[block_index]);
                hidden_states.slice(1, txt_tokens, txt_tokens + img_tokens).copy_(slice);
            }
            size_t local_layer_idx = layer - transformer_blocks.size();
            if (residual_callback && local_layer_idx % 4 == 0) {
                Tensor callback_input = hidden_states.slice(1, txt_tokens, txt_tokens + img_tokens);
                Tensor cpu_input      = callback_input.copy(Device::cpu());
                pybind11::gil_scoped_acquire gil;
                Tensor cpu_output = residual_callback(cpu_input);
                Tensor residual   = cpu_output.copy(Device::cuda());
                auto slice        = hidden_states.slice(1, txt_tokens, txt_tokens + img_tokens);
                slice             = kernels::add(slice, residual);
                hidden_states.slice(1, txt_tokens, txt_tokens + img_tokens).copy_(slice);
            }
        }
    };
    auto load = [&](int layer) {
        if (size_t(layer) < transformer_blocks.size()) {
            auto &block = transformer_blocks.at(layer);
            block->loadLazyParams();
        } else {
            auto &block = single_transformer_blocks.at(layer - transformer_blocks.size());
            block->loadLazyParams();
        }
    };
    auto unload = [&](int layer) {
        if (size_t(layer) < transformer_blocks.size()) {
            auto &block = transformer_blocks.at(layer);
            block->releaseLazyParams();
        } else {
            auto &block = single_transformer_blocks.at(layer - transformer_blocks.size());
            block->releaseLazyParams();
        }
    };

    LayerOffloadHelper helper(this->offload, numLayers, compute, load, unload);
    helper.run();

    return hidden_states;
}

// Forward pass for a single specified layer of the OminiFluxModel.
// This allows for inspecting or manipulating intermediate layer outputs.
// Parameters are similar to the main `forward` method.
// Returns a tuple of (hidden_states, encoder_hidden_states, cond_hidden_states) after the specified layer.
std::tuple<Tensor, Tensor, Tensor> OminiFluxModel::forward_layer(size_t layer,
                                                    Tensor hidden_states,
                                                    Tensor cond_hidden_states,
                                                    Tensor encoder_hidden_states,
                                                    Tensor temb,
                                                    Tensor cond_temb,
                                                    Tensor rotary_emb_img,
                                                    Tensor rotary_emb_context,
                                                    Tensor rotary_emb_cond,
                                                    Tensor controlnet_block_samples,
                                                    Tensor controlnet_single_block_samples) {

    if (layer < transformer_blocks.size()) {
        std::tie(hidden_states, cond_hidden_states, encoder_hidden_states) = transformer_blocks.at(layer)->forward(
            hidden_states, cond_hidden_states, encoder_hidden_states, temb, cond_temb, rotary_emb_img, rotary_emb_context, rotary_emb_cond, 0.0f);
    } else {
        // For single blocks, concatenate hidden_states and encoder_hidden_states similar to the forward method
        const int batch_size = hidden_states.shape[0];
        const Tensor::ScalarType dtype = hidden_states.dtype();
        const Device device = hidden_states.device();
        const int txt_tokens = encoder_hidden_states.shape[1];
        const int img_tokens = hidden_states.shape[1];
        
        Tensor concat = Tensor::allocate(TensorShape{batch_size, txt_tokens + img_tokens, hidden_states.shape[2]}, dtype, device);
        for (int i = 0; i < batch_size; i++) {
            // Copy text tokens first
            concat.slice(0, i, i + 1)
                .slice(1, 0, txt_tokens)
                .copy_(encoder_hidden_states.slice(0, i, i + 1));
            
            // Copy image tokens second
            concat.slice(0, i, i + 1)
                .slice(1, txt_tokens, txt_tokens + img_tokens)
                .copy_(hidden_states.slice(0, i, i + 1));
            
        }
        hidden_states = concat;
        
        Tensor single_hidden_states;
        std::tie(hidden_states, cond_hidden_states) = 
            single_transformer_blocks.at(layer - transformer_blocks.size())->forward(
                hidden_states, cond_hidden_states, temb, cond_temb, rotary_emb_img, rotary_emb_cond);
        
        // Extract encoder_hidden_states back from the concatenated tensor
        encoder_hidden_states = hidden_states.slice(1, 0, txt_tokens);
        hidden_states = hidden_states.slice(1, txt_tokens, txt_tokens + img_tokens);
    }

    const int txt_tokens = encoder_hidden_states.shape[1];
    const int img_tokens = hidden_states.shape[1];
    const int cond_tokens = cond_hidden_states.shape[1];

    if (layer < transformer_blocks.size() && controlnet_block_samples.valid()) {
        const int num_controlnet_block_samples = controlnet_block_samples.shape[0];

        int interval_control = ceilDiv(transformer_blocks.size(), static_cast<size_t>(num_controlnet_block_samples));
        int block_index      = layer / interval_control;
        // Xlabs ControlNet
        // block_index = layer % num_controlnet_block_samples;

        hidden_states = kernels::add(hidden_states, controlnet_block_samples[block_index]);
    } else if (layer >= transformer_blocks.size() && controlnet_single_block_samples.valid()) {
        const int num_controlnet_single_block_samples = controlnet_single_block_samples.shape[0];

        int interval_control =
            ceilDiv(single_transformer_blocks.size(), static_cast<size_t>(num_controlnet_single_block_samples));
        int block_index = (layer - transformer_blocks.size()) / interval_control;
        // Xlabs ControlNet
        // block_index = layer % num_controlnet_single_block_samples

        auto slice = hidden_states.slice(1, txt_tokens, txt_tokens + img_tokens);
        slice      = kernels::add(slice, controlnet_single_block_samples[block_index]);
        hidden_states.slice(1, txt_tokens, txt_tokens + img_tokens).copy_(slice);
    }

    return {hidden_states, encoder_hidden_states, cond_hidden_states};
}

// Sets the attention implementation (e.g., FlashAttention2, NunchakuFP16)
// for all attention modules within the model.
void OminiFluxModel::setAttentionImpl(OminiAttentionImpl impl) {
    for (auto &&block : this->transformer_blocks) {
        block->attnImpl = impl;
    }
    for (auto &&block : this->single_transformer_blocks) {
        block->attnImpl = impl;
    }
}
// Sets a callback function that can be used to inject or modify residuals
// at certain points within the transformer blocks.
void OminiFluxModel::set_residual_callback(std::function<Tensor(const Tensor &)> cb) {
    residual_callback = std::move(cb);
}
