#pragma once

#include "interop/torch.h"
#include "FluxModel.h"
#include "Serialization.h"
#include "debug.h"
#include "Linear.h"
#include "module.h"

class QuantizedFluxModel : public ModuleWrapper<FluxModel> { // : public torch::CustomClassHolder {
public:
    void init(bool use_fp4, bool offload, bool bf16, int8_t deviceId) {
        spdlog::info("Initializing QuantizedFluxModel on device {}", deviceId);
        if (!bf16) {
            spdlog::info("Use FP16 model");
        }
        if (offload) {
            spdlog::info("Layer offloading enabled");
        }
        ModuleWrapper::init(deviceId);

        CUDADeviceContext ctx(this->deviceId);
        net = std::make_unique<FluxModel>(use_fp4, offload, bf16 ? Tensor::BF16 : Tensor::FP16, Device::cuda((int)deviceId));
    }

    bool isBF16() {
        checkModel();
        return net->dtype == Tensor::BF16;
    }

    torch::Tensor forward(
        torch::Tensor hidden_states,
        torch::Tensor encoder_hidden_states,
        torch::Tensor temb,
        torch::Tensor rotary_emb_img,
        torch::Tensor rotary_emb_context,
        torch::Tensor rotary_emb_single,
        std::optional<torch::Tensor> controlnet_block_samples = std::nullopt,
        std::optional<torch::Tensor> controlnet_single_block_samples = std::nullopt,
        bool skip_first_layer = false)
    {
        checkModel();
        CUDADeviceContext ctx(deviceId);

        spdlog::debug("QuantizedFluxModel forward");

        hidden_states = hidden_states.contiguous();
        encoder_hidden_states = encoder_hidden_states.contiguous();
        temb = temb.contiguous();
        rotary_emb_img = rotary_emb_img.contiguous();
        rotary_emb_context = rotary_emb_context.contiguous();
        rotary_emb_single = rotary_emb_single.contiguous();

        Tensor result = net->forward(
            from_torch(hidden_states),
            from_torch(encoder_hidden_states),
            from_torch(temb),
            from_torch(rotary_emb_img),
            from_torch(rotary_emb_context),
            from_torch(rotary_emb_single),
            controlnet_block_samples.has_value() ? from_torch(controlnet_block_samples.value().contiguous()) : Tensor{},
            controlnet_single_block_samples.has_value() ? from_torch(controlnet_single_block_samples.value().contiguous()) : Tensor{},
            skip_first_layer
        );

        torch::Tensor output = to_torch(result);
        Tensor::synchronizeDevice();

        return output;
    }

    std::tuple<torch::Tensor, torch::Tensor> forward_layer(
        int64_t idx,
        torch::Tensor hidden_states,
        torch::Tensor encoder_hidden_states,
        torch::Tensor temb,
        torch::Tensor rotary_emb_img,
        torch::Tensor rotary_emb_context,
        std::optional<torch::Tensor> controlnet_block_samples = std::nullopt,
        std::optional<torch::Tensor> controlnet_single_block_samples = std::nullopt)
    {
        CUDADeviceContext ctx(deviceId);

        spdlog::debug("QuantizedFluxModel forward_layer {}", idx);

        hidden_states = hidden_states.contiguous();
        encoder_hidden_states = encoder_hidden_states.contiguous();
        temb = temb.contiguous();
        rotary_emb_img = rotary_emb_img.contiguous();
        rotary_emb_context = rotary_emb_context.contiguous();

        auto &&[hidden_states_, encoder_hidden_states_] = net->forward_layer(
            idx,
            from_torch(hidden_states),
            from_torch(encoder_hidden_states),
            from_torch(temb),
            from_torch(rotary_emb_img),
            from_torch(rotary_emb_context),
            controlnet_block_samples.has_value() ? from_torch(controlnet_block_samples.value().contiguous()) : Tensor{},
            controlnet_single_block_samples.has_value() ? from_torch(controlnet_single_block_samples.value().contiguous()) : Tensor{}
        );

        hidden_states = to_torch(hidden_states_);
        encoder_hidden_states = to_torch(encoder_hidden_states_);
        Tensor::synchronizeDevice();

        return { hidden_states, encoder_hidden_states };
    }

    torch::Tensor forward_single_layer(
        int64_t idx,
        torch::Tensor hidden_states,
        torch::Tensor temb,
        torch::Tensor rotary_emb_single)
    {
        CUDADeviceContext ctx(deviceId);

        spdlog::debug("QuantizedFluxModel forward_single_layer {}", idx);

        hidden_states = hidden_states.contiguous();
        temb = temb.contiguous();
        rotary_emb_single = rotary_emb_single.contiguous();

        Tensor result = net->single_transformer_blocks.at(idx)->forward(
            from_torch(hidden_states),
            from_torch(temb),
            from_torch(rotary_emb_single)
        );

        hidden_states = to_torch(result);
        Tensor::synchronizeDevice();

        return hidden_states;
    }

    // expose the norm1 forward method of the transformer blocks
    // this is used by TeaCache to get the norm1 output
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> norm1_forward(
        int64_t idx,
        torch::Tensor hidden_states,
        torch::Tensor temb
    ) {
        AdaLayerNormZero::Output result = net->transformer_blocks.at(idx)->norm1.forward(from_torch(hidden_states), from_torch(temb));
        return {
            to_torch(result.x),
            to_torch(result.gate_msa),
            to_torch(result.shift_mlp),
            to_torch(result.scale_mlp),
            to_torch(result.gate_mlp)
        };
    }

    // must be called after loading lora
    // skip specific ranks in W4A4 layers
    void setLoraScale(int skipRanks, float scale) {
        if (skipRanks % 16 != 0) {
            throw std::invalid_argument("skipRanks must be multiples of 16");
        }

        CUDADeviceContext ctx(deviceId);

        spdlog::info("Set lora scale to {} (skip {} ranks)", scale, skipRanks);

        net->traverse([&](Module *module) {
            if (auto *m = dynamic_cast<GEMV_AWQ *>(module)) {
                m->lora_scale = scale;
            } else if (auto *m = dynamic_cast<GEMM_W4A4 *>(module)) {
                for (int i = 0; i < skipRanks / 16; i++) {
                    m->lora_scales[i] = 1.0f;
                }
                for (int i = skipRanks / 16; i < (int)m->lora_scales.size(); i++) {
                    m->lora_scales[i] = scale;
                }
            }
        });
    }

    void setAttentionImpl(std::string name) {
        if (name.empty() || name == "default") {
            name = "flashattn2";
        }

        spdlog::info("Set attention implementation to {}", name);

        if (name == "flashattn2") {
            net->setAttentionImpl(AttentionImpl::FlashAttention2);
        } else if (name == "nunchaku-fp16") {
            net->setAttentionImpl(AttentionImpl::NunchakuFP16);
        } else {
            throw std::invalid_argument(spdlog::fmt_lib::format("Invalid attention implementation {}", name));
        }
    }

};