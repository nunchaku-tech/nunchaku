#pragma once

#include "interop/torch.h"
#include "FluxModel.h"
#include "Serialization.h"
#include "debug.h"
#include "Linear.h"
#include "module.h"

class QuantizedFluxModel : public ModuleWrapper<FluxModel> { // : public torch::CustomClassHolder {
public:
    void init(bool use_fp4, bool bf16, int8_t deviceId) {
        spdlog::info("Initializing QuantizedFluxModel");
        net = std::make_unique<FluxModel>(use_fp4, bf16 ? Tensor::BF16 : Tensor::FP16, Device::cuda((int)deviceId));
    }

    torch::Tensor forward(
        torch::Tensor hidden_states,
        torch::Tensor encoder_hidden_states,
        torch::Tensor temb,
        torch::Tensor rotary_emb_img,
        torch::Tensor rotary_emb_context,
        torch::Tensor rotary_emb_single,
        bool skip_first_layer = false)
    {
        checkModel();

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
        torch::Tensor rotary_emb_context)
    {
        spdlog::debug("QuantizedFluxModel forward_layer {}", idx);

        hidden_states = hidden_states.contiguous();
        encoder_hidden_states = encoder_hidden_states.contiguous();
        temb = temb.contiguous();
        rotary_emb_img = rotary_emb_img.contiguous();
        rotary_emb_context = rotary_emb_context.contiguous();

        auto &&[result_img, result_txt] = net->transformer_blocks.at(idx)->forward(
            from_torch(hidden_states),
            from_torch(encoder_hidden_states),
            from_torch(temb),
            from_torch(rotary_emb_img),
            from_torch(rotary_emb_context),
            0.0f
        );

        hidden_states = to_torch(result_img);
        encoder_hidden_states = to_torch(result_txt);
        Tensor::synchronizeDevice();

        return { hidden_states, encoder_hidden_states };
    }

    torch::Tensor forward_single_layer(
        int64_t idx,
        torch::Tensor hidden_states,
        torch::Tensor temb,
        torch::Tensor rotary_emb_single)
    {
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


    // must be called after loading lora
    // skip specific ranks in W4A4 layers
    void setLoraScale(int skipRanks, float scale) {
        if (skipRanks % 16 != 0) {
            throw std::invalid_argument("skipRanks must be multiples of 16");
        }

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

    void forceFP16Attention(bool enable) {
        Attention::setForceFP16(net.get(), enable);
    }

};