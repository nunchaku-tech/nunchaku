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
    pybind11::function residual_callback;
    void set_residual_callback(pybind11::function callback) {
        //spdlog::info(">>> [set_residual_callback] START");
    
        pybind11::gil_scoped_acquire gil;
        //spdlog::info(">>> [set_residual_callback] Acquired GIL");
    
        //spdlog::info(">>> [set_residual_callback] Callback repr (before store): {}", std::string(pybind11::str(callback)));
    
        residual_callback = std::move(callback);  // 保存一份，防止 lambda 中悬挂
        //spdlog::info(">>> [set_residual_callback] Stored to residual_callback");
    
        if (net) {
            pybind11::object cb = residual_callback;
            //spdlog::info(">>> [set_residual_callback] Captured as pybind11::object, ptr={}", (void*)cb.ptr());
    
            net->set_residual_callback([cb](const Tensor &x) -> Tensor {
                //spdlog::info(">>> [lambda] ENTER");
    
                pybind11::gil_scoped_acquire gil;
                //spdlog::info(">>> [lambda] GIL acquired");
    
                //spdlog::info(">>> [lambda] cb repr: {}", std::string(pybind11::str(cb)));
                //spdlog::info(">>> [lambda] cb ptr: {}", (void*)cb.ptr());

                //spdlog::info(">>> [lambda] x shape: {}", x.shape.str());
    
                torch::Tensor torch_x = to_torch(x);
                //spdlog::info(">>> [lambda] Converted to torch_x");
    
                pybind11::object result = cb(torch_x);
                //spdlog::info(">>> [lambda] Callback called, got result");

    
                torch::Tensor torch_y = result.cast<torch::Tensor>();
                //spdlog::info(">>> [lambda] Casted to torch::Tensor");
    
                Tensor y = from_torch(torch_y);
                //spdlog::info(">>> [lambda] Converted back to Tensor");
                //spdlog::info(">>> [lambda] Converted back to Tensor, shape: {}", y.shape.str());
    
                return y;
            });
    
            //spdlog::info(">>> [set_residual_callback] Callback set on net");
        } else {
            //spdlog::warn(">>> [set_residual_callback] net is null!");
        }
    
        //spdlog::info(">>> [set_residual_callback] END");
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