#pragma once

#include "interop/torch.h"
#include "SanaModel.h"
#include "Serialization.h"
#include "debug.h"
#include "module.h"

class QuantizedSanaModel : public ModuleWrapper<SanaModel> {
public:
    void init(pybind11::dict config, std::vector<int> pag_layers, bool use_fp4, bool bf16, int8_t deviceId) {
        spdlog::info("Initializing QuantizedSanaModel on device {}", deviceId);
        SanaConfig cfg{
            .num_layers                = config["num_layers"].cast<int>(),
            .num_attention_heads       = config["num_attention_heads"].cast<int>(),
            .attention_head_dim        = config["attention_head_dim"].cast<int>(),
            .num_cross_attention_heads = config["num_cross_attention_heads"].cast<int>(),
            .expand_ratio              = config["mlp_ratio"].cast<double>(),
            .pag_layers                = pag_layers,
            .use_fp4                   = use_fp4,
        };

        ModuleWrapper::init(deviceId);
        CUDADeviceContext ctx(this->deviceId);
        net = std::make_unique<SanaModel>(cfg, bf16 ? Tensor::BF16 : Tensor::FP16, Device::cuda((int)deviceId));
    }

    torch::Tensor forward(torch::Tensor hidden_states,
                          torch::Tensor encoder_hidden_states,
                          torch::Tensor timestep,
                          torch::Tensor cu_seqlens_img,
                          torch::Tensor cu_seqlens_txt,
                          int H,
                          int W,
                          bool pag,
                          bool cfg,
                          bool skip_first_layer = false) {
        checkModel();
        CUDADeviceContext ctx(deviceId);

        spdlog::debug("QuantizedSanaModel forward");

        hidden_states         = hidden_states.contiguous();
        encoder_hidden_states = encoder_hidden_states.contiguous();
        timestep              = timestep.contiguous();
        cu_seqlens_img        = cu_seqlens_img.contiguous();
        cu_seqlens_txt        = cu_seqlens_txt.contiguous();

        Tensor result = net->forward(from_torch(hidden_states),
                                     from_torch(encoder_hidden_states),
                                     from_torch(timestep),
                                     from_torch(cu_seqlens_img),
                                     from_torch(cu_seqlens_txt),
                                     H,
                                     W,
                                     pag,
                                     cfg,
                                     skip_first_layer);

        torch::Tensor output = to_torch(result);
        // Tensor::synchronizeDevice();

        return output;
    }

    torch::Tensor forward_layer(int64_t idx,
                                torch::Tensor hidden_states,
                                torch::Tensor encoder_hidden_states,
                                torch::Tensor timestep,
                                torch::Tensor cu_seqlens_img,
                                torch::Tensor cu_seqlens_txt,
                                int H,
                                int W,
                                bool pag,
                                bool cfg) {
        checkModel();
        CUDADeviceContext ctx(deviceId);

        spdlog::debug("QuantizedSanaModel forward_layer {}", idx);

        hidden_states         = hidden_states.contiguous();
        encoder_hidden_states = encoder_hidden_states.contiguous();
        timestep              = timestep.contiguous();
        cu_seqlens_img        = cu_seqlens_img.contiguous();
        cu_seqlens_txt        = cu_seqlens_txt.contiguous();

        Tensor result = net->transformer_blocks.at(idx)->forward(from_torch(hidden_states),
                                                                 from_torch(encoder_hidden_states),
                                                                 from_torch(timestep),
                                                                 from_torch(cu_seqlens_img),
                                                                 from_torch(cu_seqlens_txt),
                                                                 H,
                                                                 W,
                                                                 pag,
                                                                 cfg);

        torch::Tensor output = to_torch(result);
        // Tensor::synchronizeDevice();

        return output;
    }
};
