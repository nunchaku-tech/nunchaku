#pragma once

#include "interop/torch.h"
#include "FluxModel.h"
#include "Serialization.h"
#include "debug.h"
#include "Linear.h"
#include "module.h"
#include "OminiFluxModel.h"

// Wrapper class for the FluxModel, providing a PyTorch C++ API (torch::CustomClassHolder style).
// This class is likely for an earlier or different version of the Flux model compared to OminiFluxModel.
class QuantizedFluxModel : public ModuleWrapper<FluxModel> { // : public torch::CustomClassHolder {
public:
    // Initializes the QuantizedFluxModel.
    // - use_fp4: Enables 4-bit quantization.
    // - offload: Enables layer offloading to CPU.
    // - bf16: Uses bfloat16 precision if true, otherwise float16.
    // - deviceId: The CUDA device ID to run the model on.
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
        net = std::make_unique<FluxModel>(
            use_fp4, offload, bf16 ? Tensor::BF16 : Tensor::FP16, Device::cuda((int)deviceId));
    }

    // Returns true if the model is configured to use bfloat16 precision.
    bool isBF16() {
        checkModel();
        return net->dtype == Tensor::BF16;
    }
    pybind11::function residual_callback; // Stores the Python callback function.

    // Sets or clears the residual callback function.
    // This callback allows Python code to interact with the C++ forward pass,
    // typically for injecting residuals (e.g., for PuLID-like mechanisms).
    void set_residual_callback(pybind11::function callback) {
        pybind11::gil_scoped_acquire gil; // Acquire the Python Global Interpreter Lock
        if (!callback || callback.is_none()) {
            residual_callback = pybind11::function(); // Clear if None
            if (net) {
                net->set_residual_callback(nullptr); // Inform the C++ model
            }
            return;
        }
        residual_callback = std::move(callback);
        if (net) {
            pybind11::object cb = residual_callback;
            // Lambda to bridge C++ Tensor to PyTorch Tensor for the callback
            net->set_residual_callback([cb](const Tensor &x) -> Tensor {
                pybind11::gil_scoped_acquire gil;
                torch::Tensor torch_x   = to_torch(x); // Convert Nunchaku Tensor to PyTorch Tensor
                pybind11::object result = cb(torch_x);  // Call Python function
                torch::Tensor torch_y   = result.cast<torch::Tensor>();
                Tensor y                = from_torch(torch_y); // Convert result back to Nunchaku Tensor
                return y;
            });
        } else {
        }
    }

    // Main forward pass for the QuantizedFluxModel.
    // Arguments are PyTorch tensors, which are converted to Nunchaku tensors internally.
    torch::Tensor forward(torch::Tensor hidden_states,
                          torch::Tensor encoder_hidden_states,
                          torch::Tensor temb,
                          torch::Tensor rotary_emb_img,
                          torch::Tensor rotary_emb_context,
                          torch::Tensor rotary_emb_single,
                          std::optional<torch::Tensor> controlnet_block_samples        = std::nullopt,
                          std::optional<torch::Tensor> controlnet_single_block_samples = std::nullopt,
                          bool skip_first_layer                                        = false) {
        checkModel(); // Ensure the underlying C++ model is initialized
        CUDADeviceContext ctx(deviceId); // Set the CUDA device context

        spdlog::debug("QuantizedFluxModel forward");

        // Ensure input tensors are contiguous in memory for performance
        hidden_states         = hidden_states.contiguous();
        encoder_hidden_states = encoder_hidden_states.contiguous();
        temb                  = temb.contiguous();
        rotary_emb_img        = rotary_emb_img.contiguous();
        rotary_emb_context    = rotary_emb_context.contiguous();
        rotary_emb_single     = rotary_emb_single.contiguous();

        Tensor result = net->forward(
            from_torch(hidden_states),
            from_torch(encoder_hidden_states),
            from_torch(temb),
            from_torch(rotary_emb_img),
            from_torch(rotary_emb_context),
            from_torch(rotary_emb_single),
            controlnet_block_samples.has_value() ? from_torch(controlnet_block_samples.value().contiguous()) : Tensor{},
            controlnet_single_block_samples.has_value()
                ? from_torch(controlnet_single_block_samples.value().contiguous())
                : Tensor{},
            skip_first_layer);

        torch::Tensor output = to_torch(result);
        Tensor::synchronizeDevice();

        return output;
    }

    // Forward pass for a single specified layer of the QuantizedFluxModel.
    // `idx`: The index of the layer to execute.
    std::tuple<torch::Tensor, torch::Tensor>
    forward_layer(int64_t idx,
                  torch::Tensor hidden_states,
                  torch::Tensor encoder_hidden_states,
                  torch::Tensor temb,
                  torch::Tensor rotary_emb_img,
                  torch::Tensor rotary_emb_context,
                  std::optional<torch::Tensor> controlnet_block_samples        = std::nullopt,
                  std::optional<torch::Tensor> controlnet_single_block_samples = std::nullopt) {
        CUDADeviceContext ctx(deviceId);

        spdlog::debug("QuantizedFluxModel forward_layer {}", idx);

        hidden_states         = hidden_states.contiguous();
        encoder_hidden_states = encoder_hidden_states.contiguous();
        temb                  = temb.contiguous();
        rotary_emb_img        = rotary_emb_img.contiguous();
        rotary_emb_context    = rotary_emb_context.contiguous();

        auto &&[hidden_states_, encoder_hidden_states_] = net->forward_layer(
            idx,
            from_torch(hidden_states),
            from_torch(encoder_hidden_states),
            from_torch(temb),
            from_torch(rotary_emb_img),
            from_torch(rotary_emb_context),
            controlnet_block_samples.has_value() ? from_torch(controlnet_block_samples.value().contiguous()) : Tensor{},
            controlnet_single_block_samples.has_value()
                ? from_torch(controlnet_single_block_samples.value().contiguous())
                : Tensor{});

        hidden_states         = to_torch(hidden_states_);
        encoder_hidden_states = to_torch(encoder_hidden_states_);
        Tensor::synchronizeDevice();

        return {hidden_states, encoder_hidden_states};
    }

    // Forward pass for a single layer of the `single_transformer_blocks` part of the model.
    // `idx`: The index within the `single_transformer_blocks` vector.
    torch::Tensor forward_single_layer(int64_t idx,
                                       torch::Tensor hidden_states,
                                       torch::Tensor temb,
                                       torch::Tensor rotary_emb_single) {
        CUDADeviceContext ctx(deviceId);

        spdlog::debug("QuantizedFluxModel forward_single_layer {}", idx);

        hidden_states     = hidden_states.contiguous();
        temb              = temb.contiguous();
        rotary_emb_single = rotary_emb_single.contiguous();

        Tensor result = net->single_transformer_blocks.at(idx)->forward(
            from_torch(hidden_states), from_torch(temb), from_torch(rotary_emb_single));

        hidden_states = to_torch(result);
        Tensor::synchronizeDevice();

        return hidden_states;
    }

    // Exposes the `forward` method of the `norm1` (AdaLayerNormZero) submodule
    // of a specific transformer block. This is useful for techniques like TeaCache
    // that might need direct access to the output of this normalization layer.
    // `idx`: The index of the transformer block.
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    norm_one_forward(int64_t idx, torch::Tensor hidden_states, torch::Tensor temb) {
        AdaLayerNormZero::Output result =
            net->transformer_blocks.at(idx)->norm1.forward(from_torch(hidden_states), from_torch(temb));
        return {to_torch(result.x),
                to_torch(result.gate_msa),
                to_torch(result.shift_mlp),
                to_torch(result.scale_mlp),
                to_torch(result.gate_mlp)};
    }

    // Sets the LoRA (Low-Rank Adaptation) scaling factor for quantized layers.
    // `skipRanks`: Number of initial ranks to skip applying the LoRA scale to (must be multiple of 16 for W4A4).
    // `scale`: The scaling factor to apply to LoRA weights.
    // This is typically called after LoRA weights have been loaded into the model.
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

// Wrapper class for the OminiFluxModel, providing a PyTorch C++ API.
// This is the primary model class used for the OminiFlux architecture in Nunchaku.
class QuantizedOminiFluxModel : public ModuleWrapper<OminiFluxModel> {
public:
    // Initializes the QuantizedOminiFluxModel.
    // - use_fp4: Enables 4-bit quantization.
    // - offload: Enables layer offloading to CPU for memory saving.
    // - bf16: Uses bfloat16 precision if true, otherwise float16.
    // - deviceId: The CUDA device ID to run the model on.
    void init(bool use_fp4, bool offload, bool bf16, int8_t deviceId) {
        spdlog::info("Initializing QuantizedOminiFluxModel on device {}", deviceId);
        if (!bf16) {
            spdlog::info("Use FP16 model");
        }
        if (offload) {
            spdlog::info("Layer offloading enabled");
        }
        ModuleWrapper::init(deviceId);

        CUDADeviceContext ctx(this->deviceId);
        net = std::make_unique<OminiFluxModel>(
            use_fp4, offload, bf16 ? Tensor::BF16 : Tensor::FP16, Device::cuda((int)deviceId));
    }

    // Returns true if the model is configured to use bfloat16 precision.
    bool isBF16() {
        checkModel();
        return net->dtype == Tensor::BF16;
    }
    pybind11::function residual_callback; // Stores the Python callback function.

    // Sets or clears the residual callback function.
    // Allows Python code to inject residuals during the C++ forward pass (e.g., for PuLID).
    void set_residual_callback(pybind11::function callback) {
        pybind11::gil_scoped_acquire gil; // Acquire Python GIL
        if (!callback || callback.is_none()) {
            residual_callback = pybind11::function(); // Clear if None
            if (net) {
                net->set_residual_callback(nullptr); // Inform C++ model
            }
            return;
        }
        residual_callback = std::move(callback);
        if (net) {
            pybind11::object cb = residual_callback;
            // Lambda to bridge C++ Tensor to PyTorch Tensor for the callback
            net->set_residual_callback([cb](const Tensor &x) -> Tensor {
                pybind11::gil_scoped_acquire gil;
                torch::Tensor torch_x   = to_torch(x); // Nunchaku Tensor to PyTorch Tensor
                pybind11::object result = cb(torch_x);  // Call Python function
                torch::Tensor torch_y   = result.cast<torch::Tensor>();
                Tensor y                = from_torch(torch_y); // PyTorch Tensor back to Nunchaku Tensor
                return y;
            });
        } else {
        }
    }

    // Main forward pass for the QuantizedOminiFluxModel.
    // All tensor arguments are PyTorch tensors, converted internally.
    torch::Tensor forward(torch::Tensor hidden_states,         // Image latents
                          torch::Tensor cond_hidden_states,    // Conditional latents
                          torch::Tensor encoder_hidden_states, // Text encoder latents
                          torch::Tensor temb,                  // Time embeddings
                          torch::Tensor cond_temb,             // Conditional time embeddings
                          torch::Tensor rotary_emb_img,        // Rotary embeddings for image features
                          torch::Tensor rotary_emb_context,    // Rotary embeddings for context/text features
                          torch::Tensor rotary_emb_single,     // Rotary embeddings for single (concatenated) features
                          torch::Tensor rotary_emb_cond,       // Rotary embeddings for conditional features
                          std::optional<torch::Tensor> controlnet_block_samples        = std::nullopt, // Optional ControlNet features for joint blocks
                          std::optional<torch::Tensor> controlnet_single_block_samples = std::nullopt, // Optional ControlNet features for single blocks
                          bool skip_first_layer                                        = false) { // Option to skip the first transformer layer
        checkModel(); // Ensure C++ model is initialized
        CUDADeviceContext ctx(deviceId); // Set CUDA device context

        spdlog::debug("QuantizedOminiFluxModel forward");

        // Ensure input tensors are contiguous
        hidden_states         = hidden_states.contiguous();
        cond_hidden_states    = cond_hidden_states.contiguous();
        encoder_hidden_states = encoder_hidden_states.contiguous();
        temb                  = temb.contiguous();
        cond_temb             = cond_temb.contiguous();
        rotary_emb_img        = rotary_emb_img.contiguous();
        rotary_emb_context    = rotary_emb_context.contiguous();
        rotary_emb_single     = rotary_emb_single.contiguous();
        rotary_emb_cond       = rotary_emb_cond.contiguous();

        Tensor result = net->forward(
            from_torch(hidden_states),
            from_torch(cond_hidden_states),
            from_torch(encoder_hidden_states),
            from_torch(temb),
            from_torch(cond_temb),
            from_torch(rotary_emb_img),
            from_torch(rotary_emb_context),
            from_torch(rotary_emb_single),
            from_torch(rotary_emb_cond),
            controlnet_block_samples.has_value() ? from_torch(controlnet_block_samples.value().contiguous()) : Tensor{},
            controlnet_single_block_samples.has_value()
                ? from_torch(controlnet_single_block_samples.value().contiguous())
                : Tensor{},
            skip_first_layer);

        torch::Tensor output = to_torch(result);
        Tensor::synchronizeDevice();

        return output;
    }

    // Forward pass for a single specified layer of the QuantizedOminiFluxModel.
    // `idx`: The index of the layer to execute.
    // Returns a tuple: (hidden_states, cond_hidden_states, encoder_hidden_states) after the layer.
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    forward_layer(int64_t idx,
                  torch::Tensor hidden_states,
                  torch::Tensor cond_hidden_states,
                  torch::Tensor encoder_hidden_states,
                  torch::Tensor temb,
                  torch::Tensor cond_temb,
                  torch::Tensor rotary_emb_img,
                  torch::Tensor rotary_emb_context,
                  torch::Tensor rotary_emb_cond,
                  std::optional<torch::Tensor> controlnet_block_samples        = std::nullopt,
                  std::optional<torch::Tensor> controlnet_single_block_samples = std::nullopt) {
        CUDADeviceContext ctx(deviceId);

        spdlog::debug("QuantizedOminiFluxModel forward_layer {}", idx);

        hidden_states         = hidden_states.contiguous();
        cond_hidden_states    = cond_hidden_states.contiguous();
        encoder_hidden_states = encoder_hidden_states.contiguous();
        temb                  = temb.contiguous();
        cond_temb             = cond_temb.contiguous();
        rotary_emb_img        = rotary_emb_img.contiguous();
        rotary_emb_context    = rotary_emb_context.contiguous();
        rotary_emb_cond       = rotary_emb_cond.contiguous();
        auto &&[hidden_states_, cond_hidden_states_, encoder_hidden_states_] = net->forward_layer(
            idx,
            from_torch(hidden_states),
            from_torch(cond_hidden_states),
            from_torch(encoder_hidden_states),
            from_torch(temb),
            from_torch(cond_temb),
            from_torch(rotary_emb_img),
            from_torch(rotary_emb_context),
            from_torch(rotary_emb_cond),
            controlnet_block_samples.has_value() ? from_torch(controlnet_block_samples.value().contiguous()) : Tensor{},
            controlnet_single_block_samples.has_value()
                ? from_torch(controlnet_single_block_samples.value().contiguous())
                : Tensor{});

        hidden_states         = to_torch(hidden_states_);
        cond_hidden_states    = to_torch(cond_hidden_states_);
        encoder_hidden_states = to_torch(encoder_hidden_states_);
        Tensor::synchronizeDevice();

        return {hidden_states, cond_hidden_states, encoder_hidden_states};
    }

    // Exposes the `forward` method of the `norm1` (OminiAdaLayerNormZero) submodule
    // of a specific OminiJointTransformerBlock. This is useful for techniques like TeaCache.
    // `idx`: The index of the joint transformer block.
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    norm_one_forward(int64_t idx, torch::Tensor hidden_states, torch::Tensor temb) {
        OminiAdaLayerNormZero::Output result =
            net->transformer_blocks.at(idx)->norm1.forward(from_torch(hidden_states), from_torch(temb));
        return {to_torch(result.x),
                to_torch(result.gate_msa),
                to_torch(result.shift_mlp),
                to_torch(result.scale_mlp),
                to_torch(result.gate_mlp)};
    }

    // Sets the LoRA (Low-Rank Adaptation) scaling factor for quantized layers.
    // `skipRanks`: Number of initial ranks to skip applying LoRA scale (must be multiple of 16 for W4A4).
    // `scale`: The scaling factor for LoRA weights.
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
            net->setAttentionImpl(OminiAttentionImpl::FlashAttention2);
        } else if (name == "nunchaku-fp16") {
            net->setAttentionImpl(OminiAttentionImpl::NunchakuFP16);
        } else {
            throw std::invalid_argument(spdlog::fmt_lib::format("Invalid attention implementation {}", name));
        }
    }
};
