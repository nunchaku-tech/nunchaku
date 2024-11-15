#pragma once

#include "interop/torch.h"
#include "FluxModel.h"
#include "Serialization.h"
#include "debug.h"
#include "Linear.h"

class QuantizedFluxModel { // : public torch::CustomClassHolder {
public:
    void init(bool bf16, int8_t deviceId) {
        spdlog::info("Initializing QuantizedFluxModel");
        net = std::make_unique<FluxModel>(bf16 ? Tensor::BF16 : Tensor::FP16, Device::cuda((int)deviceId));
    }

    void reset() {
        debugContext.reset();
        net.reset();
        Tensor::synchronizeDevice();
        trimMemory();
        Tensor::synchronizeDevice();
    }

    void load(std::string path, bool partial = false) {
        checkModel();

        spdlog::info("{} weights from {}", partial ? "Loading partial" : "Loading", path);
        
        std::shared_ptr<SafeTensors> provider = std::make_shared<SafeTensors>(path);
        net->loadParams(*provider, partial);
        Tensor::synchronizeDevice();

        spdlog::info("Done.");
    }

    torch::Tensor forward(
        torch::Tensor hidden_states, 
        torch::Tensor encoder_hidden_states, 
        torch::Tensor temb, 
        torch::Tensor rotary_emb_img, 
        torch::Tensor rotary_emb_context, 
        torch::Tensor rotary_emb_single,
        const std::vector<torch::Tensor>* controlnet_block_samples = nullptr,
        const std::vector<torch::Tensor>* controlnet_single_block_samples = nullptr) 
    {
        checkModel();

        spdlog::debug("QuantizedFluxModel forward");

        hidden_states = hidden_states.contiguous();
        encoder_hidden_states = encoder_hidden_states.contiguous();
        temb = temb.contiguous();
        rotary_emb_img = rotary_emb_img.contiguous();
        rotary_emb_context = rotary_emb_context.contiguous();
        rotary_emb_single = rotary_emb_single.contiguous();

        // 转换controlnet samples
        std::vector<Tensor> block_samples;
        std::vector<Tensor> single_block_samples;
        
        if (controlnet_block_samples) {
            for (const auto& t : *controlnet_block_samples) {
                block_samples.push_back(from_torch(t));
            }
        }
        
        if (controlnet_single_block_samples) {
            for (const auto& t : *controlnet_single_block_samples) {
                single_block_samples.push_back(from_torch(t));
            }
        }

        Tensor result = net->forward(
            from_torch(hidden_states),
            from_torch(encoder_hidden_states),
            from_torch(temb),
            from_torch(rotary_emb_img),
            from_torch(rotary_emb_context),
            from_torch(rotary_emb_single),
            block_samples.empty() ? nullptr : &block_samples,
            single_block_samples.empty() ? nullptr : &single_block_samples
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

    void disableMemoryAutoRelease() {
        int device;
        checkCUDA(cudaGetDevice(&device));
        cudaMemPool_t mempool;
        checkCUDA(cudaDeviceGetDefaultMemPool(&mempool, device));
        uint64_t threshold = UINT64_MAX;
        checkCUDA(cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold));
    }

    void trimMemory() {
        int device;
        checkCUDA(cudaGetDevice(&device));
        cudaMemPool_t mempool;
        checkCUDA(cudaDeviceGetDefaultMemPool(&mempool, device));
        size_t bytesToKeep = 0;
        checkCUDA(cudaMemPoolTrimTo(mempool, bytesToKeep));
    }

    void startDebug() {
        debugContext = std::make_unique<DebugContext>();
    }
    void stopDebug() {
        debugContext.reset();
    }

    auto getDebugResults() {
        // c10::Dict<std::string, torch::Tensor> result;
        std::map<std::string, torch::Tensor> result;

        if (debugContext) {
            for (auto &&[key, value] : debugContext->tensors) {
                // result.insert(key, to_torch(value));
                result[key] = to_torch(value);
            }
        }
        
        return result;
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
                size_t skip_size = static_cast<size_t>(skipRanks / 16);
                for (size_t i = 0; i < skip_size; i++) {
                    m->lora_scales[i] = 1.0f;
                }
                for (size_t i = skip_size; i < m->lora_scales.size(); i++) {
                    m->lora_scales[i] = scale;
                }
            }
        });
    }

    void forceFP16Attention(bool enable) {
        Attention::setForceFP16(net.get(), enable);
    }


private:
    void checkModel() {
        if (!net) {
            throw std::runtime_error("Model not initialized");
        }
    }

private:
    std::unique_ptr<FluxModel> net;
    std::unique_ptr<DebugContext> debugContext;
};