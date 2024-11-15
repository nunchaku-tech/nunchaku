#pragma once

#include "interop/torch.h"
#include "Serialization.h"
#include "Linear.h"
#include "debug.h"

#include "kernels/gemm_w4a4.h"
#include "kernels/awq/gemv_awq.h"

class QuantizedGEMM { // : public torch::CustomClassHolder {
public:
    void init(int64_t in_features, int64_t out_features, bool bias, bool bf16, int8_t deviceId) {
        spdlog::info("Initializing QuantizedGEMM");
        
        size_t val = 0;
        checkCUDA(cudaDeviceSetLimit(cudaLimitStackSize, 8192));
        checkCUDA(cudaDeviceGetLimit(&val, cudaLimitStackSize));
        spdlog::debug("Stack={}", val);

        net = std::make_unique<GEMM_W4A4>((int)in_features, (int)out_features, bias, bf16 ? Tensor::BF16 : Tensor::FP16, Device::cuda((int)deviceId));
    }

    void reset() {
        debugContext.reset();
        net.reset();
        Tensor::synchronizeDevice();
    }

    void load(std::string path) {
        checkModel();

        spdlog::info("Loading weights from {}", path);
        std::shared_ptr<SafeTensors> provider = std::make_shared<SafeTensors>(path);
        net->loadParams(*provider);
        Tensor::synchronizeDevice();
    }

    torch::Tensor forward(torch::Tensor x) {
        checkModel();

        std::cerr << "QuantizedGEMM forward" << std::endl;

        x = x.contiguous();

        Tensor result = std::get<Tensor>(net->forward(
            from_torch(x)
        ));
        
        torch::Tensor output = to_torch(result);
        Tensor::synchronizeDevice();

        return output;
    }

    std::string dumpTensorBF16(Tensor x) {
        std::stringstream ss;
        for (int i = 0; i < 256; i++) {
            ss << spdlog::fmt_lib::format("{:.3f} ", (float)(x.data_ptr<__nv_bfloat16>()[i]));
        }
        ss << std::endl;
        return ss.str();
    }

    std::string dumpTensorINT4(Tensor x) {
        using spdlog::fmt_lib::format;

        const int M = x.shape[0];
        const int K = x.shape[1] * 2;
        
        assert(x.dtype() == Tensor::INT8);

        // activation: row major, [M / BLOCK_M, K / WARP_K, NUM_WARPS, WARP_M_TILES, WARP_SIZE] of packed_act_t (uint4)

        constexpr int BLOCK_M = 256;
        constexpr int WARP_K = 64;
        constexpr int NUM_WARPS = 8;
        constexpr int WARP_M_TILES = 2;
        constexpr int WARP_SIZE = 32;

        std::stringstream ss;
        for (int bm = 0; bm < M / BLOCK_M; bm++) {
            for (int bn = 0; bn < K / WARP_K; bn++) {
                for (int warpId = 0; warpId < NUM_WARPS; warpId++) {
                    ss << format("[bm={},bn={},warp={}] ", bm, bn, warpId);
                    const int offset = ((bm * (K / WARP_K) + bn) * NUM_WARPS + warpId) * WARP_M_TILES * WARP_SIZE * 4;

                    for (int i = 0; i < 16; i++) {
                        assert(static_cast<size_t>(offset + i) < x.numel() / 4);
                        uint32_t val = x.data_ptr<uint32_t>()[offset + i];
                        ss << "{";
                        for (int j = 0; j < 8; j++) {
                            int i4val = (val >> (j * 4)) & 0xf;
                            if (i4val & 0x8) {
                                i4val = -((~i4val & 0x7) + 1);
                            }
                            ss << format("{} ", i4val);
                        }
                        ss << format("}} {:x} ", val);
                    }
                    ss << std::endl;
                }
            }
        }
        
        ss << std::endl;
        return ss.str();
    }

    void quantize(torch::Tensor x) {
        checkModel();

        spdlog::debug("QuantizedGEMM quantize");

        x = x.contiguous();

        auto qout = net->quantize(
            from_torch(x)
        );
        
        Tensor act = qout.act.copy(Device::cpu());
        Tensor ascales = qout.ascales.copy(Device::cpu());
        Tensor lora_act = qout.lora_act.copy(Device::cpu());

        Tensor::synchronizeDevice();

        spdlog::debug("act = {}", dumpTensorINT4(act));
        spdlog::debug("ascales = {}", dumpTensorBF16(ascales));
    }

    
    void gemm(
        c10::optional<torch::Tensor> act,          // packed act [M, K / 2]
        c10::optional<torch::Tensor> wgt,          // packed act [N, K / 2]
        c10::optional<torch::Tensor> out,          // linear     [M, N]
        c10::optional<torch::Tensor> qout,         // packed act [M, N / 2]
        c10::optional<torch::Tensor> ascales,      // packed as  [K / 64, M]
        c10::optional<torch::Tensor> wscales,      // packed ws  [K / 64, N]
        c10::optional<torch::Tensor> oscales,      // packed as  [N / 64, M]
        c10::optional<torch::Tensor> poolout,      // linear     [M / PoolSize, N]
        c10::optional<torch::Tensor> lora_act_in,  // packed lora_act [M, R]
        c10::optional<torch::Tensor> lora_up,      // packed lora_wgt [N, R]
        c10::optional<torch::Tensor> lora_down,    // packed lora_wgt [N, R]
        c10::optional<torch::Tensor> lora_act_out, // packed lora_act [M, R]
        c10::optional<torch::Tensor> norm_q,       // linear     [HEAD_DIM]
        c10::optional<torch::Tensor> norm_k,       // linear     [HEAD_DIM]
        c10::optional<torch::Tensor> rotary_emb,   // linear     [M, HEAD_DIM / 2, 2, 2]
        c10::optional<torch::Tensor> bias,         // packed ws  [N]
        c10::optional<torch::Tensor> smooth_factor, // packed ws  [N], for quantization of the next layer
        bool act_unsigned,
        std::vector<float> lora_scales
    ) {
        std::cerr << "running gemm_w4a4: " << std::endl;

        auto getTensor = [](c10::optional<torch::Tensor> &t) {
            Tensor ret = t.has_value() ? from_torch(t.value()) : Tensor{};
            if (ret.valid()) {
                std::cerr << "  " << ret.shape.str() << std::endl;
            } else {
                std::cerr << "  <invalid>" << std::endl;
            }
            return ret;
        };
        gemm_w4a4(
            getTensor(act          ),
            getTensor(wgt          ),
            getTensor(out          ),
            getTensor(qout         ),
            getTensor(ascales      ),
            getTensor(wscales      ),
            getTensor(oscales      ),
            getTensor(poolout      ),
            getTensor(lora_act_in  ),
            getTensor(lora_up      ),
            getTensor(lora_down    ),
            getTensor(lora_act_out ),
            getTensor(norm_q       ),
            getTensor(norm_k       ),
            getTensor(rotary_emb   ),
            getTensor(bias         ),
            getTensor(smooth_factor),
            act_unsigned,
            lora_scales
        );
        Tensor::synchronizeDevice();
    }

    torch::Tensor gemv_awq(
        torch::Tensor _in_feats,
        torch::Tensor _kernel,
        torch::Tensor _scaling_factors,
        torch::Tensor _zeros,
        int64_t m,
        int64_t n,
        int64_t k,
        int64_t group_size)
    {
        Tensor result = ::gemv_awq(
            from_torch(_in_feats.contiguous()),
            from_torch(_kernel.contiguous()),
            from_torch(_scaling_factors.contiguous()),
            from_torch(_zeros.contiguous()),
            (int)m, 
            (int)n, 
            (int)k, 
            (int)group_size
        );

        torch::Tensor output = to_torch(result);
        Tensor::synchronizeDevice();

        return output;
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

private:
    void checkModel() {
        if (!net) {
            throw std::runtime_error("Model not initialized");
        }
    }

private:
    std::unique_ptr<GEMM_W4A4> net;
    std::unique_ptr<DebugContext> debugContext;
};