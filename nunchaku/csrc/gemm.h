#pragma once

#include "interop/torch.h"
#include "Serialization.h"
#include "Linear.h"
#include "debug.h"
#include "module.h"

class QuantizedGEMM : public ModuleWrapper<GEMM_W4A4> {
public:
    void init(int64_t in_features, int64_t out_features, bool bias, bool use_fp4, bool bf16, int8_t deviceId) {
        spdlog::info("Initializing QuantizedGEMM");

        size_t val = 0;
        checkCUDA(cudaDeviceSetLimit(cudaLimitStackSize, 8192));
        checkCUDA(cudaDeviceGetLimit(&val, cudaLimitStackSize));
        spdlog::debug("Stack={}", val);

        net = std::make_unique<GEMM_W4A4>((int)in_features,
                                          (int)out_features,
                                          bias,
                                          use_fp4,
                                          bf16 ? Tensor::BF16 : Tensor::FP16,
                                          Device::cuda((int)deviceId));
    }

    torch::Tensor forward(torch::Tensor x) {
        checkModel();

        std::cerr << "QuantizedGEMM forward" << std::endl;

        x = x.contiguous();

        Tensor result = net->forward(from_torch(x));

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

        constexpr int BLOCK_M      = 256;
        constexpr int WARP_K       = 64;
        constexpr int NUM_WARPS    = 8;
        constexpr int WARP_M_TILES = 2;
        constexpr int WARP_SIZE    = 32;

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

    void quantize(torch::Tensor x, bool fuse_glu) {
        checkModel();

        spdlog::debug("QuantizedGEMM quantize");

        x = x.contiguous();

        auto qout = net->quantize(from_torch(x), fuse_glu);

        Tensor act      = qout.act.copy(Device::cpu());
        Tensor ascales  = qout.ascales.copy(Device::cpu());
        Tensor lora_act = qout.lora_act.copy(Device::cpu());

        Tensor::synchronizeDevice();

        spdlog::debug("act = {}", dumpTensorINT4(act));
        spdlog::debug("ascales = {}", dumpTensorBF16(ascales));
    }
};
