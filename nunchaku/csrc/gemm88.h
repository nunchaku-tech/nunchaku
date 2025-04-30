#pragma once

#include "interop/torch.h"
#include "Serialization.h"
#include "Linear.h"
#include "debug.h"
#include "module.h"

class QuantizedGEMM88 : public ModuleWrapper<GEMM_W8A8> {
public:
    void init(int64_t in_features, int64_t out_features, bool bias, bool bf16, int8_t deviceId) {
        spdlog::info("Initializing QuantizedGEMM88");

        size_t val = 0;
        checkCUDA(cudaDeviceSetLimit(cudaLimitStackSize, 8192));
        checkCUDA(cudaDeviceGetLimit(&val, cudaLimitStackSize));
        spdlog::debug("Stack={}", val);

        net = std::make_unique<GEMM_W8A8>(
            (int)in_features, (int)out_features, bias, bf16 ? Tensor::BF16 : Tensor::FP16, Device::cuda((int)deviceId));
    }

    torch::Tensor forward(torch::Tensor x) {
        checkModel();

        std::cerr << "QuantizedGEMM88 forward" << std::endl;

        x = x.contiguous();

        Tensor result = net->forward(from_torch(x));

        torch::Tensor output = to_torch(result);
        Tensor::synchronizeDevice();

        return output;
    }
};
