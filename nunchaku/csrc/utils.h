#pragma once

#include "common.h"
#include "Tensor.h"
#include "kernels/zgemm/zgemm.h"

namespace nunchaku::utils {

void set_cuda_stack_limit(int64_t newval) {
    size_t val = 0;
    checkCUDA(cudaDeviceSetLimit(cudaLimitStackSize, (size_t)newval));
    checkCUDA(cudaDeviceGetLimit(&val, cudaLimitStackSize));
    spdlog::debug("Stack={}", val);
}

void disable_memory_auto_release() {
    int device;
    checkCUDA(cudaGetDevice(&device));
    cudaMemPool_t mempool;
    checkCUDA(cudaDeviceGetDefaultMemPool(&mempool, device));
    uint64_t threshold = UINT64_MAX;
    checkCUDA(cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold));
}

void trim_memory() {
    int device;
    checkCUDA(cudaGetDevice(&device));
    cudaMemPool_t mempool;
    checkCUDA(cudaDeviceGetDefaultMemPool(&mempool, device));
    size_t bytesToKeep = 0;
    checkCUDA(cudaMemPoolTrimTo(mempool, bytesToKeep));
}

void set_faster_i2f_mode(std::string mode) {
    spdlog::info("Set fasteri2f mode to {}", mode);
    kernels::set_faster_i2f_mode(mode);
}

}; // namespace nunchaku::utils
