#include "zgemm.h"
#include "gemm_w4a4.cuh"
#include "epilogues.cuh"

namespace nunchaku::kernels {

void test_rmsnorm_rope(Tensor input, Tensor output, Tensor norm_q, Tensor norm_k, Tensor rotary_emb) {
    assert(input.ndims() == 2);
    const int M = input.shape[0];
    const int N = input.shape[1];
    assert(input.shape.dataExtent == output.shape.dataExtent);
    assert(input.scalar_type() == Tensor::FP16);

    using GEMM     = Epilogues<GEMMConfig_W4A4_FP16>;
    using Epilogue = GEMM::EpilogueRMSNormRope;

    assert(M % GEMM::BLOCK_M == 0);
    assert(N % GEMM::BLOCK_N == 0);

    using kernel = typename GEMM::test_epilogue_kernel<Epilogue>;

    auto func = invoke_kernel<kernel, typename kernel::Arguments>;

    checkCUDA(cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, kernel::SHMEM_SIZE));

    dim3 grid(M / GEMM::BLOCK_M, N / GEMM::BLOCK_N);

    func<<<grid, GEMM::WARP_SIZE * GEMM::NUM_WARPS, kernel::SHMEM_SIZE, getCurrentCUDAStream()>>>(
        typename kernel::Arguments{.input        = input.data_ptr<GEMM::half_t>(),
                                   .output       = output.data_ptr<GEMM::half_t>(),
                                   .M            = M,
                                   .N            = N,
                                   .actualM      = M,
                                   .actualN      = N,
                                   .argsEpilogue = typename Epilogue::Arguments{
                                       .rotary_emb       = rotary_emb.data_ptr<typename Epilogue::packed_rotemb_t>(),
                                       .rmsnorm_weight_q = norm_q.data_ptr<GEMM::half_t>(),
                                       .rmsnorm_weight_k = norm_k.data_ptr<GEMM::half_t>(),
                                       .epsilon          = 1e-6,
                                   }});
    checkCUDA(cudaGetLastError());
}

void test_pack_qkv(Tensor input, Tensor out_q, Tensor out_k, Tensor out_v, int numTokens) {
    assert(input.ndims() == 2);
    const int M = input.shape[0];
    const int N = input.shape[1];
    assert(input.scalar_type() == Tensor::FP16);

    Tensor output = Tensor::empty_like(input);

    using GEMM     = Epilogues<GEMMConfig_W4A4_FP16>;
    using Epilogue = GEMM::EpiloguePackQKV;

    assert(M % GEMM::BLOCK_M == 0);
    assert(N % GEMM::BLOCK_N == 0);

    using kernel = typename GEMM::test_epilogue_kernel<Epilogue>;

    auto func = invoke_kernel<kernel, typename kernel::Arguments>;

    checkCUDA(cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, kernel::SHMEM_SIZE));

    dim3 grid(M / GEMM::BLOCK_M, N / GEMM::BLOCK_N);

    func<<<grid, GEMM::WARP_SIZE * GEMM::NUM_WARPS, kernel::SHMEM_SIZE, getCurrentCUDAStream()>>>(
        typename kernel::Arguments{
            .input        = input.data_ptr<GEMM::half_t>(),
            .output       = output.data_ptr<GEMM::half_t>(),
            .M            = M,
            .N            = N,
            .actualM      = M,
            .actualN      = N,
            .argsEpilogue = typename Epilogue::Arguments{
                .out_q   = out_q.data_ptr<typename Epilogue::packed_qkv_t>(),
                .out_k   = out_k.data_ptr<typename Epilogue::packed_qkv_t>(),
                .out_v   = out_v.data_ptr<typename Epilogue::packed_qkv_t>(),
                .actualM = numTokens,
                .strideHead_q =
                    int(out_q.stride(1) * out_q.scalar_size() / sizeof(GEMM::EpiloguePackQKV::packed_qkv_t)),
                .strideHead_k =
                    int(out_k.stride(1) * out_k.scalar_size() / sizeof(GEMM::EpiloguePackQKV::packed_qkv_t)),
                .strideHead_v =
                    int(out_v.stride(1) * out_v.scalar_size() / sizeof(GEMM::EpiloguePackQKV::packed_qkv_t)),
            }});
    checkCUDA(cudaGetLastError());
}

}; // namespace nunchaku::kernels
