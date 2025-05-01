#include "zgemm.h"
#include "attention.cuh"

#ifndef M_LOG2E
#define M_LOG2E 1.4426950408889634074
#endif

namespace nunchaku::kernels {

void attention_fp16(Tensor q, // packed [Batch, Head, TokensQ, HEAD_DIM]
                    Tensor k, // packed [Batch, Head, TokensKV, HEAD_DIM]
                    Tensor v, // packed [Batch, Head, TokensKV, HEAD_DIM]
                    Tensor o, // linear [Batch, TokensQ, Head * HEAD_DIM]
                    float scale) {
    int sizeBatch   = q.shape[0];
    int numHeads    = q.shape[1];
    int numTokensQ  = q.shape[2];
    int headDim     = q.shape[3];
    int numTokensKV = k.shape[2];

    assert(o.ndims() == 3);
    assert(o.shape[0] == sizeBatch);
    assert(o.shape[1] == numTokensQ);
    assert(o.shape[2] == numHeads * headDim);

    spdlog::trace("attention_fp16: B={} H={} NQ={} NK={}", sizeBatch, numHeads, numTokensQ, numTokensKV);
    spdlog::trace("q at {}", q.data_ptr());
    spdlog::trace("k at {}", k.data_ptr());
    spdlog::trace("v at {}", v.data_ptr());
    spdlog::trace("o at {}", o.data_ptr());
    spdlog::trace("scale={}", scale);

    dispatchBool(o.scalar_type() == Tensor::BF16, [&]<bool bf16out>() {
#ifndef __INTELLISENSE__
        using Attention = typename nunchaku::kernels::Attention<AttentionFP16Config<bf16out>>;
#else
        using Attention = typename nunchaku::kernels::Attention<AttentionFP16Config<true>>;
#endif
        using GEMM = typename Attention::GEMM;

        assert(isTypeMatch<typename Attention::half_t>(q.scalar_type()));
        assert(isTypeMatch<typename Attention::half_t>(k.scalar_type()));
        assert(isTypeMatch<typename Attention::half_t>(v.scalar_type()));
        assert(isTypeMatch<typename Attention::epilogue_half_t>(o.scalar_type()));

        int shmem = 0;

        // we use exp2 instead of exp in the kernel
        scale *= M_LOG2E;

        assert(numTokensQ % Attention::BLOCK_M == 0);
        assert(numTokensKV % Attention::WARP_K == 0);
        assert(headDim == Attention::HEAD_DIM);

        auto launch = [&]<typename Epilogue>(Epilogue::Arguments args) {
            dim3 grid(numTokensQ / Attention::BLOCK_M, numHeads, sizeBatch);
            using packed_q_t = typename Attention::packed_q_t;
            using packed_k_t = typename Attention::packed_k_t;
            using packed_v_t = typename Attention::packed_v_t;

            auto func = invoke_kernel<typename Attention::attention_fp16_kernel<Epilogue>,
                                      const packed_q_t *,
                                      const packed_k_t *,
                                      const packed_v_t *,
                                      float,
                                      int,
                                      int,
                                      typename Epilogue::Arguments,
                                      bool>;

            shmem = std::max(shmem, Attention::template attention_fp16_kernel<Epilogue>::SHMEM_SIZE);

            if (shmem >= 24 * 1024) {
                checkCUDA(cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem));
            }

            func<<<grid, GEMM::WARP_SIZE * GEMM::NUM_WARPS, shmem, getCurrentCUDAStream()>>>(q.data_ptr<packed_q_t>(),
                                                                                             k.data_ptr<packed_k_t>(),
                                                                                             v.data_ptr<packed_v_t>(),
                                                                                             scale,
                                                                                             numTokensQ,
                                                                                             numTokensKV,
                                                                                             args,
                                                                                             false);
            checkCUDA(cudaGetLastError());
        };

        launch.template operator()<typename GEMM::EpilogueDefault>(typename GEMM::EpilogueDefault::Arguments{
            .out     = o.data_ptr<typename GEMM::half_t>(),
            .actualM = sizeBatch * numTokensQ,
            .actualN = numHeads * headDim,
        });
    });
}

}; // namespace nunchaku::kernels
