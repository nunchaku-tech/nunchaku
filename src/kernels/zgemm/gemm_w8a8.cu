#include "zgemm.h"
#include "gemm_w8a8.cuh"

namespace nunchaku::kernels {

void quantize_w8a8_act(Tensor input, Tensor output, Tensor oscales, bool fuse_glu) {
    using GEMM = GEMM_W8A8;

    int M = input.numel() / input.shape[-1];
    int K = input.shape[-1];

    assert(output.dtype() == Tensor::INT8);
    assert(output.numel() / output.shape[-1] == M);
    assert(output.shape[-1] == fuse_glu ? K / 2 : K);

    assert(isTypeMatch<GEMM::half_t>(oscales.dtype()));
    assert(oscales.numel() == M * 1);

    auto launch = [&]<bool FUSE_GLU>() {
        using kernel = GEMM::quantize_w8a8_act_kernel<FUSE_GLU>;

        assert(kernel::check(M, K));
        dim3 grid  = kernel::gridSize(M, K);
        dim3 block = kernel::blockSize(M, K);

        auto func =
            invoke_kernel<kernel, const GEMM::half_t *, GEMM::packed_act_t *, GEMM::packed_ascale_t *, int, bool>;

        checkCUDA(cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, 92160));

        func<<<grid, block, kernel::smemSize(M, K)>>>(input.data_ptr<GEMM::half_t>(),
                                                      output.data_ptr<GEMM::packed_act_t>(),
                                                      oscales.data_ptr<GEMM::packed_ascale_t>(),
                                                      K,
                                                      false);
        checkCUDA(cudaGetLastError());
    };

    if (fuse_glu) {
        launch.template operator()<true>();
    } else {
        launch.template operator()<false>();
    }
}

void gemm_w8a8(Tensor act,     // [M, K]
               Tensor wgt,     // [N, K]
               Tensor out,     // [M, N]
               Tensor ascales, // [1, M]
               Tensor wscales, // [1, N]
               Tensor bias) {
    using GEMM = GEMM_W8A8;

    int M = act.numel() / act.shape[-1];
    int N = wgt.shape[0];
    int K = act.shape[-1];
    assert(K == wgt.shape[1]);

    int actualM = 0;
    int actualN = 0;
    if (out.valid()) {
        actualM = out.numel() / out.shape[-1];
        actualN = out.shape[-1];

        assert(actualM <= M && M - actualM < GEMM::BLOCK_M);
        assert(actualN <= N && N - actualN < GEMM::BLOCK_N);
    }

    auto launch = [&]<typename Epilogue>(Epilogue::Arguments args) {
        dim3 grid(M / GEMM::BLOCK_M, N / GEMM::BLOCK_N);

        bool swapBlockMN = M > N * 2;
        if (swapBlockMN) {
            std::swap(grid.x, grid.y);
        }

        invoke_kernel<GEMM::gemm_w8a8_kernel<Epilogue>>
            <<<grid, GEMM::WARP_SIZE * GEMM::NUM_WARPS>>>(act.data_ptr<GEMM::packed_act_t>(),
                                                          wgt.data_ptr<GEMM::packed_wgt_t>(),
                                                          ascales.data_ptr<GEMM::packed_ascale_t>(),
                                                          wscales.data_ptr<GEMM::packed_wscale_t>(),
                                                          // out.valid() ? out.data_ptr<GEMM::half_t>() : nullptr,
                                                          M,
                                                          N,
                                                          K,
                                                          args,
                                                          swapBlockMN,
                                                          false);
        checkCUDA(cudaGetLastError());
    };

    auto launch_bias = [&]<typename NextEpilogue>(NextEpilogue::Arguments nextArgs) {
        if (!bias.valid()) {
            return launch.template operator()<NextEpilogue>(nextArgs);
        }

        assert(bias.numel() == N);

        // append EpilgoueNop to workaround mismatched memory layout of std::tuple between device and host code on
        // Windows
        // ** sizeof(std::tuple<std::tuple<int>>) == 8 on device **
        using Epilogue = GEMM::EpilogueCombination<GEMM::EpilogueBias<true, false>, NextEpilogue, GEMM::EpilogueNop>;
        return launch.template operator()<Epilogue>({GEMM::EpilogueBias<true, false>::Arguments{
                                                         .bias = bias.data_ptr<GEMM::packed_wscale_t>(),
                                                     },
                                                     nextArgs,
                                                     {}});
    };

    launch_bias.template operator()<GEMM::EpilogueDefault>(GEMM::EpilogueDefault::Arguments{
        .out     = out.data_ptr<GEMM::half_t>(),
        .actualM = actualM,
        .actualN = actualN,
    });
}

#if 0
void gemm_w8a8_fuse_litela(
    Tensor act,      // [B, (M), K]
    Tensor wgt,      // [N, K]
    Tensor out_q,    // [B, (M), N / 3]
    Tensor out_vk,   // [B, num_heads, head_dim + 1, head_dim]
    Tensor ascales,  // [1, M]
    Tensor wscales   // [1, N]
) {
    using GEMM = GEMM_W8A8;
    using Epilogue = GEMM::EpilogueLiteLA;

    int M = act.numel() / act.shape[-1];
    int N = wgt.shape[0];
    int K = act.shape[-1];
    assert(K == wgt.shape[1]);

    assert(out_vk.ndims() == 4);
    assert(out_vk.shape[2] == Epilogue::LITELA_HEAD_DIM + 1);
    assert(out_vk.shape[3] == Epilogue::LITELA_HEAD_DIM);
    assert(out_vk.shape[1] * Epilogue::LITELA_HEAD_DIM * 3 == N);

    int batch_size = out_vk.shape[0];
    int num_heads = out_vk.shape[1];

    assert(M % batch_size == 0);
    int batch_m = M / batch_size;

    Epilogue::Arguments epilogueArgs;
    epilogueArgs.batch_m = act.shape[1];
    epilogueArgs.out_q = out_q.data_ptr<GEMM::half_t>();
    epilogueArgs.out_vk = out_vk.data_ptr<float>();

    checkCUDA(cudaMemsetAsync(out_vk.data_ptr(), 0, out_vk.buffer->getSize()));

    auto func = invoke_kernel<GEMM::gemm_w8a8_kernel<Epilogue>,
        const GEMM::packed_act_t *,
        const GEMM::packed_wgt_t *,
        const GEMM::packed_ascale_t *,
        const GEMM::packed_wscale_t *,
        // GEMM::half_t *,
        int, int, int,
        Epilogue::Arguments,
        bool,
        bool>;

    checkCUDA(cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, Epilogue::SHMEM_SIZE));

    dim3 grid(M / GEMM::BLOCK_M, N / GEMM::BLOCK_N);

    bool swapBlockMN = M > N * 2;
    if (swapBlockMN) {
        std::swap(grid.x, grid.y);
    }

    func<<<grid, GEMM::WARP_SIZE * GEMM::NUM_WARPS, Epilogue::SHMEM_SIZE>>>(
        act.data_ptr<GEMM::packed_act_t>(),
        wgt.data_ptr<GEMM::packed_wgt_t>(),
        ascales.data_ptr<GEMM::packed_ascale_t>(),
        wscales.data_ptr<GEMM::packed_wscale_t>(),
        // nullptr,
        M, N, K, epilogueArgs,
        swapBlockMN,
        false
    );
    checkCUDA(cudaGetLastError());

    invoke_kernel<Epilogue::vk_mul_q_kernel><<<dim3(batch_m / 128, num_heads, batch_size), 128>>>(
        out_q.data_ptr<GEMM::half_t>(),
        out_vk.data_ptr<float>(),
        1e-6f
    );
    checkCUDA(cudaGetLastError());
}
#endif

}; // namespace nunchaku::kernels
