#include "layernorm_kernels_impl.cuh"
#include "dispatch_utils.h"

void rms_norm(Tensor &out,    // [..., hidden_size]
              Tensor &input,  // [..., hidden_size]
              Tensor &weight, // [hidden_size]
              float epsilon,
              bool use_quant) {
    int hidden_size = input.size(-1);
    int num_tokens  = input.numel() / hidden_size;
    dim3 grid(num_tokens);
    dim3 block(std::min(hidden_size, 1024));
    const cudaStream_t stream = getCurrentCUDAStream();
    VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "rms_norm_kernel", [&] {
        if (use_quant) {
            vllm::rms_norm_kernel<scalar_t, int8_t, true><<<grid, block, 0, stream>>>(out.data_ptr<int8_t>(),
                                                                                      input.data_ptr<scalar_t>(),
                                                                                      weight.data_ptr<scalar_t>(),
                                                                                      epsilon,
                                                                                      num_tokens,
                                                                                      hidden_size);
        } else {
            vllm::rms_norm_kernel<scalar_t, scalar_t, false><<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(),
                                                                                         input.data_ptr<scalar_t>(),
                                                                                         weight.data_ptr<scalar_t>(),
                                                                                         epsilon,
                                                                                         num_tokens,
                                                                                         hidden_size);
        }
    });
}

void layernorm_general(Tensor out, Tensor input, Tensor weight, Tensor bias, float epsilon) {
    int hidden_size = input.size(-1);
    int num_tokens  = input.numel() / hidden_size;
    dim3 grid(num_tokens);
    dim3 block(std::min(hidden_size, 256));
    block.x = 32 * ((block.x + 31) / 32);

    size_t size_shmem = input.scalar_size() * hidden_size;

    const cudaStream_t stream = getCurrentCUDAStream();
    VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "generalLayerNorm", [&] {
        using T = typename packed_as<scalar_t, 2>::type;
        vllm::generalLayerNorm<T, half, true><<<grid, block, size_shmem, stream>>>(
            reinterpret_cast<T *>(input.data_ptr<scalar_t>()),
            weight.valid() ? reinterpret_cast<T *>(weight.data_ptr<scalar_t>()) : nullptr,
            bias.valid() ? reinterpret_cast<T *>(bias.data_ptr<scalar_t>()) : nullptr,
            reinterpret_cast<T *>(out.data_ptr<scalar_t>()),
            epsilon,
            num_tokens,
            hidden_size,
            nullptr,
            nullptr,
            nullptr,
            true);
    });
}

void rms_norm_general(Tensor &out,     // [..., hidden_size]
                      Tensor &input,   // [..., hidden_size]
                      Tensor &weight,  // [hidden_size]
                      Tensor &scaling, // [tokens] or [1]
                      float epsilon,
                      bool use_per_token_quant) {
    int hidden_size = input.size(-1);
    int num_tokens  = input.numel() / hidden_size;
    dim3 grid(num_tokens);
    dim3 block(std::min(hidden_size, 1024));
    block.x = 32 * ((block.x + 31) / 32);

    const cudaStream_t stream = getCurrentCUDAStream();
    VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "generalLayerNorm", [&] {
        using T = scalar_t;
        if (use_per_token_quant) {
            // per-token
            vllm::generalLayerNorm<T, half>
                <<<grid, block, 0, stream>>>(reinterpret_cast<T *>(input.data_ptr<scalar_t>()),
                                             reinterpret_cast<T *>(weight.data_ptr<scalar_t>()),
                                             nullptr,
                                             nullptr,
                                             epsilon,
                                             num_tokens,
                                             hidden_size,
                                             nullptr,
                                             scaling.data_ptr<half>(),
                                             out.data_ptr<int8_t>(),
                                             false);
            // input, gamma, beta, normed_output, eps, tokens, hidden_dim, per_tensor_scale, per_token_scale
            // normed_output_quant, use_shmem
            // out.data_ptr<int8_t>(), input.data_ptr<scalar_t>(),
            // weight.data_ptr<scalar_t>(), epsilon, num_tokens, hidden_size);
        } else {
            // per-tensor
            vllm::generalLayerNorm<T, half>
                <<<grid, block, 0, stream>>>(reinterpret_cast<T *>(input.data_ptr<scalar_t>()),
                                             reinterpret_cast<T *>(weight.data_ptr<scalar_t>()),
                                             nullptr,
                                             nullptr,
                                             epsilon,
                                             num_tokens,
                                             hidden_size,
                                             scaling.data_ptr<half>(),
                                             nullptr,
                                             out.data_ptr<int8_t>(),
                                             false);
        }
    });
}

void rms_norm_general_fuse_sum(Tensor &out,       // [..., hidden_size]
                               Tensor &input,     // [..., hidden_size]
                               Tensor &weight,    // [hidden_size]
                               Tensor &input_sum, // [tokens] or [1]
                               Tensor &scaling,   // [tokens] or [1]
                               float epsilon,
                               bool use_per_token_quant) {
    int hidden_size = input.size(-1);
    int num_tokens  = input.numel() / hidden_size;
    dim3 grid(num_tokens);
    dim3 block(std::min(hidden_size, 1024));
    block.x = 32 * ((block.x + 31) / 32);

    const cudaStream_t stream = getCurrentCUDAStream();
    VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "generalLayerNorm_fuse_sum", [&] {
        using T = scalar_t;
        if (use_per_token_quant) {
            // per-token
            vllm::generalLayerNorm_fuse_sum<T, half>
                <<<grid, block, 0, stream>>>(reinterpret_cast<T *>(input.data_ptr<scalar_t>()),
                                             reinterpret_cast<T *>(weight.data_ptr<scalar_t>()),
                                             nullptr,
                                             nullptr,
                                             epsilon,
                                             num_tokens,
                                             hidden_size,
                                             input_sum.data_ptr<half>(),
                                             nullptr,
                                             scaling.data_ptr<half>(),
                                             out.data_ptr<int8_t>(),
                                             false);
            // input, gamma, beta, normed_output, eps, tokens, hidden_dim, per_tensor_scale, per_token_scale
            // normed_output_quant, use_shmem
            // out.data_ptr<int8_t>(), input.data_ptr<scalar_t>(),
            // weight.data_ptr<scalar_t>(), epsilon, num_tokens, hidden_size);
        } else {
            // per-tensor
            // Rasing error here
            // Not implemented per-tensor input_sum
            assert(false);

            vllm::generalLayerNorm_fuse_sum<T, half>
                <<<grid, block, 0, stream>>>(reinterpret_cast<T *>(input.data_ptr<scalar_t>()),
                                             reinterpret_cast<T *>(weight.data_ptr<scalar_t>()),
                                             nullptr,
                                             nullptr,
                                             epsilon,
                                             num_tokens,
                                             hidden_size,
                                             nullptr,
                                             scaling.data_ptr<half>(),
                                             nullptr,
                                             out.data_ptr<int8_t>(),
                                             false);
        }
    });
}

void invoke_dequant_add_residual_rms_norm_quant(Tensor &out,      // [..., hidden_size]
                                                Tensor &input,    // [..., hidden_size]
                                                Tensor &residual, // [..., hidden_size]
                                                Tensor &gamma,    // [hidden_size]
                                                half scale,
                                                float epsilon) {
    int hidden_size = input.size(-1);
    int num_tokens  = input.numel() / hidden_size;
    dim3 grid(num_tokens);
    dim3 block(std::min(hidden_size, 1024));
    const cudaStream_t stream = getCurrentCUDAStream();
    VLLM_DISPATCH_FLOATING_TYPES(residual.scalar_type(), "dequant_add_residual_rms_norm_quant_kernel", [&] {
        vllm::dequant_add_residual_rms_norm_quant_kernel<scalar_t, half, false>
            <<<grid, block, 0, stream>>>(input.data_ptr<int32_t>(),
                                         residual.data_ptr<scalar_t>(),
                                         out.data_ptr<int8_t>(),
                                         gamma.data_ptr<scalar_t>(),
                                         epsilon,
                                         scale,
                                         num_tokens,
                                         hidden_size);
    });
}

void invoke_dequant_add_residual_rms_norm_quant(Tensor &out,      // [..., hidden_size]
                                                Tensor &input,    // [..., hidden_size]
                                                Tensor &residual, // [..., hidden_size]
                                                Tensor &gamma,    // [hidden_size]
                                                Tensor &scale,    // [num_tokens]
                                                float epsilon) {
    int hidden_size = input.size(-1);
    int num_tokens  = input.numel() / hidden_size;

    dim3 grid(num_tokens);
    dim3 block(std::min(hidden_size, 1024));

    const cudaStream_t stream = getCurrentCUDAStream();
    VLLM_DISPATCH_FLOATING_TYPES(residual.scalar_type(), "dequant_add_residual_rms_norm_quant_kernel", [&] {
        vllm::dequant_add_residual_rms_norm_quant_kernel<scalar_t, half *, true>
            <<<grid, block, 0, stream>>>(input.data_ptr<int32_t>(),
                                         residual.data_ptr<scalar_t>(),
                                         out.data_ptr<int8_t>(),
                                         gamma.data_ptr<scalar_t>(),
                                         epsilon,
                                         scale.data_ptr<half>(),
                                         num_tokens,
                                         hidden_size);
    });
}
