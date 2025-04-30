#include "activation_kernels_impl.cuh"
#include "activation_kernels.h"
#include "dispatch_utils.h"

// Launch element-wise activation kernel.
#define LAUNCH_ACTIVATION_KERNEL(KERNEL)                                                                               \
    int d          = input.size(-1);                                                                                   \
    int num_tokens = input.numel() / d;                                                                                \
    dim3 grid(num_tokens);                                                                                             \
    dim3 block(std::min(d, 1024));                                                                                     \
    const cudaStream_t stream = getCurrentCUDAStream();                                                                \
    VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "activation_kernel", [&] {                                       \
        vllm::activation_kernel<scalar_t, KERNEL<scalar_t>>                                                            \
            <<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), d);                     \
    });

void silu_and_mul(Tensor &out,   // [..., d]
                  Tensor &input) // [..., 2 * d]
{
    int64_t num_tokens = input.numel() / input.size(-1);
    int d              = input.size(-1) / 2;
    dim3 grid(num_tokens);
    dim3 block(std::min(d, 1024));
    const cudaStream_t stream = getCurrentCUDAStream();
    //   dispatchFloat(input.scalar_type(), [&]<typename scalar_t>() {
    //     vllm::silu_and_mul_kernel<scalar_t><<<grid, block, 0, stream>>>(
    //         out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), d);
    //   });
    VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "silu_and_mul_kernel", [&] {
        vllm::silu_and_mul_kernel<scalar_t>
            <<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), d);
    });
}

void invoke_dequant_silu_and_mul_quant(Tensor &out,   // [..., d]
                                       Tensor &input, // [..., 2 * d]
                                       const float scale_gate,
                                       const float scale_up,
                                       const float scale_out) {
    int64_t num_tokens = input.numel() / input.size(-1);
    int d              = input.size(-1) / 2;
    dim3 grid(num_tokens);
    dim3 block(std::min(d, 1024));
    const cudaStream_t stream = getCurrentCUDAStream();
    vllm::dequant_silu_and_mul_quant_kernel<float, false><<<grid, block, 0, stream>>>(
        out.data_ptr<int8_t>(), input.data_ptr<int32_t>(), d, scale_gate, scale_up, scale_out);
}

void invoke_dequant_silu_and_mul_quant(Tensor &out,   // [..., d]
                                       Tensor &input, // [..., 2 * d]
                                       const float scale_gate,
                                       const float scale_up,
                                       Tensor &scale_out, // [num_tokens]
                                       Tensor &tmp        // [..., d]
) {
    int64_t num_tokens = input.numel() / input.size(-1);
    int d              = input.size(-1) / 2;
    dim3 grid(num_tokens);
    dim3 block(std::min(d, 1024));
    const cudaStream_t stream = getCurrentCUDAStream();
    vllm::dequant_silu_and_mul_quant_kernel<float *, true><<<grid, block, 0, stream>>>(out.data_ptr<int8_t>(),
                                                                                       input.data_ptr<int32_t>(),
                                                                                       d,
                                                                                       scale_gate,
                                                                                       scale_up,
                                                                                       scale_out.data_ptr<float>(),
                                                                                       tmp.data_ptr<float>());
}

void silu(Tensor &out,   // [..., d]
          Tensor &input) // [..., d]
{
    LAUNCH_ACTIVATION_KERNEL(vllm::silu);
}

void gelu_new(Tensor &out,   // [..., d]
              Tensor &input) // [..., d]
{
    LAUNCH_ACTIVATION_KERNEL(vllm::gelu_new_kernel);
}

void gelu_fast(Tensor &out,   // [..., d]
               Tensor &input) // [..., d]
{
    LAUNCH_ACTIVATION_KERNEL(vllm::gelu_fast_kernel);
}
