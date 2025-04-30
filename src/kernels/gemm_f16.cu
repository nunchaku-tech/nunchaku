#include "gemm_f16.h"

#include "dispatch_cutlass.h"

#include <cutlass/core_io.h>
#include <cutlass/cutlass.h>
#include <cutlass/half.h>
#include <cutlass/bfloat16.h>

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>

using spdlog::fmt_lib::format;

Tensor gemm_f16(Tensor input,  // FP16
                Tensor weight, // FP16
                Tensor out,    // FP16
                Tensor bias,
                float alpha) {
    auto N = weight.size(0);
    auto K = input.size(-1);
    auto M = input.numel() / K;
    assert(weight.size(1) == K);

    spdlog::debug("gemm_f16: M={} K={} N={}", M, K, N);

    dispatchF16(weight.dtype(), [&]<typename half_t>() {
        using ElementOutput          = half_t;
        using ElementAccumulator     = float;
        using ElementComputeEpilogue = half_t;
        using ElementInputA          = half_t; // <- data type of elements in input matrix A
        using ElementInputB          = half_t; // <- data type of elements in input matrix B

        using LayoutInputA = cutlass::layout::RowMajor;
        using LayoutInputB = cutlass::layout::ColumnMajor;
        using LayoutOutput = cutlass::layout::RowMajor;

        // #if CUDA_ARCH >= 800
        using Gemm = cutlass::gemm::device::Gemm<ElementInputA,
                                                 cutlass::layout::RowMajor,
                                                 ElementInputB,
                                                 cutlass::layout::ColumnMajor,
                                                 ElementOutput,
                                                 cutlass::layout::RowMajor,
                                                 ElementAccumulator,
                                                 cutlass::arch::OpClassTensorOp,
                                                 cutlass::arch::Sm75>;
        // cutlass::gemm::GemmShape<128, 128, 64>,
        // cutlass::gemm::GemmShape<32, 64, 64>, cutlass::gemm::GemmShape<16, 8, 16>,
        // cutlass::epilogue::thread::LinearCombination<
        //     ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
        //     ElementAccumulator, ElementComputeEpilogue>,
        // cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;

        auto input_size  = cutlass::MatrixCoord(M, K);
        auto weight_size = cutlass::MatrixCoord(K, N);
        auto output_size = cutlass::MatrixCoord(M, N);

        auto device = input.device();
        // use the broadcasted bias as the output
        // auto out = bias.to(device).view({1, -1}).repeat({M, 1});

        if (!out.valid()) {
            auto out_shape = TensorShape(input.shape.dataExtent);
            out_shape[-1]  = N;
            out            = Tensor::empty(out_shape, input.scalar_type(), input.device());
        }

        // FIXME: check contiguous of input if dims >= 3
        assert(input.stride(-1) == 1);
        // assert(input.is_contiguous());
        assert(weight.is_contiguous());

        assert(out.dtype() == input.scalar_type());
        assert(out.shape[-1] == N);
        assert(out.numel() / out.shape[-1] == M);
        assert(out.stride(-1) == 1);
        // FIXME: check contiguous of output if dims >= 3

        assert(!bias.valid() || (bias.ndims() == 1 && bias.shape[0] == N));

        // constexpr int kSparse = Gemm::kSparse;
        // How many elements of A are covered per ElementE
        // constexpr int kElementsPerElementE = Gemm::kElementsPerElementE;
        // The size of individual meta data
        // constexpr int kMetaSizeInBits = Gemm::kMetaSizeInBits;
        cutlass::gemm::GemmCoord problem_size(M, N, K);

        cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(input.data_ptr<ElementInputA>(),
                                                                  LayoutInputA(input.stride(-2)));
        cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(weight.data_ptr<ElementInputB>(),
                                                                   LayoutInputB::packed(weight_size));
        cutlass::TensorRef<ElementOutput, LayoutOutput> bias_ref(
            bias.valid() ? bias.data_ptr<ElementOutput>() : out.data_ptr<ElementOutput>(), LayoutOutput(0));
        cutlass::TensorRef<ElementOutput, LayoutOutput> out_ref(out.data_ptr<ElementOutput>(),
                                                                LayoutOutput(out.stride(-2)));

        typename Gemm::Arguments arguments{problem_size, // <- problem size of matrix multiplication
                                           input_ref,    // <- reference to matrix A on device
                                           weight_ref,   // <- reference to matrix B on device
                                           bias_ref,     // <- reference to matrix C on device
                                           out_ref,      // <- reference to matrix D on device
                                           {ElementOutput(alpha), ElementOutput(bias.valid() ? 1.0f : 0.0f)},
                                           1};
        Gemm gemm_op;

        // Using the arguments, query for extra workspace required for matrix
        // multiplication computation
        size_t workspace_size = Gemm::get_workspace_size(arguments);

        // Allocate workspace memory
        // cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

        BufferCUDA workspace(workspace_size);

        // Check the problem size is supported or not
        cutlass::Status status = gemm_op.can_implement(arguments);
        if (status != cutlass::Status::kSuccess) {
            throw std::runtime_error(format("cutlass cannot implement M={} N={} K={}", M, N, K));
        }

        // Initialize CUTLASS kernel with arguments and workspace pointer
        status = gemm_op.initialize(arguments, workspace.getPtr());
        if (status != cutlass::Status::kSuccess) {
            throw std::runtime_error("cutlass cannot initialize");
        }

        status = gemm_op();
        if (status != cutlass::Status::kSuccess) {
            throw std::runtime_error("cutlass cannot run");
        }
    });

    return out;
}
