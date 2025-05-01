#include "gemm_batched.h"

#include <cutlass/core_io.h>
#include <cutlass/cutlass.h>
#include <cutlass/half.h>

#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/numeric_types.h>

using spdlog::fmt_lib::format;

Tensor gemm_batched_fp16(Tensor a,  // FP16 row-major [(... batch ...), M, K]
                         Tensor b,  // FP16 col-major [(... batch ...), N, K]
                         Tensor out // FP32 row-major [(... batch ...), M, N]
) {
    const int M     = a.shape[-2];
    const int K     = a.shape[-1];
    const int N     = a.shape[-2];
    const int batch = a.numel() / (M * K);

    using ElementInput  = cutlass::half_t;
    using ElementOutput = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutO = cutlass::layout::RowMajor;

    using Gemm = cutlass::gemm::device::GemmBatched<
        ElementInput,
        LayoutA,
        ElementInput,
        LayoutB,
        ElementOutput,
        LayoutO,
        ElementOutput,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<32, 32, 64>,
        cutlass::gemm::GemmShape<32, 32, 64>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<ElementOutput,
                                                     128 / cutlass::sizeof_bits<ElementOutput>::value,
                                                     ElementOutput,
                                                     ElementOutput>,
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
        2>;

    auto sizeA = cutlass::MatrixCoord(M, K);
    auto sizeB = cutlass::MatrixCoord(K, N);
    auto sizeO = cutlass::MatrixCoord(M, N);

    if (!out.valid()) {
        auto outShape = TensorShape(a.shape.dataExtent);
        outShape[-1]  = N;
        out           = Tensor::empty(outShape, Tensor::FP32, a.device());
    }

    assert(K == b.shape[-1]);
    assert(M == out.shape[-2]);
    assert(N == out.shape[-1]);

    assert(a.dtype() == Tensor::FP16);
    assert(a.dtype() == b.dtype());
    assert(out.dtype() == Tensor::FP32);

    cutlass::gemm::GemmCoord problemSize(M, N, K);

    cutlass::TensorRef<ElementInput, LayoutA> refA(a.data_ptr<ElementInput>(), LayoutA(a.stride(-2)));
    cutlass::TensorRef<ElementInput, LayoutB> refB(b.data_ptr<ElementInput>(), LayoutB(b.stride(-2)));
    cutlass::TensorRef<ElementOutput, LayoutO> refO(out.data_ptr<ElementOutput>(), LayoutO(out.stride(-2)));

    typename Gemm::Arguments arguments{problemSize,
                                       refA,
                                       (int)a.stride(-3),
                                       refB,
                                       (int)b.stride(-3),
                                       refO,
                                       (int)out.stride(-3),
                                       refO,
                                       (int)out.stride(-3),
                                       {ElementOutput(1), ElementOutput(0)},
                                       batch};

    Gemm op;
    BufferCUDA workspace(Gemm::get_workspace_size(arguments));

    cutlass::Status status = op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error(format("cutlass cannot implement M={} N={} K={}", M, N, K));
    }

    status = op.initialize(arguments, workspace.getPtr());
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("cutlass cannot initialize");
    }

    status = op();
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("cutlass cannot run");
    }

    return out;
}
