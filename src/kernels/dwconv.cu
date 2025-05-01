#include "common.h"
#include "Tensor.h"

#include "dispatch_cutlass.h"

#include <cuda_runtime.h>
#include "cutlass/cutlass.h"

#include "cutlass/conv/device/direct_convolution.h"
#include "cutlass/conv/kernel/default_depthwise_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"

// depthwise_Conv2d operation cutlass_sm80_tensorop_f16_s16x8x16fprop_analytic_f16_256x128_64x3_nhwc_align8

#if 0
using ThreadBlockOutputShape = cutlass::conv::TensorNHWCShape<1, 8, 8, 64>;

using FilterShape = cutlass::MatrixShape<3, 3>;

using ThreadblockShape =
    cutlass::gemm::GemmShape<ThreadBlockOutputShape::kNHW, 64, FilterShape::kCount>;

using WarpShape = cutlass::gemm::GemmShape<16, 64, FilterShape::kCount>;

using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

using DepthwiseDirect2dConv = typename cutlass::conv::kernel::DefaultDepthwiseDirect2dConvFprop<
    cutlass::half_t,
    cutlass::layout::TensorNHWC,
    cutlass::half_t,
    cutlass::layout::TensorNHWC,
    cutlass::half_t,
    cutlass::layout::TensorNHWC,
    cutlass::half_t,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80, // TODO
    ThreadblockShape,
    ThreadBlockOutputShape,
    FilterShape,
    WarpShape,
    InstructionShape,
    cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, cutlass::half_t, float, cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>,
    cutlass::conv::threadblock::DepthwiseDirect2dConvIdentityThreadblockSwizzle<
        1,
        ThreadBlockOutputShape::kN,
        ThreadBlockOutputShape::kH,
        ThreadBlockOutputShape::kW>,
    4,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kFixedStrideDilation,
    cutlass::conv::StrideSupport::kFixed,
    cutlass::MatrixShape<1, 1>,
    cutlass::MatrixShape<1, 1>>::Kernel;

using DeviceKernel =
    typename cutlass::conv::device::DirectConvolution<DepthwiseDirect2dConv>;

using UnderlyingKernel = typename DeviceKernel::UnderlyingKernel;
namespace {
    using TensorRefA = typename UnderlyingKernel::TensorRefA;
    using TensorRefB = typename UnderlyingKernel::TensorRefB;
    using TensorRefC = typename UnderlyingKernel::TensorRefC;
    using ElementCompute = typename UnderlyingKernel::EpilogueOutputOp::ElementCompute;
}

template <typename TensorRef, typename Element>
TensorRef get_tensor_ref(cutlass::Tensor4DCoord tensor_coord, Element *ptr) {
    cutlass::layout::TensorNHWC layout = cutlass::layout::TensorNHWC::packed(tensor_coord);
    TensorRef tensor_ref(ptr, layout);
    return tensor_ref;
}

static cutlass::Status depthwise_conv2d_kernel_run(cutlass::conv::Conv2dProblemSize *problem_size,
                                            UnderlyingKernel::ElementA *A, UnderlyingKernel::ElementB *B,
                                            UnderlyingKernel::ElementC *C, UnderlyingKernel::ElementC *D,
                                            ElementCompute alpha, ElementCompute beta, std::string split_k_mode,
                                            cudaStream_t stream, int device_id = 0)
{
    // create the tensor references
    cutlass::Tensor4DCoord tensor_coord_A = cutlass::conv::implicit_gemm_tensor_a_extent(
        cutlass::conv::Operator::kFprop, *problem_size);
    cutlass::Tensor4DCoord tensor_coord_B = cutlass::conv::implicit_gemm_tensor_b_extent(
        cutlass::conv::Operator::kFprop, *problem_size);
    cutlass::Tensor4DCoord tensor_coord_C = cutlass::conv::implicit_gemm_tensor_c_extent(
        cutlass::conv::Operator::kFprop, *problem_size);

    TensorRefA tensor_ref_A = get_tensor_ref<TensorRefA, UnderlyingKernel::ElementA>(tensor_coord_A, A);
    TensorRefB tensor_ref_B = get_tensor_ref<TensorRefB, UnderlyingKernel::ElementB>(tensor_coord_B, B);
    TensorRefC tensor_ref_C = get_tensor_ref<TensorRefC, UnderlyingKernel::ElementC>(tensor_coord_C, C);
    TensorRefC tensor_ref_D = get_tensor_ref<TensorRefC, UnderlyingKernel::ElementC>(tensor_coord_C, D);

    cutlass::conv::SplitKMode mode;
    if (split_k_mode == "serial") {
        mode = cutlass::conv::SplitKMode::kSerial;
    } else if (split_k_mode == "parallel") {
        mode = cutlass::conv::SplitKMode::kParallel;
    } else {
        throw std::runtime_error("Invalid split_k_mode: " + split_k_mode);
    }

    typename DeviceKernel::Arguments arguments{
        *problem_size,
        tensor_ref_A,
        tensor_ref_B,
        tensor_ref_C,
        tensor_ref_D,
        {alpha, beta},
        tensor_ref_B,
        // mode
    };

    DeviceKernel implicit_gemm_op;

    size_t workspace_size = implicit_gemm_op.get_workspace_size(arguments);

    BufferCUDA workspace(workspace_size);

    cutlass::Status status = implicit_gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        return status;
    }

    status = implicit_gemm_op.initialize(arguments, workspace.getPtr(), stream);
    if (status != cutlass::Status::kSuccess) {
        return status;
    }

    //
    // Launch initialized CUTLASS kernel
    //
    status = implicit_gemm_op(stream);

    return status;
}

Tensor depthwise_conv2d_kernel(Tensor A, Tensor B) {
    int N, H, W, C_, K, C__, R, S, P, Q;
    N = A.size(0);
    H = A.size(1);
    W = A.size(2);
    C_ = A.size(3);

    // std::cout << A.dtype() << std::endl;
    // std::cout << "A: " << N << ", " << H << ", " << W << ", " << C_ << std::endl;
    // for (int h = 0; h < H; ++h)
    // {
    //     for (int w = 0; w < W; ++w)
    //         std::cout << A[0][h][w][0].item() << " ";
    //     std::cout << std::endl;
    // }

    K = B.size(0);
    R = B.size(1);
    S = B.size(2);
    C__ = B.size(3);

    // std::cout << B.dtype() << std::endl;
    // std::cout << "B: " << K << ", " << R << ", " << S << ", " << C__ << std::endl;
    // for (int h = 0; h < R; ++h)
    // {
    //     for (int w = 0; w < S; ++w)
    //         std::cout << B[0][h][w][0].item() << " ";
    //     std::cout << std::endl;
    // }

    cutlass::conv::Conv2dProblemSize problem_size(
        cutlass::Tensor4DCoord(N, H, W, C_),
        cutlass::Tensor4DCoord(K, R, S, C__),
        cutlass::Tensor4DCoord(1, 1, 1, 1),
        cutlass::MatrixCoord(1, 1),
        cutlass::MatrixCoord(1, 1),
        cutlass::conv::Mode::kCrossCorrelation,
        1,
        C_ // groups
    );

    P = problem_size.P;
    Q = problem_size.Q;

    // printf("P=%d Q=%d\n", P, Q);

    typename UnderlyingKernel::ElementC *ptrC = nullptr;

    Tensor D = Tensor::allocate({N, P, Q, K}, A.dtype(), A.device());

    auto stream = getCurrentCUDAStream();

    cutlass::Status status = depthwise_conv2d_kernel_run(
        &problem_size,
        reinterpret_cast<typename UnderlyingKernel::ElementA *>(A.data_ptr()),
        reinterpret_cast<typename UnderlyingKernel::ElementB *>(B.data_ptr()),
        ptrC,
        reinterpret_cast<typename UnderlyingKernel::ElementC *>(D.data_ptr()),
        1, 0,
        "serial", stream, B.device().idx);
    assert(status == cutlass::Status::kSuccess);

    return D;
}

#endif

Tensor dwconv_f16(Tensor input, Tensor weight, Tensor out, Tensor bias) {

    assert(input.ndims() == 4);

    const int N  = input.size(0);
    const int H  = input.size(1);
    const int W  = input.size(2);
    const int C_ = input.size(3);

    assert(weight.ndims() == 4);

    const int K   = weight.size(0);
    const int R   = weight.size(1);
    const int S   = weight.size(2);
    const int C__ = weight.size(3);

    // weight = weight.copy(weight.device());

    dispatchF16(weight.dtype(), [&]<typename half_t>() {
        using ElementOutput          = half_t;
        using ElementAccumulator     = half_t;
        using ElementComputeEpilogue = half_t;
        using ElementInputA          = half_t;
        using ElementInputB          = half_t;

        using LayoutInputA = cutlass::layout::TensorNHWC;
        using LayoutInputB = cutlass::layout::TensorNHWC;
        using LayoutOutput = cutlass::layout::TensorNHWC;

        using ThreadBlockOutputShape = cutlass::conv::TensorNHWCShape<1, 8, 8, 64>;
        using FilterShape            = cutlass::MatrixShape<3, 3>;

        using ThreadblockShape = cutlass::gemm::GemmShape<ThreadBlockOutputShape::kNHW, 64, FilterShape::kCount>;
        using WarpShape        = cutlass::gemm::GemmShape<16, 64, FilterShape::kCount>;
        using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

        using DepthwiseDirect2dConv = typename cutlass::conv::kernel::DefaultDepthwiseDirect2dConvFprop<
            ElementInputA,
            LayoutInputA,
            ElementInputB,
            LayoutInputB,
            ElementOutput,
            LayoutOutput,
            ElementAccumulator,
            cutlass::arch::OpClassSimt,
            cutlass::arch::Sm80,
            ThreadblockShape,
            ThreadBlockOutputShape,
            FilterShape,
            WarpShape,
            InstructionShape,
            cutlass::epilogue::thread::LinearCombination<ElementOutput,
                                                         128 / cutlass::sizeof_bits<ElementOutput>::value,
                                                         ElementOutput,
                                                         ElementComputeEpilogue>,
            cutlass::conv::threadblock::DepthwiseDirect2dConvIdentityThreadblockSwizzle<1,
                                                                                        ThreadBlockOutputShape::kN,
                                                                                        ThreadBlockOutputShape::kH,
                                                                                        ThreadBlockOutputShape::kW>,
            4,
            cutlass::arch::OpMultiplyAdd,
            cutlass::conv::IteratorAlgorithm::kFixedStrideDilation,
            cutlass::conv::StrideSupport::kFixed,
            cutlass::MatrixShape<1, 1>,
            cutlass::MatrixShape<1, 1>>::Kernel;

        using DeviceKernel = typename cutlass::conv::device::DirectConvolution<DepthwiseDirect2dConv>;

        cutlass::conv::Conv2dProblemSize problem_size(cutlass::Tensor4DCoord(N, H, W, C_),
                                                      cutlass::Tensor4DCoord(K, R, S, C__),
                                                      cutlass::Tensor4DCoord(1, 1, 1, 1),
                                                      cutlass::MatrixCoord(1, 1),
                                                      cutlass::MatrixCoord(1, 1),
                                                      cutlass::conv::Mode::kCrossCorrelation,
                                                      1,
                                                      C_ // groups
        );

        const int P = problem_size.P;
        const int Q = problem_size.Q;

        if (!out.valid()) {
            out = Tensor::allocate({N, P, Q, K}, input.dtype(), input.device());
        }
        assert(out.ndims() == 4);
        assert(out.size(0) == N);
        assert(out.size(1) == P);
        assert(out.size(2) == Q);
        assert(out.size(3) == K);

        Tensor tmp_weight = Tensor::empty_like(weight);

        cutlass::TensorRef<ElementInputA, LayoutInputA> a_ref(
            input.data_ptr<ElementInputA>(), LayoutInputA(input.stride(2), input.stride(1), input.stride(0)));
        cutlass::TensorRef<ElementInputB, LayoutInputB> b_ref(
            weight.data_ptr<ElementInputB>(), LayoutInputB(weight.stride(2), weight.stride(1), weight.stride(0)));
        cutlass::TensorRef<ElementOutput, LayoutOutput> c_ref(
            bias.valid() ? bias.data_ptr<ElementOutput>() : out.data_ptr<ElementOutput>(), LayoutOutput(0, 0, 0));
        cutlass::TensorRef<ElementOutput, LayoutOutput> d_ref(
            out.data_ptr<ElementOutput>(), LayoutOutput(out.stride(2), out.stride(1), out.stride(0)));
        cutlass::TensorRef<ElementOutput, LayoutOutput> tmpw_ref(
            tmp_weight.data_ptr<ElementOutput>(),
            LayoutOutput(tmp_weight.stride(2), tmp_weight.stride(1), tmp_weight.stride(0)));

        typename DeviceKernel::Arguments arguments{
            problem_size,
            a_ref,
            b_ref,
            c_ref,
            d_ref,
            {ElementOutput(1.0f), ElementOutput(bias.valid() ? 1.0f : 0.0f)},
            tmpw_ref,
        };

        DeviceKernel implicit_gemm_op;

        size_t workspace_size = implicit_gemm_op.get_workspace_size(arguments);

        BufferCUDA workspace(workspace_size);
        auto stream = getCurrentCUDAStream();

        cutlass::Status status = implicit_gemm_op.can_implement(arguments);
        if (status != cutlass::Status::kSuccess) {
            throw std::runtime_error("cutlass cannot implement");
        }

        status = implicit_gemm_op.initialize(arguments, workspace.getPtr(), stream);
        if (status != cutlass::Status::kSuccess) {
            throw std::runtime_error("cutlass cannot initialize");
        }

        status = implicit_gemm_op(stream);
        if (status != cutlass::Status::kSuccess) {
            throw std::runtime_error("cutlass cannot run");
        }
    });

    return out;
}
