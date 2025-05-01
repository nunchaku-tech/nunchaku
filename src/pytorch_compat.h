#pragma once

#include "common.h"
#include "Tensor.h"

namespace pytorch_compat {
inline void TORCH_CHECK(bool cond, const std::string &msg = "") {
    assert(cond);
}

template<typename T>
inline void C10_CUDA_CHECK(T ret) {
    return checkCUDA(ret);
}

namespace at {
using ::Tensor;

constexpr auto kFloat32  = Tensor::FP32;
constexpr auto kFloat    = Tensor::FP32;
constexpr auto kFloat16  = Tensor::FP16;
constexpr auto kBFloat16 = Tensor::BF16;
constexpr auto kInt32    = Tensor::INT32;
constexpr auto kInt64    = Tensor::INT64;

struct Generator {
    Generator() {
        throw std::runtime_error("Not implemented");
    }
    std::mutex mutex_;
};

namespace cuda {
using ::getCurrentDeviceProperties;

struct StreamWrapper {
    cudaStream_t st;
    cudaStream_t stream() const {
        return st;
    }
};
inline StreamWrapper getCurrentCUDAStream() {
    return StreamWrapper(::getCurrentCUDAStream());
}

struct CUDAGuard {
    int dev;
};

namespace detail {
inline Generator getDefaultCUDAGenerator() {
    return Generator();
}
} // namespace detail
} // namespace cuda

using CUDAGeneratorImpl = Generator;

template<typename T>
std::unique_ptr<Generator> get_generator_or_default(std::optional<Generator> gen, T gen2) {
    throw std::runtime_error("Not implemented");
}
} // namespace at

namespace torch {
using at::kFloat32;
using at::kFloat;
using at::kFloat16;
using at::kBFloat16;
using at::kInt32;
using at::kInt64;
constexpr Device kCUDA = Device::cuda();

using IntArrayRef   = std::vector<int>;
using TensorOptions = Tensor::TensorOptions;

inline Tensor empty_like(const Tensor &tensor) {
    return Tensor::empty_like(tensor);
}
inline Tensor empty(TensorShape shape, Tensor::TensorOptions options) {
    return Tensor::empty(shape, options.dtype(), options.device());
}
inline Tensor zeros(TensorShape shape, Tensor::TensorOptions options) {
    return Tensor::empty(shape, options.dtype(), options.device()).zero_();
}

namespace nn {
namespace functional {
using PadFuncOptions = std::vector<int>;
inline Tensor pad(Tensor x, PadFuncOptions options) {
    throw std::runtime_error("Not implemented");
}
} // namespace functional
} // namespace nn

namespace indexing {
constexpr int None = 0;
struct Slice {
    int a;
    int b;
};
} // namespace indexing
} // namespace torch

namespace c10 {
using std::optional;
}

} // namespace pytorch_compat
