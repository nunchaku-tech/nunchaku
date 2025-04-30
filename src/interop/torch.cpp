#include "torch.h"

#include <ATen/cuda/CUDAContext.h>

using spdlog::fmt_lib::format;

template<typename To, typename Ti>
static To int_cast(Ti x) {
    if (x < std::numeric_limits<To>::min() || x > std::numeric_limits<To>::max()) {
        throw std::runtime_error("integer overflow");
    }
    return static_cast<To>(x);
}

Tensor from_torch(at::Tensor input) {
    Tensor result;

    const int ndims = int_cast<int>(input.ndimension());
    for (int i = 0; i < ndims; i++) {
        result.shape.dataExtent.push_back(int_cast<decltype(result.shape.dataExtent)::value_type>(input.size(i)));
        result.shape.dataStride.push_back(int_cast<decltype(result.shape.dataStride)::value_type>(input.stride(i)));
    }

    static const std::map<at::ScalarType, Tensor::ScalarType> mapType = {
        {at::ScalarType::Char, Tensor::INT8},
        {at::ScalarType::Byte, Tensor::INT8},
        {at::ScalarType::Int, Tensor::INT32},
        {at::ScalarType::Long, Tensor::INT64},
        {at::ScalarType::Float, Tensor::FP32},
        {at::ScalarType::Half, Tensor::FP16},
        {at::ScalarType::BFloat16, Tensor::BF16},
        {at::ScalarType::Short, Tensor::INT16},
        {at::ScalarType::Float8_e4m3fn, Tensor::FP8_E4M3},
        {at::ScalarType::Float8_e5m2, Tensor::FP8_E5M2},
    };

    result.scalarType = mapType.at(input.scalar_type());
    result.buffer     = std::make_shared<BufferTorchTensor>(std::move(input));

    Tensor::lockBuffer(result.buffer, getCurrentCUDAStream());

    return result;
}

at::Tensor to_torch(Tensor input) {
    assert(input.is_contiguous());

    std::vector<int64_t> shape;
    for (size_t i = 0; i < input.ndims(); i++) {
        shape.push_back(input.size(i));
    }

    static const std::map<Tensor::ScalarType, at::ScalarType> mapType = {
        {Tensor::INT8, at::ScalarType::Byte},
        {Tensor::INT32, at::ScalarType::Int},
        {Tensor::INT64, at::ScalarType::Long},
        {Tensor::FP32, at::ScalarType::Float},
        {Tensor::FP16, at::ScalarType::Half},
        {Tensor::BF16, at::ScalarType::BFloat16},
        {Tensor::INT16, at::ScalarType::Short},
        {Tensor::FP8_E4M3, at::ScalarType::Float8_e4m3fn},
        {Tensor::FP8_E5M2, at::ScalarType::Float8_e5m2},
    };

    c10::TensorOptions opts(mapType.at(input.scalar_type()));
    if (input.device().type == Device::CPU) {
        opts = opts.device("cpu");
    } else {
        opts = opts.device(format("cuda:{}", input.device().idx));
    }

    at::Tensor result = torch::empty(at::IntArrayRef(shape), opts);
    from_torch(result).copy_(input);

    return result;
}

TorchOpContext::TorchOpContext() {
    stackCUDAStreams.push(at::cuda::getCurrentCUDAStream().stream());
}

TorchOpContext::~TorchOpContext() {
    assert(stackCUDAStreams.top() == at::cuda::getCurrentCUDAStream().stream());
    stackCUDAStreams.pop();
}
