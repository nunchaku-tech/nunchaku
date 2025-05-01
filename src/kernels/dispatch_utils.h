#pragma once

#include "common.h"
#include "Tensor.h"
#include <cuda_fp16.h>

template<typename F>
inline auto dispatchFloat(Tensor::ScalarType scalarType, F &&func) {
    switch (scalarType) {
    case Tensor::BF16:
        return func.template operator()<__nv_bfloat16>();
    case Tensor::FP16:
        return func.template operator()<half>();
    case Tensor::FP32:
        return func.template operator()<float>();
    default:
        assert(false);
        throw std::invalid_argument("scalarType is not a floating type");
    }
}

template<typename F>
inline auto dispatchFloat16(Tensor::ScalarType scalarType, F &&func) {
    switch (scalarType) {
    case Tensor::BF16:
        return func.template operator()<__nv_bfloat16>();
    case Tensor::FP16:
        return func.template operator()<half>();
    default:
        assert(false);
        throw std::invalid_argument("scalarType is not a float16 type");
    }
}

template<typename F>
inline auto dispatch(Tensor::ScalarType scalarType, F &&func) {
    switch (scalarType) {
    case Tensor::BF16:
        return func.template operator()<__nv_bfloat16>();
    case Tensor::FP16:
        return func.template operator()<half>();
    case Tensor::FP32:
        return func.template operator()<float>();
    case Tensor::INT8:
        return func.template operator()<int8_t>();
    case Tensor::INT32:
        return func.template operator()<int32_t>();
    case Tensor::INT64:
        return func.template operator()<int64_t>();
    default:
        throw std::runtime_error("Unsupported scalar type");
    }
}

#pragma nv_diagnostic push
// warning #445-D: template parameter "scalar_t" is not used in declaring the parameter types of function template
// "lambda []()->auto::operator auto (*)()"
#pragma nv_diag_suppress 445
template<typename T>
inline bool isTypeMatch(Tensor::ScalarType scalarType) {
    return dispatch(scalarType, []<typename scalar_t>() { return std::is_same_v<scalar_t, T>; });
}
#pragma nv_diagnostic pop

template<typename F, int... N>
inline auto dispatchVal(int val, std::integer_sequence<int, N...>, F &&func) {
    auto call = [&]<int i>() {
        if (val == i) {
            func.template operator()<i>();
        }
    };
    (call.template operator()<N>(), ...);
}

template<typename F>
inline auto dispatchBool(bool val, F &&func) {
    if (val) {
        func.template operator()<true>();
    } else {
        func.template operator()<false>();
    }
}

#define VLLM_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) dispatchFloat(TYPE, [&]<typename scalar_t>() { __VA_ARGS__(); });
