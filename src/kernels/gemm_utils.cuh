#pragma once

#include <cstdint>
#include "common.h"
#include "utils.cuh"

static constexpr int clamp(int val, int min, int max) {
    if (val < min) 
        return min;
    if (val > max)
        return max;
    return val;
}

template<bool shmem = false, typename T>
__device__ __forceinline__
static T load(const T *addr) {
    if constexpr (shmem) {
        if constexpr (sizeof(T) == 8) {
            uint2 data;
            asm volatile ("ld.shared.v2.b32 {%0, %1}, [%2];" : "=r"(data.x), "=r"(data.y) : "l"(__cvta_generic_to_shared(addr)));
            return *reinterpret_cast<T *>(&data);
        }
        if constexpr (sizeof(T) == 16) {
            uint4 data;
            asm volatile ("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(data.x), "=r"(data.y), "=r"(data.z), "=r"(data.w) : "l"(__cvta_generic_to_shared(addr)));
            return *reinterpret_cast<T *>(&data);
        }
        return *addr;
    }

    if constexpr (sizeof(T) == 8) {
        uint2 data = __ldg(reinterpret_cast<const uint2 *>(addr));
        return *reinterpret_cast<T *>(&data);
    }
    if constexpr (sizeof(T) == 16) {
        uint4 data = __ldg(reinterpret_cast<const uint4 *>(addr));
        return *reinterpret_cast<T *>(&data);
    }

    return *addr;
}

template<bool shmem = false, typename T>
__device__ __forceinline__
static void store(T *addr, T val) {
    if constexpr (shmem) {
        if constexpr (sizeof(T) == 8) {
            uint2 data = *reinterpret_cast<uint2 *>(&val);
            asm volatile ("st.shared.v2.b32 [%0], {%1, %2};" ::  "l"(__cvta_generic_to_shared(addr)), "r"(data.x), "r"(data.y));
            return;
        }
        if constexpr (sizeof(T) == 16) {
            uint4 data = *reinterpret_cast<uint4 *>(&val);
            asm volatile ("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(__cvta_generic_to_shared(addr)), "r"(data.x), "r"(data.y), "r"(data.z), "r"(data.w));
            return;
        }
        *addr = val;
        return;
    }

    if constexpr (sizeof(T) == 4) {
        __stcg(reinterpret_cast<unsigned int *>(addr), *reinterpret_cast<unsigned int *>(&val));
        return;
    }
    if constexpr (sizeof(T) == 8) {
        __stcg(reinterpret_cast<uint2 *>(addr), *reinterpret_cast<uint2 *>(&val));
        return;
    }
    if constexpr (sizeof(T) == 16) {
        __stcg(reinterpret_cast<uint4 *>(addr), *reinterpret_cast<uint4 *>(&val));
        return;
    } 
    *addr = val;
}

template<typename T>
__device__ __forceinline__
float2 half22float2(T val);

template<>
__device__ __forceinline__
float2 half22float2<half2>(half2 val) {
    return __half22float2(val);
}

template<>
__device__ __forceinline__
float2 half22float2<__nv_bfloat162>(__nv_bfloat162 val) {
    return __bfloat1622float2(val);
}

template<typename T>
__device__ __forceinline__
T float22half2(float2 val);

template<>
__device__ __forceinline__
half2 float22half2<half2>(float2 val) {
    return __float22half2_rn(val);
}

template<>
__device__ __forceinline__
__nv_bfloat162 float22half2<__nv_bfloat162>(float2 val) {
    return __float22bfloat162_rn(val);
}

template<typename T>
__device__ __forceinline__
void unused_var(T &val, bool alwaysfalse) {
    volatile T *ptr = nullptr;
    if (alwaysfalse) {
        *ptr = val;
    }
}

__device__ __forceinline__ 
static void ldmatrix(const void *ptr, uint4 &out) {
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w)
        : "l"(__cvta_generic_to_shared(ptr))
    );
}


// x in low bit, y in high bit
template<int bitwidth, bool use_unsigned>
__device__ __forceinline__
uint32_t quantize_float2(float2 value) = delete;

template<>
__device__ __forceinline__
uint32_t quantize_float2<4, false>(float2 value) {
    int v1, v2;
    uint32_t result;
    asm volatile ("cvt.rni.s32.f32 %0, %1;" : "=r"(v1) : "f"(value.x));
    asm volatile ("cvt.rni.s32.f32 %0, %1;" : "=r"(v2) : "f"(value.y));
    asm volatile ("cvt.pack.sat.s4.s32.b32 %0, %1, %2, 0;" : "=r"(result) : "r"(v2), "r"(v1));
    return result;
}

template<>
__device__ __forceinline__
uint32_t quantize_float2<4, true>(float2 value) {
    int v1, v2;
    uint32_t result;
    asm volatile ("cvt.rni.s32.f32 %0, %1;" : "=r"(v1) : "f"(value.x));
    asm volatile ("cvt.rni.s32.f32 %0, %1;" : "=r"(v2) : "f"(value.y));
    asm volatile ("cvt.pack.sat.u4.s32.b32 %0, %1, %2, 0;" : "=r"(result) : "r"(v2), "r"(v1));
    return result;
}

template<>
__device__ __forceinline__
uint32_t quantize_float2<8, false>(float2 value) {
    int v1, v2;
    uint32_t result;
    asm volatile ("cvt.rni.s32.f32 %0, %1;" : "=r"(v1) : "f"(value.x));
    asm volatile ("cvt.rni.s32.f32 %0, %1;" : "=r"(v2) : "f"(value.y));
    asm volatile ("cvt.pack.sat.s8.s32.b32 %0, %1, %2, 0;" : "=r"(result) : "r"(v2), "r"(v1));
    return result;
}

__device__ __forceinline__
static float cuda_tanhf(float x) {
    float result;
    asm ("tanh.approx.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
}

__device__ __forceinline__
static float cuda_frcp(float x) {
    float result;
    asm ("rcp.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
}

__device__ __forceinline__
static float cuda_frsqrt(float x) {
    float result;
    asm ("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
}

__device__ __forceinline__
static float cuda_sin(float x) {
    float result;
    asm ("sin.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
}

__device__ __forceinline__
static float cuda_cos(float x) {
    float result;
    asm ("cos.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
}

// https://forums.developer.nvidia.com/t/hardware-accelerated-computation-of-the-sigmoid-logistic-function/266206
__forceinline__ __device__ 
static float cuda_sigmoidf (float a)
{
#if USE_TANH
    return fmaf (0.5, __tanhf (0.5f * a), 0.5f);
#else // USE_TANH
    const float L2E = 1.442695041f; // log2(exp(1))
    float t, d, e, r;
    t = -L2E * a;
    asm ("ex2.approx.ftz.f32 %0,%1;\n\t" : "=f"(e) : "f"(t));
    d = e + 1.0f;
    asm ("rcp.approx.ftz.f32 %0,%1;\n\t" : "=f"(r) : "f"(d));
    return r;
#endif // USE_TANH
}

template<typename T>
__device__ __forceinline__ 
static T gelu_half2(T x) {
    float2 xf  = half22float2<T>(x);
    float2 x3f = xf * xf * xf;
    float t1 = 0.5f + 0.5f * cuda_tanhf(0.79788456f * (xf.x + (0.044715f * x3f.x)));
    float t2 = 0.5f + 0.5f * cuda_tanhf(0.79788456f * (xf.y + (0.044715f * x3f.y)));
    return float22half2<T>(xf * make_float2(t1, t2));
}

template<typename T>
__device__ __forceinline__ 
static T gelu_half(T x) {
    float xf  = float(x);
    float x3f = xf * xf * xf;
    float t = 0.5f + 0.5f * cuda_tanhf(0.79788456f * (xf + (0.044715f * x3f)));
    return (T)(xf * t);
}

template <typename T>
__device__ __forceinline__ 
static T silu(const T &x) {
  // x * sigmoid(x)
  return (T)((float)x * cuda_sigmoidf((float)x));
  // return (T)__fdividef((float)x, 1.0f + __expf((float)-x));
}

__device__ __forceinline__
static void reduce_add(float *addr, float val) {
    asm volatile ("red.relaxed.gpu.global.add.f32 [%0], %1;" :: "l"(addr), "f"(val));
}

template<int cnt, typename F>
__device__ __forceinline__
static void unrolled_loop(F &&lambda) {
    auto call = [&]<int ...Is>(std::integer_sequence<int, Is...>) {
        (lambda.template operator()<Is>(), ...);
    };
    call(std::make_integer_sequence<int, cnt>());
}