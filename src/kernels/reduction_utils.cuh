/*
 * Adapted from
 * https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/reduce_kernel_utils.cuh
 * Copyright (c) 2023, The vLLM team.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#define FINAL_MASK 0xffffffff

namespace vllm {

template<typename T>
__inline__ __device__ T warpReduceSum(T val) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, mask, 32);
    return val;
}

template<typename T, int NUM>
__inline__ __device__ T warpReduceSumV2(T *val) {
#pragma unroll
    for (int i = 0; i < NUM; i++) {
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            val[i] += __shfl_xor_sync(FINAL_MASK, val[i], mask, 32);
    }
    return (T)(0.0f);
}

/* Calculate the sum of all elements in a block */
template<typename T>
__inline__ __device__ T blockReduceSum(T val) {
    static __shared__ T shared[32];
    int lane = threadIdx.x & 0x1f;
    int wid  = threadIdx.x >> 5;

    val = warpReduceSum<T>(val);

    if (lane == 0)
        shared[wid] = val;

    __syncthreads();

    // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
    // blockDim.x is not divided by 32
    val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : (T)(0.0f);
    val = warpReduceSum<T>(val);
    return val;
}

/* Calculate the sum of all elements in a block */
template<typename T>
__inline__ __device__ T blockAllReduceSum(T val) {
    static __shared__ T shared[32];
    int lane = threadIdx.x & 0x1f;
    int wid  = threadIdx.x >> 5;

    val = warpReduceSum<T>(val);

    if (lane == 0)
        shared[wid] = val;

    __syncthreads();

    // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
    // blockDim.x is not divided by 32
    val = (lane < (blockDim.x / 32.f)) ? shared[lane] : (T)(0.0f);
    val = warpReduceSum<T>(val);
    return val;
}

template<typename T, int NUM>
__inline__ __device__ T blockReduceSumV2(T *val) {
    static __shared__ T shared[NUM][33];
    int lane = threadIdx.x & 0x1f;
    int wid  = threadIdx.x >> 5;

    warpReduceSumV2<T, NUM>(val);

    if (lane == 0) {
#pragma unroll
        for (int i = 0; i < NUM; i++) {
            shared[i][wid] = val[i];
        }
    }

    __syncthreads();

    bool is_mask = threadIdx.x < (blockDim.x / 32.f);
#pragma unroll
    for (int i = 0; i < NUM; i++) {
        val[i] = is_mask ? shared[i][lane] : (T)(0.0f);
    }
    warpReduceSumV2<T, NUM>(val);
    return (T)0.0f;
}

template<typename T>
__inline__ __device__ T warpReduceMax(T val) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val = max(val, __shfl_xor_sync(0xffffffff, val, mask, 32));
    return val;
}
/* Calculate the maximum of all elements in a block */
template<typename T>
__inline__ __device__ T blockReduceMax(T val) {
    static __shared__ T shared[32];
    int lane = threadIdx.x & 0x1f; // in-warp idx
    int wid  = threadIdx.x >> 5;   // warp idx
    val      = warpReduceMax(val); // get maxx in each warp
    if (lane == 0)                 // record in-warp maxx by warp Idx
        shared[wid] = val;
    __syncthreads();
    // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
    // blockDim.x is not divided by 32
    val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : -1e20f;
    val = warpReduceMax(val);
    return val;
}

/* Calculate the maximum of all elements in a block */
template<typename T>
__inline__ __device__ T blockAllReduceMax(T val) {
    static __shared__ T shared[32];
    int lane = threadIdx.x & 0x1f; // in-warp idx
    int wid  = threadIdx.x >> 5;   // warp idx

    val = warpReduceMax(val); // get maxx in each warp

    if (lane == 0) // record in-warp maxx by warp Idx
        shared[wid] = val;

    __syncthreads();

    // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
    // blockDim.x is not divided by 32
    val = (lane < (blockDim.x / 32.f)) ? shared[lane] : -1e20f;
    val = warpReduceMax(val);

    return val;
}

} // namespace vllm
