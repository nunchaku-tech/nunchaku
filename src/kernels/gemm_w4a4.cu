#include "common.h"
#include "Tensor.h"

#include "utils.cuh"
#include "gemm_utils.cuh"

#include "dispatch_utils.h"

#pragma nv_diag_suppress 177

#ifdef _MSC_VER
#define ALWAYSINLINE [[msvc::forceinline]]
#else
#define ALWAYSINLINE __attribute__((always_inline))
#endif

// #define ENABLE_NAN_CHECK 1
#if ENABLE_NAN_CHECK
#define STRINGIZE(x) STRINGIZE2(x)
#define STRINGIZE2(x) #x
#define CHECK_NAN(data, name) checkNan(data, name " at " STRINGIZE(__LINE__))
#else
#define CHECK_NAN(data, name)
#endif

class GEMMConfig_W4A4 {
public:
    // BE CAREFUL: weights need to be repacked when the tiling size changes

    static constexpr int BLOCK_M = 256;
    static constexpr int BLOCK_N = 128;
    static constexpr int WARP_SIZE = 32;
    static constexpr int NUM_WARPS = 8;

    static constexpr int INSN_M = 16;
    static constexpr int INSN_N = 16;
    static constexpr int INSN_K = 64;

#if 0
    using half_t  = half;
    using half2_t = half2;
#else
    using half_t  = __nv_bfloat16;
    using half2_t = __nv_bfloat162;
#endif
};

class GEMMConfig_W8A8 {
public:
    static constexpr int BLOCK_M = 256;
    static constexpr int BLOCK_N = 128;
    static constexpr int WARP_SIZE = 32;
    static constexpr int NUM_WARPS = 8;

    static constexpr int INSN_M = 16;
    static constexpr int INSN_N = 16;
    static constexpr int INSN_K = 32;

    using half_t  = half;
    using half2_t = half2;
};

template<class Config>
class GEMMBase : public Config {
public:
    using Config::BLOCK_M;
    using Config::BLOCK_N;
    using Config::WARP_SIZE;
    using Config::NUM_WARPS;
    using Config::INSN_M;
    using Config::INSN_N;
    using Config::INSN_K;

    using typename Config::half_t;
    using typename Config::half2_t;

    static constexpr int WARP_M = BLOCK_M / NUM_WARPS;
    static constexpr int WARP_N = BLOCK_N;
    static constexpr int WARP_K = INSN_K;

    static constexpr int WARP_M_TILES = WARP_M / INSN_M;
    static constexpr int WARP_N_TILES = WARP_N / INSN_N;
    static constexpr int WARP_K_TILES = WARP_K / INSN_K;

    /**
     * refer to https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#mma-16864-c
     * 
     * wscales store order: (pack = 4)
     *  0   1   8   9   <-- load by lane 0, broadcast to lane {0, 4, 8, ..., 28} (8x)
     *  2   3   10  11  <-- load by lane 1, broadcast to lane {1, 5, 9, ..., 29} (8x)
     *  4   5   12  13  <-- load by lane 2, broadcast to lane {2, 6, 10, ..., 30} (8x)
     *  6   7   14  15  <-- load by lane 3, broadcast to lane {3, 7, 11, ..., 31} (8x)
     * 
     *  16  17  24  25  <-- load by lane 4, broadcast to lane {0, 4, 8, ..., 28} (8x)
     *  ...
     *  22  23  30  31  <-- load by lane 7, broadcast to lane {3, 7, 11, ..., 31} (8x)
     *  ... ...
     *  112 113 120 121 <-- load by lane 28, broadcast to lane {0, 4, 8, ..., 28} (8x)
     *  ...
     *  118 119 126 127 <-- load by lane 31, broadcast to lane {3, 7, 11, ..., 31} (8x)
     *  
     * wscales store order: (pack = 8)
     *  0   1   8   9   16  17  24  25  <-- load by lane 0, broadcast to lane {0, 4, 8, ..., 28} (8x)
     *  2   3   10  11  18  19  26  27  <-- load by lane 1, broadcast to lane {1, 5, 9, ..., 29} (8x)
     *  4   5   12  13  20  21  28  29  <-- load by lane 2, broadcast to lane {2, 6, 10, ..., 30} (8x)
     *  6   7   14  15  22  23  30  31  <-- load by lane 3, broadcast to lane {3, 7, 11, ..., 31} (8x)
     * 
     *  224 225 232 233 240 241 248 249 <-- load by lane 28, broadcast to lane {0, 4, 8, ..., 28} (8x)
     *  ...
     *  230 231 238 239 246 247 254 255 <-- load by lane 31, broadcast to lane {3, 7, 11, ..., 31} (8x)
     * 
     * {k}-th wscale used by lane {i} => {k // (WSCALES_PACK_SIZE * WARP_SIZE)}-th pack, in lane {4*(k // WSCALES_PACK_SIZE) + i % 4}, element {k % WSCALES_PACK_SIZE}
     * 
     * max pack size set to 8 since max load size is 16 bytes / lane
     * min pack size set to 2 since shuffle granularity is 32b 2*half
     * */ 
    static constexpr int WSCALES_PACK_SIZE = clamp(WARP_N / WARP_SIZE, 4 / sizeof(half), 16 / sizeof(half));
    static constexpr int WSCALES_NUM_PACKS = ceilDiv(WARP_N, (WSCALES_PACK_SIZE * WARP_SIZE));
    static constexpr int WSCALES_VALID_LANES = std::min(WARP_SIZE, WARP_N / WSCALES_PACK_SIZE);

    /**
     * ascales store order: (pack = 2)
     *  0   8   <-- load by lane 0, broadcast to lane {0, 1, 2, 3} (4x)
     *  1   9   <-- load by lane 1, broadcast to lane {4, 5, 6, 7} (4x)
     *  2   10
     *  ...
     *  6   14
     *  7   15  <-- load by lane 7, broadcast to lane {28, 29, 30, 31} (4x)
     *  ... ...
     *  48  56  <-- load by lane 24, broadcast to lane {0, 1, 2, 3} (4x)
     *  49  57
     *  ...
     *  54  62
     *  55  63  <-- load by lane 31, broadcast to lane {28, 29, 30, 31}  (4x)
     * 
     * {k}-th wscale used by lane {i} => {k // (ASCALES_PACK_SIZE * WARP_SIZE)}-th pack, in lane {8*(k // ASCALES_PACK_SIZE) + i // 4}, element {k % ASCALES_PACK_SIZE}
     */
    static constexpr int ASCALES_PACK_SIZE = clamp(WARP_M / WARP_SIZE, 4 / sizeof(half), 16 / sizeof(half));
    static constexpr int ASCALES_NUM_PACKS = ceilDiv(WARP_M, (ASCALES_PACK_SIZE * WARP_SIZE));
    static constexpr int ASCALES_VALID_LANES = std::min(WARP_SIZE, WARP_M / ASCALES_PACK_SIZE);

    using packed_act_t = uint4;
    using packed_wgt_t = uint4;
    
    struct alignas(32) packed_psum_t {
        int data[8];
    };
    struct alignas(16) packed_fpsum_t {
        half2_t data[4];
    };
    struct alignas(8) packed_gated_fpsum_t {
        half_t data[4];
    };
    // 16 * 16 matrix
    struct alignas(32) packed_f32psum_t {
        float data[8];

        static constexpr packed_f32psum_t zeros() {
            packed_f32psum_t result;
            for (int i = 0; i < 8; i++) {
                result.data[i] = 0;
            }
            return result;
        };
    };

    struct packed_wscale_t {
        half2_t data[WSCALES_PACK_SIZE / 2];
    };
    struct packed_ascale_t {
        half2_t data[ASCALES_PACK_SIZE / 2];
    };

    using act_warp = std::array<packed_act_t, WARP_M_TILES>;
    using wgt_warp = std::array<packed_wgt_t, WARP_N_TILES>;
    using ascale_warp = std::array<packed_ascale_t, ASCALES_NUM_PACKS>;
    using wscale_warp = std::array<packed_wscale_t, WSCALES_NUM_PACKS>;
    using fpsum_warp = std::array<packed_fpsum_t, WARP_M_TILES * WARP_N_TILES>;
    using f32psum_warp = std::array<packed_f32psum_t, WARP_M_TILES * WARP_N_TILES>;
    using gated_fpsum_warp = std::array<packed_gated_fpsum_t, WARP_M_TILES * WARP_N_TILES>;


    struct BlockInfo {
        int bm;
        int bn;
        int numBlocksM;
        int numBlocksN;
    };

    __device__ __forceinline__
    static packed_f32psum_t mma_f16xf16_f32(packed_fpsum_t a, packed_fpsum_t b, packed_f32psum_t psum) {
        static_assert(std::is_same_v<half_t, half> || std::is_same_v<half_t, __nv_bfloat16>);

        if constexpr (std::is_same_v<half_t, half>) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,  %1,  %2,  %3},"
                "{%4,  %5,  %6,  %7},"
                "{%8,  %9},"
                "{%10,  %11,  %12,  %13};\n"
                : 
                "=f"(psum.data[0]), "=f"(psum.data[1]), "=f"(psum.data[2]), "=f"(psum.data[3])
                : 
                "r"(*reinterpret_cast<unsigned int *>(&a.data[0])), 
                "r"(*reinterpret_cast<unsigned int *>(&a.data[1])), 
                "r"(*reinterpret_cast<unsigned int *>(&a.data[2])), 
                "r"(*reinterpret_cast<unsigned int *>(&a.data[3])),
                "r"(*reinterpret_cast<unsigned int *>(&b.data[0])), 
                "r"(*reinterpret_cast<unsigned int *>(&b.data[1])),

                // "r"(0), "r"(0), "r"(0), "r"(0)
                "f"(psum.data[0]), "f"(psum.data[1]), "f"(psum.data[2]), "f"(psum.data[3])
            );
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,  %1,  %2,  %3},"
                "{%4,  %5,  %6,  %7},"
                "{%8,  %9},"
                "{%10,  %11,  %12,  %13};\n"
                : 
                "=f"(psum.data[4]), "=f"(psum.data[5]), "=f"(psum.data[6]), "=f"(psum.data[7])
                : 
                "r"(*reinterpret_cast<unsigned int *>(&a.data[0])), 
                "r"(*reinterpret_cast<unsigned int *>(&a.data[1])), 
                "r"(*reinterpret_cast<unsigned int *>(&a.data[2])), 
                "r"(*reinterpret_cast<unsigned int *>(&a.data[3])),
                "r"(*reinterpret_cast<unsigned int *>(&b.data[2])), 
                "r"(*reinterpret_cast<unsigned int *>(&b.data[3])),
                // "r"(0), "r"(0), "r"(0), "r"(0)
                "f"(psum.data[4]), "f"(psum.data[5]), "f"(psum.data[6]), "f"(psum.data[7])
            );
        }

        if constexpr (std::is_same_v<half_t, __nv_bfloat16>) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,  %1,  %2,  %3},"
                "{%4,  %5,  %6,  %7},"
                "{%8,  %9},"
                "{%10,  %11,  %12,  %13};\n"
                : 
                "=f"(psum.data[0]), "=f"(psum.data[1]), "=f"(psum.data[2]), "=f"(psum.data[3])
                : 
                "r"(*reinterpret_cast<unsigned int *>(&a.data[0])), 
                "r"(*reinterpret_cast<unsigned int *>(&a.data[1])), 
                "r"(*reinterpret_cast<unsigned int *>(&a.data[2])), 
                "r"(*reinterpret_cast<unsigned int *>(&a.data[3])),
                "r"(*reinterpret_cast<unsigned int *>(&b.data[0])), 
                "r"(*reinterpret_cast<unsigned int *>(&b.data[1])),

                // "r"(0), "r"(0), "r"(0), "r"(0)
                "f"(psum.data[0]), "f"(psum.data[1]), "f"(psum.data[2]), "f"(psum.data[3])
            );
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,  %1,  %2,  %3},"
                "{%4,  %5,  %6,  %7},"
                "{%8,  %9},"
                "{%10,  %11,  %12,  %13};\n"
                : 
                "=f"(psum.data[4]), "=f"(psum.data[5]), "=f"(psum.data[6]), "=f"(psum.data[7])
                : 
                "r"(*reinterpret_cast<unsigned int *>(&a.data[0])), 
                "r"(*reinterpret_cast<unsigned int *>(&a.data[1])), 
                "r"(*reinterpret_cast<unsigned int *>(&a.data[2])), 
                "r"(*reinterpret_cast<unsigned int *>(&a.data[3])),
                "r"(*reinterpret_cast<unsigned int *>(&b.data[2])), 
                "r"(*reinterpret_cast<unsigned int *>(&b.data[3])),
                // "r"(0), "r"(0), "r"(0), "r"(0)
                "f"(psum.data[4]), "f"(psum.data[5]), "f"(psum.data[6]), "f"(psum.data[7])
            );
        }
        
        return psum;
    }

    __device__ __forceinline__
    static packed_fpsum_t packed_fp32_to_fp16(packed_f32psum_t input) {
        packed_fpsum_t results;
        for (int i = 0; i < 4; i++) {
            results.data[i] = float22half2<half2_t>(float2(input.data[i * 2], input.data[i * 2 + 1]));
        }
        return results;
    }

    __device__ __forceinline__
    static packed_f32psum_t packed_fp16_to_fp32(packed_fpsum_t input) {
        packed_f32psum_t results;
        for (int i = 0; i < 4; i++) {
            float2 tmp = half22float2(input.data[i]);
            results.data[i * 2] = tmp.x;
            results.data[i * 2 + 1] = tmp.y;
        }
        return results;
    }

    __device__ __forceinline__
    static fpsum_warp packed_fp32_to_fp16(f32psum_warp input) {
        fpsum_warp results;
    #pragma unroll
        for (int i = 0; i < results.size(); i++) {
            results[i] = packed_fp32_to_fp16(input[i]);
        }
        return results;
    }

    // activation: row major, [M / BLOCK_M, K / WARP_K, NUM_WARPS, WARP_M_TILES, WARP_SIZE] of packed_act_t
    __device__ __forceinline__ 
    static void load_act(const packed_act_t *act, int k, int K, act_warp &out, bool pred) {
        int laneId = threadIdx.x % WARP_SIZE;
        int warpId = threadIdx.x / WARP_SIZE;
    #pragma unroll
        for (int i = 0; i < WARP_M_TILES; i++) {
            if (pred) {
                // out[i] = load(&act[((warpId * WARP_M_TILES + i) * K / WARP_K + k) * WARP_SIZE + laneId]);
                out[i] = load(&act[((k * NUM_WARPS + warpId) * WARP_M_TILES + i) * WARP_SIZE + laneId]);
            }
        }
    }

    // weight: column major: [N / BLOCK_N, 1, K / WARP_K, WARP_N_TILES, WARP_SIZE] of packed_wgt_t
    __device__ __forceinline__ 
    static void load_wgt(const packed_wgt_t *wgt, int k, int K, wgt_warp &out, bool pred) {
        int laneId = threadIdx.x % WARP_SIZE;
        
        // const packed_wgt_t *ptr = &wgt[(0 * K / WARP_K + k) * WARP_SIZE + laneId];
        const packed_wgt_t *ptr = &wgt[(0 + k * WARP_N_TILES) * WARP_SIZE + laneId];
        // int offset = K / WARP_K * WARP_SIZE;
    #pragma unroll
        for (int i = 0; i < WARP_N_TILES; i++) {
            if (pred) {
                // out[i] = load(&wgt[(i * K / WARP_K + k) * WARP_SIZE + laneId]);
                // out[i] = load(&wgt[(i + k * WARP_N_TILES) * WARP_SIZE + laneId]);
                out[i] = load(&ptr[i * WARP_SIZE]);
                // ptr += offset;
            }
        }
    }

    // ascales: row major [M / BLOCK_M, K / group size, NUM_WARPS, ASCALES_NUM_PACKS, ASCALES_VALID_LANES] of packed_ascale_t
    __device__ __forceinline__ 
    static void load_ascale(const packed_ascale_t *ascales, int group, int M, ascale_warp &out, bool pred) {
        int laneId = threadIdx.x % WARP_SIZE;
        int warpId = threadIdx.x / WARP_SIZE;
    #pragma unroll
        for (int i = 0; i < ASCALES_NUM_PACKS; i++) {
            if (pred && laneId < ASCALES_VALID_LANES) {
                // out[i] = ascales[(group * M / WARP_M + warpId) * ASCALES_VALID_LANES * ASCALES_NUM_PACKS + i * ASCALES_VALID_LANES + laneId];
                out[i] = ascales[(group * NUM_WARPS + warpId) * ASCALES_NUM_PACKS * ASCALES_VALID_LANES + i * ASCALES_VALID_LANES + laneId];

            }
        }
    }

    // wscales: column major [N / BLOCK_N, K / group size, 1, WSCALES_NUM_PACKS, WSCALES_VALID_LANES] of packed_wscale_t </del>
    __device__ __forceinline__
    static void load_wscale(const packed_wscale_t *wscales, int group, int N, wscale_warp &out, bool pred) {
        int laneId = threadIdx.x % WARP_SIZE;

        // static_assert(WSCALES_NUM_PACKS * WSCALES_VALID_LANES == 32);
        // static_assert(sizeof(packed_wscale_t) == 8);

        // const packed_wscale_t *ptr = &wscales[(group * WSCALES_NUM_PACKS + 0) * WSCALES_VALID_LANES + laneId];
        // // const packed_wscale_t *ptr = (const packed_wscale_t *)((const char *)wscales) + ((group * WSCALES_NUM_PACKS + 0) * WSCALES_VALID_LANES + laneId) * sizeof(packed_wscale_t);

    #pragma unroll
        for (int i = 0; i < WSCALES_NUM_PACKS; i++) {
            if (pred && laneId < WSCALES_VALID_LANES) {
                
                // out[i] = wscales[group * N / WARP_N * WSCALES_VALID_LANES * WSCALES_NUM_PACKS + i * WSCALES_VALID_LANES + laneId];
                // out[i] = load(&wscales[group * N / WARP_N * WSCALES_VALID_LANES * WSCALES_NUM_PACKS + i * WSCALES_VALID_LANES + laneId]);
                out[i] = load(&wscales[(group * WSCALES_NUM_PACKS + i) * WSCALES_VALID_LANES + laneId]);
                // out[i] = load(&ptr[i * WSCALES_VALID_LANES]);
            }
        }
    }

    // get {k}-th and {k+1}-th wscale from the block, k must be multiples of 2, k must be uniform across all lanes
    __device__ __forceinline__
    static half2_t broadcast_wscale(wscale_warp block, int k, int laneId) {
        const int packIdx = k / (WSCALES_PACK_SIZE * WARP_SIZE);
        const int srcLane = 4 * (k / WSCALES_PACK_SIZE) + laneId % 4;
        const int elementIdx = k % WSCALES_PACK_SIZE / 2;
        return __shfl_sync(~0, block[packIdx].data[elementIdx], srcLane);
    }
    // get {k}-th and {k+1}-th ascale from the block, k must be multiples of 2, k must be uniform across all lanes
    __device__ __forceinline__
    static half2_t broadcast_ascale(ascale_warp block, int k, int laneId) {
        const int packIdx = k / (ASCALES_PACK_SIZE * WARP_SIZE);
        const int srcLane = 8 * (k / ASCALES_PACK_SIZE) + laneId / 4;
        const int elementIdx = k % ASCALES_PACK_SIZE / 2;
        return __shfl_sync(~0, block[packIdx].data[elementIdx], srcLane);
    }

    template<typename F>
    __device__ __forceinline__
    static void apply_scales(F &&getpsum, ascale_warp ascale, wscale_warp wscale, fpsum_warp &fpsum) {
        const int laneId = threadIdx.x % WARP_SIZE;
        const int warpId = threadIdx.x / WARP_SIZE;

        half2_t asx[WARP_M_TILES];
        half2_t asy[WARP_M_TILES];

        for (int i = 0; i < WARP_M_TILES; i++) {
            half2_t as = broadcast_ascale(ascale, i * 2, laneId);
            asx[i] = half2_t(as.x, as.x);
            asy[i] = half2_t(as.y, as.y);
        }

        for (int j = 0; j < WARP_N_TILES; j++) {
            half2_t ws1 = broadcast_wscale(wscale, j * 4, laneId);
            half2_t ws2 = broadcast_wscale(wscale, j * 4 + 2, laneId);

            for (int i = 0; i < WARP_M_TILES; i++) {
                auto &fsum = fpsum[i * WARP_N_TILES + j];

                packed_psum_t psum = getpsum(i, j);

                // constexpr int target = 0;
                // if (threadIdx.x == 3 && j == 1 && i == 0) {
                    
                //     printf("before ws2 = %f %f fsum.data[%d] = %f %f\n", (float)ws2.x, (float)ws2.y, target, (float)fsum.data[target].x, (float)fsum.data[target].y);
                // }
                
                fsum.data[0] = __hfma2(float22half2<half2_t>(make_float2(__int2float_rn(psum.data[0]), __int2float_rn(psum.data[1]))), __hmul2(asx[i], ws1), fsum.data[0]);
                fsum.data[1] = __hfma2(float22half2<half2_t>(make_float2(__int2float_rn(psum.data[2]), __int2float_rn(psum.data[3]))), __hmul2(asy[i], ws1), fsum.data[1]);
                fsum.data[2] = __hfma2(float22half2<half2_t>(make_float2(__int2float_rn(psum.data[4]), __int2float_rn(psum.data[5]))), __hmul2(asx[i], ws2), fsum.data[2]);
                fsum.data[3] = __hfma2(float22half2<half2_t>(make_float2(__int2float_rn(psum.data[6]), __int2float_rn(psum.data[7]))), __hmul2(asy[i], ws2), fsum.data[3]);

                // if (threadIdx.x == 3 && j == 1 && i == 0) {
                //     printf("before ws2 = %f %f fsum.data[%d] = %f %f\n", (float)ws2.x, (float)ws2.y, target, (float)fsum.data[target].x, (float)fsum.data[target].y);
                // }
            }
        }
    }

    template<typename F>
    __device__ __forceinline__
    static void apply_scales(F &&getpsum, ascale_warp ascale, wscale_warp wscale, f32psum_warp &fpsum) {
        const int laneId = threadIdx.x % WARP_SIZE;
        const int warpId = threadIdx.x / WARP_SIZE;

        float2 asx[WARP_M_TILES];
        float2 asy[WARP_M_TILES];

        for (int i = 0; i < WARP_M_TILES; i++) {
            half2_t as = broadcast_ascale(ascale, i * 2, laneId);
            asx[i] = half22float2(half2_t(as.x, as.x));
            asy[i] = half22float2(half2_t(as.y, as.y));
        }

        auto fma2 = [](float2 a, float2 b, float &cx, float &cy) ALWAYSINLINE {
            cx += a.x * b.x;
            cy += a.y * b.y;
        };

        for (int j = 0; j < WARP_N_TILES; j++) {
            float2 ws1 = half22float2(broadcast_wscale(wscale, j * 4, laneId));
            float2 ws2 = half22float2(broadcast_wscale(wscale, j * 4 + 2, laneId));

            for (int i = 0; i < WARP_M_TILES; i++) {
                auto &fsum = fpsum[i * WARP_N_TILES + j];

                packed_psum_t psum = getpsum(i, j);

                fma2(make_float2(__int2float_rn(psum.data[0]), __int2float_rn(psum.data[1])), asx[i] * ws1, fsum.data[0], fsum.data[1]);
                fma2(make_float2(__int2float_rn(psum.data[2]), __int2float_rn(psum.data[3])), asy[i] * ws1, fsum.data[2], fsum.data[3]);
                fma2(make_float2(__int2float_rn(psum.data[4]), __int2float_rn(psum.data[5])), asx[i] * ws2, fsum.data[4], fsum.data[5]);
                fma2(make_float2(__int2float_rn(psum.data[6]), __int2float_rn(psum.data[7])), asy[i] * ws2, fsum.data[6], fsum.data[7]);
            }
        }
    }

    /**
     * input: WARP_M of half (in shared memory, per-warp)
     * output: [..., ASCALES_NUM_PACKS, ASCALES_VALID_LANES] in global memory, per-warp
     */
    __device__ __forceinline__
    static void pack_ascales(const half_t *input, packed_ascale_t *output) {
        const int laneId = threadIdx.x % WARP_SIZE;
    #pragma unroll
        for (int j = 0; j < ASCALES_NUM_PACKS; j++) {
            if (laneId < ASCALES_VALID_LANES) {
                packed_ascale_t tmp;
    #pragma unroll
                for (int i = 0; i < ASCALES_PACK_SIZE; i += 2) {
                    tmp.data[i / 2].x = input[j * ASCALES_PACK_SIZE * WARP_SIZE + laneId / 8 * 8 * ASCALES_PACK_SIZE + laneId % 8 + i * 8];
                    tmp.data[i / 2].y = input[j * ASCALES_PACK_SIZE * WARP_SIZE + laneId / 8 * 8 * ASCALES_PACK_SIZE + laneId % 8 + (i + 1) * 8];
                }
                output[j * ASCALES_VALID_LANES + laneId] = tmp;
            }
        }
    }

    /**
     * input: WARP_N of half (in shared memory, per-warp)
     * output: [..., WSCALES_NUM_PACKS, WSCALES_VALID_LANES] in global memory, per-warp
     */
    __device__ __forceinline__
    static void pack_wscales(const half_t *input, packed_wscale_t *output) {
        const int laneId = threadIdx.x % WARP_SIZE;
    #pragma unroll
        for (int j = 0; j < WSCALES_NUM_PACKS; j++) {
            if (laneId < WSCALES_VALID_LANES) {
                packed_wscale_t tmp;
    #pragma unroll
                for (int i = 0; i < WSCALES_PACK_SIZE; i += 2) {
                    tmp.data[i / 2] = *reinterpret_cast<const half2_t *>(&input[j * WSCALES_PACK_SIZE * WARP_SIZE + laneId / 4 * 4 * WSCALES_PACK_SIZE + laneId % 4 * 2 + i * 4]);
                }
                store(&output[j * WSCALES_VALID_LANES + laneId], tmp);
            }
        }
    }

    struct unpack_fpsum {
        // +8 to prevent bank conflicts
        using matrix_t = half_t[8][WARP_N + 8];

        static constexpr int SHMEM_SIZE = sizeof(matrix_t);
        static constexpr int PACK_SIZE = WARP_N / WARP_SIZE;
        using pack_t = std::array<half_t, PACK_SIZE>;

        // F (int rowId, pack_t &pack)
        template<typename ...F>
        __device__ __forceinline__
        void operator()(fpsum_warp fpsum, half_t *output, int stride, void *shmem, F &&...plugins) {
            const int laneId = threadIdx.x % WARP_SIZE;
            
            matrix_t &mat = *reinterpret_cast<matrix_t *>(shmem);


            // pack_t reduce_tmp;
            // constexpr bool enableReduce = !std::is_void_v<FuncReduce>;

            // if constexpr (enableReduce) {
            //     reduce_tmp.fill(reduce_initval);
            //     // reduce_tmp = load<true>(reinterpret_cast<pack_t *>(&reduce_result[laneId * PACK_SIZE]));
            // }
            // auto doReduce = [&reduce_tmp](pack_t pack) {
            //     if constexpr (enableReduce) {
            //         for (int i = 0; i < PACK_SIZE; i++) {
            //             reduce_tmp[i] = FuncReduce()(reduce_tmp[i], pack[i]);
            //         }
            //     }
            // };

            #pragma unroll
            for (int i = 0; i < WARP_M_TILES; i++) {
            #pragma unroll
                for (int j = 0; j < WARP_N_TILES; j++) {
                    packed_fpsum_t &fsum = fpsum[i * WARP_N_TILES + j];
                    int row = laneId / 4;
                    int col = laneId % 4 * 2 + j * INSN_N;
                    *reinterpret_cast<half2_t *>(&mat[row][col + 0]) = fsum.data[0];
                    *reinterpret_cast<half2_t *>(&mat[row][col + 8]) = fsum.data[2];
                }
                __syncwarp();

            #pragma unroll
                for (int row = 0; row < 8; row++) {
                    pack_t pack = *reinterpret_cast<pack_t *>(&mat[row][laneId * PACK_SIZE]);

                    // if constexpr (enableReduce) {
                    //     doReduce(pack);
                    // }

                    (plugins(i * INSN_M + row, pack), ...);

                    store(reinterpret_cast<pack_t *>(&output[(i * INSN_M + row) * stride + laneId * PACK_SIZE]), pack);
                }
                __syncwarp();

            #pragma unroll
                for (int j = 0; j < WARP_N_TILES; j++) {
                    packed_fpsum_t &fsum = fpsum[i * WARP_N_TILES + j];
                    int row = laneId / 4;
                    int col = laneId % 4 * 2 + j * INSN_N;
                    *reinterpret_cast<half2_t *>(&mat[row][col + 0]) = fsum.data[1];
                    *reinterpret_cast<half2_t *>(&mat[row][col + 8]) = fsum.data[3];
                }
                __syncwarp();

            #pragma unroll
                for (int row = 0; row < 8; row++) {
                    pack_t pack = *reinterpret_cast<pack_t *>(&mat[row][laneId * PACK_SIZE]);

                    // if constexpr (enableReduce) {
                    //     doReduce(pack);
                    // }

                    (plugins(i * INSN_M + 8 + row, pack), ...);

                    store(reinterpret_cast<pack_t *>(&output[(i * INSN_M + 8 + row) * stride + laneId * PACK_SIZE]), pack);
                }
                __syncwarp();
            }
            // if (enableReduce) {
            //     store<true>(reinterpret_cast<pack_t *>(&reduce_result[laneId * PACK_SIZE]), reduce_tmp);
            // }
        }
    };

    
    
    

    struct EpilogueDefault {
        // workaround for layout mismatch between host and device code
        struct Arguments { size_t unused; };

        __device__ __forceinline__
        void operator()(const BlockInfo binfo, fpsum_warp fpsum, half_t *out, int M, int N, int K, Arguments args) {
            const int warpId = threadIdx.x / WARP_SIZE;
            
            __shared__ alignas(128) uint8_t shmem[NUM_WARPS][ceilDiv(unpack_fpsum::SHMEM_SIZE, 128) * 128];
            unpack_fpsum()(fpsum, out + warpId * WARP_M * N, N, shmem[warpId]);
        }
    };

    struct EpilogueNop {
        struct Arguments { size_t unused; };

        __device__ __forceinline__
        void operator()(const BlockInfo binfo, fpsum_warp fpsum, half_t *out, int M, int N, int K, Arguments args) {
        }
    };

    template<typename ...Epilogues>
    struct EpilogueCombination {
        using Arguments = std::tuple<typename Epilogues::Arguments...>;

        __device__ __forceinline__
        void operator()(const BlockInfo binfo, fpsum_warp &fpsum, half_t *out, int M, int N, int K, Arguments args) {
            // this function makes intellisense crashes :(
    #if __INTELLISENSE__
            __trap();   // should not happen when actually compiling
    #else
            std::tuple<Epilogues...> epilogues;
            auto run = [&]<size_t idx>() {
                std::get<idx>(epilogues).operator()(binfo, fpsum, out, M, N, K, std::get<idx>(args));
            };
            auto foreach = [&]<size_t ...Is>(std::index_sequence<Is...>) {
                (run.template operator()<Is>(), ...);
            };
            foreach(std::make_index_sequence<sizeof...(Epilogues)>());
    #endif
        }
    };


};

class GEMM_W4A4 : public GEMMBase<GEMMConfig_W4A4> {
public:
    template<bool ACT_UNSIGNED>
    __device__ __forceinline__
    static packed_psum_t mma(packed_act_t act, packed_wgt_t wgt) {
        packed_psum_t psum;

        if constexpr (!ACT_UNSIGNED) {
            asm volatile(
                "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
                "{%0,  %1,  %2,  %3},"
                "{%4,  %5,  %6,  %7},"
                "{%8,  %9},"
                "{%10,  %11,  %12,  %13};\n"
                : 
                "=r"(psum.data[0]), "=r"(psum.data[1]), "=r"(psum.data[2]), "=r"(psum.data[3])
                : 
                "r"(act.x), "r"(act.y), "r"(act.z), "r"(act.w),
                "r"(wgt.x), "r"(wgt.y),
                "r"(0), "r"(0), "r"(0), "r"(0)
                // "r"(psum.data[0]), "r"(psum.data[1]), "r"(psum.data[2]), "r"(psum.data[3])
            );
            asm volatile(
                "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
                "{%0,  %1,  %2,  %3},"
                "{%4,  %5,  %6,  %7},"
                "{%8,  %9},"
                "{%10,  %11,  %12,  %13};\n"
                : 
                "=r"(psum.data[4]), "=r"(psum.data[5]), "=r"(psum.data[6]), "=r"(psum.data[7])
                : 
                "r"(act.x), "r"(act.y), "r"(act.z), "r"(act.w),
                "r"(wgt.z), "r"(wgt.w),
                "r"(0), "r"(0), "r"(0), "r"(0)
                // "r"(psum.data[4]), "r"(psum.data[5]), "r"(psum.data[6]), "r"(psum.data[7])
            );
        }

        if constexpr (ACT_UNSIGNED) {
            asm volatile(
                "mma.sync.aligned.m16n8k64.row.col.s32.u4.s4.s32 "
                "{%0,  %1,  %2,  %3},"
                "{%4,  %5,  %6,  %7},"
                "{%8,  %9},"
                "{%10,  %11,  %12,  %13};\n"
                : 
                "=r"(psum.data[0]), "=r"(psum.data[1]), "=r"(psum.data[2]), "=r"(psum.data[3])
                : 
                "r"(act.x), "r"(act.y), "r"(act.z), "r"(act.w),
                "r"(wgt.x), "r"(wgt.y),
                "r"(0), "r"(0), "r"(0), "r"(0)
                // "r"(psum.data[0]), "r"(psum.data[1]), "r"(psum.data[2]), "r"(psum.data[3])
            );
            asm volatile(
                "mma.sync.aligned.m16n8k64.row.col.s32.u4.s4.s32 "
                "{%0,  %1,  %2,  %3},"
                "{%4,  %5,  %6,  %7},"
                "{%8,  %9},"
                "{%10,  %11,  %12,  %13};\n"
                : 
                "=r"(psum.data[4]), "=r"(psum.data[5]), "=r"(psum.data[6]), "=r"(psum.data[7])
                : 
                "r"(act.x), "r"(act.y), "r"(act.z), "r"(act.w),
                "r"(wgt.z), "r"(wgt.w),
                "r"(0), "r"(0), "r"(0), "r"(0)
                // "r"(psum.data[4]), "r"(psum.data[5]), "r"(psum.data[6]), "r"(psum.data[7])
            );
        }
        
        return psum;
    }

    // template<bool si>
    template<bool use_unsigned>
    __device__ __forceinline__
    static void quantize_w4a4_from_fpsum_warp(const packed_fpsum_t (&fpsum)[INSN_K / INSN_N], packed_act_t &output, half_t *output_scale) {
        const int laneId = threadIdx.x % WARP_SIZE;

        constexpr float QVALUE_MAX_SIGNED = 7.0f;
        constexpr float QVALUE_MAX_UNSIGNED = 15.0f;
        constexpr float RECPI_QVALUE_MAX_SIGNED = 1 / QVALUE_MAX_SIGNED;
        constexpr float RECPI_QVALUE_MAX_UNSIGNED = 1 / QVALUE_MAX_UNSIGNED;

        constexpr float QVALUE_MAX = use_unsigned ? QVALUE_MAX_UNSIGNED : QVALUE_MAX_SIGNED;
        constexpr float RECPI_QVALUE_MAX = use_unsigned ? RECPI_QVALUE_MAX_UNSIGNED : RECPI_QVALUE_MAX_SIGNED;
        // constexpr int QUANTIZE_BITMASK = 0xf;

        // 0 for row 0-7; 1 for row 8-15
        half2_t input[2][INSN_K / INSN_N * 2];

    #pragma unroll
        for (int i = 0; i < INSN_K / INSN_N; i++) {
            input[0][i * 2 + 0] = fpsum[i].data[0];
            input[0][i * 2 + 1] = fpsum[i].data[2];
            input[1][i * 2 + 0] = fpsum[i].data[1];
            input[1][i * 2 + 1] = fpsum[i].data[3];
        }

        half_t maxvalue[2];
        maxvalue[0] = 0;
        maxvalue[1] = 0;
    #pragma unroll
        for (int i = 0; i < INSN_K / INSN_M * 2; i++) {
            half2_t abs0 = __habs2(input[0][i]);
            half2_t abs1 = __habs2(input[1][i]);
            maxvalue[0] = __hmax(maxvalue[0], __hmax(abs0.x, abs0.y));
            maxvalue[1] = __hmax(maxvalue[1], __hmax(abs1.x, abs1.y));
        }
    #pragma unroll
        for (int mask = 2; mask > 0; mask /= 2) {
            maxvalue[0] = __hmax(maxvalue[0], __shfl_xor_sync(~0, maxvalue[0], mask));
            maxvalue[1] = __hmax(maxvalue[1], __shfl_xor_sync(~0, maxvalue[1], mask));
        }
        maxvalue[0] = __shfl_sync(~0, maxvalue[0], laneId / 4 * 4);
        maxvalue[1] = __shfl_sync(~0, maxvalue[1], laneId / 4 * 4);

        float scale[2];
        // scale[0] = float(maxvalue[0]) / QVALUE_MAX;
        // scale[1] = float(maxvalue[1]) / QVALUE_MAX;
        scale[0] = float(maxvalue[0]) * RECPI_QVALUE_MAX;
        scale[1] = float(maxvalue[1]) * RECPI_QVALUE_MAX;
        if (laneId % 4 == 0) {
            output_scale[laneId / 4] = half_t(scale[0]);
            output_scale[laneId / 4 + 8] = half_t(scale[1]);
        }

        float rscale[2];
        // rscale[0] = QVALUE_MAX / float(maxvalue[0]);
        // rscale[1] = QVALUE_MAX / float(maxvalue[1]);
        rscale[0] = cuda_frcp(scale[0]);
        rscale[1] = cuda_frcp(scale[1]);

        uint32_t qpacks[2][INSN_K / INSN_M * 2];
    #pragma unroll
        for (int i = 0; i < INSN_K / INSN_M * 2; i++) {
    #pragma unroll
            for (int j = 0; j < 2; j++) {
                // half2_t hval = __hmul2(input[j][i], half2_t(rscale[j], rscale[j]));
                // float2 fval = half22float2(hval);
                float2 fval = half22float2(input[j][i]) * make_float2(rscale[j], rscale[j]);
                qpacks[j][i] = quantize_float2<4, use_unsigned>(fval) << (laneId % 4 * 8);
            }
        }
        
        // 2 * 8 * 2 = 32 instructions => 256 cycles
    #pragma unroll
        for (int mask = 1; mask <= 2; mask *= 2) {
    #pragma unroll
            for (int i = 0; i < INSN_K / INSN_M * 2; i++) {
    #pragma unroll
                for (int j = 0; j < 2; j++) {
                    qpacks[j][i] |= __shfl_xor_sync(~0, qpacks[j][i], mask);
                }
            }
        }
        // lane 0,1,2,3 / 4,5,6,7 / ...  should have identical qpacks now

    #pragma unroll
        for (int i = 0; i < 4; i++) {
            if (laneId % 4 == i) {
                output.x = qpacks[0][0 + i];
                output.y = qpacks[1][0 + i];
                output.z = qpacks[0][4 + i];
                output.w = qpacks[1][4 + i];
            }
        }
    }

    // loads act of [WARP_M, WARP_N] and stores to fpsum_warp
    struct load_act_to_fpsum {
        using matrix_t = half_t[WARP_M][WARP_N + 8];
        static constexpr size_t SHMEM_SIZE = sizeof(matrix_t);

        __device__ __forceinline__
        void operator()(const half_t *input, int stride, fpsum_warp &out, void *shmem /*, const half_t *smooth_factor */) {
            const int laneId = threadIdx.x % WARP_SIZE;

            matrix_t &mat = *reinterpret_cast<matrix_t *>(shmem);

            constexpr int PACK_SIZE = WARP_N / WARP_SIZE;
            using packed_input = std::array<half_t, PACK_SIZE>;

            // packed_input pack_smooth;
            // if (smooth_factor) {
            //     pack_smooth = load(reinterpret_cast<const packed_input *>(input + laneId * PACK_SIZE));
            // }

            for (int row = 0; row < WARP_M; row++) {
                auto pack = load(reinterpret_cast<const packed_input *>(input + row * stride + laneId * PACK_SIZE));
                // if (smooth_factor) {
                //     for (int i = 0; i < PACK_SIZE; i++) {
                //         pack[i] *= pack_smooth[i];
                //     }
                // }
                store<true>(reinterpret_cast<packed_input *>(&mat[row][laneId * PACK_SIZE]), pack);
            }
            __syncwarp();

            for (int m = 0; m < WARP_M_TILES; m++) {
                for (int n = 0; n < WARP_N_TILES; n++) {
                    const int row = m * INSN_M + laneId % 16;
                    const int col = n * INSN_N + laneId / 16 * 8;
                    uint4 tmp;
                    ldmatrix(&mat[row][col], tmp);
                    *reinterpret_cast<uint4 *>(&out[m * WARP_N_TILES + n]) = tmp;
                }
            }
            __syncwarp();
        }
    };

    

    /**
     * each warp quantizes a INSN_M * INSN_K (16 * 64) matrix
     * input is per-warp (in global memory)
     * output is per-thread (in regs)
     * output_scale is per-warp (in shared memory)
     * shmem must be at least INSN_M * INSN_K * sizeof(element) (16 * 64 * 0.5 = 512 Bytes)
     * default to quantize activation, if quantize weight, input should be column-majored and output should be transposed ({x, y, z, w} = {x, z, y, w})
     */
    __device__ __forceinline__
    static void quantize_w4a4_warp(const half_t *input, int stride, packed_act_t &output, half_t *output_scale, void *shmem) {
        const int laneId = threadIdx.x % WARP_SIZE;

        constexpr int QUANTIZE_BITWIDTH = 4;
        constexpr int QVALUE_MAX = 7;   // 4 bit => [-8, 7]

        // 1 lane = 1 pack
        // 1 warp = 32 lanes = 32 packs = 1 packwarp
        // a pack is {a0, ..., a7} in figure https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=ex2#mma-16864-a
        // PACK_SIZE * 4 = INSN_K / 2
        constexpr int PACK_SIZE = INSN_K / 8;  // = 8 for 4bit
        constexpr int NUM_PACKS_PER_ROW = INSN_K / PACK_SIZE;
        constexpr int NUM_ROWS_PER_PACKWARP = PACK_SIZE * WARP_SIZE / INSN_K;
        constexpr int NUM_PACKWARPS = INSN_M / NUM_ROWS_PER_PACKWARP;
        using packed_input = std::array<half_t, PACK_SIZE>;

        packed_input packs[NUM_PACKWARPS];

        // load
    #pragma unroll
        for (int i = 0; i < NUM_PACKWARPS; i++) {
            int rowId = i * NUM_ROWS_PER_PACKWARP + laneId / NUM_PACKS_PER_ROW;
            int colId = laneId % NUM_PACKS_PER_ROW * PACK_SIZE;
            packs[i] = load(reinterpret_cast<const packed_input *>(input + rowId * stride + colId));
        }

        // find max
        half_t maxvalue[NUM_PACKWARPS];
    #pragma unroll
        for (int i = 0; i < NUM_PACKWARPS; i++) {
            maxvalue[i] =  __habs(packs[i][0]);
    #pragma unroll
            for (int j = 1; j < PACK_SIZE; j++) {
                
                maxvalue[i] = __hmax(maxvalue[i], __habs(packs[i][j]));
            }
        }

        // warp reduce (max)
    #pragma unroll
        for (int mask = NUM_PACKS_PER_ROW / 2; mask > 0; mask /= 2) {
    #pragma unroll
            for (int i = 0; i < NUM_PACKWARPS; i++) {
                maxvalue[i] = __hmax(maxvalue[i], __shfl_xor_sync(~0, maxvalue[i], mask));
            }
        }

        // broadcast (max)
    #pragma unroll
        for (int i = 0; i < NUM_PACKWARPS; i++) {
            maxvalue[i] = __shfl_sync(~0, maxvalue[i], laneId / NUM_PACKS_PER_ROW * NUM_PACKS_PER_ROW);
        }

        // quantize
        using matrix_t = uint32_t[INSN_M][NUM_PACKS_PER_ROW];
        matrix_t &mat = *reinterpret_cast<matrix_t *>(shmem);
    #pragma unroll
        for (int i = 0; i < NUM_PACKWARPS; i++) {
            half_t scale  = maxvalue[i] / half_t(QVALUE_MAX);
            half_t rscale = half_t(QVALUE_MAX) / maxvalue[i];
            if (laneId % NUM_PACKS_PER_ROW == 0) {
                output_scale[i * NUM_ROWS_PER_PACKWARP + laneId / NUM_PACKS_PER_ROW] = scale;
            }

            uint32_t qpack = 0;
    // #pragma unroll
    //         for (int j = 0; j < PACK_SIZE; j++) {
    //             int intvalue = __half2int_rn(packs[i][j] / scale);
    //             intvalue = clamp(intvalue, -QVALUE_MAX, QVALUE_MAX);
    //             qpack |= (intvalue & QUANTIZE_BITMASK) << (QUANTIZE_BITWIDTH * j);
    //         }
    #pragma unroll
            for (int j = 0; j < PACK_SIZE; j += 2) {
                half2_t hval = __hmul2(half2_t(rscale, rscale), half2_t(packs[i][j], packs[i][j + 1]));
                qpack |= quantize_float2<QUANTIZE_BITWIDTH, false>(half22float2(hval)) << (j * QUANTIZE_BITWIDTH);
            }
            mat[i * NUM_ROWS_PER_PACKWARP + laneId / NUM_PACKS_PER_ROW][laneId % NUM_PACKS_PER_ROW] = qpack;
        }
        __syncwarp();
        
        // convert to imma format
        int row = laneId % 16;
        int col = laneId / 16 * 4;
        ldmatrix(&mat[row][col], output);

        __syncwarp();
    }

    // each thread block (1 warp) quantize WARP_M * WARP_K tile (32 * 64)
    struct quantize_w4a4_act_kernel {
        __device__ 
        void operator()(const half_t *input, packed_act_t *output, packed_ascale_t *oscales, int K) {
            const int laneId = threadIdx.x % WARP_SIZE;

            const int bm = blockIdx.x / (BLOCK_M / WARP_M);
            const int bk = blockIdx.y;
            const int warpId = blockIdx.x % (BLOCK_M / WARP_M);

            const int row = blockIdx.x * WARP_M;
            const int col = blockIdx.y * WARP_K;

            __shared__ alignas(128) half_t oscale_shmem[WARP_M];
            __shared__ alignas(128) uint8_t tmp_shmem[INSN_M * INSN_K / 2];

            for (int tileId = 0; tileId < WARP_M_TILES; tileId++) {
                packed_act_t tmpout;

                quantize_w4a4_warp(
                    input + (row + tileId * INSN_M) * K + col,
                    K,
                    tmpout,
                    oscale_shmem + tileId * INSN_M,
                    tmp_shmem
                );

                store(&output[(((bm * K / WARP_K + bk) * NUM_WARPS + warpId) * WARP_M_TILES + tileId) * WARP_SIZE + laneId], tmpout);
            }

            // if (threadIdx.x == 0) {
            //     printf("Block (%d, %d) => offset = %d\n", blockIdx.x, blockIdx.y, (bm * K / WARP_K + bk) * NUM_WARPS + warpId);
            // }
            pack_ascales(oscale_shmem, &oscales[((bm * K / WARP_K + bk) * NUM_WARPS + warpId) * ASCALES_NUM_PACKS * ASCALES_VALID_LANES]);
        }
    };
    
    // each thread block (1 warp) quantize WARP_N * WARP_K tile (128 * 64)
    struct quantize_w4a4_wgt_kernel {
        __device__ 
        void operator()(const half_t *input, packed_wgt_t *output, packed_wscale_t *oscales, int K) {
            const int laneId = threadIdx.x % WARP_SIZE;

            const int bn = blockIdx.x / (BLOCK_N / WARP_N);
            const int bk = blockIdx.y;

            const int col = blockIdx.x * WARP_N;
            const int row = blockIdx.y * WARP_K;

            __shared__ alignas(128) half_t oscale_shmem[WARP_N];
            __shared__ alignas(128) uint8_t tmp_shmem[INSN_M * INSN_K / 2];

            for (int tileId = 0; tileId < WARP_N_TILES; tileId++) {
                packed_wgt_t tmpout;

                quantize_w4a4_warp(
                    input + (col + tileId * INSN_N) * K + row,
                    K,
                    tmpout,
                    oscale_shmem + tileId * INSN_N,
                    tmp_shmem
                );

                std::swap(tmpout.y, tmpout.z);

                store(&output[((bn * K / WARP_K + bk) * WARP_N_TILES + tileId) * WARP_SIZE + laneId], tmpout);
            }

            pack_wscales(oscale_shmem, &oscales[(bn * K / WARP_K + bk) * WSCALES_NUM_PACKS * WSCALES_VALID_LANES]);
        }
    };

    template<bool ACT_UNSIGNED, typename T>
    __device__ __forceinline__
    static void compute(act_warp A, wgt_warp W, ascale_warp ascale, wscale_warp wscale, T &fpsum) {
        apply_scales([&](int i, int j) {
            return mma<ACT_UNSIGNED>(A[i], W[j]);
        }, ascale, wscale, fpsum);
    }

    __device__ __forceinline__
    static void checkNan(fpsum_warp fpsum, const char *info = "") {
#if ENABLE_NAN_CHECK
        const int laneId = threadIdx.x % WARP_SIZE;
        const int warpId = threadIdx.x / WARP_SIZE;

        for (int i = 0; i < fpsum.size(); i++) {
            for (int j = 0; j < 4; j++) {
                bool abnormal = !isfinite((float)fpsum[i].data[j].x) || !isfinite((float)fpsum[i].data[j].y);
                if (abnormal) {
                    printf("abnormal value detected at block.x=%d block.y=%d warpId=%d laneId=%d fpsum_warp (%s) i=%d j=%d data.x=%f data.y=%f\n", 
                        blockIdx.x, blockIdx.y,
                        warpId, laneId,
                        info,
                        i, j,
                        (float)fpsum[i].data[j].x,
                        (float)fpsum[i].data[j].y
                    );
                    __trap();
                }
            }
        }
#endif
    }

    __device__ __forceinline__
    static void checkNan(packed_f32psum_t fpsum, const char *info = "") {
#if ENABLE_NAN_CHECK
        const int laneId = threadIdx.x % WARP_SIZE;
        const int warpId = threadIdx.x / WARP_SIZE;

        for (int j = 0; j < 8; j++) {
            bool abnormal = !isfinite(fpsum.data[j]);
            if (abnormal) {
                printf("abnormal value detected at bm=%d bn=%d warpId=%d laneId=%d packed_f32psum_t (%s) j=%d data=%f\n", 
                    blockIdx.x, blockIdx.y,
                    warpId, laneId,
                    info,
                    j,
                    fpsum.data[j]
                );
                __trap();
            }
        }
#endif
    }

    __device__ __forceinline__
    static void checkNan(packed_fpsum_t fpsum, const char *info = "") {
#if ENABLE_NAN_CHECK
        const int laneId = threadIdx.x % WARP_SIZE;
        const int warpId = threadIdx.x / WARP_SIZE;

        for (int j = 0; j < 4; j++) {
            bool abnormal = !isfinite((float)fpsum.data[j].x) || !isfinite((float)fpsum.data[j].y);
            if (abnormal) {
                printf("abnormal value detected at bm=%d bn=%d warpId=%d laneId=%d packed_fpsum_t (%s) j=%d data.x=%f data.y=%f\n", 
                    blockIdx.x, blockIdx.y,
                    warpId, laneId,
                    info,
                    j,
                    (float)fpsum.data[j].x,
                    (float)fpsum.data[j].y
                );
                __trap();
            }
        }
#endif
    }

    __device__ __forceinline__
    static void checkNan(float data, const char *info = "") {
#if ENABLE_NAN_CHECK
        const int laneId = threadIdx.x % WARP_SIZE;
        const int warpId = threadIdx.x / WARP_SIZE;

        bool abnormal = !isfinite(data);
        if (abnormal) {
            printf("abnormal value detected at bm=%d bn=%d warpId=%d laneId=%d packed_fpsum_t (%s) data=%f\n", 
                blockIdx.x, blockIdx.y,
                warpId, laneId,
                info,
                data
            );
            __trap();
        }
#endif
    }

    // out: [M / BLOCK_M, N / BLOCK_N, NUM_WARPS, 1, NUM_M_TILES, NUM_N_TILES, WARP_SIZE] of fpsum_warp
    template<typename Epilogue, bool ACT_UNSIGNED>
    __device__ __forceinline__
    static void gemm_w4a4_block(
        const BlockInfo binfo,
        const packed_act_t *act,
        const packed_wgt_t *wgt,
        const packed_ascale_t *ascales,
        const packed_wscale_t *wscales,
        // const packed_wscale_t *bias_ptr,
        half_t *out,
        int M, int N, int K, 
        Epilogue::Arguments epilogueArgs,
        bool alwaysfalse)
    {
        constexpr int NUM_STAGES = 2;

        const int laneId = threadIdx.x % WARP_SIZE;
        const int warpId = threadIdx.x / WARP_SIZE;

        act_warp A[NUM_STAGES];  // 8
        wgt_warp W[NUM_STAGES];  // 32
        ascale_warp ascale[NUM_STAGES];  // 1
        wscale_warp wscale[NUM_STAGES];  // 2
        fpsum_warp fpsum;   // 64

        // load_wscale<true>(wscales, wscale[0], true);
        // load_wscale<false>(wscales, wscale[1], true);
        // load_wscale<false>(wscales, wscale[2], true);

        for (int k = 0; k < NUM_STAGES - 1; k++) {
            load_act(act, k, K, A[k], true);
            load_wgt(wgt, k, K, W[k], true);
            load_ascale(ascales, k, M, ascale[k], true);
            load_wscale(wscales, k, N, wscale[k], true);
        }

        for (auto &pack : fpsum) {
    #if 1
            for (int i = 0; i < 4; i++) {
                pack.data[i].x = 0;
                pack.data[i].y = 0;
            }
    #else 
            for (int i = 0; i < 8; i++) {
                pack.data[i] = 0;
            }
    #endif
        }
        
        int dummy = 0;

        for (int k1 = 0; k1 < K / WARP_K; k1 += NUM_STAGES) {
    #pragma unroll
            for (int k2 = 0; k2 < NUM_STAGES; k2++) {
                int nextk = k1 + k2 + NUM_STAGES - 1;
                int idx = (k2 + NUM_STAGES - 1) % NUM_STAGES;
                bool pred = nextk < K / WARP_K;
                load_act(act, nextk, K, A[idx], pred);
                load_wgt(wgt, nextk, K, W[idx], pred);
                load_ascale(ascales, nextk, M, ascale[idx], pred);
                load_wscale(wscales, nextk, N, wscale[idx], pred);
                // load_wscale<false>(wscales, wscale[idx], pred);

                // __syncthreads();
                // if (alwaysfalse) {
                //     dummy = clock();
                // }

                compute<ACT_UNSIGNED>(A[k2], W[k2], ascale[k2], wscale[k2], fpsum);

                if (alwaysfalse) {
                    dummy = clock();
                }

                // asm volatile ("membar.cta;");
            }
        }

        unused_var(dummy, alwaysfalse);

    #if 0
        auto f16psum = packed_fp32_to_fp16(fpsum);
    #else
        auto f16psum = fpsum;
    #endif

        CHECK_NAN(f16psum, "f16psum");

        Epilogue()(binfo, f16psum, out, M, N, K, epilogueArgs);
    }

    template<bool FUSE_GELU, bool USE_UNSIGNED>
    struct EpilogueQuantize {
        struct Arguments {
            packed_act_t *qout;
            packed_ascale_t *oscales;

            half_t shift_value;
            const packed_wscale_t *smooth_factor;
        };

        static constexpr int NUM_PACKS = INSN_K / INSN_N;
        static constexpr int NUM_GROUPS = WARP_N_TILES / NUM_PACKS;

        __device__ __forceinline__
        void apply_quantize(fpsum_warp fpsum, half_t *out, int M, int N, int K, packed_act_t *qout, packed_ascale_t *oscales, half_t shift_value, const packed_wscale_t *smooth_factor) {
            const int laneId = threadIdx.x % WARP_SIZE;
            const int warpId = threadIdx.x / WARP_SIZE;

            __shared__ half_t oscale_shmem[NUM_WARPS][WARP_M];

            wscale_warp smooth;
            load_wscale(smooth_factor, 0, N, smooth, true);

        #pragma unroll
            for (int group = 0; group < NUM_GROUPS; group++) {
        #pragma unroll
                for (int i = 0; i < WARP_M_TILES; i++) {
                    packed_fpsum_t tmp[NUM_PACKS];

        #pragma unroll
                    for (int j = 0; j < NUM_PACKS; j++) {
                        half2_t ws1 = broadcast_wscale(smooth, (group * NUM_PACKS + j) * 4, laneId);
                        half2_t ws2 = broadcast_wscale(smooth, (group * NUM_PACKS + j) * 4 + 2, laneId);
        #pragma unroll
                        for (int k = 0; k < 4; k++) {
                            half2_t  src = fpsum[i * WARP_N_TILES + group * NUM_PACKS + j].data[k];
                            half2_t &dst = tmp[j].data[k];

                            // dst.x = gelu(src.x);
                            // dst.y = gelu(src.y);
                            if constexpr (FUSE_GELU) {
                                dst = gelu_half2(src);
                            } else {
                                dst = src;
                            }

                            dst += half2_t(shift_value, shift_value);
                            // dst = src;
                        }

                        auto h2div = [](half2_t a, half2_t b) ALWAYSINLINE {
                            float2 af = half22float2(a);
                            float2 bf = half22float2(b);
                            float2 of;
                            of.x = __fdividef(af.x, bf.x);
                            of.y = __fdividef(af.y, bf.y);
                            return float22half2<half2_t>(of);
                        };

                        tmp[j].data[0] = h2div(tmp[j].data[0], ws1);
                        tmp[j].data[1] = h2div(tmp[j].data[1], ws1);
                        tmp[j].data[2] = h2div(tmp[j].data[2], ws2);
                        tmp[j].data[3] = h2div(tmp[j].data[3], ws2);
                    }

                    packed_act_t qresult;
                    quantize_w4a4_from_fpsum_warp<USE_UNSIGNED>(tmp, qresult, &oscale_shmem[warpId][i * INSN_M]);
                    store(&qout[((group * NUM_WARPS + warpId) * WARP_M_TILES + i) * WARP_SIZE + laneId], qresult);
                }

                __syncwarp();
                pack_ascales(&oscale_shmem[warpId][0], &oscales[(group * NUM_WARPS + warpId) * ASCALES_NUM_PACKS * ASCALES_VALID_LANES]);
                __syncwarp();
            }
        }

        __device__ __forceinline__
        void operator()(const BlockInfo binfo, fpsum_warp fpsum, half_t *out, int M, int N, int K, Arguments args) {
            const int bm = binfo.bm;
            const int bn = binfo.bn;

            apply_quantize(
                fpsum, out, M, N, K, 
                args.qout + (bm * N / WARP_K + bn * NUM_GROUPS) * NUM_WARPS * WARP_M_TILES * WARP_SIZE,
                args.oscales + (bm * N / WARP_K + bn * NUM_GROUPS) * NUM_WARPS * ASCALES_NUM_PACKS * ASCALES_VALID_LANES,
                args.shift_value,
                args.smooth_factor + bn * WSCALES_NUM_PACKS * WSCALES_VALID_LANES
            );
        }
    };
    // using EpilogueQuantizeFuseGelu = EpilogueQuantize<true>;

    template<int rank = 32>
    struct Lora {
        static_assert(rank % 16 == 0);

        static constexpr int LORA_RANK = rank;
        static constexpr int LORA_M_TILES = WARP_M / 16;
        static constexpr int LORA_R_TILES = LORA_RANK / 16;
        static constexpr int LORA_N_TILES = WARP_N / 16;

        static_assert(LORA_M_TILES == WARP_M_TILES);
        static_assert(LORA_N_TILES == WARP_N_TILES);
        
        // lora_down: [WARP_M, WARP_N] x [WARP_N, R] (row-wise) = [WARP_M, R]
        // lora up:   [WARP_M, R]      x [WARP_N, R] (col-wise) = [WARP_M, WARP_N]
        // we use fp32 for lora activation since there's no bf16 reduction in sm_89 :(

        using lora_act_warp   = std::array<packed_f32psum_t, LORA_M_TILES * LORA_R_TILES>;
        using lora_act16_warp = std::array<packed_fpsum_t, LORA_M_TILES * LORA_R_TILES>;
        using lora_wgt_warp = std::array<packed_fpsum_t, LORA_N_TILES * LORA_R_TILES>;

        using scale_t = std::array<float, LORA_R_TILES>;

        // lora_wgt:   [N / 16, LORA_R_TILES, WARP_SIZE] of packed_fpsum_t
        __device__ __forceinline__
        static lora_wgt_warp load_lora_wgt(const packed_fpsum_t *ptr) {
            const int laneId = threadIdx.x % WARP_SIZE;

            const packed_fpsum_t *ptr_lane = ptr + laneId;

            lora_wgt_warp result;
    #if 0
        #pragma unroll
            for (int n = 0; n < LORA_N_TILES; n++) {
        #pragma unroll
                for (int r = 0; r < LORA_R_TILES; r++) {
                    result[n * LORA_R_TILES + r] = load(ptr_lane + (n * LORA_R_TILES + r) * WARP_SIZE);
                }
            }
    #else
            unrolled_loop<LORA_N_TILES>([&]<int n>() {
                unrolled_loop<LORA_R_TILES>([&]<int r>() {
                    constexpr int offset = (n * LORA_R_TILES + r) * WARP_SIZE;
                    result[n * LORA_R_TILES + r] = load(ptr_lane + offset);
                });
            });
    #endif
            return result;
        }

        // lora_act: [M / BLOCK_M, NUM_WARPS, LORA_M_TILES, LORA_R_TILES, 8, WARP_SIZE] of float
        __device__ __forceinline__
        static lora_act16_warp load_lora_act(const float *ptr, scale_t scales) {
            const int laneId = threadIdx.x % WARP_SIZE;

            const float *ptrlane = ptr + laneId;

            lora_act16_warp result;
    #if 0
        #pragma unroll
            for (int i = 0; i < LORA_M_TILES * LORA_R_TILES; i++) {
                packed_f32psum_t tmp;
        #pragma unroll
                for (int j = 0; j < 8; j++) {
                    const int offset = i * 8 * WARP_SIZE + j * WARP_SIZE;
                    tmp.data[j] = ptrlane[offset];
                    // tmp.data[j] = ptr[i * 8 * WARP_SIZE + j * WARP_SIZE + laneId];
                }
                CHECK_NAN(tmp, "load_lora_act.tmp");
                result[i] = packed_fp32_to_fp16(tmp);
            }
    #else
            unrolled_loop<LORA_M_TILES>([&]<int m>() {
                unrolled_loop<LORA_R_TILES>([&]<int r>{
                    constexpr int i = m * LORA_R_TILES + r;
                    packed_f32psum_t tmp;
                    unrolled_loop<8>([&]<int j>() { 
                        constexpr int offset = i * 8 * WARP_SIZE + j * WARP_SIZE;
                        tmp.data[j] = ptrlane[offset] * scales[r];
                    });
                    CHECK_NAN(tmp, "load_lora_act.tmp");
                    result[i] = packed_fp32_to_fp16(tmp);
                });
            });
    #endif
            return result;
        }
        // no vector reduction in sm_89 :(
        __device__ __forceinline__
        static void reduce_lora_act(float *ptr, lora_act_warp val) {
            const int laneId = threadIdx.x % WARP_SIZE;

            float *ptrlane = ptr + laneId;

        // #pragma unroll
        //     for (int i = 0; i < LORA_M_TILES * LORA_R_TILES; i++) {
        // #pragma unroll
        //         for (int j = 0; j < 8; j++) {
        //             int offset = i * 8 * WARP_SIZE + j * WARP_SIZE;
        //             reduce_add(&ptrlane[offset], val[i].data[j]);
        //         }
        //     }

            unrolled_loop<LORA_M_TILES * LORA_R_TILES>([&]<int i>() {
                unrolled_loop<8>([&]<int j>() {
                    constexpr int offset = i * 8 * WARP_SIZE + j * WARP_SIZE;
                    reduce_add(&ptrlane[offset], val[i].data[j]);
                });
            });
        }

        // __device__ __forceinline__
        // static void reduce_lora_act(float *ptr, lora_act_warp val, int m) {
        //     const int laneId = threadIdx.x % WARP_SIZE;

        //     float *ptrlane = ptr + laneId + m * LORA_R_TILES * 8 * WARP_SIZE;

        //     unrolled_loop<LORA_R_TILES>([&]<int r>() {
        //         unrolled_loop<8>([&]<int j>() {
        //             constexpr int offset = r * 8 * WARP_SIZE + j * WARP_SIZE;
        //             reduce_add(&ptrlane[offset], val[m * LORA_R_TILES + r].data[j]);
        //         });
        //     });
        // }


        struct EpilogueLoraUp {
            struct Arguments {
                const float *lora_act;
                const packed_fpsum_t *lora_wgt_up;
                scale_t scales;
            };

            __device__ __forceinline__
            static void apply_lora_up(fpsum_warp &fpsum, half_t *out, int M, int N, int K, const float *act, const packed_fpsum_t *wgt, const scale_t scales, const BlockInfo binfo) {
                const int laneId = threadIdx.x % WARP_SIZE;
                const int warpId = threadIdx.x / WARP_SIZE;

                if constexpr (rank > 0) {
                    lora_act16_warp lora_act = load_lora_act(act + warpId * (LORA_M_TILES * LORA_R_TILES * 8 * WARP_SIZE), scales);
                    lora_wgt_warp lora_wgt   = load_lora_wgt(wgt);
                    for (int m = 0; m < LORA_M_TILES; m++) {
                        for (int n = 0; n < LORA_N_TILES; n++) {
                            packed_f32psum_t psum = packed_fp16_to_fp32(fpsum[m * WARP_N_TILES + n]);
                            for (int r = 0; r < LORA_R_TILES; r++) {
                                CHECK_NAN(lora_act[m * LORA_R_TILES + r], "lora_act");
                                CHECK_NAN(lora_wgt[n * LORA_R_TILES + r], "lora_wgt");
                                psum = mma_f16xf16_f32(lora_act[m * LORA_R_TILES + r], lora_wgt[n * LORA_R_TILES + r], psum);
                            }
                            fpsum[m * WARP_N_TILES + n] = packed_fp32_to_fp16(psum);
                        }
                    }
                }
            }

            __device__ __forceinline__
            void operator()(const BlockInfo binfo, fpsum_warp &fpsum, half_t *out, int M, int N, int K, Arguments args) {
                const int bm = binfo.bm;
                const int bn = binfo.bn;

                CHECK_NAN(fpsum, "fpsum");

                if constexpr (rank == 0) {
                    return;
                }

                apply_lora_up(
                    fpsum, out, M, N, K,
                    args.lora_act + bm * (NUM_WARPS * LORA_M_TILES * LORA_R_TILES * 8 *  WARP_SIZE),
                    args.lora_wgt_up + bn * (BLOCK_N / 16) * LORA_R_TILES * WARP_SIZE,
                    args.scales,
                    binfo   // for debug
                );

                CHECK_NAN(fpsum, "fpsum");
            }
        };

        struct EpilogueLoraDown {
            struct Arguments {
                const packed_fpsum_t *lora_wgt_down;
                float *lora_act;
            };

            __device__ __forceinline__
            static void apply_lora_down(fpsum_warp &fpsum, half_t *out, int M, int N, int K, float *act, const packed_fpsum_t *wgt) {
                const int laneId = threadIdx.x % WARP_SIZE;
                const int warpId = threadIdx.x / WARP_SIZE;

                if constexpr (rank > 0) {
                    lora_act_warp lora_act;
                    lora_act.fill(packed_f32psum_t::zeros());

                    lora_wgt_warp lora_wgt = load_lora_wgt(wgt);

                    // clock_t dummy = 0;

                #pragma unroll
                    for (int m = 0; m < LORA_M_TILES; m++) {
                #pragma unroll
                        for (int n = 0; n < LORA_N_TILES; n++) {
                #pragma unroll
                            for (int r = 0; r < LORA_R_TILES; r++) {
                                auto &psum = lora_act[m * LORA_R_TILES + r];

                                CHECK_NAN(fpsum[m * WARP_N_TILES + n], "apply_lora_down.fpsum");
                                CHECK_NAN(lora_wgt[n * LORA_R_TILES + r], "apply_lora_down.lora_wgt");

                                psum = mma_f16xf16_f32(fpsum[m * WARP_N_TILES + n], lora_wgt[n * LORA_R_TILES + r], psum);

                                CHECK_NAN(psum, "apply_lora_down.psum");
                            }
                        }
                        // reduce_lora_act(act + warpId * (LORA_M_TILES * LORA_R_TILES * 8 * WARP_SIZE), lora_act, m);

                        // if (alwaysfalse) {
                        //     dummy = clock();
                        // }
                    }

                    reduce_lora_act(act + warpId * (LORA_M_TILES * LORA_R_TILES * 8 * WARP_SIZE), lora_act);

                    // unused_var(dummy, alwaysfalse);
                }

            }

            __device__ __forceinline__
            void operator()(const BlockInfo binfo, fpsum_warp &fpsum, half_t *out, int M, int N, int K, Arguments args) {
                const int bm = binfo.bm;
                const int bn = binfo.bn;

                if constexpr (rank == 0) {
                    return;
                }

                apply_lora_down(
                    fpsum, out, M, N, K,
                    args.lora_act + bm * (NUM_WARPS * LORA_M_TILES * LORA_R_TILES * 8 *  WARP_SIZE),
                    args.lora_wgt_down + bn * (BLOCK_N / 16) * LORA_R_TILES * WARP_SIZE
                );
            }
        };

        struct quantize_w4a4_fuse_lora_kernel {
            static constexpr size_t SHMEM_PER_WARP = ceilDiv<size_t>(load_act_to_fpsum::SHMEM_SIZE, 128) * 128;
            static constexpr size_t SHMEM_SIZE = SHMEM_PER_WARP * NUM_WARPS;

            struct Arguments {
                const half_t *input;
                const packed_wscale_t *smooth_factor;
                packed_act_t *output;
                packed_ascale_t *oscales;
                const packed_fpsum_t *lora_wgt_down;
                float *lora_act;
                int M, N;
            };

            __device__ __forceinline__
            void operator()(Arguments args) 
            {
                const BlockInfo binfo = {
                    .bm = (int)blockIdx.x,
                    .bn = (int)blockIdx.y,
                    .numBlocksM = (int)gridDim.x,
                    .numBlocksN = (int)gridDim.y,
                };

                const int bm = binfo.bm;
                const int bn = binfo.bn;
                const int warpId = threadIdx.x / WARP_SIZE;

                const int m_offset = bm * BLOCK_M + warpId * WARP_M;
                const int n_offset = bn * BLOCK_N;

                extern __shared__ uint8_t shmem[];

                fpsum_warp fpsum;

                // FIXME: smooth factor should change to EpilogueQuantize
                load_act_to_fpsum()(
                    args.input + m_offset * args.N + n_offset,
                    args.N,
                    fpsum,
                    shmem + warpId * SHMEM_PER_WARP
                    // args.smooth_factor ? args.smooth_factor + n_offset : nullptr
                );

                CHECK_NAN(fpsum, "fpsum");
                // for (int i = 0; i < 16; i++) {
                //     printf("bm=%d bn=%d warp=%d lane=%d fpsum[%d][0:1]=%f %f\n", 
                //         bm, bn, warpId, threadIdx.x % WARP_SIZE, i,
                //         (float)fpsum[i].data[0].x, (float)fpsum[i].data[0].y);
                // }

                EpilogueLoraDown()(binfo, fpsum, nullptr, args.M, args.N, 0, typename EpilogueLoraDown::Arguments{
                    .lora_wgt_down = args.lora_wgt_down,
                    .lora_act = args.lora_act,
                });

                EpilogueQuantize<false, false>()(binfo, fpsum, nullptr, args.M, args.N, 0, EpilogueQuantize<false, false>::Arguments{
                    .qout = args.output,
                    .oscales = args.oscales,
                    .shift_value = 0,
                    .smooth_factor = args.smooth_factor
                });

            }
        };
    };
    
    struct EpilogueBias {
        struct Arguments {
            const packed_wscale_t *bias;  // [N / BLOCK_N, WSCALES_NUM_PACKS, WSCALES_VALID_LANES] of packed_wscale_t
        };

        __device__ __forceinline__
        void apply_bias(fpsum_warp &fpsum, half_t *out, int M, int N, int K, const packed_wscale_t *bias) {
            const int laneId = threadIdx.x % WARP_SIZE;

            // if (laneId == 0) {
            //     printf("block.x=%d block.y=%d warpId=%d bias=%p\n", blockIdx.x, blockIdx.y, threadIdx.x / WARP_SIZE, bias);
            // }

            wscale_warp b;
            load_wscale(bias, 0, N, b, true);

            for (int j = 0; j < WARP_N_TILES; j++) {
                half2_t b1 = broadcast_wscale(b, j * 4, laneId);
                half2_t b2 = broadcast_wscale(b, j * 4 + 2, laneId);

                for (int i = 0; i < WARP_M_TILES; i++) {
                    auto &fsum = fpsum[i * WARP_N_TILES + j];

                    fsum.data[0] = __hadd2(fsum.data[0], b1);
                    fsum.data[1] = __hadd2(fsum.data[1], b1);
                    fsum.data[2] = __hadd2(fsum.data[2], b2);
                    fsum.data[3] = __hadd2(fsum.data[3], b2);
                }
            }
        }

        __device__ __forceinline__
        void operator()(const BlockInfo binfo, fpsum_warp &fpsum, half_t *out, int M, int N, int K, Arguments args) {
            const int bn = binfo.bn;
            apply_bias(
                fpsum, out, M, N, K,
                args.bias + bn * WSCALES_NUM_PACKS * WSCALES_VALID_LANES
            );
        }
    };

    struct EpilogueGelu {
        struct Arguments { size_t unused; };

        // static constexpr float SHIFT_VALUE = 0.171875f;

        __device__ __forceinline__
        void operator()(const BlockInfo binfo, fpsum_warp &fpsum, half_t *out, int M, int N, int K, Arguments args) {
        #pragma unroll
            for (int i = 0; i < WARP_M_TILES; i++) {
        #pragma unroll
                for (int j = 0; j < WARP_N_TILES; j++) {
        #pragma unroll
                    for (int k = 0; k < 4; k++) {
                        half2_t &data = fpsum[i * WARP_N_TILES + j].data[k];
                        data = gelu_half2(data);
                        // data = __hadd2(data, half2_t(SHIFT_VALUE, SHIFT_VALUE));
                    }
                }
            }
        }
    };

    // template<int PoolSize = 128>
    struct EpilogueQKVProj {
        struct Arguments {
            half_t *pool_out;         // [M / PoolSize, N]
            const float *rotary_emb;        // [M, HEAD_DIM / 2, ROTARY_EMB_NUM_ELEMENTS]
            const half_t *rmsnorm_weight_q; // [HEAD_DIM]
            const half_t *rmsnorm_weight_k; // [HEAD_DIM]
            float epsilon;
        };

        static constexpr int HEAD_DIM = 128;
        static constexpr int NUM_HEADS_PER_WARP = WARP_N / HEAD_DIM;

        static constexpr int PoolSize = 128;
        static constexpr int NUM_WARPS_PER_POOL = PoolSize / WARP_M;
        static constexpr int NUM_POOLS_PER_BLOCK = BLOCK_M / PoolSize;

        static constexpr int ROTARY_EMB_NUM_ELEMENTS = 2;   // 1 for theta, 2 for {sin, cos} pair

        __device__ __forceinline__
        static void apply(fpsum_warp fpsum, half_t *out, int M, int N, int K, half_t *pool_out, const float *rotary_emb, const half_t *rmsnorm_weight, float epsilon) {
            const int laneId = threadIdx.x % WARP_SIZE;
            const int warpId = threadIdx.x / WARP_SIZE;
            
            __shared__ alignas(128) uint8_t shmem[NUM_WARPS][ceilDiv(unpack_fpsum::SHMEM_SIZE, 128) * 128];

            constexpr int PACK_SIZE = unpack_fpsum::PACK_SIZE;
            using pack_t = unpack_fpsum::pack_t;

            using pack_rope_t = std::array<float, PACK_SIZE / 2 * ROTARY_EMB_NUM_ELEMENTS>;
            constexpr int LANES_PER_HEAD = HEAD_DIM / PACK_SIZE;

            pack_t reduce_tmp;
            __shared__ alignas(128) pack_t pool[NUM_WARPS];

            // load rmsnorm scales
            pack_t rms;
            if (laneId < LANES_PER_HEAD) {
                rms = load(reinterpret_cast<const pack_t *>(&rmsnorm_weight[laneId * PACK_SIZE]));
            }
            if constexpr (LANES_PER_HEAD < WARP_SIZE) {
                for (int i = 0; i < PACK_SIZE; i++) {
                    rms[i] = __shfl_sync(~0, rms[i], laneId % LANES_PER_HEAD);
                }
            }

            const float *rotary_emb_base_addr = &rotary_emb[(warpId * WARP_M) * HEAD_DIM / 2 * ROTARY_EMB_NUM_ELEMENTS + laneId * PACK_SIZE / 2 * ROTARY_EMB_NUM_ELEMENTS];

            CHECK_NAN(fpsum, "fpsum");

            unpack_fpsum()(fpsum, out + warpId * WARP_M * N, N, shmem[warpId], [&](int rowId, pack_t &pack) ALWAYSINLINE {
                // load rope
                pack_rope_t rope; 
                if (laneId < LANES_PER_HEAD) {
                    // freq = load(reinterpret_cast<pack_freq_t *>(&freqs_cis[(warpId * WARP_M + rowId) * HEAD_DIM * 2 + laneId * PACK_SIZE * 2]));
                    rope = load(reinterpret_cast<const pack_rope_t *>(&rotary_emb_base_addr[rowId * HEAD_DIM / 2 * ROTARY_EMB_NUM_ELEMENTS]));
                }
                if constexpr (LANES_PER_HEAD < WARP_SIZE) {
                    for (int i = 0; i < rope.size(); i++) {
                        rope[i] = __shfl_sync(~0, rope[i], laneId % LANES_PER_HEAD);
                    }
                }

                // rmsnorm
                float sqrsum = 0.0f;
                for (int i = 0; i < PACK_SIZE; i++) {
                    sqrsum += float(pack[i]) * float(pack[i]);
                    CHECK_NAN(sqrsum, "sqrsum");
                }
            #pragma unroll
                for (int mask = LANES_PER_HEAD / 2; mask > 0; mask /= 2) {
                    sqrsum += __shfl_xor_sync(~0, sqrsum, mask);
                }
                sqrsum /= HEAD_DIM;
                float coef = cuda_frsqrt(sqrsum + epsilon);
                CHECK_NAN(coef, "coef");

                for (int i = 0; i < PACK_SIZE; i++) {
                    pack[i] *= coef * float(rms[i]);

                    CHECK_NAN(rms[i], "rms.wgt");
                    CHECK_NAN(pack[i], "rms.out");
                }

#if 1
                // rope
                for (int i = 0; i < PACK_SIZE; i += 2) {
                    float2 pack2 = half22float2(half2_t(pack[i], pack[i+1]));
                    
                    CHECK_NAN(freq[i].x, "rope.freq");
                    CHECK_NAN(freq[i].y, "rope.freq");
                    CHECK_NAN(freq[i+1].x, "rope.freq");
                    CHECK_NAN(freq[i+1].y, "rope.freq");

                    // half2_t tmp = __hmul2(freq[i], pack2);
                    // tmp = __hfma2(freq[i+1], pack2, tmp);
                    // pack[i] = tmp.x;
                    // pack[i+1] = tmp.y;

                    // printf("block.x=%d block.y=%d warpId=%d rowId=%d (%d) freqs = %f %f %f %f\n", 
                    //     blockIdx.x, blockIdx.y, warpId, rowId,
                    //     blockIdx.x * BLOCK_M + warpId * WARP_M + rowId,
                    //     (float)freq[i].x, (float)freq[i].y, (float)freq[i+1].x, (float)freq[i+1].y
                    // );
                    // __trap();

                    // half2_t tmp = __hmul2(half2_t(pack2.x, pack2.x), freq[i]);
                    // tmp = __hfma2(half2_t(pack2.y, pack2.y), freq[i+1], tmp);
                    // pack[i] = tmp.x;
                    // pack[i+1] = tmp.y;

                    float sin, cos;

                    if constexpr (ROTARY_EMB_NUM_ELEMENTS == 1) {
                        sin = cuda_sin(rope[i / 2]);
                        cos = cuda_cos(rope[i / 2]);
                    }
                    if constexpr (ROTARY_EMB_NUM_ELEMENTS == 2) {
                        sin = rope[i];
                        cos = rope[i+1];
                    }

                    // pack[i]   = pack2.x * freq[i].x   + pack2.y * freq[i].y;
                    // pack[i+1] = pack2.x * freq[i+1].x + pack2.y * freq[i+1].y;

                    pack[i]   = half_t(pack2.x * cos - pack2.y * sin);
                    pack[i+1] = half_t(pack2.x * sin + pack2.y * cos);

                    CHECK_NAN(pack[i], "rope.out");
                    CHECK_NAN(pack[i+1], "rope.out");
                }
#endif

                // mean pool
                for (int i = 0; i < PACK_SIZE; i++) {
                    reduce_tmp[i] += pack[i];
                }
            });

            if (!pool_out) {
                return;
            }

            store<true>(&pool[warpId], reduce_tmp);
            __syncthreads();

            if (warpId < NUM_POOLS_PER_BLOCK) {
                const int row = warpId * NUM_WARPS_PER_POOL;
                reduce_tmp = load<true>(&pool[row]);

                for (int i = 1; i < NUM_WARPS_PER_POOL; i++) {
                    pack_t pack = load<true>(&pool[row + i]);
                    for (int j = 0; j < PACK_SIZE; j++) {
                        reduce_tmp[j] += pack[j];
                    }
                }
                for (int j = 0; j < PACK_SIZE; j++) {
                    reduce_tmp[j] /= PoolSize;
                }
                
                store(reinterpret_cast<pack_t *>(pool_out + warpId * N), reduce_tmp);
            }
            __syncthreads();
        }

        __device__ __forceinline__
        void operator()(const BlockInfo binfo, fpsum_warp fpsum, half_t *out, int M, int N, int K, Arguments args) {
            const int bm = binfo.bm;
            const int bn = binfo.bn;

            assert(binfo.numBlocksN % 3 == 0);
            const bool is_q = bn < binfo.numBlocksN / 3;
            const bool is_k = !is_q && bn < binfo.numBlocksN / 3 * 2;

            if (is_q || is_k) {
                apply(
                    fpsum, out, M, N, K, 
                    args.pool_out ? args.pool_out + bm * BLOCK_M / PoolSize * N : nullptr,
                    args.rotary_emb + bm * BLOCK_M * (HEAD_DIM / 2 * ROTARY_EMB_NUM_ELEMENTS),
                    is_q ? args.rmsnorm_weight_q : args.rmsnorm_weight_k,
                    args.epsilon
                );
            } else {
                EpilogueDefault()(binfo, fpsum, out, M, N, K, {});
            }
        }
    };

    template<typename Epilogue, bool ACT_UNSIGNED>
    struct gemm_w4a4_kernel {
        __device__
        void operator()(
            const packed_act_t *act,
            const packed_wgt_t *wgt,
            const packed_ascale_t *ascales,
            const packed_wscale_t *wscales,
            // const packed_wscale_t *bias,
            half_t *out,
            int M, int N, int K, 
            Epilogue::Arguments epilogueArgs,
            bool swapBlockXY,
            bool alwaysfalse)
        {
            // printf("Device sizeof(args) = %d", (int)sizeof(epilogueArgs));

            BlockInfo binfo = {
                .bm = (int)blockIdx.x,
                .bn = (int)blockIdx.y,
                .numBlocksM = (int)gridDim.x,
                .numBlocksN = (int)gridDim.y,
            };

            if (swapBlockXY) {
                std::swap(binfo.bm, binfo.bn);
                std::swap(binfo.numBlocksM, binfo.numBlocksN);
            }

            const int bm = binfo.bm;
            const int bn = binfo.bn;

            // bool fusequant = !out;

            gemm_w4a4_block<Epilogue, ACT_UNSIGNED>(
                binfo,
                act + bm * (K / WARP_K) * NUM_WARPS * WARP_M_TILES * WARP_SIZE,
                wgt + bn * (K / WARP_K) * WARP_N_TILES * WARP_SIZE,
                ascales + bm * (K / WARP_K) * NUM_WARPS * ASCALES_NUM_PACKS * ASCALES_VALID_LANES,
                wscales + bn * (K / WARP_K) * WSCALES_NUM_PACKS * WSCALES_VALID_LANES,
                // bias ? bias + bn * WSCALES_NUM_PACKS * WSCALES_VALID_LANES : nullptr,
                out + (bm * BLOCK_M * N) + bn * BLOCK_N,
                // out + (bm * N / BLOCK_N + bn) * NUM_WARPS * WARP_M_TILES * WARP_N_TILES * WARP_SIZE,
                M, N, K,
                epilogueArgs,
                alwaysfalse
            );
        }
    };
    
};

class GEMM_W8A8 : public GEMMBase<GEMMConfig_W8A8> {
public:
    using psum_warp = std::array<packed_psum_t, WARP_M_TILES * WARP_N_TILES>;

    __device__ __forceinline__
    static packed_psum_t mma(packed_act_t act, packed_wgt_t wgt, packed_psum_t psum) {
        // packed_psum_t psum;
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,  %1,  %2,  %3},"
            "{%4,  %5,  %6,  %7},"
            "{%8,  %9},"
            "{%10,  %11,  %12,  %13};\n"
            : 
            "=r"(psum.data[0]), "=r"(psum.data[1]), "=r"(psum.data[2]), "=r"(psum.data[3])
            : 
            "r"(act.x), "r"(act.y), "r"(act.z), "r"(act.w),
            "r"(wgt.x), "r"(wgt.y),
            // "r"(0), "r"(0), "r"(0), "r"(0)
            "r"(psum.data[0]), "r"(psum.data[1]), "r"(psum.data[2]), "r"(psum.data[3])
        );
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,  %1,  %2,  %3},"
            "{%4,  %5,  %6,  %7},"
            "{%8,  %9},"
            "{%10,  %11,  %12,  %13};\n"
            : 
            "=r"(psum.data[4]), "=r"(psum.data[5]), "=r"(psum.data[6]), "=r"(psum.data[7])
            : 
            "r"(act.x), "r"(act.y), "r"(act.z), "r"(act.w),
            "r"(wgt.z), "r"(wgt.w),
            // "r"(0), "r"(0), "r"(0), "r"(0)
            "r"(psum.data[4]), "r"(psum.data[5]), "r"(psum.data[6]), "r"(psum.data[7])
        );
        return psum;
    }

    __device__ __forceinline__
    static void compute(act_warp A, wgt_warp W, psum_warp &psum) {
        const int laneId = threadIdx.x % WARP_SIZE;
        const int warpId = threadIdx.x / WARP_SIZE;

    #pragma unroll
        for (int j = 0; j < WARP_N_TILES; j++) {
    #pragma unroll
            for (int i = 0; i < WARP_M_TILES; i++) {
                psum[i * WARP_N_TILES + j] = mma(A[i], W[j], psum[i * WARP_N_TILES + j]);
            }
        }
    }

    /**
     * each warp quantizes a INSN_M * INSN_K (16 * 32) matrix
     * input is per-warp (in global memory / shared memory)
     * rscales is per-warp (in shared memory)
     * output is per-thread (in regs)
     * shmem must be at least INSN_M * (INSN_K * sizeof(element) + 16) (16 * 32 = 512 Bytes)
     * default to quantize activation, if quantize weight, input should be column-majored and output should be transposed ({x, y, z, w} = {x, z, y, w})
     */
    template<bool input_shmem = false>
    __device__ __forceinline__
    static void quantize_w8a8_warp(const half *input, const half *rscales, int stride, packed_act_t &output,  void *shmem) {
        const int laneId = threadIdx.x % WARP_SIZE;

        constexpr int QUANTIZE_BITWIDTH = 8;
        // constexpr int QUANTIZE_BITMASK = 0xff;
        // constexpr int QVALUE_MAX = 128;   // 4 bit => [-128, 127]

        // 1 lane = 1 pack
        // 1 warp = 32 lanes = 32 packs = 1 packwarp
        // a pack is {a0, ..., a7} in figure https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=ex2#mma-16864-a
        // PACK_SIZE * 4 = INSN_K / 2
        constexpr int PACK_SIZE = INSN_K / 8;  // = 4 for 8bit
        constexpr int NUM_PACKS_PER_ROW = INSN_K / PACK_SIZE;
        constexpr int NUM_ROWS_PER_PACKWARP = PACK_SIZE * WARP_SIZE / INSN_K;
        constexpr int NUM_PACKWARPS = INSN_M / NUM_ROWS_PER_PACKWARP;
        using packed_input = std::array<half, PACK_SIZE>;

        packed_input packs[NUM_PACKWARPS];

        // load
    #pragma unroll
        for (int i = 0; i < NUM_PACKWARPS; i++) {
            int rowId = i * NUM_ROWS_PER_PACKWARP + laneId / NUM_PACKS_PER_ROW;
            int colId = laneId % NUM_PACKS_PER_ROW * PACK_SIZE;
            packs[i] = load<input_shmem>(reinterpret_cast<const packed_input *>(input + rowId * stride + colId));
        }

        // quantize
        using matrix_t = uint32_t[INSN_M][NUM_PACKS_PER_ROW];
        matrix_t &mat = *reinterpret_cast<matrix_t *>(shmem);
    #pragma unroll
        for (int i = 0; i < NUM_PACKWARPS; i++) {
            const int row = i * NUM_ROWS_PER_PACKWARP + laneId / NUM_PACKS_PER_ROW;
            const int col = laneId % NUM_PACKS_PER_ROW;

            half rscale = rscales[row];

            uint32_t qpack = 0;
    #pragma unroll
            for (int j = 0; j < PACK_SIZE; j += 2) {
                half2 hval = __hmul2(make_half2(rscale, rscale), make_half2(packs[i][j], packs[i][j + 1]));
                qpack |= quantize_float2<QUANTIZE_BITWIDTH, false>(__half22float2(hval)) << (j * QUANTIZE_BITWIDTH);
            }
            mat[row][col] = qpack;
        }
        __syncwarp();
        
        // convert to imma format
        int row = laneId % 16;
        int col = laneId / 16 * 4;
        ldmatrix(&mat[row][col], output);

        __syncwarp();
    }

    /**
     * each warp finds absmax from a row
     */
    template<bool fuse_glu = false>
    __device__ __forceinline__
    static half findmax_warp(const half_t *input, half_t *output_shmem, int K, bool alwaysfalse) {
        const int laneId = threadIdx.x % WARP_SIZE;

        using packed_input = std::array<half2_t, 4>;
        using packed_gated_input = std::array<half_t, 4>;

        constexpr int PACK_SIZE = sizeof(packed_input) / sizeof(half_t);
        constexpr int NUM_STAGES = 2;

        half2_t maxvalue2 = { 0, 0 };
        packed_input pack[NUM_STAGES];

    #pragma unroll
        for (int k = 0; k < NUM_STAGES - 1; k++) {
            const int idx = k * PACK_SIZE * WARP_SIZE + laneId * PACK_SIZE;
            if (idx < K) {
                pack[k] = load(reinterpret_cast<const packed_input *>(&input[idx]));
            } else {
                pack[k].fill(make_half2(0, 0));
            }
        }

        // int dummy = 0;

        // FIXME: pipeline does not work
        // TODO: store quantized data to shmem (instead of half)

        for (int k1 = 0; k1 < ceilDiv(K, PACK_SIZE * WARP_SIZE); k1 += NUM_STAGES) {
    #pragma unroll
            for (int k2 = 0; k2 < NUM_STAGES; k2++) {
                
                const int nextidx = (k1 + k2 + NUM_STAGES - 1) * PACK_SIZE * WARP_SIZE + laneId * PACK_SIZE;
                const int nextk2 = (k2 + NUM_STAGES - 1) % NUM_STAGES;

                if (nextidx < K) {
                    pack[nextk2] = load(reinterpret_cast<const packed_input *>(&input[nextidx]));
                } else {
                    pack[nextk2].fill(make_half2(0, 0));
                }

                packed_input &p = pack[k2];

                if constexpr (fuse_glu) {
                    packed_gated_input gated;

                    #pragma unroll
                    for (int j = 0; j < p.size(); j++) {
                        gated[j] = p[j].x * gelu_half(p[j].y);
                        p[j].x = gated[j];
                        p[j].y = 0;
                    }

                    int idx = (k1 + k2) * PACK_SIZE / 2 * WARP_SIZE + laneId * PACK_SIZE / 2;
                    if (idx < K) {
                        store<true>(reinterpret_cast<packed_gated_input *>(&output_shmem[idx]), gated);
                    }
                }

    #pragma unroll
                for (int j = 0; j < p.size(); j++) {
                    maxvalue2 = __hmax2(maxvalue2, __habs2(p[j]));
                }
            }
        }

        // unused_var(dummy, alwaysfalse);

    #pragma unroll
        for (int mask = 32 / 2; mask > 0; mask /= 2) {
            maxvalue2 = __hmax2(maxvalue2, __shfl_xor_sync(~0, maxvalue2, mask));
        }

        return __hmax(maxvalue2.x, maxvalue2.y);
    }

    // each thread block quantize WARP_M * K tile (32 * K)
    template<bool fuse_glu>
    struct quantize_w8a8_act_kernel {
        static bool check(int M, int K) {
            const int K2 = fuse_glu ? K / 2 : K;
            return M % WARP_M == 0 && K2 % WARP_K == 0;
        }
        static dim3 gridSize(int M, int K) {
            return dim3(M / WARP_M);
        }
        static dim3 blockSize(int M, int K) {
            return dim3(NUM_WARPS * 32);
        }
        static size_t smemSize(int M, int K) {
            if constexpr (!fuse_glu) {
                return 0;
            }
            const int K2 = fuse_glu ? K / 2 : K;
            return INSN_M * K2 * sizeof(half_t);
        }

        __device__ 
        void operator()(const half *input, packed_act_t *output, packed_ascale_t *oscales, int K, bool alwaysfalse) {
            // for quantize kernel
            const int laneId = threadIdx.x % WARP_SIZE;
            const int warpId = threadIdx.x / WARP_SIZE;

            const int numWarps = blockDim.x / WARP_SIZE;

            // for GEMM kernel
            const int bm = blockIdx.x / (BLOCK_M / WARP_M);
            const int gemmWarpId = blockIdx.x % (BLOCK_M / WARP_M);


            __shared__ alignas(128) half_t oscale_shmem[WARP_M];
            __shared__ alignas(128) half_t rscale_shmem[WARP_M];
            __shared__ alignas(128) uint8_t tmp_shmem[NUM_WARPS][512];

            const int K2 = fuse_glu ? K / 2 : K;

            // INSN_M * K2
            extern __shared__ uint8_t smem[];
            half_t *shmem = reinterpret_cast<half_t *>(smem);

            for (int tileM = 0; tileM < WARP_M_TILES; tileM++) {

                for (int i = warpId; i < INSN_M; i += numWarps) {
                    const int rowLocal = tileM * INSN_M + i;
                    const int rowGlobal = blockIdx.x * WARP_M + rowLocal;

                    half maxv = findmax_warp<fuse_glu>(input + rowGlobal * K, shmem + i * K2, K, alwaysfalse);
                    oscale_shmem[rowLocal] = maxv / half(127);
                    rscale_shmem[rowLocal] = half(127) / maxv;
                }
                __syncthreads();

                for (int bk = warpId; bk < K2 / WARP_K; bk += numWarps) {
                    const int rowLocal = tileM * INSN_M;
                    const int rowGlobal = blockIdx.x * WARP_M + rowLocal;
                    const int col = bk * WARP_K;

                    packed_act_t tmpout;

                    if constexpr (fuse_glu) {
                        quantize_w8a8_warp<true>(
                            shmem + col,
                            rscale_shmem + rowLocal,
                            K2,
                            tmpout,
                            &tmp_shmem[warpId]
                        );
                    } else {
                        quantize_w8a8_warp<false>(
                            input + rowGlobal * K + col,
                            rscale_shmem + rowLocal,
                            K,
                            tmpout,
                            &tmp_shmem[warpId]
                        );
                    }
                    
                    store(&output[(((bm * K2 / WARP_K + bk) * NUM_WARPS + gemmWarpId) * WARP_M_TILES + tileM) * WARP_SIZE + laneId], tmpout);
                }
                __syncthreads();
            }

            // [M / BLOCK_M, 1, NUM_WARPS, ASCALES_NUM_PACKS, ASCALES_VALID_LANES] of packed_ascale_t
            pack_ascales(oscale_shmem, &oscales[(bm * NUM_WARPS + gemmWarpId) * ASCALES_NUM_PACKS * ASCALES_VALID_LANES]);
        }
    };


    __device__ __forceinline__
    static gated_fpsum_warp apply_glu(fpsum_warp fpsum) {
        gated_fpsum_warp result;
        for (int i = 0; i < WARP_M_TILES; i++) {
            for (int j = 0; j < WARP_N_TILES; j++) {
                for (int k = 0; k < 4; k++) {
                    half_t &dst = result[i * WARP_N_TILES + j].data[k];
                    half2_t src = fpsum[i * WARP_N_TILES + j].data[k];
                    dst = src.x * gelu_half(src.y);
                }
            }
        }
        return result;
    }

    template<typename F>
    __device__ __forceinline__
    static fpsum_warp apply_act(fpsum_warp fpsum, F func) {
        fpsum_warp result;
        for (int i = 0; i < WARP_M_TILES; i++) {
            for (int j = 0; j < WARP_N_TILES; j++) {
                for (int k = 0; k < 4; k++) {
                    half2_t &dst = result[i * WARP_N_TILES + j].data[k];
                    half2_t src = fpsum[i * WARP_N_TILES + j].data[k];
                    dst.x = func(src.x);
                    dst.y = func(src.y);
                }
            }
        }
        return result;
    }

    

    static constexpr int unpack_gated_fpsum_shmem_size = INSN_M * (WARP_N / 2 + 8) * sizeof(half_t);
    __device__ __forceinline__
    static void unpack_gated_fpsum(gated_fpsum_warp fpsum, half_t *output, int stride, void *shmem) {
        const int laneId = threadIdx.x % WARP_SIZE;

        constexpr int PACK_SIZE = WARP_N / 2 / WARP_SIZE;
        using pack_t = std::array<half_t, PACK_SIZE>;

        // +8 to prevent bank conflicts
        using matrix_t = half_t[INSN_M][WARP_N / 2 + 8];
        matrix_t &mat = *reinterpret_cast<matrix_t *>(shmem);

        for (int i = 0; i < WARP_M_TILES; i++) {
            for (int j = 0; j < WARP_N_TILES; j++) {
                packed_gated_fpsum_t &fsum = fpsum[i * WARP_N_TILES + j];
                int row = laneId / 4;
                int col = laneId % 4 + j * INSN_N / 2;
                *reinterpret_cast<half_t *>(&mat[row][col + 0]) = fsum.data[0];
                *reinterpret_cast<half_t *>(&mat[row][col + 4]) = fsum.data[2];
                *reinterpret_cast<half_t *>(&mat[row + 8][col + 4]) = fsum.data[1];
                *reinterpret_cast<half_t *>(&mat[row + 8][col + 4]) = fsum.data[3];
            }
            __syncwarp();

            for (int row = 0; row < INSN_M; row++) {
                pack_t pack = *reinterpret_cast<pack_t *>(&mat[row][laneId * PACK_SIZE]);
                store(reinterpret_cast<pack_t *>(&output[(i * INSN_M + row) * stride + laneId * PACK_SIZE]), pack);
            }
            __syncwarp();
        }
    }

    // out: [M, N] <=> [..., NUM_WARPS, WARP_M, N] of half
    template<typename Epilogue>
    __device__ __forceinline__
    static void gemm_w8a8_block(
        const BlockInfo binfo,
        const packed_act_t *act,
        const packed_wgt_t *wgt,
        const packed_ascale_t *ascales,
        const packed_wscale_t *wscales,
        half_t *out,
        int M, int N, int K, 
        Epilogue::Arguments epilogeParams,
        bool alwaysfalse)
    {
        constexpr int NUM_STAGES = 2;

        const int laneId = threadIdx.x % WARP_SIZE;
        const int warpId = threadIdx.x / WARP_SIZE;

        act_warp A[NUM_STAGES];  // 8
        wgt_warp W[NUM_STAGES];  // 32
        ascale_warp ascale;  // 1
        wscale_warp wscale;  // 2
        psum_warp psum;   // 128

        for (auto &pack : psum) {
            for (int i = 0; i < 8; i++) {
                pack.data[i] = 0;
            }
        }

        // load_wscale<true>(wscales, wscale[0], true);
        // load_wscale<false>(wscales, wscale[1], true);
        // load_wscale<false>(wscales, wscale[2], true);
        
        load_ascale(ascales, 0, M, ascale, true);
        load_wscale(wscales, 0, N, wscale, true);

        for (int k = 0; k < NUM_STAGES - 1; k++) {
            load_act(act, k, K, A[k], true);
            load_wgt(wgt, k, K, W[k], true);
        }
        
        int dummy = 0;

        for (int k1 = 0; k1 < K / WARP_K; k1 += NUM_STAGES) {
    #pragma unroll
            for (int k2 = 0; k2 < NUM_STAGES; k2++) {
                int nextk = k1 + k2 + NUM_STAGES - 1;
                int idx = (k2 + NUM_STAGES - 1) % NUM_STAGES;
                bool pred = nextk < K / WARP_K;
                load_act(act, nextk, K, A[idx], pred);
                load_wgt(wgt, nextk, K, W[idx], pred);
                // load_wscale<false>(wscales, wscale[idx], pred);

                // __syncthreads();
                // if (alwaysfalse) {
                //     dummy = clock();
                // }

                // if (alwaysfalse) {
                //     dummy = clock();
                // }

                compute(A[k2], W[k2], psum);

                // if (alwaysfalse) {
                //     dummy = clock();
                // }

                // asm volatile ("membar.cta;");
            }
        }

        unused_var(dummy, alwaysfalse);

        fpsum_warp fpsum;
        apply_scales([&](int i, int j) {
            return psum[i * WARP_N_TILES + j];
        }, ascale, wscale, fpsum);

        Epilogue()(binfo, fpsum, out, M, N, K, epilogeParams);
    }

    

    // out : [M / BLOCK_M, BLOCK_M, N / BLOCK_N, BLOCK_N]
    template<typename Epilogue>
    struct gemm_w8a8_kernel {
        __device__
        void operator()(
            const packed_act_t *act,
            const packed_wgt_t *wgt,
            const packed_ascale_t *ascales,
            const packed_wscale_t *wscales,
            half_t *out,
            int M, int N, int K, 
            Epilogue::Arguments epilogueArgs,
            bool swapBlockXY,
            bool alwaysfalse)
        {
            BlockInfo binfo = {
                .bm = (int)blockIdx.x,
                .bn = (int)blockIdx.y,
                .numBlocksM = (int)gridDim.x,
                .numBlocksN = (int)gridDim.y,
            };

            if (swapBlockXY) {
                std::swap(binfo.bm, binfo.bn);
                std::swap(binfo.numBlocksM, binfo.numBlocksN);
            }

            const int bm = binfo.bm;
            const int bn = binfo.bn;

            gemm_w8a8_block<Epilogue>(
                binfo,
                act + bm * (K / WARP_K) * NUM_WARPS * WARP_M_TILES * WARP_SIZE,
                wgt + bn * (K / WARP_K) * WARP_N_TILES * WARP_SIZE,
                ascales + bm * (1) * NUM_WARPS * ASCALES_NUM_PACKS * ASCALES_VALID_LANES,   // only 1 group in W8A8
                wscales + bn * (1) * WSCALES_NUM_PACKS * WSCALES_VALID_LANES,
#if 1
                out + (bm * BLOCK_M * N) + bn * BLOCK_N,
#else
                out + (bm * BLOCK_M * N / 2) + bn * BLOCK_N / 2,
#endif
                M, N, K,
                epilogueArgs,
                alwaysfalse
            );
        }
    };

    

    struct EpilogueGLU {
        struct Arguments { size_t unused; };

        __device__ __forceinline__
        void operator()(const BlockInfo binfo, fpsum_warp fpsum, half_t *out, int M, int N, int K, Arguments args) {
            const int warpId = threadIdx.x / WARP_SIZE;

            gated_fpsum_warp gated_fpsum = apply_glu(fpsum);

            __shared__ alignas(128) uint8_t shmem[NUM_WARPS][ceilDiv(unpack_gated_fpsum_shmem_size, 128) * 128];
            unpack_gated_fpsum(gated_fpsum, out + warpId * WARP_M * N / 2, N / 2, shmem[warpId]);
        }
    };

    struct EpilogueSilu {
        struct Arguments { size_t unused; };

        __device__ __forceinline__
        void operator()(const BlockInfo binfo, fpsum_warp fpsum, half_t *out, int M, int N, int K, Arguments args) {
            const int warpId = threadIdx.x / WARP_SIZE;
            
            fpsum = apply_act(fpsum, [](half_t x) { return silu(x); });

            __shared__ alignas(128) uint8_t shmem[NUM_WARPS][ceilDiv(unpack_fpsum::SHMEM_SIZE, 128) * 128];
            unpack_fpsum()(fpsum, out + warpId * WARP_M * N, N, shmem[warpId]);
        }
    };

    struct EpilogueLiteLA {
        __device__ __forceinline__
        static half2_t movmatrix(half2_t x) {
            asm volatile ("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(*reinterpret_cast<uint32_t *>(&x)) : "r"(*reinterpret_cast<uint32_t *>(&x)));
            return x;
        }

        // __device__ __forceinline__ 
        // static uint4 hmma_fp32(uint4 a, uint2 b, uint4 c) {
        //     asm volatile(
        //         "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        //         "{%0,  %1,  %2,  %3},"
        //         "{%4,  %5,  %6,  %7},"
        //         "{%8,  %9},"
        //         "{%10,  %11,  %12,  %13};\n"
        //         : 
        //         "=r"(c.x), "=r"(c.y), "=r"(c.z), "=r"(c.w)
        //         : 
        //         "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w),
        //         "r"(b.x), "r"(b.y),
        //         // "r"(0), "r"(0), "r"(0), "r"(0)
        //         "r"(c.x), "r"(c.y), "r"(c.z), "r"(c.w)
        //     );
        //     return c;
        // }

        
        
        __device__ __forceinline__
        static packed_f32psum_t mma_litela(packed_fpsum_t k, packed_fpsum_t v, packed_f32psum_t psum) {
            for (int i = 0; i < 4; i++) {
                k.data[i] = movmatrix(k.data[i]);
                v.data[i] = movmatrix(v.data[i]);
            }
            std::swap(v.data[1], v.data[2]);
            return mma_f16xf16_f32(v, k, psum);
        }

        static constexpr int LITELA_HEAD_DIM = 32;
        static constexpr int LITELA_K_TILES = LITELA_HEAD_DIM / 16;
        static constexpr int LITELA_V_TILES = LITELA_HEAD_DIM / 16;

        static constexpr int SHMEM_SIZE = NUM_WARPS * (LITELA_HEAD_DIM + 1) * (LITELA_HEAD_DIM + 8) * sizeof(float);

        // out_vk: [batch_size, num_heads, head_dim + 1, head_dim]
        __device__ __forceinline__
        static void apply_litela(const BlockInfo binfo, fpsum_warp fpsum, float *out_vk, int batch_m) {
            const int laneId = threadIdx.x % WARP_SIZE;
            const int warpId = threadIdx.x / WARP_SIZE;

            using vk_t = float[NUM_WARPS][LITELA_HEAD_DIM + 1][LITELA_HEAD_DIM + 8];
            extern __shared__ uint8_t shmem[];
            
            vk_t &shmem_vk = *reinterpret_cast<vk_t *>(shmem);

            static_assert(sizeof(vk_t) == SHMEM_SIZE);
            static_assert(WARP_N == BLOCK_N);
            assert(binfo.numBlocksN % 3 == 0);

            const int num_heads = binfo.numBlocksN / 3 * 2 * (WARP_N / (LITELA_HEAD_DIM * 2));
            const int batch_id = binfo.bm * BLOCK_M / batch_m;

            for (int head_id = 0; head_id < WARP_N / (LITELA_HEAD_DIM * 2); head_id++) {
                const int global_head_id = (binfo.bn - binfo.numBlocksN / 3) * (WARP_N / (LITELA_HEAD_DIM * 2)) + head_id;
                float *out_vk_current_head = out_vk + (batch_id * num_heads + global_head_id) * (LITELA_HEAD_DIM + 1) * LITELA_HEAD_DIM;

                for (int i = laneId; i < sizeof(shmem_vk) / sizeof(float) / NUM_WARPS; i += WARP_SIZE) {
                    *((&shmem_vk[warpId][0][0]) + i) = 0;
                }
                __syncwarp();

                for (int tile_v = 0; tile_v < LITELA_V_TILES; tile_v++) {
                    for (int tile_k = 0; tile_k < LITELA_K_TILES; tile_k++) {
                        packed_f32psum_t attn_sum = { 0 };
                        for (int i = 0; i < WARP_M_TILES; i++) {
                            packed_fpsum_t k = fpsum[i * WARP_N_TILES + head_id * (LITELA_HEAD_DIM * 2) / 16 + tile_k];
                            packed_fpsum_t v = fpsum[i * WARP_N_TILES + head_id * (LITELA_HEAD_DIM * 2) / 16 + LITELA_HEAD_DIM / 16 + tile_v];
                            for (int j = 0; j < 4; j++) {
                                k.data[j] = __hmax2(k.data[j], make_half2(0, 0));  // relu
                            }
                            attn_sum = mma_litela(k, v, attn_sum);
                        }

                        const int row = tile_v * 16 + laneId / 4;
                        const int col = tile_k * 16 + laneId % 4 * 2;

                        shmem_vk[warpId][row + 0][col + 0] = attn_sum.data[0];
                        shmem_vk[warpId][row + 0][col + 1] = attn_sum.data[1];
                        shmem_vk[warpId][row + 8][col + 0] = attn_sum.data[2];
                        shmem_vk[warpId][row + 8][col + 1] = attn_sum.data[3];
                        shmem_vk[warpId][row + 0][col + 8] = attn_sum.data[4];
                        shmem_vk[warpId][row + 0][col + 9] = attn_sum.data[5];
                        shmem_vk[warpId][row + 8][col + 8] = attn_sum.data[6];
                        shmem_vk[warpId][row + 8][col + 9] = attn_sum.data[7];
                    }
                }
                for (int tile_k = 0; tile_k < LITELA_K_TILES; tile_k++) {
                    packed_f32psum_t attn_sum = { 0 };
                    for (int i = 0; i < WARP_M_TILES; i++) {
                        packed_fpsum_t k = fpsum[i * WARP_N_TILES + head_id * (LITELA_HEAD_DIM * 2) / 16 + tile_k];
                        packed_fpsum_t v = {};  // TODO fill to 0
                        if (laneId < 4) {
                            v.data[0] = make_half2(1, 1);
                            v.data[2] = make_half2(1, 1);
                        }
                        attn_sum = mma_litela(k, v, attn_sum);
                    }
                    const int row = LITELA_HEAD_DIM + laneId / 4;
                    const int col = tile_k * 16 + laneId % 4 * 2;

                    if (laneId < 4) {
                        shmem_vk[warpId][row + 0][col + 0] = attn_sum.data[0];
                        shmem_vk[warpId][row + 0][col + 1] = attn_sum.data[1];
                        shmem_vk[warpId][row + 0][col + 8] = attn_sum.data[4];
                        shmem_vk[warpId][row + 0][col + 9] = attn_sum.data[5];
                    }
                }
                __syncthreads();

                for (int i = warpId; i < LITELA_HEAD_DIM + 1; i += NUM_WARPS) {
                    for (int j = laneId; j < LITELA_HEAD_DIM; j += WARP_SIZE) {
                        float sum = 0;
                        for (int k = 0; k < NUM_WARPS; k++) {
                            sum += shmem_vk[k][i][j];
                        }
                        reduce_add(&out_vk_current_head[i * LITELA_HEAD_DIM + j], sum);
                    }
                }
                __syncthreads();
            }
        }

        struct Arguments {
            half_t *out_q;
            float *out_vk;
            int batch_m;
        };

        __device__ __forceinline__
        void operator()(const BlockInfo binfo, fpsum_warp fpsum, half_t *out, int M, int N, int K, Arguments args) {
            const int bm = binfo.bm;
            const int bn = binfo.bn;

            if (bn < binfo.numBlocksN / 3) {
                fpsum = apply_act(fpsum, [](half_t x) { return __hmax(x, 0); });    // relu
                return EpilogueDefault()(
                    binfo,
                    fpsum, 
                    args.out_q + (bm * BLOCK_M * N / 3) + bn * BLOCK_N, 
                    M, N / 3, K, EpilogueDefault::Arguments{});
            }

            return apply_litela(binfo, fpsum, args.out_vk, args.batch_m);
        }

        // each thread block mults BlockSize*HEAD_DIM q and (HEAD_DIM+1)*HEAD_DIM vk, in-place writes back to q
        // q:   [batch_size, #blocks, block_size, #heads, HEAD_DIM]
        // vk:  [batch_size, #heads, HEAD_DIM+1, HEAD_DIM]
        struct vk_mul_q_kernel {
            // FIXME FIXME FIXME
            __device__
            void operator()(half_t *q, const float *vk, float eps) {
                const int block_id = blockIdx.x;
                const int head_id  = blockIdx.y;
                const int batch_id = blockIdx.z;

                const int num_blocks = gridDim.x;
                const int num_heads = gridDim.y;
                const int block_size = blockDim.x;

                half_t *localq = &q[(((batch_id * num_blocks + block_id) * block_size + threadIdx.x) * num_heads + head_id) * LITELA_HEAD_DIM];
                const float *localvk = &vk[(batch_id * num_heads + head_id) * (LITELA_HEAD_DIM + 1) * LITELA_HEAD_DIM];
                // half_t *localout = &out[(((batch_id * num_blocks + block_id) * block_size + threadIdx.x) * num_heads + head_id) * LITELA_HEAD_DIM];

                using packed_q = std::array<half_t, 8>;
                using packed_vk = std::array<float, 4>;

                half_t qblock[LITELA_HEAD_DIM];
                for (int i = 0; i < LITELA_HEAD_DIM; i += sizeof(packed_q) / sizeof(half_t)) {
                    *reinterpret_cast<packed_q *>(&qblock[i]) = load(reinterpret_cast<const packed_q *>(&localq[i]));
                }

                float outblock[LITELA_HEAD_DIM + 1];
            #pragma unroll
                for (int j = 0; j < LITELA_HEAD_DIM + 1; j++) {
                    outblock[j] = 0;
            #pragma unroll
                    for (int i = 0; i < LITELA_HEAD_DIM; i += sizeof(packed_vk) / sizeof(float)) {
                        packed_vk vkpack = load(reinterpret_cast<const packed_vk *>(&localvk[j * LITELA_HEAD_DIM + i]));
            #pragma unroll
                        for (int k = 0; k < vkpack.size(); k++) {
                            outblock[j] += (float)qblock[i + k] * vkpack[k];
                        }
                    }
                }
                
                for (int i = 0; i < LITELA_HEAD_DIM; i += sizeof(packed_q) / sizeof(half_t)) {
                    packed_q opack;
                    for (int k = 0; k < opack.size(); k++) {
                        opack[k] = __fdividef(outblock[i + k], outblock[LITELA_HEAD_DIM] + eps);
                    }
                    store(reinterpret_cast<packed_q *>(&localq[i]), opack);
                }
            }
        };
    };

    
};

template<typename kernel, typename ...T>
__global__
static void invoke_kernel(T ...args) {
    kernel()(args...);
}

template<typename T>
__global__
static void test_sizeof_device() {
    printf("sizeof on device = %d\n", (int)sizeof(T));
}

template<typename T>
static void test_sizeof_host() {
    printf("sizeof on host = %d\n", (int)sizeof(T));
}

template<typename T>
static void test_sizeof() {
    printf("typeid = %s\n", typeid(T).name());
    test_sizeof_host<T>();
    test_sizeof_device<T><<<1, 1>>>();
    checkCUDA(cudaDeviceSynchronize());
}

void gemm_w4a4(
        Tensor act,          // packed act [M, K / 2]
        Tensor wgt,          // packed act [N, K / 2]
        Tensor out,          // linear     [M, N]
        Tensor qout,         // packed act [M, N / 2]
        Tensor ascales,      // packed as  [K / 64, M]
        Tensor wscales,      // packed ws  [K / 64, N]
        Tensor oscales,      // packed as  [N / 64, M]
        Tensor poolout,      // linear     [M / PoolSize, N]
        Tensor lora_act_in,  // packed lora_act [M, R]
        Tensor lora_up,      // packed lora_wgt [N, R]
        Tensor lora_down,    // packed lora_wgt [N, R]
        Tensor lora_act_out, // packed lora_act [M, R]
        Tensor norm_q,       // linear     [HEAD_DIM]
        Tensor norm_k,       // linear     [HEAD_DIM]
        Tensor rotary_emb,   // linear     [M, HEAD_DIM / 2, 2, 2]
        Tensor bias,         // packed ws  [N]
        Tensor smooth_factor, // packed ws  [N], for quantization of the next layer
        bool act_unsigned,
        std::vector<float> lora_scales  // [R / 16]
) {
    using GEMM = GEMM_W4A4;

    int M = act.numel() / act.shape[-1];
    int N = wgt.shape[0];
    int K = act.shape[-1] * 2;
    assert(K == wgt.shape[1] * 2);

    // spdlog::info("M={} N={} K={}", M, N, K);
    // spdlog::info("act at {}", act.data_ptr());
    // spdlog::info("wgt at {}", wgt.data_ptr());
    // spdlog::info("ascales at {}", ascales.data_ptr());
    // spdlog::info("wscales at {}", wscales.data_ptr());
    // spdlog::info("bias at {}", bias.data_ptr());

    auto launch = [&]<typename Epilogue>(Epilogue::Arguments args) {
        dim3 grid(M / GEMM::BLOCK_M, N / GEMM::BLOCK_N);

        bool swapBlockMN = M > N * 2;
        if (swapBlockMN) {
            std::swap(grid.x, grid.y);
        }

        dispatchBool(act_unsigned, [&]<bool ACT_UNSIGNED>() {
            // test_sizeof<typename Epilogue::Arguments>();
            // std::apply([](auto ...args) {
            //     (test_sizeof<decltype(args)>(), ...);
            // }, args);
            invoke_kernel<GEMM::gemm_w4a4_kernel<Epilogue, ACT_UNSIGNED>><<<grid, GEMM::WARP_SIZE * GEMM::NUM_WARPS>>>(
                act.data_ptr<GEMM::packed_act_t>(),
                wgt.data_ptr<GEMM::packed_wgt_t>(),
                ascales.data_ptr<GEMM::packed_ascale_t>(),
                wscales.data_ptr<GEMM::packed_wscale_t>(),
                // bias.valid() ? bias.data_ptr<GEMM::packed_wscale_t>() : nullptr,
                out.valid() ? out.data_ptr<GEMM::half_t>() : nullptr,
                M, N, K,
                args,
                swapBlockMN,
                false
            );
            checkCUDA(cudaGetLastError());
        });
    };

    auto launch_bias = [&]<typename NextEpilogue>(NextEpilogue::Arguments nextArgs) {
        if (!bias.valid()) {
            return launch.template operator()<NextEpilogue>(nextArgs);
        }

        assert(bias.numel() == N);

        // append EpilgoueNop to workaround mismatched memory layout of std::tuple between device and host code on Windows
        // ** sizeof(std::tuple<std::tuple<int>>) == 8 on device **
        using Epilogue = GEMM::EpilogueCombination<GEMM::EpilogueBias, NextEpilogue, GEMM::EpilogueNop>;
        return launch.template operator()<Epilogue>({
            GEMM::EpilogueBias::Arguments{
                .bias = bias.data_ptr<GEMM::packed_wscale_t>(),
            },
            nextArgs,
            {}
        });
    };
    // auto launch_bias = launch;

    auto launch_lora = [&]<typename NextEpilogue, typename MidEpilogue>(NextEpilogue::Arguments nextArgs, MidEpilogue::Arguments midArgs) {
        assert(lora_up.valid() == lora_act_in.valid());
        assert(lora_down.valid() == lora_act_out.valid());

        if (!lora_up.valid()) {
            assert(!lora_down.valid());
            return launch_bias.template operator()<GEMM::EpilogueCombination<MidEpilogue, NextEpilogue>>({midArgs, nextArgs});
        }

        const int rank_up = lora_up.shape[1];

        assert(lora_up.shape[0] == N);
        // assert(lora_up.shape[1] == Lora::LORA_RANK);
        assert(lora_act_in.shape[0] == M);
        assert(lora_act_in.shape[1] == rank_up);

        dispatchVal(rank_up, std::integer_sequence<int, 0, 32, 48, 64, 80, 96>(), [&]<int RANK_UP>() {
            using LoraUp = GEMM::Lora<RANK_UP>;
            using scale_t = typename LoraUp::scale_t;

            scale_t scales;
            if constexpr (scales.size() > 0) {
                assert(lora_scales.size() >= scales.size());
                for (size_t i = 0; i < scales.size(); i++) {
                    scales[i] = lora_scales[i];
                }
            }

            if (!lora_down.valid()) {
                using Epilogue = typename GEMM::EpilogueCombination<typename LoraUp::EpilogueLoraUp, MidEpilogue, NextEpilogue, GEMM::EpilogueNop>;
                return launch_bias.template operator()<Epilogue>({
                    typename LoraUp::EpilogueLoraUp::Arguments{
                        .lora_act = lora_act_in.data_ptr<float>(),
                        .lora_wgt_up = lora_up.data_ptr<GEMM::packed_fpsum_t>(),
                        .scales = scales,
                    },
                    midArgs,
                    nextArgs,
                    {}
                });
            }

            const int rank_down = lora_down.shape[1];

            assert(rank_down == rank_up);

            assert(lora_down.shape[0] == N);
            // assert(lora_down.shape[1] == Lora::LORA_RANK);
            assert(lora_act_out.shape[0] == M);
            assert(lora_act_out.shape[1] == rank_down);

            lora_act_out.zero_();

            // dispatchVal(rank_down, std::integer_sequence<int, 16, 32, 48, 64, 80>(), [&]<int RANK_DOWN>() {

            using LoraDown = LoraUp; // GEMM::Lora<RANK_DOWN>;
            using Epilogue = GEMM::EpilogueCombination<typename LoraUp::EpilogueLoraUp, MidEpilogue, typename LoraDown::EpilogueLoraDown, NextEpilogue, GEMM::EpilogueNop>;
            return launch_bias.template operator()<Epilogue>({
                typename LoraUp::EpilogueLoraUp::Arguments{
                    .lora_act = lora_act_in.data_ptr<float>(),
                    .lora_wgt_up = lora_up.data_ptr<GEMM::packed_fpsum_t>(),
                    .scales = scales,
                },
                midArgs,
                typename LoraDown::EpilogueLoraDown::Arguments{
                    .lora_wgt_down = lora_down.data_ptr<GEMM::packed_fpsum_t>(),
                    .lora_act = lora_act_out.data_ptr<float>(),
                },
                nextArgs,
                {}
            });

            // });
        });
    };

    if (qout.valid() && oscales.valid()) {

        // dispatchBool(qout_unsigned, [&]<bool USE_UNSIGNED>() {

        static constexpr float SHIFT_GELU = 0.171875f;


        constexpr bool USE_UNSIGNED = true;
        using EpilogueQuantize = typename GEMM::EpilogueQuantize<false, USE_UNSIGNED>;
        auto argsQuantize = typename EpilogueQuantize::Arguments{
            .qout = qout.data_ptr<GEMM::packed_act_t>(),
            .oscales = oscales.data_ptr<GEMM::packed_ascale_t>(),
            .shift_value = SHIFT_GELU,
            .smooth_factor = smooth_factor.data_ptr<GEMM::packed_wscale_t>()
        };

        // TODO: check if gelu is needed
        if (out.valid()) {
            launch_lora.template operator()<GEMM::EpilogueCombination<GEMM::EpilogueDefault, EpilogueQuantize>, GEMM::EpilogueGelu>({
                GEMM::EpilogueDefault::Arguments{},
                argsQuantize
            }, {});
        } else {
            launch_lora.template operator()<EpilogueQuantize, GEMM::EpilogueGelu>(argsQuantize, {});
        }

        // });
        
    } else if (rotary_emb.valid()) {
        assert(norm_q.valid());
        assert(norm_k.valid());
        // assert(isTypeMatch<GEMM::half_t>(rotary_emb.scalar_type()));
        assert(rotary_emb.scalar_type() == Tensor::FP32);
        assert(rotary_emb.numel() == M * GEMM::EpilogueQKVProj::HEAD_DIM / 2 * GEMM::EpilogueQKVProj::ROTARY_EMB_NUM_ELEMENTS);
        launch_lora.template operator()<GEMM::EpilogueQKVProj, GEMM::EpilogueNop>(GEMM::EpilogueQKVProj::Arguments{
            .pool_out = poolout.valid() ? poolout.data_ptr<GEMM::half_t>() : nullptr,
            .rotary_emb = rotary_emb.data_ptr<float>(),
            .rmsnorm_weight_q = norm_q.data_ptr<GEMM::half_t>(),
            .rmsnorm_weight_k = norm_k.data_ptr<GEMM::half_t>(),
            .epsilon = 1e-6,
        }, {});
    } else if (out.valid()) {
        launch_lora.template operator()<GEMM::EpilogueDefault, GEMM::EpilogueNop>({}, {});
    } else {
        assert(false);
    }
}

void quantize_w4a4_act_fuse_lora(Tensor input, Tensor output, Tensor oscales, Tensor lora_down, Tensor lora_act_out, Tensor smooth) {
    using GEMM = GEMM_W4A4;
    // using Lora = GEMM::Lora;

    int M = input.numel() / input.shape[-1];
    int N = input.shape[-1];

    assert(output.dtype() == Tensor::INT8);
    assert(output.numel() / output.shape[-1] == M);
    assert(output.shape[-1] == N / 2);

    // assert(oscales.dtype() == Tensor::FP16);
    assert(isTypeMatch<GEMM::half_t>(oscales.dtype()));
    assert(oscales.numel() == M * N / GEMM::WARP_K);

    const int rank = lora_down.shape[1];

    assert(lora_down.shape[0] == N);
    // assert(lora_down.shape[1] == Lora::LORA_RANK);
    assert(lora_act_out.shape[0] == M);
    assert(lora_act_out.shape[1] == rank);

    lora_act_out.zero_();

    dim3 grid(M / GEMM::BLOCK_M, N / GEMM::BLOCK_N);

    dispatchVal(rank, std::integer_sequence<int, 0, 32, 48, 64, 80, 96>(), [&]<int RANK>() {
        using Lora = typename GEMM::Lora<RANK>;
        using kernel = typename Lora::quantize_w4a4_fuse_lora_kernel;

        auto func = invoke_kernel<kernel, typename kernel::Arguments>;

        checkCUDA(cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, kernel::SHMEM_SIZE));

        // log(std::format("quantize_w4a4_act_fuse_lora M={} N={} input={} output={} (size={} numel={})", M, N, input.data_ptr(), output.data_ptr(), output.buffer->getSize(), output.numel()));

        func<<<grid, GEMM::WARP_SIZE * GEMM::NUM_WARPS, kernel::SHMEM_SIZE>>>(
            typename kernel::Arguments{
                .input = input.data_ptr<GEMM::half_t>(),
                .smooth_factor = smooth.valid() ? smooth.data_ptr<GEMM::packed_wscale_t>() : nullptr,
                .output = output.data_ptr<GEMM::packed_act_t>(),
                .oscales = oscales.data_ptr<GEMM::packed_ascale_t>(),
                .lora_wgt_down = lora_down.data_ptr<GEMM::packed_fpsum_t>(),
                .lora_act = lora_act_out.data_ptr<float>(),
                .M = M,
                .N = N,
            }
        );
        checkCUDA(cudaGetLastError());
    });
}

void quantize_w4a4_act(Tensor input, Tensor output, Tensor oscales) {
    using GEMM = GEMM_W4A4;

    int M = input.numel() / input.shape[-1];
    int K = input.shape[-1];

    assert(output.dtype() == Tensor::INT8);
    assert(output.numel() / output.shape[-1] == M);
    assert(output.shape[-1] == K / 2);

    // assert(oscales.dtype() == Tensor::FP16);
    assert(isTypeMatch<GEMM::half_t>(oscales.dtype()));
    assert(oscales.numel() == M * K / GEMM::WARP_K);

    dim3 grid(M / GEMM::WARP_M, K / GEMM::WARP_K);
    invoke_kernel<GEMM::quantize_w4a4_act_kernel><<<grid, GEMM::WARP_SIZE>>>(
        input.data_ptr<GEMM::half_t>(),
        output.data_ptr<GEMM::packed_act_t>(),
        oscales.data_ptr<GEMM::packed_ascale_t>(),
        K
    );
    checkCUDA(cudaGetLastError());
}

void quantize_w4a4_wgt(Tensor input, Tensor output, Tensor oscales) {
    using GEMM = GEMM_W4A4;

    int N = input.numel() / input.shape[-1];
    int K = input.shape[-1];

    assert(output.dtype() == Tensor::INT8);
    assert(output.ndims() == 2);
    assert(output.shape[0] == N);
    assert(output.shape[1] == K / 2);
    
    assert(isTypeMatch<GEMM::half_t>(oscales.dtype()));
    // assert(oscales.dtype() == Tensor::FP16);
    assert(oscales.numel() == N * K / GEMM::WARP_K);

    dim3 grid(N / GEMM::WARP_N, K / GEMM::WARP_K);
    invoke_kernel<GEMM::quantize_w4a4_wgt_kernel><<<grid, GEMM::WARP_SIZE>>>(
        input.data_ptr<GEMM::half_t>(),
        output.data_ptr<GEMM::packed_wgt_t>(),
        oscales.data_ptr<GEMM::packed_wscale_t>(),
        K
    );
    checkCUDA(cudaGetLastError());
}

void quantize_w8a8_act(Tensor input, Tensor output, Tensor oscales, bool fuse_glu) {
    int M = input.numel() / input.shape[-1];
    int K = input.shape[-1];

    assert(output.dtype() == Tensor::INT8);
    assert(output.numel() / output.shape[-1] == M);
    assert(output.shape[-1] == fuse_glu ? K / 2 : K);

    assert(oscales.dtype() == Tensor::FP16);
    assert(oscales.numel() == M * 1);

    auto launch = [&]<bool FUSE_GLU>() {
        using GEMM = GEMM_W8A8;
        using kernel = GEMM::quantize_w8a8_act_kernel<FUSE_GLU>;

        assert(kernel::check(M, K));
        dim3 grid = kernel::gridSize(M, K);
        dim3 block = kernel::blockSize(M, K);

        auto func = invoke_kernel<kernel, const GEMM::half_t *, GEMM::packed_act_t *, GEMM::packed_ascale_t *, int, bool>;

        checkCUDA(cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, 92160));

        func<<<grid, block, kernel::smemSize(M, K)>>>(
            input.data_ptr<GEMM::half_t>(),
            output.data_ptr<GEMM::packed_act_t>(),
            oscales.data_ptr<GEMM::packed_ascale_t>(),
            K,
            false
        );
        checkCUDA(cudaGetLastError());
    };

    if (fuse_glu) {
        launch.template operator()<true>();
    } else {
        launch.template operator()<false>();
    }
}

void gemm_w8a8(Tensor act,      // [M, K]
               Tensor wgt,      // [N, K]
               Tensor out,      // [M, N]
               Tensor ascales,  // [1, M]
               Tensor wscales   // [1, N]
               )  
{
    using GEMM = GEMM_W8A8;
    using Epilogue = GEMM::EpilogueSilu;

    int M = act.numel() / act.shape[-1];
    int N = wgt.shape[0];
    int K = act.shape[-1];
    assert(K == wgt.shape[1]);

    Epilogue::Arguments epilogueArgs;


    dim3 grid(M / GEMM::BLOCK_M, N / GEMM::BLOCK_N);

    bool swapBlockMN = M > N * 2;
    if (swapBlockMN) {
        std::swap(grid.x, grid.y);
    }

    invoke_kernel<GEMM::gemm_w8a8_kernel<Epilogue>><<<grid, GEMM::WARP_SIZE * GEMM::NUM_WARPS>>>(
        act.data_ptr<GEMM::packed_act_t>(),
        wgt.data_ptr<GEMM::packed_wgt_t>(),
        ascales.data_ptr<GEMM::packed_ascale_t>(),
        wscales.data_ptr<GEMM::packed_wscale_t>(),
        out.data_ptr<GEMM::half_t>(),
        M, N, K, epilogueArgs,
        swapBlockMN,
        false
    );
    checkCUDA(cudaGetLastError());
}

void gemm_w8a8_fuse_litela(
    Tensor act,      // [B, (M), K]
    Tensor wgt,      // [N, K]
    Tensor out_q,    // [B, (M), N / 3]
    Tensor out_vk,   // [B, num_heads, head_dim + 1, head_dim]
    Tensor ascales,  // [1, M]
    Tensor wscales   // [1, N]
) {
    using GEMM = GEMM_W8A8;
    using Epilogue = GEMM::EpilogueLiteLA;

    int M = act.numel() / act.shape[-1];
    int N = wgt.shape[0];
    int K = act.shape[-1];
    assert(K == wgt.shape[1]);

    assert(out_vk.ndims() == 4);
    assert(out_vk.shape[2] == Epilogue::LITELA_HEAD_DIM + 1);
    assert(out_vk.shape[3] == Epilogue::LITELA_HEAD_DIM);
    assert(out_vk.shape[1] * Epilogue::LITELA_HEAD_DIM * 3 == N);

    int batch_size = out_vk.shape[0];
    int num_heads = out_vk.shape[1];

    assert(M % batch_size == 0);
    int batch_m = M / batch_size;

    Epilogue::Arguments epilogueArgs;
    epilogueArgs.batch_m = act.shape[1];
    epilogueArgs.out_q = out_q.data_ptr<GEMM::half_t>();
    epilogueArgs.out_vk = out_vk.data_ptr<float>();

    checkCUDA(cudaMemsetAsync(out_vk.data_ptr(), 0, out_vk.buffer->getSize()));

    auto func = invoke_kernel<GEMM::gemm_w8a8_kernel<Epilogue>, 
        const GEMM::packed_act_t *, 
        const GEMM::packed_wgt_t *, 
        const GEMM::packed_ascale_t *,
        const GEMM::packed_wscale_t *,
        GEMM::half_t *,
        int, int, int,
        Epilogue::Arguments,
        bool,
        bool>;

    checkCUDA(cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, Epilogue::SHMEM_SIZE));

    dim3 grid(M / GEMM::BLOCK_M, N / GEMM::BLOCK_N);

    bool swapBlockMN = M > N * 2;
    if (swapBlockMN) {
        std::swap(grid.x, grid.y);
    }

    func<<<grid, GEMM::WARP_SIZE * GEMM::NUM_WARPS, Epilogue::SHMEM_SIZE>>>(
        act.data_ptr<GEMM::packed_act_t>(),
        wgt.data_ptr<GEMM::packed_wgt_t>(),
        ascales.data_ptr<GEMM::packed_ascale_t>(),
        wscales.data_ptr<GEMM::packed_wscale_t>(),
        nullptr,
        M, N, K, epilogueArgs, 
        swapBlockMN,
        false
    );
    checkCUDA(cudaGetLastError());

    invoke_kernel<Epilogue::vk_mul_q_kernel><<<dim3(batch_m / 128, num_heads, batch_size), 128>>>(
        out_q.data_ptr<GEMM::half_t>(),
        out_vk.data_ptr<float>(),
        1e-6f
    );
    checkCUDA(cudaGetLastError());
}