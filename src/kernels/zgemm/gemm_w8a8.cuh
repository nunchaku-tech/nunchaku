#pragma once

#include "gemm_base.cuh"

namespace nunchaku::kernels {

class GEMM_W8A8 : public GEMMBase<GEMMConfig_W8A8> {
public:
    using psum_warp = std::array<packed_psum_t, WARP_M_TILES * WARP_N_TILES>;

    __device__ __forceinline__ static packed_psum_t mma(packed_act_t act, packed_wgt_t wgt, packed_psum_t psum) {
        // packed_psum_t psum;
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                     "{%0,  %1,  %2,  %3},"
                     "{%4,  %5,  %6,  %7},"
                     "{%8,  %9},"
                     "{%10,  %11,  %12,  %13};\n"
                     : "=r"(psum.data[0]), "=r"(psum.data[1]), "=r"(psum.data[2]), "=r"(psum.data[3])
                     : "r"(act.x),
                       "r"(act.y),
                       "r"(act.z),
                       "r"(act.w),
                       "r"(wgt.x),
                       "r"(wgt.y),
                       // "r"(0), "r"(0), "r"(0), "r"(0)
                       "r"(psum.data[0]),
                       "r"(psum.data[1]),
                       "r"(psum.data[2]),
                       "r"(psum.data[3]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                     "{%0,  %1,  %2,  %3},"
                     "{%4,  %5,  %6,  %7},"
                     "{%8,  %9},"
                     "{%10,  %11,  %12,  %13};\n"
                     : "=r"(psum.data[4]), "=r"(psum.data[5]), "=r"(psum.data[6]), "=r"(psum.data[7])
                     : "r"(act.x),
                       "r"(act.y),
                       "r"(act.z),
                       "r"(act.w),
                       "r"(wgt.z),
                       "r"(wgt.w),
                       // "r"(0), "r"(0), "r"(0), "r"(0)
                       "r"(psum.data[4]),
                       "r"(psum.data[5]),
                       "r"(psum.data[6]),
                       "r"(psum.data[7]));
        return psum;
    }

    __device__ __forceinline__ static void compute(act_warp A, wgt_warp W, psum_warp &psum) {
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
     * oscales is per-warp (in shared memory)
     * output is per-thread (in regs)
     * shmem must be at least INSN_M * (INSN_K * sizeof(element) + 16) (16 * 32 = 512 Bytes)
     * default to quantize activation, if quantize weight, input should be column-majored and output should be
     * transposed ({x, y, z, w} = {x, z, y, w})
     */
    template<bool input_shmem = false>
    __device__ __forceinline__ static void
    quantize_w8a8_warp(const half_t *input, const half_t *oscales, int stride, packed_act_t &output, void *shmem) {
        const int laneId = threadIdx.x % WARP_SIZE;

        constexpr int QUANTIZE_BITWIDTH = 8;
        // constexpr int QUANTIZE_BITMASK = 0xff;
        // constexpr int QVALUE_MAX = 128;   // 4 bit => [-128, 127]

        // 1 lane = 1 pack
        // 1 warp = 32 lanes = 32 packs = 1 packwarp
        // a pack is {a0, ..., a7} in figure
        // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=ex2#mma-16864-a PACK_SIZE * 4 =
        // INSN_K / 2
        constexpr int PACK_SIZE             = INSN_K / 8; // = 4 for 8bit
        constexpr int NUM_PACKS_PER_ROW     = INSN_K / PACK_SIZE;
        constexpr int NUM_ROWS_PER_PACKWARP = PACK_SIZE * WARP_SIZE / INSN_K;
        constexpr int NUM_PACKWARPS         = INSN_M / NUM_ROWS_PER_PACKWARP;
        using packed_input                  = std::array<half_t, PACK_SIZE>;

        packed_input packs[NUM_PACKWARPS];

        // load
#pragma unroll
        for (int i = 0; i < NUM_PACKWARPS; i++) {
            int rowId = i * NUM_ROWS_PER_PACKWARP + laneId / NUM_PACKS_PER_ROW;
            int colId = laneId % NUM_PACKS_PER_ROW * PACK_SIZE;
            packs[i]  = load<input_shmem>(reinterpret_cast<const packed_input *>(input + rowId * stride + colId));
        }

        // quantize
        using matrix_t = uint32_t[INSN_M][NUM_PACKS_PER_ROW];
        matrix_t &mat  = *reinterpret_cast<matrix_t *>(shmem);
#pragma unroll
        for (int i = 0; i < NUM_PACKWARPS; i++) {
            const int row = i * NUM_ROWS_PER_PACKWARP + laneId / NUM_PACKS_PER_ROW;
            const int col = laneId % NUM_PACKS_PER_ROW;

            float rscale = cuda_frcp(float(oscales[row]));

            uint32_t qpack = 0;
#pragma unroll
            for (int j = 0; j < PACK_SIZE; j += 2) {
                // half2_t hval = __hmul2(half2_t(rscale, rscale), half2_t(packs[i][j], packs[i][j + 1]));
                float2 fval = half22float2(half2_t(packs[i][j], packs[i][j + 1])) * float2(rscale, rscale);
                qpack |= quantize_float2<QUANTIZE_BITWIDTH, false>(fval) << (j * QUANTIZE_BITWIDTH);
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
    __device__ __forceinline__ static half_t
    findmax_warp(const half_t *input, half_t *output_shmem, int K, bool alwaysfalse) {
        const int laneId = threadIdx.x % WARP_SIZE;

        using packed_input       = std::array<half2_t, 4>;
        using packed_gated_input = std::array<half_t, 4>;

        constexpr int PACK_SIZE  = sizeof(packed_input) / sizeof(half_t);
        constexpr int NUM_STAGES = 2;

        half2_t maxvalue2 = {0, 0};
        packed_input pack[NUM_STAGES];

#pragma unroll
        for (int k = 0; k < NUM_STAGES - 1; k++) {
            const int idx = k * PACK_SIZE * WARP_SIZE + laneId * PACK_SIZE;
            if (idx < K) {
                pack[k] = load(reinterpret_cast<const packed_input *>(&input[idx]));
            } else {
                pack[k].fill(half2_t(0, 0));
            }
        }

        // int dummy = 0;

        // FIXME: pipeline does not work
        // TODO: store quantized data to shmem (instead of half)

        for (int k1 = 0; k1 < ceilDiv(K, PACK_SIZE * WARP_SIZE); k1 += NUM_STAGES) {
#pragma unroll
            for (int k2 = 0; k2 < NUM_STAGES; k2++) {

                const int nextidx = (k1 + k2 + NUM_STAGES - 1) * PACK_SIZE * WARP_SIZE + laneId * PACK_SIZE;
                const int nextk2  = (k2 + NUM_STAGES - 1) % NUM_STAGES;

                if (nextidx < K) {
                    pack[nextk2] = load(reinterpret_cast<const packed_input *>(&input[nextidx]));
                } else {
                    pack[nextk2].fill(half2_t(0, 0));
                }

                packed_input &p = pack[k2];

                if constexpr (fuse_glu) {
                    packed_gated_input gated;

#pragma unroll
                    for (int j = 0; j < p.size(); j++) {
                        gated[j] = p[j].x * gelu_half(p[j].y);
                        p[j].x   = gated[j];
                        p[j].y   = 0;
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
        static constexpr bool check(int M, int K) {
            const int K2 = fuse_glu ? K / 2 : K;
            return M % WARP_M == 0 && K2 % WARP_K == 0;
        }
        static constexpr dim3 gridSize(int M, int K) {
            return dim3(M / WARP_M);
        }
        static constexpr dim3 blockSize(int M, int K) {
            return dim3(NUM_WARPS * 32);
        }
        static constexpr size_t smemSize(int M, int K) {
            if constexpr (!fuse_glu) {
                return 0;
            }
            const int K2 = fuse_glu ? K / 2 : K;
            return INSN_M * K2 * sizeof(half_t);
        }

        __device__ void
        operator()(const half_t *input, packed_act_t *output, packed_ascale_t *oscales, int K, bool alwaysfalse) {
            // for quantize kernel
            const int laneId = threadIdx.x % WARP_SIZE;
            const int warpId = threadIdx.x / WARP_SIZE;

            const int numWarps = blockDim.x / WARP_SIZE;

            // for GEMM kernel
            const int bm         = blockIdx.x / (BLOCK_M / WARP_M);
            const int gemmWarpId = blockIdx.x % (BLOCK_M / WARP_M);

            __shared__ alignas(128) half_t oscale_shmem[WARP_M];
            // __shared__ alignas(128) half_t maxv_shmem[WARP_M];
            __shared__ alignas(128) uint8_t tmp_shmem[NUM_WARPS][512];

            const int K2 = fuse_glu ? K / 2 : K;

            // INSN_M * K2
            extern __shared__ uint8_t smem[];
            half_t *shmem = reinterpret_cast<half_t *>(smem);

            for (int tileM = 0; tileM < WARP_M_TILES; tileM++) {

                for (int i = warpId; i < INSN_M; i += numWarps) {
                    const int rowLocal  = tileM * INSN_M + i;
                    const int rowGlobal = blockIdx.x * WARP_M + rowLocal;

                    half_t maxv = findmax_warp<fuse_glu>(input + rowGlobal * K, shmem + i * K2, K, alwaysfalse);
                    oscale_shmem[rowLocal] = maxv / half_t(127);
                    // rscale_shmem[rowLocal] = half_t(127) / maxv;
                    // maxv_shmem[rowLocal] = maxv;
                }
                __syncthreads();

                for (int bk = warpId; bk < K2 / WARP_K; bk += numWarps) {
                    const int rowLocal  = tileM * INSN_M;
                    const int rowGlobal = blockIdx.x * WARP_M + rowLocal;
                    const int col       = bk * WARP_K;

                    packed_act_t tmpout;

                    if constexpr (fuse_glu) {
                        quantize_w8a8_warp<true>(shmem + col, oscale_shmem + rowLocal, K2, tmpout, &tmp_shmem[warpId]);
                    } else {
                        quantize_w8a8_warp<false>(
                            input + rowGlobal * K + col, oscale_shmem + rowLocal, K, tmpout, &tmp_shmem[warpId]);
                    }

                    store(&output[(((bm * K2 / WARP_K + bk) * NUM_WARPS + gemmWarpId) * WARP_M_TILES + tileM) *
                                      WARP_SIZE +
                                  laneId],
                          tmpout);
                }
                __syncthreads();
            }

            // [M / BLOCK_M, 1, NUM_WARPS, ASCALES_NUM_PACKS, ASCALES_VALID_LANES] of packed_ascale_t
            pack_ascales(oscale_shmem,
                         &oscales[(bm * NUM_WARPS + gemmWarpId) * ASCALES_NUM_PACKS * ASCALES_VALID_LANES]);
        }
    };

    __device__ __forceinline__ static gated_fpsum_warp apply_glu(fpsum_warp fpsum) {
        gated_fpsum_warp result;
        for (int i = 0; i < WARP_M_TILES; i++) {
            for (int j = 0; j < WARP_N_TILES; j++) {
                for (int k = 0; k < 4; k++) {
                    half_t &dst = result[i * WARP_N_TILES + j].data[k];
                    half2_t src = fpsum[i * WARP_N_TILES + j].data[k];
                    dst         = src.x * gelu_half(src.y);
                }
            }
        }
        return result;
    }

    static constexpr int unpack_gated_fpsum_shmem_size = INSN_M * (WARP_N / 2 + 8) * sizeof(half_t);
    __device__ __forceinline__ static void
    unpack_gated_fpsum(gated_fpsum_warp fpsum, half_t *output, int stride, void *shmem) {
        const int laneId = threadIdx.x % WARP_SIZE;

        constexpr int PACK_SIZE = WARP_N / 2 / WARP_SIZE;
        using pack_t            = std::array<half_t, PACK_SIZE>;

        // +8 to prevent bank conflicts
        using matrix_t = half_t[INSN_M][WARP_N / 2 + 8];
        matrix_t &mat  = *reinterpret_cast<matrix_t *>(shmem);

        for (int i = 0; i < WARP_M_TILES; i++) {
            for (int j = 0; j < WARP_N_TILES; j++) {
                packed_gated_fpsum_t &fsum                          = fpsum[i * WARP_N_TILES + j];
                int row                                             = laneId / 4;
                int col                                             = laneId % 4 + j * INSN_N / 2;
                *reinterpret_cast<half_t *>(&mat[row][col + 0])     = fsum.data[0];
                *reinterpret_cast<half_t *>(&mat[row][col + 4])     = fsum.data[2];
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
    __device__ __forceinline__ static void gemm_w8a8_block(const BlockInfo binfo,
                                                           const packed_act_t *act,
                                                           const packed_wgt_t *wgt,
                                                           const packed_ascale_t *ascales,
                                                           const packed_wscale_t *wscales,
                                                           // half_t *out,
                                                           int M,
                                                           int N,
                                                           int K,
                                                           Epilogue::Arguments epilogeParams,
                                                           bool alwaysfalse) {
        constexpr int NUM_STAGES = 2;

        const int laneId = threadIdx.x % WARP_SIZE;
        const int warpId = threadIdx.x / WARP_SIZE;

        act_warp A[NUM_STAGES]; // 8
        wgt_warp W[NUM_STAGES]; // 32
        ascale_warp ascale;     // 1
        wscale_warp wscale;     // 2
        psum_warp psum;         // 128

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
                int idx   = (k2 + NUM_STAGES - 1) % NUM_STAGES;
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

        f32psum_warp f32psum;

#pragma unroll
        for (int i = 0; i < f32psum.size(); i++) {
#pragma unroll
            for (int j = 0; j < 8; j++) {
                f32psum[i].data[j] = 0;
            }
        }

        apply_scales([&](int i, int j) { return psum[i * WARP_N_TILES + j]; }, ascale, wscale, f32psum);

        fpsum_warp fpsum = packed_fp32_to_fp16(f32psum);

        // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x % 32 == 0) {

        //     printf("warpId = %d fpsum = %f\n", warpId, (float)fpsum[0].data[0].x);
        // }

        Epilogue()(binfo, fpsum, M, N, K, epilogeParams);
    }

    // out : [M / BLOCK_M, BLOCK_M, N / BLOCK_N, BLOCK_N]
    template<typename Epilogue>
    struct gemm_w8a8_kernel {
        static constexpr int MIN_ARCH = std::is_same_v<half_t, __nv_bfloat16> ? 800 : 750;
        __device__ void operator()(const packed_act_t *act,
                                   const packed_wgt_t *wgt,
                                   const packed_ascale_t *ascales,
                                   const packed_wscale_t *wscales,
                                   // half_t *out,
                                   int M,
                                   int N,
                                   int K,
                                   Epilogue::Arguments epilogueArgs,
                                   bool swapBlockXY,
                                   bool alwaysfalse) {
            BlockInfo binfo = {
                .bm         = (int)blockIdx.x,
                .bn         = (int)blockIdx.y,
                .numBlocksM = (int)gridDim.x,
                .numBlocksN = (int)gridDim.y,
            };

            if (swapBlockXY) {
                std::swap(binfo.bm, binfo.bn);
                std::swap(binfo.numBlocksM, binfo.numBlocksN);
            }

            const int bm = binfo.bm;
            const int bn = binfo.bn;

            gemm_w8a8_block<Epilogue>(binfo,
                                      act + bm * (K / WARP_K) * NUM_WARPS * WARP_M_TILES * WARP_SIZE,
                                      wgt + bn * (K / WARP_K) * WARP_N_TILES * WARP_SIZE,
                                      ascales + bm * (1) * NUM_WARPS * ASCALES_NUM_PACKS *
                                                    ASCALES_VALID_LANES, // only 1 group in W8A8
                                      wscales + bn * (1) * WSCALES_NUM_PACKS * WSCALES_VALID_LANES,
                                      // #if 1
                                      //                 out + (bm * BLOCK_M * N) + bn * BLOCK_N,
                                      // #else
                                      //                 out + (bm * BLOCK_M * N / 2) + bn * BLOCK_N / 2,
                                      // #endif
                                      M,
                                      N,
                                      K,
                                      epilogueArgs,
                                      alwaysfalse);
        }
    };

#if 0
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
#endif
};

}; // namespace nunchaku::kernels
