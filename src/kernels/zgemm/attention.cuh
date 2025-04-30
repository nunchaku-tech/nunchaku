#pragma once

#include "gemm_base.cuh"

namespace nunchaku::kernels {

// M: Q tokens
// N: V HEAD_DIM
// K: K tokens
// D: QK HEAD_DIM
template<bool bf16out>
struct AttentionFP16Config {
    static constexpr int HEAD_DIM = 128;

    static constexpr int BLOCK_M   = 128;
    static constexpr int WARP_SIZE = 32;
    static constexpr int NUM_WARPS = 8;

    static constexpr int WARP_K = 32;

    static constexpr int INSN_M    = 16;
    static constexpr int INSN_N    = 16;
    static constexpr int INSN_K_QK = 16;
    static constexpr int INSN_K_PV = 16;

    using half_t  = half;
    using half2_t = half2;

    using epilogue_half_t  = typename std::conditional_t<bf16out, __nv_bfloat16, half>;
    using epilogue_half2_t = typename std::conditional_t<bf16out, __nv_bfloat162, half2>;
};

using AttentionFP16Config_FP16 = AttentionFP16Config<false>;
using AttentionFP16Config_BF16 = AttentionFP16Config<true>;

template<typename AttentionConfig>
class Attention;

#ifndef __INTELLISENSE__
template<typename AttentionConfig>
class Attention : public AttentionConfig {
#else
template<>
class Attention<AttentionFP16Config_BF16> : public AttentionFP16Config_BF16 {
    using AttentionConfig = AttentionFP16Config_BF16;
#endif

public:
    using AttentionConfig::HEAD_DIM;
    using AttentionConfig::BLOCK_M;
    using AttentionConfig::WARP_SIZE;
    using AttentionConfig::NUM_WARPS;
    using AttentionConfig::WARP_K;
    using AttentionConfig::INSN_M;
    using AttentionConfig::INSN_N;
    using AttentionConfig::INSN_K_QK;
    using AttentionConfig::INSN_K_PV;
    using typename AttentionConfig::half_t;
    using typename AttentionConfig::half2_t;
    using typename AttentionConfig::epilogue_half_t;
    using typename AttentionConfig::epilogue_half2_t;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    static constexpr bool IS_SM80 = true;
#else
    static constexpr bool IS_SM80 = false;
#endif

    struct GEMMConfig {
        static constexpr int BLOCK_M   = AttentionConfig::BLOCK_M;
        static constexpr int BLOCK_N   = AttentionConfig::HEAD_DIM;
        static constexpr int WARP_SIZE = AttentionConfig::WARP_SIZE;
        static constexpr int NUM_WARPS = AttentionConfig::NUM_WARPS;

        static constexpr int INSN_M = AttentionConfig::INSN_M;
        static constexpr int INSN_N = AttentionConfig::INSN_N;
        static constexpr int INSN_K = AttentionConfig::INSN_K_PV;

        using half_t  = typename AttentionConfig::epilogue_half_t;
        using half2_t = typename AttentionConfig::epilogue_half2_t;
    };

    using GEMM = typename nunchaku::kernels::GEMMBase<GEMMConfig>;

    static constexpr int WARP_M = BLOCK_M / NUM_WARPS;
    static constexpr int WARP_N = HEAD_DIM;
    static constexpr int WARP_D = HEAD_DIM;

    static constexpr int WARP_M_TILES = WARP_M / INSN_M;
    static constexpr int WARP_N_TILES = WARP_N / INSN_N;
    static constexpr int WARP_K_TILES_QK =
        WARP_K / INSN_N; // when multiplying Q*K, K is on dimension of N in MMA instruction
    static constexpr int WARP_K_TILES_PV = WARP_K / INSN_K_PV;
    static constexpr int WARP_D_TILES    = WARP_D / INSN_K_QK;

    using packed_q_t = uint4;
    using packed_k_t = uint4;
    using packed_v_t = uint4;
    using q_warp     = std::array<packed_q_t, WARP_M_TILES * WARP_D_TILES>;
    using k_warp     = std::array<packed_k_t, WARP_K_TILES_QK * WARP_D_TILES>;
    using v_warp     = std::array<packed_v_t, WARP_K_TILES_PV * WARP_N_TILES>;

    using packed_p_t = uint4;
    using p_warp     = std::array<packed_v_t, WARP_M_TILES * WARP_K_TILES_PV>;

    using packed_fpsum_t   = uint4;
    using packed_f32psum_t = typename GEMM::packed_f32psum_t;

    using qk_warp = std::array<packed_f32psum_t, WARP_M_TILES * WARP_K_TILES_QK>;
    // using o_warp = std::array<packed_f32psum_t, WARP_M_TILES * WARP_N_TILES>;
    using o_warp = typename GEMM::f32psum_warp;

    using rowval_warp = std::array<float2, WARP_M_TILES>;

    struct BlockInfo {
        int bm;    // M: Q tokens, bm: block id of M
        int head;  // H: head
        int batch; // B: batch
        int numBlocksM;
        int numHeads;
        int numBatch;
    };

    __device__ __forceinline__ static packed_fpsum_t packed_fp32_to_fp16(packed_f32psum_t input) {
        std::array<half2_t, 4> results;
        for (int i = 0; i < 4; i++) {
            results[i] = float22half2<half2_t>(float2(input.data[i * 2], input.data[i * 2 + 1]));
        }
        return kernels::bit_cast<packed_fpsum_t>(results);
    }

    __device__ __forceinline__ static packed_f32psum_t packed_fp16_to_fp32(packed_fpsum_t input) {
        auto arr = kernels::bit_cast<std::array<half2_t, 4>>(input);
        packed_f32psum_t results;
        for (int i = 0; i < 4; i++) {
            float2 tmp              = half22float2(arr[i]);
            results.data[i * 2]     = tmp.x;
            results.data[i * 2 + 1] = tmp.y;
        }
        return results;
    }

    // q: [batch, head, bm, NUM_WARPS, WARP_M_TILES, WARP_D_TILES, WARP_SIZE] of packed_q_t
    __device__ __forceinline__ static void load_q(const packed_q_t *ptr, q_warp &out, bool pred) {
        const int laneId = threadIdx.x % WARP_SIZE;
        const int warpId = threadIdx.x / WARP_SIZE;

        const packed_q_t *base = &ptr[((warpId * WARP_M_TILES + 0) * WARP_D_TILES + 0) * WARP_SIZE + laneId];

        unrolled_loop<WARP_M_TILES>([&]<int m>() {
            unrolled_loop<WARP_D_TILES>([&]<int d>() {
                out[m * WARP_D_TILES + d] = load_pred(&base[(m * WARP_D_TILES + d) * WARP_SIZE], pred);
            });
        });
    }

    // k: [batch, head, ktile, WARP_K_TILES_QK, WARP_D_TILES, WARP_SIZE] of packed_k_t
    __device__ __forceinline__ static void load_k(const packed_k_t *ptr, int ktile, k_warp &out, bool pred) {
        const int laneId = threadIdx.x % WARP_SIZE;
        const int warpId = threadIdx.x / WARP_SIZE;

        const packed_k_t *base = &ptr[((ktile * WARP_K_TILES_QK + 0) * WARP_D_TILES + 0) * WARP_SIZE + laneId];

        unrolled_loop<WARP_K_TILES_QK>([&]<int k>() {
            unrolled_loop<WARP_D_TILES>([&]<int d>() {
                out[k * WARP_D_TILES + d] = load_pred(&base[(k * WARP_D_TILES + d) * WARP_SIZE], pred);
            });
        });
    }

    // v: [batch, head, ktile, WARP_K_TILES_PV, WARP_N_TILES, WARP_SIZE] of packed_v_t
    __device__ __forceinline__ static void load_v(const packed_v_t *ptr, int ktile, v_warp &out, bool pred) {
        const int laneId = threadIdx.x % WARP_SIZE;
        const int warpId = threadIdx.x / WARP_SIZE;

        const packed_v_t *base = &ptr[((ktile * WARP_K_TILES_PV + 0) * WARP_N_TILES + 0) * WARP_SIZE + laneId];

        unrolled_loop<WARP_K_TILES_PV>([&]<int k>() {
            unrolled_loop<WARP_N_TILES>([&]<int n>() {
                out[n * WARP_K_TILES_PV + k] = load_pred(&base[(k * WARP_N_TILES + n) * WARP_SIZE], pred);
            });
        });
    }

    __device__ __forceinline__ static packed_fpsum_t
    mma_f16xf16_f16(packed_fpsum_t a, packed_fpsum_t b, packed_fpsum_t psum) {
        uint2 out1 = mma_m16n8k16_f16f16f16f16(a, uint2(b.x, b.y), uint2(psum.x, psum.y));
        uint2 out2 = mma_m16n8k16_f16f16f16f16(a, uint2(b.z, b.w), uint2(psum.z, psum.w));
        return packed_fpsum_t{out1.x, out1.y, out2.x, out2.y};
    }

    // set nan values to -inf
    __device__ __forceinline__ static half2_t fix_nan(half2_t input) {
        static constexpr float neginf = -std::numeric_limits<float>::infinity();
        /**
         * In accordance to the IEEE-754R standard,
         * if one of the input parameters to fminf(), fmin(), fmaxf(), or fmax() is NaN,
         * but not the other,
         * the result is the non-NaN parameter.
         */
        return __hmax2(input, half2_t(neginf, neginf));
    }

    __device__ __forceinline__ static float fix_nan(float input) {
        static constexpr float neginf = -std::numeric_limits<float>::infinity();
        return fmaxf(input, neginf);
    }

    __device__ __forceinline__ static packed_fpsum_t fix_nan(packed_fpsum_t input) {
        input.x = kernels::bit_cast<int>(fix_nan(kernels::bit_cast<half2_t>(input.x)));
        input.y = kernels::bit_cast<int>(fix_nan(kernels::bit_cast<half2_t>(input.y)));
        input.z = kernels::bit_cast<int>(fix_nan(kernels::bit_cast<half2_t>(input.z)));
        input.w = kernels::bit_cast<int>(fix_nan(kernels::bit_cast<half2_t>(input.w)));
        return input;
    }

    __device__ __forceinline__ static packed_f32psum_t fix_nan(packed_f32psum_t input) {
#pragma unroll
        for (int i = 0; i < 8; i++) {
            input.data[i] = fix_nan(input.data[i]);
        }
        return input;
    }

    __device__ __forceinline__ static qk_warp compute_qk(q_warp Q, k_warp K) {
        qk_warp QK;
#pragma unroll
        for (int m = 0; m < WARP_M_TILES; m++) {
#pragma unroll
            for (int k = 0; k < WARP_K_TILES_QK; k++) {

#if 0
#pragma unroll
                for (int d = 0; d < WARP_D_TILES; d++) {
                    packed_fpsum_t psum = make_uint4(0, 0, 0, 0);
                    psum = mma_f16xf16_f16(Q[m * WARP_D_TILES + d], K[k * WARP_D_TILES + d], psum);
                    auto f32psum = packed_fp16_to_fp32(psum);
#pragma unroll
                    for (int i = 0; i < 8; i++) {
                        QK[m * WARP_K_TILES_QK + k].data[i] += f32psum.data[i];
                    }
                }

#else
                packed_fpsum_t psum = make_uint4(0, 0, 0, 0);
#pragma unroll
                for (int d = 0; d < WARP_D_TILES; d++) {
                    psum = mma_f16xf16_f16(Q[m * WARP_D_TILES + d], K[k * WARP_D_TILES + d], psum);
                }

                if constexpr (IS_SM80) {
                    psum                        = fix_nan(psum);
                    QK[m * WARP_K_TILES_QK + k] = packed_fp16_to_fp32(psum);
                } else {
                    QK[m * WARP_K_TILES_QK + k] = fix_nan(packed_fp16_to_fp32(psum));
                }
#endif
            }
        }
        return QK;
    }

    __device__ __forceinline__ static rowval_warp compute_rowmax(qk_warp QK, rowval_warp rowmax, float scale) {
#pragma unroll
        for (int m = 0; m < WARP_M_TILES; m++) {
            float2 maxv;
#pragma unroll
            for (int k = 0; k < WARP_K_TILES_QK; k++) {
                packed_f32psum_t &val = QK[m * WARP_K_TILES_QK + k];
                float x               = fmaxf(fmaxf(val.data[0], val.data[1]), fmaxf(val.data[4], val.data[5]));
                float y               = fmaxf(fmaxf(val.data[2], val.data[3]), fmaxf(val.data[6], val.data[7]));
                if (k == 0) {
                    maxv = make_float2(x, y);
                } else {
                    maxv.x = fmaxf(maxv.x, x);
                    maxv.y = fmaxf(maxv.y, y);
                }
            }
#pragma unroll
            for (int mask = 1; mask <= 2; mask *= 2) {
                maxv.x = fmaxf(maxv.x, __shfl_xor_sync(~0, maxv.x, mask));
                maxv.y = fmaxf(maxv.y, __shfl_xor_sync(~0, maxv.y, mask));
            }
            rowmax[m].x = fmaxf(rowmax[m].x, maxv.x * scale);
            rowmax[m].y = fmaxf(rowmax[m].y, maxv.y * scale);
        }
        return rowmax;
    }

    __device__ __forceinline__ static qk_warp softmax(qk_warp QK, rowval_warp rowmax_scaled, float scale) {
#pragma unroll
        for (int m = 0; m < WARP_M_TILES; m++) {
            float2 shift = rowmax_scaled[m];
#pragma unroll
            for (int k = 0; k < WARP_K_TILES_QK; k++) {
                packed_f32psum_t &val = QK[m * WARP_K_TILES_QK + k];
                val.data[0]           = cuda_exp2(fmaf(val.data[0], scale, -shift.x));
                val.data[1]           = cuda_exp2(fmaf(val.data[1], scale, -shift.x));
                val.data[4]           = cuda_exp2(fmaf(val.data[4], scale, -shift.x));
                val.data[5]           = cuda_exp2(fmaf(val.data[5], scale, -shift.x));
                val.data[2]           = cuda_exp2(fmaf(val.data[2], scale, -shift.y));
                val.data[3]           = cuda_exp2(fmaf(val.data[3], scale, -shift.y));
                val.data[6]           = cuda_exp2(fmaf(val.data[6], scale, -shift.y));
                val.data[7]           = cuda_exp2(fmaf(val.data[7], scale, -shift.y));
            }
        }
        return QK;
    }

    __device__ __forceinline__ static rowval_warp compute_rowsum(qk_warp QK) {
        rowval_warp rowsum;
#pragma unroll
        for (int m = 0; m < WARP_M_TILES; m++) {
            float2 sumv = make_float2(0.0f, 0.0f);
#pragma unroll
            for (int k = 0; k < WARP_K_TILES_QK; k++) {
                packed_f32psum_t &val = QK[m * WARP_K_TILES_QK + k];
                sumv.x += val.data[0] + val.data[1] + val.data[4] + val.data[5];
                sumv.y += val.data[2] + val.data[3] + val.data[6] + val.data[7];
            }
#pragma unroll
            for (int mask = 1; mask <= 2; mask *= 2) {
                sumv.x += __shfl_xor_sync(~0, sumv.x, mask);
                sumv.y += __shfl_xor_sync(~0, sumv.y, mask);
            }
            rowsum[m] = sumv;
        }
        return rowsum;
    }

    __device__ __forceinline__ static rowval_warp compute_rescale(rowval_warp rowmax0, rowval_warp rowmax1) {
        rowval_warp rescale;
#pragma unroll
        for (int m = 0; m < WARP_M_TILES; m++) {
            rescale[m].x = cuda_exp2(rowmax0[m].x - rowmax1[m].x);
            rescale[m].y = cuda_exp2(rowmax0[m].y - rowmax1[m].y);
        }
        return rescale;
    }

    __device__ __forceinline__ static o_warp compute_pv(p_warp P, v_warp V, o_warp O, rowval_warp rescale) {
#pragma unroll
        for (int m = 0; m < WARP_M_TILES; m++) {
#pragma unroll
            for (int n = 0; n < WARP_N_TILES; n++) {
                packed_fpsum_t psum = make_uint4(0, 0, 0, 0);
#pragma unroll
                for (int k = 0; k < WARP_K_TILES_PV; k++) {
                    psum = mma_f16xf16_f16(P[m * WARP_K_TILES_PV + k], V[n * WARP_K_TILES_PV + k], psum);
                }

                packed_f32psum_t pv    = packed_fp16_to_fp32(psum);
                packed_f32psum_t &oval = O[m * WARP_N_TILES + n];
                oval.data[0]           = oval.data[0] * rescale[m].x + pv.data[0];
                oval.data[1]           = oval.data[1] * rescale[m].x + pv.data[1];
                oval.data[4]           = oval.data[4] * rescale[m].x + pv.data[4];
                oval.data[5]           = oval.data[5] * rescale[m].x + pv.data[5];
                oval.data[2]           = oval.data[2] * rescale[m].y + pv.data[2];
                oval.data[3]           = oval.data[3] * rescale[m].y + pv.data[3];
                oval.data[6]           = oval.data[6] * rescale[m].y + pv.data[6];
                oval.data[7]           = oval.data[7] * rescale[m].y + pv.data[7];
            }
        }
        return O;
    }

    __device__ __forceinline__ static rowval_warp compute_l(rowval_warp L, rowval_warp rescale, rowval_warp rowsum) {
#pragma unroll
        for (int m = 0; m < WARP_M_TILES; m++) {
            L[m].x = fmaf(L[m].x, rescale[m].x, rowsum[m].x);
            L[m].y = fmaf(L[m].y, rescale[m].y, rowsum[m].y);
        }
        return L;
    }

    __device__ __forceinline__ static p_warp qk_to_p(qk_warp QK) {
        static_assert(WARP_K_TILES_QK == WARP_K_TILES_PV);
        p_warp P;
#pragma unroll
        for (int m = 0; m < WARP_M_TILES; m++) {
#pragma unroll
            for (int k = 0; k < WARP_K_TILES_PV; k++) {
                P[m * WARP_K_TILES_PV + k] = packed_fp32_to_fp16(QK[m * WARP_K_TILES_QK + k]);
            }
        }
        return P;
    }

    // __device__ __forceinline__
    // static void compute(q_warp Q, k_warp K, v_warp V, o_warp &O, rowval_warp &M, rowval_warp &L, float scale) {
    //     qk_warp qk = compute_qk(Q, K);
    //     rowval_warp M1 = compute_rowmax(qk, M, scale);
    //     qk = softmax(qk, M1, scale);
    //     rowval_warp rowsum = compute_rowsum(qk);
    //     p_warp P = qk_to_p(qk);
    //     rowval_warp rescale = compute_rescale(M, M1);
    //     M = M1;
    //     L = compute_l(L, rescale, rowsum);
    //     O = compute_pv(P, V, O, rescale);
    // }

    __device__ __forceinline__ static std::tuple<p_warp, rowval_warp>
    compute(q_warp Q, k_warp K, rowval_warp &M, rowval_warp &L, float scale) {
        qk_warp qk          = compute_qk(Q, K);
        rowval_warp M1      = compute_rowmax(qk, M, scale);
        qk                  = softmax(qk, M1, scale);
        rowval_warp rowsum  = compute_rowsum(qk);
        p_warp P            = qk_to_p(qk);
        rowval_warp rescale = compute_rescale(M, M1);
        M                   = M1;
        L                   = compute_l(L, rescale, rowsum);
        return {P, rescale};
    }

    __device__ __forceinline__ static o_warp compute_o(o_warp O, rowval_warp L) {
#pragma unroll
        for (int m = 0; m < WARP_M_TILES; m++) {
            float2 inv;
            inv.x = cuda_frcp(L[m].x);
            inv.y = cuda_frcp(L[m].y);
#pragma unroll
            for (int n = 0; n < WARP_N_TILES; n++) {
                packed_f32psum_t &oval = O[m * WARP_N_TILES + n];
                oval.data[0]           = oval.data[0] * inv.x;
                oval.data[1]           = oval.data[1] * inv.x;
                oval.data[4]           = oval.data[4] * inv.x;
                oval.data[5]           = oval.data[5] * inv.x;
                oval.data[2]           = oval.data[2] * inv.y;
                oval.data[3]           = oval.data[3] * inv.y;
                oval.data[6]           = oval.data[6] * inv.y;
                oval.data[7]           = oval.data[7] * inv.y;
            }
        }
        return O;
    }

#if 0
    template<typename Epilogue>
    __device__ __forceinline__
    static void attention_fp16_block(
        const BlockInfo binfo,
        const packed_q_t *ptr_q,
        const packed_k_t *ptr_k,
        const packed_v_t *ptr_v,
        float scale,
        int ntokens_q,
        int ntokens_kv,
        Epilogue::Arguments epilogueArgs,
        bool alwaysfalse)
    {
        constexpr int NUM_STAGES = 2;

        const int laneId = threadIdx.x % WARP_SIZE;
        const int warpId = threadIdx.x / WARP_SIZE;

        q_warp Q;   // 32
        k_warp K[NUM_STAGES];  // 32 * 2
        v_warp V[NUM_STAGES];  // 32 * 2
        o_warp O;   // 64
        rowval_warp L;  // 2
        rowval_warp M;  // 2

        load_q(ptr_q, Q, true);
        for (int k = 0; k < NUM_STAGES - 1; k++) {
            load_k(ptr_k, k, K[k], true);
            load_v(ptr_v, k, V[k], true);
        }

#pragma unroll
        for (auto &pack : O) {
#pragma unroll
            for (int i = 0; i < 8; i++) {
                pack.data[i] = 0;
            }
        }

        static constexpr float neginf = -std::numeric_limits<float>::infinity();
        L.fill(make_float2(0.0f, 0.0f));
        M.fill(make_float2(neginf, neginf));

        __shared__ q_warp Q_shmem[NUM_WARPS];
#pragma unroll
        for (int i = 0; i < Q.size(); i++) {
            store<true>(&Q_shmem[warpId][i], Q[i]);
        }

        int dummy = 0;

        // TODO: mask tokens in last block
        for (int k1 = 0; k1 < ntokens_kv / WARP_K; k1 += NUM_STAGES) {
#pragma unroll
            for (int k2 = 0; k2 < NUM_STAGES; k2++) {
#pragma unroll
                for (int i = 0; i < Q.size(); i++) {
                    Q[i] = load<true>(&Q_shmem[warpId][i]);
                }

                int nextk = k1 + k2 + NUM_STAGES - 1;
                int idx = (k2 + NUM_STAGES - 1) % NUM_STAGES;
                bool pred = nextk < ntokens_kv / WARP_K;
                load_k(ptr_k, nextk, K[idx], pred);
                load_v(ptr_v, nextk, V[idx], pred);

                // __syncthreads();
                // if (alwaysfalse) {
                //     dummy = clock();
                // }
                auto [P, rescale] = compute(Q, K[k2], V[k2], M, L, scale);
                O = compute_pv(P, V[idx], O, rescale);

                if (alwaysfalse) {
                    dummy = clock();
                }
                // asm volatile ("membar.cta;");
            }
        }

        unused_var(dummy, alwaysfalse);

        O = compute_o(O, L);

        auto f16psum = GEMM::packed_fp32_to_fp16(O);

        Epilogue()(typename GEMM::BlockInfo{
            .bm = binfo.batch * binfo.numBlocksM + binfo.bm,
            .bn = binfo.head,
            .numBlocksM = binfo.numBatch * binfo.numBlocksM,
            .numBlocksN = binfo.numHeads,
        }, f16psum, binfo.numBatch * binfo.numBlocksM * BLOCK_M, binfo.numHeads * HEAD_DIM, 0, epilogueArgs);
    }
#else
    template<typename Epilogue>
    __device__ __forceinline__ static void attention_fp16_block(const BlockInfo binfo,
                                                                const packed_q_t *ptr_q,
                                                                const packed_k_t *ptr_k,
                                                                const packed_v_t *ptr_v,
                                                                float scale,
                                                                int ntokens_q,
                                                                int ntokens_kv,
                                                                Epilogue::Arguments epilogueArgs,
                                                                bool alwaysfalse) {
        // constexpr int NUM_STAGES = 2;

        const int laneId = threadIdx.x % WARP_SIZE;
        const int warpId = threadIdx.x / WARP_SIZE;

        q_warp Q;      // 32
        k_warp K;      // 64
        v_warp V;      // 64
        o_warp O;      // 64
        rowval_warp L; // 2
        rowval_warp M; // 2

        load_q(ptr_q, Q, true);
        load_k(ptr_k, 0, K, true);

#pragma unroll
        for (auto &pack : O) {
#pragma unroll
            for (int i = 0; i < 8; i++) {
                pack.data[i] = 0;
            }
        }

        static constexpr float neginf =
            -std::numeric_limits<float>::max(); // not real inf, to prevent nan during computation
        L.fill(make_float2(0.0f, 0.0f));
        M.fill(make_float2(neginf, neginf));

        static constexpr int SHMEM_TILES = IS_SM80 ? 4 : 7;
        static_assert(SHMEM_TILES <= Q.size());
        using q_shmem_t = packed_q_t[NUM_WARPS][SHMEM_TILES][WARP_SIZE];
        __shared__ q_shmem_t Q_shmem;

#pragma unroll
        for (int i = 0; i < SHMEM_TILES; i++) {
            store<true>(&Q_shmem[warpId][i][laneId], Q[Q.size() - 1 - i]);
        }
        __syncwarp();

        int dummy = 0;

        // TODO: mask tokens in last block
        for (int k1 = 0; k1 < ntokens_kv / WARP_K; k1++) {
            if (alwaysfalse) {
                ptr_v += K[0].x;
            }

#pragma unroll
            for (int i = 0; i < SHMEM_TILES; i++) {
                Q[Q.size() - 1 - i] = load<true>(&Q_shmem[warpId][i][laneId]);
            }

            if constexpr (!IS_SM80) {
                if (k1 % 2 == 1) {
                    __syncthreads();
                }
            }

            if (alwaysfalse) {
                dummy = clock();
            }

            load_v(ptr_v, k1, V, true);

            if (alwaysfalse) {
                dummy = clock();
            }

            auto [P, rescale] = compute(Q, K, M, L, scale);

            if (alwaysfalse) {
                dummy = clock();
            }

            if (alwaysfalse) {
                ptr_k += V[0].x;
            }

            // if (alwaysfalse) {
            //     dummy = clock();
            // }

            load_k(ptr_k, k1 + 1, K, k1 + 1 < ntokens_kv / WARP_K);

            // if (alwaysfalse) {
            //     dummy = clock();
            // }

            O = compute_pv(P, V, O, rescale);

            if (alwaysfalse) {
                dummy = clock();
            }
        }

        unused_var(dummy, alwaysfalse);

        O = compute_o(O, L);

        auto f16psum = GEMM::packed_fp32_to_fp16(O);

        Epilogue()(
            typename GEMM::BlockInfo{
                .bm         = binfo.batch * binfo.numBlocksM + binfo.bm,
                .bn         = binfo.head,
                .numBlocksM = binfo.numBatch * binfo.numBlocksM,
                .numBlocksN = binfo.numHeads,
            },
            f16psum,
            binfo.numBatch * binfo.numBlocksM * BLOCK_M,
            binfo.numHeads * HEAD_DIM,
            0,
            epilogueArgs);
    }
#endif

    template<typename Epilogue>
    struct attention_fp16_kernel {
        static constexpr int MIN_ARCH   = std::is_same_v<half_t, __nv_bfloat16> ? 800 : 750;
        static constexpr int SHMEM_SIZE = 0; // sizeof(q_shmem_t);

        __device__ void operator()(const packed_q_t *ptr_q,
                                   const packed_k_t *ptr_k,
                                   const packed_v_t *ptr_v,
                                   float scale,
                                   int ntokens_q,
                                   int ntokens_kv,
                                   Epilogue::Arguments epilogueArgs,
                                   bool alwaysfalse) {
            BlockInfo binfo = {
                .bm         = (int)blockIdx.x,
                .head       = (int)blockIdx.y,
                .batch      = (int)blockIdx.z,
                .numBlocksM = (int)gridDim.x,
                .numHeads   = (int)gridDim.y,
                .numBatch   = (int)gridDim.z,
            };

            // extern __shared__ uint8_t shmem[];
            // q_shmem_t *Q_shmem = reinterpret_cast<q_shmem_t *>(shmem);

            const int ktiles = ceilDiv(ntokens_kv, WARP_K);

            attention_fp16_block<Epilogue>(
                binfo,
                ptr_q + ((binfo.batch * binfo.numHeads + binfo.head) * binfo.numBlocksM + binfo.bm) * NUM_WARPS *
                            WARP_M_TILES * WARP_D_TILES * WARP_SIZE,
                ptr_k +
                    (binfo.batch * binfo.numHeads + binfo.head) * ktiles * WARP_K_TILES_QK * WARP_D_TILES * WARP_SIZE,
                ptr_v +
                    (binfo.batch * binfo.numHeads + binfo.head) * ktiles * WARP_K_TILES_PV * WARP_N_TILES * WARP_SIZE,
                scale,
                ntokens_q,
                ntokens_kv,
                // *Q_shmem,
                epilogueArgs,
                alwaysfalse);
        }
    };
};

}; // namespace nunchaku::kernels
