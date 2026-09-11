/* Private micro8 B kernel. Caller validates spans and scales C by beta. */
#ifndef CAMBLAS_SVE_INTERLEAVED_H
#define CAMBLAS_SVE_INTERLEAVED_H
#include "sve_lane_ops.h"
#if defined(CAMBLAS_MICRO8_PAIRED_ASM) && CAMBLAS_MICRO8_PAIRED_ASM
#include "paired_micro8.h"
#endif
#if defined(CAMBLAS_MICRO8_PIPE_B) && CAMBLAS_MICRO8_PIPE_B
#include "sve_pipe_b.h"
#endif
#if defined(CAMBLAS_MICRO8_NEON) && CAMBLAS_MICRO8_NEON
#include "neon_micro8.h"
#endif
/* Grouped loads preserve the intrinsic layout while exposing constant offsets.
 * The memory clobber declares reads hidden inside the assembly to the compiler.
 * These helpers are used only for full, already validated eight-column groups. */
static inline void camblas_micro8_load_b_f32(svbool_t pg, const float *p, svfloat32_t *b0,
                                             svfloat32_t *b1)
{
#if defined(CAMBLAS_MICRO8_BASE_LOADS) && CAMBLAS_MICRO8_BASE_LOADS
    __asm__ volatile("ld1rqw {%0.s}, %2/z, [%3]\n\t"
                     "ld1rqw {%1.s}, %2/z, [%3, #16]"
                     : "=y"(*b0), "=y"(*b1)
                     : "Upl"(pg), "r"(p)
                     : "memory");
#else
    *b0 = svld1rq_f32(pg, p);
    *b1 = svld1rq_f32(pg, p + 4);
#endif
}
static inline void camblas_micro8_load_b_f64(svbool_t pg, const double *p, svfloat64_t *b0,
                                             svfloat64_t *b1, svfloat64_t *b2, svfloat64_t *b3)
{
#if defined(CAMBLAS_MICRO8_BASE_LOADS) && CAMBLAS_MICRO8_BASE_LOADS
    __asm__ volatile("ld1rqd {%0.d}, %4/z, [%5]\n\t"
                     "ld1rqd {%1.d}, %4/z, [%5, #16]\n\t"
                     "ld1rqd {%2.d}, %4/z, [%5, #32]\n\t"
                     "ld1rqd {%3.d}, %4/z, [%5, #48]"
                     : "=y"(*b0), "=y"(*b1), "=y"(*b2), "=y"(*b3)
                     : "Upl"(pg), "r"(p)
                     : "memory");
#else
    *b0 = svld1rq_f64(pg, p);
    *b1 = svld1rq_f64(pg, p + 2);
    *b2 = svld1rq_f64(pg, p + 4);
    *b3 = svld1rq_f64(pg, p + 6);
#endif
}
#ifndef CAMBLAS_MICRO8_ROWS
#define CAMBLAS_MICRO8_ROWS 3
#endif
#ifndef CAMBLAS_MICRO8_PREFETCH_K
#define CAMBLAS_MICRO8_PREFETCH_K 0
#endif
#ifndef CAMBLAS_MICRO8_UNROLL_FACTOR
#define CAMBLAS_MICRO8_UNROLL_FACTOR 2
#endif
#if CAMBLAS_MICRO8_UNROLL_FACTOR != 2 && CAMBLAS_MICRO8_UNROLL_FACTOR != 4
#error "micro8 guarded unroll factor must be two or four"
#endif
#if CAMBLAS_MICRO8_ROWS != 2 && CAMBLAS_MICRO8_ROWS != 3
#error "micro8 supports two or three vector rows"
#endif
static inline void camblas_micro8_store_f32(svbool_t pg, float *p, svfloat32_t acc, float alpha,
                                            int initialize)
{
    svfloat32_t value = svmul_n_f32_x(pg, acc, alpha);
    if (!initialize)
        value = svadd_f32_x(pg, svld1_f32(pg, p), value);
    svst1_f32(pg, p, value);
}
static inline void camblas_micro8_store_f64(svbool_t pg, double *p, svfloat64_t acc, double alpha,
                                            int initialize)
{
    svfloat64_t value = svmul_n_f64_x(pg, acc, alpha);
    if (!initialize)
        value = svadd_f64_x(pg, svld1_f64(pg, p), value);
    svst1_f64(pg, p, value);
}
static inline size_t camblas_micro8_a_index(size_t row, size_t depth, size_t ld, size_t group,
                                            int micro)
{
    return micro ? (row / group * group) * ld + depth * group + row % group : row + depth * ld;
}
static inline int camblas_micro8_layout_f32(int m, int n, int k, float alpha, const float *A,
                                            size_t lda, const float *B, size_t ldb, float *C,
                                            size_t ldc, int a_micro, int initialize)
{
#if defined(CAMBLAS_MICRO8_NEON) && CAMBLAS_MICRO8_NEON
    if (!a_micro && svcntb() == 16)
        return camblas_neon_micro8_f32(m, n, k, alpha, A, lda, B, ldb, C, ldc, initialize);
#endif
    const size_t vl = svcntw();
    const svbool_t all = svptrue_b32();
    for (size_t j = 0; j < (size_t)n; j += 8) {
        size_t i = 0;
        for (; i + CAMBLAS_MICRO8_ROWS * vl <= (size_t)m; i += CAMBLAS_MICRO8_ROWS * vl) {
#if defined(CAMBLAS_MICRO8_PAIRED_ASM) && CAMBLAS_MICRO8_PAIRED_ASM && CAMBLAS_MICRO8_ROWS == 3
            if (!a_micro && vl == 4 && j + 8 <= (size_t)n) {
                camblas_paired_f32_block(k, A + i, lda, B + j * ldb, C + i + j * ldc, ldc, alpha,
                                         initialize);
                continue;
            }
#endif
#if defined(CAMBLAS_MICRO8_PIPE_B) && CAMBLAS_MICRO8_PIPE_B && CAMBLAS_MICRO8_ROWS == 3
            if (!a_micro && vl == 4 && j + 8 <= (size_t)n) {
                camblas_pipe_b_f32_block(k, A + i, lda, B + j * ldb, C + i + j * ldc, ldc, alpha,
                                         initialize);
                continue;
            }
#endif
            svfloat32_t c00 = svdup_f32(0);
            svfloat32_t c10 = svdup_f32(0);
#if CAMBLAS_MICRO8_ROWS == 3
            svfloat32_t c20 = svdup_f32(0);
#endif
            svfloat32_t c01 = svdup_f32(0);
            svfloat32_t c11 = svdup_f32(0);
#if CAMBLAS_MICRO8_ROWS == 3
            svfloat32_t c21 = svdup_f32(0);
#endif
            svfloat32_t c02 = svdup_f32(0);
            svfloat32_t c12 = svdup_f32(0);
#if CAMBLAS_MICRO8_ROWS == 3
            svfloat32_t c22 = svdup_f32(0);
#endif
            svfloat32_t c03 = svdup_f32(0);
            svfloat32_t c13 = svdup_f32(0);
#if CAMBLAS_MICRO8_ROWS == 3
            svfloat32_t c23 = svdup_f32(0);
#endif
            svfloat32_t c04 = svdup_f32(0);
            svfloat32_t c14 = svdup_f32(0);
#if CAMBLAS_MICRO8_ROWS == 3
            svfloat32_t c24 = svdup_f32(0);
#endif
            svfloat32_t c05 = svdup_f32(0);
            svfloat32_t c15 = svdup_f32(0);
#if CAMBLAS_MICRO8_ROWS == 3
            svfloat32_t c25 = svdup_f32(0);
#endif
            svfloat32_t c06 = svdup_f32(0);
            svfloat32_t c16 = svdup_f32(0);
#if CAMBLAS_MICRO8_ROWS == 3
            svfloat32_t c26 = svdup_f32(0);
#endif
            svfloat32_t c07 = svdup_f32(0);
            svfloat32_t c17 = svdup_f32(0);
#if CAMBLAS_MICRO8_ROWS == 3
            svfloat32_t c27 = svdup_f32(0);
#endif
#if defined(CAMBLAS_MICRO8_PIPE_A) && CAMBLAS_MICRO8_PIPE_A && CAMBLAS_MICRO8_ROWS == 3
            if (k > 0) {
                svfloat32_t a0 =
                    svld1_f32(all, A + camblas_micro8_a_index(i + 0 * vl, 0, lda,
                                                              CAMBLAS_MICRO8_ROWS * vl, a_micro));
                svfloat32_t a1 =
                    svld1_f32(all, A + camblas_micro8_a_index(i + 1 * vl, 0, lda,
                                                              CAMBLAS_MICRO8_ROWS * vl, a_micro));
                svfloat32_t a2 =
                    svld1_f32(all, A + camblas_micro8_a_index(i + 2 * vl, 0, lda,
                                                              CAMBLAS_MICRO8_ROWS * vl, a_micro));
#if CAMBLAS_MICRO8_PIPE_A == 2
                /* Peel the final K step: steady-state loads remain in bounds. */
#if defined(CAMBLAS_MICRO8_UNROLL2) && CAMBLAS_MICRO8_UNROLL2
#if CAMBLAS_MICRO8_UNROLL_FACTOR == 4
#pragma GCC unroll 4
#else
#pragma GCC unroll 2
#endif
#endif
                for (size_t l = 0; l + 1 < (size_t)k; l++) {
#if CAMBLAS_MICRO8_PREFETCH_K > 0
                    /* Form addresses only for a future, valid A column.
                       The first and last vector cover possible line splits. */
                    if ((size_t)k - l > CAMBLAS_MICRO8_PREFETCH_K) {
                        size_t ahead = l + CAMBLAS_MICRO8_PREFETCH_K;
                        __builtin_prefetch(A + camblas_micro8_a_index(i, ahead, lda,
                                                                      CAMBLAS_MICRO8_ROWS * vl,
                                                                      a_micro),
                                           0, 3);
                        __builtin_prefetch(A + camblas_micro8_a_index(i + 2 * vl, ahead, lda,
                                                                      CAMBLAS_MICRO8_ROWS * vl,
                                                                      a_micro),
                                           0, 3);
                    }
#endif
                    svfloat32_t b0, b1;
                    camblas_micro8_load_b_f32(all, B + j * ldb + l * 8, &b0, &b1);
                    CAMBLAS_LANE_F32(c00, a0, b0, 0);
                    CAMBLAS_LANE_F32(c01, a0, b0, 1);
                    CAMBLAS_LANE_F32(c02, a0, b0, 2);
                    CAMBLAS_LANE_F32(c03, a0, b0, 3);
                    CAMBLAS_LANE_F32(c04, a0, b1, 0);
                    CAMBLAS_LANE_F32(c05, a0, b1, 1);
                    CAMBLAS_LANE_F32(c06, a0, b1, 2);
                    CAMBLAS_LANE_F32(c07, a0, b1, 3);
                    a0 = svld1_f32(all,
                                   A + camblas_micro8_a_index(i + 0 * vl, l + 1, lda,
                                                              CAMBLAS_MICRO8_ROWS * vl, a_micro));
                    CAMBLAS_LANE_F32(c10, a1, b0, 0);
                    CAMBLAS_LANE_F32(c11, a1, b0, 1);
                    CAMBLAS_LANE_F32(c12, a1, b0, 2);
                    CAMBLAS_LANE_F32(c13, a1, b0, 3);
                    CAMBLAS_LANE_F32(c14, a1, b1, 0);
                    CAMBLAS_LANE_F32(c15, a1, b1, 1);
                    CAMBLAS_LANE_F32(c16, a1, b1, 2);
                    CAMBLAS_LANE_F32(c17, a1, b1, 3);
                    a1 = svld1_f32(all,
                                   A + camblas_micro8_a_index(i + 1 * vl, l + 1, lda,
                                                              CAMBLAS_MICRO8_ROWS * vl, a_micro));
                    CAMBLAS_LANE_F32(c20, a2, b0, 0);
                    CAMBLAS_LANE_F32(c21, a2, b0, 1);
                    CAMBLAS_LANE_F32(c22, a2, b0, 2);
                    CAMBLAS_LANE_F32(c23, a2, b0, 3);
                    CAMBLAS_LANE_F32(c24, a2, b1, 0);
                    CAMBLAS_LANE_F32(c25, a2, b1, 1);
                    CAMBLAS_LANE_F32(c26, a2, b1, 2);
                    CAMBLAS_LANE_F32(c27, a2, b1, 3);
                    a2 = svld1_f32(all,
                                   A + camblas_micro8_a_index(i + 2 * vl, l + 1, lda,
                                                              CAMBLAS_MICRO8_ROWS * vl, a_micro));
#if defined(CAMBLAS_MICRO8_UNROLL2) && CAMBLAS_MICRO8_UNROLL2
                    /* Keep the next unrolled B loads behind this K step. */
                    __asm__ volatile("" ::: "memory");
#endif
                }
                {
                    const size_t l = (size_t)k - 1;
                    svfloat32_t b0, b1;
                    camblas_micro8_load_b_f32(all, B + j * ldb + l * 8, &b0, &b1);
                    CAMBLAS_LANE_F32(c00, a0, b0, 0);
                    CAMBLAS_LANE_F32(c01, a0, b0, 1);
                    CAMBLAS_LANE_F32(c02, a0, b0, 2);
                    CAMBLAS_LANE_F32(c03, a0, b0, 3);
                    CAMBLAS_LANE_F32(c04, a0, b1, 0);
                    CAMBLAS_LANE_F32(c05, a0, b1, 1);
                    CAMBLAS_LANE_F32(c06, a0, b1, 2);
                    CAMBLAS_LANE_F32(c07, a0, b1, 3);
                    CAMBLAS_LANE_F32(c10, a1, b0, 0);
                    CAMBLAS_LANE_F32(c11, a1, b0, 1);
                    CAMBLAS_LANE_F32(c12, a1, b0, 2);
                    CAMBLAS_LANE_F32(c13, a1, b0, 3);
                    CAMBLAS_LANE_F32(c14, a1, b1, 0);
                    CAMBLAS_LANE_F32(c15, a1, b1, 1);
                    CAMBLAS_LANE_F32(c16, a1, b1, 2);
                    CAMBLAS_LANE_F32(c17, a1, b1, 3);
                    CAMBLAS_LANE_F32(c20, a2, b0, 0);
                    CAMBLAS_LANE_F32(c21, a2, b0, 1);
                    CAMBLAS_LANE_F32(c22, a2, b0, 2);
                    CAMBLAS_LANE_F32(c23, a2, b0, 3);
                    CAMBLAS_LANE_F32(c24, a2, b1, 0);
                    CAMBLAS_LANE_F32(c25, a2, b1, 1);
                    CAMBLAS_LANE_F32(c26, a2, b1, 2);
                    CAMBLAS_LANE_F32(c27, a2, b1, 3);
                }
#else
                for (size_t l = 0; l < (size_t)k; l++) {
                    svfloat32_t b0 = svld1rq_f32(all, B + j * ldb + l * 8 + 0);
                    svfloat32_t b1 = svld1rq_f32(all, B + j * ldb + l * 8 + 4);
                    CAMBLAS_LANE_F32(c00, a0, b0, 0);
                    CAMBLAS_LANE_F32(c01, a0, b0, 1);
                    CAMBLAS_LANE_F32(c02, a0, b0, 2);
                    CAMBLAS_LANE_F32(c03, a0, b0, 3);
                    CAMBLAS_LANE_F32(c04, a0, b1, 0);
                    CAMBLAS_LANE_F32(c05, a0, b1, 1);
                    CAMBLAS_LANE_F32(c06, a0, b1, 2);
                    CAMBLAS_LANE_F32(c07, a0, b1, 3);
                    if (l + 1 < (size_t)k)
                        a0 = svld1_f32(all, A + camblas_micro8_a_index(i + 0 * vl, l + 1, lda,
                                                                       CAMBLAS_MICRO8_ROWS * vl,
                                                                       a_micro));
                    CAMBLAS_LANE_F32(c10, a1, b0, 0);
                    CAMBLAS_LANE_F32(c11, a1, b0, 1);
                    CAMBLAS_LANE_F32(c12, a1, b0, 2);
                    CAMBLAS_LANE_F32(c13, a1, b0, 3);
                    CAMBLAS_LANE_F32(c14, a1, b1, 0);
                    CAMBLAS_LANE_F32(c15, a1, b1, 1);
                    CAMBLAS_LANE_F32(c16, a1, b1, 2);
                    CAMBLAS_LANE_F32(c17, a1, b1, 3);
                    if (l + 1 < (size_t)k)
                        a1 = svld1_f32(all, A + camblas_micro8_a_index(i + 1 * vl, l + 1, lda,
                                                                       CAMBLAS_MICRO8_ROWS * vl,
                                                                       a_micro));
                    CAMBLAS_LANE_F32(c20, a2, b0, 0);
                    CAMBLAS_LANE_F32(c21, a2, b0, 1);
                    CAMBLAS_LANE_F32(c22, a2, b0, 2);
                    CAMBLAS_LANE_F32(c23, a2, b0, 3);
                    CAMBLAS_LANE_F32(c24, a2, b1, 0);
                    CAMBLAS_LANE_F32(c25, a2, b1, 1);
                    CAMBLAS_LANE_F32(c26, a2, b1, 2);
                    CAMBLAS_LANE_F32(c27, a2, b1, 3);
                    if (l + 1 < (size_t)k)
                        a2 = svld1_f32(all, A + camblas_micro8_a_index(i + 2 * vl, l + 1, lda,
                                                                       CAMBLAS_MICRO8_ROWS * vl,
                                                                       a_micro));
                }
#endif
            }
#else
#if defined(CAMBLAS_MICRO8_UNROLL2) && CAMBLAS_MICRO8_UNROLL2
#if CAMBLAS_MICRO8_UNROLL_FACTOR == 4
#pragma GCC unroll 4
#else
#pragma GCC unroll 2
#endif
#endif
            for (size_t l = 0; l < (size_t)k; l++) {
                svfloat32_t a0 =
                    svld1_f32(all, A + camblas_micro8_a_index(i + 0 * vl, l, lda,
                                                              CAMBLAS_MICRO8_ROWS * vl, a_micro));
                svfloat32_t a1 =
                    svld1_f32(all, A + camblas_micro8_a_index(i + 1 * vl, l, lda,
                                                              CAMBLAS_MICRO8_ROWS * vl, a_micro));
#if CAMBLAS_MICRO8_ROWS == 3
                svfloat32_t a2 =
                    svld1_f32(all, A + camblas_micro8_a_index(i + 2 * vl, l, lda,
                                                              CAMBLAS_MICRO8_ROWS * vl, a_micro));
#endif
                svfloat32_t b0 = svld1rq_f32(all, B + j * ldb + l * 8 + 0);
                svfloat32_t b1 = svld1rq_f32(all, B + j * ldb + l * 8 + 4);
#if defined(CAMBLAS_MICRO8_ALTERNATE) && CAMBLAS_MICRO8_ALTERNATE
                CAMBLAS_LANE_F32(c00, a0, b0, 0);
                CAMBLAS_LANE_F32(c10, a1, b0, 0);
#if CAMBLAS_MICRO8_ROWS == 3
                CAMBLAS_LANE_F32(c20, a2, b0, 0);
#endif
                CAMBLAS_LANE_F32(c14, a1, b1, 0);
#if CAMBLAS_MICRO8_ROWS == 3
                CAMBLAS_LANE_F32(c24, a2, b1, 0);
#endif
                CAMBLAS_LANE_F32(c04, a0, b1, 0);
#if CAMBLAS_MICRO8_ROWS == 3
                CAMBLAS_LANE_F32(c21, a2, b0, 1);
#endif
                CAMBLAS_LANE_F32(c01, a0, b0, 1);
                CAMBLAS_LANE_F32(c11, a1, b0, 1);
                CAMBLAS_LANE_F32(c05, a0, b1, 1);
                CAMBLAS_LANE_F32(c15, a1, b1, 1);
#if CAMBLAS_MICRO8_ROWS == 3
                CAMBLAS_LANE_F32(c25, a2, b1, 1);
#endif
                CAMBLAS_LANE_F32(c12, a1, b0, 2);
#if CAMBLAS_MICRO8_ROWS == 3
                CAMBLAS_LANE_F32(c22, a2, b0, 2);
#endif
                CAMBLAS_LANE_F32(c02, a0, b0, 2);
#if CAMBLAS_MICRO8_ROWS == 3
                CAMBLAS_LANE_F32(c26, a2, b1, 2);
#endif
                CAMBLAS_LANE_F32(c06, a0, b1, 2);
                CAMBLAS_LANE_F32(c16, a1, b1, 2);
                CAMBLAS_LANE_F32(c03, a0, b0, 3);
                CAMBLAS_LANE_F32(c13, a1, b0, 3);
#if CAMBLAS_MICRO8_ROWS == 3
                CAMBLAS_LANE_F32(c23, a2, b0, 3);
#endif
                CAMBLAS_LANE_F32(c17, a1, b1, 3);
#if CAMBLAS_MICRO8_ROWS == 3
                CAMBLAS_LANE_F32(c27, a2, b1, 3);
#endif
                CAMBLAS_LANE_F32(c07, a0, b1, 3);
#else
                CAMBLAS_LANE_F32(c00, a0, b0, 0);
                CAMBLAS_LANE_F32(c10, a1, b0, 0);
#if CAMBLAS_MICRO8_ROWS == 3
                CAMBLAS_LANE_F32(c20, a2, b0, 0);
#endif
                CAMBLAS_LANE_F32(c01, a0, b0, 1);
                CAMBLAS_LANE_F32(c11, a1, b0, 1);
#if CAMBLAS_MICRO8_ROWS == 3
                CAMBLAS_LANE_F32(c21, a2, b0, 1);
#endif
                CAMBLAS_LANE_F32(c02, a0, b0, 2);
                CAMBLAS_LANE_F32(c12, a1, b0, 2);
#if CAMBLAS_MICRO8_ROWS == 3
                CAMBLAS_LANE_F32(c22, a2, b0, 2);
#endif
                CAMBLAS_LANE_F32(c03, a0, b0, 3);
                CAMBLAS_LANE_F32(c13, a1, b0, 3);
#if CAMBLAS_MICRO8_ROWS == 3
                CAMBLAS_LANE_F32(c23, a2, b0, 3);
#endif
                CAMBLAS_LANE_F32(c04, a0, b1, 0);
                CAMBLAS_LANE_F32(c14, a1, b1, 0);
#if CAMBLAS_MICRO8_ROWS == 3
                CAMBLAS_LANE_F32(c24, a2, b1, 0);
#endif
                CAMBLAS_LANE_F32(c05, a0, b1, 1);
                CAMBLAS_LANE_F32(c15, a1, b1, 1);
#if CAMBLAS_MICRO8_ROWS == 3
                CAMBLAS_LANE_F32(c25, a2, b1, 1);
#endif
                CAMBLAS_LANE_F32(c06, a0, b1, 2);
                CAMBLAS_LANE_F32(c16, a1, b1, 2);
#if CAMBLAS_MICRO8_ROWS == 3
                CAMBLAS_LANE_F32(c26, a2, b1, 2);
#endif
                CAMBLAS_LANE_F32(c07, a0, b1, 3);
                CAMBLAS_LANE_F32(c17, a1, b1, 3);
#if CAMBLAS_MICRO8_ROWS == 3
                CAMBLAS_LANE_F32(c27, a2, b1, 3);
#endif
#endif
            }
#endif
            if (j + 0 < (size_t)n) {
                float *p0 = C + i + 0 * vl + (j + 0) * ldc;
                camblas_micro8_store_f32(all, p0, c00, alpha, initialize);
                float *p1 = C + i + 1 * vl + (j + 0) * ldc;
                camblas_micro8_store_f32(all, p1, c10, alpha, initialize);
#if CAMBLAS_MICRO8_ROWS == 3
                float *p2 = C + i + 2 * vl + (j + 0) * ldc;
#endif
#if CAMBLAS_MICRO8_ROWS == 3
                camblas_micro8_store_f32(all, p2, c20, alpha, initialize);
#endif
            }
            if (j + 1 < (size_t)n) {
                float *p0 = C + i + 0 * vl + (j + 1) * ldc;
                camblas_micro8_store_f32(all, p0, c01, alpha, initialize);
                float *p1 = C + i + 1 * vl + (j + 1) * ldc;
                camblas_micro8_store_f32(all, p1, c11, alpha, initialize);
#if CAMBLAS_MICRO8_ROWS == 3
                float *p2 = C + i + 2 * vl + (j + 1) * ldc;
#endif
#if CAMBLAS_MICRO8_ROWS == 3
                camblas_micro8_store_f32(all, p2, c21, alpha, initialize);
#endif
            }
            if (j + 2 < (size_t)n) {
                float *p0 = C + i + 0 * vl + (j + 2) * ldc;
                camblas_micro8_store_f32(all, p0, c02, alpha, initialize);
                float *p1 = C + i + 1 * vl + (j + 2) * ldc;
                camblas_micro8_store_f32(all, p1, c12, alpha, initialize);
#if CAMBLAS_MICRO8_ROWS == 3
                float *p2 = C + i + 2 * vl + (j + 2) * ldc;
#endif
#if CAMBLAS_MICRO8_ROWS == 3
                camblas_micro8_store_f32(all, p2, c22, alpha, initialize);
#endif
            }
            if (j + 3 < (size_t)n) {
                float *p0 = C + i + 0 * vl + (j + 3) * ldc;
                camblas_micro8_store_f32(all, p0, c03, alpha, initialize);
                float *p1 = C + i + 1 * vl + (j + 3) * ldc;
                camblas_micro8_store_f32(all, p1, c13, alpha, initialize);
#if CAMBLAS_MICRO8_ROWS == 3
                float *p2 = C + i + 2 * vl + (j + 3) * ldc;
#endif
#if CAMBLAS_MICRO8_ROWS == 3
                camblas_micro8_store_f32(all, p2, c23, alpha, initialize);
#endif
            }
            if (j + 4 < (size_t)n) {
                float *p0 = C + i + 0 * vl + (j + 4) * ldc;
                camblas_micro8_store_f32(all, p0, c04, alpha, initialize);
                float *p1 = C + i + 1 * vl + (j + 4) * ldc;
                camblas_micro8_store_f32(all, p1, c14, alpha, initialize);
#if CAMBLAS_MICRO8_ROWS == 3
                float *p2 = C + i + 2 * vl + (j + 4) * ldc;
#endif
#if CAMBLAS_MICRO8_ROWS == 3
                camblas_micro8_store_f32(all, p2, c24, alpha, initialize);
#endif
            }
            if (j + 5 < (size_t)n) {
                float *p0 = C + i + 0 * vl + (j + 5) * ldc;
                camblas_micro8_store_f32(all, p0, c05, alpha, initialize);
                float *p1 = C + i + 1 * vl + (j + 5) * ldc;
                camblas_micro8_store_f32(all, p1, c15, alpha, initialize);
#if CAMBLAS_MICRO8_ROWS == 3
                float *p2 = C + i + 2 * vl + (j + 5) * ldc;
#endif
#if CAMBLAS_MICRO8_ROWS == 3
                camblas_micro8_store_f32(all, p2, c25, alpha, initialize);
#endif
            }
            if (j + 6 < (size_t)n) {
                float *p0 = C + i + 0 * vl + (j + 6) * ldc;
                camblas_micro8_store_f32(all, p0, c06, alpha, initialize);
                float *p1 = C + i + 1 * vl + (j + 6) * ldc;
                camblas_micro8_store_f32(all, p1, c16, alpha, initialize);
#if CAMBLAS_MICRO8_ROWS == 3
                float *p2 = C + i + 2 * vl + (j + 6) * ldc;
#endif
#if CAMBLAS_MICRO8_ROWS == 3
                camblas_micro8_store_f32(all, p2, c26, alpha, initialize);
#endif
            }
            if (j + 7 < (size_t)n) {
                float *p0 = C + i + 0 * vl + (j + 7) * ldc;
                camblas_micro8_store_f32(all, p0, c07, alpha, initialize);
                float *p1 = C + i + 1 * vl + (j + 7) * ldc;
                camblas_micro8_store_f32(all, p1, c17, alpha, initialize);
#if CAMBLAS_MICRO8_ROWS == 3
                float *p2 = C + i + 2 * vl + (j + 7) * ldc;
#endif
#if CAMBLAS_MICRO8_ROWS == 3
                camblas_micro8_store_f32(all, p2, c27, alpha, initialize);
#endif
            }
        }
        for (; i < (size_t)m; i += vl) {
            svbool_t pg = svwhilelt_b32((uint64_t)i, (uint64_t)m);
#if defined(CAMBLAS_MICRO8_WIDE_TAIL) && CAMBLAS_MICRO8_WIDE_TAIL
            svfloat32_t t0 = svdup_f32(0);
            svfloat32_t t1 = svdup_f32(0);
            svfloat32_t t2 = svdup_f32(0);
            svfloat32_t t3 = svdup_f32(0);
            svfloat32_t t4 = svdup_f32(0);
            svfloat32_t t5 = svdup_f32(0);
            svfloat32_t t6 = svdup_f32(0);
            svfloat32_t t7 = svdup_f32(0);
            for (size_t l = 0; l < (size_t)k; l++) {
                svfloat32_t a = svld1_f32(
                    pg, A + camblas_micro8_a_index(i, l, lda, CAMBLAS_MICRO8_ROWS * vl, a_micro));
                svfloat32_t b0 = svld1rq_f32(all, B + j * ldb + l * 8 + 0);
                svfloat32_t b1 = svld1rq_f32(all, B + j * ldb + l * 8 + 4);
                CAMBLAS_LANE_F32(t0, a, b0, 0);
                CAMBLAS_LANE_F32(t1, a, b0, 1);
                CAMBLAS_LANE_F32(t2, a, b0, 2);
                CAMBLAS_LANE_F32(t3, a, b0, 3);
                CAMBLAS_LANE_F32(t4, a, b1, 0);
                CAMBLAS_LANE_F32(t5, a, b1, 1);
                CAMBLAS_LANE_F32(t6, a, b1, 2);
                CAMBLAS_LANE_F32(t7, a, b1, 3);
            }
            if (j + 0 < (size_t)n) {
                float *p = C + i + (j + 0) * ldc;
                camblas_micro8_store_f32(pg, p, t0, alpha, initialize);
            }
            if (j + 1 < (size_t)n) {
                float *p = C + i + (j + 1) * ldc;
                camblas_micro8_store_f32(pg, p, t1, alpha, initialize);
            }
            if (j + 2 < (size_t)n) {
                float *p = C + i + (j + 2) * ldc;
                camblas_micro8_store_f32(pg, p, t2, alpha, initialize);
            }
            if (j + 3 < (size_t)n) {
                float *p = C + i + (j + 3) * ldc;
                camblas_micro8_store_f32(pg, p, t3, alpha, initialize);
            }
            if (j + 4 < (size_t)n) {
                float *p = C + i + (j + 4) * ldc;
                camblas_micro8_store_f32(pg, p, t4, alpha, initialize);
            }
            if (j + 5 < (size_t)n) {
                float *p = C + i + (j + 5) * ldc;
                camblas_micro8_store_f32(pg, p, t5, alpha, initialize);
            }
            if (j + 6 < (size_t)n) {
                float *p = C + i + (j + 6) * ldc;
                camblas_micro8_store_f32(pg, p, t6, alpha, initialize);
            }
            if (j + 7 < (size_t)n) {
                float *p = C + i + (j + 7) * ldc;
                camblas_micro8_store_f32(pg, p, t7, alpha, initialize);
            }
#else
            for (size_t q = 0; q < 8 && j + q < (size_t)n; q++) {
                svfloat32_t acc = svdup_f32(0);
                for (size_t l = 0; l < (size_t)k; l++)
                    acc = svmla_n_f32_x(
                        pg, acc,
                        svld1_f32(pg, A + camblas_micro8_a_index(
                                              i, l, lda, CAMBLAS_MICRO8_ROWS * vl, a_micro)),
                        B[j * ldb + l * 8 + q]);
                float *p = C + i + (j + q) * ldc;
                camblas_micro8_store_f32(pg, p, acc, alpha, initialize);
            }
#endif
        }
    }
    return 0;
}
static inline int camblas_micro8_layout_f64(int m, int n, int k, double alpha, const double *A,
                                            size_t lda, const double *B, size_t ldb, double *C,
                                            size_t ldc, int a_micro, int initialize)
{
#if defined(CAMBLAS_MICRO8_NEON) && CAMBLAS_MICRO8_NEON
    if (!a_micro && svcntb() == 16)
        return camblas_neon_micro8_f64(m, n, k, alpha, A, lda, B, ldb, C, ldc, initialize);
#endif
    const size_t vl = svcntd();
    const svbool_t all = svptrue_b64();
    for (size_t j = 0; j < (size_t)n; j += 8) {
        size_t i = 0;
        for (; i + CAMBLAS_MICRO8_ROWS * vl <= (size_t)m; i += CAMBLAS_MICRO8_ROWS * vl) {
#if defined(CAMBLAS_MICRO8_PAIRED_ASM) && CAMBLAS_MICRO8_PAIRED_ASM && CAMBLAS_MICRO8_ROWS == 3
            if (!a_micro && vl == 2 && j + 8 <= (size_t)n) {
                camblas_paired_f64_block(k, A + i, lda, B + j * ldb, C + i + j * ldc, ldc, alpha,
                                         initialize);
                continue;
            }
#endif
            svfloat64_t c00 = svdup_f64(0);
            svfloat64_t c10 = svdup_f64(0);
#if CAMBLAS_MICRO8_ROWS == 3
            svfloat64_t c20 = svdup_f64(0);
#endif
            svfloat64_t c01 = svdup_f64(0);
            svfloat64_t c11 = svdup_f64(0);
#if CAMBLAS_MICRO8_ROWS == 3
            svfloat64_t c21 = svdup_f64(0);
#endif
            svfloat64_t c02 = svdup_f64(0);
            svfloat64_t c12 = svdup_f64(0);
#if CAMBLAS_MICRO8_ROWS == 3
            svfloat64_t c22 = svdup_f64(0);
#endif
            svfloat64_t c03 = svdup_f64(0);
            svfloat64_t c13 = svdup_f64(0);
#if CAMBLAS_MICRO8_ROWS == 3
            svfloat64_t c23 = svdup_f64(0);
#endif
            svfloat64_t c04 = svdup_f64(0);
            svfloat64_t c14 = svdup_f64(0);
#if CAMBLAS_MICRO8_ROWS == 3
            svfloat64_t c24 = svdup_f64(0);
#endif
            svfloat64_t c05 = svdup_f64(0);
            svfloat64_t c15 = svdup_f64(0);
#if CAMBLAS_MICRO8_ROWS == 3
            svfloat64_t c25 = svdup_f64(0);
#endif
            svfloat64_t c06 = svdup_f64(0);
            svfloat64_t c16 = svdup_f64(0);
#if CAMBLAS_MICRO8_ROWS == 3
            svfloat64_t c26 = svdup_f64(0);
#endif
            svfloat64_t c07 = svdup_f64(0);
            svfloat64_t c17 = svdup_f64(0);
#if CAMBLAS_MICRO8_ROWS == 3
            svfloat64_t c27 = svdup_f64(0);
#endif
#if defined(CAMBLAS_MICRO8_PIPE_A) && CAMBLAS_MICRO8_PIPE_A && CAMBLAS_MICRO8_ROWS == 3
            if (k > 0) {
                svfloat64_t a0 =
                    svld1_f64(all, A + camblas_micro8_a_index(i + 0 * vl, 0, lda,
                                                              CAMBLAS_MICRO8_ROWS * vl, a_micro));
                svfloat64_t a1 =
                    svld1_f64(all, A + camblas_micro8_a_index(i + 1 * vl, 0, lda,
                                                              CAMBLAS_MICRO8_ROWS * vl, a_micro));
                svfloat64_t a2 =
                    svld1_f64(all, A + camblas_micro8_a_index(i + 2 * vl, 0, lda,
                                                              CAMBLAS_MICRO8_ROWS * vl, a_micro));
#if CAMBLAS_MICRO8_PIPE_A == 2
                /* Peel the final K step: steady-state loads remain in bounds. */
#if defined(CAMBLAS_MICRO8_UNROLL2) && CAMBLAS_MICRO8_UNROLL2
#if CAMBLAS_MICRO8_UNROLL_FACTOR == 4
#pragma GCC unroll 4
#else
#pragma GCC unroll 2
#endif
#endif
                for (size_t l = 0; l + 1 < (size_t)k; l++) {
#if CAMBLAS_MICRO8_PREFETCH_K > 0
                    if ((size_t)k - l > CAMBLAS_MICRO8_PREFETCH_K) {
                        size_t ahead = l + CAMBLAS_MICRO8_PREFETCH_K;
                        __builtin_prefetch(A + camblas_micro8_a_index(i, ahead, lda,
                                                                      CAMBLAS_MICRO8_ROWS * vl,
                                                                      a_micro),
                                           0, 3);
                        __builtin_prefetch(A + camblas_micro8_a_index(i + 2 * vl, ahead, lda,
                                                                      CAMBLAS_MICRO8_ROWS * vl,
                                                                      a_micro),
                                           0, 3);
                    }
#endif
                    svfloat64_t b0, b1, b2, b3;
                    camblas_micro8_load_b_f64(all, B + j * ldb + l * 8, &b0, &b1, &b2, &b3);
                    CAMBLAS_LANE_F64(c00, a0, b0, 0);
                    CAMBLAS_LANE_F64(c01, a0, b0, 1);
                    CAMBLAS_LANE_F64(c02, a0, b1, 0);
                    CAMBLAS_LANE_F64(c03, a0, b1, 1);
                    CAMBLAS_LANE_F64(c04, a0, b2, 0);
                    CAMBLAS_LANE_F64(c05, a0, b2, 1);
                    CAMBLAS_LANE_F64(c06, a0, b3, 0);
                    CAMBLAS_LANE_F64(c07, a0, b3, 1);
                    a0 = svld1_f64(all,
                                   A + camblas_micro8_a_index(i + 0 * vl, l + 1, lda,
                                                              CAMBLAS_MICRO8_ROWS * vl, a_micro));
                    CAMBLAS_LANE_F64(c10, a1, b0, 0);
                    CAMBLAS_LANE_F64(c11, a1, b0, 1);
                    CAMBLAS_LANE_F64(c12, a1, b1, 0);
                    CAMBLAS_LANE_F64(c13, a1, b1, 1);
                    CAMBLAS_LANE_F64(c14, a1, b2, 0);
                    CAMBLAS_LANE_F64(c15, a1, b2, 1);
                    CAMBLAS_LANE_F64(c16, a1, b3, 0);
                    CAMBLAS_LANE_F64(c17, a1, b3, 1);
                    a1 = svld1_f64(all,
                                   A + camblas_micro8_a_index(i + 1 * vl, l + 1, lda,
                                                              CAMBLAS_MICRO8_ROWS * vl, a_micro));
                    CAMBLAS_LANE_F64(c20, a2, b0, 0);
                    CAMBLAS_LANE_F64(c21, a2, b0, 1);
                    CAMBLAS_LANE_F64(c22, a2, b1, 0);
                    CAMBLAS_LANE_F64(c23, a2, b1, 1);
                    CAMBLAS_LANE_F64(c24, a2, b2, 0);
                    CAMBLAS_LANE_F64(c25, a2, b2, 1);
                    CAMBLAS_LANE_F64(c26, a2, b3, 0);
                    CAMBLAS_LANE_F64(c27, a2, b3, 1);
                    a2 = svld1_f64(all,
                                   A + camblas_micro8_a_index(i + 2 * vl, l + 1, lda,
                                                              CAMBLAS_MICRO8_ROWS * vl, a_micro));
#if defined(CAMBLAS_MICRO8_UNROLL2) && CAMBLAS_MICRO8_UNROLL2
                    __asm__ volatile("" ::: "memory");
#endif
                }
                {
                    const size_t l = (size_t)k - 1;
                    svfloat64_t b0, b1, b2, b3;
                    camblas_micro8_load_b_f64(all, B + j * ldb + l * 8, &b0, &b1, &b2, &b3);
                    CAMBLAS_LANE_F64(c00, a0, b0, 0);
                    CAMBLAS_LANE_F64(c01, a0, b0, 1);
                    CAMBLAS_LANE_F64(c02, a0, b1, 0);
                    CAMBLAS_LANE_F64(c03, a0, b1, 1);
                    CAMBLAS_LANE_F64(c04, a0, b2, 0);
                    CAMBLAS_LANE_F64(c05, a0, b2, 1);
                    CAMBLAS_LANE_F64(c06, a0, b3, 0);
                    CAMBLAS_LANE_F64(c07, a0, b3, 1);
                    CAMBLAS_LANE_F64(c10, a1, b0, 0);
                    CAMBLAS_LANE_F64(c11, a1, b0, 1);
                    CAMBLAS_LANE_F64(c12, a1, b1, 0);
                    CAMBLAS_LANE_F64(c13, a1, b1, 1);
                    CAMBLAS_LANE_F64(c14, a1, b2, 0);
                    CAMBLAS_LANE_F64(c15, a1, b2, 1);
                    CAMBLAS_LANE_F64(c16, a1, b3, 0);
                    CAMBLAS_LANE_F64(c17, a1, b3, 1);
                    CAMBLAS_LANE_F64(c20, a2, b0, 0);
                    CAMBLAS_LANE_F64(c21, a2, b0, 1);
                    CAMBLAS_LANE_F64(c22, a2, b1, 0);
                    CAMBLAS_LANE_F64(c23, a2, b1, 1);
                    CAMBLAS_LANE_F64(c24, a2, b2, 0);
                    CAMBLAS_LANE_F64(c25, a2, b2, 1);
                    CAMBLAS_LANE_F64(c26, a2, b3, 0);
                    CAMBLAS_LANE_F64(c27, a2, b3, 1);
                }
#else
                for (size_t l = 0; l < (size_t)k; l++) {
                    svfloat64_t b0 = svld1rq_f64(all, B + j * ldb + l * 8 + 0);
                    svfloat64_t b1 = svld1rq_f64(all, B + j * ldb + l * 8 + 2);
                    svfloat64_t b2 = svld1rq_f64(all, B + j * ldb + l * 8 + 4);
                    svfloat64_t b3 = svld1rq_f64(all, B + j * ldb + l * 8 + 6);
                    CAMBLAS_LANE_F64(c00, a0, b0, 0);
                    CAMBLAS_LANE_F64(c01, a0, b0, 1);
                    CAMBLAS_LANE_F64(c02, a0, b1, 0);
                    CAMBLAS_LANE_F64(c03, a0, b1, 1);
                    CAMBLAS_LANE_F64(c04, a0, b2, 0);
                    CAMBLAS_LANE_F64(c05, a0, b2, 1);
                    CAMBLAS_LANE_F64(c06, a0, b3, 0);
                    CAMBLAS_LANE_F64(c07, a0, b3, 1);
                    if (l + 1 < (size_t)k)
                        a0 = svld1_f64(all, A + camblas_micro8_a_index(i + 0 * vl, l + 1, lda,
                                                                       CAMBLAS_MICRO8_ROWS * vl,
                                                                       a_micro));
                    CAMBLAS_LANE_F64(c10, a1, b0, 0);
                    CAMBLAS_LANE_F64(c11, a1, b0, 1);
                    CAMBLAS_LANE_F64(c12, a1, b1, 0);
                    CAMBLAS_LANE_F64(c13, a1, b1, 1);
                    CAMBLAS_LANE_F64(c14, a1, b2, 0);
                    CAMBLAS_LANE_F64(c15, a1, b2, 1);
                    CAMBLAS_LANE_F64(c16, a1, b3, 0);
                    CAMBLAS_LANE_F64(c17, a1, b3, 1);
                    if (l + 1 < (size_t)k)
                        a1 = svld1_f64(all, A + camblas_micro8_a_index(i + 1 * vl, l + 1, lda,
                                                                       CAMBLAS_MICRO8_ROWS * vl,
                                                                       a_micro));
                    CAMBLAS_LANE_F64(c20, a2, b0, 0);
                    CAMBLAS_LANE_F64(c21, a2, b0, 1);
                    CAMBLAS_LANE_F64(c22, a2, b1, 0);
                    CAMBLAS_LANE_F64(c23, a2, b1, 1);
                    CAMBLAS_LANE_F64(c24, a2, b2, 0);
                    CAMBLAS_LANE_F64(c25, a2, b2, 1);
                    CAMBLAS_LANE_F64(c26, a2, b3, 0);
                    CAMBLAS_LANE_F64(c27, a2, b3, 1);
                    if (l + 1 < (size_t)k)
                        a2 = svld1_f64(all, A + camblas_micro8_a_index(i + 2 * vl, l + 1, lda,
                                                                       CAMBLAS_MICRO8_ROWS * vl,
                                                                       a_micro));
                }
#endif
            }
#else
#if defined(CAMBLAS_MICRO8_UNROLL2) && CAMBLAS_MICRO8_UNROLL2
#if CAMBLAS_MICRO8_UNROLL_FACTOR == 4
#pragma GCC unroll 4
#else
#pragma GCC unroll 2
#endif
#endif
            for (size_t l = 0; l < (size_t)k; l++) {
                svfloat64_t a0 =
                    svld1_f64(all, A + camblas_micro8_a_index(i + 0 * vl, l, lda,
                                                              CAMBLAS_MICRO8_ROWS * vl, a_micro));
                svfloat64_t a1 =
                    svld1_f64(all, A + camblas_micro8_a_index(i + 1 * vl, l, lda,
                                                              CAMBLAS_MICRO8_ROWS * vl, a_micro));
#if CAMBLAS_MICRO8_ROWS == 3
                svfloat64_t a2 =
                    svld1_f64(all, A + camblas_micro8_a_index(i + 2 * vl, l, lda,
                                                              CAMBLAS_MICRO8_ROWS * vl, a_micro));
#endif
                svfloat64_t b0 = svld1rq_f64(all, B + j * ldb + l * 8 + 0);
                svfloat64_t b1 = svld1rq_f64(all, B + j * ldb + l * 8 + 2);
                svfloat64_t b2 = svld1rq_f64(all, B + j * ldb + l * 8 + 4);
                svfloat64_t b3 = svld1rq_f64(all, B + j * ldb + l * 8 + 6);
#if defined(CAMBLAS_MICRO8_ALTERNATE) && CAMBLAS_MICRO8_ALTERNATE
                CAMBLAS_LANE_F64(c00, a0, b0, 0);
                CAMBLAS_LANE_F64(c10, a1, b0, 0);
#if CAMBLAS_MICRO8_ROWS == 3
                CAMBLAS_LANE_F64(c20, a2, b0, 0);
#endif
                CAMBLAS_LANE_F64(c12, a1, b1, 0);
#if CAMBLAS_MICRO8_ROWS == 3
                CAMBLAS_LANE_F64(c22, a2, b1, 0);
#endif
                CAMBLAS_LANE_F64(c02, a0, b1, 0);
#if CAMBLAS_MICRO8_ROWS == 3
                CAMBLAS_LANE_F64(c24, a2, b2, 0);
#endif
                CAMBLAS_LANE_F64(c04, a0, b2, 0);
                CAMBLAS_LANE_F64(c14, a1, b2, 0);
                CAMBLAS_LANE_F64(c06, a0, b3, 0);
                CAMBLAS_LANE_F64(c16, a1, b3, 0);
#if CAMBLAS_MICRO8_ROWS == 3
                CAMBLAS_LANE_F64(c26, a2, b3, 0);
#endif
                CAMBLAS_LANE_F64(c11, a1, b0, 1);
#if CAMBLAS_MICRO8_ROWS == 3
                CAMBLAS_LANE_F64(c21, a2, b0, 1);
#endif
                CAMBLAS_LANE_F64(c01, a0, b0, 1);
#if CAMBLAS_MICRO8_ROWS == 3
                CAMBLAS_LANE_F64(c23, a2, b1, 1);
#endif
                CAMBLAS_LANE_F64(c03, a0, b1, 1);
                CAMBLAS_LANE_F64(c13, a1, b1, 1);
                CAMBLAS_LANE_F64(c05, a0, b2, 1);
                CAMBLAS_LANE_F64(c15, a1, b2, 1);
#if CAMBLAS_MICRO8_ROWS == 3
                CAMBLAS_LANE_F64(c25, a2, b2, 1);
#endif
                CAMBLAS_LANE_F64(c17, a1, b3, 1);
#if CAMBLAS_MICRO8_ROWS == 3
                CAMBLAS_LANE_F64(c27, a2, b3, 1);
#endif
                CAMBLAS_LANE_F64(c07, a0, b3, 1);
#else
                CAMBLAS_LANE_F64(c00, a0, b0, 0);
                CAMBLAS_LANE_F64(c10, a1, b0, 0);
#if CAMBLAS_MICRO8_ROWS == 3
                CAMBLAS_LANE_F64(c20, a2, b0, 0);
#endif
                CAMBLAS_LANE_F64(c01, a0, b0, 1);
                CAMBLAS_LANE_F64(c11, a1, b0, 1);
#if CAMBLAS_MICRO8_ROWS == 3
                CAMBLAS_LANE_F64(c21, a2, b0, 1);
#endif
                CAMBLAS_LANE_F64(c02, a0, b1, 0);
                CAMBLAS_LANE_F64(c12, a1, b1, 0);
#if CAMBLAS_MICRO8_ROWS == 3
                CAMBLAS_LANE_F64(c22, a2, b1, 0);
#endif
                CAMBLAS_LANE_F64(c03, a0, b1, 1);
                CAMBLAS_LANE_F64(c13, a1, b1, 1);
#if CAMBLAS_MICRO8_ROWS == 3
                CAMBLAS_LANE_F64(c23, a2, b1, 1);
#endif
                CAMBLAS_LANE_F64(c04, a0, b2, 0);
                CAMBLAS_LANE_F64(c14, a1, b2, 0);
#if CAMBLAS_MICRO8_ROWS == 3
                CAMBLAS_LANE_F64(c24, a2, b2, 0);
#endif
                CAMBLAS_LANE_F64(c05, a0, b2, 1);
                CAMBLAS_LANE_F64(c15, a1, b2, 1);
#if CAMBLAS_MICRO8_ROWS == 3
                CAMBLAS_LANE_F64(c25, a2, b2, 1);
#endif
                CAMBLAS_LANE_F64(c06, a0, b3, 0);
                CAMBLAS_LANE_F64(c16, a1, b3, 0);
#if CAMBLAS_MICRO8_ROWS == 3
                CAMBLAS_LANE_F64(c26, a2, b3, 0);
#endif
                CAMBLAS_LANE_F64(c07, a0, b3, 1);
                CAMBLAS_LANE_F64(c17, a1, b3, 1);
#if CAMBLAS_MICRO8_ROWS == 3
                CAMBLAS_LANE_F64(c27, a2, b3, 1);
#endif
#endif
            }
#endif
            if (j + 0 < (size_t)n) {
                double *p0 = C + i + 0 * vl + (j + 0) * ldc;
                camblas_micro8_store_f64(all, p0, c00, alpha, initialize);
                double *p1 = C + i + 1 * vl + (j + 0) * ldc;
                camblas_micro8_store_f64(all, p1, c10, alpha, initialize);
#if CAMBLAS_MICRO8_ROWS == 3
                double *p2 = C + i + 2 * vl + (j + 0) * ldc;
#endif
#if CAMBLAS_MICRO8_ROWS == 3
                camblas_micro8_store_f64(all, p2, c20, alpha, initialize);
#endif
            }
            if (j + 1 < (size_t)n) {
                double *p0 = C + i + 0 * vl + (j + 1) * ldc;
                camblas_micro8_store_f64(all, p0, c01, alpha, initialize);
                double *p1 = C + i + 1 * vl + (j + 1) * ldc;
                camblas_micro8_store_f64(all, p1, c11, alpha, initialize);
#if CAMBLAS_MICRO8_ROWS == 3
                double *p2 = C + i + 2 * vl + (j + 1) * ldc;
#endif
#if CAMBLAS_MICRO8_ROWS == 3
                camblas_micro8_store_f64(all, p2, c21, alpha, initialize);
#endif
            }
            if (j + 2 < (size_t)n) {
                double *p0 = C + i + 0 * vl + (j + 2) * ldc;
                camblas_micro8_store_f64(all, p0, c02, alpha, initialize);
                double *p1 = C + i + 1 * vl + (j + 2) * ldc;
                camblas_micro8_store_f64(all, p1, c12, alpha, initialize);
#if CAMBLAS_MICRO8_ROWS == 3
                double *p2 = C + i + 2 * vl + (j + 2) * ldc;
#endif
#if CAMBLAS_MICRO8_ROWS == 3
                camblas_micro8_store_f64(all, p2, c22, alpha, initialize);
#endif
            }
            if (j + 3 < (size_t)n) {
                double *p0 = C + i + 0 * vl + (j + 3) * ldc;
                camblas_micro8_store_f64(all, p0, c03, alpha, initialize);
                double *p1 = C + i + 1 * vl + (j + 3) * ldc;
                camblas_micro8_store_f64(all, p1, c13, alpha, initialize);
#if CAMBLAS_MICRO8_ROWS == 3
                double *p2 = C + i + 2 * vl + (j + 3) * ldc;
#endif
#if CAMBLAS_MICRO8_ROWS == 3
                camblas_micro8_store_f64(all, p2, c23, alpha, initialize);
#endif
            }
            if (j + 4 < (size_t)n) {
                double *p0 = C + i + 0 * vl + (j + 4) * ldc;
                camblas_micro8_store_f64(all, p0, c04, alpha, initialize);
                double *p1 = C + i + 1 * vl + (j + 4) * ldc;
                camblas_micro8_store_f64(all, p1, c14, alpha, initialize);
#if CAMBLAS_MICRO8_ROWS == 3
                double *p2 = C + i + 2 * vl + (j + 4) * ldc;
#endif
#if CAMBLAS_MICRO8_ROWS == 3
                camblas_micro8_store_f64(all, p2, c24, alpha, initialize);
#endif
            }
            if (j + 5 < (size_t)n) {
                double *p0 = C + i + 0 * vl + (j + 5) * ldc;
                camblas_micro8_store_f64(all, p0, c05, alpha, initialize);
                double *p1 = C + i + 1 * vl + (j + 5) * ldc;
                camblas_micro8_store_f64(all, p1, c15, alpha, initialize);
#if CAMBLAS_MICRO8_ROWS == 3
                double *p2 = C + i + 2 * vl + (j + 5) * ldc;
#endif
#if CAMBLAS_MICRO8_ROWS == 3
                camblas_micro8_store_f64(all, p2, c25, alpha, initialize);
#endif
            }
            if (j + 6 < (size_t)n) {
                double *p0 = C + i + 0 * vl + (j + 6) * ldc;
                camblas_micro8_store_f64(all, p0, c06, alpha, initialize);
                double *p1 = C + i + 1 * vl + (j + 6) * ldc;
                camblas_micro8_store_f64(all, p1, c16, alpha, initialize);
#if CAMBLAS_MICRO8_ROWS == 3
                double *p2 = C + i + 2 * vl + (j + 6) * ldc;
#endif
#if CAMBLAS_MICRO8_ROWS == 3
                camblas_micro8_store_f64(all, p2, c26, alpha, initialize);
#endif
            }
            if (j + 7 < (size_t)n) {
                double *p0 = C + i + 0 * vl + (j + 7) * ldc;
                camblas_micro8_store_f64(all, p0, c07, alpha, initialize);
                double *p1 = C + i + 1 * vl + (j + 7) * ldc;
                camblas_micro8_store_f64(all, p1, c17, alpha, initialize);
#if CAMBLAS_MICRO8_ROWS == 3
                double *p2 = C + i + 2 * vl + (j + 7) * ldc;
#endif
#if CAMBLAS_MICRO8_ROWS == 3
                camblas_micro8_store_f64(all, p2, c27, alpha, initialize);
#endif
            }
        }
        for (; i < (size_t)m; i += vl) {
            svbool_t pg = svwhilelt_b64((uint64_t)i, (uint64_t)m);
#if defined(CAMBLAS_MICRO8_WIDE_TAIL) && CAMBLAS_MICRO8_WIDE_TAIL
            svfloat64_t t0 = svdup_f64(0);
            svfloat64_t t1 = svdup_f64(0);
            svfloat64_t t2 = svdup_f64(0);
            svfloat64_t t3 = svdup_f64(0);
            svfloat64_t t4 = svdup_f64(0);
            svfloat64_t t5 = svdup_f64(0);
            svfloat64_t t6 = svdup_f64(0);
            svfloat64_t t7 = svdup_f64(0);
            for (size_t l = 0; l < (size_t)k; l++) {
                svfloat64_t a = svld1_f64(
                    pg, A + camblas_micro8_a_index(i, l, lda, CAMBLAS_MICRO8_ROWS * vl, a_micro));
                svfloat64_t b0 = svld1rq_f64(all, B + j * ldb + l * 8 + 0);
                svfloat64_t b1 = svld1rq_f64(all, B + j * ldb + l * 8 + 2);
                svfloat64_t b2 = svld1rq_f64(all, B + j * ldb + l * 8 + 4);
                svfloat64_t b3 = svld1rq_f64(all, B + j * ldb + l * 8 + 6);
                CAMBLAS_LANE_F64(t0, a, b0, 0);
                CAMBLAS_LANE_F64(t1, a, b0, 1);
                CAMBLAS_LANE_F64(t2, a, b1, 0);
                CAMBLAS_LANE_F64(t3, a, b1, 1);
                CAMBLAS_LANE_F64(t4, a, b2, 0);
                CAMBLAS_LANE_F64(t5, a, b2, 1);
                CAMBLAS_LANE_F64(t6, a, b3, 0);
                CAMBLAS_LANE_F64(t7, a, b3, 1);
            }
            if (j + 0 < (size_t)n) {
                double *p = C + i + (j + 0) * ldc;
                camblas_micro8_store_f64(pg, p, t0, alpha, initialize);
            }
            if (j + 1 < (size_t)n) {
                double *p = C + i + (j + 1) * ldc;
                camblas_micro8_store_f64(pg, p, t1, alpha, initialize);
            }
            if (j + 2 < (size_t)n) {
                double *p = C + i + (j + 2) * ldc;
                camblas_micro8_store_f64(pg, p, t2, alpha, initialize);
            }
            if (j + 3 < (size_t)n) {
                double *p = C + i + (j + 3) * ldc;
                camblas_micro8_store_f64(pg, p, t3, alpha, initialize);
            }
            if (j + 4 < (size_t)n) {
                double *p = C + i + (j + 4) * ldc;
                camblas_micro8_store_f64(pg, p, t4, alpha, initialize);
            }
            if (j + 5 < (size_t)n) {
                double *p = C + i + (j + 5) * ldc;
                camblas_micro8_store_f64(pg, p, t5, alpha, initialize);
            }
            if (j + 6 < (size_t)n) {
                double *p = C + i + (j + 6) * ldc;
                camblas_micro8_store_f64(pg, p, t6, alpha, initialize);
            }
            if (j + 7 < (size_t)n) {
                double *p = C + i + (j + 7) * ldc;
                camblas_micro8_store_f64(pg, p, t7, alpha, initialize);
            }
#else
            for (size_t q = 0; q < 8 && j + q < (size_t)n; q++) {
                svfloat64_t acc = svdup_f64(0);
                for (size_t l = 0; l < (size_t)k; l++)
                    acc = svmla_n_f64_x(
                        pg, acc,
                        svld1_f64(pg, A + camblas_micro8_a_index(
                                              i, l, lda, CAMBLAS_MICRO8_ROWS * vl, a_micro)),
                        B[j * ldb + l * 8 + q]);
                double *p = C + i + (j + q) * ldc;
                camblas_micro8_store_f64(pg, p, acc, alpha, initialize);
            }
#endif
        }
    }
    return 0;
}
static inline int camblas_micro8_f32(int m, int n, int k, float alpha, const float *A, size_t lda,
                                     const float *B, size_t ldb, float *C, size_t ldc)
{
    return camblas_micro8_layout_f32(m, n, k, alpha, A, lda, B, ldb, C, ldc, 0, 0);
}
static inline int camblas_micro8_amicro_f32(int m, int n, int k, float alpha, const float *A,
                                            size_t lda, const float *B, size_t ldb, float *C,
                                            size_t ldc)
{
    return camblas_micro8_layout_f32(m, n, k, alpha, A, lda, B, ldb, C, ldc, 1, 0);
}
static inline int camblas_micro8_f64(int m, int n, int k, double alpha, const double *A, size_t lda,
                                     const double *B, size_t ldb, double *C, size_t ldc)
{
    return camblas_micro8_layout_f64(m, n, k, alpha, A, lda, B, ldb, C, ldc, 0, 0);
}
static inline int camblas_micro8_amicro_f64(int m, int n, int k, double alpha, const double *A,
                                            size_t lda, const double *B, size_t ldb, double *C,
                                            size_t ldc)
{
    return camblas_micro8_layout_f64(m, n, k, alpha, A, lda, B, ldb, C, ldc, 1, 0);
}
static inline int camblas_micro8_init_f32(int m, int n, int k, float alpha, const float *A,
                                          size_t lda, const float *B, size_t ldb, float *C,
                                          size_t ldc)
{
    return camblas_micro8_layout_f32(m, n, k, alpha, A, lda, B, ldb, C, ldc, 0, 1);
}
static inline int camblas_micro8_init_f64(int m, int n, int k, double alpha, const double *A,
                                          size_t lda, const double *B, size_t ldb, double *C,
                                          size_t ldc)
{
    return camblas_micro8_layout_f64(m, n, k, alpha, A, lda, B, ldb, C, ldc, 0, 1);
}
#endif
