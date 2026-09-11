#ifndef CAMBLAS_SVE_PIPE_B_H
#define CAMBLAS_SVE_PIPE_B_H
#include <arm_sve.h>
#include <stddef.h>
#include "sve_lane_ops.h"
/* Experimental FP32 VL128 full 12x8 block. Double-buffer B inside the bounded K loop:
 * 24 accumulators + 3 carried A vectors + 4 B vectors leave one Z register.
 * No source for K+1 is loaded on the final step. Output order is unchanged. */
static inline void camblas_pipe_b_f32_block(int k, const float *A, size_t lda, const float *B,
                                            float *C, size_t ldc, float alpha, int initialize)
{
    svbool_t pg = svptrue_b32();
#define PB_DECLARE(col) \
    svfloat32_t c0##col = svdup_f32(0), c1##col = svdup_f32(0), c2##col = svdup_f32(0)
    PB_DECLARE(0);
    PB_DECLARE(1);
    PB_DECLARE(2);
    PB_DECLARE(3);
    PB_DECLARE(4);
    PB_DECLARE(5);
    PB_DECLARE(6);
    PB_DECLARE(7);
#undef PB_DECLARE
#define PB_ROW(row, av, bv0, bv1)            \
    CAMBLAS_LANE_F32(c##row##0, av, bv0, 0); \
    CAMBLAS_LANE_F32(c##row##1, av, bv0, 1); \
    CAMBLAS_LANE_F32(c##row##2, av, bv0, 2); \
    CAMBLAS_LANE_F32(c##row##3, av, bv0, 3); \
    CAMBLAS_LANE_F32(c##row##4, av, bv1, 0); \
    CAMBLAS_LANE_F32(c##row##5, av, bv1, 1); \
    CAMBLAS_LANE_F32(c##row##6, av, bv1, 2); \
    CAMBLAS_LANE_F32(c##row##7, av, bv1, 3)
    if (k > 0) {
        svfloat32_t a0 = svld1_f32(pg, A), a1 = svld1_f32(pg, A + 4), a2 = svld1_f32(pg, A + 8);
        svfloat32_t b0 = svld1rq_f32(pg, B), b1 = svld1rq_f32(pg, B + 4);
        size_t l = 0;
        /* Explicit alternation avoids register-to-register B swaps in the
         * steady-state pair. Both prefetched depths are proven in bounds. */
#pragma GCC unroll 1
        for (; l + 2 < (size_t)k; l += 2) {
            svfloat32_t next0 = svld1rq_f32(pg, B + (l + 1) * 8);
            svfloat32_t next1 = svld1rq_f32(pg, B + (l + 1) * 8 + 4);
            PB_ROW(0, a0, b0, b1);
            a0 = svld1_f32(pg, A + (l + 1) * lda);
            PB_ROW(1, a1, b0, b1);
            a1 = svld1_f32(pg, A + (l + 1) * lda + 4);
            PB_ROW(2, a2, b0, b1);
            a2 = svld1_f32(pg, A + (l + 1) * lda + 8);
            b0 = svld1rq_f32(pg, B + (l + 2) * 8);
            b1 = svld1rq_f32(pg, B + (l + 2) * 8 + 4);
            PB_ROW(0, a0, next0, next1);
            a0 = svld1_f32(pg, A + (l + 2) * lda);
            PB_ROW(1, a1, next0, next1);
            a1 = svld1_f32(pg, A + (l + 2) * lda + 4);
            PB_ROW(2, a2, next0, next1);
            a2 = svld1_f32(pg, A + (l + 2) * lda + 8);
            __asm__ volatile("" ::: "memory");
        }
        if (l + 1 < (size_t)k) {
            svfloat32_t next0 = svld1rq_f32(pg, B + (l + 1) * 8),
                        next1 = svld1rq_f32(pg, B + (l + 1) * 8 + 4);
            PB_ROW(0, a0, b0, b1);
            a0 = svld1_f32(pg, A + (l + 1) * lda);
            PB_ROW(1, a1, b0, b1);
            a1 = svld1_f32(pg, A + (l + 1) * lda + 4);
            PB_ROW(2, a2, b0, b1);
            a2 = svld1_f32(pg, A + (l + 1) * lda + 8);
            b0 = next0;
            b1 = next1;
        }
        PB_ROW(0, a0, b0, b1);
        PB_ROW(1, a1, b0, b1);
        PB_ROW(2, a2, b0, b1);
    }
#undef PB_ROW
#define PB_STORE(row, col)                                         \
    do {                                                           \
        float *out = C + (size_t)(col) * ldc + (row) * 4;          \
        svfloat32_t value = svmul_n_f32_x(pg, c##row##col, alpha); \
        if (!initialize)                                           \
            value = svadd_f32_x(pg, svld1_f32(pg, out), value);    \
        svst1_f32(pg, out, value);                                 \
    } while (0)
#define PB_STORE_COLUMN(col) \
    PB_STORE(0, col);        \
    PB_STORE(1, col);        \
    PB_STORE(2, col)
    PB_STORE_COLUMN(0);
    PB_STORE_COLUMN(1);
    PB_STORE_COLUMN(2);
    PB_STORE_COLUMN(3);
    PB_STORE_COLUMN(4);
    PB_STORE_COLUMN(5);
    PB_STORE_COLUMN(6);
    PB_STORE_COLUMN(7);
#undef PB_STORE_COLUMN
#undef PB_STORE
}
#endif
