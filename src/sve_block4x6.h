/* Explicit 4-vector x 6-column full tile; 24 independent K-order accumulators. */
#ifndef CAMBLAS_SVE_BLOCK4X6_H
#define CAMBLAS_SVE_BLOCK4X6_H
#include "sve_lane_ops.h"
static inline __attribute__((always_inline)) void camblas_block4x6_f32(int k, float alpha,
                                                                       const float *A, size_t lda,
                                                                       const float *B, size_t ldb,
                                                                       float *C, size_t ldc)
{
    size_t lanes = svcntw();
    svbool_t pg = svptrue_b32();
    svfloat32_t acc00 = svdup_f32(0.0);
    svfloat32_t acc01 = svdup_f32(0.0);
    svfloat32_t acc02 = svdup_f32(0.0);
    svfloat32_t acc03 = svdup_f32(0.0);
    svfloat32_t acc04 = svdup_f32(0.0);
    svfloat32_t acc05 = svdup_f32(0.0);
    svfloat32_t acc10 = svdup_f32(0.0);
    svfloat32_t acc11 = svdup_f32(0.0);
    svfloat32_t acc12 = svdup_f32(0.0);
    svfloat32_t acc13 = svdup_f32(0.0);
    svfloat32_t acc14 = svdup_f32(0.0);
    svfloat32_t acc15 = svdup_f32(0.0);
    svfloat32_t acc20 = svdup_f32(0.0);
    svfloat32_t acc21 = svdup_f32(0.0);
    svfloat32_t acc22 = svdup_f32(0.0);
    svfloat32_t acc23 = svdup_f32(0.0);
    svfloat32_t acc24 = svdup_f32(0.0);
    svfloat32_t acc25 = svdup_f32(0.0);
    svfloat32_t acc30 = svdup_f32(0.0);
    svfloat32_t acc31 = svdup_f32(0.0);
    svfloat32_t acc32 = svdup_f32(0.0);
    svfloat32_t acc33 = svdup_f32(0.0);
    svfloat32_t acc34 = svdup_f32(0.0);
    svfloat32_t acc35 = svdup_f32(0.0);
    int l = 0;
#if CAMBLAS_SVE_LANE_FMA
    /* Load one 128-bit K group per B column, then use indexed FMAs.
       Each accumulator keeps the original K order; the scalar remainder
       below never loads beyond the final logical K element. */
    for (; l <= k - 4; l += 4) {
        svfloat32_t b0 = svld1rq_f32(pg, B + (size_t)l + 0 * ldb);
        svfloat32_t b1 = svld1rq_f32(pg, B + (size_t)l + 1 * ldb);
        svfloat32_t b2 = svld1rq_f32(pg, B + (size_t)l + 2 * ldb);
        svfloat32_t b3 = svld1rq_f32(pg, B + (size_t)l + 3 * ldb);
        svfloat32_t b4 = svld1rq_f32(pg, B + (size_t)l + 4 * ldb);
        svfloat32_t b5 = svld1rq_f32(pg, B + (size_t)l + 5 * ldb);
        {
            svfloat32_t a0 = svld1_f32(pg, A + ((size_t)l + 0) * lda + 0 * lanes);
            CAMBLAS_LANE_F32(acc00, a0, b0, 0);
            CAMBLAS_LANE_F32(acc01, a0, b1, 0);
            CAMBLAS_LANE_F32(acc02, a0, b2, 0);
            CAMBLAS_LANE_F32(acc03, a0, b3, 0);
            CAMBLAS_LANE_F32(acc04, a0, b4, 0);
            CAMBLAS_LANE_F32(acc05, a0, b5, 0);
            svfloat32_t a1 = svld1_f32(pg, A + ((size_t)l + 0) * lda + 1 * lanes);
            CAMBLAS_LANE_F32(acc10, a1, b0, 0);
            CAMBLAS_LANE_F32(acc11, a1, b1, 0);
            CAMBLAS_LANE_F32(acc12, a1, b2, 0);
            CAMBLAS_LANE_F32(acc13, a1, b3, 0);
            CAMBLAS_LANE_F32(acc14, a1, b4, 0);
            CAMBLAS_LANE_F32(acc15, a1, b5, 0);
            svfloat32_t a2 = svld1_f32(pg, A + ((size_t)l + 0) * lda + 2 * lanes);
            CAMBLAS_LANE_F32(acc20, a2, b0, 0);
            CAMBLAS_LANE_F32(acc21, a2, b1, 0);
            CAMBLAS_LANE_F32(acc22, a2, b2, 0);
            CAMBLAS_LANE_F32(acc23, a2, b3, 0);
            CAMBLAS_LANE_F32(acc24, a2, b4, 0);
            CAMBLAS_LANE_F32(acc25, a2, b5, 0);
            svfloat32_t a3 = svld1_f32(pg, A + ((size_t)l + 0) * lda + 3 * lanes);
            CAMBLAS_LANE_F32(acc30, a3, b0, 0);
            CAMBLAS_LANE_F32(acc31, a3, b1, 0);
            CAMBLAS_LANE_F32(acc32, a3, b2, 0);
            CAMBLAS_LANE_F32(acc33, a3, b3, 0);
            CAMBLAS_LANE_F32(acc34, a3, b4, 0);
            CAMBLAS_LANE_F32(acc35, a3, b5, 0);
        }
        {
            svfloat32_t a0 = svld1_f32(pg, A + ((size_t)l + 1) * lda + 0 * lanes);
            CAMBLAS_LANE_F32(acc00, a0, b0, 1);
            CAMBLAS_LANE_F32(acc01, a0, b1, 1);
            CAMBLAS_LANE_F32(acc02, a0, b2, 1);
            CAMBLAS_LANE_F32(acc03, a0, b3, 1);
            CAMBLAS_LANE_F32(acc04, a0, b4, 1);
            CAMBLAS_LANE_F32(acc05, a0, b5, 1);
            svfloat32_t a1 = svld1_f32(pg, A + ((size_t)l + 1) * lda + 1 * lanes);
            CAMBLAS_LANE_F32(acc10, a1, b0, 1);
            CAMBLAS_LANE_F32(acc11, a1, b1, 1);
            CAMBLAS_LANE_F32(acc12, a1, b2, 1);
            CAMBLAS_LANE_F32(acc13, a1, b3, 1);
            CAMBLAS_LANE_F32(acc14, a1, b4, 1);
            CAMBLAS_LANE_F32(acc15, a1, b5, 1);
            svfloat32_t a2 = svld1_f32(pg, A + ((size_t)l + 1) * lda + 2 * lanes);
            CAMBLAS_LANE_F32(acc20, a2, b0, 1);
            CAMBLAS_LANE_F32(acc21, a2, b1, 1);
            CAMBLAS_LANE_F32(acc22, a2, b2, 1);
            CAMBLAS_LANE_F32(acc23, a2, b3, 1);
            CAMBLAS_LANE_F32(acc24, a2, b4, 1);
            CAMBLAS_LANE_F32(acc25, a2, b5, 1);
            svfloat32_t a3 = svld1_f32(pg, A + ((size_t)l + 1) * lda + 3 * lanes);
            CAMBLAS_LANE_F32(acc30, a3, b0, 1);
            CAMBLAS_LANE_F32(acc31, a3, b1, 1);
            CAMBLAS_LANE_F32(acc32, a3, b2, 1);
            CAMBLAS_LANE_F32(acc33, a3, b3, 1);
            CAMBLAS_LANE_F32(acc34, a3, b4, 1);
            CAMBLAS_LANE_F32(acc35, a3, b5, 1);
        }
        {
            svfloat32_t a0 = svld1_f32(pg, A + ((size_t)l + 2) * lda + 0 * lanes);
            CAMBLAS_LANE_F32(acc00, a0, b0, 2);
            CAMBLAS_LANE_F32(acc01, a0, b1, 2);
            CAMBLAS_LANE_F32(acc02, a0, b2, 2);
            CAMBLAS_LANE_F32(acc03, a0, b3, 2);
            CAMBLAS_LANE_F32(acc04, a0, b4, 2);
            CAMBLAS_LANE_F32(acc05, a0, b5, 2);
            svfloat32_t a1 = svld1_f32(pg, A + ((size_t)l + 2) * lda + 1 * lanes);
            CAMBLAS_LANE_F32(acc10, a1, b0, 2);
            CAMBLAS_LANE_F32(acc11, a1, b1, 2);
            CAMBLAS_LANE_F32(acc12, a1, b2, 2);
            CAMBLAS_LANE_F32(acc13, a1, b3, 2);
            CAMBLAS_LANE_F32(acc14, a1, b4, 2);
            CAMBLAS_LANE_F32(acc15, a1, b5, 2);
            svfloat32_t a2 = svld1_f32(pg, A + ((size_t)l + 2) * lda + 2 * lanes);
            CAMBLAS_LANE_F32(acc20, a2, b0, 2);
            CAMBLAS_LANE_F32(acc21, a2, b1, 2);
            CAMBLAS_LANE_F32(acc22, a2, b2, 2);
            CAMBLAS_LANE_F32(acc23, a2, b3, 2);
            CAMBLAS_LANE_F32(acc24, a2, b4, 2);
            CAMBLAS_LANE_F32(acc25, a2, b5, 2);
            svfloat32_t a3 = svld1_f32(pg, A + ((size_t)l + 2) * lda + 3 * lanes);
            CAMBLAS_LANE_F32(acc30, a3, b0, 2);
            CAMBLAS_LANE_F32(acc31, a3, b1, 2);
            CAMBLAS_LANE_F32(acc32, a3, b2, 2);
            CAMBLAS_LANE_F32(acc33, a3, b3, 2);
            CAMBLAS_LANE_F32(acc34, a3, b4, 2);
            CAMBLAS_LANE_F32(acc35, a3, b5, 2);
        }
        {
            svfloat32_t a0 = svld1_f32(pg, A + ((size_t)l + 3) * lda + 0 * lanes);
            CAMBLAS_LANE_F32(acc00, a0, b0, 3);
            CAMBLAS_LANE_F32(acc01, a0, b1, 3);
            CAMBLAS_LANE_F32(acc02, a0, b2, 3);
            CAMBLAS_LANE_F32(acc03, a0, b3, 3);
            CAMBLAS_LANE_F32(acc04, a0, b4, 3);
            CAMBLAS_LANE_F32(acc05, a0, b5, 3);
            svfloat32_t a1 = svld1_f32(pg, A + ((size_t)l + 3) * lda + 1 * lanes);
            CAMBLAS_LANE_F32(acc10, a1, b0, 3);
            CAMBLAS_LANE_F32(acc11, a1, b1, 3);
            CAMBLAS_LANE_F32(acc12, a1, b2, 3);
            CAMBLAS_LANE_F32(acc13, a1, b3, 3);
            CAMBLAS_LANE_F32(acc14, a1, b4, 3);
            CAMBLAS_LANE_F32(acc15, a1, b5, 3);
            svfloat32_t a2 = svld1_f32(pg, A + ((size_t)l + 3) * lda + 2 * lanes);
            CAMBLAS_LANE_F32(acc20, a2, b0, 3);
            CAMBLAS_LANE_F32(acc21, a2, b1, 3);
            CAMBLAS_LANE_F32(acc22, a2, b2, 3);
            CAMBLAS_LANE_F32(acc23, a2, b3, 3);
            CAMBLAS_LANE_F32(acc24, a2, b4, 3);
            CAMBLAS_LANE_F32(acc25, a2, b5, 3);
            svfloat32_t a3 = svld1_f32(pg, A + ((size_t)l + 3) * lda + 3 * lanes);
            CAMBLAS_LANE_F32(acc30, a3, b0, 3);
            CAMBLAS_LANE_F32(acc31, a3, b1, 3);
            CAMBLAS_LANE_F32(acc32, a3, b2, 3);
            CAMBLAS_LANE_F32(acc33, a3, b3, 3);
            CAMBLAS_LANE_F32(acc34, a3, b4, 3);
            CAMBLAS_LANE_F32(acc35, a3, b5, 3);
        }
    }
#endif
    for (; l < k; ++l) {
        svfloat32_t b0 = camblas_sve_load_replicate_f32(pg, B + (size_t)l + 0 * ldb);
        svfloat32_t b1 = camblas_sve_load_replicate_f32(pg, B + (size_t)l + 1 * ldb);
        svfloat32_t b2 = camblas_sve_load_replicate_f32(pg, B + (size_t)l + 2 * ldb);
        svfloat32_t b3 = camblas_sve_load_replicate_f32(pg, B + (size_t)l + 3 * ldb);
        svfloat32_t b4 = camblas_sve_load_replicate_f32(pg, B + (size_t)l + 4 * ldb);
        svfloat32_t b5 = camblas_sve_load_replicate_f32(pg, B + (size_t)l + 5 * ldb);
        svfloat32_t a0 = svld1_f32(pg, A + (size_t)l * lda + 0 * lanes);
        acc00 = svmla_f32_x(pg, acc00, a0, b0);
        acc01 = svmla_f32_x(pg, acc01, a0, b1);
        acc02 = svmla_f32_x(pg, acc02, a0, b2);
        acc03 = svmla_f32_x(pg, acc03, a0, b3);
        acc04 = svmla_f32_x(pg, acc04, a0, b4);
        acc05 = svmla_f32_x(pg, acc05, a0, b5);
        svfloat32_t a1 = svld1_f32(pg, A + (size_t)l * lda + 1 * lanes);
        acc10 = svmla_f32_x(pg, acc10, a1, b0);
        acc11 = svmla_f32_x(pg, acc11, a1, b1);
        acc12 = svmla_f32_x(pg, acc12, a1, b2);
        acc13 = svmla_f32_x(pg, acc13, a1, b3);
        acc14 = svmla_f32_x(pg, acc14, a1, b4);
        acc15 = svmla_f32_x(pg, acc15, a1, b5);
        svfloat32_t a2 = svld1_f32(pg, A + (size_t)l * lda + 2 * lanes);
        acc20 = svmla_f32_x(pg, acc20, a2, b0);
        acc21 = svmla_f32_x(pg, acc21, a2, b1);
        acc22 = svmla_f32_x(pg, acc22, a2, b2);
        acc23 = svmla_f32_x(pg, acc23, a2, b3);
        acc24 = svmla_f32_x(pg, acc24, a2, b4);
        acc25 = svmla_f32_x(pg, acc25, a2, b5);
        svfloat32_t a3 = svld1_f32(pg, A + (size_t)l * lda + 3 * lanes);
        acc30 = svmla_f32_x(pg, acc30, a3, b0);
        acc31 = svmla_f32_x(pg, acc31, a3, b1);
        acc32 = svmla_f32_x(pg, acc32, a3, b2);
        acc33 = svmla_f32_x(pg, acc33, a3, b3);
        acc34 = svmla_f32_x(pg, acc34, a3, b4);
        acc35 = svmla_f32_x(pg, acc35, a3, b5);
    }
    {
        float *out = C + 0 * lanes + 0 * ldc;
        svfloat32_t value = acc00;
        if (alpha != 1.0)
            value = svmul_n_f32_x(pg, value, alpha);
        svst1_f32(pg, out, svadd_f32_x(pg, svld1_f32(pg, out), value));
    }
    {
        float *out = C + 0 * lanes + 1 * ldc;
        svfloat32_t value = acc01;
        if (alpha != 1.0)
            value = svmul_n_f32_x(pg, value, alpha);
        svst1_f32(pg, out, svadd_f32_x(pg, svld1_f32(pg, out), value));
    }
    {
        float *out = C + 0 * lanes + 2 * ldc;
        svfloat32_t value = acc02;
        if (alpha != 1.0)
            value = svmul_n_f32_x(pg, value, alpha);
        svst1_f32(pg, out, svadd_f32_x(pg, svld1_f32(pg, out), value));
    }
    {
        float *out = C + 0 * lanes + 3 * ldc;
        svfloat32_t value = acc03;
        if (alpha != 1.0)
            value = svmul_n_f32_x(pg, value, alpha);
        svst1_f32(pg, out, svadd_f32_x(pg, svld1_f32(pg, out), value));
    }
    {
        float *out = C + 0 * lanes + 4 * ldc;
        svfloat32_t value = acc04;
        if (alpha != 1.0)
            value = svmul_n_f32_x(pg, value, alpha);
        svst1_f32(pg, out, svadd_f32_x(pg, svld1_f32(pg, out), value));
    }
    {
        float *out = C + 0 * lanes + 5 * ldc;
        svfloat32_t value = acc05;
        if (alpha != 1.0)
            value = svmul_n_f32_x(pg, value, alpha);
        svst1_f32(pg, out, svadd_f32_x(pg, svld1_f32(pg, out), value));
    }
    {
        float *out = C + 1 * lanes + 0 * ldc;
        svfloat32_t value = acc10;
        if (alpha != 1.0)
            value = svmul_n_f32_x(pg, value, alpha);
        svst1_f32(pg, out, svadd_f32_x(pg, svld1_f32(pg, out), value));
    }
    {
        float *out = C + 1 * lanes + 1 * ldc;
        svfloat32_t value = acc11;
        if (alpha != 1.0)
            value = svmul_n_f32_x(pg, value, alpha);
        svst1_f32(pg, out, svadd_f32_x(pg, svld1_f32(pg, out), value));
    }
    {
        float *out = C + 1 * lanes + 2 * ldc;
        svfloat32_t value = acc12;
        if (alpha != 1.0)
            value = svmul_n_f32_x(pg, value, alpha);
        svst1_f32(pg, out, svadd_f32_x(pg, svld1_f32(pg, out), value));
    }
    {
        float *out = C + 1 * lanes + 3 * ldc;
        svfloat32_t value = acc13;
        if (alpha != 1.0)
            value = svmul_n_f32_x(pg, value, alpha);
        svst1_f32(pg, out, svadd_f32_x(pg, svld1_f32(pg, out), value));
    }
    {
        float *out = C + 1 * lanes + 4 * ldc;
        svfloat32_t value = acc14;
        if (alpha != 1.0)
            value = svmul_n_f32_x(pg, value, alpha);
        svst1_f32(pg, out, svadd_f32_x(pg, svld1_f32(pg, out), value));
    }
    {
        float *out = C + 1 * lanes + 5 * ldc;
        svfloat32_t value = acc15;
        if (alpha != 1.0)
            value = svmul_n_f32_x(pg, value, alpha);
        svst1_f32(pg, out, svadd_f32_x(pg, svld1_f32(pg, out), value));
    }
    {
        float *out = C + 2 * lanes + 0 * ldc;
        svfloat32_t value = acc20;
        if (alpha != 1.0)
            value = svmul_n_f32_x(pg, value, alpha);
        svst1_f32(pg, out, svadd_f32_x(pg, svld1_f32(pg, out), value));
    }
    {
        float *out = C + 2 * lanes + 1 * ldc;
        svfloat32_t value = acc21;
        if (alpha != 1.0)
            value = svmul_n_f32_x(pg, value, alpha);
        svst1_f32(pg, out, svadd_f32_x(pg, svld1_f32(pg, out), value));
    }
    {
        float *out = C + 2 * lanes + 2 * ldc;
        svfloat32_t value = acc22;
        if (alpha != 1.0)
            value = svmul_n_f32_x(pg, value, alpha);
        svst1_f32(pg, out, svadd_f32_x(pg, svld1_f32(pg, out), value));
    }
    {
        float *out = C + 2 * lanes + 3 * ldc;
        svfloat32_t value = acc23;
        if (alpha != 1.0)
            value = svmul_n_f32_x(pg, value, alpha);
        svst1_f32(pg, out, svadd_f32_x(pg, svld1_f32(pg, out), value));
    }
    {
        float *out = C + 2 * lanes + 4 * ldc;
        svfloat32_t value = acc24;
        if (alpha != 1.0)
            value = svmul_n_f32_x(pg, value, alpha);
        svst1_f32(pg, out, svadd_f32_x(pg, svld1_f32(pg, out), value));
    }
    {
        float *out = C + 2 * lanes + 5 * ldc;
        svfloat32_t value = acc25;
        if (alpha != 1.0)
            value = svmul_n_f32_x(pg, value, alpha);
        svst1_f32(pg, out, svadd_f32_x(pg, svld1_f32(pg, out), value));
    }
    {
        float *out = C + 3 * lanes + 0 * ldc;
        svfloat32_t value = acc30;
        if (alpha != 1.0)
            value = svmul_n_f32_x(pg, value, alpha);
        svst1_f32(pg, out, svadd_f32_x(pg, svld1_f32(pg, out), value));
    }
    {
        float *out = C + 3 * lanes + 1 * ldc;
        svfloat32_t value = acc31;
        if (alpha != 1.0)
            value = svmul_n_f32_x(pg, value, alpha);
        svst1_f32(pg, out, svadd_f32_x(pg, svld1_f32(pg, out), value));
    }
    {
        float *out = C + 3 * lanes + 2 * ldc;
        svfloat32_t value = acc32;
        if (alpha != 1.0)
            value = svmul_n_f32_x(pg, value, alpha);
        svst1_f32(pg, out, svadd_f32_x(pg, svld1_f32(pg, out), value));
    }
    {
        float *out = C + 3 * lanes + 3 * ldc;
        svfloat32_t value = acc33;
        if (alpha != 1.0)
            value = svmul_n_f32_x(pg, value, alpha);
        svst1_f32(pg, out, svadd_f32_x(pg, svld1_f32(pg, out), value));
    }
    {
        float *out = C + 3 * lanes + 4 * ldc;
        svfloat32_t value = acc34;
        if (alpha != 1.0)
            value = svmul_n_f32_x(pg, value, alpha);
        svst1_f32(pg, out, svadd_f32_x(pg, svld1_f32(pg, out), value));
    }
    {
        float *out = C + 3 * lanes + 5 * ldc;
        svfloat32_t value = acc35;
        if (alpha != 1.0)
            value = svmul_n_f32_x(pg, value, alpha);
        svst1_f32(pg, out, svadd_f32_x(pg, svld1_f32(pg, out), value));
    }
}
static inline __attribute__((always_inline)) void camblas_block4x6_f64(int k, double alpha,
                                                                       const double *A, size_t lda,
                                                                       const double *B, size_t ldb,
                                                                       double *C, size_t ldc)
{
    size_t lanes = svcntd();
    svbool_t pg = svptrue_b64();
    svfloat64_t acc00 = svdup_f64(0.0);
    svfloat64_t acc01 = svdup_f64(0.0);
    svfloat64_t acc02 = svdup_f64(0.0);
    svfloat64_t acc03 = svdup_f64(0.0);
    svfloat64_t acc04 = svdup_f64(0.0);
    svfloat64_t acc05 = svdup_f64(0.0);
    svfloat64_t acc10 = svdup_f64(0.0);
    svfloat64_t acc11 = svdup_f64(0.0);
    svfloat64_t acc12 = svdup_f64(0.0);
    svfloat64_t acc13 = svdup_f64(0.0);
    svfloat64_t acc14 = svdup_f64(0.0);
    svfloat64_t acc15 = svdup_f64(0.0);
    svfloat64_t acc20 = svdup_f64(0.0);
    svfloat64_t acc21 = svdup_f64(0.0);
    svfloat64_t acc22 = svdup_f64(0.0);
    svfloat64_t acc23 = svdup_f64(0.0);
    svfloat64_t acc24 = svdup_f64(0.0);
    svfloat64_t acc25 = svdup_f64(0.0);
    svfloat64_t acc30 = svdup_f64(0.0);
    svfloat64_t acc31 = svdup_f64(0.0);
    svfloat64_t acc32 = svdup_f64(0.0);
    svfloat64_t acc33 = svdup_f64(0.0);
    svfloat64_t acc34 = svdup_f64(0.0);
    svfloat64_t acc35 = svdup_f64(0.0);
    int l = 0;
#if CAMBLAS_SVE_LANE_FMA
    /* Load one 128-bit K group per B column, then use indexed FMAs.
       Each accumulator keeps the original K order; the scalar remainder
       below never loads beyond the final logical K element. */
    for (; l <= k - 2; l += 2) {
        svfloat64_t b0 = svld1rq_f64(pg, B + (size_t)l + 0 * ldb);
        svfloat64_t b1 = svld1rq_f64(pg, B + (size_t)l + 1 * ldb);
        svfloat64_t b2 = svld1rq_f64(pg, B + (size_t)l + 2 * ldb);
        svfloat64_t b3 = svld1rq_f64(pg, B + (size_t)l + 3 * ldb);
        svfloat64_t b4 = svld1rq_f64(pg, B + (size_t)l + 4 * ldb);
        svfloat64_t b5 = svld1rq_f64(pg, B + (size_t)l + 5 * ldb);
        {
            svfloat64_t a0 = svld1_f64(pg, A + ((size_t)l + 0) * lda + 0 * lanes);
            CAMBLAS_LANE_F64(acc00, a0, b0, 0);
            CAMBLAS_LANE_F64(acc01, a0, b1, 0);
            CAMBLAS_LANE_F64(acc02, a0, b2, 0);
            CAMBLAS_LANE_F64(acc03, a0, b3, 0);
            CAMBLAS_LANE_F64(acc04, a0, b4, 0);
            CAMBLAS_LANE_F64(acc05, a0, b5, 0);
            svfloat64_t a1 = svld1_f64(pg, A + ((size_t)l + 0) * lda + 1 * lanes);
            CAMBLAS_LANE_F64(acc10, a1, b0, 0);
            CAMBLAS_LANE_F64(acc11, a1, b1, 0);
            CAMBLAS_LANE_F64(acc12, a1, b2, 0);
            CAMBLAS_LANE_F64(acc13, a1, b3, 0);
            CAMBLAS_LANE_F64(acc14, a1, b4, 0);
            CAMBLAS_LANE_F64(acc15, a1, b5, 0);
            svfloat64_t a2 = svld1_f64(pg, A + ((size_t)l + 0) * lda + 2 * lanes);
            CAMBLAS_LANE_F64(acc20, a2, b0, 0);
            CAMBLAS_LANE_F64(acc21, a2, b1, 0);
            CAMBLAS_LANE_F64(acc22, a2, b2, 0);
            CAMBLAS_LANE_F64(acc23, a2, b3, 0);
            CAMBLAS_LANE_F64(acc24, a2, b4, 0);
            CAMBLAS_LANE_F64(acc25, a2, b5, 0);
            svfloat64_t a3 = svld1_f64(pg, A + ((size_t)l + 0) * lda + 3 * lanes);
            CAMBLAS_LANE_F64(acc30, a3, b0, 0);
            CAMBLAS_LANE_F64(acc31, a3, b1, 0);
            CAMBLAS_LANE_F64(acc32, a3, b2, 0);
            CAMBLAS_LANE_F64(acc33, a3, b3, 0);
            CAMBLAS_LANE_F64(acc34, a3, b4, 0);
            CAMBLAS_LANE_F64(acc35, a3, b5, 0);
        }
        {
            svfloat64_t a0 = svld1_f64(pg, A + ((size_t)l + 1) * lda + 0 * lanes);
            CAMBLAS_LANE_F64(acc00, a0, b0, 1);
            CAMBLAS_LANE_F64(acc01, a0, b1, 1);
            CAMBLAS_LANE_F64(acc02, a0, b2, 1);
            CAMBLAS_LANE_F64(acc03, a0, b3, 1);
            CAMBLAS_LANE_F64(acc04, a0, b4, 1);
            CAMBLAS_LANE_F64(acc05, a0, b5, 1);
            svfloat64_t a1 = svld1_f64(pg, A + ((size_t)l + 1) * lda + 1 * lanes);
            CAMBLAS_LANE_F64(acc10, a1, b0, 1);
            CAMBLAS_LANE_F64(acc11, a1, b1, 1);
            CAMBLAS_LANE_F64(acc12, a1, b2, 1);
            CAMBLAS_LANE_F64(acc13, a1, b3, 1);
            CAMBLAS_LANE_F64(acc14, a1, b4, 1);
            CAMBLAS_LANE_F64(acc15, a1, b5, 1);
            svfloat64_t a2 = svld1_f64(pg, A + ((size_t)l + 1) * lda + 2 * lanes);
            CAMBLAS_LANE_F64(acc20, a2, b0, 1);
            CAMBLAS_LANE_F64(acc21, a2, b1, 1);
            CAMBLAS_LANE_F64(acc22, a2, b2, 1);
            CAMBLAS_LANE_F64(acc23, a2, b3, 1);
            CAMBLAS_LANE_F64(acc24, a2, b4, 1);
            CAMBLAS_LANE_F64(acc25, a2, b5, 1);
            svfloat64_t a3 = svld1_f64(pg, A + ((size_t)l + 1) * lda + 3 * lanes);
            CAMBLAS_LANE_F64(acc30, a3, b0, 1);
            CAMBLAS_LANE_F64(acc31, a3, b1, 1);
            CAMBLAS_LANE_F64(acc32, a3, b2, 1);
            CAMBLAS_LANE_F64(acc33, a3, b3, 1);
            CAMBLAS_LANE_F64(acc34, a3, b4, 1);
            CAMBLAS_LANE_F64(acc35, a3, b5, 1);
        }
    }
#endif
    for (; l < k; ++l) {
        svfloat64_t b0 = camblas_sve_load_replicate_f64(pg, B + (size_t)l + 0 * ldb);
        svfloat64_t b1 = camblas_sve_load_replicate_f64(pg, B + (size_t)l + 1 * ldb);
        svfloat64_t b2 = camblas_sve_load_replicate_f64(pg, B + (size_t)l + 2 * ldb);
        svfloat64_t b3 = camblas_sve_load_replicate_f64(pg, B + (size_t)l + 3 * ldb);
        svfloat64_t b4 = camblas_sve_load_replicate_f64(pg, B + (size_t)l + 4 * ldb);
        svfloat64_t b5 = camblas_sve_load_replicate_f64(pg, B + (size_t)l + 5 * ldb);
        svfloat64_t a0 = svld1_f64(pg, A + (size_t)l * lda + 0 * lanes);
        acc00 = svmla_f64_x(pg, acc00, a0, b0);
        acc01 = svmla_f64_x(pg, acc01, a0, b1);
        acc02 = svmla_f64_x(pg, acc02, a0, b2);
        acc03 = svmla_f64_x(pg, acc03, a0, b3);
        acc04 = svmla_f64_x(pg, acc04, a0, b4);
        acc05 = svmla_f64_x(pg, acc05, a0, b5);
        svfloat64_t a1 = svld1_f64(pg, A + (size_t)l * lda + 1 * lanes);
        acc10 = svmla_f64_x(pg, acc10, a1, b0);
        acc11 = svmla_f64_x(pg, acc11, a1, b1);
        acc12 = svmla_f64_x(pg, acc12, a1, b2);
        acc13 = svmla_f64_x(pg, acc13, a1, b3);
        acc14 = svmla_f64_x(pg, acc14, a1, b4);
        acc15 = svmla_f64_x(pg, acc15, a1, b5);
        svfloat64_t a2 = svld1_f64(pg, A + (size_t)l * lda + 2 * lanes);
        acc20 = svmla_f64_x(pg, acc20, a2, b0);
        acc21 = svmla_f64_x(pg, acc21, a2, b1);
        acc22 = svmla_f64_x(pg, acc22, a2, b2);
        acc23 = svmla_f64_x(pg, acc23, a2, b3);
        acc24 = svmla_f64_x(pg, acc24, a2, b4);
        acc25 = svmla_f64_x(pg, acc25, a2, b5);
        svfloat64_t a3 = svld1_f64(pg, A + (size_t)l * lda + 3 * lanes);
        acc30 = svmla_f64_x(pg, acc30, a3, b0);
        acc31 = svmla_f64_x(pg, acc31, a3, b1);
        acc32 = svmla_f64_x(pg, acc32, a3, b2);
        acc33 = svmla_f64_x(pg, acc33, a3, b3);
        acc34 = svmla_f64_x(pg, acc34, a3, b4);
        acc35 = svmla_f64_x(pg, acc35, a3, b5);
    }
    {
        double *out = C + 0 * lanes + 0 * ldc;
        svfloat64_t value = acc00;
        if (alpha != 1.0)
            value = svmul_n_f64_x(pg, value, alpha);
        svst1_f64(pg, out, svadd_f64_x(pg, svld1_f64(pg, out), value));
    }
    {
        double *out = C + 0 * lanes + 1 * ldc;
        svfloat64_t value = acc01;
        if (alpha != 1.0)
            value = svmul_n_f64_x(pg, value, alpha);
        svst1_f64(pg, out, svadd_f64_x(pg, svld1_f64(pg, out), value));
    }
    {
        double *out = C + 0 * lanes + 2 * ldc;
        svfloat64_t value = acc02;
        if (alpha != 1.0)
            value = svmul_n_f64_x(pg, value, alpha);
        svst1_f64(pg, out, svadd_f64_x(pg, svld1_f64(pg, out), value));
    }
    {
        double *out = C + 0 * lanes + 3 * ldc;
        svfloat64_t value = acc03;
        if (alpha != 1.0)
            value = svmul_n_f64_x(pg, value, alpha);
        svst1_f64(pg, out, svadd_f64_x(pg, svld1_f64(pg, out), value));
    }
    {
        double *out = C + 0 * lanes + 4 * ldc;
        svfloat64_t value = acc04;
        if (alpha != 1.0)
            value = svmul_n_f64_x(pg, value, alpha);
        svst1_f64(pg, out, svadd_f64_x(pg, svld1_f64(pg, out), value));
    }
    {
        double *out = C + 0 * lanes + 5 * ldc;
        svfloat64_t value = acc05;
        if (alpha != 1.0)
            value = svmul_n_f64_x(pg, value, alpha);
        svst1_f64(pg, out, svadd_f64_x(pg, svld1_f64(pg, out), value));
    }
    {
        double *out = C + 1 * lanes + 0 * ldc;
        svfloat64_t value = acc10;
        if (alpha != 1.0)
            value = svmul_n_f64_x(pg, value, alpha);
        svst1_f64(pg, out, svadd_f64_x(pg, svld1_f64(pg, out), value));
    }
    {
        double *out = C + 1 * lanes + 1 * ldc;
        svfloat64_t value = acc11;
        if (alpha != 1.0)
            value = svmul_n_f64_x(pg, value, alpha);
        svst1_f64(pg, out, svadd_f64_x(pg, svld1_f64(pg, out), value));
    }
    {
        double *out = C + 1 * lanes + 2 * ldc;
        svfloat64_t value = acc12;
        if (alpha != 1.0)
            value = svmul_n_f64_x(pg, value, alpha);
        svst1_f64(pg, out, svadd_f64_x(pg, svld1_f64(pg, out), value));
    }
    {
        double *out = C + 1 * lanes + 3 * ldc;
        svfloat64_t value = acc13;
        if (alpha != 1.0)
            value = svmul_n_f64_x(pg, value, alpha);
        svst1_f64(pg, out, svadd_f64_x(pg, svld1_f64(pg, out), value));
    }
    {
        double *out = C + 1 * lanes + 4 * ldc;
        svfloat64_t value = acc14;
        if (alpha != 1.0)
            value = svmul_n_f64_x(pg, value, alpha);
        svst1_f64(pg, out, svadd_f64_x(pg, svld1_f64(pg, out), value));
    }
    {
        double *out = C + 1 * lanes + 5 * ldc;
        svfloat64_t value = acc15;
        if (alpha != 1.0)
            value = svmul_n_f64_x(pg, value, alpha);
        svst1_f64(pg, out, svadd_f64_x(pg, svld1_f64(pg, out), value));
    }
    {
        double *out = C + 2 * lanes + 0 * ldc;
        svfloat64_t value = acc20;
        if (alpha != 1.0)
            value = svmul_n_f64_x(pg, value, alpha);
        svst1_f64(pg, out, svadd_f64_x(pg, svld1_f64(pg, out), value));
    }
    {
        double *out = C + 2 * lanes + 1 * ldc;
        svfloat64_t value = acc21;
        if (alpha != 1.0)
            value = svmul_n_f64_x(pg, value, alpha);
        svst1_f64(pg, out, svadd_f64_x(pg, svld1_f64(pg, out), value));
    }
    {
        double *out = C + 2 * lanes + 2 * ldc;
        svfloat64_t value = acc22;
        if (alpha != 1.0)
            value = svmul_n_f64_x(pg, value, alpha);
        svst1_f64(pg, out, svadd_f64_x(pg, svld1_f64(pg, out), value));
    }
    {
        double *out = C + 2 * lanes + 3 * ldc;
        svfloat64_t value = acc23;
        if (alpha != 1.0)
            value = svmul_n_f64_x(pg, value, alpha);
        svst1_f64(pg, out, svadd_f64_x(pg, svld1_f64(pg, out), value));
    }
    {
        double *out = C + 2 * lanes + 4 * ldc;
        svfloat64_t value = acc24;
        if (alpha != 1.0)
            value = svmul_n_f64_x(pg, value, alpha);
        svst1_f64(pg, out, svadd_f64_x(pg, svld1_f64(pg, out), value));
    }
    {
        double *out = C + 2 * lanes + 5 * ldc;
        svfloat64_t value = acc25;
        if (alpha != 1.0)
            value = svmul_n_f64_x(pg, value, alpha);
        svst1_f64(pg, out, svadd_f64_x(pg, svld1_f64(pg, out), value));
    }
    {
        double *out = C + 3 * lanes + 0 * ldc;
        svfloat64_t value = acc30;
        if (alpha != 1.0)
            value = svmul_n_f64_x(pg, value, alpha);
        svst1_f64(pg, out, svadd_f64_x(pg, svld1_f64(pg, out), value));
    }
    {
        double *out = C + 3 * lanes + 1 * ldc;
        svfloat64_t value = acc31;
        if (alpha != 1.0)
            value = svmul_n_f64_x(pg, value, alpha);
        svst1_f64(pg, out, svadd_f64_x(pg, svld1_f64(pg, out), value));
    }
    {
        double *out = C + 3 * lanes + 2 * ldc;
        svfloat64_t value = acc32;
        if (alpha != 1.0)
            value = svmul_n_f64_x(pg, value, alpha);
        svst1_f64(pg, out, svadd_f64_x(pg, svld1_f64(pg, out), value));
    }
    {
        double *out = C + 3 * lanes + 3 * ldc;
        svfloat64_t value = acc33;
        if (alpha != 1.0)
            value = svmul_n_f64_x(pg, value, alpha);
        svst1_f64(pg, out, svadd_f64_x(pg, svld1_f64(pg, out), value));
    }
    {
        double *out = C + 3 * lanes + 4 * ldc;
        svfloat64_t value = acc34;
        if (alpha != 1.0)
            value = svmul_n_f64_x(pg, value, alpha);
        svst1_f64(pg, out, svadd_f64_x(pg, svld1_f64(pg, out), value));
    }
    {
        double *out = C + 3 * lanes + 5 * ldc;
        svfloat64_t value = acc35;
        if (alpha != 1.0)
            value = svmul_n_f64_x(pg, value, alpha);
        svst1_f64(pg, out, svadd_f64_x(pg, svld1_f64(pg, out), value));
    }
}
#endif
