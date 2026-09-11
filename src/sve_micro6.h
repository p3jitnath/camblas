/* Experimental four-vector by six-column kernel for tagged micro6 B. */
#ifndef CAMBLAS_SVE_MICRO6_H
#define CAMBLAS_SVE_MICRO6_H
#include "sve_lane_ops.h"
static inline int camblas_micro6_stride_f32(int m, int n, int k, float alpha, const float *A,
                                            size_t lda, const float *B, size_t ldb, float *C,
                                            size_t ldc, size_t step)
{
    const size_t vl = svcntw();
    const svbool_t all = svptrue_b32();
    for (size_t j = 0; j < (size_t)n; j += 6) {
        size_t i = 0;
        for (; i + 4 * vl <= (size_t)m; i += 4 * vl) {
            svfloat32_t c00 = svdup_f32(0);
            svfloat32_t c10 = svdup_f32(0);
            svfloat32_t c20 = svdup_f32(0);
            svfloat32_t c30 = svdup_f32(0);
            svfloat32_t c01 = svdup_f32(0);
            svfloat32_t c11 = svdup_f32(0);
            svfloat32_t c21 = svdup_f32(0);
            svfloat32_t c31 = svdup_f32(0);
            svfloat32_t c02 = svdup_f32(0);
            svfloat32_t c12 = svdup_f32(0);
            svfloat32_t c22 = svdup_f32(0);
            svfloat32_t c32 = svdup_f32(0);
            svfloat32_t c03 = svdup_f32(0);
            svfloat32_t c13 = svdup_f32(0);
            svfloat32_t c23 = svdup_f32(0);
            svfloat32_t c33 = svdup_f32(0);
            svfloat32_t c04 = svdup_f32(0);
            svfloat32_t c14 = svdup_f32(0);
            svfloat32_t c24 = svdup_f32(0);
            svfloat32_t c34 = svdup_f32(0);
            svfloat32_t c05 = svdup_f32(0);
            svfloat32_t c15 = svdup_f32(0);
            svfloat32_t c25 = svdup_f32(0);
            svfloat32_t c35 = svdup_f32(0);
            for (size_t l = 0; l < (size_t)k; l++) {
                svfloat32_t a0 = svld1_f32(all, A + i + 0 * vl + l * lda);
                svfloat32_t a1 = svld1_f32(all, A + i + 1 * vl + l * lda);
                svfloat32_t a2 = svld1_f32(all, A + i + 2 * vl + l * lda);
                svfloat32_t a3 = svld1_f32(all, A + i + 3 * vl + l * lda);
                svfloat32_t b0 = svld1rq_f32(all, B + (j / 6 * step) * ldb + l * step + 0);
                svfloat32_t b1 = svld1rq_f32(svwhilelt_b32((uint64_t)0, (uint64_t)2),
                                             B + (j / 6 * step) * ldb + l * step + 4);
                CAMBLAS_LANE_F32(c00, a0, b0, 0);
                CAMBLAS_LANE_F32(c10, a1, b0, 0);
                CAMBLAS_LANE_F32(c20, a2, b0, 0);
                CAMBLAS_LANE_F32(c30, a3, b0, 0);
                CAMBLAS_LANE_F32(c01, a0, b0, 1);
                CAMBLAS_LANE_F32(c11, a1, b0, 1);
                CAMBLAS_LANE_F32(c21, a2, b0, 1);
                CAMBLAS_LANE_F32(c31, a3, b0, 1);
                CAMBLAS_LANE_F32(c02, a0, b0, 2);
                CAMBLAS_LANE_F32(c12, a1, b0, 2);
                CAMBLAS_LANE_F32(c22, a2, b0, 2);
                CAMBLAS_LANE_F32(c32, a3, b0, 2);
                CAMBLAS_LANE_F32(c03, a0, b0, 3);
                CAMBLAS_LANE_F32(c13, a1, b0, 3);
                CAMBLAS_LANE_F32(c23, a2, b0, 3);
                CAMBLAS_LANE_F32(c33, a3, b0, 3);
                CAMBLAS_LANE_F32(c04, a0, b1, 0);
                CAMBLAS_LANE_F32(c14, a1, b1, 0);
                CAMBLAS_LANE_F32(c24, a2, b1, 0);
                CAMBLAS_LANE_F32(c34, a3, b1, 0);
                CAMBLAS_LANE_F32(c05, a0, b1, 1);
                CAMBLAS_LANE_F32(c15, a1, b1, 1);
                CAMBLAS_LANE_F32(c25, a2, b1, 1);
                CAMBLAS_LANE_F32(c35, a3, b1, 1);
            }
            if (j + 0 < (size_t)n) {
                float *p0 = C + i + 0 * vl + (j + 0) * ldc;
                svst1_f32(all, p0,
                          svadd_f32_x(all, svld1_f32(all, p0), svmul_n_f32_x(all, c00, alpha)));
                float *p1 = C + i + 1 * vl + (j + 0) * ldc;
                svst1_f32(all, p1,
                          svadd_f32_x(all, svld1_f32(all, p1), svmul_n_f32_x(all, c10, alpha)));
                float *p2 = C + i + 2 * vl + (j + 0) * ldc;
                svst1_f32(all, p2,
                          svadd_f32_x(all, svld1_f32(all, p2), svmul_n_f32_x(all, c20, alpha)));
                float *p3 = C + i + 3 * vl + (j + 0) * ldc;
                svst1_f32(all, p3,
                          svadd_f32_x(all, svld1_f32(all, p3), svmul_n_f32_x(all, c30, alpha)));
            }
            if (j + 1 < (size_t)n) {
                float *p0 = C + i + 0 * vl + (j + 1) * ldc;
                svst1_f32(all, p0,
                          svadd_f32_x(all, svld1_f32(all, p0), svmul_n_f32_x(all, c01, alpha)));
                float *p1 = C + i + 1 * vl + (j + 1) * ldc;
                svst1_f32(all, p1,
                          svadd_f32_x(all, svld1_f32(all, p1), svmul_n_f32_x(all, c11, alpha)));
                float *p2 = C + i + 2 * vl + (j + 1) * ldc;
                svst1_f32(all, p2,
                          svadd_f32_x(all, svld1_f32(all, p2), svmul_n_f32_x(all, c21, alpha)));
                float *p3 = C + i + 3 * vl + (j + 1) * ldc;
                svst1_f32(all, p3,
                          svadd_f32_x(all, svld1_f32(all, p3), svmul_n_f32_x(all, c31, alpha)));
            }
            if (j + 2 < (size_t)n) {
                float *p0 = C + i + 0 * vl + (j + 2) * ldc;
                svst1_f32(all, p0,
                          svadd_f32_x(all, svld1_f32(all, p0), svmul_n_f32_x(all, c02, alpha)));
                float *p1 = C + i + 1 * vl + (j + 2) * ldc;
                svst1_f32(all, p1,
                          svadd_f32_x(all, svld1_f32(all, p1), svmul_n_f32_x(all, c12, alpha)));
                float *p2 = C + i + 2 * vl + (j + 2) * ldc;
                svst1_f32(all, p2,
                          svadd_f32_x(all, svld1_f32(all, p2), svmul_n_f32_x(all, c22, alpha)));
                float *p3 = C + i + 3 * vl + (j + 2) * ldc;
                svst1_f32(all, p3,
                          svadd_f32_x(all, svld1_f32(all, p3), svmul_n_f32_x(all, c32, alpha)));
            }
            if (j + 3 < (size_t)n) {
                float *p0 = C + i + 0 * vl + (j + 3) * ldc;
                svst1_f32(all, p0,
                          svadd_f32_x(all, svld1_f32(all, p0), svmul_n_f32_x(all, c03, alpha)));
                float *p1 = C + i + 1 * vl + (j + 3) * ldc;
                svst1_f32(all, p1,
                          svadd_f32_x(all, svld1_f32(all, p1), svmul_n_f32_x(all, c13, alpha)));
                float *p2 = C + i + 2 * vl + (j + 3) * ldc;
                svst1_f32(all, p2,
                          svadd_f32_x(all, svld1_f32(all, p2), svmul_n_f32_x(all, c23, alpha)));
                float *p3 = C + i + 3 * vl + (j + 3) * ldc;
                svst1_f32(all, p3,
                          svadd_f32_x(all, svld1_f32(all, p3), svmul_n_f32_x(all, c33, alpha)));
            }
            if (j + 4 < (size_t)n) {
                float *p0 = C + i + 0 * vl + (j + 4) * ldc;
                svst1_f32(all, p0,
                          svadd_f32_x(all, svld1_f32(all, p0), svmul_n_f32_x(all, c04, alpha)));
                float *p1 = C + i + 1 * vl + (j + 4) * ldc;
                svst1_f32(all, p1,
                          svadd_f32_x(all, svld1_f32(all, p1), svmul_n_f32_x(all, c14, alpha)));
                float *p2 = C + i + 2 * vl + (j + 4) * ldc;
                svst1_f32(all, p2,
                          svadd_f32_x(all, svld1_f32(all, p2), svmul_n_f32_x(all, c24, alpha)));
                float *p3 = C + i + 3 * vl + (j + 4) * ldc;
                svst1_f32(all, p3,
                          svadd_f32_x(all, svld1_f32(all, p3), svmul_n_f32_x(all, c34, alpha)));
            }
            if (j + 5 < (size_t)n) {
                float *p0 = C + i + 0 * vl + (j + 5) * ldc;
                svst1_f32(all, p0,
                          svadd_f32_x(all, svld1_f32(all, p0), svmul_n_f32_x(all, c05, alpha)));
                float *p1 = C + i + 1 * vl + (j + 5) * ldc;
                svst1_f32(all, p1,
                          svadd_f32_x(all, svld1_f32(all, p1), svmul_n_f32_x(all, c15, alpha)));
                float *p2 = C + i + 2 * vl + (j + 5) * ldc;
                svst1_f32(all, p2,
                          svadd_f32_x(all, svld1_f32(all, p2), svmul_n_f32_x(all, c25, alpha)));
                float *p3 = C + i + 3 * vl + (j + 5) * ldc;
                svst1_f32(all, p3,
                          svadd_f32_x(all, svld1_f32(all, p3), svmul_n_f32_x(all, c35, alpha)));
            }
        }
        for (; i < (size_t)m; i += vl) {
            svbool_t pg = svwhilelt_b32((uint64_t)i, (uint64_t)m);
            svfloat32_t t0 = svdup_f32(0);
            svfloat32_t t1 = svdup_f32(0);
            svfloat32_t t2 = svdup_f32(0);
            svfloat32_t t3 = svdup_f32(0);
            svfloat32_t t4 = svdup_f32(0);
            svfloat32_t t5 = svdup_f32(0);
            for (size_t l = 0; l < (size_t)k; l++) {
                svfloat32_t a = svld1_f32(pg, A + i + l * lda);
                t0 = svmla_n_f32_x(pg, t0, a, B[(j / 6 * step) * ldb + l * step + 0]);
                t1 = svmla_n_f32_x(pg, t1, a, B[(j / 6 * step) * ldb + l * step + 1]);
                t2 = svmla_n_f32_x(pg, t2, a, B[(j / 6 * step) * ldb + l * step + 2]);
                t3 = svmla_n_f32_x(pg, t3, a, B[(j / 6 * step) * ldb + l * step + 3]);
                t4 = svmla_n_f32_x(pg, t4, a, B[(j / 6 * step) * ldb + l * step + 4]);
                t5 = svmla_n_f32_x(pg, t5, a, B[(j / 6 * step) * ldb + l * step + 5]);
            }
            if (j + 0 < (size_t)n) {
                float *p = C + i + (j + 0) * ldc;
                svst1_f32(pg, p, svadd_f32_x(pg, svld1_f32(pg, p), svmul_n_f32_x(pg, t0, alpha)));
            }
            if (j + 1 < (size_t)n) {
                float *p = C + i + (j + 1) * ldc;
                svst1_f32(pg, p, svadd_f32_x(pg, svld1_f32(pg, p), svmul_n_f32_x(pg, t1, alpha)));
            }
            if (j + 2 < (size_t)n) {
                float *p = C + i + (j + 2) * ldc;
                svst1_f32(pg, p, svadd_f32_x(pg, svld1_f32(pg, p), svmul_n_f32_x(pg, t2, alpha)));
            }
            if (j + 3 < (size_t)n) {
                float *p = C + i + (j + 3) * ldc;
                svst1_f32(pg, p, svadd_f32_x(pg, svld1_f32(pg, p), svmul_n_f32_x(pg, t3, alpha)));
            }
            if (j + 4 < (size_t)n) {
                float *p = C + i + (j + 4) * ldc;
                svst1_f32(pg, p, svadd_f32_x(pg, svld1_f32(pg, p), svmul_n_f32_x(pg, t4, alpha)));
            }
            if (j + 5 < (size_t)n) {
                float *p = C + i + (j + 5) * ldc;
                svst1_f32(pg, p, svadd_f32_x(pg, svld1_f32(pg, p), svmul_n_f32_x(pg, t5, alpha)));
            }
        }
    }
    return 0;
}
static inline int camblas_micro6_stride_f64(int m, int n, int k, double alpha, const double *A,
                                            size_t lda, const double *B, size_t ldb, double *C,
                                            size_t ldc, size_t step)
{
    const size_t vl = svcntd();
    const svbool_t all = svptrue_b64();
    for (size_t j = 0; j < (size_t)n; j += 6) {
        size_t i = 0;
        for (; i + 4 * vl <= (size_t)m; i += 4 * vl) {
            svfloat64_t c00 = svdup_f64(0);
            svfloat64_t c10 = svdup_f64(0);
            svfloat64_t c20 = svdup_f64(0);
            svfloat64_t c30 = svdup_f64(0);
            svfloat64_t c01 = svdup_f64(0);
            svfloat64_t c11 = svdup_f64(0);
            svfloat64_t c21 = svdup_f64(0);
            svfloat64_t c31 = svdup_f64(0);
            svfloat64_t c02 = svdup_f64(0);
            svfloat64_t c12 = svdup_f64(0);
            svfloat64_t c22 = svdup_f64(0);
            svfloat64_t c32 = svdup_f64(0);
            svfloat64_t c03 = svdup_f64(0);
            svfloat64_t c13 = svdup_f64(0);
            svfloat64_t c23 = svdup_f64(0);
            svfloat64_t c33 = svdup_f64(0);
            svfloat64_t c04 = svdup_f64(0);
            svfloat64_t c14 = svdup_f64(0);
            svfloat64_t c24 = svdup_f64(0);
            svfloat64_t c34 = svdup_f64(0);
            svfloat64_t c05 = svdup_f64(0);
            svfloat64_t c15 = svdup_f64(0);
            svfloat64_t c25 = svdup_f64(0);
            svfloat64_t c35 = svdup_f64(0);
            for (size_t l = 0; l < (size_t)k; l++) {
                svfloat64_t a0 = svld1_f64(all, A + i + 0 * vl + l * lda);
                svfloat64_t a1 = svld1_f64(all, A + i + 1 * vl + l * lda);
                svfloat64_t a2 = svld1_f64(all, A + i + 2 * vl + l * lda);
                svfloat64_t a3 = svld1_f64(all, A + i + 3 * vl + l * lda);
                svfloat64_t b0 = svld1rq_f64(all, B + (j / 6 * step) * ldb + l * step + 0);
                svfloat64_t b1 = svld1rq_f64(all, B + (j / 6 * step) * ldb + l * step + 2);
                svfloat64_t b2 = svld1rq_f64(all, B + (j / 6 * step) * ldb + l * step + 4);
                CAMBLAS_LANE_F64(c00, a0, b0, 0);
                CAMBLAS_LANE_F64(c10, a1, b0, 0);
                CAMBLAS_LANE_F64(c20, a2, b0, 0);
                CAMBLAS_LANE_F64(c30, a3, b0, 0);
                CAMBLAS_LANE_F64(c01, a0, b0, 1);
                CAMBLAS_LANE_F64(c11, a1, b0, 1);
                CAMBLAS_LANE_F64(c21, a2, b0, 1);
                CAMBLAS_LANE_F64(c31, a3, b0, 1);
                CAMBLAS_LANE_F64(c02, a0, b1, 0);
                CAMBLAS_LANE_F64(c12, a1, b1, 0);
                CAMBLAS_LANE_F64(c22, a2, b1, 0);
                CAMBLAS_LANE_F64(c32, a3, b1, 0);
                CAMBLAS_LANE_F64(c03, a0, b1, 1);
                CAMBLAS_LANE_F64(c13, a1, b1, 1);
                CAMBLAS_LANE_F64(c23, a2, b1, 1);
                CAMBLAS_LANE_F64(c33, a3, b1, 1);
                CAMBLAS_LANE_F64(c04, a0, b2, 0);
                CAMBLAS_LANE_F64(c14, a1, b2, 0);
                CAMBLAS_LANE_F64(c24, a2, b2, 0);
                CAMBLAS_LANE_F64(c34, a3, b2, 0);
                CAMBLAS_LANE_F64(c05, a0, b2, 1);
                CAMBLAS_LANE_F64(c15, a1, b2, 1);
                CAMBLAS_LANE_F64(c25, a2, b2, 1);
                CAMBLAS_LANE_F64(c35, a3, b2, 1);
            }
            if (j + 0 < (size_t)n) {
                double *p0 = C + i + 0 * vl + (j + 0) * ldc;
                svst1_f64(all, p0,
                          svadd_f64_x(all, svld1_f64(all, p0), svmul_n_f64_x(all, c00, alpha)));
                double *p1 = C + i + 1 * vl + (j + 0) * ldc;
                svst1_f64(all, p1,
                          svadd_f64_x(all, svld1_f64(all, p1), svmul_n_f64_x(all, c10, alpha)));
                double *p2 = C + i + 2 * vl + (j + 0) * ldc;
                svst1_f64(all, p2,
                          svadd_f64_x(all, svld1_f64(all, p2), svmul_n_f64_x(all, c20, alpha)));
                double *p3 = C + i + 3 * vl + (j + 0) * ldc;
                svst1_f64(all, p3,
                          svadd_f64_x(all, svld1_f64(all, p3), svmul_n_f64_x(all, c30, alpha)));
            }
            if (j + 1 < (size_t)n) {
                double *p0 = C + i + 0 * vl + (j + 1) * ldc;
                svst1_f64(all, p0,
                          svadd_f64_x(all, svld1_f64(all, p0), svmul_n_f64_x(all, c01, alpha)));
                double *p1 = C + i + 1 * vl + (j + 1) * ldc;
                svst1_f64(all, p1,
                          svadd_f64_x(all, svld1_f64(all, p1), svmul_n_f64_x(all, c11, alpha)));
                double *p2 = C + i + 2 * vl + (j + 1) * ldc;
                svst1_f64(all, p2,
                          svadd_f64_x(all, svld1_f64(all, p2), svmul_n_f64_x(all, c21, alpha)));
                double *p3 = C + i + 3 * vl + (j + 1) * ldc;
                svst1_f64(all, p3,
                          svadd_f64_x(all, svld1_f64(all, p3), svmul_n_f64_x(all, c31, alpha)));
            }
            if (j + 2 < (size_t)n) {
                double *p0 = C + i + 0 * vl + (j + 2) * ldc;
                svst1_f64(all, p0,
                          svadd_f64_x(all, svld1_f64(all, p0), svmul_n_f64_x(all, c02, alpha)));
                double *p1 = C + i + 1 * vl + (j + 2) * ldc;
                svst1_f64(all, p1,
                          svadd_f64_x(all, svld1_f64(all, p1), svmul_n_f64_x(all, c12, alpha)));
                double *p2 = C + i + 2 * vl + (j + 2) * ldc;
                svst1_f64(all, p2,
                          svadd_f64_x(all, svld1_f64(all, p2), svmul_n_f64_x(all, c22, alpha)));
                double *p3 = C + i + 3 * vl + (j + 2) * ldc;
                svst1_f64(all, p3,
                          svadd_f64_x(all, svld1_f64(all, p3), svmul_n_f64_x(all, c32, alpha)));
            }
            if (j + 3 < (size_t)n) {
                double *p0 = C + i + 0 * vl + (j + 3) * ldc;
                svst1_f64(all, p0,
                          svadd_f64_x(all, svld1_f64(all, p0), svmul_n_f64_x(all, c03, alpha)));
                double *p1 = C + i + 1 * vl + (j + 3) * ldc;
                svst1_f64(all, p1,
                          svadd_f64_x(all, svld1_f64(all, p1), svmul_n_f64_x(all, c13, alpha)));
                double *p2 = C + i + 2 * vl + (j + 3) * ldc;
                svst1_f64(all, p2,
                          svadd_f64_x(all, svld1_f64(all, p2), svmul_n_f64_x(all, c23, alpha)));
                double *p3 = C + i + 3 * vl + (j + 3) * ldc;
                svst1_f64(all, p3,
                          svadd_f64_x(all, svld1_f64(all, p3), svmul_n_f64_x(all, c33, alpha)));
            }
            if (j + 4 < (size_t)n) {
                double *p0 = C + i + 0 * vl + (j + 4) * ldc;
                svst1_f64(all, p0,
                          svadd_f64_x(all, svld1_f64(all, p0), svmul_n_f64_x(all, c04, alpha)));
                double *p1 = C + i + 1 * vl + (j + 4) * ldc;
                svst1_f64(all, p1,
                          svadd_f64_x(all, svld1_f64(all, p1), svmul_n_f64_x(all, c14, alpha)));
                double *p2 = C + i + 2 * vl + (j + 4) * ldc;
                svst1_f64(all, p2,
                          svadd_f64_x(all, svld1_f64(all, p2), svmul_n_f64_x(all, c24, alpha)));
                double *p3 = C + i + 3 * vl + (j + 4) * ldc;
                svst1_f64(all, p3,
                          svadd_f64_x(all, svld1_f64(all, p3), svmul_n_f64_x(all, c34, alpha)));
            }
            if (j + 5 < (size_t)n) {
                double *p0 = C + i + 0 * vl + (j + 5) * ldc;
                svst1_f64(all, p0,
                          svadd_f64_x(all, svld1_f64(all, p0), svmul_n_f64_x(all, c05, alpha)));
                double *p1 = C + i + 1 * vl + (j + 5) * ldc;
                svst1_f64(all, p1,
                          svadd_f64_x(all, svld1_f64(all, p1), svmul_n_f64_x(all, c15, alpha)));
                double *p2 = C + i + 2 * vl + (j + 5) * ldc;
                svst1_f64(all, p2,
                          svadd_f64_x(all, svld1_f64(all, p2), svmul_n_f64_x(all, c25, alpha)));
                double *p3 = C + i + 3 * vl + (j + 5) * ldc;
                svst1_f64(all, p3,
                          svadd_f64_x(all, svld1_f64(all, p3), svmul_n_f64_x(all, c35, alpha)));
            }
        }
        for (; i < (size_t)m; i += vl) {
            svbool_t pg = svwhilelt_b64((uint64_t)i, (uint64_t)m);
            svfloat64_t t0 = svdup_f64(0);
            svfloat64_t t1 = svdup_f64(0);
            svfloat64_t t2 = svdup_f64(0);
            svfloat64_t t3 = svdup_f64(0);
            svfloat64_t t4 = svdup_f64(0);
            svfloat64_t t5 = svdup_f64(0);
            for (size_t l = 0; l < (size_t)k; l++) {
                svfloat64_t a = svld1_f64(pg, A + i + l * lda);
                t0 = svmla_n_f64_x(pg, t0, a, B[(j / 6 * step) * ldb + l * step + 0]);
                t1 = svmla_n_f64_x(pg, t1, a, B[(j / 6 * step) * ldb + l * step + 1]);
                t2 = svmla_n_f64_x(pg, t2, a, B[(j / 6 * step) * ldb + l * step + 2]);
                t3 = svmla_n_f64_x(pg, t3, a, B[(j / 6 * step) * ldb + l * step + 3]);
                t4 = svmla_n_f64_x(pg, t4, a, B[(j / 6 * step) * ldb + l * step + 4]);
                t5 = svmla_n_f64_x(pg, t5, a, B[(j / 6 * step) * ldb + l * step + 5]);
            }
            if (j + 0 < (size_t)n) {
                double *p = C + i + (j + 0) * ldc;
                svst1_f64(pg, p, svadd_f64_x(pg, svld1_f64(pg, p), svmul_n_f64_x(pg, t0, alpha)));
            }
            if (j + 1 < (size_t)n) {
                double *p = C + i + (j + 1) * ldc;
                svst1_f64(pg, p, svadd_f64_x(pg, svld1_f64(pg, p), svmul_n_f64_x(pg, t1, alpha)));
            }
            if (j + 2 < (size_t)n) {
                double *p = C + i + (j + 2) * ldc;
                svst1_f64(pg, p, svadd_f64_x(pg, svld1_f64(pg, p), svmul_n_f64_x(pg, t2, alpha)));
            }
            if (j + 3 < (size_t)n) {
                double *p = C + i + (j + 3) * ldc;
                svst1_f64(pg, p, svadd_f64_x(pg, svld1_f64(pg, p), svmul_n_f64_x(pg, t3, alpha)));
            }
            if (j + 4 < (size_t)n) {
                double *p = C + i + (j + 4) * ldc;
                svst1_f64(pg, p, svadd_f64_x(pg, svld1_f64(pg, p), svmul_n_f64_x(pg, t4, alpha)));
            }
            if (j + 5 < (size_t)n) {
                double *p = C + i + (j + 5) * ldc;
                svst1_f64(pg, p, svadd_f64_x(pg, svld1_f64(pg, p), svmul_n_f64_x(pg, t5, alpha)));
            }
        }
    }
    return 0;
}
static inline int camblas_micro6_f32(int m, int n, int k, float alpha, const float *A, size_t lda,
                                     const float *B, size_t ldb, float *C, size_t ldc)
{
    return camblas_micro6_stride_f32(m, n, k, alpha, A, lda, B, ldb, C, ldc, 6);
}

static inline int camblas_micro6_f32_padded(int m, int n, int k, float alpha, const float *A,
                                            size_t lda, const float *B, size_t ldb, float *C,
                                            size_t ldc)
{
    return camblas_micro6_stride_f32(m, n, k, alpha, A, lda, B, ldb, C, ldc, 8);
}

static inline int camblas_micro6_f64(int m, int n, int k, double alpha, const double *A, size_t lda,
                                     const double *B, size_t ldb, double *C, size_t ldc)
{
    return camblas_micro6_stride_f64(m, n, k, alpha, A, lda, B, ldb, C, ldc, 6);
}

static inline int camblas_micro6_f64_padded(int m, int n, int k, double alpha, const double *A,
                                            size_t lda, const double *B, size_t ldb, double *C,
                                            size_t ldc)
{
    return camblas_micro6_stride_f64(m, n, k, alpha, A, lda, B, ldb, C, ldc, 8);
}

#endif
