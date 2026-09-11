/* Fixed-width NEON control for ordinary-A, micro8-B GEMM. */
#ifndef CAMBLAS_NEON_MICRO8_H
#define CAMBLAS_NEON_MICRO8_H
#include <arm_neon.h>
static inline int camblas_neon_micro8_f32(int m, int n, int k, float alpha, const float *A,
                                          size_t lda, const float *B, size_t ldb, float *C,
                                          size_t ldc, int initialize)
{
    for (size_t j = 0; j < (size_t)n; j += 8) {
        size_t i = 0;
        for (; i + 12 <= (size_t)m; i += 12) {
            float32x4_t c00 = vdupq_n_f32(0);
            float32x4_t c01 = vdupq_n_f32(0);
            float32x4_t c02 = vdupq_n_f32(0);
            float32x4_t c03 = vdupq_n_f32(0);
            float32x4_t c04 = vdupq_n_f32(0);
            float32x4_t c05 = vdupq_n_f32(0);
            float32x4_t c06 = vdupq_n_f32(0);
            float32x4_t c07 = vdupq_n_f32(0);
            float32x4_t c10 = vdupq_n_f32(0);
            float32x4_t c11 = vdupq_n_f32(0);
            float32x4_t c12 = vdupq_n_f32(0);
            float32x4_t c13 = vdupq_n_f32(0);
            float32x4_t c14 = vdupq_n_f32(0);
            float32x4_t c15 = vdupq_n_f32(0);
            float32x4_t c16 = vdupq_n_f32(0);
            float32x4_t c17 = vdupq_n_f32(0);
            float32x4_t c20 = vdupq_n_f32(0);
            float32x4_t c21 = vdupq_n_f32(0);
            float32x4_t c22 = vdupq_n_f32(0);
            float32x4_t c23 = vdupq_n_f32(0);
            float32x4_t c24 = vdupq_n_f32(0);
            float32x4_t c25 = vdupq_n_f32(0);
            float32x4_t c26 = vdupq_n_f32(0);
            float32x4_t c27 = vdupq_n_f32(0);
            for (size_t l = 0; l < (size_t)k; ++l) {
                float32x4_t a0 = vld1q_f32(A + i + 0 + l * lda);
                float32x4_t a1 = vld1q_f32(A + i + 4 + l * lda);
                float32x4_t a2 = vld1q_f32(A + i + 8 + l * lda);
                float32x4_t b0 = vld1q_f32(B + j * ldb + l * 8 + 0);
                float32x4_t b1 = vld1q_f32(B + j * ldb + l * 8 + 4);
                c00 = vfmaq_laneq_f32(c00, a0, b0, 0);
                c01 = vfmaq_laneq_f32(c01, a0, b0, 1);
                c02 = vfmaq_laneq_f32(c02, a0, b0, 2);
                c03 = vfmaq_laneq_f32(c03, a0, b0, 3);
                c04 = vfmaq_laneq_f32(c04, a0, b1, 0);
                c05 = vfmaq_laneq_f32(c05, a0, b1, 1);
                c06 = vfmaq_laneq_f32(c06, a0, b1, 2);
                c07 = vfmaq_laneq_f32(c07, a0, b1, 3);
                c10 = vfmaq_laneq_f32(c10, a1, b0, 0);
                c11 = vfmaq_laneq_f32(c11, a1, b0, 1);
                c12 = vfmaq_laneq_f32(c12, a1, b0, 2);
                c13 = vfmaq_laneq_f32(c13, a1, b0, 3);
                c14 = vfmaq_laneq_f32(c14, a1, b1, 0);
                c15 = vfmaq_laneq_f32(c15, a1, b1, 1);
                c16 = vfmaq_laneq_f32(c16, a1, b1, 2);
                c17 = vfmaq_laneq_f32(c17, a1, b1, 3);
                c20 = vfmaq_laneq_f32(c20, a2, b0, 0);
                c21 = vfmaq_laneq_f32(c21, a2, b0, 1);
                c22 = vfmaq_laneq_f32(c22, a2, b0, 2);
                c23 = vfmaq_laneq_f32(c23, a2, b0, 3);
                c24 = vfmaq_laneq_f32(c24, a2, b1, 0);
                c25 = vfmaq_laneq_f32(c25, a2, b1, 1);
                c26 = vfmaq_laneq_f32(c26, a2, b1, 2);
                c27 = vfmaq_laneq_f32(c27, a2, b1, 3);
            }
            if (j + 0 < (size_t)n) {
                float32x4_t v0 = vmulq_n_f32(c00, alpha);
                if (!initialize)
                    v0 = vaddq_f32(v0, vld1q_f32(C + i + 0 + (j + 0) * ldc));
                vst1q_f32(C + i + 0 + (j + 0) * ldc, v0);
                float32x4_t v1 = vmulq_n_f32(c10, alpha);
                if (!initialize)
                    v1 = vaddq_f32(v1, vld1q_f32(C + i + 4 + (j + 0) * ldc));
                vst1q_f32(C + i + 4 + (j + 0) * ldc, v1);
                float32x4_t v2 = vmulq_n_f32(c20, alpha);
                if (!initialize)
                    v2 = vaddq_f32(v2, vld1q_f32(C + i + 8 + (j + 0) * ldc));
                vst1q_f32(C + i + 8 + (j + 0) * ldc, v2);
            }
            if (j + 1 < (size_t)n) {
                float32x4_t v0 = vmulq_n_f32(c01, alpha);
                if (!initialize)
                    v0 = vaddq_f32(v0, vld1q_f32(C + i + 0 + (j + 1) * ldc));
                vst1q_f32(C + i + 0 + (j + 1) * ldc, v0);
                float32x4_t v1 = vmulq_n_f32(c11, alpha);
                if (!initialize)
                    v1 = vaddq_f32(v1, vld1q_f32(C + i + 4 + (j + 1) * ldc));
                vst1q_f32(C + i + 4 + (j + 1) * ldc, v1);
                float32x4_t v2 = vmulq_n_f32(c21, alpha);
                if (!initialize)
                    v2 = vaddq_f32(v2, vld1q_f32(C + i + 8 + (j + 1) * ldc));
                vst1q_f32(C + i + 8 + (j + 1) * ldc, v2);
            }
            if (j + 2 < (size_t)n) {
                float32x4_t v0 = vmulq_n_f32(c02, alpha);
                if (!initialize)
                    v0 = vaddq_f32(v0, vld1q_f32(C + i + 0 + (j + 2) * ldc));
                vst1q_f32(C + i + 0 + (j + 2) * ldc, v0);
                float32x4_t v1 = vmulq_n_f32(c12, alpha);
                if (!initialize)
                    v1 = vaddq_f32(v1, vld1q_f32(C + i + 4 + (j + 2) * ldc));
                vst1q_f32(C + i + 4 + (j + 2) * ldc, v1);
                float32x4_t v2 = vmulq_n_f32(c22, alpha);
                if (!initialize)
                    v2 = vaddq_f32(v2, vld1q_f32(C + i + 8 + (j + 2) * ldc));
                vst1q_f32(C + i + 8 + (j + 2) * ldc, v2);
            }
            if (j + 3 < (size_t)n) {
                float32x4_t v0 = vmulq_n_f32(c03, alpha);
                if (!initialize)
                    v0 = vaddq_f32(v0, vld1q_f32(C + i + 0 + (j + 3) * ldc));
                vst1q_f32(C + i + 0 + (j + 3) * ldc, v0);
                float32x4_t v1 = vmulq_n_f32(c13, alpha);
                if (!initialize)
                    v1 = vaddq_f32(v1, vld1q_f32(C + i + 4 + (j + 3) * ldc));
                vst1q_f32(C + i + 4 + (j + 3) * ldc, v1);
                float32x4_t v2 = vmulq_n_f32(c23, alpha);
                if (!initialize)
                    v2 = vaddq_f32(v2, vld1q_f32(C + i + 8 + (j + 3) * ldc));
                vst1q_f32(C + i + 8 + (j + 3) * ldc, v2);
            }
            if (j + 4 < (size_t)n) {
                float32x4_t v0 = vmulq_n_f32(c04, alpha);
                if (!initialize)
                    v0 = vaddq_f32(v0, vld1q_f32(C + i + 0 + (j + 4) * ldc));
                vst1q_f32(C + i + 0 + (j + 4) * ldc, v0);
                float32x4_t v1 = vmulq_n_f32(c14, alpha);
                if (!initialize)
                    v1 = vaddq_f32(v1, vld1q_f32(C + i + 4 + (j + 4) * ldc));
                vst1q_f32(C + i + 4 + (j + 4) * ldc, v1);
                float32x4_t v2 = vmulq_n_f32(c24, alpha);
                if (!initialize)
                    v2 = vaddq_f32(v2, vld1q_f32(C + i + 8 + (j + 4) * ldc));
                vst1q_f32(C + i + 8 + (j + 4) * ldc, v2);
            }
            if (j + 5 < (size_t)n) {
                float32x4_t v0 = vmulq_n_f32(c05, alpha);
                if (!initialize)
                    v0 = vaddq_f32(v0, vld1q_f32(C + i + 0 + (j + 5) * ldc));
                vst1q_f32(C + i + 0 + (j + 5) * ldc, v0);
                float32x4_t v1 = vmulq_n_f32(c15, alpha);
                if (!initialize)
                    v1 = vaddq_f32(v1, vld1q_f32(C + i + 4 + (j + 5) * ldc));
                vst1q_f32(C + i + 4 + (j + 5) * ldc, v1);
                float32x4_t v2 = vmulq_n_f32(c25, alpha);
                if (!initialize)
                    v2 = vaddq_f32(v2, vld1q_f32(C + i + 8 + (j + 5) * ldc));
                vst1q_f32(C + i + 8 + (j + 5) * ldc, v2);
            }
            if (j + 6 < (size_t)n) {
                float32x4_t v0 = vmulq_n_f32(c06, alpha);
                if (!initialize)
                    v0 = vaddq_f32(v0, vld1q_f32(C + i + 0 + (j + 6) * ldc));
                vst1q_f32(C + i + 0 + (j + 6) * ldc, v0);
                float32x4_t v1 = vmulq_n_f32(c16, alpha);
                if (!initialize)
                    v1 = vaddq_f32(v1, vld1q_f32(C + i + 4 + (j + 6) * ldc));
                vst1q_f32(C + i + 4 + (j + 6) * ldc, v1);
                float32x4_t v2 = vmulq_n_f32(c26, alpha);
                if (!initialize)
                    v2 = vaddq_f32(v2, vld1q_f32(C + i + 8 + (j + 6) * ldc));
                vst1q_f32(C + i + 8 + (j + 6) * ldc, v2);
            }
            if (j + 7 < (size_t)n) {
                float32x4_t v0 = vmulq_n_f32(c07, alpha);
                if (!initialize)
                    v0 = vaddq_f32(v0, vld1q_f32(C + i + 0 + (j + 7) * ldc));
                vst1q_f32(C + i + 0 + (j + 7) * ldc, v0);
                float32x4_t v1 = vmulq_n_f32(c17, alpha);
                if (!initialize)
                    v1 = vaddq_f32(v1, vld1q_f32(C + i + 4 + (j + 7) * ldc));
                vst1q_f32(C + i + 4 + (j + 7) * ldc, v1);
                float32x4_t v2 = vmulq_n_f32(c27, alpha);
                if (!initialize)
                    v2 = vaddq_f32(v2, vld1q_f32(C + i + 8 + (j + 7) * ldc));
                vst1q_f32(C + i + 8 + (j + 7) * ldc, v2);
            }
        }
        for (; i + 4 <= (size_t)m; i += 4) {
            float32x4_t c00 = vdupq_n_f32(0);
            float32x4_t c01 = vdupq_n_f32(0);
            float32x4_t c02 = vdupq_n_f32(0);
            float32x4_t c03 = vdupq_n_f32(0);
            float32x4_t c04 = vdupq_n_f32(0);
            float32x4_t c05 = vdupq_n_f32(0);
            float32x4_t c06 = vdupq_n_f32(0);
            float32x4_t c07 = vdupq_n_f32(0);
            for (size_t l = 0; l < (size_t)k; ++l) {
                float32x4_t a0 = vld1q_f32(A + i + 0 + l * lda);
                float32x4_t b0 = vld1q_f32(B + j * ldb + l * 8 + 0);
                float32x4_t b1 = vld1q_f32(B + j * ldb + l * 8 + 4);
                c00 = vfmaq_laneq_f32(c00, a0, b0, 0);
                c01 = vfmaq_laneq_f32(c01, a0, b0, 1);
                c02 = vfmaq_laneq_f32(c02, a0, b0, 2);
                c03 = vfmaq_laneq_f32(c03, a0, b0, 3);
                c04 = vfmaq_laneq_f32(c04, a0, b1, 0);
                c05 = vfmaq_laneq_f32(c05, a0, b1, 1);
                c06 = vfmaq_laneq_f32(c06, a0, b1, 2);
                c07 = vfmaq_laneq_f32(c07, a0, b1, 3);
            }
            if (j + 0 < (size_t)n) {
                float32x4_t v0 = vmulq_n_f32(c00, alpha);
                if (!initialize)
                    v0 = vaddq_f32(v0, vld1q_f32(C + i + 0 + (j + 0) * ldc));
                vst1q_f32(C + i + 0 + (j + 0) * ldc, v0);
            }
            if (j + 1 < (size_t)n) {
                float32x4_t v0 = vmulq_n_f32(c01, alpha);
                if (!initialize)
                    v0 = vaddq_f32(v0, vld1q_f32(C + i + 0 + (j + 1) * ldc));
                vst1q_f32(C + i + 0 + (j + 1) * ldc, v0);
            }
            if (j + 2 < (size_t)n) {
                float32x4_t v0 = vmulq_n_f32(c02, alpha);
                if (!initialize)
                    v0 = vaddq_f32(v0, vld1q_f32(C + i + 0 + (j + 2) * ldc));
                vst1q_f32(C + i + 0 + (j + 2) * ldc, v0);
            }
            if (j + 3 < (size_t)n) {
                float32x4_t v0 = vmulq_n_f32(c03, alpha);
                if (!initialize)
                    v0 = vaddq_f32(v0, vld1q_f32(C + i + 0 + (j + 3) * ldc));
                vst1q_f32(C + i + 0 + (j + 3) * ldc, v0);
            }
            if (j + 4 < (size_t)n) {
                float32x4_t v0 = vmulq_n_f32(c04, alpha);
                if (!initialize)
                    v0 = vaddq_f32(v0, vld1q_f32(C + i + 0 + (j + 4) * ldc));
                vst1q_f32(C + i + 0 + (j + 4) * ldc, v0);
            }
            if (j + 5 < (size_t)n) {
                float32x4_t v0 = vmulq_n_f32(c05, alpha);
                if (!initialize)
                    v0 = vaddq_f32(v0, vld1q_f32(C + i + 0 + (j + 5) * ldc));
                vst1q_f32(C + i + 0 + (j + 5) * ldc, v0);
            }
            if (j + 6 < (size_t)n) {
                float32x4_t v0 = vmulq_n_f32(c06, alpha);
                if (!initialize)
                    v0 = vaddq_f32(v0, vld1q_f32(C + i + 0 + (j + 6) * ldc));
                vst1q_f32(C + i + 0 + (j + 6) * ldc, v0);
            }
            if (j + 7 < (size_t)n) {
                float32x4_t v0 = vmulq_n_f32(c07, alpha);
                if (!initialize)
                    v0 = vaddq_f32(v0, vld1q_f32(C + i + 0 + (j + 7) * ldc));
                vst1q_f32(C + i + 0 + (j + 7) * ldc, v0);
            }
        }
        for (; i < (size_t)m; ++i)
            for (size_t q = 0; q < 8 && j + q < (size_t)n; ++q) {
                float sum = 0;
                for (size_t l = 0; l < (size_t)k; ++l)
                    sum += A[i + l * lda] * B[j * ldb + l * 8 + q];
                C[i + (j + q) * ldc] = alpha * sum + (initialize ? 0 : C[i + (j + q) * ldc]);
            }
    }
    return 0;
}
static inline int camblas_neon_micro8_f64(int m, int n, int k, double alpha, const double *A,
                                          size_t lda, const double *B, size_t ldb, double *C,
                                          size_t ldc, int initialize)
{
    for (size_t j = 0; j < (size_t)n; j += 8) {
        size_t i = 0;
        for (; i + 6 <= (size_t)m; i += 6) {
            float64x2_t c00 = vdupq_n_f64(0);
            float64x2_t c01 = vdupq_n_f64(0);
            float64x2_t c02 = vdupq_n_f64(0);
            float64x2_t c03 = vdupq_n_f64(0);
            float64x2_t c04 = vdupq_n_f64(0);
            float64x2_t c05 = vdupq_n_f64(0);
            float64x2_t c06 = vdupq_n_f64(0);
            float64x2_t c07 = vdupq_n_f64(0);
            float64x2_t c10 = vdupq_n_f64(0);
            float64x2_t c11 = vdupq_n_f64(0);
            float64x2_t c12 = vdupq_n_f64(0);
            float64x2_t c13 = vdupq_n_f64(0);
            float64x2_t c14 = vdupq_n_f64(0);
            float64x2_t c15 = vdupq_n_f64(0);
            float64x2_t c16 = vdupq_n_f64(0);
            float64x2_t c17 = vdupq_n_f64(0);
            float64x2_t c20 = vdupq_n_f64(0);
            float64x2_t c21 = vdupq_n_f64(0);
            float64x2_t c22 = vdupq_n_f64(0);
            float64x2_t c23 = vdupq_n_f64(0);
            float64x2_t c24 = vdupq_n_f64(0);
            float64x2_t c25 = vdupq_n_f64(0);
            float64x2_t c26 = vdupq_n_f64(0);
            float64x2_t c27 = vdupq_n_f64(0);
            for (size_t l = 0; l < (size_t)k; ++l) {
                float64x2_t a0 = vld1q_f64(A + i + 0 + l * lda);
                float64x2_t a1 = vld1q_f64(A + i + 2 + l * lda);
                float64x2_t a2 = vld1q_f64(A + i + 4 + l * lda);
                float64x2_t b0 = vld1q_f64(B + j * ldb + l * 8 + 0);
                float64x2_t b1 = vld1q_f64(B + j * ldb + l * 8 + 2);
                float64x2_t b2 = vld1q_f64(B + j * ldb + l * 8 + 4);
                float64x2_t b3 = vld1q_f64(B + j * ldb + l * 8 + 6);
                c00 = vfmaq_laneq_f64(c00, a0, b0, 0);
                c01 = vfmaq_laneq_f64(c01, a0, b0, 1);
                c02 = vfmaq_laneq_f64(c02, a0, b1, 0);
                c03 = vfmaq_laneq_f64(c03, a0, b1, 1);
                c04 = vfmaq_laneq_f64(c04, a0, b2, 0);
                c05 = vfmaq_laneq_f64(c05, a0, b2, 1);
                c06 = vfmaq_laneq_f64(c06, a0, b3, 0);
                c07 = vfmaq_laneq_f64(c07, a0, b3, 1);
                c10 = vfmaq_laneq_f64(c10, a1, b0, 0);
                c11 = vfmaq_laneq_f64(c11, a1, b0, 1);
                c12 = vfmaq_laneq_f64(c12, a1, b1, 0);
                c13 = vfmaq_laneq_f64(c13, a1, b1, 1);
                c14 = vfmaq_laneq_f64(c14, a1, b2, 0);
                c15 = vfmaq_laneq_f64(c15, a1, b2, 1);
                c16 = vfmaq_laneq_f64(c16, a1, b3, 0);
                c17 = vfmaq_laneq_f64(c17, a1, b3, 1);
                c20 = vfmaq_laneq_f64(c20, a2, b0, 0);
                c21 = vfmaq_laneq_f64(c21, a2, b0, 1);
                c22 = vfmaq_laneq_f64(c22, a2, b1, 0);
                c23 = vfmaq_laneq_f64(c23, a2, b1, 1);
                c24 = vfmaq_laneq_f64(c24, a2, b2, 0);
                c25 = vfmaq_laneq_f64(c25, a2, b2, 1);
                c26 = vfmaq_laneq_f64(c26, a2, b3, 0);
                c27 = vfmaq_laneq_f64(c27, a2, b3, 1);
            }
            if (j + 0 < (size_t)n) {
                float64x2_t v0 = vmulq_n_f64(c00, alpha);
                if (!initialize)
                    v0 = vaddq_f64(v0, vld1q_f64(C + i + 0 + (j + 0) * ldc));
                vst1q_f64(C + i + 0 + (j + 0) * ldc, v0);
                float64x2_t v1 = vmulq_n_f64(c10, alpha);
                if (!initialize)
                    v1 = vaddq_f64(v1, vld1q_f64(C + i + 2 + (j + 0) * ldc));
                vst1q_f64(C + i + 2 + (j + 0) * ldc, v1);
                float64x2_t v2 = vmulq_n_f64(c20, alpha);
                if (!initialize)
                    v2 = vaddq_f64(v2, vld1q_f64(C + i + 4 + (j + 0) * ldc));
                vst1q_f64(C + i + 4 + (j + 0) * ldc, v2);
            }
            if (j + 1 < (size_t)n) {
                float64x2_t v0 = vmulq_n_f64(c01, alpha);
                if (!initialize)
                    v0 = vaddq_f64(v0, vld1q_f64(C + i + 0 + (j + 1) * ldc));
                vst1q_f64(C + i + 0 + (j + 1) * ldc, v0);
                float64x2_t v1 = vmulq_n_f64(c11, alpha);
                if (!initialize)
                    v1 = vaddq_f64(v1, vld1q_f64(C + i + 2 + (j + 1) * ldc));
                vst1q_f64(C + i + 2 + (j + 1) * ldc, v1);
                float64x2_t v2 = vmulq_n_f64(c21, alpha);
                if (!initialize)
                    v2 = vaddq_f64(v2, vld1q_f64(C + i + 4 + (j + 1) * ldc));
                vst1q_f64(C + i + 4 + (j + 1) * ldc, v2);
            }
            if (j + 2 < (size_t)n) {
                float64x2_t v0 = vmulq_n_f64(c02, alpha);
                if (!initialize)
                    v0 = vaddq_f64(v0, vld1q_f64(C + i + 0 + (j + 2) * ldc));
                vst1q_f64(C + i + 0 + (j + 2) * ldc, v0);
                float64x2_t v1 = vmulq_n_f64(c12, alpha);
                if (!initialize)
                    v1 = vaddq_f64(v1, vld1q_f64(C + i + 2 + (j + 2) * ldc));
                vst1q_f64(C + i + 2 + (j + 2) * ldc, v1);
                float64x2_t v2 = vmulq_n_f64(c22, alpha);
                if (!initialize)
                    v2 = vaddq_f64(v2, vld1q_f64(C + i + 4 + (j + 2) * ldc));
                vst1q_f64(C + i + 4 + (j + 2) * ldc, v2);
            }
            if (j + 3 < (size_t)n) {
                float64x2_t v0 = vmulq_n_f64(c03, alpha);
                if (!initialize)
                    v0 = vaddq_f64(v0, vld1q_f64(C + i + 0 + (j + 3) * ldc));
                vst1q_f64(C + i + 0 + (j + 3) * ldc, v0);
                float64x2_t v1 = vmulq_n_f64(c13, alpha);
                if (!initialize)
                    v1 = vaddq_f64(v1, vld1q_f64(C + i + 2 + (j + 3) * ldc));
                vst1q_f64(C + i + 2 + (j + 3) * ldc, v1);
                float64x2_t v2 = vmulq_n_f64(c23, alpha);
                if (!initialize)
                    v2 = vaddq_f64(v2, vld1q_f64(C + i + 4 + (j + 3) * ldc));
                vst1q_f64(C + i + 4 + (j + 3) * ldc, v2);
            }
            if (j + 4 < (size_t)n) {
                float64x2_t v0 = vmulq_n_f64(c04, alpha);
                if (!initialize)
                    v0 = vaddq_f64(v0, vld1q_f64(C + i + 0 + (j + 4) * ldc));
                vst1q_f64(C + i + 0 + (j + 4) * ldc, v0);
                float64x2_t v1 = vmulq_n_f64(c14, alpha);
                if (!initialize)
                    v1 = vaddq_f64(v1, vld1q_f64(C + i + 2 + (j + 4) * ldc));
                vst1q_f64(C + i + 2 + (j + 4) * ldc, v1);
                float64x2_t v2 = vmulq_n_f64(c24, alpha);
                if (!initialize)
                    v2 = vaddq_f64(v2, vld1q_f64(C + i + 4 + (j + 4) * ldc));
                vst1q_f64(C + i + 4 + (j + 4) * ldc, v2);
            }
            if (j + 5 < (size_t)n) {
                float64x2_t v0 = vmulq_n_f64(c05, alpha);
                if (!initialize)
                    v0 = vaddq_f64(v0, vld1q_f64(C + i + 0 + (j + 5) * ldc));
                vst1q_f64(C + i + 0 + (j + 5) * ldc, v0);
                float64x2_t v1 = vmulq_n_f64(c15, alpha);
                if (!initialize)
                    v1 = vaddq_f64(v1, vld1q_f64(C + i + 2 + (j + 5) * ldc));
                vst1q_f64(C + i + 2 + (j + 5) * ldc, v1);
                float64x2_t v2 = vmulq_n_f64(c25, alpha);
                if (!initialize)
                    v2 = vaddq_f64(v2, vld1q_f64(C + i + 4 + (j + 5) * ldc));
                vst1q_f64(C + i + 4 + (j + 5) * ldc, v2);
            }
            if (j + 6 < (size_t)n) {
                float64x2_t v0 = vmulq_n_f64(c06, alpha);
                if (!initialize)
                    v0 = vaddq_f64(v0, vld1q_f64(C + i + 0 + (j + 6) * ldc));
                vst1q_f64(C + i + 0 + (j + 6) * ldc, v0);
                float64x2_t v1 = vmulq_n_f64(c16, alpha);
                if (!initialize)
                    v1 = vaddq_f64(v1, vld1q_f64(C + i + 2 + (j + 6) * ldc));
                vst1q_f64(C + i + 2 + (j + 6) * ldc, v1);
                float64x2_t v2 = vmulq_n_f64(c26, alpha);
                if (!initialize)
                    v2 = vaddq_f64(v2, vld1q_f64(C + i + 4 + (j + 6) * ldc));
                vst1q_f64(C + i + 4 + (j + 6) * ldc, v2);
            }
            if (j + 7 < (size_t)n) {
                float64x2_t v0 = vmulq_n_f64(c07, alpha);
                if (!initialize)
                    v0 = vaddq_f64(v0, vld1q_f64(C + i + 0 + (j + 7) * ldc));
                vst1q_f64(C + i + 0 + (j + 7) * ldc, v0);
                float64x2_t v1 = vmulq_n_f64(c17, alpha);
                if (!initialize)
                    v1 = vaddq_f64(v1, vld1q_f64(C + i + 2 + (j + 7) * ldc));
                vst1q_f64(C + i + 2 + (j + 7) * ldc, v1);
                float64x2_t v2 = vmulq_n_f64(c27, alpha);
                if (!initialize)
                    v2 = vaddq_f64(v2, vld1q_f64(C + i + 4 + (j + 7) * ldc));
                vst1q_f64(C + i + 4 + (j + 7) * ldc, v2);
            }
        }
        for (; i + 2 <= (size_t)m; i += 2) {
            float64x2_t c00 = vdupq_n_f64(0);
            float64x2_t c01 = vdupq_n_f64(0);
            float64x2_t c02 = vdupq_n_f64(0);
            float64x2_t c03 = vdupq_n_f64(0);
            float64x2_t c04 = vdupq_n_f64(0);
            float64x2_t c05 = vdupq_n_f64(0);
            float64x2_t c06 = vdupq_n_f64(0);
            float64x2_t c07 = vdupq_n_f64(0);
            for (size_t l = 0; l < (size_t)k; ++l) {
                float64x2_t a0 = vld1q_f64(A + i + 0 + l * lda);
                float64x2_t b0 = vld1q_f64(B + j * ldb + l * 8 + 0);
                float64x2_t b1 = vld1q_f64(B + j * ldb + l * 8 + 2);
                float64x2_t b2 = vld1q_f64(B + j * ldb + l * 8 + 4);
                float64x2_t b3 = vld1q_f64(B + j * ldb + l * 8 + 6);
                c00 = vfmaq_laneq_f64(c00, a0, b0, 0);
                c01 = vfmaq_laneq_f64(c01, a0, b0, 1);
                c02 = vfmaq_laneq_f64(c02, a0, b1, 0);
                c03 = vfmaq_laneq_f64(c03, a0, b1, 1);
                c04 = vfmaq_laneq_f64(c04, a0, b2, 0);
                c05 = vfmaq_laneq_f64(c05, a0, b2, 1);
                c06 = vfmaq_laneq_f64(c06, a0, b3, 0);
                c07 = vfmaq_laneq_f64(c07, a0, b3, 1);
            }
            if (j + 0 < (size_t)n) {
                float64x2_t v0 = vmulq_n_f64(c00, alpha);
                if (!initialize)
                    v0 = vaddq_f64(v0, vld1q_f64(C + i + 0 + (j + 0) * ldc));
                vst1q_f64(C + i + 0 + (j + 0) * ldc, v0);
            }
            if (j + 1 < (size_t)n) {
                float64x2_t v0 = vmulq_n_f64(c01, alpha);
                if (!initialize)
                    v0 = vaddq_f64(v0, vld1q_f64(C + i + 0 + (j + 1) * ldc));
                vst1q_f64(C + i + 0 + (j + 1) * ldc, v0);
            }
            if (j + 2 < (size_t)n) {
                float64x2_t v0 = vmulq_n_f64(c02, alpha);
                if (!initialize)
                    v0 = vaddq_f64(v0, vld1q_f64(C + i + 0 + (j + 2) * ldc));
                vst1q_f64(C + i + 0 + (j + 2) * ldc, v0);
            }
            if (j + 3 < (size_t)n) {
                float64x2_t v0 = vmulq_n_f64(c03, alpha);
                if (!initialize)
                    v0 = vaddq_f64(v0, vld1q_f64(C + i + 0 + (j + 3) * ldc));
                vst1q_f64(C + i + 0 + (j + 3) * ldc, v0);
            }
            if (j + 4 < (size_t)n) {
                float64x2_t v0 = vmulq_n_f64(c04, alpha);
                if (!initialize)
                    v0 = vaddq_f64(v0, vld1q_f64(C + i + 0 + (j + 4) * ldc));
                vst1q_f64(C + i + 0 + (j + 4) * ldc, v0);
            }
            if (j + 5 < (size_t)n) {
                float64x2_t v0 = vmulq_n_f64(c05, alpha);
                if (!initialize)
                    v0 = vaddq_f64(v0, vld1q_f64(C + i + 0 + (j + 5) * ldc));
                vst1q_f64(C + i + 0 + (j + 5) * ldc, v0);
            }
            if (j + 6 < (size_t)n) {
                float64x2_t v0 = vmulq_n_f64(c06, alpha);
                if (!initialize)
                    v0 = vaddq_f64(v0, vld1q_f64(C + i + 0 + (j + 6) * ldc));
                vst1q_f64(C + i + 0 + (j + 6) * ldc, v0);
            }
            if (j + 7 < (size_t)n) {
                float64x2_t v0 = vmulq_n_f64(c07, alpha);
                if (!initialize)
                    v0 = vaddq_f64(v0, vld1q_f64(C + i + 0 + (j + 7) * ldc));
                vst1q_f64(C + i + 0 + (j + 7) * ldc, v0);
            }
        }
        for (; i < (size_t)m; ++i)
            for (size_t q = 0; q < 8 && j + q < (size_t)n; ++q) {
                double sum = 0;
                for (size_t l = 0; l < (size_t)k; ++l)
                    sum += A[i + l * lda] * B[j * ldb + l * 8 + q];
                C[i + (j + q) * ldc] = alpha * sum + (initialize ? 0 : C[i + (j + q) * ldc]);
            }
    }
    return 0;
}
#endif
