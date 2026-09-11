/* Private 12-column NEON experiment. Caller supplies zero-padded A/B
 * panels and a padded temporary C tile; no writes reach user C here. */
#ifndef CAMBLAS_NEON_MICRO12_H
#define CAMBLAS_NEON_MICRO12_H
#include <arm_neon.h>
#include <stddef.h>
static inline void camblas_neon_micro12_f32(int m, int n, int k, const float *a, size_t lda,
                                            const float *b, size_t ldb, float *c, size_t ldc,
                                            int initialize)
{
    for (int j = 0; j < n; j += 12)
        for (int i = 0; i < m; i += 8) {
            float32x4_t c0_0 = vdupq_n_f32(0);
            float32x4_t c1_0 = vdupq_n_f32(0);
            float32x4_t c0_1 = vdupq_n_f32(0);
            float32x4_t c1_1 = vdupq_n_f32(0);
            float32x4_t c0_2 = vdupq_n_f32(0);
            float32x4_t c1_2 = vdupq_n_f32(0);
            float32x4_t c0_3 = vdupq_n_f32(0);
            float32x4_t c1_3 = vdupq_n_f32(0);
            float32x4_t c0_4 = vdupq_n_f32(0);
            float32x4_t c1_4 = vdupq_n_f32(0);
            float32x4_t c0_5 = vdupq_n_f32(0);
            float32x4_t c1_5 = vdupq_n_f32(0);
            float32x4_t c0_6 = vdupq_n_f32(0);
            float32x4_t c1_6 = vdupq_n_f32(0);
            float32x4_t c0_7 = vdupq_n_f32(0);
            float32x4_t c1_7 = vdupq_n_f32(0);
            float32x4_t c0_8 = vdupq_n_f32(0);
            float32x4_t c1_8 = vdupq_n_f32(0);
            float32x4_t c0_9 = vdupq_n_f32(0);
            float32x4_t c1_9 = vdupq_n_f32(0);
            float32x4_t c0_10 = vdupq_n_f32(0);
            float32x4_t c1_10 = vdupq_n_f32(0);
            float32x4_t c0_11 = vdupq_n_f32(0);
            float32x4_t c1_11 = vdupq_n_f32(0);
            for (int q = 0; q < k; q++) {
                float32x4_t a0 = vld1q_f32(a + (size_t)q * lda + i);
                float32x4_t a1 = vld1q_f32(a + (size_t)q * lda + i + 4);
                float32x4_t b0 = vld1q_f32(b + (size_t)j * ldb + (size_t)q * 12 + 0);
                c0_0 = vfmaq_laneq_f32(c0_0, a0, b0, 0);
                c1_0 = vfmaq_laneq_f32(c1_0, a1, b0, 0);
                c0_1 = vfmaq_laneq_f32(c0_1, a0, b0, 1);
                c1_1 = vfmaq_laneq_f32(c1_1, a1, b0, 1);
                c0_2 = vfmaq_laneq_f32(c0_2, a0, b0, 2);
                c1_2 = vfmaq_laneq_f32(c1_2, a1, b0, 2);
                c0_3 = vfmaq_laneq_f32(c0_3, a0, b0, 3);
                c1_3 = vfmaq_laneq_f32(c1_3, a1, b0, 3);
                float32x4_t b4 = vld1q_f32(b + (size_t)j * ldb + (size_t)q * 12 + 4);
                c0_4 = vfmaq_laneq_f32(c0_4, a0, b4, 0);
                c1_4 = vfmaq_laneq_f32(c1_4, a1, b4, 0);
                c0_5 = vfmaq_laneq_f32(c0_5, a0, b4, 1);
                c1_5 = vfmaq_laneq_f32(c1_5, a1, b4, 1);
                c0_6 = vfmaq_laneq_f32(c0_6, a0, b4, 2);
                c1_6 = vfmaq_laneq_f32(c1_6, a1, b4, 2);
                c0_7 = vfmaq_laneq_f32(c0_7, a0, b4, 3);
                c1_7 = vfmaq_laneq_f32(c1_7, a1, b4, 3);
                float32x4_t b8 = vld1q_f32(b + (size_t)j * ldb + (size_t)q * 12 + 8);
                c0_8 = vfmaq_laneq_f32(c0_8, a0, b8, 0);
                c1_8 = vfmaq_laneq_f32(c1_8, a1, b8, 0);
                c0_9 = vfmaq_laneq_f32(c0_9, a0, b8, 1);
                c1_9 = vfmaq_laneq_f32(c1_9, a1, b8, 1);
                c0_10 = vfmaq_laneq_f32(c0_10, a0, b8, 2);
                c1_10 = vfmaq_laneq_f32(c1_10, a1, b8, 2);
                c0_11 = vfmaq_laneq_f32(c0_11, a0, b8, 3);
                c1_11 = vfmaq_laneq_f32(c1_11, a1, b8, 3);
            }
            if (!initialize)
                c0_0 = vaddq_f32(c0_0, vld1q_f32(c + (size_t)(j + 0) * ldc + i + 0));
            vst1q_f32(c + (size_t)(j + 0) * ldc + i + 0, c0_0);
            if (!initialize)
                c1_0 = vaddq_f32(c1_0, vld1q_f32(c + (size_t)(j + 0) * ldc + i + 4));
            vst1q_f32(c + (size_t)(j + 0) * ldc + i + 4, c1_0);
            if (!initialize)
                c0_1 = vaddq_f32(c0_1, vld1q_f32(c + (size_t)(j + 1) * ldc + i + 0));
            vst1q_f32(c + (size_t)(j + 1) * ldc + i + 0, c0_1);
            if (!initialize)
                c1_1 = vaddq_f32(c1_1, vld1q_f32(c + (size_t)(j + 1) * ldc + i + 4));
            vst1q_f32(c + (size_t)(j + 1) * ldc + i + 4, c1_1);
            if (!initialize)
                c0_2 = vaddq_f32(c0_2, vld1q_f32(c + (size_t)(j + 2) * ldc + i + 0));
            vst1q_f32(c + (size_t)(j + 2) * ldc + i + 0, c0_2);
            if (!initialize)
                c1_2 = vaddq_f32(c1_2, vld1q_f32(c + (size_t)(j + 2) * ldc + i + 4));
            vst1q_f32(c + (size_t)(j + 2) * ldc + i + 4, c1_2);
            if (!initialize)
                c0_3 = vaddq_f32(c0_3, vld1q_f32(c + (size_t)(j + 3) * ldc + i + 0));
            vst1q_f32(c + (size_t)(j + 3) * ldc + i + 0, c0_3);
            if (!initialize)
                c1_3 = vaddq_f32(c1_3, vld1q_f32(c + (size_t)(j + 3) * ldc + i + 4));
            vst1q_f32(c + (size_t)(j + 3) * ldc + i + 4, c1_3);
            if (!initialize)
                c0_4 = vaddq_f32(c0_4, vld1q_f32(c + (size_t)(j + 4) * ldc + i + 0));
            vst1q_f32(c + (size_t)(j + 4) * ldc + i + 0, c0_4);
            if (!initialize)
                c1_4 = vaddq_f32(c1_4, vld1q_f32(c + (size_t)(j + 4) * ldc + i + 4));
            vst1q_f32(c + (size_t)(j + 4) * ldc + i + 4, c1_4);
            if (!initialize)
                c0_5 = vaddq_f32(c0_5, vld1q_f32(c + (size_t)(j + 5) * ldc + i + 0));
            vst1q_f32(c + (size_t)(j + 5) * ldc + i + 0, c0_5);
            if (!initialize)
                c1_5 = vaddq_f32(c1_5, vld1q_f32(c + (size_t)(j + 5) * ldc + i + 4));
            vst1q_f32(c + (size_t)(j + 5) * ldc + i + 4, c1_5);
            if (!initialize)
                c0_6 = vaddq_f32(c0_6, vld1q_f32(c + (size_t)(j + 6) * ldc + i + 0));
            vst1q_f32(c + (size_t)(j + 6) * ldc + i + 0, c0_6);
            if (!initialize)
                c1_6 = vaddq_f32(c1_6, vld1q_f32(c + (size_t)(j + 6) * ldc + i + 4));
            vst1q_f32(c + (size_t)(j + 6) * ldc + i + 4, c1_6);
            if (!initialize)
                c0_7 = vaddq_f32(c0_7, vld1q_f32(c + (size_t)(j + 7) * ldc + i + 0));
            vst1q_f32(c + (size_t)(j + 7) * ldc + i + 0, c0_7);
            if (!initialize)
                c1_7 = vaddq_f32(c1_7, vld1q_f32(c + (size_t)(j + 7) * ldc + i + 4));
            vst1q_f32(c + (size_t)(j + 7) * ldc + i + 4, c1_7);
            if (!initialize)
                c0_8 = vaddq_f32(c0_8, vld1q_f32(c + (size_t)(j + 8) * ldc + i + 0));
            vst1q_f32(c + (size_t)(j + 8) * ldc + i + 0, c0_8);
            if (!initialize)
                c1_8 = vaddq_f32(c1_8, vld1q_f32(c + (size_t)(j + 8) * ldc + i + 4));
            vst1q_f32(c + (size_t)(j + 8) * ldc + i + 4, c1_8);
            if (!initialize)
                c0_9 = vaddq_f32(c0_9, vld1q_f32(c + (size_t)(j + 9) * ldc + i + 0));
            vst1q_f32(c + (size_t)(j + 9) * ldc + i + 0, c0_9);
            if (!initialize)
                c1_9 = vaddq_f32(c1_9, vld1q_f32(c + (size_t)(j + 9) * ldc + i + 4));
            vst1q_f32(c + (size_t)(j + 9) * ldc + i + 4, c1_9);
            if (!initialize)
                c0_10 = vaddq_f32(c0_10, vld1q_f32(c + (size_t)(j + 10) * ldc + i + 0));
            vst1q_f32(c + (size_t)(j + 10) * ldc + i + 0, c0_10);
            if (!initialize)
                c1_10 = vaddq_f32(c1_10, vld1q_f32(c + (size_t)(j + 10) * ldc + i + 4));
            vst1q_f32(c + (size_t)(j + 10) * ldc + i + 4, c1_10);
            if (!initialize)
                c0_11 = vaddq_f32(c0_11, vld1q_f32(c + (size_t)(j + 11) * ldc + i + 0));
            vst1q_f32(c + (size_t)(j + 11) * ldc + i + 0, c0_11);
            if (!initialize)
                c1_11 = vaddq_f32(c1_11, vld1q_f32(c + (size_t)(j + 11) * ldc + i + 4));
            vst1q_f32(c + (size_t)(j + 11) * ldc + i + 4, c1_11);
        }
}
static inline void camblas_neon_micro12_f64(int m, int n, int k, const double *a, size_t lda,
                                            const double *b, size_t ldb, double *c, size_t ldc,
                                            int initialize)
{
    for (int j = 0; j < n; j += 12)
        for (int i = 0; i < m; i += 4) {
            float64x2_t c0_0 = vdupq_n_f64(0);
            float64x2_t c1_0 = vdupq_n_f64(0);
            float64x2_t c0_1 = vdupq_n_f64(0);
            float64x2_t c1_1 = vdupq_n_f64(0);
            float64x2_t c0_2 = vdupq_n_f64(0);
            float64x2_t c1_2 = vdupq_n_f64(0);
            float64x2_t c0_3 = vdupq_n_f64(0);
            float64x2_t c1_3 = vdupq_n_f64(0);
            float64x2_t c0_4 = vdupq_n_f64(0);
            float64x2_t c1_4 = vdupq_n_f64(0);
            float64x2_t c0_5 = vdupq_n_f64(0);
            float64x2_t c1_5 = vdupq_n_f64(0);
            float64x2_t c0_6 = vdupq_n_f64(0);
            float64x2_t c1_6 = vdupq_n_f64(0);
            float64x2_t c0_7 = vdupq_n_f64(0);
            float64x2_t c1_7 = vdupq_n_f64(0);
            float64x2_t c0_8 = vdupq_n_f64(0);
            float64x2_t c1_8 = vdupq_n_f64(0);
            float64x2_t c0_9 = vdupq_n_f64(0);
            float64x2_t c1_9 = vdupq_n_f64(0);
            float64x2_t c0_10 = vdupq_n_f64(0);
            float64x2_t c1_10 = vdupq_n_f64(0);
            float64x2_t c0_11 = vdupq_n_f64(0);
            float64x2_t c1_11 = vdupq_n_f64(0);
            for (int q = 0; q < k; q++) {
                float64x2_t a0 = vld1q_f64(a + (size_t)q * lda + i);
                float64x2_t a1 = vld1q_f64(a + (size_t)q * lda + i + 2);
                float64x2_t b0 = vld1q_f64(b + (size_t)j * ldb + (size_t)q * 12 + 0);
                c0_0 = vfmaq_laneq_f64(c0_0, a0, b0, 0);
                c1_0 = vfmaq_laneq_f64(c1_0, a1, b0, 0);
                c0_1 = vfmaq_laneq_f64(c0_1, a0, b0, 1);
                c1_1 = vfmaq_laneq_f64(c1_1, a1, b0, 1);
                float64x2_t b2 = vld1q_f64(b + (size_t)j * ldb + (size_t)q * 12 + 2);
                c0_2 = vfmaq_laneq_f64(c0_2, a0, b2, 0);
                c1_2 = vfmaq_laneq_f64(c1_2, a1, b2, 0);
                c0_3 = vfmaq_laneq_f64(c0_3, a0, b2, 1);
                c1_3 = vfmaq_laneq_f64(c1_3, a1, b2, 1);
                float64x2_t b4 = vld1q_f64(b + (size_t)j * ldb + (size_t)q * 12 + 4);
                c0_4 = vfmaq_laneq_f64(c0_4, a0, b4, 0);
                c1_4 = vfmaq_laneq_f64(c1_4, a1, b4, 0);
                c0_5 = vfmaq_laneq_f64(c0_5, a0, b4, 1);
                c1_5 = vfmaq_laneq_f64(c1_5, a1, b4, 1);
                float64x2_t b6 = vld1q_f64(b + (size_t)j * ldb + (size_t)q * 12 + 6);
                c0_6 = vfmaq_laneq_f64(c0_6, a0, b6, 0);
                c1_6 = vfmaq_laneq_f64(c1_6, a1, b6, 0);
                c0_7 = vfmaq_laneq_f64(c0_7, a0, b6, 1);
                c1_7 = vfmaq_laneq_f64(c1_7, a1, b6, 1);
                float64x2_t b8 = vld1q_f64(b + (size_t)j * ldb + (size_t)q * 12 + 8);
                c0_8 = vfmaq_laneq_f64(c0_8, a0, b8, 0);
                c1_8 = vfmaq_laneq_f64(c1_8, a1, b8, 0);
                c0_9 = vfmaq_laneq_f64(c0_9, a0, b8, 1);
                c1_9 = vfmaq_laneq_f64(c1_9, a1, b8, 1);
                float64x2_t b10 = vld1q_f64(b + (size_t)j * ldb + (size_t)q * 12 + 10);
                c0_10 = vfmaq_laneq_f64(c0_10, a0, b10, 0);
                c1_10 = vfmaq_laneq_f64(c1_10, a1, b10, 0);
                c0_11 = vfmaq_laneq_f64(c0_11, a0, b10, 1);
                c1_11 = vfmaq_laneq_f64(c1_11, a1, b10, 1);
            }
            if (!initialize)
                c0_0 = vaddq_f64(c0_0, vld1q_f64(c + (size_t)(j + 0) * ldc + i + 0));
            vst1q_f64(c + (size_t)(j + 0) * ldc + i + 0, c0_0);
            if (!initialize)
                c1_0 = vaddq_f64(c1_0, vld1q_f64(c + (size_t)(j + 0) * ldc + i + 2));
            vst1q_f64(c + (size_t)(j + 0) * ldc + i + 2, c1_0);
            if (!initialize)
                c0_1 = vaddq_f64(c0_1, vld1q_f64(c + (size_t)(j + 1) * ldc + i + 0));
            vst1q_f64(c + (size_t)(j + 1) * ldc + i + 0, c0_1);
            if (!initialize)
                c1_1 = vaddq_f64(c1_1, vld1q_f64(c + (size_t)(j + 1) * ldc + i + 2));
            vst1q_f64(c + (size_t)(j + 1) * ldc + i + 2, c1_1);
            if (!initialize)
                c0_2 = vaddq_f64(c0_2, vld1q_f64(c + (size_t)(j + 2) * ldc + i + 0));
            vst1q_f64(c + (size_t)(j + 2) * ldc + i + 0, c0_2);
            if (!initialize)
                c1_2 = vaddq_f64(c1_2, vld1q_f64(c + (size_t)(j + 2) * ldc + i + 2));
            vst1q_f64(c + (size_t)(j + 2) * ldc + i + 2, c1_2);
            if (!initialize)
                c0_3 = vaddq_f64(c0_3, vld1q_f64(c + (size_t)(j + 3) * ldc + i + 0));
            vst1q_f64(c + (size_t)(j + 3) * ldc + i + 0, c0_3);
            if (!initialize)
                c1_3 = vaddq_f64(c1_3, vld1q_f64(c + (size_t)(j + 3) * ldc + i + 2));
            vst1q_f64(c + (size_t)(j + 3) * ldc + i + 2, c1_3);
            if (!initialize)
                c0_4 = vaddq_f64(c0_4, vld1q_f64(c + (size_t)(j + 4) * ldc + i + 0));
            vst1q_f64(c + (size_t)(j + 4) * ldc + i + 0, c0_4);
            if (!initialize)
                c1_4 = vaddq_f64(c1_4, vld1q_f64(c + (size_t)(j + 4) * ldc + i + 2));
            vst1q_f64(c + (size_t)(j + 4) * ldc + i + 2, c1_4);
            if (!initialize)
                c0_5 = vaddq_f64(c0_5, vld1q_f64(c + (size_t)(j + 5) * ldc + i + 0));
            vst1q_f64(c + (size_t)(j + 5) * ldc + i + 0, c0_5);
            if (!initialize)
                c1_5 = vaddq_f64(c1_5, vld1q_f64(c + (size_t)(j + 5) * ldc + i + 2));
            vst1q_f64(c + (size_t)(j + 5) * ldc + i + 2, c1_5);
            if (!initialize)
                c0_6 = vaddq_f64(c0_6, vld1q_f64(c + (size_t)(j + 6) * ldc + i + 0));
            vst1q_f64(c + (size_t)(j + 6) * ldc + i + 0, c0_6);
            if (!initialize)
                c1_6 = vaddq_f64(c1_6, vld1q_f64(c + (size_t)(j + 6) * ldc + i + 2));
            vst1q_f64(c + (size_t)(j + 6) * ldc + i + 2, c1_6);
            if (!initialize)
                c0_7 = vaddq_f64(c0_7, vld1q_f64(c + (size_t)(j + 7) * ldc + i + 0));
            vst1q_f64(c + (size_t)(j + 7) * ldc + i + 0, c0_7);
            if (!initialize)
                c1_7 = vaddq_f64(c1_7, vld1q_f64(c + (size_t)(j + 7) * ldc + i + 2));
            vst1q_f64(c + (size_t)(j + 7) * ldc + i + 2, c1_7);
            if (!initialize)
                c0_8 = vaddq_f64(c0_8, vld1q_f64(c + (size_t)(j + 8) * ldc + i + 0));
            vst1q_f64(c + (size_t)(j + 8) * ldc + i + 0, c0_8);
            if (!initialize)
                c1_8 = vaddq_f64(c1_8, vld1q_f64(c + (size_t)(j + 8) * ldc + i + 2));
            vst1q_f64(c + (size_t)(j + 8) * ldc + i + 2, c1_8);
            if (!initialize)
                c0_9 = vaddq_f64(c0_9, vld1q_f64(c + (size_t)(j + 9) * ldc + i + 0));
            vst1q_f64(c + (size_t)(j + 9) * ldc + i + 0, c0_9);
            if (!initialize)
                c1_9 = vaddq_f64(c1_9, vld1q_f64(c + (size_t)(j + 9) * ldc + i + 2));
            vst1q_f64(c + (size_t)(j + 9) * ldc + i + 2, c1_9);
            if (!initialize)
                c0_10 = vaddq_f64(c0_10, vld1q_f64(c + (size_t)(j + 10) * ldc + i + 0));
            vst1q_f64(c + (size_t)(j + 10) * ldc + i + 0, c0_10);
            if (!initialize)
                c1_10 = vaddq_f64(c1_10, vld1q_f64(c + (size_t)(j + 10) * ldc + i + 2));
            vst1q_f64(c + (size_t)(j + 10) * ldc + i + 2, c1_10);
            if (!initialize)
                c0_11 = vaddq_f64(c0_11, vld1q_f64(c + (size_t)(j + 11) * ldc + i + 0));
            vst1q_f64(c + (size_t)(j + 11) * ldc + i + 0, c0_11);
            if (!initialize)
                c1_11 = vaddq_f64(c1_11, vld1q_f64(c + (size_t)(j + 11) * ldc + i + 2));
            vst1q_f64(c + (size_t)(j + 11) * ldc + i + 2, c1_11);
        }
}
#endif
