#ifndef CAMBLAS_AMICRO_TRANSPOSE_H
#define CAMBLAS_AMICRO_TRANSPOSE_H
#include <stddef.h>
#include <string.h>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

/* The caller has checked the logical source span and the padded destination
 * capacity. Transpose directly into twelve-row A microgroups; no intermediate
 * panel is allocated. Only complete 4x4 source tiles use vector accesses. */
static inline void camblas_pack_transposed_amicro12(const float *source, size_t lda, size_t row0,
                                                    size_t depth0, size_t rows, size_t depth,
                                                    float *output)
{
    for (size_t group = 0; group < rows; group += 12) {
        size_t count = rows - group < 12 ? rows - group : 12;
        float *panel = output + group * depth;
        for (size_t kk = 0; kk < depth; kk += 16) {
            size_t end = depth - kk < 16 ? depth : kk + 16;
            size_t row = 0;
#if defined(__aarch64__)
            for (; row + 4 <= count; row += 4) {
                size_t k = kk;
                for (; k + 4 <= end; k += 4) {
                    const float *p = source + depth0 + k + (row0 + group + row) * lda;
                    float32x4_t a = vld1q_f32(p), b = vld1q_f32(p + lda);
                    float32x4_t c = vld1q_f32(p + 2 * lda), d = vld1q_f32(p + 3 * lda);
                    float64x2_t x = vreinterpretq_f64_f32(vtrn1q_f32(a, b));
                    float64x2_t y = vreinterpretq_f64_f32(vtrn2q_f32(a, b));
                    float64x2_t z = vreinterpretq_f64_f32(vtrn1q_f32(c, d));
                    float64x2_t t = vreinterpretq_f64_f32(vtrn2q_f32(c, d));
                    vst1q_f32(panel + row + k * 12, vreinterpretq_f32_f64(vzip1q_f64(x, z)));
                    vst1q_f32(panel + row + (k + 1) * 12, vreinterpretq_f32_f64(vzip1q_f64(y, t)));
                    vst1q_f32(panel + row + (k + 2) * 12, vreinterpretq_f32_f64(vzip2q_f64(x, z)));
                    vst1q_f32(panel + row + (k + 3) * 12, vreinterpretq_f32_f64(vzip2q_f64(y, t)));
                }
                for (; k < end; ++k)
                    for (size_t lane = 0; lane < 4; ++lane)
                        panel[row + lane + k * 12] =
                            source[depth0 + k + (row0 + group + row + lane) * lda];
            }
#endif
            for (; row < count; ++row)
                for (size_t k = kk; k < end; ++k)
                    panel[row + k * 12] = source[depth0 + k + (row0 + group + row) * lda];
        }
        if (count < 12)
            for (size_t k = 0; k < depth; ++k)
                memset(panel + count + k * 12, 0, (12 - count) * sizeof(float));
    }
}
#endif
