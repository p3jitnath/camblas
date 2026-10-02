/* Bounded Grace FP32 path. Inputs are packed afresh on every call.
 * A uses eight-row micro-panels and B uses twelve-column micro-panels.
 * A private pool or a reused OpenMP team owns one pack/compute batch;
 * a one-shot barrier also orders output page preparation before C writes.
 * Dispatch excludes nested calls and shapes outside the bounded task grid. */
#ifndef CAMBLAS_FRAMEWORK_SHARED_GRID32_H
#define CAMBLAS_FRAMEWORK_SHARED_GRID32_H
#include <arm_neon.h>
static inline void shared32_full(int depth, const float *a, const float *b, float *c, int ldc,
                                 int first)
{
    const float *ap = a, *bp = b;
    float *cp = c;
    uintptr_t stride = (uintptr_t)ldc * sizeof(float);
    __asm__ volatile("movi v8.16b, #0\n\t"
                     "movi v9.16b, #0\n\t"
                     "movi v10.16b, #0\n\t"
                     "movi v11.16b, #0\n\t"
                     "movi v12.16b, #0\n\t"
                     "movi v13.16b, #0\n\t"
                     "movi v14.16b, #0\n\t"
                     "movi v15.16b, #0\n\t"
                     "movi v16.16b, #0\n\t"
                     "movi v17.16b, #0\n\t"
                     "movi v18.16b, #0\n\t"
                     "movi v19.16b, #0\n\t"
                     "movi v20.16b, #0\n\t"
                     "movi v21.16b, #0\n\t"
                     "movi v22.16b, #0\n\t"
                     "movi v23.16b, #0\n\t"
                     "movi v24.16b, #0\n\t"
                     "movi v25.16b, #0\n\t"
                     "movi v26.16b, #0\n\t"
                     "movi v27.16b, #0\n\t"
                     "movi v28.16b, #0\n\t"
                     "movi v29.16b, #0\n\t"
                     "movi v30.16b, #0\n\t"
                     "movi v31.16b, #0\n\t"
                     "prfm pstl1keep, [%[cp]]\n\t"
                     "add %[cp], %[cp], %[stride]\n\t"
                     "prfm pstl1keep, [%[cp]]\n\t"
                     "add %[cp], %[cp], %[stride]\n\t"
                     "prfm pstl1keep, [%[cp]]\n\t"
                     "add %[cp], %[cp], %[stride]\n\t"
                     "prfm pstl1keep, [%[cp]]\n\t"
                     "add %[cp], %[cp], %[stride]\n\t"
                     "prfm pstl1keep, [%[cp]]\n\t"
                     "add %[cp], %[cp], %[stride]\n\t"
                     "prfm pstl1keep, [%[cp]]\n\t"
                     "add %[cp], %[cp], %[stride]\n\t"
                     "prfm pstl1keep, [%[cp]]\n\t"
                     "add %[cp], %[cp], %[stride]\n\t"
                     "prfm pstl1keep, [%[cp]]\n\t"
                     "add %[cp], %[cp], %[stride]\n\t"
                     "prfm pstl1keep, [%[cp]]\n\t"
                     "add %[cp], %[cp], %[stride]\n\t"
                     "prfm pstl1keep, [%[cp]]\n\t"
                     "add %[cp], %[cp], %[stride]\n\t"
                     "prfm pstl1keep, [%[cp]]\n\t"
                     "add %[cp], %[cp], %[stride]\n\t"
                     "prfm pstl1keep, [%[cp]]\n\t"
                     "add %[cp], %[cp], %[stride]\n\t"
                     "mov %[cp], %[base]\n\t"
                     "cmp %w[depth], #4\n\t"
                     "b.lt 2f\n\t"
                     "1:\n\t"
                     "cmp %w[depth], #36\n\t"
                     "b.lt 7f\n\t"
                     "prfm pldl1keep, [%[ap], #1024]\n\t"
                     "prfm pldl1keep, [%[ap], #1088]\n\t"
                     "prfm pldl1keep, [%[bp], #1536]\n\t"
                     "prfm pldl1keep, [%[bp], #1600]\n\t"
                     "prfm pldl1keep, [%[bp], #1664]\n\t"
                     "7:\n\t"
                     "ld1 {v0.4s-v1.4s}, [%[ap]], #32\n\t"
                     "ld1 {v2.4s-v4.4s}, [%[bp]], #48\n\t"
                     "fmla v8.4s, v0.4s, v2.s[0]\n\t"
                     "fmla v9.4s, v1.4s, v2.s[0]\n\t"
                     "fmla v10.4s, v0.4s, v2.s[1]\n\t"
                     "fmla v11.4s, v1.4s, v2.s[1]\n\t"
                     "fmla v12.4s, v0.4s, v2.s[2]\n\t"
                     "fmla v13.4s, v1.4s, v2.s[2]\n\t"
                     "fmla v14.4s, v0.4s, v2.s[3]\n\t"
                     "fmla v15.4s, v1.4s, v2.s[3]\n\t"
                     "fmla v16.4s, v0.4s, v3.s[0]\n\t"
                     "fmla v17.4s, v1.4s, v3.s[0]\n\t"
                     "fmla v18.4s, v0.4s, v3.s[1]\n\t"
                     "fmla v19.4s, v1.4s, v3.s[1]\n\t"
                     "fmla v20.4s, v0.4s, v3.s[2]\n\t"
                     "fmla v21.4s, v1.4s, v3.s[2]\n\t"
                     "fmla v22.4s, v0.4s, v3.s[3]\n\t"
                     "fmla v23.4s, v1.4s, v3.s[3]\n\t"
                     "fmla v24.4s, v0.4s, v4.s[0]\n\t"
                     "fmla v25.4s, v1.4s, v4.s[0]\n\t"
                     "fmla v26.4s, v0.4s, v4.s[1]\n\t"
                     "fmla v27.4s, v1.4s, v4.s[1]\n\t"
                     "fmla v28.4s, v0.4s, v4.s[2]\n\t"
                     "fmla v29.4s, v1.4s, v4.s[2]\n\t"
                     "fmla v30.4s, v0.4s, v4.s[3]\n\t"
                     "fmla v31.4s, v1.4s, v4.s[3]\n\t"
                     "ld1 {v0.4s-v1.4s}, [%[ap]], #32\n\t"
                     "ld1 {v2.4s-v4.4s}, [%[bp]], #48\n\t"
                     "fmla v8.4s, v0.4s, v2.s[0]\n\t"
                     "fmla v9.4s, v1.4s, v2.s[0]\n\t"
                     "fmla v10.4s, v0.4s, v2.s[1]\n\t"
                     "fmla v11.4s, v1.4s, v2.s[1]\n\t"
                     "fmla v12.4s, v0.4s, v2.s[2]\n\t"
                     "fmla v13.4s, v1.4s, v2.s[2]\n\t"
                     "fmla v14.4s, v0.4s, v2.s[3]\n\t"
                     "fmla v15.4s, v1.4s, v2.s[3]\n\t"
                     "fmla v16.4s, v0.4s, v3.s[0]\n\t"
                     "fmla v17.4s, v1.4s, v3.s[0]\n\t"
                     "fmla v18.4s, v0.4s, v3.s[1]\n\t"
                     "fmla v19.4s, v1.4s, v3.s[1]\n\t"
                     "fmla v20.4s, v0.4s, v3.s[2]\n\t"
                     "fmla v21.4s, v1.4s, v3.s[2]\n\t"
                     "fmla v22.4s, v0.4s, v3.s[3]\n\t"
                     "fmla v23.4s, v1.4s, v3.s[3]\n\t"
                     "fmla v24.4s, v0.4s, v4.s[0]\n\t"
                     "fmla v25.4s, v1.4s, v4.s[0]\n\t"
                     "fmla v26.4s, v0.4s, v4.s[1]\n\t"
                     "fmla v27.4s, v1.4s, v4.s[1]\n\t"
                     "fmla v28.4s, v0.4s, v4.s[2]\n\t"
                     "fmla v29.4s, v1.4s, v4.s[2]\n\t"
                     "fmla v30.4s, v0.4s, v4.s[3]\n\t"
                     "fmla v31.4s, v1.4s, v4.s[3]\n\t"
                     "ld1 {v0.4s-v1.4s}, [%[ap]], #32\n\t"
                     "ld1 {v2.4s-v4.4s}, [%[bp]], #48\n\t"
                     "fmla v8.4s, v0.4s, v2.s[0]\n\t"
                     "fmla v9.4s, v1.4s, v2.s[0]\n\t"
                     "fmla v10.4s, v0.4s, v2.s[1]\n\t"
                     "fmla v11.4s, v1.4s, v2.s[1]\n\t"
                     "fmla v12.4s, v0.4s, v2.s[2]\n\t"
                     "fmla v13.4s, v1.4s, v2.s[2]\n\t"
                     "fmla v14.4s, v0.4s, v2.s[3]\n\t"
                     "fmla v15.4s, v1.4s, v2.s[3]\n\t"
                     "fmla v16.4s, v0.4s, v3.s[0]\n\t"
                     "fmla v17.4s, v1.4s, v3.s[0]\n\t"
                     "fmla v18.4s, v0.4s, v3.s[1]\n\t"
                     "fmla v19.4s, v1.4s, v3.s[1]\n\t"
                     "fmla v20.4s, v0.4s, v3.s[2]\n\t"
                     "fmla v21.4s, v1.4s, v3.s[2]\n\t"
                     "fmla v22.4s, v0.4s, v3.s[3]\n\t"
                     "fmla v23.4s, v1.4s, v3.s[3]\n\t"
                     "fmla v24.4s, v0.4s, v4.s[0]\n\t"
                     "fmla v25.4s, v1.4s, v4.s[0]\n\t"
                     "fmla v26.4s, v0.4s, v4.s[1]\n\t"
                     "fmla v27.4s, v1.4s, v4.s[1]\n\t"
                     "fmla v28.4s, v0.4s, v4.s[2]\n\t"
                     "fmla v29.4s, v1.4s, v4.s[2]\n\t"
                     "fmla v30.4s, v0.4s, v4.s[3]\n\t"
                     "fmla v31.4s, v1.4s, v4.s[3]\n\t"
                     "ld1 {v0.4s-v1.4s}, [%[ap]], #32\n\t"
                     "ld1 {v2.4s-v4.4s}, [%[bp]], #48\n\t"
                     "fmla v8.4s, v0.4s, v2.s[0]\n\t"
                     "fmla v9.4s, v1.4s, v2.s[0]\n\t"
                     "fmla v10.4s, v0.4s, v2.s[1]\n\t"
                     "fmla v11.4s, v1.4s, v2.s[1]\n\t"
                     "fmla v12.4s, v0.4s, v2.s[2]\n\t"
                     "fmla v13.4s, v1.4s, v2.s[2]\n\t"
                     "fmla v14.4s, v0.4s, v2.s[3]\n\t"
                     "fmla v15.4s, v1.4s, v2.s[3]\n\t"
                     "fmla v16.4s, v0.4s, v3.s[0]\n\t"
                     "fmla v17.4s, v1.4s, v3.s[0]\n\t"
                     "fmla v18.4s, v0.4s, v3.s[1]\n\t"
                     "fmla v19.4s, v1.4s, v3.s[1]\n\t"
                     "fmla v20.4s, v0.4s, v3.s[2]\n\t"
                     "fmla v21.4s, v1.4s, v3.s[2]\n\t"
                     "fmla v22.4s, v0.4s, v3.s[3]\n\t"
                     "fmla v23.4s, v1.4s, v3.s[3]\n\t"
                     "fmla v24.4s, v0.4s, v4.s[0]\n\t"
                     "fmla v25.4s, v1.4s, v4.s[0]\n\t"
                     "fmla v26.4s, v0.4s, v4.s[1]\n\t"
                     "fmla v27.4s, v1.4s, v4.s[1]\n\t"
                     "fmla v28.4s, v0.4s, v4.s[2]\n\t"
                     "fmla v29.4s, v1.4s, v4.s[2]\n\t"
                     "fmla v30.4s, v0.4s, v4.s[3]\n\t"
                     "fmla v31.4s, v1.4s, v4.s[3]\n\t"
                     "sub %w[depth], %w[depth], #4\n\t"
                     "cmp %w[depth], #4\n\t"
                     "b.ge 1b\n\t"
                     "2:\n\t"
                     "cbz %w[depth], 4f\n\t"
                     "3:\n\t"
                     "ld1 {v0.4s-v1.4s}, [%[ap]], #32\n\t"
                     "ld1 {v2.4s-v4.4s}, [%[bp]], #48\n\t"
                     "fmla v8.4s, v0.4s, v2.s[0]\n\t"
                     "fmla v9.4s, v1.4s, v2.s[0]\n\t"
                     "fmla v10.4s, v0.4s, v2.s[1]\n\t"
                     "fmla v11.4s, v1.4s, v2.s[1]\n\t"
                     "fmla v12.4s, v0.4s, v2.s[2]\n\t"
                     "fmla v13.4s, v1.4s, v2.s[2]\n\t"
                     "fmla v14.4s, v0.4s, v2.s[3]\n\t"
                     "fmla v15.4s, v1.4s, v2.s[3]\n\t"
                     "fmla v16.4s, v0.4s, v3.s[0]\n\t"
                     "fmla v17.4s, v1.4s, v3.s[0]\n\t"
                     "fmla v18.4s, v0.4s, v3.s[1]\n\t"
                     "fmla v19.4s, v1.4s, v3.s[1]\n\t"
                     "fmla v20.4s, v0.4s, v3.s[2]\n\t"
                     "fmla v21.4s, v1.4s, v3.s[2]\n\t"
                     "fmla v22.4s, v0.4s, v3.s[3]\n\t"
                     "fmla v23.4s, v1.4s, v3.s[3]\n\t"
                     "fmla v24.4s, v0.4s, v4.s[0]\n\t"
                     "fmla v25.4s, v1.4s, v4.s[0]\n\t"
                     "fmla v26.4s, v0.4s, v4.s[1]\n\t"
                     "fmla v27.4s, v1.4s, v4.s[1]\n\t"
                     "fmla v28.4s, v0.4s, v4.s[2]\n\t"
                     "fmla v29.4s, v1.4s, v4.s[2]\n\t"
                     "fmla v30.4s, v0.4s, v4.s[3]\n\t"
                     "fmla v31.4s, v1.4s, v4.s[3]\n\t"
                     "subs %w[depth], %w[depth], #1\n\t"
                     "b.ne 3b\n\t"
                     "4:\n\t"
                     "cbnz %w[first], 6f\n\t"
                     "ldp q0, q1, [%[cp]]\n\t"
                     "fadd v8.4s, v8.4s, v0.4s\n\t"
                     "fadd v9.4s, v9.4s, v1.4s\n\t"
                     "add %[cp], %[cp], %[stride]\n\t"
                     "ldp q0, q1, [%[cp]]\n\t"
                     "fadd v10.4s, v10.4s, v0.4s\n\t"
                     "fadd v11.4s, v11.4s, v1.4s\n\t"
                     "add %[cp], %[cp], %[stride]\n\t"
                     "ldp q0, q1, [%[cp]]\n\t"
                     "fadd v12.4s, v12.4s, v0.4s\n\t"
                     "fadd v13.4s, v13.4s, v1.4s\n\t"
                     "add %[cp], %[cp], %[stride]\n\t"
                     "ldp q0, q1, [%[cp]]\n\t"
                     "fadd v14.4s, v14.4s, v0.4s\n\t"
                     "fadd v15.4s, v15.4s, v1.4s\n\t"
                     "add %[cp], %[cp], %[stride]\n\t"
                     "ldp q0, q1, [%[cp]]\n\t"
                     "fadd v16.4s, v16.4s, v0.4s\n\t"
                     "fadd v17.4s, v17.4s, v1.4s\n\t"
                     "add %[cp], %[cp], %[stride]\n\t"
                     "ldp q0, q1, [%[cp]]\n\t"
                     "fadd v18.4s, v18.4s, v0.4s\n\t"
                     "fadd v19.4s, v19.4s, v1.4s\n\t"
                     "add %[cp], %[cp], %[stride]\n\t"
                     "ldp q0, q1, [%[cp]]\n\t"
                     "fadd v20.4s, v20.4s, v0.4s\n\t"
                     "fadd v21.4s, v21.4s, v1.4s\n\t"
                     "add %[cp], %[cp], %[stride]\n\t"
                     "ldp q0, q1, [%[cp]]\n\t"
                     "fadd v22.4s, v22.4s, v0.4s\n\t"
                     "fadd v23.4s, v23.4s, v1.4s\n\t"
                     "add %[cp], %[cp], %[stride]\n\t"
                     "ldp q0, q1, [%[cp]]\n\t"
                     "fadd v24.4s, v24.4s, v0.4s\n\t"
                     "fadd v25.4s, v25.4s, v1.4s\n\t"
                     "add %[cp], %[cp], %[stride]\n\t"
                     "ldp q0, q1, [%[cp]]\n\t"
                     "fadd v26.4s, v26.4s, v0.4s\n\t"
                     "fadd v27.4s, v27.4s, v1.4s\n\t"
                     "add %[cp], %[cp], %[stride]\n\t"
                     "ldp q0, q1, [%[cp]]\n\t"
                     "fadd v28.4s, v28.4s, v0.4s\n\t"
                     "fadd v29.4s, v29.4s, v1.4s\n\t"
                     "add %[cp], %[cp], %[stride]\n\t"
                     "ldp q0, q1, [%[cp]]\n\t"
                     "fadd v30.4s, v30.4s, v0.4s\n\t"
                     "fadd v31.4s, v31.4s, v1.4s\n\t"
                     "add %[cp], %[cp], %[stride]\n\t"
                     "mov %[cp], %[base]\n\t"
                     "6:\n\t"
                     "stp q8, q9, [%[cp]]\n\t"
                     "add %[cp], %[cp], %[stride]\n\t"
                     "stp q10, q11, [%[cp]]\n\t"
                     "add %[cp], %[cp], %[stride]\n\t"
                     "stp q12, q13, [%[cp]]\n\t"
                     "add %[cp], %[cp], %[stride]\n\t"
                     "stp q14, q15, [%[cp]]\n\t"
                     "add %[cp], %[cp], %[stride]\n\t"
                     "stp q16, q17, [%[cp]]\n\t"
                     "add %[cp], %[cp], %[stride]\n\t"
                     "stp q18, q19, [%[cp]]\n\t"
                     "add %[cp], %[cp], %[stride]\n\t"
                     "stp q20, q21, [%[cp]]\n\t"
                     "add %[cp], %[cp], %[stride]\n\t"
                     "stp q22, q23, [%[cp]]\n\t"
                     "add %[cp], %[cp], %[stride]\n\t"
                     "stp q24, q25, [%[cp]]\n\t"
                     "add %[cp], %[cp], %[stride]\n\t"
                     "stp q26, q27, [%[cp]]\n\t"
                     "add %[cp], %[cp], %[stride]\n\t"
                     "stp q28, q29, [%[cp]]\n\t"
                     "add %[cp], %[cp], %[stride]\n\t"
                     "stp q30, q31, [%[cp]]\n\t"
                     "add %[cp], %[cp], %[stride]\n\t"
                     : [ap] "+&r"(ap), [bp] "+&r"(bp), [cp] "+&r"(cp), [depth] "+&r"(depth)
                     : [stride] "r"(stride), [first] "r"(first), [base] "r"(c)
                     : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                       "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
                       "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
}
static inline void shared32_tail4(int depth, const float *a, const float *b, float *c, int ldc,
                                  int first)
{
    float32x4_t c0_0 = vdupq_n_f32(0);
    float32x4_t c0_1 = vdupq_n_f32(0);
    float32x4_t c1_0 = vdupq_n_f32(0);
    float32x4_t c1_1 = vdupq_n_f32(0);
    float32x4_t c2_0 = vdupq_n_f32(0);
    float32x4_t c2_1 = vdupq_n_f32(0);
    float32x4_t c3_0 = vdupq_n_f32(0);
    float32x4_t c3_1 = vdupq_n_f32(0);
    for (int p = 0; p < depth; p++) {
        float32x4_t a0 = vld1q_f32(a + 0);
        float32x4_t a1 = vld1q_f32(a + 4);
        float32x4_t b0 = vld1q_f32(b + 0);
        c0_0 = vfmaq_laneq_f32(c0_0, a0, b0, 0);
        c0_1 = vfmaq_laneq_f32(c0_1, a1, b0, 0);
        c1_0 = vfmaq_laneq_f32(c1_0, a0, b0, 1);
        c1_1 = vfmaq_laneq_f32(c1_1, a1, b0, 1);
        c2_0 = vfmaq_laneq_f32(c2_0, a0, b0, 2);
        c2_1 = vfmaq_laneq_f32(c2_1, a1, b0, 2);
        c3_0 = vfmaq_laneq_f32(c3_0, a0, b0, 3);
        c3_1 = vfmaq_laneq_f32(c3_1, a1, b0, 3);
        a += 8;
        b += 12;
    }
    {
        float *out = c + (size_t)0 * ldc + 0;
        if (!first)
            c0_0 = vaddq_f32(c0_0, vld1q_f32(out));
        vst1q_f32(out, c0_0);
    }
    {
        float *out = c + (size_t)0 * ldc + 4;
        if (!first)
            c0_1 = vaddq_f32(c0_1, vld1q_f32(out));
        vst1q_f32(out, c0_1);
    }
    {
        float *out = c + (size_t)1 * ldc + 0;
        if (!first)
            c1_0 = vaddq_f32(c1_0, vld1q_f32(out));
        vst1q_f32(out, c1_0);
    }
    {
        float *out = c + (size_t)1 * ldc + 4;
        if (!first)
            c1_1 = vaddq_f32(c1_1, vld1q_f32(out));
        vst1q_f32(out, c1_1);
    }
    {
        float *out = c + (size_t)2 * ldc + 0;
        if (!first)
            c2_0 = vaddq_f32(c2_0, vld1q_f32(out));
        vst1q_f32(out, c2_0);
    }
    {
        float *out = c + (size_t)2 * ldc + 4;
        if (!first)
            c2_1 = vaddq_f32(c2_1, vld1q_f32(out));
        vst1q_f32(out, c2_1);
    }
    {
        float *out = c + (size_t)3 * ldc + 0;
        if (!first)
            c3_0 = vaddq_f32(c3_0, vld1q_f32(out));
        vst1q_f32(out, c3_0);
    }
    {
        float *out = c + (size_t)3 * ldc + 4;
        if (!first)
            c3_1 = vaddq_f32(c3_1, vld1q_f32(out));
        vst1q_f32(out, c3_1);
    }
}
static inline void shared32_tail8(int depth, const float *a, const float *b, float *c, int ldc,
                                  int first)
{
    float32x4_t c0_0 = vdupq_n_f32(0);
    float32x4_t c0_1 = vdupq_n_f32(0);
    float32x4_t c1_0 = vdupq_n_f32(0);
    float32x4_t c1_1 = vdupq_n_f32(0);
    float32x4_t c2_0 = vdupq_n_f32(0);
    float32x4_t c2_1 = vdupq_n_f32(0);
    float32x4_t c3_0 = vdupq_n_f32(0);
    float32x4_t c3_1 = vdupq_n_f32(0);
    float32x4_t c4_0 = vdupq_n_f32(0);
    float32x4_t c4_1 = vdupq_n_f32(0);
    float32x4_t c5_0 = vdupq_n_f32(0);
    float32x4_t c5_1 = vdupq_n_f32(0);
    float32x4_t c6_0 = vdupq_n_f32(0);
    float32x4_t c6_1 = vdupq_n_f32(0);
    float32x4_t c7_0 = vdupq_n_f32(0);
    float32x4_t c7_1 = vdupq_n_f32(0);
    for (int p = 0; p < depth; p++) {
        float32x4_t a0 = vld1q_f32(a + 0);
        float32x4_t a1 = vld1q_f32(a + 4);
        float32x4_t b0 = vld1q_f32(b + 0);
        float32x4_t b1 = vld1q_f32(b + 4);
        c0_0 = vfmaq_laneq_f32(c0_0, a0, b0, 0);
        c0_1 = vfmaq_laneq_f32(c0_1, a1, b0, 0);
        c1_0 = vfmaq_laneq_f32(c1_0, a0, b0, 1);
        c1_1 = vfmaq_laneq_f32(c1_1, a1, b0, 1);
        c2_0 = vfmaq_laneq_f32(c2_0, a0, b0, 2);
        c2_1 = vfmaq_laneq_f32(c2_1, a1, b0, 2);
        c3_0 = vfmaq_laneq_f32(c3_0, a0, b0, 3);
        c3_1 = vfmaq_laneq_f32(c3_1, a1, b0, 3);
        c4_0 = vfmaq_laneq_f32(c4_0, a0, b1, 0);
        c4_1 = vfmaq_laneq_f32(c4_1, a1, b1, 0);
        c5_0 = vfmaq_laneq_f32(c5_0, a0, b1, 1);
        c5_1 = vfmaq_laneq_f32(c5_1, a1, b1, 1);
        c6_0 = vfmaq_laneq_f32(c6_0, a0, b1, 2);
        c6_1 = vfmaq_laneq_f32(c6_1, a1, b1, 2);
        c7_0 = vfmaq_laneq_f32(c7_0, a0, b1, 3);
        c7_1 = vfmaq_laneq_f32(c7_1, a1, b1, 3);
        a += 8;
        b += 12;
    }
    {
        float *out = c + (size_t)0 * ldc + 0;
        if (!first)
            c0_0 = vaddq_f32(c0_0, vld1q_f32(out));
        vst1q_f32(out, c0_0);
    }
    {
        float *out = c + (size_t)0 * ldc + 4;
        if (!first)
            c0_1 = vaddq_f32(c0_1, vld1q_f32(out));
        vst1q_f32(out, c0_1);
    }
    {
        float *out = c + (size_t)1 * ldc + 0;
        if (!first)
            c1_0 = vaddq_f32(c1_0, vld1q_f32(out));
        vst1q_f32(out, c1_0);
    }
    {
        float *out = c + (size_t)1 * ldc + 4;
        if (!first)
            c1_1 = vaddq_f32(c1_1, vld1q_f32(out));
        vst1q_f32(out, c1_1);
    }
    {
        float *out = c + (size_t)2 * ldc + 0;
        if (!first)
            c2_0 = vaddq_f32(c2_0, vld1q_f32(out));
        vst1q_f32(out, c2_0);
    }
    {
        float *out = c + (size_t)2 * ldc + 4;
        if (!first)
            c2_1 = vaddq_f32(c2_1, vld1q_f32(out));
        vst1q_f32(out, c2_1);
    }
    {
        float *out = c + (size_t)3 * ldc + 0;
        if (!first)
            c3_0 = vaddq_f32(c3_0, vld1q_f32(out));
        vst1q_f32(out, c3_0);
    }
    {
        float *out = c + (size_t)3 * ldc + 4;
        if (!first)
            c3_1 = vaddq_f32(c3_1, vld1q_f32(out));
        vst1q_f32(out, c3_1);
    }
    {
        float *out = c + (size_t)4 * ldc + 0;
        if (!first)
            c4_0 = vaddq_f32(c4_0, vld1q_f32(out));
        vst1q_f32(out, c4_0);
    }
    {
        float *out = c + (size_t)4 * ldc + 4;
        if (!first)
            c4_1 = vaddq_f32(c4_1, vld1q_f32(out));
        vst1q_f32(out, c4_1);
    }
    {
        float *out = c + (size_t)5 * ldc + 0;
        if (!first)
            c5_0 = vaddq_f32(c5_0, vld1q_f32(out));
        vst1q_f32(out, c5_0);
    }
    {
        float *out = c + (size_t)5 * ldc + 4;
        if (!first)
            c5_1 = vaddq_f32(c5_1, vld1q_f32(out));
        vst1q_f32(out, c5_1);
    }
    {
        float *out = c + (size_t)6 * ldc + 0;
        if (!first)
            c6_0 = vaddq_f32(c6_0, vld1q_f32(out));
        vst1q_f32(out, c6_0);
    }
    {
        float *out = c + (size_t)6 * ldc + 4;
        if (!first)
            c6_1 = vaddq_f32(c6_1, vld1q_f32(out));
        vst1q_f32(out, c6_1);
    }
    {
        float *out = c + (size_t)7 * ldc + 0;
        if (!first)
            c7_0 = vaddq_f32(c7_0, vld1q_f32(out));
        vst1q_f32(out, c7_0);
    }
    {
        float *out = c + (size_t)7 * ldc + 4;
        if (!first)
            c7_1 = vaddq_f32(c7_1, vld1q_f32(out));
        vst1q_f32(out, c7_1);
    }
}
typedef struct {
    const float *a, *b;
    float *c, *workspace;
    int m, n, k, lda, ldb, ldc, tb, bstride, used;
    const camblas_executor_t *executor;
    _Alignas(64) atomic_int ready;
} shared32_work_t;
static void *shared32_workspace;
static size_t shared32_capacity;
static void shared32_task(const camblas_task_t *task, void *opaque)
{
    shared32_work_t *w = opaque;
    int id = task->i0, row = id % 16, col = id / 16;
    int agroups = w->m / 8, j0 = w->n * col / 4, je = w->n * (col + 1) / 4,
        bgroups = (je - j0 + 11) / 12;
    float *ap = w->workspace, *bp = ap + (size_t)w->m * w->k + (size_t)col * w->bstride;
    /* Deep products keep all micro-panels for one 512-depth block adjacent.
     * This layout bounds the working set of each compute pass without
     * changing operand values or retaining packed input across calls. */
    for (int p0 = 0; p0 < w->k; p0 += 16) {
        int pe = p0 + 16 < w->k ? p0 + 16 : w->k;
        for (int g = agroups * id / 64; g < agroups * (id + 1) / 64; g++)
            for (int p = p0; p < pe; p++) {
                int pc = p / 512 * 512, depth = w->k - pc < 512 ? w->k - pc : 512;
                size_t offset = w->k > 1024
                                    ? (size_t)pc * w->m + (size_t)g * 8 * depth + (p - pc) * 8
                                    : (size_t)g * 8 * w->k + p * 8;
                memcpy(ap + offset, w->a + g * 8 + (size_t)p * w->lda, 8 * sizeof(float));
            }
        for (int g = bgroups * row / 16; g < bgroups * (row + 1) / 16; g++) {
            int j = g * 12;
            for (int p = p0; p < pe; p++) {
                int pc = p / 512 * 512, depth = w->k - pc < 512 ? w->k - pc : 512;
                size_t offset = w->k > 1024
                                    ? (size_t)pc * bgroups * 12 + (size_t)j * depth + (p - pc) * 12
                                    : (size_t)j * w->k + p * 12;
                float *out = bp + offset;
                if (w->tb && j0 + j + 12 <= je)
                    memcpy(out, w->b + j0 + j + (size_t)p * w->ldb, 12 * sizeof(float));
                else
                    for (int q = 0; q < 12; q++)
                        out[q] = j0 + j + q < je ? (!w->tb ? w->b[p + (size_t)(j0 + j + q) * w->ldb]
                                                           : w->b[j0 + j + q + (size_t)p * w->ldb])
                                                 : 0;
            }
        }
    }
    atomic_fetch_add_explicit(&w->ready, 1, memory_order_acq_rel);
    while (atomic_load_explicit(&w->ready, memory_order_acquire) != 64)
        spin_pause();
    int i0 = agroups * row / 16 * 8, ie = agroups * (row + 1) / 16 * 8;
    for (int pc = 0; pc < w->k; pc += 512) {
        int depth = w->k - pc < 512 ? w->k - pc : 512;
        for (int j = 0; j < je - j0; j += 12)
            for (int i = i0; i < ie; i += 8) {
                const float *a = ap + (w->k > 1024 ? (size_t)pc * w->m + (size_t)i * depth
                                                   : (size_t)i * w->k + pc * 8);
                const float *b = bp + (w->k > 1024 ? (size_t)pc * bgroups * 12 + (size_t)j * depth
                                                   : (size_t)j * w->k + pc * 12);
                float *c = w->c + i + (size_t)(j0 + j) * w->ldc;
                int columns = je - j0 - j < 12 ? je - j0 - j : 12;
                if (columns == 12)
                    shared32_full(depth, a, b, c, w->ldc, pc == 0);
                else if (columns == 8)
                    shared32_tail8(depth, a, b, c, w->ldc, pc == 0);
                else
                    shared32_tail4(depth, a, b, c, w->ldc, pc == 0);
            }
    }
}
/* The private pool and a reused OpenMP team each assign one task to every
 * worker. Decline before the pack barrier if the OpenMP team is smaller. */
static void shared32_execute(void *opaque)
{
    shared32_work_t *w = opaque;
    camblas_task_t tasks[64];
    for (int t = 0; t < 64; ++t)
        tasks[t] = (camblas_task_t){t, t + 1, 0, 1};
    if (w->executor != &framework_spin_executor) {
#if CAMBLAS_FRAMEWORK_TEAM_REUSE
        if (!active_team || active_team->threads != 64)
            return;
#else
        return;
#endif
    }
    if (w->executor->run(shared32_task, tasks, 64, w, w->executor->user_data))
        fail("shared FP32 GEMM");
    w->used = 1;
}

static int try_base_shared32_s(camblas_ctx_t *ctx, char ta, char tb, int m, int n, int k,
                               float alpha, const float *a, int lda, const float *b, int ldb,
                               float beta, float *c, int ldc)
{
    if (ctx->num_threads != 64 || spin_inside || omp_in_parallel() || ta != 'N' ||
        (tb != 'N' && tb != 'T') || alpha != 1 || beta != 0 || m < n || m < 128 || m > 8192 ||
        n < 128 || n > 2048 || k < 128 || k > 4096 || m % 8 || n % 16)
        return 0;
    if (k > 1024) {
        /* The reused OpenMP team keeps its depth-major packed route.
         * The private pool uses bounded panels in try_shared32_s below. */
#if CAMBLAS_FRAMEWORK_TEAM_REUSE
        if (m <= n || tb != 'N' || ctx->executor != &framework_omp_executor)
            return 0;
#else
        return 0;
#endif
    } else if (m > 2048 || ctx->executor != &framework_spin_executor)
        return 0;
    shared32_work_t work = {.a = a,
                            .b = b,
                            .c = c,
                            .m = m,
                            .n = n,
                            .k = k,
                            .lda = lda,
                            .ldb = ldb,
                            .ldc = ldc,
                            .tb = tb == 'T'};
    work.executor = ctx->executor;
    work.bstride = ((n / 4 + 11) / 12 * 12) * k;
    size_t bytes = ((size_t)m * k + 4 * (size_t)work.bstride) * sizeof(float);
    if (bytes > shared32_capacity) {
        void *next = benchmark_alloc(bytes);
        if (!next)
            return 0;
        free(shared32_workspace);
        shared32_workspace = next;
        shared32_capacity = bytes;
    }
    work.workspace = shared32_workspace;
    atomic_init(&work.ready, 0);
    if (framework_team_call(shared32_execute, &work, m, n, k))
        fail("shared FP32 team");
    if (!work.used)
        return 0;
    atomic_fetch_add_explicit(&counters[PACKED], 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&shared_grid32_calls, 1, memory_order_relaxed);
    if (trace_calls)
        fprintf(
            stderr,
            "FRAMEWORK_BLAS backend=camblas operation=gemm precision=fp32 algorithm=shared-packed m=%d n=%d k=%d threads=64\n",
            m, n, k);
    return 1;
}
static void release_base_shared32(void)
{
    free(shared32_workspace);
    shared32_workspace = NULL;
    shared32_capacity = 0;
}

#ifndef CAMBLAS_STREAM32_KC
#define CAMBLAS_STREAM32_KC 768
#endif
#ifndef CAMBLAS_STREAM32_ROWS64
#define CAMBLAS_STREAM32_ROWS64 32
#endif
#if CAMBLAS_STREAM32_KC < 4 || CAMBLAS_STREAM32_KC > 1024 || CAMBLAS_STREAM32_KC % 4
#error "Streaming depth must be a multiple of four in [4, 1024]"
#endif
#if CAMBLAS_STREAM32_ROWS64 != 8 && CAMBLAS_STREAM32_ROWS64 != 16 && CAMBLAS_STREAM32_ROWS64 != 32
#error "Streaming row groups must divide sixty-four"
#endif
#ifndef CAMBLAS_OMP_STREAM32_KC
#define CAMBLAS_OMP_STREAM32_KC 512
#endif
#ifndef CAMBLAS_OMP_STREAM32_ROWS
#define CAMBLAS_OMP_STREAM32_ROWS 16
#endif
#if CAMBLAS_OMP_STREAM32_KC < 4 || CAMBLAS_OMP_STREAM32_KC > 1024 || CAMBLAS_OMP_STREAM32_KC % 4
#error "OpenMP streaming depth must be a multiple of four in [4, 1024]"
#endif
#if CAMBLAS_OMP_STREAM32_ROWS != 8 && CAMBLAS_OMP_STREAM32_ROWS != 16 && \
    CAMBLAS_OMP_STREAM32_ROWS != 32
#error "OpenMP streaming row groups must divide sixty-four"
#endif
enum { STREAM32_KC = CAMBLAS_STREAM32_KC, STREAM32_PANELS = 192 };
typedef struct {
    _Alignas(64) atomic_int count;
} stream32_counter_t;
typedef struct {
    const float *a, *b;
    float *c, *packed_a;
    float **panels;
    int m, n, k, lda, ldb, ldc, threads, rows, columns, groups, used, kc;
    const camblas_executor_t *executor;
    stream32_counter_t row_ready[32], column_ready[8];
} stream32_work_t;
/* Each A row panel has one consumer per column group; each B column
 * panel has one consumer per row group. Arrival and completion phases
 * protect both reads and subsequent overwrites of the shared panels. */
static void stream32_barrier(stream32_work_t *w, int row, int column, int phase)
{
    atomic_fetch_add_explicit(&w->row_ready[row].count, 1, memory_order_acq_rel);
    atomic_fetch_add_explicit(&w->column_ready[column].count, 1, memory_order_acq_rel);
    while (atomic_load_explicit(&w->row_ready[row].count, memory_order_acquire) <
           w->columns * phase)
        spin_pause();
    while (atomic_load_explicit(&w->column_ready[column].count, memory_order_acquire) <
           w->rows * phase)
        spin_pause();
}
/* Transpose four source columns into consecutive depth rows of one B panel. */
static void stream32_pack_b4(float *out, const float *in, int ld, int columns)
{
    float32x4_t a = vld1q_f32(in);
    float32x4_t b = columns > 1 ? vld1q_f32(in + ld) : vdupq_n_f32(0);
    float32x4_t c = columns > 2 ? vld1q_f32(in + 2 * ld) : vdupq_n_f32(0);
    float32x4_t d = columns > 3 ? vld1q_f32(in + 3 * ld) : vdupq_n_f32(0);
    float32x4_t x = vtrn1q_f32(a, b), y = vtrn2q_f32(a, b);
    float32x4_t z = vtrn1q_f32(c, d), t = vtrn2q_f32(c, d);
    vst1q_f32(out, vcombine_f32(vget_low_f32(x), vget_low_f32(z)));
    vst1q_f32(out + 12, vcombine_f32(vget_low_f32(y), vget_low_f32(t)));
    vst1q_f32(out + 24, vcombine_f32(vget_high_f32(x), vget_high_f32(z)));
    vst1q_f32(out + 36, vcombine_f32(vget_high_f32(y), vget_high_f32(t)));
}
static void stream32_task(const camblas_task_t *task, void *opaque)
{
    stream32_work_t *w = opaque;
    int id = task->i0, row = id % w->rows, column = id / w->rows;
    int agroups = w->m / 8, width = w->n / w->columns, j0 = column * width;
    int first_group = agroups * row / w->rows, last_group = agroups * (row + 1) / w->rows;
    int i0 = first_group * 8, ie = last_group * 8;
    int pack_first = first_group + (last_group - first_group) * column / w->columns;
    int pack_last = first_group + (last_group - first_group) * (column + 1) / w->columns;
    for (int pc = 0, phase = 1; pc < w->k; pc += w->kc, phase += 2) {
        int depth = w->k - pc < w->kc ? w->k - pc : w->kc;
        for (int p0 = 0; p0 < depth; p0 += 16) {
            int pe = p0 + 16 < depth ? p0 + 16 : depth;
            for (int g = pack_first; g < pack_last; ++g)
                for (int p = p0; p < pe; ++p)
                    memcpy(w->packed_a + (size_t)g * 8 * w->kc + (size_t)p * 8,
                           w->a + g * 8 + (size_t)(pc + p) * w->lda, 8 * sizeof(float));
        }
        for (int g = w->groups * row / w->rows; g < w->groups * (row + 1) / w->rows; ++g) {
            float *panel = w->panels[column * w->groups + g];
            int count = width - g * 12 < 12 ? width - g * 12 : 12;
            int p = 0;
            for (; p + 4 <= depth; p += 4)
                for (int j = 0; j < 12; j += 4) {
                    float *out = panel + (size_t)p * 12 + j;
                    if (j < count)
                        stream32_pack_b4(out, w->b + pc + p + (size_t)(j0 + g * 12 + j) * w->ldb,
                                         w->ldb, count - j < 4 ? count - j : 4);
                    else
                        for (int q = 0; q < 4; ++q)
                            vst1q_f32(out + q * 12, vdupq_n_f32(0));
                }
            for (; p < depth; ++p)
                for (int j = 0; j < 12; ++j)
                    panel[(size_t)p * 12 + j] =
                        j < count ? w->b[pc + p + (size_t)(j0 + g * 12 + j) * w->ldb] : 0;
        }
        stream32_barrier(w, row, column, phase);
        for (int g = 0; g < w->groups; ++g) {
            int count = width - g * 12 < 12 ? width - g * 12 : 12;
            const float *b = w->panels[column * w->groups + g];
            for (int i = i0; i < ie; i += 8) {
                const float *a = w->packed_a + (size_t)i * w->kc;
                float *c = w->c + i + (size_t)(j0 + g * 12) * w->ldc;
                if (count == 12)
                    shared32_full(depth, a, b, c, w->ldc, pc == 0);
                else if (count == 8)
                    shared32_tail8(depth, a, b, c, w->ldc, pc == 0);
                else
                    shared32_tail4(depth, a, b, c, w->ldc, pc == 0);
            }
        }
        /* All products finish before any packed panel can be overwritten. */
        stream32_barrier(w, row, column, phase + 1);
    }
}
static void stream32_execute(void *opaque)
{
    stream32_work_t *w = opaque;
    camblas_task_t tasks[64];
    for (int t = 0; t < w->threads; ++t)
        tasks[t] = (camblas_task_t){t, t + 1, 0, 1};
    if (w->executor != &framework_spin_executor) {
#if CAMBLAS_FRAMEWORK_TEAM_REUSE
        if (!active_team || active_team->threads != w->threads)
            return;
#else
        return;
#endif
    }
    if (w->executor->run(stream32_task, tasks, w->threads, w, w->executor->user_data))
        fail("streaming FP32 GEMM");
    w->used = 1;
}
static int try_shared32_s(camblas_ctx_t *ctx, char ta, char tb, int m, int n, int k, float alpha,
                          const float *a, int lda, const float *b, int ldb, float beta, float *c,
                          int ldc)
{
    int eligible = ctx->num_threads == 64 && (ctx->executor == &framework_spin_executor ||
                                              ctx->executor == &framework_omp_executor);
    if (!eligible || spin_inside || omp_in_parallel() || ta != 'N' || tb != 'N' || alpha != 1 ||
        beta != 0 || m <= n || m > 8192 || m < 128 || n < 128 || n > 2048 || k <= 1024 ||
        k > 4096 || m % 8 || n % 16)
        return try_base_shared32_s(ctx, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    int omp = ctx->executor == &framework_omp_executor;
    int kc = omp ? CAMBLAS_OMP_STREAM32_KC : STREAM32_KC;
    int depth = k < kc ? k : kc;
    size_t a_bytes = (size_t)m * depth * sizeof(float);
    int rows = omp ? CAMBLAS_OMP_STREAM32_ROWS : CAMBLAS_STREAM32_ROWS64, columns = 64 / rows;
    int groups = (n / columns + 11) / 12, count = groups * columns;
    if (count > STREAM32_PANELS)
        return 0;
    size_t panel_bytes = 12 * kc * sizeof(float);
    size_t bytes = a_bytes + count * panel_bytes;
    /* A single owned allocation can be recycled by libc between calls;
     * align its payload for vector loads without page-aligning the request. */
    void *allocation = malloc(bytes + 63);
    unsigned char *frame =
        allocation ? (unsigned char *)(((uintptr_t)allocation + 63u) & ~(uintptr_t)63u) : NULL;
    if (!frame)
        return 0;
    float *panels[STREAM32_PANELS];
    for (int t = 0; t < count; ++t)
        panels[t] = (float *)(frame + a_bytes + (size_t)t * panel_bytes);
    stream32_work_t w = {.kc = kc,
                         .a = a,
                         .b = b,
                         .c = c,
                         .packed_a = (float *)frame,
                         .panels = panels,
                         .m = m,
                         .n = n,
                         .k = k,
                         .lda = lda,
                         .ldb = ldb,
                         .ldc = ldc,
                         .threads = ctx->num_threads,
                         .rows = rows,
                         .columns = columns,
                         .groups = groups,
                         .executor = ctx->executor};
    for (int t = 0; t < 32; ++t)
        atomic_init(&w.row_ready[t].count, 0);
    for (int t = 0; t < 8; ++t)
        atomic_init(&w.column_ready[t].count, 0);
    if (framework_team_call(stream32_execute, &w, m, n, k))
        fail("streaming FP32 team");
    free(allocation);
    if (!w.used)
        return 0;
    atomic_fetch_add_explicit(&counters[PACKED], 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&shared_grid32_calls, 1, memory_order_relaxed);
    if (trace_calls)
        fprintf(
            stderr,
            "FRAMEWORK_BLAS backend=camblas operation=gemm precision=fp32 algorithm=shared-stream m=%d n=%d k=%d threads=64\n",
            m, n, k);
    return 1;
}
static void release_shared32(void)
{
    release_base_shared32();
}
#define try_shared32_d(...) 0
#endif
