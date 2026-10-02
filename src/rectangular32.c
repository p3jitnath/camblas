/* Experimental FP32 rectangular Strassen with fused packed transforms.
 * Deep products reuse bounded depth panels; packing and compute complete
 * synchronously before panels are reused, and C is written only at recombination.
 * No input-dependent values survive the call. This is a different floating-
 * point algorithm from classical GEMM and requires separate error checks. */
#include "camblas.h"
#include "camblas_executor.h"
#include "rectangular32.h"
#ifndef CAMBLAS_RECT32_NEON12
#define CAMBLAS_RECT32_NEON12 0
#endif
#include "packed.h"
#include "kernels.h"
#include "gemm_bounds.h"
#include <arm_neon.h>
#include <stdatomic.h>
#include <math.h>
#include <stdint.h>
#ifndef CAMBLAS_RECT_ROWS16
#define CAMBLAS_RECT_ROWS16 8
#endif
#if CAMBLAS_RECT_ROWS16 < 1 || CAMBLAS_RECT_ROWS16 > 16 || 16 % CAMBLAS_RECT_ROWS16
#error "The sixteen-worker row grid must divide sixteen"
#endif
#ifndef CAMBLAS_RECT_KC
#define CAMBLAS_RECT_KC 256
#endif
#if CAMBLAS_RECT_KC < 4 || CAMBLAS_RECT_KC > 1024 || CAMBLAS_RECT_KC % 4
#error "Rectangular depth blocks must be multiples of four in [4, 1024]"
#endif

/* Packing and recombination share this seven-product order:
 * P1 = (A11 + A22) * (B11 + B22), P2 = (A21 + A22) * B11,
 * P3 = A11 * (B12 - B22),         P4 = A22 * (B21 - B11),
 * P5 = (A11 + A12) * B22,         P6 = (A21 - A11) * (B11 + B12),
 * P7 = (A12 - A22) * (B21 + B22).
 * Quadrant indices refer to logical matrices, including transposed B. */

/* Research-only exports; the public BLAS ABI does not expose this policy. */
int camblas_experimental_rectangular32_available(void)
{
    camblas_kernel_runtime_t runtime;
    camblas_kernel_runtime_query(&runtime);
    return runtime.compiled_sve && runtime.hw_sve && runtime.vl_bits == 128;
}

int camblas_experimental_rectangular32_bytes(int m, int n, int k, size_t *bytes)
{
    if (!bytes || m < 2 || n < 2 || k < 2 || m > 8192 || n > 8192 || k > 8192 || m % 2 || n % 2 ||
        k % 2)
        return -1;
    uint64_t hm = (unsigned)m / 2, hn = (unsigned)n / 2, hk = (unsigned)k / 2;
    uint64_t apad = (hm + 11) / 12 * 12, bpad = (hn + 7) / 8 * 8;
    /* Preserve the full-depth scratch contract, including streamed calls. */
    uint64_t elements = 7 * ((apad + bpad) * hk + hm * hn);
    if (elements > PTRDIFF_MAX / sizeof(float) || elements > SIZE_MAX / sizeof(float))
        return -1;
    *bytes = (size_t)elements * sizeof(float);
    return 0;
}

int camblas_experimental_rectangular32_segment_bytes(int workers, int m, int n, int k,
                                                     size_t sizes[3])
{
    size_t legacy;
    if (!sizes || workers < 1 || workers > 64 ||
        camblas_experimental_rectangular32_bytes(m, n, k, &legacy))
        return -1;
    int hm = m / 2, hn = n / 2, hk = k / 2;
    int stream = workers >= 16 && workers <= 32 && m >= (int64_t)n * 4 && k > 1024;
    int packed_k = stream && hk > CAMBLAS_RECT_KC ? CAMBLAS_RECT_KC : hk;
    sizes[0] = 7 * (size_t)((hm + 11) / 12 * 12) * packed_k * sizeof(float);
    sizes[1] = 7 * (size_t)((hn + 7) / 8 * 8) * packed_k * sizeof(float);
    sizes[2] = 7 * (size_t)hm * hn * sizeof(float);
    return 0;
}

typedef struct {
    int tb, hm, hn, hk, source_pc, lda, ldb, ldc, workers, row_groups, column_groups, depth_blocks,
        neon_kernel;
    size_t a_plane, b_plane, product_plane;
    const float *a[7], *a2[7], *b[7], *b2[7];
    float *packed_a, *packed_b, *products, *c;
    float a_max[64], b_max[64];
    atomic_int failed;
} rectangular32_t;

/* The range preflight rides on the pack pass: every input element is already
 * loaded exactly here, so the abs-max scan costs no extra memory traffic.
 * Vector bodies keep the NEON max NaN-suppressing behaviour of the standalone
 * scan; scalar tails flag non-finite values like the standalone scalar tail. */
static float32x4_t rectangular32_abs_max4(float32x4_t acc, float32x4_t value)
{
    return vmaxq_f32(acc, vabsq_f32(value));
}

static void rectangular32_scan_max(float *acc, float value)
{
    float magnitude = fabsf(value);
    if (!isfinite(magnitude))
        *acc = INFINITY;
    else if (magnitude > *acc)
        *acc = magnitude;
}

static void rectangular32_store_max(float32x4_t acc, float tail, float *out)
{
    float peak = vmaxvq_f32(acc);
    float merged = isfinite(peak) ? fmaxf(peak, tail) : INFINITY;
    if (merged > *out)
        *out = merged;
}

/* Return a small logical B tile as contiguous output-column vectors. */
static void rectangular32_load_b_tile(const float *base, int stride, int transposed,
                                      float32x4_t out[4], float32x4_t *absmax)
{
    if (transposed) {
        for (int q = 0; q < 4; ++q) {
            out[q] = vld1q_f32(base + (size_t)q * stride);
            if (absmax)
                *absmax = rectangular32_abs_max4(*absmax, out[q]);
        }
    } else {
        float32x4_t a = vld1q_f32(base), b = vld1q_f32(base + stride);
        float32x4_t c = vld1q_f32(base + 2 * stride), d = vld1q_f32(base + 3 * stride);
        if (absmax) {
            *absmax = rectangular32_abs_max4(*absmax, a);
            *absmax = rectangular32_abs_max4(*absmax, b);
            *absmax = rectangular32_abs_max4(*absmax, c);
            *absmax = rectangular32_abs_max4(*absmax, d);
        }
        float32x4_t t0 = vtrn1q_f32(a, b), t1 = vtrn2q_f32(a, b);
        float32x4_t t2 = vtrn1q_f32(c, d), t3 = vtrn2q_f32(c, d);
        out[0] = vcombine_f32(vget_low_f32(t0), vget_low_f32(t2));
        out[2] = vcombine_f32(vget_high_f32(t0), vget_high_f32(t2));
        out[1] = vcombine_f32(vget_low_f32(t1), vget_low_f32(t3));
        out[3] = vcombine_f32(vget_high_f32(t1), vget_high_f32(t3));
    }
}

/* Load four quadrants once and produce all seven B transforms. */
static void rectangular32_pack_all_b(rectangular32_t *w, int column, int pc, int bk, float *bmax)
{
    int count = w->hn - column < 8 ? w->hn - column : 8;
    const float *quadrants[4] = {w->b[0], w->b[2], w->b[3], w->b[4]};
    size_t base_offset = w->tb ? column + (size_t)(w->source_pc + pc) * w->ldb
                               : w->source_pc + pc + (size_t)column * w->ldb;
    float *base = w->packed_b + (size_t)pc * w->column_groups * 8 + (size_t)column * bk;
    size_t plane = w->b_plane;
    float32x4_t acc = vdupq_n_f32(0);
    float tail = 0;
    int q = 0;
    if (count == 8)
        for (; q + 4 <= bk; q += 4)
            for (int j = 0; j < 8; j += 4) {
                size_t offset =
                    base_offset + (w->tb ? j + (size_t)q * w->ldb : q + (size_t)j * w->ldb);
                float32x4_t b11[4], b12[4], b21[4], b22[4];
                rectangular32_load_b_tile(quadrants[0] + offset, w->ldb, w->tb, b11, &acc);
                rectangular32_load_b_tile(quadrants[1] + offset, w->ldb, w->tb, b12, &acc);
                rectangular32_load_b_tile(quadrants[2] + offset, w->ldb, w->tb, b21, &acc);
                rectangular32_load_b_tile(quadrants[3] + offset, w->ldb, w->tb, b22, &acc);
                for (int l = 0; l < 4; ++l) {
                    float *out = base + (size_t)(q + l) * 8 + j;
                    vst1q_f32(out, vaddq_f32(b11[l], b22[l]));
                    vst1q_f32(out + plane, b11[l]);
                    vst1q_f32(out + 2 * plane, vsubq_f32(b12[l], b22[l]));
                    vst1q_f32(out + 3 * plane, vsubq_f32(b21[l], b11[l]));
                    vst1q_f32(out + 4 * plane, b22[l]);
                    vst1q_f32(out + 5 * plane, vaddq_f32(b11[l], b12[l]));
                    vst1q_f32(out + 6 * plane, vaddq_f32(b21[l], b22[l]));
                }
            }
    for (; q < bk; ++q)
        for (int j = 0; j < 8; ++j) {
            size_t offset = base_offset + (w->tb ? j + (size_t)q * w->ldb : q + (size_t)j * w->ldb);
            float b11 = j < count ? quadrants[0][offset] : 0;
            float b12 = j < count ? quadrants[1][offset] : 0;
            float b21 = j < count ? quadrants[2][offset] : 0;
            float b22 = j < count ? quadrants[3][offset] : 0;
            if (j < count) {
                rectangular32_scan_max(&tail, b11);
                rectangular32_scan_max(&tail, b12);
                rectangular32_scan_max(&tail, b21);
                rectangular32_scan_max(&tail, b22);
            }
            float *out = base + (size_t)q * 8 + j;
            out[0] = b11 + b22;
            out[plane] = b11;
            out[2 * plane] = b12 - b22;
            out[3 * plane] = b21 - b11;
            out[4 * plane] = b22;
            out[5 * plane] = b11 + b12;
            out[6 * plane] = b21 + b22;
        }
    rectangular32_store_max(acc, tail, bmax);
}
/* Reuse four loaded quadrants for all seven A transforms. A small row/depth
 * tile limits the live input set when original columns have a large stride. */
static void rectangular32_pack_all_a(rectangular32_t *w, int first, int last, int pc, int bk,
                                     float *amax)
{
    size_t plane = w->a_plane;
    float32x4_t acc = vdupq_n_f32(0);
    float tail = 0;
    for (int q0 = 0; q0 < bk; q0 += 16) {
        int end = bk - q0 < 16 ? bk : q0 + 16;
        for (int group = first; group < last; ++group) {
            int row = group * 12, count = w->hm - row < 12 ? w->hm - row : 12;
            float *base = w->packed_a + (size_t)pc * w->row_groups * 12 + (size_t)row * bk;
            for (int q = q0; q < end; ++q) {
                size_t source = row + (size_t)(w->source_pc + pc + q) * w->lda;
                float *out = base + (size_t)q * 12;
                int i = 0;
                for (; i + 4 <= count; i += 4) {
                    float32x4_t a11 = vld1q_f32(w->a[0] + source + i);
                    float32x4_t a21 = vld1q_f32(w->a[1] + source + i);
                    float32x4_t a12 = vld1q_f32(w->a[6] + source + i);
                    float32x4_t a22 = vld1q_f32(w->a[3] + source + i);
                    acc = rectangular32_abs_max4(acc, a11);
                    acc = rectangular32_abs_max4(acc, a21);
                    acc = rectangular32_abs_max4(acc, a12);
                    acc = rectangular32_abs_max4(acc, a22);
                    vst1q_f32(out + i, vaddq_f32(a11, a22));
                    vst1q_f32(out + plane + i, vaddq_f32(a21, a22));
                    vst1q_f32(out + 2 * plane + i, a11);
                    vst1q_f32(out + 3 * plane + i, a22);
                    vst1q_f32(out + 4 * plane + i, vaddq_f32(a11, a12));
                    vst1q_f32(out + 5 * plane + i, vsubq_f32(a21, a11));
                    vst1q_f32(out + 6 * plane + i, vsubq_f32(a12, a22));
                }
                for (; i < count; ++i) {
                    float a11 = w->a[0][source + i], a21 = w->a[1][source + i];
                    float a12 = w->a[6][source + i], a22 = w->a[3][source + i];
                    rectangular32_scan_max(&tail, a11);
                    rectangular32_scan_max(&tail, a21);
                    rectangular32_scan_max(&tail, a12);
                    rectangular32_scan_max(&tail, a22);
                    out[i] = a11 + a22;
                    out[plane + i] = a21 + a22;
                    out[2 * plane + i] = a11;
                    out[3 * plane + i] = a22;
                    out[4 * plane + i] = a11 + a12;
                    out[5 * plane + i] = a21 - a11;
                    out[6 * plane + i] = a12 - a22;
                }
                for (; i < 12; ++i)
                    for (int p = 0; p < 7; ++p)
                        out[(size_t)p * plane + i] = 0;
            }
        }
    }
    rectangular32_store_max(acc, tail, amax);
}

static void rectangular32_pack(const camblas_task_t *task, void *opaque)
{
    rectangular32_t *w = opaque;
    float a_max = 0, b_max = 0;
    int per_block = 4;
    while (per_block > 1 &&
           ((w->row_groups + per_block - 1) / per_block) * w->depth_blocks < w->workers)
        per_block /= 2;
    int a_blocks = (w->row_groups + per_block - 1) / per_block;
    /* Weight an all-seven A block more heavily than one B micro-panel when
     * distributing work. Only the first index in each weighted slot owns it. */
    int weight = 12 * per_block, groups = a_blocks * weight + 7 * w->column_groups;
    int total = groups * w->depth_blocks, worker = task->i0;
    for (int index = total * worker / w->workers; index < total * (worker + 1) / w->workers;
         ++index) {
        int group = index % groups, pc = (index / groups) * CAMBLAS_RECT_KC;
        int bk = w->hk - pc < CAMBLAS_RECT_KC ? w->hk - pc : CAMBLAS_RECT_KC;
        if (group < a_blocks * weight) {
            if (group % weight)
                continue;
            int first = group / weight * per_block, last = first + per_block;
            if (last > w->row_groups)
                last = w->row_groups;
            rectangular32_pack_all_a(w, first, last, pc, bk, &a_max);
        } else {
            group -= a_blocks * weight;
            if (group % 7 == 0)
                rectangular32_pack_all_b(w, group / 7 * 8, pc, bk, &b_max);
        }
    }
    w->a_max[worker] = a_max;
    w->b_max[worker] = b_max;
}

#if CAMBLAS_RECT32_NEON12
/* Independent NEON 12x8 kernel matching existing binary32 Strassen panels. */
static inline void rectangular32_neon12_kernel(int depth, const float *a, const float *b, float *c,
                                               int ldc, int first, int rows, int columns)
{
    const float *ap = a, *bp = b;
    float *cp = c;
    uintptr_t stride = (uintptr_t)ldc * sizeof(float);
    __asm__ volatile(
        "movi v8.16b, #0\n\t"
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
        "cmp %w[depth], #4\n\t"
        "b.lt 2f\n\t"
        "1:\n\t"
        "cmp %w[depth], #36\n\t"
        "b.lt 7f\n\t"
        "prfm pldl1keep, [%[ap], #1536]\n\t"
        "prfm pldl1keep, [%[ap], #1600]\n\t"
        "prfm pldl1keep, [%[ap], #1664]\n\t"
        "prfm pldl1keep, [%[bp], #1024]\n\t"
        "prfm pldl1keep, [%[bp], #1088]\n\t"
        "7:\n\t"
        "ld1 {v0.4s-v2.4s}, [%[ap]], #48\n\t"
        "ld1 {v3.4s-v4.4s}, [%[bp]], #32\n\t"
        "fmla v8.4s, v0.4s, v3.s[0]\n\t"
        "fmla v9.4s, v1.4s, v3.s[0]\n\t"
        "fmla v10.4s, v2.4s, v3.s[0]\n\t"
        "fmla v11.4s, v0.4s, v3.s[1]\n\t"
        "fmla v12.4s, v1.4s, v3.s[1]\n\t"
        "fmla v13.4s, v2.4s, v3.s[1]\n\t"
        "fmla v14.4s, v0.4s, v3.s[2]\n\t"
        "fmla v15.4s, v1.4s, v3.s[2]\n\t"
        "fmla v16.4s, v2.4s, v3.s[2]\n\t"
        "fmla v17.4s, v0.4s, v3.s[3]\n\t"
        "fmla v18.4s, v1.4s, v3.s[3]\n\t"
        "fmla v19.4s, v2.4s, v3.s[3]\n\t"
        "fmla v20.4s, v0.4s, v4.s[0]\n\t"
        "fmla v21.4s, v1.4s, v4.s[0]\n\t"
        "fmla v22.4s, v2.4s, v4.s[0]\n\t"
        "fmla v23.4s, v0.4s, v4.s[1]\n\t"
        "fmla v24.4s, v1.4s, v4.s[1]\n\t"
        "fmla v25.4s, v2.4s, v4.s[1]\n\t"
        "fmla v26.4s, v0.4s, v4.s[2]\n\t"
        "fmla v27.4s, v1.4s, v4.s[2]\n\t"
        "fmla v28.4s, v2.4s, v4.s[2]\n\t"
        "fmla v29.4s, v0.4s, v4.s[3]\n\t"
        "fmla v30.4s, v1.4s, v4.s[3]\n\t"
        "fmla v31.4s, v2.4s, v4.s[3]\n\t"
        "ld1 {v0.4s-v2.4s}, [%[ap]], #48\n\t"
        "ld1 {v3.4s-v4.4s}, [%[bp]], #32\n\t"
        "fmla v8.4s, v0.4s, v3.s[0]\n\t"
        "fmla v9.4s, v1.4s, v3.s[0]\n\t"
        "fmla v10.4s, v2.4s, v3.s[0]\n\t"
        "fmla v11.4s, v0.4s, v3.s[1]\n\t"
        "fmla v12.4s, v1.4s, v3.s[1]\n\t"
        "fmla v13.4s, v2.4s, v3.s[1]\n\t"
        "fmla v14.4s, v0.4s, v3.s[2]\n\t"
        "fmla v15.4s, v1.4s, v3.s[2]\n\t"
        "fmla v16.4s, v2.4s, v3.s[2]\n\t"
        "fmla v17.4s, v0.4s, v3.s[3]\n\t"
        "fmla v18.4s, v1.4s, v3.s[3]\n\t"
        "fmla v19.4s, v2.4s, v3.s[3]\n\t"
        "fmla v20.4s, v0.4s, v4.s[0]\n\t"
        "fmla v21.4s, v1.4s, v4.s[0]\n\t"
        "fmla v22.4s, v2.4s, v4.s[0]\n\t"
        "fmla v23.4s, v0.4s, v4.s[1]\n\t"
        "fmla v24.4s, v1.4s, v4.s[1]\n\t"
        "fmla v25.4s, v2.4s, v4.s[1]\n\t"
        "fmla v26.4s, v0.4s, v4.s[2]\n\t"
        "fmla v27.4s, v1.4s, v4.s[2]\n\t"
        "fmla v28.4s, v2.4s, v4.s[2]\n\t"
        "fmla v29.4s, v0.4s, v4.s[3]\n\t"
        "fmla v30.4s, v1.4s, v4.s[3]\n\t"
        "fmla v31.4s, v2.4s, v4.s[3]\n\t"
        "ld1 {v0.4s-v2.4s}, [%[ap]], #48\n\t"
        "ld1 {v3.4s-v4.4s}, [%[bp]], #32\n\t"
        "fmla v8.4s, v0.4s, v3.s[0]\n\t"
        "fmla v9.4s, v1.4s, v3.s[0]\n\t"
        "fmla v10.4s, v2.4s, v3.s[0]\n\t"
        "fmla v11.4s, v0.4s, v3.s[1]\n\t"
        "fmla v12.4s, v1.4s, v3.s[1]\n\t"
        "fmla v13.4s, v2.4s, v3.s[1]\n\t"
        "fmla v14.4s, v0.4s, v3.s[2]\n\t"
        "fmla v15.4s, v1.4s, v3.s[2]\n\t"
        "fmla v16.4s, v2.4s, v3.s[2]\n\t"
        "fmla v17.4s, v0.4s, v3.s[3]\n\t"
        "fmla v18.4s, v1.4s, v3.s[3]\n\t"
        "fmla v19.4s, v2.4s, v3.s[3]\n\t"
        "fmla v20.4s, v0.4s, v4.s[0]\n\t"
        "fmla v21.4s, v1.4s, v4.s[0]\n\t"
        "fmla v22.4s, v2.4s, v4.s[0]\n\t"
        "fmla v23.4s, v0.4s, v4.s[1]\n\t"
        "fmla v24.4s, v1.4s, v4.s[1]\n\t"
        "fmla v25.4s, v2.4s, v4.s[1]\n\t"
        "fmla v26.4s, v0.4s, v4.s[2]\n\t"
        "fmla v27.4s, v1.4s, v4.s[2]\n\t"
        "fmla v28.4s, v2.4s, v4.s[2]\n\t"
        "fmla v29.4s, v0.4s, v4.s[3]\n\t"
        "fmla v30.4s, v1.4s, v4.s[3]\n\t"
        "fmla v31.4s, v2.4s, v4.s[3]\n\t"
        "ld1 {v0.4s-v2.4s}, [%[ap]], #48\n\t"
        "ld1 {v3.4s-v4.4s}, [%[bp]], #32\n\t"
        "fmla v8.4s, v0.4s, v3.s[0]\n\t"
        "fmla v9.4s, v1.4s, v3.s[0]\n\t"
        "fmla v10.4s, v2.4s, v3.s[0]\n\t"
        "fmla v11.4s, v0.4s, v3.s[1]\n\t"
        "fmla v12.4s, v1.4s, v3.s[1]\n\t"
        "fmla v13.4s, v2.4s, v3.s[1]\n\t"
        "fmla v14.4s, v0.4s, v3.s[2]\n\t"
        "fmla v15.4s, v1.4s, v3.s[2]\n\t"
        "fmla v16.4s, v2.4s, v3.s[2]\n\t"
        "fmla v17.4s, v0.4s, v3.s[3]\n\t"
        "fmla v18.4s, v1.4s, v3.s[3]\n\t"
        "fmla v19.4s, v2.4s, v3.s[3]\n\t"
        "fmla v20.4s, v0.4s, v4.s[0]\n\t"
        "fmla v21.4s, v1.4s, v4.s[0]\n\t"
        "fmla v22.4s, v2.4s, v4.s[0]\n\t"
        "fmla v23.4s, v0.4s, v4.s[1]\n\t"
        "fmla v24.4s, v1.4s, v4.s[1]\n\t"
        "fmla v25.4s, v2.4s, v4.s[1]\n\t"
        "fmla v26.4s, v0.4s, v4.s[2]\n\t"
        "fmla v27.4s, v1.4s, v4.s[2]\n\t"
        "fmla v28.4s, v2.4s, v4.s[2]\n\t"
        "fmla v29.4s, v0.4s, v4.s[3]\n\t"
        "fmla v30.4s, v1.4s, v4.s[3]\n\t"
        "fmla v31.4s, v2.4s, v4.s[3]\n\t"
        "sub %w[depth], %w[depth], #4\n\t"
        "cmp %w[depth], #4\n\t"
        "b.ge 1b\n\t"
        "2:\n\t"
        "cbz %w[depth], 4f\n\t"
        "3:\n\t"
        "ld1 {v0.4s-v2.4s}, [%[ap]], #48\n\t"
        "ld1 {v3.4s-v4.4s}, [%[bp]], #32\n\t"
        "fmla v8.4s, v0.4s, v3.s[0]\n\t"
        "fmla v9.4s, v1.4s, v3.s[0]\n\t"
        "fmla v10.4s, v2.4s, v3.s[0]\n\t"
        "fmla v11.4s, v0.4s, v3.s[1]\n\t"
        "fmla v12.4s, v1.4s, v3.s[1]\n\t"
        "fmla v13.4s, v2.4s, v3.s[1]\n\t"
        "fmla v14.4s, v0.4s, v3.s[2]\n\t"
        "fmla v15.4s, v1.4s, v3.s[2]\n\t"
        "fmla v16.4s, v2.4s, v3.s[2]\n\t"
        "fmla v17.4s, v0.4s, v3.s[3]\n\t"
        "fmla v18.4s, v1.4s, v3.s[3]\n\t"
        "fmla v19.4s, v2.4s, v3.s[3]\n\t"
        "fmla v20.4s, v0.4s, v4.s[0]\n\t"
        "fmla v21.4s, v1.4s, v4.s[0]\n\t"
        "fmla v22.4s, v2.4s, v4.s[0]\n\t"
        "fmla v23.4s, v0.4s, v4.s[1]\n\t"
        "fmla v24.4s, v1.4s, v4.s[1]\n\t"
        "fmla v25.4s, v2.4s, v4.s[1]\n\t"
        "fmla v26.4s, v0.4s, v4.s[2]\n\t"
        "fmla v27.4s, v1.4s, v4.s[2]\n\t"
        "fmla v28.4s, v2.4s, v4.s[2]\n\t"
        "fmla v29.4s, v0.4s, v4.s[3]\n\t"
        "fmla v30.4s, v1.4s, v4.s[3]\n\t"
        "fmla v31.4s, v2.4s, v4.s[3]\n\t"
        "subs %w[depth], %w[depth], #1\n\t"
        "b.ne 3b\n\t"
        "4:\n\t"
        "cbnz %w[first], 10f\n\t"
        "ldr q7, [%[cp], #0]\n\t"
        "fadd v8.4s, v8.4s, v7.4s\n\t"
        "10:\n\t"
        "str q8, [%[cp], #0]\n\t"
        "cmp %w[rows], #8\n\t"
        "b.lt 50f\n\t"
        "cbnz %w[first], 11f\n\t"
        "ldr q7, [%[cp], #16]\n\t"
        "fadd v9.4s, v9.4s, v7.4s\n\t"
        "11:\n\t"
        "str q9, [%[cp], #16]\n\t"
        "cmp %w[rows], #12\n\t"
        "b.lt 50f\n\t"
        "cbnz %w[first], 12f\n\t"
        "ldr q7, [%[cp], #32]\n\t"
        "fadd v10.4s, v10.4s, v7.4s\n\t"
        "12:\n\t"
        "str q10, [%[cp], #32]\n\t"
        "50:\n\t"
        "add %[cp], %[cp], %[stride]\n\t"
        "cbnz %w[first], 13f\n\t"
        "ldr q7, [%[cp], #0]\n\t"
        "fadd v11.4s, v11.4s, v7.4s\n\t"
        "13:\n\t"
        "str q11, [%[cp], #0]\n\t"
        "cmp %w[rows], #8\n\t"
        "b.lt 51f\n\t"
        "cbnz %w[first], 14f\n\t"
        "ldr q7, [%[cp], #16]\n\t"
        "fadd v12.4s, v12.4s, v7.4s\n\t"
        "14:\n\t"
        "str q12, [%[cp], #16]\n\t"
        "cmp %w[rows], #12\n\t"
        "b.lt 51f\n\t"
        "cbnz %w[first], 15f\n\t"
        "ldr q7, [%[cp], #32]\n\t"
        "fadd v13.4s, v13.4s, v7.4s\n\t"
        "15:\n\t"
        "str q13, [%[cp], #32]\n\t"
        "51:\n\t"
        "add %[cp], %[cp], %[stride]\n\t"
        "cbnz %w[first], 16f\n\t"
        "ldr q7, [%[cp], #0]\n\t"
        "fadd v14.4s, v14.4s, v7.4s\n\t"
        "16:\n\t"
        "str q14, [%[cp], #0]\n\t"
        "cmp %w[rows], #8\n\t"
        "b.lt 52f\n\t"
        "cbnz %w[first], 17f\n\t"
        "ldr q7, [%[cp], #16]\n\t"
        "fadd v15.4s, v15.4s, v7.4s\n\t"
        "17:\n\t"
        "str q15, [%[cp], #16]\n\t"
        "cmp %w[rows], #12\n\t"
        "b.lt 52f\n\t"
        "cbnz %w[first], 18f\n\t"
        "ldr q7, [%[cp], #32]\n\t"
        "fadd v16.4s, v16.4s, v7.4s\n\t"
        "18:\n\t"
        "str q16, [%[cp], #32]\n\t"
        "52:\n\t"
        "add %[cp], %[cp], %[stride]\n\t"
        "cbnz %w[first], 19f\n\t"
        "ldr q7, [%[cp], #0]\n\t"
        "fadd v17.4s, v17.4s, v7.4s\n\t"
        "19:\n\t"
        "str q17, [%[cp], #0]\n\t"
        "cmp %w[rows], #8\n\t"
        "b.lt 53f\n\t"
        "cbnz %w[first], 20f\n\t"
        "ldr q7, [%[cp], #16]\n\t"
        "fadd v18.4s, v18.4s, v7.4s\n\t"
        "20:\n\t"
        "str q18, [%[cp], #16]\n\t"
        "cmp %w[rows], #12\n\t"
        "b.lt 53f\n\t"
        "cbnz %w[first], 21f\n\t"
        "ldr q7, [%[cp], #32]\n\t"
        "fadd v19.4s, v19.4s, v7.4s\n\t"
        "21:\n\t"
        "str q19, [%[cp], #32]\n\t"
        "53:\n\t"
        "add %[cp], %[cp], %[stride]\n\t"
        "cmp %w[columns], #4\n\t"
        "b.le 6f\n\t"
        "cbnz %w[first], 22f\n\t"
        "ldr q7, [%[cp], #0]\n\t"
        "fadd v20.4s, v20.4s, v7.4s\n\t"
        "22:\n\t"
        "str q20, [%[cp], #0]\n\t"
        "cmp %w[rows], #8\n\t"
        "b.lt 54f\n\t"
        "cbnz %w[first], 23f\n\t"
        "ldr q7, [%[cp], #16]\n\t"
        "fadd v21.4s, v21.4s, v7.4s\n\t"
        "23:\n\t"
        "str q21, [%[cp], #16]\n\t"
        "cmp %w[rows], #12\n\t"
        "b.lt 54f\n\t"
        "cbnz %w[first], 24f\n\t"
        "ldr q7, [%[cp], #32]\n\t"
        "fadd v22.4s, v22.4s, v7.4s\n\t"
        "24:\n\t"
        "str q22, [%[cp], #32]\n\t"
        "54:\n\t"
        "add %[cp], %[cp], %[stride]\n\t"
        "cbnz %w[first], 25f\n\t"
        "ldr q7, [%[cp], #0]\n\t"
        "fadd v23.4s, v23.4s, v7.4s\n\t"
        "25:\n\t"
        "str q23, [%[cp], #0]\n\t"
        "cmp %w[rows], #8\n\t"
        "b.lt 55f\n\t"
        "cbnz %w[first], 26f\n\t"
        "ldr q7, [%[cp], #16]\n\t"
        "fadd v24.4s, v24.4s, v7.4s\n\t"
        "26:\n\t"
        "str q24, [%[cp], #16]\n\t"
        "cmp %w[rows], #12\n\t"
        "b.lt 55f\n\t"
        "cbnz %w[first], 27f\n\t"
        "ldr q7, [%[cp], #32]\n\t"
        "fadd v25.4s, v25.4s, v7.4s\n\t"
        "27:\n\t"
        "str q25, [%[cp], #32]\n\t"
        "55:\n\t"
        "add %[cp], %[cp], %[stride]\n\t"
        "cbnz %w[first], 28f\n\t"
        "ldr q7, [%[cp], #0]\n\t"
        "fadd v26.4s, v26.4s, v7.4s\n\t"
        "28:\n\t"
        "str q26, [%[cp], #0]\n\t"
        "cmp %w[rows], #8\n\t"
        "b.lt 56f\n\t"
        "cbnz %w[first], 29f\n\t"
        "ldr q7, [%[cp], #16]\n\t"
        "fadd v27.4s, v27.4s, v7.4s\n\t"
        "29:\n\t"
        "str q27, [%[cp], #16]\n\t"
        "cmp %w[rows], #12\n\t"
        "b.lt 56f\n\t"
        "cbnz %w[first], 30f\n\t"
        "ldr q7, [%[cp], #32]\n\t"
        "fadd v28.4s, v28.4s, v7.4s\n\t"
        "30:\n\t"
        "str q28, [%[cp], #32]\n\t"
        "56:\n\t"
        "add %[cp], %[cp], %[stride]\n\t"
        "cbnz %w[first], 31f\n\t"
        "ldr q7, [%[cp], #0]\n\t"
        "fadd v29.4s, v29.4s, v7.4s\n\t"
        "31:\n\t"
        "str q29, [%[cp], #0]\n\t"
        "cmp %w[rows], #8\n\t"
        "b.lt 57f\n\t"
        "cbnz %w[first], 32f\n\t"
        "ldr q7, [%[cp], #16]\n\t"
        "fadd v30.4s, v30.4s, v7.4s\n\t"
        "32:\n\t"
        "str q30, [%[cp], #16]\n\t"
        "cmp %w[rows], #12\n\t"
        "b.lt 57f\n\t"
        "cbnz %w[first], 33f\n\t"
        "ldr q7, [%[cp], #32]\n\t"
        "fadd v31.4s, v31.4s, v7.4s\n\t"
        "33:\n\t"
        "str q31, [%[cp], #32]\n\t"
        "57:\n\t"
        "6:\n\t"
        : [ap] "+&r"(ap), [bp] "+&r"(bp), [cp] "+&r"(cp), [depth] "+&r"(depth)
        : [stride] "r"(stride), [first] "r"(first), [rows] "r"(rows), [columns] "r"(columns)
        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
          "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24",
          "v25", "v26", "v27", "v28", "v29", "v30", "v31");
}
static void rectangular32_neon12_tile(int m, int n, int k, const float *a, const float *b, float *c,
                                      int ldc, int first)
{
    for (int j = 0; j < n; j += 8)
        for (int i = 0; i < m; i += 12)
            rectangular32_neon12_kernel(k, a + (size_t)i * k, b + (size_t)j * k,
                                        c + i + (size_t)j * ldc, ldc, first,
                                        m - i < 12 ? m - i : 12, n - j < 8 ? n - j : 8);
}
#endif

static void rectangular32_product(const camblas_task_t *task, void *opaque)
{
    rectangular32_t *w = opaque;
    /* Product index is encoded by stacking seven disjoint output planes.
     * Rows within each plane begin on a complete twelve-row A group. */
    int product = task->i0 / w->hm, row = task->i0 % w->hm;
    int bm = task->i1 - task->i0, column = task->j0, bn = task->j1 - column;
    int end_pc = w->hk;
    for (int pc = 0; pc < end_pc; pc += CAMBLAS_RECT_KC) {
        int bk = w->hk - pc < CAMBLAS_RECT_KC ? w->hk - pc : CAMBLAS_RECT_KC;
        const float *a = w->packed_a + (size_t)product * w->a_plane +
                         (size_t)pc * w->row_groups * 12 + (size_t)row * bk;
        const float *b = w->packed_b + (size_t)product * w->b_plane +
                         (size_t)pc * w->column_groups * 8 + (size_t)column * bk;
        float *c = w->products + (size_t)product * w->product_plane + row + (size_t)column * w->hm;
#if CAMBLAS_RECT32_NEON12
        if (w->neon_kernel) {
            rectangular32_neon12_tile(bm, bn, bk, a, b, c, w->hm, pc == 0 && w->source_pc == 0);
            continue;
        }
#endif
        if (camblas_sgemm_sve_amicro12_tile(bm, bn, bk, 1.0f, a, bk, b, bk, c, w->hm,
                                            pc == 0 && w->source_pc == 0)) {
            atomic_store_explicit(&w->failed, 1, memory_order_relaxed);
            return;
        }
    }
}

static void rectangular32_combine(const camblas_task_t *task, void *opaque)
{
    rectangular32_t *w = opaque;
    size_t plane = w->product_plane;
    for (int j = task->j0; j < task->j1; ++j) {
        const float *p = w->products + (size_t)j * w->hm;
        float *left = w->c + (size_t)j * w->ldc;
        float *right = left + (size_t)w->hn * w->ldc;
        int i = 0;
        for (; i + 4 <= w->hm; i += 4) {
            float32x4_t p1 = vld1q_f32(p + i), p2 = vld1q_f32(p + plane + i);
            float32x4_t p3 = vld1q_f32(p + 2 * plane + i), p4 = vld1q_f32(p + 3 * plane + i);
            float32x4_t p5 = vld1q_f32(p + 4 * plane + i), p6 = vld1q_f32(p + 5 * plane + i);
            float32x4_t p7 = vld1q_f32(p + 6 * plane + i);
            vst1q_f32(left + i, vaddq_f32(vsubq_f32(vaddq_f32(p1, p4), p5), p7));
            vst1q_f32(left + w->hm + i, vaddq_f32(p2, p4));
            vst1q_f32(right + i, vaddq_f32(p3, p5));
            vst1q_f32(right + w->hm + i, vaddq_f32(vaddq_f32(vsubq_f32(p1, p2), p3), p6));
        }
        for (; i < w->hm; ++i) {
            float p1 = p[i], p2 = p[plane + i], p3 = p[2 * plane + i], p4 = p[3 * plane + i];
            float p5 = p[4 * plane + i], p6 = p[5 * plane + i], p7 = p[6 * plane + i];
            left[i] = ((p1 + p4) - p5) + p7;
            left[w->hm + i] = p2 + p4;
            right[i] = p3 + p5;
            right[w->hm + i] = ((p1 - p2) + p3) + p6;
        }
    }
}

/* Core packed route. levels > 0 fuses the conservative range preflight into
 * the pack pass and returns 1 when the classical path should own the call;
 * levels == 0 keeps the historical unconditional behaviour. */
static int rectangular32_execute(int tb, const camblas_executor_t *executor, int workers, int m,
                                 int n, int k, const float *a, int lda, const float *b, int ldb,
                                 float *c, int ldc, void *scratch, size_t bytes, int levels,
                                 const camblas_rectangular32_segments_t *segments)
{
    size_t needed;
    if ((tb != 0 && tb != 1) || !executor || !executor->run || workers < 1 || workers > 64 || !a ||
        !b || !c || (!scratch && !segments) || (uintptr_t)scratch % _Alignof(float) ||
        camblas_experimental_rectangular32_bytes(m, n, k, &needed) ||
        (!segments && bytes < needed) || lda < m || ldb < (tb ? n : k) || ldc < m ||
        camblas_matrix_span_fits(m, k, lda, sizeof(float)) ||
        camblas_matrix_span_fits(tb ? n : k, tb ? k : n, ldb, sizeof(float)) ||
        camblas_matrix_span_fits(m, n, ldc, sizeof(float)))
        return -1;
    camblas_kernel_runtime_t runtime;
    camblas_kernel_runtime_query(&runtime);
    if (!runtime.compiled_sve || !runtime.hw_sve || runtime.vl_bits != 128)
        return -1;
    int hm = m / 2, hn = n / 2, hk = k / 2;
    rectangular32_t work = {.tb = tb,
                            .hm = hm,
                            .hn = hn,
                            .hk = hk,
                            .lda = lda,
                            .ldb = ldb,
                            .ldc = ldc,
                            .workers = workers,
                            .row_groups = (hm + 11) / 12,
                            .column_groups = (hn + 7) / 8,
                            .depth_blocks = (hk + CAMBLAS_RECT_KC - 1) / CAMBLAS_RECT_KC,
                            .c = c};
    work.neon_kernel = CAMBLAS_RECT32_NEON12 && workers >= 16 && workers <= 32 && !tb &&
                       m >= (int64_t)n * 4 && k > 1024 && !(m % 8) && !(n % 8);
    int stream = workers >= 16 && workers <= 32 && m >= (int64_t)n * 4 && k > 1024;
    int packed_k = stream && hk > CAMBLAS_RECT_KC ? CAMBLAS_RECT_KC : hk;
    work.a_plane = (size_t)work.row_groups * 12 * packed_k;
    work.b_plane = (size_t)work.column_groups * 8 * packed_k;
    work.product_plane = (size_t)hm * hn;
    if (segments) {
        size_t sizes[3];
        if (camblas_experimental_rectangular32_segment_bytes(workers, m, n, k, sizes))
            return -1;
        for (int t = 0; t < 3; ++t)
            if (!segments->data[t] || (uintptr_t)segments->data[t] % _Alignof(float) ||
                segments->bytes[t] < sizes[t])
                return -1;
        work.packed_a = segments->data[0];
        work.packed_b = segments->data[1];
        work.products = segments->data[2];
    } else {
        work.packed_a = scratch;
        work.packed_b = work.packed_a + 7 * work.a_plane;
        work.products = work.packed_b + 7 * work.b_plane;
    }
    const float *a12 = a + (size_t)hk * lda, *a21 = a + hm, *a22 = a12 + hm;
    const float *b12 = b + (tb ? (size_t)hn : (size_t)hn * ldb);
    const float *b21 = b + (tb ? (size_t)hk * ldb : (size_t)hk);
    const float *b22 = b12 + (tb ? (size_t)hk * ldb : (size_t)hk);
    const float *left[7] = {a, a21, a, a22, a, a21, a12};
    const float *left2[7] = {a22, a22, NULL, NULL, a12, a, a22};
    const float *right[7] = {b, b, b12, b21, b22, b, b21};
    const float *right2[7] = {b22, NULL, b22, b, NULL, b12, b22};
    for (int p = 0; p < 7; ++p) {
        work.a[p] = left[p];
        work.a2[p] = left2[p];
        work.b[p] = right[p];
        work.b2[p] = right2[p];
    }
    atomic_init(&work.failed, 0);
    camblas_task_t workers_tasks[64], product_tasks[7 * 64];
    for (int t = 0; t < workers; ++t)
        workers_tasks[t] = (camblas_task_t){t, t + 1, hn * t / workers, hn * (t + 1) / workers};
    int row_grid = workers >= 64 ? 8 : workers >= 16 ? 4 : 1;
    int column_grid = workers >= 64 ? 8 : workers >= 16 ? 4 : 1;
    int rows = work.row_groups < row_grid ? work.row_groups : row_grid;
    if (stream && workers == 16) {
        row_grid = CAMBLAS_RECT_ROWS16;
        column_grid = 16 / row_grid;
        rows = work.row_groups < row_grid ? work.row_groups : row_grid;
    }
    int columns = work.column_groups < column_grid ? work.column_groups : column_grid;
    int count = 0;
    if (work.neon_kernel) {
        for (int rb = 0; rb < rows; ++rb)
            for (int cb = 0; cb < columns; ++cb)
                for (int p = 0; p < 7; ++p) {
                    int r = (rb + p) % rows, cc = (cb + p) % columns;
                    int i0 = work.row_groups * r / rows * 12;
                    int i1 = work.row_groups * (r + 1) / rows * 12;
                    int j0 = work.column_groups * cc / columns * 8;
                    int j1 = work.column_groups * (cc + 1) / columns * 8;
                    if (i1 > hm)
                        i1 = hm;
                    if (j1 > hn)
                        j1 = hn;
                    product_tasks[count++] = (camblas_task_t){p * hm + i0, p * hm + i1, j0, j1};
                }
    } else {
        for (int p = 0; p < 7; ++p)
            for (int rb = 0; rb < rows; ++rb)
                for (int cb = 0; cb < columns; ++cb) {
                    int i0 = work.row_groups * rb / rows * 12;
                    int i1 = work.row_groups * (rb + 1) / rows * 12;
                    int j0 = work.column_groups * cb / columns * 8;
                    int j1 = work.column_groups * (cb + 1) / columns * 8;
                    if (i1 > hm)
                        i1 = hm;
                    if (j1 > hn)
                        j1 = hn;
                    product_tasks[count++] = (camblas_task_t){p * hm + i0, p * hm + i1, j0, j1};
                }
    }
    if (!stream) {
        /* Each synchronous stage completes before its outputs are consumed. The
     * pack pass also produces the per-worker operand maxima, so the range
     * preflight costs no separate scan of A and B. */
        if (executor->run(rectangular32_pack, workers_tasks, workers, &work, executor->user_data))
            return -1;
        if (levels) {
            float a_peak = 0, b_peak = 0;
            for (int t = 0; t < workers; ++t) {
                a_peak = fmaxf(a_peak, work.a_max[t]);
                b_peak = fmaxf(b_peak, work.b_max[t]);
            }
            if (!rectangular32_range_verdict(a_peak, b_peak, k, levels))
                return 1;
        }
        if (executor->run(rectangular32_product, product_tasks, count, &work,
                          executor->user_data) ||
            atomic_load_explicit(&work.failed, memory_order_relaxed))
            return -1;
        return executor->run(rectangular32_combine, workers_tasks, workers, &work,
                             executor->user_data);
    }
    /* Private products accumulate across slabs. A late range rejection
     * leaves caller C untouched so the bridge can use classical GEMM. */
    float a_peak = 0, b_peak = 0;
    for (int origin = 0; origin < hk; origin += CAMBLAS_RECT_KC) {
        work.source_pc = origin;
        work.hk = hk - origin < CAMBLAS_RECT_KC ? hk - origin : CAMBLAS_RECT_KC;
        work.depth_blocks = 1;
        if (executor->run(rectangular32_pack, workers_tasks, workers, &work, executor->user_data))
            return -1;
        if (levels) {
            for (int t = 0; t < workers; t++) {
                a_peak = fmaxf(a_peak, work.a_max[t]);
                b_peak = fmaxf(b_peak, work.b_max[t]);
            }
            if (!rectangular32_range_verdict(a_peak, b_peak, k, levels))
                return 1;
        }
        if (executor->run(rectangular32_product, product_tasks, count, &work,
                          executor->user_data) ||
            atomic_load_explicit(&work.failed, memory_order_relaxed))
            return -1;
    }
    return executor->run(rectangular32_combine, workers_tasks, workers, &work, executor->user_data);
}

int camblas_experimental_rectangular32_f32_op(int tb, const camblas_executor_t *executor,
                                              int workers, int m, int n, int k, const float *a,
                                              int lda, const float *b, int ldb, float *c, int ldc,
                                              void *scratch, size_t bytes)
{
    return rectangular32_execute(tb, executor, workers, m, n, k, a, lda, b, ldb, c, ldc, scratch,
                                 bytes, 0, NULL);
}

int camblas_experimental_rectangular32_f32_op_checked(int tb, const camblas_executor_t *executor,
                                                      int workers, int m, int n, int k,
                                                      const float *a, int lda, const float *b,
                                                      int ldb, float *c, int ldc, void *scratch,
                                                      size_t bytes, int levels)
{
    return rectangular32_execute(tb, executor, workers, m, n, k, a, lda, b, ldb, c, ldc, scratch,
                                 bytes, levels, NULL);
}

/* Convenience wrapper for an untransposed B operand. */
int camblas_experimental_rectangular32_f32(const camblas_executor_t *executor, int workers, int m,
                                           int n, int k, const float *a, int lda, const float *b,
                                           int ldb, float *c, int ldc, void *scratch, size_t bytes)
{
    return camblas_experimental_rectangular32_f32_op(0, executor, workers, m, n, k, a, lda, b, ldb,
                                                     c, ldc, scratch, bytes);
}

int camblas_experimental_rectangular32_f32_segments_checked(
    int tb, const camblas_executor_t *executor, int workers, int m, int n, int k, const float *a,
    int lda, const float *b, int ldb, float *c, int ldc,
    const camblas_rectangular32_segments_t *segments, int levels)
{
    if (!segments)
        return -1;
    return rectangular32_execute(tb, executor, workers, m, n, k, a, lda, b, ldb, c, ldc, NULL, 0,
                                 levels, segments);
}
