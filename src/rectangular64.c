/* Experimental FP64 rectangular Strassen with fused packed transforms.
 * Packing, all seven products and recombination each use one executor batch.
 * No input-dependent values survive the call. This is a different floating-
 * point algorithm from classical GEMM and requires separate error checks. */
#include "camblas.h"
#include "camblas_executor.h"
#include "rectangular64.h"
#include "packed.h"
#include "kernels.h"
#include "gemm_bounds.h"
#include <arm_neon.h>
#include <stdatomic.h>
#include <math.h>
#include <stdint.h>

int camblas_batch_dgemm_amicro6_tile(int m, int n, int k, double alpha, const double *a, int lda,
                                     const double *b, int ldb, double *c, int ldc, int initialize);

/* Packing and recombination share this seven-product order:
 * P1 = (A11 + A22) * (B11 + B22), P2 = (A21 + A22) * B11,
 * P3 = A11 * (B12 - B22),         P4 = A22 * (B21 - B11),
 * P5 = (A11 + A12) * B22,         P6 = (A21 - A11) * (B11 + B12),
 * P7 = (A12 - A22) * (B21 + B22).
 * Quadrant indices refer to logical matrices, including transposed B. */

/* Research-only exports; the public BLAS ABI does not expose this policy. */
int camblas_experimental_rectangular64_available(void)
{
    camblas_kernel_runtime_t runtime;
    camblas_kernel_runtime_query(&runtime);
    return runtime.compiled_sve && runtime.hw_sve && runtime.vl_bits == 128;
}

int camblas_experimental_rectangular64_bytes(int m, int n, int k, size_t *bytes)
{
    if (!bytes || m < 2 || n < 2 || k < 2 || m > 8192 || n > 8192 || k > 8192 || m % 2 || n % 2 ||
        k % 2)
        return -1;
    uint64_t hm = (unsigned)m / 2, hn = (unsigned)n / 2, hk = (unsigned)k / 2;
    uint64_t apad = (hm + 5) / 6 * 6, bpad = (hn + 7) / 8 * 8;
    uint64_t packed_k = hk;
    uint64_t elements = 7 * ((apad + bpad) * packed_k + hm * hn);
    if (elements > PTRDIFF_MAX / sizeof(double) || elements > SIZE_MAX / sizeof(double))
        return -1;
    *bytes = (size_t)elements * sizeof(double);
    return 0;
}

typedef struct {
    int tb, hm, hn, hk, lda, ldb, ldc, workers, row_groups, column_groups, depth_blocks;
    size_t a_plane, b_plane, product_plane;
    const double *a[7], *a2[7], *b[7], *b2[7];
    double *packed_a, *packed_b, *products, *c;
    double a_max[64], b_max[64];
    atomic_int failed;
} rectangular64_t;

static const int rectangular64_sign_b[7] = {1, 0, -1, -1, 0, 1, 1};

/* The range preflight rides on the pack pass: every input element is already
 * loaded exactly here, so the abs-max scan costs no extra memory traffic.
 * Vector bodies keep the NEON max NaN-suppressing behaviour of the standalone
 * scan; scalar tails flag non-finite values like the standalone scalar tail. */
static float64x2_t rectangular64_abs_max2(float64x2_t acc, float64x2_t value)
{
    return vmaxq_f64(acc, vabsq_f64(value));
}

static void rectangular64_scan_max(double *acc, double value)
{
    double magnitude = fabs(value);
    if (!isfinite(magnitude))
        *acc = INFINITY;
    else if (magnitude > *acc)
        *acc = magnitude;
}

static void rectangular64_store_max(float64x2_t acc, double tail, double *out)
{
    double peak = vmaxvq_f64(acc);
    double merged = isfinite(peak) ? fmax(peak, tail) : INFINITY;
    if (merged > *out)
        *out = merged;
}

static float64x2_t rectangular64_load_sum(const double *left, const double *right, int sign,
                                          float64x2_t *absmax)
{
    float64x2_t value = vld1q_f64(left);
    if (absmax)
        *absmax = rectangular64_abs_max2(*absmax, value);
    if (right) {
        float64x2_t second = vld1q_f64(right);
        if (absmax)
            *absmax = rectangular64_abs_max2(*absmax, second);
        value = sign == 1 ? vaddq_f64(value, second) : vsubq_f64(value, second);
    }
    return value;
}

/* Transpose two adjacent columns and depths into an eight-column B group. */
static void rectangular64_transpose_two(const double *left, const double *right, int sign,
                                        size_t stride, double *destination, float64x2_t *absmax)
{
    float64x2_t a = rectangular64_load_sum(left, right, sign, absmax);
    float64x2_t b =
        rectangular64_load_sum(left + stride, right ? right + stride : NULL, sign, absmax);
    vst1q_f64(destination, vtrn1q_f64(a, b));
    vst1q_f64(destination + 8, vtrn2q_f64(a, b));
}

static void rectangular64_pack_b_block(rectangular64_t *w, int product, int column, int pc, int bk,
                                       double *bmax)
{
    int count = w->hn - column < 8 ? w->hn - column : 8;
    double *out = w->packed_b + (size_t)product * w->b_plane + (size_t)pc * w->column_groups * 8 +
                  (size_t)column * bk;
    size_t initial_offset = w->tb ? column + (size_t)pc * w->ldb : pc + (size_t)column * w->ldb;
    const double *left = w->b[product] + initial_offset;
    const double *right = w->b2[product] ? w->b2[product] + initial_offset : NULL;
    float64x2_t acc = vdupq_n_f64(0);
    double tail = 0;

    if (w->tb) {
        left = w->b[product] + column + (size_t)pc * w->ldb;
        right = w->b2[product] ? w->b2[product] + column + (size_t)pc * w->ldb : NULL;
        for (int q = 0; q < bk; ++q) {
            int j = 0;
            for (; j + 2 <= count; j += 2)
                vst1q_f64(out + (size_t)q * 8 + j,
                          rectangular64_load_sum(left + j + (size_t)q * w->ldb,
                                                 right ? right + j + (size_t)q * w->ldb : NULL,
                                                 rectangular64_sign_b[product], &acc));
            for (; j < 8; ++j) {
                double value = j < count ? left[j + (size_t)q * w->ldb] : 0;
                if (j < count)
                    rectangular64_scan_max(&tail, value);
                if (right && j < count) {
                    double second = right[j + (size_t)q * w->ldb];
                    rectangular64_scan_max(&tail, second);
                    value = rectangular64_sign_b[product] == 1 ? value + second : value - second;
                }
                out[(size_t)q * 8 + j] = value;
            }
        }
        rectangular64_store_max(acc, tail, bmax);
        return;
    }
    int q = 0;
    if (count == 8)
        for (; q + 2 <= bk; q += 2)
            for (int j = 0; j < 8; j += 2) {
                size_t source = q + (size_t)j * w->ldb;
                rectangular64_transpose_two(left + source, right ? right + source : NULL,
                                            rectangular64_sign_b[product], w->ldb,
                                            out + (size_t)q * 8 + j, &acc);
            }
    for (; q < bk; ++q)
        for (int j = 0; j < 8; ++j) {
            double value = j < count ? left[q + (size_t)j * w->ldb] : 0;
            if (j < count)
                rectangular64_scan_max(&tail, value);
            if (right && j < count) {
                double second = right[q + (size_t)j * w->ldb];
                rectangular64_scan_max(&tail, second);
                value = rectangular64_sign_b[product] == 1 ? value + second : value - second;
            }
            out[(size_t)q * 8 + j] = value;
        }
    rectangular64_store_max(acc, tail, bmax);
}

/* Return a small logical B tile as contiguous output-column vectors. */
static void rectangular64_load_b_tile(const double *base, int stride, int transposed,
                                      float64x2_t out[2])
{
    if (transposed) {
        for (int q = 0; q < 2; ++q)
            out[q] = vld1q_f64(base + (size_t)q * stride);
    } else {
        float64x2_t a = vld1q_f64(base), b = vld1q_f64(base + stride);
        out[0] = vtrn1q_f64(a, b);
        out[1] = vtrn2q_f64(a, b);
    }
}

/* Load four quadrants once and produce all seven B transforms. */
static void rectangular64_pack_all_b(rectangular64_t *w, int column, int pc, int bk, double *bmax)
{
    int count = w->hn - column < 8 ? w->hn - column : 8;
    const double *quadrants[4] = {w->b[0], w->b[2], w->b[3], w->b[4]};
    size_t base_offset = w->tb ? column + (size_t)pc * w->ldb : pc + (size_t)column * w->ldb;
    double *base = w->packed_b + (size_t)pc * w->column_groups * 8 + (size_t)column * bk;
    size_t plane = w->b_plane;
    float64x2_t acc = vdupq_n_f64(0);
    double tail = 0;
    int q = 0;
    if (count == 8)
        for (; q + 2 <= bk; q += 2)
            for (int j = 0; j < 8; j += 2) {
                size_t offset =
                    base_offset + (w->tb ? j + (size_t)q * w->ldb : q + (size_t)j * w->ldb);
                float64x2_t b11[2], b12[2], b21[2], b22[2];
                rectangular64_load_b_tile(quadrants[0] + offset, w->ldb, w->tb, b11);
                rectangular64_load_b_tile(quadrants[1] + offset, w->ldb, w->tb, b12);
                rectangular64_load_b_tile(quadrants[2] + offset, w->ldb, w->tb, b21);
                rectangular64_load_b_tile(quadrants[3] + offset, w->ldb, w->tb, b22);
                for (int l = 0; l < 2; ++l) {
                    acc = rectangular64_abs_max2(acc, b11[l]);
                    acc = rectangular64_abs_max2(acc, b12[l]);
                    acc = rectangular64_abs_max2(acc, b21[l]);
                    acc = rectangular64_abs_max2(acc, b22[l]);
                    double *out = base + (size_t)(q + l) * 8 + j;
                    vst1q_f64(out, vaddq_f64(b11[l], b22[l]));
                    vst1q_f64(out + plane, b11[l]);
                    vst1q_f64(out + 2 * plane, vsubq_f64(b12[l], b22[l]));
                    vst1q_f64(out + 3 * plane, vsubq_f64(b21[l], b11[l]));
                    vst1q_f64(out + 4 * plane, b22[l]);
                    vst1q_f64(out + 5 * plane, vaddq_f64(b11[l], b12[l]));
                    vst1q_f64(out + 6 * plane, vaddq_f64(b21[l], b22[l]));
                }
            }
    for (; q < bk; ++q)
        for (int j = 0; j < 8; ++j) {
            size_t offset = base_offset + (w->tb ? j + (size_t)q * w->ldb : q + (size_t)j * w->ldb);
            double b11 = j < count ? quadrants[0][offset] : 0;
            double b12 = j < count ? quadrants[1][offset] : 0;
            double b21 = j < count ? quadrants[2][offset] : 0;
            double b22 = j < count ? quadrants[3][offset] : 0;
            if (j < count) {
                rectangular64_scan_max(&tail, b11);
                rectangular64_scan_max(&tail, b12);
                rectangular64_scan_max(&tail, b21);
                rectangular64_scan_max(&tail, b22);
            }
            double *out = base + (size_t)q * 8 + j;
            out[0] = b11 + b22;
            out[plane] = b11;
            out[2 * plane] = b12 - b22;
            out[3 * plane] = b21 - b11;
            out[4 * plane] = b22;
            out[5 * plane] = b11 + b12;
            out[6 * plane] = b21 + b22;
        }
    rectangular64_store_max(acc, tail, bmax);
}
/* Reuse four loaded quadrants for all seven A transforms. A small row/depth
 * tile limits the live input set when original columns have a large stride. */
static void rectangular64_pack_all_a(rectangular64_t *w, int first, int last, int pc, int bk,
                                     double *amax)
{
    size_t plane = w->a_plane;
    float64x2_t acc = vdupq_n_f64(0);
    double tail = 0;
    for (int q0 = 0; q0 < bk; q0 += 16) {
        int end = bk - q0 < 16 ? bk : q0 + 16;
        for (int group = first; group < last; ++group) {
            int row = group * 6, count = w->hm - row < 6 ? w->hm - row : 6;
            double *base = w->packed_a + (size_t)pc * w->row_groups * 6 + (size_t)row * bk;
            for (int q = q0; q < end; ++q) {
                size_t source = row + (size_t)(pc + q) * w->lda;
                double *out = base + (size_t)q * 6;
                int i = 0;
                for (; i + 2 <= count; i += 2) {
                    float64x2_t a11 = vld1q_f64(w->a[0] + source + i);
                    float64x2_t a21 = vld1q_f64(w->a[1] + source + i);
                    float64x2_t a12 = vld1q_f64(w->a[6] + source + i);
                    float64x2_t a22 = vld1q_f64(w->a[3] + source + i);
                    acc = rectangular64_abs_max2(acc, a11);
                    acc = rectangular64_abs_max2(acc, a21);
                    acc = rectangular64_abs_max2(acc, a12);
                    acc = rectangular64_abs_max2(acc, a22);
                    vst1q_f64(out + i, vaddq_f64(a11, a22));
                    vst1q_f64(out + plane + i, vaddq_f64(a21, a22));
                    vst1q_f64(out + 2 * plane + i, a11);
                    vst1q_f64(out + 3 * plane + i, a22);
                    vst1q_f64(out + 4 * plane + i, vaddq_f64(a11, a12));
                    vst1q_f64(out + 5 * plane + i, vsubq_f64(a21, a11));
                    vst1q_f64(out + 6 * plane + i, vsubq_f64(a12, a22));
                }
                for (; i < count; ++i) {
                    double a11 = w->a[0][source + i], a21 = w->a[1][source + i];
                    double a12 = w->a[6][source + i], a22 = w->a[3][source + i];
                    rectangular64_scan_max(&tail, a11);
                    rectangular64_scan_max(&tail, a21);
                    rectangular64_scan_max(&tail, a12);
                    rectangular64_scan_max(&tail, a22);
                    out[i] = a11 + a22;
                    out[plane + i] = a21 + a22;
                    out[2 * plane + i] = a11;
                    out[3 * plane + i] = a22;
                    out[4 * plane + i] = a11 + a12;
                    out[5 * plane + i] = a21 - a11;
                    out[6 * plane + i] = a12 - a22;
                }
                for (; i < 6; ++i)
                    for (int p = 0; p < 7; ++p)
                        out[(size_t)p * plane + i] = 0;
            }
        }
    }
    rectangular64_store_max(acc, tail, amax);
}

#ifndef RECT64_PACK_WEIGHT
#define RECT64_PACK_WEIGHT 6
#endif
static void rectangular64_pack(const camblas_task_t *task, void *opaque)
{
    rectangular64_t *w = opaque;
    double a_max = 0, b_max = 0;
    int per_block = 4;
    while (per_block > 1 &&
           ((w->row_groups + per_block - 1) / per_block) * w->depth_blocks < w->workers)
        per_block /= 2;
    int a_blocks = (w->row_groups + per_block - 1) / per_block;
    /* Weight an all-seven A block more heavily than one B micro-panel when
     * distributing work. Only the first index in each weighted slot owns it. */
    int weight = RECT64_PACK_WEIGHT * per_block, groups = a_blocks * weight + 7 * w->column_groups;
    int total = groups * w->depth_blocks, worker = task->i0;
    for (int index = total * worker / w->workers; index < total * (worker + 1) / w->workers;
         ++index) {
        int group = index % groups, pc = (index / groups) * 256;
        int bk = w->hk - pc < 256 ? w->hk - pc : 256;
        if (group < a_blocks * weight) {
            if (group % weight)
                continue;
            int first = group / weight * per_block, last = first + per_block;
            if (last > w->row_groups)
                last = w->row_groups;
            rectangular64_pack_all_a(w, first, last, pc, bk, &a_max);
        } else {
            group -= a_blocks * weight;
            if (!w->tb)
                rectangular64_pack_b_block(w, group / w->column_groups,
                                           group % w->column_groups * 8, pc, bk, &b_max);
            else if (group % 7 == 0)
                rectangular64_pack_all_b(w, group / 7 * 8, pc, bk, &b_max);
        }
    }
    w->a_max[worker] = a_max;
    w->b_max[worker] = b_max;
}

static void rectangular64_product(const camblas_task_t *task, void *opaque)
{
    rectangular64_t *w = opaque;
    /* Product index is encoded by stacking seven disjoint output planes.
     * Rows within each plane begin on a complete six-row A group. */
    int product = task->i0 / w->hm, row = task->i0 % w->hm;
    int bm = task->i1 - task->i0, column = task->j0, bn = task->j1 - column;
    int end_pc = w->hk;
    for (int pc = 0; pc < end_pc; pc += 256) {
        int bk = w->hk - pc < 256 ? w->hk - pc : 256;
        const double *a = w->packed_a + (size_t)product * w->a_plane +
                          (size_t)pc * w->row_groups * 6 + (size_t)row * bk;
        const double *b = w->packed_b + (size_t)product * w->b_plane +
                          (size_t)pc * w->column_groups * 8 + (size_t)column * bk;
        double *c = w->products + (size_t)product * w->product_plane + row + (size_t)column * w->hm;
        int status;
        if (!w->tb)
            status =
                camblas_dgemm_sve_amicro6_tile(bm, bn, bk, 1.0, a, bk, b, bk, c, w->hm, pc == 0);
        else
            status =
                camblas_batch_dgemm_amicro6_tile(bm, bn, bk, 1.0, a, bk, b, bk, c, w->hm, pc == 0);
        if (status) {
            atomic_store_explicit(&w->failed, 1, memory_order_relaxed);
            return;
        }
    }
}

static void rectangular64_combine(const camblas_task_t *task, void *opaque)
{
    rectangular64_t *w = opaque;
    size_t plane = w->product_plane;
    for (int j = task->j0; j < task->j1; ++j) {
        const double *p = w->products + (size_t)j * w->hm;
        double *left = w->c + (size_t)j * w->ldc;
        double *right = left + (size_t)w->hn * w->ldc;
        int i = 0;
        for (; i + 2 <= w->hm; i += 2) {
            float64x2_t p1 = vld1q_f64(p + i), p2 = vld1q_f64(p + plane + i);
            float64x2_t p3 = vld1q_f64(p + 2 * plane + i), p4 = vld1q_f64(p + 3 * plane + i);
            float64x2_t p5 = vld1q_f64(p + 4 * plane + i), p6 = vld1q_f64(p + 5 * plane + i);
            float64x2_t p7 = vld1q_f64(p + 6 * plane + i);
            vst1q_f64(left + i, vaddq_f64(vsubq_f64(vaddq_f64(p1, p4), p5), p7));
            vst1q_f64(left + w->hm + i, vaddq_f64(p2, p4));
            vst1q_f64(right + i, vaddq_f64(p3, p5));
            vst1q_f64(right + w->hm + i, vaddq_f64(vaddq_f64(vsubq_f64(p1, p2), p3), p6));
        }
        for (; i < w->hm; ++i) {
            double p1 = p[i], p2 = p[plane + i], p3 = p[2 * plane + i], p4 = p[3 * plane + i];
            double p5 = p[4 * plane + i], p6 = p[5 * plane + i], p7 = p[6 * plane + i];
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
static int rectangular64_execute(int tb, const camblas_executor_t *executor, int workers, int m,
                                 int n, int k, const double *a, int lda, const double *b, int ldb,
                                 double *c, int ldc, void *scratch, size_t bytes, int levels)
{
    size_t needed;
    if ((tb != 0 && tb != 1) || !executor || !executor->run || workers < 1 || workers > 64 || !a ||
        !b || !c || !scratch || (uintptr_t)scratch % _Alignof(double) ||
        camblas_experimental_rectangular64_bytes(m, n, k, &needed) || bytes < needed || lda < m ||
        ldb < (tb ? n : k) || ldc < m || camblas_matrix_span_fits(m, k, lda, sizeof(double)) ||
        camblas_matrix_span_fits(tb ? n : k, tb ? k : n, ldb, sizeof(double)) ||
        camblas_matrix_span_fits(m, n, ldc, sizeof(double)))
        return -1;
    camblas_kernel_runtime_t runtime;
    camblas_kernel_runtime_query(&runtime);
    if (!runtime.compiled_sve || !runtime.hw_sve || runtime.vl_bits != 128)
        return -1;
    int hm = m / 2, hn = n / 2, hk = k / 2;
    rectangular64_t work = {.tb = tb,
                            .hm = hm,
                            .hn = hn,
                            .hk = hk,
                            .lda = lda,
                            .ldb = ldb,
                            .ldc = ldc,
                            .workers = workers,
                            .row_groups = (hm + 5) / 6,
                            .column_groups = (hn + 7) / 8,
                            .depth_blocks = (hk + 256 - 1) / 256,
                            .c = c};
    int packed_k = hk;
    work.a_plane = (size_t)work.row_groups * 6 * packed_k;
    work.b_plane = (size_t)work.column_groups * 8 * packed_k;
    work.product_plane = (size_t)hm * hn;
    work.packed_a = scratch;
    work.packed_b = work.packed_a + 7 * work.a_plane;
    work.products = work.packed_b + 7 * work.b_plane;
    const double *a12 = a + (size_t)hk * lda, *a21 = a + hm, *a22 = a12 + hm;
    const double *b12 = b + (tb ? (size_t)hn : (size_t)hn * ldb);
    const double *b21 = b + (tb ? (size_t)hk * ldb : (size_t)hk);
    const double *b22 = b12 + (tb ? (size_t)hk * ldb : (size_t)hk);
    const double *left[7] = {a, a21, a, a22, a, a21, a12};
    const double *left2[7] = {a22, a22, NULL, NULL, a12, a, a22};
    const double *right[7] = {b, b, b12, b21, b22, b, b21};
    const double *right2[7] = {b22, NULL, b22, b, NULL, b12, b22};
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
    int columns = work.column_groups < column_grid ? work.column_groups : column_grid;
    int count = 0;
    /* Full 16/64-thread grids give each static executor chunk seven products.
     * Rotate row and column tiles to spread partial micro-panels across workers.
     * Each product/tile pair occurs once, with disjoint output ownership. */
    for (int rb = 0; rb < rows; ++rb)
        for (int cb = 0; cb < columns; ++cb)
            for (int p = 0; p < 7; ++p) {
                int r = (rb + p) % rows, c = (cb + p) % columns;
                int i0 = work.row_groups * r / rows * 6;
                int i1 = work.row_groups * (r + 1) / rows * 6;
                int j0 = work.column_groups * c / columns * 8;
                int j1 = work.column_groups * (c + 1) / columns * 8;
                if (i1 > hm)
                    i1 = hm;
                if (j1 > hn)
                    j1 = hn;
                product_tasks[count++] = (camblas_task_t){p * hm + i0, p * hm + i1, j0, j1};
            }
    /* Each synchronous stage completes before its outputs are consumed. The
     * pack pass also produces the per-worker operand maxima, so the range
     * preflight costs no separate scan of A and B. */
    if (executor->run(rectangular64_pack, workers_tasks, workers, &work, executor->user_data))
        return -1;
    if (levels) {
        double a_peak = 0, b_peak = 0;
        for (int t = 0; t < workers; ++t) {
            a_peak = fmax(a_peak, work.a_max[t]);
            b_peak = fmax(b_peak, work.b_max[t]);
        }
        if (!rectangular64_range_verdict(a_peak, b_peak, k, levels))
            return 1;
    }
    if (executor->run(rectangular64_product, product_tasks, count, &work, executor->user_data) ||
        atomic_load_explicit(&work.failed, memory_order_relaxed))
        return -1;
    return executor->run(rectangular64_combine, workers_tasks, workers, &work, executor->user_data);
}

int camblas_experimental_rectangular64_f64_op(int tb, const camblas_executor_t *executor,
                                              int workers, int m, int n, int k, const double *a,
                                              int lda, const double *b, int ldb, double *c, int ldc,
                                              void *scratch, size_t bytes)
{
    return rectangular64_execute(tb, executor, workers, m, n, k, a, lda, b, ldb, c, ldc, scratch,
                                 bytes, 0);
}

int camblas_experimental_rectangular64_f64_op_checked(int tb, const camblas_executor_t *executor,
                                                      int workers, int m, int n, int k,
                                                      const double *a, int lda, const double *b,
                                                      int ldb, double *c, int ldc, void *scratch,
                                                      size_t bytes, int levels)
{
    return rectangular64_execute(tb, executor, workers, m, n, k, a, lda, b, ldb, c, ldc, scratch,
                                 bytes, levels);
}

/* Convenience wrapper for an untransposed B operand. */
int camblas_experimental_rectangular64_f64(const camblas_executor_t *executor, int workers, int m,
                                           int n, int k, const double *a, int lda, const double *b,
                                           int ldb, double *c, int ldc, void *scratch, size_t bytes)
{
    return camblas_experimental_rectangular64_f64_op(0, executor, workers, m, n, k, a, lda, b, ldb,
                                                     c, ldc, scratch, bytes);
}
