#ifndef CAMBLAS_EXPERIMENTAL_STRASSEN_ONE_LEVEL_H
#define CAMBLAS_EXPERIMENTAL_STRASSEN_ONE_LEVEL_H
/* Experimental square NN, alpha=1, beta=0 algorithm. Not public BLAS dispatch.
 * Scratch allocation is reusable, but every sum, product and recombination
 * is recomputed on every call. Numerical behaviour differs from classical
 * GEMM, particularly for cancellation and extreme magnitudes. */
#include "camblas.h"
#include "camblas_executor.h"
#include "camblas_planner.h"
#include "camblas_workspace.h"
#include "fused_sum_api.h"
#include "benchmark_alloc.h"
#include <stddef.h>
#include <stdlib.h>
#include <arm_neon.h>
#include <time.h>
#ifndef STRASSEN_FUSED_SUMS
#define STRASSEN_FUSED_SUMS 0
#endif
#define STRASSEN_PLANES (STRASSEN_FUSED_SUMS ? 7 : 17)
#define STRASSEN_PRODUCT_BASE (STRASSEN_FUSED_SUMS ? 0 : 10)
#ifndef STRASSEN_LEVELS
#define STRASSEN_LEVELS 1
#endif
#ifndef STRASSEN_DIRECT_CHILD
#define STRASSEN_DIRECT_CHILD 0
#endif
#if STRASSEN_DIRECT_CHILD != 0 && STRASSEN_DIRECT_CHILD != 1
#error "Direct child selector must be Boolean"
#endif
#if STRASSEN_LEVELS < 1 || STRASSEN_LEVELS > 2
#error "Research recursion supports one or two levels"
#endif
#if STRASSEN_LEVELS > 1 && STRASSEN_FUSED_SUMS
#error "Recursive materialised experiment cannot use fused sums"
#endif
#ifndef STRASSEN_PROFILE
#define STRASSEN_PROFILE 0
#endif
#ifndef STRASSEN_WINOGRAD
#define STRASSEN_WINOGRAD 0
#endif
#if STRASSEN_FUSED_SUMS && STRASSEN_WINOGRAD
#error "Fused sums currently support the ordinary Strassen schedule only"
#endif
#ifndef STRASSEN_VECTOR_TRANSFORMS
#define STRASSEN_VECTOR_TRANSFORMS 0
#endif
#ifndef STRASSEN_PLANE_PAD_BYTES
#define STRASSEN_PLANE_PAD_BYTES 0
#endif
#if STRASSEN_PLANE_PAD_BYTES < 0 || STRASSEN_PLANE_PAD_BYTES % 64 != 0
#error "Strassen plane padding must be a nonnegative multiple of 64 bytes"
#endif

typedef struct {
    int n, fp64, levels;
    size_t bytes;
    void *data;
    long long phase_ns[3];
} strassen_scratch_t;

static long long strassen_now(void)
{
    if (!STRASSEN_PROFILE)
        return 0;
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t))
        abort();
    return (long long)t.tv_sec * 1000000000LL + t.tv_nsec;
}

static int strassen_scratch_init(strassen_scratch_t *s, int n, int fp64)
{
    if (!s || n < 2 || n > 8192 || n % 2 || (fp64 != 0 && fp64 != 1))
        return -1;
    size_t h = (size_t)n / 2,
           bytes = STRASSEN_PLANES *
                   (h * h * (fp64 ? sizeof(double) : sizeof(float)) + STRASSEN_PLANE_PAD_BYTES);
    int levels = STRASSEN_LEVELS == 2 && n % 4 == 0 ? 2 : 1;
    if (levels == 2)
        bytes += ((STRASSEN_DIRECT_CHILD ? 0 : 2 * h * h) + STRASSEN_PLANES * (h / 2) * (h / 2)) *
                     (fp64 ? sizeof(double) : sizeof(float)) +
                 STRASSEN_PLANES * STRASSEN_PLANE_PAD_BYTES;
    void *data = benchmark_alloc(bytes);
    if (!data)
        return -1;
    *s = (strassen_scratch_t){.n = n, .fp64 = fp64, .levels = levels, .bytes = bytes, .data = data};
    return 0;
}

typedef struct {
    int n, combine, lda, ldb;
    const void *a, *b;
    void *c, *scratch;
} strassen_task_context_t;

#define DEFINE_STRASSEN_CALLBACK(SUFFIX, TYPE, VTYPE, LANES, LOAD, STORE, ADD, SUB)                \
    static void strassen_task_##SUFFIX(const camblas_task_t *task, void *opaque)                   \
    {                                                                                              \
        strassen_task_context_t *g = opaque;                                                       \
        int n = g->n, h = n / 2;                                                                   \
        size_t plane = (size_t)h * h + STRASSEN_PLANE_PAD_BYTES / sizeof(TYPE);                    \
        TYPE *s = g->scratch, *c = g->c;                                                           \
        const TYPE *a = g->a, *b = g->b;                                                           \
        for (int j = task->j0; j < task->j1; ++j) {                                                \
            int i = 0;                                                                             \
            if (STRASSEN_VECTOR_TRANSFORMS)                                                        \
                for (; i + LANES <= h; i += LANES) {                                               \
                    size_t q = (size_t)j * h + i, p = (size_t)j * n + i, off = (size_t)h * n;      \
                    if (!g->combine) {                                                             \
                        size_t pa = (size_t)j * g->lda + i, pb = (size_t)j * g->ldb + i,           \
                               oa = (size_t)h * g->lda, ob = (size_t)h * g->ldb;                   \
                        VTYPE a11 = LOAD(a + pa), a21 = LOAD(a + pa + h), a12 = LOAD(a + pa + oa), \
                              a22 = LOAD(a + pa + oa + h);                                         \
                        VTYPE b11 = LOAD(b + pb), b21 = LOAD(b + pb + h), b12 = LOAD(b + pb + ob), \
                              b22 = LOAD(b + pb + ob + h);                                         \
                        if (STRASSEN_WINOGRAD) {                                                   \
                            VTYPE s1 = ADD(a21, a22), s2 = SUB(s1, a11);                           \
                            VTYPE t1 = SUB(b12, b11), t2 = SUB(b22, t1);                           \
                            STORE(s + q, s1);                                                      \
                            STORE(s + plane + q, s2);                                              \
                            STORE(s + 2 * plane + q, SUB(a11, a21));                               \
                            STORE(s + 3 * plane + q, SUB(a12, s2));                                \
                            STORE(s + 4 * plane + q, t1);                                          \
                            STORE(s + 5 * plane + q, t2);                                          \
                            STORE(s + 6 * plane + q, SUB(b22, b12));                               \
                            STORE(s + 7 * plane + q, SUB(t2, b21));                                \
                        } else {                                                                   \
                            STORE(s + q, ADD(a11, a22));                                           \
                            STORE(s + plane + q, ADD(b11, b22));                                   \
                            STORE(s + 2 * plane + q, ADD(a21, a22));                               \
                            STORE(s + 3 * plane + q, SUB(b12, b22));                               \
                            STORE(s + 4 * plane + q, SUB(b21, b11));                               \
                            STORE(s + 5 * plane + q, ADD(a11, a12));                               \
                            STORE(s + 6 * plane + q, SUB(a21, a11));                               \
                            STORE(s + 7 * plane + q, ADD(b11, b12));                               \
                            STORE(s + 8 * plane + q, SUB(a12, a22));                               \
                            STORE(s + 9 * plane + q, ADD(b21, b22));                               \
                        }                                                                          \
                    } else {                                                                       \
                        VTYPE m1 = LOAD(s + (STRASSEN_PRODUCT_BASE + 0) * plane + q),              \
                              m2 = LOAD(s + (STRASSEN_PRODUCT_BASE + 1) * plane + q),              \
                              m3 = LOAD(s + (STRASSEN_PRODUCT_BASE + 2) * plane + q);              \
                        VTYPE m4 = LOAD(s + (STRASSEN_PRODUCT_BASE + 3) * plane + q),              \
                              m5 = LOAD(s + (STRASSEN_PRODUCT_BASE + 4) * plane + q);              \
                        VTYPE m6 = LOAD(s + (STRASSEN_PRODUCT_BASE + 5) * plane + q),              \
                              m7 = LOAD(s + (STRASSEN_PRODUCT_BASE + 6) * plane + q);              \
                        if (STRASSEN_WINOGRAD) {                                                   \
                            VTYPE u2 = ADD(m1, m6), u3 = ADD(u2, m7), u4 = ADD(u2, m5);            \
                            STORE(c + p, ADD(m1, m2));                                             \
                            STORE(c + p + off, ADD(u4, m3));                                       \
                            STORE(c + p + h, SUB(u3, m4));                                         \
                            STORE(c + p + off + h, ADD(u3, m5));                                   \
                        } else {                                                                   \
                            STORE(c + p, ADD(SUB(ADD(m1, m4), m5), m7));                           \
                            STORE(c + p + off, ADD(m3, m5));                                       \
                            STORE(c + p + h, ADD(m2, m4));                                         \
                            STORE(c + p + off + h, ADD(ADD(SUB(m1, m2), m3), m6));                 \
                        }                                                                          \
                    }                                                                              \
                }                                                                                  \
            for (; i < h; ++i) {                                                                   \
                size_t q = (size_t)j * h + i, p = (size_t)j * n + i, off = (size_t)h * n;          \
                if (!g->combine) {                                                                 \
                    size_t pa = (size_t)j * g->lda + i, pb = (size_t)j * g->ldb + i,               \
                           oa = (size_t)h * g->lda, ob = (size_t)h * g->ldb;                       \
                    TYPE a11 = a[pa], a21 = a[pa + h], a12 = a[pa + oa], a22 = a[pa + oa + h];     \
                    TYPE b11 = b[pb], b21 = b[pb + h], b12 = b[pb + ob], b22 = b[pb + ob + h];     \
                    if (STRASSEN_WINOGRAD) {                                                       \
                        TYPE s1 = a21 + a22, s2 = s1 - a11, t1 = b12 - b11, t2 = b22 - t1;         \
                        s[q] = s1;                                                                 \
                        s[plane + q] = s2;                                                         \
                        s[2 * plane + q] = a11 - a21;                                              \
                        s[3 * plane + q] = a12 - s2;                                               \
                        s[4 * plane + q] = t1;                                                     \
                        s[5 * plane + q] = t2;                                                     \
                        s[6 * plane + q] = b22 - b12;                                              \
                        s[7 * plane + q] = t2 - b21;                                               \
                    } else {                                                                       \
                        s[q] = a11 + a22;                                                          \
                        s[plane + q] = b11 + b22;                                                  \
                        s[2 * plane + q] = a21 + a22;                                              \
                        s[3 * plane + q] = b12 - b22;                                              \
                        s[4 * plane + q] = b21 - b11;                                              \
                        s[5 * plane + q] = a11 + a12;                                              \
                        s[6 * plane + q] = a21 - a11;                                              \
                        s[7 * plane + q] = b11 + b12;                                              \
                        s[8 * plane + q] = a12 - a22;                                              \
                        s[9 * plane + q] = b21 + b22;                                              \
                    }                                                                              \
                } else {                                                                           \
                    TYPE m1 = s[(STRASSEN_PRODUCT_BASE + 0) * plane + q],                          \
                         m2 = s[(STRASSEN_PRODUCT_BASE + 1) * plane + q],                          \
                         m3 = s[(STRASSEN_PRODUCT_BASE + 2) * plane + q];                          \
                    TYPE m4 = s[(STRASSEN_PRODUCT_BASE + 3) * plane + q],                          \
                         m5 = s[(STRASSEN_PRODUCT_BASE + 4) * plane + q];                          \
                    TYPE m6 = s[(STRASSEN_PRODUCT_BASE + 5) * plane + q],                          \
                         m7 = s[(STRASSEN_PRODUCT_BASE + 6) * plane + q];                          \
                    if (STRASSEN_WINOGRAD) {                                                       \
                        TYPE u2 = m1 + m6, u3 = u2 + m7, u4 = u2 + m5;                             \
                        c[p] = m1 + m2;                                                            \
                        c[p + off] = u4 + m3;                                                      \
                        c[p + h] = u3 - m4;                                                        \
                        c[p + off + h] = u3 + m5;                                                  \
                    } else {                                                                       \
                        c[p] = ((m1 + m4) - m5) + m7;                                              \
                        c[p + off] = m3 + m5;                                                      \
                        c[p + h] = m2 + m4;                                                        \
                        c[p + off + h] = ((m1 - m2) + m3) + m6;                                    \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
    }
DEFINE_STRASSEN_CALLBACK(f32, float, float32x4_t, 4, vld1q_f32, vst1q_f32, vaddq_f32, vsubq_f32)
DEFINE_STRASSEN_CALLBACK(f64, double, float64x2_t, 2, vld1q_f64, vst1q_f64, vaddq_f64, vsubq_f64)
#undef DEFINE_STRASSEN_CALLBACK

/* Contiguous child operands keep the existing one-level implementation and
 * its leading-dimension contract intact. Copies are inside the timed call. */
#if !STRASSEN_FUSED_SUMS
typedef struct {
    int n, lda, ldb, fp64;
    const void *a, *b;
    void *ac, *bc;
} strassen_copy_t;
static void strassen_copy_task(const camblas_task_t *task, void *opaque)
{
    strassen_copy_t *g = opaque;
    for (int j = task->j0; j < task->j1; ++j)
        for (int i = 0; i < g->n; ++i) {
            if (g->fp64) {
                ((double *)g->ac)[(size_t)j * g->n + i] =
                    ((const double *)g->a)[(size_t)j * g->lda + i];
                ((double *)g->bc)[(size_t)j * g->n + i] =
                    ((const double *)g->b)[(size_t)j * g->ldb + i];
            } else {
                ((float *)g->ac)[(size_t)j * g->n + i] =
                    ((const float *)g->a)[(size_t)j * g->lda + i];
                ((float *)g->bc)[(size_t)j * g->n + i] =
                    ((const float *)g->b)[(size_t)j * g->ldb + i];
            }
        }
}
#endif

static int strassen_execute(const camblas_ctx_t *ctx, int fp64, int n, const void *a, int a_stride,
                            const void *b, int b_stride, void *c, camblas_plan_t *plan,
                            void *workspace, size_t bytes, strassen_scratch_t *scratch)
{
    if (!ctx || !a || !b || !c || !plan || !workspace || !scratch || !scratch->data ||
        scratch->n != n || scratch->fp64 != fp64 || n < 2 || n > 8192 || n % 2 || a_stride < n ||
        b_stride < n || ctx->num_threads < 1 || ctx->num_threads > 64)
        return -1;
    int h = n / 2, workers = ctx->num_threads;
    size_t element = fp64 ? sizeof(double) : sizeof(float),
           plane = (size_t)h * h + STRASSEN_PLANE_PAD_BYTES / element;
    if ((scratch->levels != 1 && scratch->levels != 2) || (scratch->levels == 2 && n % 4))
        return -1;
    size_t scratch_required = STRASSEN_PLANES * plane * element;
    if (scratch->levels == 2)
        scratch_required += ((STRASSEN_DIRECT_CHILD ? 0 : 2 * (size_t)h * h) +
                             STRASSEN_PLANES * (size_t)(h / 2) * (h / 2)) *
                                element +
                            STRASSEN_PLANES * STRASSEN_PLANE_PAD_BYTES;
    if (scratch->bytes < scratch_required)
        return -1;
    size_t required = 0;
    if (camblas_workspace_bytes(h, h, fp64 ? CAMBLAS_DTYPE_F64 : CAMBLAS_DTYPE_F32, &required) ||
        bytes < required)
        return -1;
    const camblas_executor_t *executor = ctx->executor ? ctx->executor : &camblas_executor_serial;
    if (!executor->run)
        return -1;
    camblas_task_t tasks[64];
    for (int t = 0; t < workers; ++t)
        tasks[t] = (camblas_task_t){0, h, h * t / workers, h * (t + 1) / workers};
    strassen_task_context_t g = {n, 0, a_stride, b_stride, a, b, c, scratch->data};
    camblas_task_fn callback = fp64 ? strassen_task_f64 : strassen_task_f32;
    long long phase_start = strassen_now();
    if (!STRASSEN_FUSED_SUMS && executor->run(callback, tasks, workers, &g, executor->user_data))
        return -1;
    long long products_start = strassen_now();
    const unsigned char *aa = a, *bb = b;
    unsigned char *s = scratch->data;
    camblas_plan_t subplan = {0};
#if !STRASSEN_FUSED_SUMS
    const void *left[7] = {s,
                           s + 2 * plane * element,
                           aa,
                           aa + ((size_t)h * a_stride + h) * element,
                           s + 5 * plane * element,
                           s + 6 * plane * element,
                           s + 8 * plane * element};
    const void *right[7] = {s + plane * element,
                            bb,
                            s + 3 * plane * element,
                            s + 4 * plane * element,
                            bb + ((size_t)h * b_stride + h) * element,
                            s + 7 * plane * element,
                            s + 9 * plane * element};
    int lda[7] = {h, h, a_stride, a_stride, h, h, h}, ldb[7] = {h, b_stride, h, h, b_stride, h, h};
    if (STRASSEN_WINOGRAD) {
        left[0] = aa;
        left[1] = aa + (size_t)h * a_stride * element;
        left[2] = s + 3 * plane * element;
        left[3] = aa + ((size_t)h * a_stride + h) * element;
        left[4] = s;
        left[5] = s + plane * element;
        left[6] = s + 2 * plane * element;
        right[0] = bb;
        right[1] = bb + (size_t)h * element;
        right[2] = bb + ((size_t)h * b_stride + h) * element;
        right[3] = s + 7 * plane * element;
        right[4] = s + 4 * plane * element;
        right[5] = s + 5 * plane * element;
        right[6] = s + 6 * plane * element;
        lda[0] = a_stride;
        lda[1] = a_stride;
        lda[2] = h;
        lda[3] = a_stride;
        ldb[0] = b_stride;
        ldb[1] = b_stride;
        ldb[2] = b_stride;
        ldb[3] = h;
        ldb[4] = h;
    }
#else
    size_t bottom = (size_t)h * element, right_offset = (size_t)h * n * element;
    const void *left[7] = {aa,          aa + bottom,      aa, aa + right_offset + bottom, aa,
                           aa + bottom, aa + right_offset};
    const void *left2[7] = {
        aa + right_offset + bottom, aa + right_offset + bottom, NULL, NULL, aa + right_offset, aa,
        aa + right_offset + bottom};
    const void *right[7] = {
        bb, bb, bb + right_offset, bb + bottom, bb + right_offset + bottom, bb, bb + bottom};
    const void *right2[7] = {
        bb + right_offset + bottom, NULL, bb + right_offset + bottom, bb, NULL, bb + right_offset,
        bb + right_offset + bottom};
    const int sign_a[7] = {1, 1, 0, 0, 1, -1, -1}, sign_b[7] = {1, 0, -1, -1, 0, 1, 1};
    camblas_op_t op = {.m = h,
                       .n = h,
                       .k = h,
                       .trans_a = 'N',
                       .trans_b = 'N',
                       .dtype = fp64 ? CAMBLAS_DTYPE_F64 : CAMBLAS_DTYPE_F32,
                       .batch_count = 1};
    if (camblas_plan_make(&op, ctx->topo, ctx, &subplan))
        return -1;
    if (subplan.kernel_id != CAMBLAS_KERNEL_PACKED) {
        /* Research-only forced packed path for small numerical tests. */
        subplan.kernel_id = CAMBLAS_KERNEL_PACKED;
        subplan.mc = 256;
        subplan.nc = 512;
        subplan.kc = 256;
        subplan.mr = fp64 ? 4 : 8;
        subplan.nr = 8;
        subplan.num_threads = workers;
    }
#endif
    for (int p = 0; p < 7; ++p) {
        void *out = s + (STRASSEN_PRODUCT_BASE + (size_t)p) * plane * element;
#if !STRASSEN_FUSED_SUMS
        int rc;
        if (scratch->levels == 2) {
            unsigned char *ac = s + STRASSEN_PLANES * plane * element;
            unsigned char *bc = ac + (size_t)h * h * element;
            strassen_scratch_t child = {
                .n = h,
                .fp64 = fp64,
                .levels = 1,
                .bytes = STRASSEN_PLANES *
                         ((size_t)(h / 2) * (h / 2) * element + STRASSEN_PLANE_PAD_BYTES),
                .data = STRASSEN_DIRECT_CHILD ? ac : bc + (size_t)h * h * element};
            strassen_copy_t copy = {h, lda[p], ldb[p], fp64, left[p], right[p], ac, bc};
            if (STRASSEN_DIRECT_CHILD)
                rc = strassen_execute(ctx, fp64, h, left[p], lda[p], right[p], ldb[p], out,
                                      &subplan, workspace, bytes, &child);
            else {
                rc = executor->run(strassen_copy_task, tasks, workers, &copy, executor->user_data);
                if (!rc)
                    rc = strassen_execute(ctx, fp64, h, ac, h, bc, h, out, &subplan, workspace,
                                          bytes, &child);
            }
        } else
            rc = fp64 ? camblas_dgemm_plan_workspace(ctx, 'N', 'N', h, h, h, 1, left[p], lda[p],
                                                     right[p], ldb[p], 0, out, h, &subplan,
                                                     workspace, bytes)
                      : camblas_sgemm_plan_workspace(ctx, 'N', 'N', h, h, h, 1, left[p], lda[p],
                                                     right[p], ldb[p], 0, out, h, &subplan,
                                                     workspace, bytes);
#else
        int rc = fp64 ? camblas_experimental_dgemm_sum_workspace(
                            'N', 'N', h, h, h, 1, left[p], n, right[p], n, 0, out, h, subplan.mc,
                            subplan.nc, subplan.kc, subplan.mr, subplan.nr, executor, workspace,
                            bytes, left2[p], n, sign_a[p], right2[p], n, sign_b[p])
                      : camblas_experimental_sgemm_sum_workspace(
                            'N', 'N', h, h, h, 1, left[p], n, right[p], n, 0, out, h, subplan.mc,
                            subplan.nc, subplan.kc, subplan.mr, subplan.nr, executor, workspace,
                            bytes, left2[p], n, sign_a[p], right2[p], n, sign_b[p]);
#endif
        if (rc)
            return rc;
    }
    g.combine = 1;
    long long combine_start = strassen_now();
    if (executor->run(callback, tasks, workers, &g, executor->user_data))
        return -1;
    if (STRASSEN_PROFILE) {
        scratch->phase_ns[0] += products_start - phase_start;
        scratch->phase_ns[1] += combine_start - products_start;
        scratch->phase_ns[2] += strassen_now() - combine_start;
    }
    *plan = subplan; /* Describes a constituent GEMM, not the outer operation. */
    return 0;
}
static int strassen_one_level(const camblas_ctx_t *ctx, int fp64, int n, const void *a,
                              const void *b, void *c, camblas_plan_t *plan, void *workspace,
                              size_t bytes, strassen_scratch_t *scratch)
{
    return strassen_execute(ctx, fp64, n, a, n, b, n, c, plan, workspace, bytes, scratch);
}
#endif
