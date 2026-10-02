#ifndef CAMBLAS_RECTANGULAR32_POLICY_H
#define CAMBLAS_RECTANGULAR32_POLICY_H
#include "rectangular32.h"
#include "rectangular32_range.h"
#include "gemm_bounds.h"

/* The bridge GEMM mutex owns this scratch. No operand values survive a call. */
static void *rectangular32_scratch;
static size_t rectangular32_capacity;

typedef struct {
    const float *source;
    float *out;
    int m, k, ld, workers;
} short32_copy_t;
static void short32_copy(const camblas_task_t *task, void *opaque)
{
    short32_copy_t *w = opaque;
    int blocks = w->k / 4, begin = blocks * task->i0 / w->workers,
        end = blocks * (task->i0 + 1) / w->workers;
    for (int block = begin; block < end; ++block) {
        int p = block * 4;
        for (int i = 0; i < w->m; i += 4) {
            const float *in = w->source + p + (size_t)i * w->ld;
            float32x4_t a = vld1q_f32(in), b = vld1q_f32(in + w->ld);
            float32x4_t c = vld1q_f32(in + 2 * w->ld), d = vld1q_f32(in + 3 * w->ld);
            float32x4_t x = vtrn1q_f32(a, b), y = vtrn2q_f32(a, b), z = vtrn1q_f32(c, d),
                        t = vtrn2q_f32(c, d);
            vst1q_f32(w->out + i + (size_t)p * w->m,
                      vcombine_f32(vget_low_f32(x), vget_low_f32(z)));
            vst1q_f32(w->out + i + (size_t)(p + 1) * w->m,
                      vcombine_f32(vget_low_f32(y), vget_low_f32(t)));
            vst1q_f32(w->out + i + (size_t)(p + 2) * w->m,
                      vcombine_f32(vget_high_f32(x), vget_high_f32(z)));
            vst1q_f32(w->out + i + (size_t)(p + 3) * w->m,
                      vcombine_f32(vget_high_f32(y), vget_high_f32(t)));
        }
    }
}

static int try_rect32_s(camblas_ctx_t *ctx, char ta, char tb, int m, int n, int k, float alpha,
                        const float *a, int lda, const float *b, int ldb, float beta, float *c,
                        int ldc, camblas_plan_t *plan)
{

    int square = tb == 'N' && m == n && n == k && n >= 512 && n <= 2048;
    int transposed = tb == 'T' && m >= n && m <= 2048 && n >= 512 && k >= 256 && k <= 1024;
    int deep = ctx->num_threads <= 32 && tb == 'N' && m >= (int64_t)n * 4 && m <= 8192 &&
               n >= 256 && n <= 2048 && k > 1024 && k <= 4096;
    int copy_a = ctx->num_threads <= 32 && ta == 'T' && tb == 'N' && m == n && m >= 512 &&
                 m <= 2048 && k >= 256 && k <= 1024 && !(m % 4) && !(k % 4);
    if ((ta != 'N' && !copy_a) || alpha != 1 || beta != 0 || ctx->num_threads < 16 ||
        ctx->num_threads > 64 || (ctx->num_threads > 32 && !deep) ||
        (!square && !transposed && !deep && !copy_a) || m % 2 || n % 2 || k % 2)
        return 0;
    if (!camblas_experimental_rectangular32_available())
        return 0;
    size_t needed;
    if (camblas_matrix_span_fits(ta == 'T' ? k : m, ta == 'T' ? m : k, lda, sizeof(float)) ||
        camblas_matrix_span_fits(tb == 'T' ? n : k, tb == 'T' ? k : n, ldb, sizeof(float)) ||
        camblas_matrix_span_fits(m, n, ldc, sizeof(float)) ||
        camblas_experimental_rectangular32_bytes(m, n, k, &needed) || needed > 512u * 1024u * 1024u)
        return 0;
    int status;
    if (copy_a) {
        size_t a_bytes = (size_t)m * k * sizeof(float);
        void *allocation = malloc(needed + a_bytes + 63);
        if (!allocation)
            return 0;
        unsigned char *frame = (void *)(((uintptr_t)allocation + 63u) & ~(uintptr_t)63u);
        short32_copy_t copy = {.source = a,
                               .out = (float *)frame,
                               .m = m,
                               .k = k,
                               .ld = lda,
                               .workers = ctx->num_threads};
        camblas_task_t tasks[64];
        for (int t = 0; t < ctx->num_threads; ++t)
            tasks[t] = (camblas_task_t){t, t + 1, 0, 1};
        if (ctx->executor->run(short32_copy, tasks, ctx->num_threads, &copy,
                               ctx->executor->user_data))
            fail("short FP32 transpose");
        status = camblas_experimental_rectangular32_f32_op_checked(
            0, ctx->executor, ctx->num_threads, m, n, k, (float *)frame, m, b, ldb, c, ldc,
            frame + a_bytes, needed, 1);
        free(allocation);
    } else if (deep) {
        size_t sizes[3];
        if (camblas_experimental_rectangular32_segment_bytes(ctx->num_threads, m, n, k, sizes))
            return 0;
        size_t total = sizes[0] + sizes[1] + sizes[2];
        /* A single owned allocation can be recycled by libc between calls;
     * align its payload for vector loads without page-aligning the request. */
        void *allocation = malloc(total + 63);
        unsigned char *frame =
            allocation ? (unsigned char *)(((uintptr_t)allocation + 63u) & ~(uintptr_t)63u) : NULL;
        if (!frame)
            return 0;
        camblas_rectangular32_segments_t segments;
        size_t offset = 0;
        for (int t = 0; t < 3; ++t) {
            segments.data[t] = frame + offset;
            segments.bytes[t] = sizes[t];
            offset += sizes[t];
        }
        status = camblas_experimental_rectangular32_f32_segments_checked(
            tb == 'T', ctx->executor, ctx->num_threads, m, n, k, a, lda, b, ldb, c, ldc, &segments,
            1);
        free(allocation);
    } else {
        if (needed > rectangular32_capacity) {
            void *next = benchmark_alloc(needed);
            if (!next)
                return 0;
            free(rectangular32_scratch);
            rectangular32_scratch = next;
            rectangular32_capacity = needed;
        }
        status = camblas_experimental_rectangular32_f32_op_checked(
            tb == 'T', ctx->executor, ctx->num_threads, m, n, k, a, lda, b, ldb, c, ldc,
            rectangular32_scratch, rectangular32_capacity, 1);
    }
    if (status == 1)
        return 0;
    if (status)
        fail("FP32 rectangular GEMM");

    memset(plan, 0, sizeof(*plan));
    plan->kernel_id = CAMBLAS_KERNEL_PACKED;
    return 1;
}

static void release_rect32(void)
{
    free(rectangular32_scratch);
}
#endif
