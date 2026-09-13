#ifndef CAMBLAS_RECTANGULAR32_POLICY_H
#define CAMBLAS_RECTANGULAR32_POLICY_H
#include "rectangular32.h"
#include "rectangular32_range.h"
#include "gemm_bounds.h"

/* The bridge GEMM mutex owns this scratch. No operand values survive a call. */
static void *rectangular32_scratch;
static size_t rectangular32_capacity;

static int try_rect32_s(camblas_ctx_t *ctx, char ta, char tb, int m, int n, int k, float alpha,
                        const float *a, int lda, const float *b, int ldb, float beta, float *c,
                        int ldc, camblas_plan_t *plan)
{

    int square = tb == 'N' && m == n && n == k && n >= 512 && n <= 2048;
    int transposed = tb == 'T' && m >= n && m <= 2048 && n >= 512 && k >= 256 && k <= 1024;
    if (ta != 'N' || alpha != 1 || beta != 0 || ctx->num_threads < 16 || ctx->num_threads > 32 ||
        (!square && !transposed) || m % 2 || n % 2 || k % 2)
        return 0;
    if (!camblas_experimental_rectangular32_available())
        return 0;
    size_t needed;
    if (camblas_matrix_span_fits(m, k, lda, sizeof(float)) ||
        camblas_matrix_span_fits(tb == 'T' ? n : k, tb == 'T' ? k : n, ldb, sizeof(float)) ||
        camblas_matrix_span_fits(m, n, ldc, sizeof(float)) ||
        camblas_experimental_rectangular32_bytes(m, n, k, &needed) || needed > 512u * 1024u * 1024u)
        return 0;
    int safe = rectangular32_range_safe_op(tb == 'T', ctx->executor, ctx->num_threads, m, n, k, a,
                                           lda, b, ldb, 1);
    if (safe < 0)
        fail("FP32 rectangular range preflight");
    if (!safe)
        return 0;
    if (needed > rectangular32_capacity) {
        void *next = benchmark_alloc(needed);
        if (!next)
            return 0;
        free(rectangular32_scratch);
        rectangular32_scratch = next;
        rectangular32_capacity = needed;
    }
    if (camblas_experimental_rectangular32_f32_op(tb == 'T', ctx->executor, ctx->num_threads, m, n,
                                                  k, a, lda, b, ldb, c, ldc, rectangular32_scratch,
                                                  rectangular32_capacity))
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
