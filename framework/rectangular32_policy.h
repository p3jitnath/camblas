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
    if (needed > rectangular32_capacity) {
        void *next = benchmark_alloc(needed);
        if (!next)
            return 0;
        free(rectangular32_scratch);
        rectangular32_scratch = next;
        rectangular32_capacity = needed;
    }
    /* The range preflight rides on the pack pass inside the op, so no
     * separate scan of A and B runs on the hot path. A verdict of one means
     * the conservative bounds prefer the classical route. */
    int status = camblas_experimental_rectangular32_f32_op_checked(
        tb == 'T', ctx->executor, ctx->num_threads, m, n, k, a, lda, b, ldb, c, ldc,
        rectangular32_scratch, rectangular32_capacity, 1);
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
