#ifndef CAMBLAS_RECTANGULAR64_POLICY_H
#define CAMBLAS_RECTANGULAR64_POLICY_H
#include "rectangular64.h"
#include "rectangular64_range.h"
#include "gemm_bounds.h"

/* The bridge GEMM mutex owns this scratch. No operand values survive a call. */
static void *rectangular64_scratch;
static size_t rectangular64_capacity;

static int try_rect64_d(camblas_ctx_t *ctx, char ta, char tb, int m, int n, int k, double alpha,
                        const double *a, int lda, const double *b, int ldb, double beta, double *c,
                        int ldc, camblas_plan_t *plan)
{

    int square = tb == 'N' && m == n && n == k && n >= 512 && n <= 2048;
    int transposed = tb == 'T' && m >= n && m <= 2048 && n >= 512 && k >= 256 && k <= 1024;
    if (ta != 'N' || alpha != 1 || beta != 0 || ctx->num_threads < 16 || ctx->num_threads > 64 ||
        (!square && !transposed) || m % 2 || n % 2 || k % 2)
        return 0;
    if (!camblas_experimental_rectangular64_available())
        return 0;
    size_t needed;
    if (camblas_matrix_span_fits(m, k, lda, sizeof(double)) ||
        camblas_matrix_span_fits(tb == 'T' ? n : k, tb == 'T' ? k : n, ldb, sizeof(double)) ||
        camblas_matrix_span_fits(m, n, ldc, sizeof(double)) ||
        camblas_experimental_rectangular64_bytes(m, n, k, &needed) || needed > 512u * 1024u * 1024u)
        return 0;
    int safe = rectangular64_range_safe_op(tb == 'T', ctx->executor, ctx->num_threads, m, n, k, a,
                                           lda, b, ldb, 1);
    if (safe < 0)
        fail("FP64 rectangular range preflight");
    if (!safe)
        return 0;
    if (needed > rectangular64_capacity) {
        void *next = benchmark_alloc(needed);
        if (!next)
            return 0;
        free(rectangular64_scratch);
        rectangular64_scratch = next;
        rectangular64_capacity = needed;
    }
    if (camblas_experimental_rectangular64_f64_op(tb == 'T', ctx->executor, ctx->num_threads, m, n,
                                                  k, a, lda, b, ldb, c, ldc, rectangular64_scratch,
                                                  rectangular64_capacity))
        fail("FP64 rectangular GEMM");

    memset(plan, 0, sizeof(*plan));
    plan->kernel_id = CAMBLAS_KERNEL_PACKED;
    return 1;
}

static void release_rect64(void)
{
    free(rectangular64_scratch);
}
#endif
