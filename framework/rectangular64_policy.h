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
    int deep = ctx->num_threads <= 32 && tb == 'N' && ta == 'N' && m > n && m <= 8192 && n >= 256 &&
               n <= 2048 && k > 1024 && k <= 4096;
    int wide = ctx->num_threads <= 32 && tb == 'N' && ta == 'N' && m < n && m >= 128 && m <= 2048 &&
               n <= 2048 && k >= 256 && k <= 1024;
    int copy_a = ctx->num_threads <= 32 && ta == 'T' && tb == 'N' && m == n && m >= 512 &&
                 m <= 2048 && k >= 256 && k <= 1024;
    if ((ta != 'N' && !copy_a) || alpha != 1 || beta != 0 || ctx->num_threads < 16 ||
        ctx->num_threads > 64 || (!square && !transposed && !wide && !deep && !copy_a) || m % 2 ||
        n % 2 || k % 2)
        return 0;
    if (!camblas_experimental_rectangular64_available())
        return 0;
    size_t needed;
    if (camblas_matrix_span_fits(ta == 'T' ? k : m, ta == 'T' ? m : k, lda, sizeof(double)) ||
        camblas_matrix_span_fits(tb == 'T' ? n : k, tb == 'T' ? k : n, ldb, sizeof(double)) ||
        camblas_matrix_span_fits(m, n, ldc, sizeof(double)) ||
        camblas_experimental_rectangular64_bytes(m, n, k, &needed) || needed > 512u * 1024u * 1024u)
        return 0;
    if (needed > rectangular64_capacity) {
        void *next = benchmark_alloc(needed);
        if (!next)
            return 0;
        free(rectangular64_scratch);
        rectangular64_scratch = next;
        rectangular64_capacity = needed;
    }
    /* The range preflight rides on the pack pass inside the op, so no
     * separate scan of A and B runs on the hot path. A verdict of one means
     * the conservative bounds prefer the classical route. */
    int status = camblas_experimental_rectangular64_f64_ops_checked(
        ta == 'T', tb == 'T', ctx->executor, ctx->num_threads, m, n, k, a, lda, b, ldb, c, ldc,
        rectangular64_scratch, rectangular64_capacity, 1);
    if (status == 1)
        return 0;
    if (status)
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
