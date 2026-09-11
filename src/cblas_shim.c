/*
 * CAMBLAS CBLAS compatibility shim.
 *
 * Provides standard CBLAS GEMM functions (cblas_sgemm, cblas_dgemm) that
 * delegate to the CAMBLAS native API. This allows CAMBLAS to be used through
 * the same cblas_* interface as OpenBLAS, BLIS, and Cray LibSci in the
 * differential test harness and benchmark driver.
 *
 * The shim uses the default context (1 thread, fast, planner "none") for
 * CBLAS-compatible calls. Callers who need per-call execution control should
 * use the native camblas_*gemm API with an explicit camblas_ctx_t.
 *
 * CBLAS enum values are the standard ones (ABI-compatible with OpenBLAS,
 * BLIS, and Cray LibSci):
 *   CblasRowMajor=101, CblasColMajor=102
 *   CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113
 *
 * Native scope: column-major, real N and T only. Row-major is supported via
 * the standard CBLAS trick (swap A/B and transpose flags) so the shim is
 * a correct drop-in, but performance optimization is column-major only.
 */
#include "camblas.h"
#include "camblas_cblas.h"
#include "gemm_bounds.h"

/*
 * CBLAS is void-valued, so keep the current call's error observable without
 * introducing process-global mutable state.  The default xerbla hook records
 * only the parameter number; the routine/form pointers are borrowed for the
 * duration of the synchronous callback and are not retained.
 */
static _Thread_local int cblas_error_parameter;

int camblas_cblas_last_error(void)
{
    return cblas_error_parameter;
}

void camblas_cblas_clear_error(void)
{
    cblas_error_parameter = CAMBLAS_CBLAS_NO_ERROR;
}

void cblas_xerbla(int p, const char *rout, const char *form, ...)
{
    (void)rout;
    (void)form;
    cblas_error_parameter = p;
}

static void report_cblas_error(int parameter, const char *routine)
{
    /* Set state before the hook so an application-provided hook can query it. */
    cblas_error_parameter = parameter;
    cblas_xerbla(parameter, routine, "invalid argument");
}

static int validate_cblas_args(camblas_cblas_order_t order, camblas_cblas_transpose_t trans_a,
                               camblas_cblas_transpose_t trans_b, int m, int n, int k,
                               const void *A, int lda, const void *B, int ldb, const void *C,
                               int ldc)
{
    int min_lda, min_ldb, min_ldc;

    if (order != CAMBLAS_CBLAS_ROW_MAJOR && order != CAMBLAS_CBLAS_COL_MAJOR)
        return 1;
    if (trans_a != CAMBLAS_CBLAS_NO_TRANS && trans_a != CAMBLAS_CBLAS_TRANS &&
        trans_a != CAMBLAS_CBLAS_CONJ_TRANS)
        return 2;
    if (trans_b != CAMBLAS_CBLAS_NO_TRANS && trans_b != CAMBLAS_CBLAS_TRANS &&
        trans_b != CAMBLAS_CBLAS_CONJ_TRANS)
        return 3;
    if (m < 0)
        return 4;
    if (n < 0)
        return 5;
    if (k < 0)
        return 6;

    /* Preserve the native ABI-0 zero-output rule: positive LDs and all
       pointers are still required, but no larger shape minimum is needed. */
    if (m == 0 || n == 0) {
        min_lda = min_ldb = min_ldc = 1;
    } else if (order == CAMBLAS_CBLAS_COL_MAJOR) {
        min_lda = (trans_a == CAMBLAS_CBLAS_NO_TRANS) ? m : k;
        min_ldb = (trans_b == CAMBLAS_CBLAS_NO_TRANS) ? k : n;
        min_ldc = m;
    } else {
        /* Row-major storage has the number of columns as its leading
           dimension.  These are the physical layouts before the swap into
           the native column-major call below. */
        min_lda = (trans_a == CAMBLAS_CBLAS_NO_TRANS) ? k : m;
        min_ldb = (trans_b == CAMBLAS_CBLAS_NO_TRANS) ? n : k;
        min_ldc = n;
    }
    if (min_lda < 1)
        min_lda = 1;
    if (min_ldb < 1)
        min_ldb = 1;
    if (min_ldc < 1)
        min_ldc = 1;
    if (lda < min_lda)
        return 9;
    if (ldb < min_ldb)
        return 11;
    if (ldc < min_ldc)
        return 14;
    if (A == NULL)
        return 8;
    if (B == NULL)
        return 10;
    if (C == NULL)
        return 13;
    return 0;
}

static char native_trans(camblas_cblas_transpose_t trans)
{
    return (trans == CAMBLAS_CBLAS_NO_TRANS) ? 'N' : 'T';
}

/*
 * Apply the native column-major view's access-span check before dispatch.
 * CBLAS has already validated the conventional LP64 argument contract; an
 * unrepresentable byte span is therefore an internal fail-closed condition,
 * not a mapped CBLAS parameter error. Keeping this check at the public shim
 * boundary also guarantees that a void-valued call cannot reach a native or
 * interposed implementation with an unsafe address calculation.
 */
static int validate_cblas_native_span(camblas_cblas_order_t order,
                                      camblas_cblas_transpose_t trans_a,
                                      camblas_cblas_transpose_t trans_b, int m, int n, int k,
                                      int lda, int ldb, int ldc, int read_inputs, int write_output,
                                      size_t element_size)
{
    char native_trans_a = native_trans(trans_a);
    char native_trans_b = native_trans(trans_b);
    int native_m = m;
    int native_n = n;
    int native_lda = lda;
    int native_ldb = ldb;

    if (order == CAMBLAS_CBLAS_ROW_MAJOR) {
        native_trans_a = native_trans(trans_b);
        native_trans_b = native_trans(trans_a);
        native_m = n;
        native_n = m;
        native_lda = ldb;
        native_ldb = lda;
    }
    return camblas_gemm_access_spans_fit(native_trans_a, native_trans_b, native_m, native_n, k,
                                         native_lda, native_ldb, ldc, read_inputs, write_output,
                                         element_size);
}

/*
 * cblas_sgemm: single-precision GEMM via CAMBLAS.
 *
 * Computes C = alpha * op(A) * op(B) + beta * C in the specified order.
 * For row-major, uses the identity: (AB)^T = B^T A^T, swapping operands.
 */
void cblas_sgemm(camblas_cblas_order_t order, camblas_cblas_transpose_t trans_a,
                 camblas_cblas_transpose_t trans_b, int m, int n, int k, float alpha,
                 const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
    const camblas_ctx_t *ctx = &camblas_ctx_default;
    int error_parameter;
    int rc;

    camblas_cblas_clear_error();
    error_parameter = validate_cblas_args(order, trans_a, trans_b, m, n, k, A, lda, B, ldb, C, ldc);
    if (error_parameter != 0) {
        report_cblas_error(error_parameter, "cblas_sgemm");
        return;
    }
    if (validate_cblas_native_span(
            order, trans_a, trans_b, m, n, k, lda, ldb, ldc, alpha != 0.0f && k > 0,
            camblas_gemm_output_access_needed(k, alpha != 0.0f, beta != 1.0f),
            sizeof(float)) != 0) {
        report_cblas_error(CAMBLAS_CBLAS_INTERNAL_ERROR, "cblas_sgemm");
        return;
    }

    if (order == CAMBLAS_CBLAS_COL_MAJOR) {
        char ta = native_trans(trans_a);
        char tb = native_trans(trans_b);
        rc = camblas_sgemm(ctx, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        /* Row-major: C^T_row = B^T * A^T in column-major. Swap A<->B, trans. */
        char ta = native_trans(trans_b);
        char tb = native_trans(trans_a);
        /* For row-major A(m×k): if noTrans, A is (k×m) col-major, lda = leading
           dim in row-major = stride between rows. In col-major view, the
           leading dimension is the row-major stride, which is k for noTrans
           or m for trans. The swap is:
             C_col = alpha * op(B)^T * op(A)^T + beta * C_col
           where C_col is C viewed as column-major (same memory, transposed). */
        rc = camblas_sgemm(ctx, ta, tb, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc);
    }
    /* The native API collapses packed/executor execution errors to -1 after
       dispatch.  CBLAS has no return slot, so expose the same internal error
       code through xerbla; the native failure contract still governs C. */
    if (rc != 0)
        report_cblas_error(CAMBLAS_CBLAS_INTERNAL_ERROR, "cblas_sgemm");
}

/*
 * cblas_dgemm: double-precision GEMM via CAMBLAS.
 */
void cblas_dgemm(camblas_cblas_order_t order, camblas_cblas_transpose_t trans_a,
                 camblas_cblas_transpose_t trans_b, int m, int n, int k, double alpha,
                 const double *A, int lda, const double *B, int ldb, double beta, double *C,
                 int ldc)
{
    const camblas_ctx_t *ctx = &camblas_ctx_default;
    int error_parameter;
    int rc;

    camblas_cblas_clear_error();
    error_parameter = validate_cblas_args(order, trans_a, trans_b, m, n, k, A, lda, B, ldb, C, ldc);
    if (error_parameter != 0) {
        report_cblas_error(error_parameter, "cblas_dgemm");
        return;
    }
    if (validate_cblas_native_span(
            order, trans_a, trans_b, m, n, k, lda, ldb, ldc, alpha != 0.0 && k > 0,
            camblas_gemm_output_access_needed(k, alpha != 0.0, beta != 1.0), sizeof(double)) != 0) {
        report_cblas_error(CAMBLAS_CBLAS_INTERNAL_ERROR, "cblas_dgemm");
        return;
    }

    if (order == CAMBLAS_CBLAS_COL_MAJOR) {
        char ta = native_trans(trans_a);
        char tb = native_trans(trans_b);
        rc = camblas_dgemm(ctx, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        char ta = native_trans(trans_b);
        char tb = native_trans(trans_a);
        rc = camblas_dgemm(ctx, ta, tb, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc);
    }
    /* As above, a post-dispatch native failure is observable only through the
       synchronous CBLAS internal-error hook and may leave C partial. */
    if (rc != 0)
        report_cblas_error(CAMBLAS_CBLAS_INTERNAL_ERROR, "cblas_dgemm");
}
