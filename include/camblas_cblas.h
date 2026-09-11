/*
 * CAMBLAS CBLAS compatibility surface.
 *
 * This is the intentionally small public header for the two CBLAS entry
 * points currently exported by CAMBLAS.  It is not a declaration of the full
 * CBLAS/LAPACK interface.  The enum values match the standard CBLAS values so
 * callers can pass the usual order and transpose constants.
 */
#ifndef CAMBLAS_CBLAS_H
#define CAMBLAS_CBLAS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { CAMBLAS_CBLAS_ROW_MAJOR = 101, CAMBLAS_CBLAS_COL_MAJOR = 102 } camblas_cblas_order_t;

typedef enum {
    CAMBLAS_CBLAS_NO_TRANS = 111,
    CAMBLAS_CBLAS_TRANS = 112,
    CAMBLAS_CBLAS_CONJ_TRANS = 113
} camblas_cblas_transpose_t;

/*
 * CBLAS-compatible real GEMM entry points.  Dimensions and leading dimensions
 * use LP64 C int, matching the native CAMBLAS API.  The functions are
 * void-valued for CBLAS compatibility.  Invalid arguments are rejected before
 * dispatch and reported through cblas_xerbla(); the default hook records the
 * 1-based offending parameter in thread-local state.  Real arithmetic treats
 * CONJ_TRANS as TRANS.
 */
void cblas_sgemm(camblas_cblas_order_t order, camblas_cblas_transpose_t trans_a,
                 camblas_cblas_transpose_t trans_b, int m, int n, int k, float alpha,
                 const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc);

void cblas_dgemm(camblas_cblas_order_t order, camblas_cblas_transpose_t trans_a,
                 camblas_cblas_transpose_t trans_b, int m, int n, int k, double alpha,
                 const double *A, int lda, const double *B, int ldb, double beta, double *C,
                 int ldc);

/*
 * CBLAS error reporting.
 *
 * cblas_* entry points clear the calling thread's previous error first.  On
 * invalid input they call cblas_xerbla with the 1-based CBLAS parameter index
 * (1=order, 2=trans_a, ..., 14=ldc) and do not modify a non-NULL C buffer.
 * The built-in hook is non-aborting and records that index; applications may
 * observe it with camblas_cblas_last_error().  A valid call leaves the query
 * at CAMBLAS_CBLAS_NO_ERROR.  An unrepresentable accessed matrix byte span is
 * reported as CAMBLAS_CBLAS_INTERNAL_ERROR before native dispatch and leaves
 * C unchanged.  Any other native execution failure after validation is
 * reported as CAMBLAS_CBLAS_INTERNAL_ERROR after native dispatch; the native
 * failure contract determines whether C is unchanged or partial.  The hook
 * is synchronous and must not retain its argument pointers.
 *
 * This is the current ABI-0 development contract, not a stable ABI-1 promise.
 */
#define CAMBLAS_CBLAS_NO_ERROR 0
#define CAMBLAS_CBLAS_INTERNAL_ERROR (-1)

int camblas_cblas_last_error(void);
void camblas_cblas_clear_error(void);
void cblas_xerbla(int p, const char *rout, const char *form, ...);

#ifdef __cplusplus
}
#endif

#endif /* CAMBLAS_CBLAS_H */
