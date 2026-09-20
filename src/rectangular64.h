#ifndef CAMBLAS_RECTANGULAR64_H
#define CAMBLAS_RECTANGULAR64_H
#include "camblas_executor.h"
#include <float.h>
#include <stddef.h>

/* Internal FP64 one-level Strassen. The caller owns disjoint scratch;
 * inputs are repacked every call. Alpha=1 and beta=0 are required. A is
 * untransposed; the _op entry accepts ordinary B (tb=0) or transposed B
 * (tb=1), with column-major physical storage and explicit leading dimensions.
 * The bridge selects bounded geometries and handles other BLAS cases.
 * The range preflight belongs to the bridge, before this routine is called.
 * Any executor failure is fatal to this attempt: C must not be retried. */
/* Query the current SVE runtime before any operand or workspace access. */
int camblas_experimental_rectangular64_available(void);
int camblas_experimental_rectangular64_bytes(int m, int n, int k, size_t *bytes);

/* Shared conservative verdict for the packed route: one for a safe finite
 * range, zero when the classical path should own the call. A sum contains at
 * most two or four source values. Conservative bounds for intermediate
 * products and recombination are 8*K and 64*K times the input maxima
 * product. A factor of two leaves rounding headroom. */
static inline int rectangular64_range_verdict(double a_peak, double b_peak, int k, int levels)
{
    double sum_bound = levels == 1 ? 4.0 : 8.0;
    double product_bound = levels == 1 ? 16.0 : 128.0;
    return a_peak <= DBL_MAX / sum_bound && b_peak <= DBL_MAX / sum_bound &&
           (a_peak == 0 || b_peak <= ((DBL_MAX / product_bound) / k) / a_peak);
}

int camblas_experimental_rectangular64_f64(const camblas_executor_t *executor, int workers, int m,
                                           int n, int k, const double *a, int lda, const double *b,
                                           int ldb, double *c, int ldc, void *scratch,
                                           size_t bytes);
int camblas_experimental_rectangular64_f64_op(int tb, const camblas_executor_t *executor,
                                              int workers, int m, int n, int k, const double *a,
                                              int lda, const double *b, int ldb, double *c, int ldc,
                                              void *scratch, size_t bytes);
/* Same route with the conservative range preflight fused into the pack pass:
 * returns 0 on success, 1 when the classical path should own the call, and
 * -1 on failure. levels mirrors the standalone preflight bound selection. */
int camblas_experimental_rectangular64_f64_op_checked(int tb, const camblas_executor_t *executor,
                                                      int workers, int m, int n, int k,
                                                      const double *a, int lda, const double *b,
                                                      int ldb, double *c, int ldc, void *scratch,
                                                      size_t bytes, int levels);
#endif
