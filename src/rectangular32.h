#ifndef CAMBLAS_RECTANGULAR32_H
#define CAMBLAS_RECTANGULAR32_H
#include "camblas_executor.h"
#include <float.h>
#include <stddef.h>

/* Internal FP32 one-level Strassen. The caller owns disjoint scratch;
 * inputs are repacked every call. Alpha=1 and beta=0 are required. A is
 * untransposed; the _op entry accepts ordinary B (tb=0) or transposed B
 * (tb=1), with column-major physical storage and explicit leading dimensions.
 * The bridge selects bounded geometries and handles other BLAS cases.
 * The range preflight belongs to the bridge, before this routine is called.
 * Any executor failure is fatal to this attempt: C must not be retried. */
/* Query the current SVE runtime before any operand or workspace access. */
int camblas_experimental_rectangular32_available(void);
int camblas_experimental_rectangular32_bytes(int m, int n, int k, size_t *bytes);

/* Shared conservative verdict for the packed route: one for a safe finite
 * range, zero when the classical path should own the call. A sum contains at
 * most two or four source values. Conservative bounds for intermediate
 * products and recombination are 8*K and 64*K times the input maxima
 * product. A factor of two leaves rounding headroom. */
static inline int rectangular32_range_verdict(float a_peak, float b_peak, int k, int levels)
{
    float sum_bound = levels == 1 ? 4.0 : 8.0;
    float product_bound = levels == 1 ? 16.0 : 128.0;
    return a_peak <= FLT_MAX / sum_bound && b_peak <= FLT_MAX / sum_bound &&
           (a_peak == 0 || b_peak <= ((FLT_MAX / product_bound) / k) / a_peak);
}
int camblas_experimental_rectangular32_f32(const camblas_executor_t *executor, int workers, int m,
                                           int n, int k, const float *a, int lda, const float *b,
                                           int ldb, float *c, int ldc, void *scratch, size_t bytes);
int camblas_experimental_rectangular32_f32_op(int tb, const camblas_executor_t *executor,
                                              int workers, int m, int n, int k, const float *a,
                                              int lda, const float *b, int ldb, float *c, int ldc,
                                              void *scratch, size_t bytes);
/* Same route with the conservative range preflight fused into the pack pass:
 * returns 0 on success, 1 when the classical path should own the call, and
 * -1 on failure. levels mirrors the standalone preflight bound selection. */
int camblas_experimental_rectangular32_f32_op_checked(int tb, const camblas_executor_t *executor,
                                                      int workers, int m, int n, int k,
                                                      const float *a, int lda, const float *b,
                                                      int ldb, float *c, int ldc, void *scratch,
                                                      size_t bytes, int levels);
#endif
