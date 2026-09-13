#ifndef CAMBLAS_RECTANGULAR32_H
#define CAMBLAS_RECTANGULAR32_H
#include "camblas_executor.h"
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
int camblas_experimental_rectangular32_f32(const camblas_executor_t *executor, int workers, int m,
                                           int n, int k, const float *a, int lda, const float *b,
                                           int ldb, float *c, int ldc, void *scratch, size_t bytes);
int camblas_experimental_rectangular32_f32_op(int tb, const camblas_executor_t *executor,
                                              int workers, int m, int n, int k, const float *a,
                                              int lda, const float *b, int ldb, float *c, int ldc,
                                              void *scratch, size_t bytes);
#endif
