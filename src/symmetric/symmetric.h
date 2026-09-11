/* Experimental, bridge-private symmetric product API; not a public BLAS ABI. */
#ifndef CAMBLAS_SYMMETRIC_H
#define CAMBLAS_SYMMETRIC_H
#include "camblas_executor.h"
#include <stddef.h>
#define CAMBLAS_SYM_PRIVATE __attribute__((visibility("hidden")))
CAMBLAS_SYM_PRIVATE int camblas_symmetric_bytes(int n, int k, int fp64, size_t *bytes);
/* Column-major op(A), upper=1/0 selects one triangle; full=1 writes both.
 * Scratch is private to the call, and its contents are rebuilt each time.
 * Caller validates BLAS arguments and serializes reusable scratch ownership. */
CAMBLAS_SYM_PRIVATE int camblas_symmetric_f32(const camblas_executor_t *, int, char, int, int,
                                              float, const float *, int, float, float *, int, int,
                                              int, void *, size_t);
CAMBLAS_SYM_PRIVATE int camblas_symmetric_f64(const camblas_executor_t *, int, char, int, int,
                                              double, const double *, int, double, double *, int,
                                              int, int, void *, size_t);
#endif
