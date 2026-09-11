/* Bridge-private no-pack T/N GEMM experiment; not a public BLAS API. */
#ifndef CAMBLAS_DOT_PRODUCT_H
#define CAMBLAS_DOT_PRODUCT_H
#include "camblas_executor.h"
#define CAMBLAS_DOT_PRIVATE __attribute__((visibility("hidden")))
CAMBLAS_DOT_PRIVATE int camblas_dot_f32(const camblas_executor_t *, int, int, int, int, float,
                                        const float *, int, const float *, int, float, float *,
                                        int);
CAMBLAS_DOT_PRIVATE int camblas_dot_f64(const camblas_executor_t *, int, int, int, int, double,
                                        const double *, int, const double *, int, double, double *,
                                        int);
#endif
