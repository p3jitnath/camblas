/* Bridge-private, opt-in batched one-level Strassen experiment. */
#ifndef CAMBLAS_BILINEAR_H
#define CAMBLAS_BILINEAR_H
#include "camblas_executor.h"
#include <stddef.h>
#define CAMBLAS_BILINEAR_PRIVATE __attribute__((visibility("hidden")))
CAMBLAS_BILINEAR_PRIVATE int camblas_bilinear_bytes(int n, int fp64, size_t *bytes);
CAMBLAS_BILINEAR_PRIVATE int camblas_bilinear_f32(const camblas_executor_t *, int, const float *,
                                                  const float *, float *, void *, size_t);
CAMBLAS_BILINEAR_PRIVATE int camblas_bilinear_f64(const camblas_executor_t *, int, const double *,
                                                  const double *, double *, void *, size_t);
#endif
