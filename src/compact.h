/* Experimental bridge-private, fully prepacked compact-panel GEMM. */
#ifndef CAMBLAS_COMPACT_H
#define CAMBLAS_COMPACT_H
#include "camblas_executor.h"
#include <stddef.h>
#define CAMBLAS_COMPACT_PRIVATE __attribute__((visibility("hidden")))
CAMBLAS_COMPACT_PRIVATE int camblas_compact_bytes(int, int, int, int, size_t *);
CAMBLAS_COMPACT_PRIVATE int camblas_compact_f32(const camblas_executor_t *, int, char, char, int,
                                                int, int, float, const float *, int, const float *,
                                                int, float, float *, int, void *, size_t);
CAMBLAS_COMPACT_PRIVATE int camblas_compact_f64(const camblas_executor_t *, int, char, char, int,
                                                int, int, double, const double *, int,
                                                const double *, int, double, double *, int, void *,
                                                size_t);
#endif
