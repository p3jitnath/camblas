/*
 * CAMBLAS internal kernel boundary.
 *
 * This header is deliberately outside include/ and is not part of the public
 * ABI.  The SVE implementation is compiled with an SVE target attribute in a
 * separate translation unit; the rest of the library remains runnable on a
 * non-SVE AArch64 CPU and on other architectures.
 */
#ifndef CAMBLAS_INTERNAL_KERNELS_H
#define CAMBLAS_INTERNAL_KERNELS_H

#include <stdint.h>

/*
 * Candidate parameterization for the first SVE microtile.  The accepted
 * baseline is nr=2; nr=1 remains a deliberately narrower candidate and nr=4
 * is the bounded Phase-4 wider-tile candidate.  Each candidate must still be
 * correctness-built and measured under the frozen compute-node protocol.  The
 * candidate set is not an optimality claim.
 */
#ifndef CAMBLAS_SVE_NR
#define CAMBLAS_SVE_NR 2
#endif

#if CAMBLAS_SVE_NR != 1 && CAMBLAS_SVE_NR != 2 && CAMBLAS_SVE_NR != 4
#error "CAMBLAS_SVE_NR must be 1, 2, or 4"
#endif

#if defined(__GNUC__) || defined(__clang__)
#define CAMBLAS_INTERNAL __attribute__((visibility("hidden")))
#define CAMBLAS_WEAK __attribute__((weak))
#else
#define CAMBLAS_INTERNAL
#define CAMBLAS_WEAK
#endif

/* Hidden no-pack SVE status contract.  UNAVAILABLE guarantees that the
 * backend returned before touching C; every other nonzero status is an
 * execution failure and may leave C partial.  The scalar caller may retry
 * only the former. */
#define CAMBLAS_SVE_UNAVAILABLE (-1)
#define CAMBLAS_SVE_EXECUTION_ERROR (-2)

typedef struct {
    int compiled_sve; /* this translation unit contains SVE instructions */
    int hw_sve;       /* the running CPU/kernel advertises SVE */
    int hw_sve2;      /* the running CPU/kernel advertises SVE2 */
    int vl_bits;      /* current process SVE VL, or -1 if unavailable */
} camblas_kernel_runtime_t;

CAMBLAS_INTERNAL void camblas_kernel_runtime_query(camblas_kernel_runtime_t *info);

CAMBLAS_INTERNAL const char *camblas_kernel_runtime_name(const camblas_kernel_runtime_t *info);

/* Return 0 when the SVE implementation performed the complete call.
 * CAMBLAS_SVE_UNAVAILABLE means that the caller may use the scalar fallback;
 * any other nonzero return is an execution failure and must not be retried
 * over a possibly partial C.  Only trans_a='N' is currently vectorized; the
 * implementation uses the build-selected CAMBLAS_SVE_NR output microtile
 * and an odd-N tail. trans_a='T' is intentionally rejected before any output
 * write.
 */
CAMBLAS_INTERNAL int camblas_sgemm_sve(char trans_a, char trans_b, int m, int n, int k, float alpha,
                                       const float *A, int lda, const float *B, int ldb, float beta,
                                       float *C, int ldc);

CAMBLAS_INTERNAL int camblas_dgemm_sve(char trans_a, char trans_b, int m, int n, int k,
                                       double alpha, const double *A, int lda, const double *B,
                                       int ldb, double beta, double *C, int ldc);

#endif /* CAMBLAS_INTERNAL_KERNELS_H */
