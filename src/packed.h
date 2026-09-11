/*
 * CAMBLAS internal packed GEMM boundary.
 *
 * This header is deliberately outside include/ and is not part of the public
 * ABI. The implementation provides serial and caller-owned-executor,
 * correctness-gated blocked paths that consume the checked panel packer. The
 * opt-in SVE library may provide a hidden packed tile implementation; the
 * ordinary-C tile remains the fail-closed fallback. Neither path is a tuned
 * performance claim.
 */
#ifndef CAMBLAS_INTERNAL_PACKED_H
#define CAMBLAS_INTERNAL_PACKED_H

#include "camblas_executor.h"

#if defined(__GNUC__) || defined(__clang__)
#define CAMBLAS_PACKED_INTERNAL __attribute__((visibility("hidden")))
#define CAMBLAS_PACKED_WEAK __attribute__((weak))
#else
#define CAMBLAS_PACKED_INTERNAL
#define CAMBLAS_PACKED_WEAK
#endif

/* A negative result from validation, span/size preflight, or capability
 * refusal. It guarantees that C was not modified and permits the public
 * wrapper to use the scalar fallback. */
#define CAMBLAS_PACKED_UNAVAILABLE (-1)
/* A negative result after executor/packed execution began. This includes a
 * per-task allocation or packing failure; C is unspecified on this path and
 * the public wrapper must not run a second GEMM. */
#define CAMBLAS_PACKED_EXECUTION_ERROR (-2)

CAMBLAS_PACKED_INTERNAL int camblas_sgemm_packed(char trans_a, char trans_b, int m, int n, int k,
                                                 float alpha, const float *A, int lda,
                                                 const float *B, int ldb, float beta, float *C,
                                                 int ldc, int mc, int nc, int kc, int mr, int nr);

CAMBLAS_PACKED_INTERNAL int camblas_dgemm_packed(char trans_a, char trans_b, int m, int n, int k,
                                                 double alpha, const double *A, int lda,
                                                 const double *B, int ldb, double beta, double *C,
                                                 int ldc, int mc, int nc, int kc, int mr, int nr);

/* The serial path performs its shared workspace allocation before beta
 * scaling. Allocation, validation, or size refusal returns UNAVAILABLE with
 * C untouched and permits scalar fallback. Once scaling or panel execution
 * begins, a pack or tile failure returns EXECUTION_ERROR; C may be partial and
 * the public wrapper must not retry the GEMM. */

/* Optional architecture-specific tile hook. A missing hook or a -1 return
 * guarantees that the caller may use its ordinary-C tile instead. Any other
 * nonzero return is an execution error and may follow a partial C update, so
 * the caller must fail closed without applying the ordinary-C tile again. The
 * caller has already applied beta to C; this hook only accumulates alpha*A*B
 * into C. The packed panels are column-major with the supplied leading
 * dimensions. A zero-alpha or zero-K tile is a complete no-op: neither the
 * panels nor C may be accessed, and the ordinary-C fallback must preserve the
 * same boundary. */
CAMBLAS_PACKED_INTERNAL CAMBLAS_PACKED_WEAK int
camblas_sgemm_sve_packed_tile(int m, int n, int k, float alpha, const float *A, int lda,
                              const float *B, int ldb, float *C, int ldc);

CAMBLAS_PACKED_INTERNAL CAMBLAS_PACKED_WEAK int
camblas_dgemm_sve_packed_tile(int m, int n, int k, double alpha, const double *A, int lda,
                              const double *B, int ldb, double *C, int ldc);

/*
 * Caller-owned executor variant. Each task receives a disjoint C rectangle
 * and owns its temporary packed panels, so synchronous custom executors may
 * run tasks concurrently without sharing mutable packing workspace. Panel
 * size/byte rejection happens before executor->run and returns
 * CAMBLAS_PACKED_UNAVAILABLE with C untouched. Once executor->run has been
 * entered, a task allocation/packing failure returns
 * CAMBLAS_PACKED_EXECUTION_ERROR; another task may already have modified its
 * disjoint C rectangle, so the public wrapper must not retry with scalar GEMM.
 */
/* B uses the private micro8 layout; other contracts match packed_tile. */
CAMBLAS_PACKED_INTERNAL CAMBLAS_PACKED_WEAK int
camblas_sgemm_sve_interleaved_tile(int m, int n, int k, float alpha, const float *A, int lda,
                                   const float *B, int ldb, float *C, int ldc);

/* Same panels as interleaved_tile; overwrite C without reading old C.
 * Only used for a positive-K, nonzero-alpha first panel with beta zero. */
CAMBLAS_PACKED_INTERNAL CAMBLAS_PACKED_WEAK int
camblas_sgemm_sve_interleaved_init_tile(int m, int n, int k, float alpha, const float *A, int lda,
                                        const float *B, int ldb, float *C, int ldc);
CAMBLAS_PACKED_INTERNAL CAMBLAS_PACKED_WEAK int
camblas_dgemm_sve_interleaved_init_tile(int m, int n, int k, double alpha, const double *A, int lda,
                                        const double *B, int ldb, double *C, int ldc);

/**
 * Compute one FP32 tile from twelve-row A and eight-column B microgroups.
 *
 * @param m,n,k Logical tile dimensions, excluding packed padding.
 * @param alpha Product scale; zero alpha or zero K leaves C untouched.
 * @param A Packed A, padded to a multiple of twelve rows.
 * @param lda Packed K stride; must equal k when k is positive.
 * @param B Packed B, padded to a multiple of eight columns.
 * @param ldb Packed K stride, at least k.
 * @param C Column-major output tile with at least m rows per column.
 * @param ldc Output column stride in scalar elements.
 * @param initialize One to overwrite C without reading it; zero to accumulate
 *        into C that the caller has already scaled by beta.
 * @return Zero on success, or CAMBLAS_PACKED_UNAVAILABLE with C unchanged when
 *         the shape, address span or SVE128 capability checks fail.
 */
CAMBLAS_PACKED_INTERNAL CAMBLAS_PACKED_WEAK int
camblas_sgemm_sve_amicro12_tile(int m, int n, int k, float alpha, const float *A, int lda,
                                const float *B, int ldb, float *C, int ldc, int initialize);
/**
 * FP64 counterpart of camblas_sgemm_sve_amicro12_tile.
 *
 * A is padded to six-row groups, B to eight-column groups, and lda equals K.
 * The scalar, output-span, initialisation and SVE128 preconditions match the
 * FP32 hook. Unavailability leaves C unchanged.
 */
CAMBLAS_PACKED_INTERNAL CAMBLAS_PACKED_WEAK int
camblas_dgemm_sve_amicro6_tile(int m, int n, int k, double alpha, const double *A, int lda,
                               const double *B, int ldb, double *C, int ldc, int initialize);
/* FP64 A uses six-row microgroups with lda=K; B uses eight-column microgroups. */
CAMBLAS_PACKED_INTERNAL CAMBLAS_PACKED_WEAK int
camblas_dgemm_sve_amicro_tile(int m, int n, int k, double alpha, const double *A, int lda,
                              const double *B, int ldb, double *C, int ldc);

/* B uses the private micro8 layout; other contracts match packed_tile. */
CAMBLAS_PACKED_INTERNAL CAMBLAS_PACKED_WEAK int
camblas_dgemm_sve_interleaved_tile(int m, int n, int k, double alpha, const double *A, int lda,
                                   const double *B, int ldb, double *C, int ldc);

CAMBLAS_PACKED_INTERNAL CAMBLAS_PACKED_WEAK int
camblas_sgemm_sve_micro6_tile(int m, int n, int k, float alpha, const float *A, int lda,
                              const float *B, int ldb, float *C, int ldc);
CAMBLAS_PACKED_INTERNAL CAMBLAS_PACKED_WEAK int
camblas_dgemm_sve_micro6_tile(int m, int n, int k, double alpha, const double *A, int lda,
                              const double *B, int ldb, double *C, int ldc);

CAMBLAS_PACKED_INTERNAL CAMBLAS_PACKED_WEAK int
camblas_sgemm_sve_micro6_padded_tile(int m, int n, int k, float alpha, const float *A, int lda,
                                     const float *B, int ldb, float *C, int ldc);
CAMBLAS_PACKED_INTERNAL CAMBLAS_PACKED_WEAK int
camblas_dgemm_sve_micro6_padded_tile(int m, int n, int k, double alpha, const double *A, int lda,
                                     const double *B, int ldb, double *C, int ldc);

CAMBLAS_PACKED_INTERNAL int camblas_sgemm_packed_executor_workspace(
    char trans_a, char trans_b, int m, int n, int k, float alpha, const float *A, int lda,
    const float *B, int ldb, float beta, float *C, int ldc, int mc, int nc, int kc, int mr, int nr,
    const camblas_executor_t *executor, void *workspace, size_t workspace_bytes);
CAMBLAS_PACKED_INTERNAL int camblas_dgemm_packed_executor_workspace(
    char trans_a, char trans_b, int m, int n, int k, double alpha, const double *A, int lda,
    const double *B, int ldb, double beta, double *C, int ldc, int mc, int nc, int kc, int mr,
    int nr, const camblas_executor_t *executor, void *workspace, size_t workspace_bytes);

CAMBLAS_PACKED_INTERNAL int camblas_sgemm_packed_executor(char trans_a, char trans_b, int m, int n,
                                                          int k, float alpha, const float *A,
                                                          int lda, const float *B, int ldb,
                                                          float beta, float *C, int ldc, int mc,
                                                          int nc, int kc, int mr, int nr,
                                                          const camblas_executor_t *executor);

CAMBLAS_PACKED_INTERNAL int camblas_dgemm_packed_executor(char trans_a, char trans_b, int m, int n,
                                                          int k, double alpha, const double *A,
                                                          int lda, const double *B, int ldb,
                                                          double beta, double *C, int ldc, int mc,
                                                          int nc, int kc, int mr, int nr,
                                                          const camblas_executor_t *executor);

#endif /* CAMBLAS_INTERNAL_PACKED_H */
