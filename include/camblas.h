/*
 * CAMBLAS declares the execution context and the column-major GEMM interface.
 * The context carries the thread count, reproducibility level, planner mode,
 * topology snapshot and optional executor. GEMM calls the planner internally,
 * so callers do not need to include camblas_planner.h.
 *
 * The interface uses LP64 dimensions. Its current identity is API 0.1.0-dev,
 * ABI 0.2, as defined in camblas_version.h; ABI major 0 remains unstable.
 * Compile callers against matching headers and libraries. Adding context
 * fields preserves source compatibility for designated initialisers, but it
 * changes the binary layout and requires existing objects to be recompiled.
 */
#ifndef CAMBLAS_H
#define CAMBLAS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include "camblas_version.h"
#include "camblas_topology.h" /* camblas_topology_t for ctx.topo */
#include "camblas_executor.h" /* camblas_executor_t for ctx.executor */

/* Reproducibility levels for the native execution context. */
#define CAMBLAS_REPRO_FAST 0         /* best-effort, nondeterministic */
#define CAMBLAS_REPRO_REPRODUCIBLE 1 /* numerical contract */
#define CAMBLAS_REPRO_BITWISE 2      /* bitwise-identical across runs */

/*
 * The caller owns the execution context and the resources it references.
 * Every GEMM receives a context pointer; NULL selects a single-threaded fast
 * context. The following requirements also apply to calls that perform no work.
 *
 * num_threads is a requested count. Planning normalises values below 1 to 1
 * and may cap positive values according to topology or planner policy. The
 * plan reports the effective count, which is 1 for policies that run serially.
 * Normalisation and capping precede publication of a plan for either no-op form.
 *
 * reproducibility must be CAMBLAS_REPRO_FAST, CAMBLAS_REPRO_REPRODUCIBLE or
 * CAMBLAS_REPRO_BITWISE. An invalid value is rejected rather than converted to
 * fast mode. Bitwise selection still requires valid planner and topology state.
 *
 * topo is an optional snapshot. Without one, the planner can use "none" or
 * "fixed"; "shape-aware", "ablation" and "sve" fail. Unknown modes are invalid.
 * Discover topology with camblas_topology_discover before setting this pointer.
 * Topology-dependent calls require an allowed-CPU count in 1..CAMBLAS_MAX_CPUS
 * that agrees with the canonical allowed_cpus bitset. Empty, stale or
 * inconsistent allowed masks are invalid.
 *
 * Zero SVE VL, NUMA-node count or online-CPU count means the optional information
 * is unavailable. A positive online count must cover the allowed CPU set; an
 * inconsistent discovered count is cleared to zero, while an inconsistent
 * caller-supplied count is rejected. Malformed nonzero SVE metadata and
 * malformed nonempty NUMA metadata are rejected. The "sve" mode additionally
 * requires a valid VL in the Linux UAPI range, in multiples of 128 bits.
 *
 * executor is an optional caller-owned task dispatcher. NULL uses the serial
 * executor, which runs tasks in order on the calling thread. An application can
 * supply its own pool so that each call uses its chosen parallelism without
 * changing process-wide thread settings. camblas_executor.h defines the contract.
 */
typedef struct {
    int num_threads;                    /* requested BLAS-internal threads; <1 -> 1 */
    int reproducibility;                /* CAMBLAS_REPRO_* */
    const char *planner_mode;           /* "none", "fixed", "shape-aware", "ablation", "sve" */
    const camblas_topology_t *topo;     /* optional topology snapshot (NULL OK) */
    const camblas_executor_t *executor; /* optional executor (NULL = serial) */
} camblas_ctx_t;

/*
 * Column-major GEMM: C = alpha * op(A) * op(B) + beta * C
 *
 * trans_a, trans_b: 'N' (no transpose) or 'T' (transpose).
 * A is (lda, k) if trans_a='N', (lda, m) if trans_a='T'.
 * B is (ldb, n) if trans_b='N', (ldb, k) if trans_b='T'.
 * C is (ldc, n), m rows.
 * When m and n are both positive, required leading dimensions are
 * lda >= max(1, trans_a=='N' ? m : k),
 * ldb >= max(1, trans_b=='N' ? k : n), and ldc >= max(1, m).
 * If m==0 or n==0, the call is a no-op and only lda/ldb/ldc >= 1 is required.
 * A, B, and C must be non-NULL even when a dimension is zero.
 * For positive output dimensions, alpha==0 or k==0 with beta==1 is also a
 * no-read, no-write no-op; the non-NULL and leading-dimension requirements
 * still apply. Executor validation remains part of the call contract: a
 * non-NULL caller-owned executor with a NULL run callback is invalid even for
 * either no-op form.
 *
 * Returns 0 on success, -1 on invalid argument or invalid execution context.
 * Context validation, including reproducibility and planner-mode selectors,
 * precedes either no-op form and plan_out publication.
 */
int camblas_sgemm(const camblas_ctx_t *ctx, char trans_a, char trans_b, int m, int n, int k,
                  float alpha, const float *A, int lda, const float *B, int ldb, float beta,
                  float *C, int ldc);

int camblas_dgemm(const camblas_ctx_t *ctx, char trans_a, char trans_b, int m, int n, int k,
                  double alpha, const double *A, int lda, const double *B, int ldb, double beta,
                  double *C, int ldc);

/* Default context: 1 thread, fast, planner "none". */
extern const camblas_ctx_t camblas_ctx_default;

#ifdef __cplusplus
}
#endif

#endif /* CAMBLAS_H */
