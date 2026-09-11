/*
 * CAMBLAS public API header.
 *
 * Declares the execution-context type and the GEMM interface. The execution
 * context carries a thread count, reproducibility level, planner mode, and
 * an optional topology snapshot. The planner is called internally by the
 * GEMM functions; callers do not need to include camblas_planner.h.
 *
 * ABI policy: LP64 initially. The struct layout is NOT part of a stable ABI
 * yet; callers must recompile against the header. The current development
 * identity is in camblas_version.h (API 0.1.0-dev, ABI 0.2); ABI major 0 is
 * explicitly unstable. A stable ABI has not yet been frozen. Adding
 * fields to camblas_ctx_t (e.g. the executor field) is SOURCE-compatible
 * (existing code using designated initializers recompiles without changes)
 * but NOT binary-ABI-compatible (the struct layout changes; old object files
 * or shared libraries built against a prior layout must be recompiled).
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
 * Execution context: caller-owned resources and policy.
 * Passed by pointer to every GEMM call. NULL is treated as a default
 * single-threaded fast context.
 *
 * topo: optional topology snapshot. NULL means "no topology available";
 *   the planner will use "none" or "fixed" mode without topology, or fail
 *   closed for "shape-aware", "ablation", and "sve" modes. Unknown planner
 *   modes are invalid. Reproducibility must be one of CAMBLAS_REPRO_FAST,
 *   CAMBLAS_REPRO_REPRODUCIBLE, or CAMBLAS_REPRO_BITWISE; an out-of-range
 *   value is an invalid context and cannot be converted to fast behavior.
 *   These context requirements remain in force when reproducibility is
 *   CAMBLAS_REPRO_BITWISE; bitwise selection cannot make malformed planner
 *   state valid. Callers who want topology-aware planning must discover
 *   topology (camblas_topology_discover) and set this pointer.
 *   A topology-dependent call also requires the snapshot's allowed-CPU count
 *   to be representable (1..CAMBLAS_MAX_CPUS) and consistent with the
 *   canonical allowed_cpus bitset. An empty, stale, or mismatched allowed
 *   mask is a fatal topology context. Zero SVE VL, zero NUMA nodes, and zero
 *   online CPUs are the documented optional/unavailable results. A positive
 *   online count must cover the allowed CPU set; an inconsistent discovered
 *   value is cleared to zero and an inconsistent caller-supplied value is
 *   rejected. Malformed nonzero SVE metadata or malformed nonempty NUMA
 *   metadata is rejected, and
 *   SVE mode additionally requires a valid 128-bit-granular VL in the Linux
 *   UAPI range.
 *   num_threads is a requested count, not an invalid-context selector:
 *   values below 1 are normalized to 1, while positive values may be capped
 *   by the available topology or the selected planner policy. The effective
 *   count is reported in the plan; modes that are serial by policy always
 *   report one thread. This normalization and capping occur during planning
 *   before either public no-op form can publish a plan.
 *
 * executor: optional caller-provided executor for tile-task dispatch. NULL
 *   means "use the default serial executor" (tasks run in the calling thread,
 *   in order — today's behavior). A non-NULL executor lets the caller own the
 *   parallelism (e.g. dispatch GEMM tiles to an existing thread pool), so
 *   BLAS parallelism composes with the caller's instead of contending on
 *   process-global thread state. See camblas_executor.h for the contract.
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
