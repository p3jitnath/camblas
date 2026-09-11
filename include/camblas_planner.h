/*
 * CAMBLAS runtime planner — shape-aware execution plan selection.
 *
 * The planner is a PURE FUNCTION: no global state, no allocation, no side
 * effects. It consumes an operation descriptor, a topology snapshot, and an
 * execution context, and produces a caller-owned plan struct.
 *
 * The planner uses explicit rules based on shape thresholds and bounded
 * cache-capacity estimates, rather than a learned cost model. The build
 * configuration selects the retained blocking and dispatch parameters.
 *
 * DESIGN PRINCIPLES:
 *   1. Separation of planning from execution. The planner produces a plan;
 *      the executor attempts to follow it, falling back to scalar if the
 *      requested kernel is unavailable. This separation is itself a research
 *      contribution (observability).
 *   2. No global mutable state. All functions are reentrant.
 *   3. Immutable string literals for name/rationale functions. Caller-owned
 *      buffers for format functions.
 *   4. Fail-closed: invalid inputs produce an error return, not a default plan.
 *
 * PRECONDITIONS:
 *   camblas_plan_make: op, ctx, and plan must be non-NULL. planner_mode must
 *     be NULL (treated as "none") or one of "none", "fixed", "shape-aware",
 *     "ablation", and "sve". topo may be NULL only for planner_mode "none"
 *     and "fixed"; "shape-aware", "ablation", and "sve" require a valid
 *     topology (non-NULL, no fatal status). For the topology fields consumed
 *     here, n_allowed_cpus must be in [1, CAMBLAS_MAX_CPUS] and must equal
 *     the set-bit count of a canonical allowed_cpus set. An empty, stale, or
 *     scalar/bit-count-mismatched allowed mask is a fatal topology context and
 *     returns -2 for shape-aware/ablation (and -1 for sve). Nonzero NUMA
 *     metadata must contain canonical, nonempty, disjoint node CPU sets whose
 *     scalar counts agree and whose CPUs are contained in allowed_cpus;
 *     malformed NUMA metadata is rejected as invalid optional context. Zero
 *     n_numa_nodes, zero n_online_cpus, and zero sve_vl_bits represent
 *     optional/unavailable metadata. A positive n_online_cpus must be at least
 *     n_allowed_cpus; a smaller positive value is malformed optional context.
 *     A nonzero sve_vl_bits must be a valid 128-bit-granular Linux SVE value
 *     in [128, 65536], and is required by "sve" mode. Malformed optional
 *     metadata is rejected rather than publishing a plan; n_online_cpus
 *     remains reference-only. reproducibility
 *     must be one of CAMBLAS_REPRO_FAST, CAMBLAS_REPRO_REPRODUCIBLE, or
 *     CAMBLAS_REPRO_BITWISE. All context checks happen before the
 *     bitwise-reproducibility override; an unknown reproducibility level
 *     cannot sanitize malformed planner state or publish a plan.
 *     num_threads is a request: values below 1 are normalized to 1, and
 *     positive values are capped by the available topology and policy where
 *     applicable. This normalization is valid context handling, not an
 *     error, and is completed before any plan is written or no-op returned.
 *   camblas_plan_format: plan and buf must be non-NULL, bufsize > 0.
 *   Name/rationale functions accept any value; out-of-range returns "unknown".
 */
#ifndef CAMBLAS_PLANNER_H
#define CAMBLAS_PLANNER_H

#ifdef __cplusplus
extern "C" {
#endif

#include "camblas.h"
#include "camblas_topology.h"
#include <stddef.h>

/* ---- Constants ---- */

/*
 * Data types for the operation descriptor.
 */
#define CAMBLAS_DTYPE_F32 0 /* float32 / SGEMM */
#define CAMBLAS_DTYPE_F64 1 /* float64 / DGEMM */

/*
 * Kernel IDs.
 * The planner selects one of these; the executor decides whether the
 * requested kernel is available and falls back to SCALAR if not.
 */
#define CAMBLAS_KERNEL_SCALAR 0  /* portable scalar reference */
#define CAMBLAS_KERNEL_NOPACK 1  /* no-pack optimized (avoids packing) */
#define CAMBLAS_KERNEL_PACKED 2  /* packed blocked (cache-aware) */
#define CAMBLAS_KERNEL_BATCHED 3 /* strided-batch aware */
#define CAMBLAS_KERNEL_SVE 4     /* single-core vector-length-agnostic SVE */

/*
 * Packing options.
 */
#define CAMBLAS_PACK_NONE 0 /* no packing (scalar, no-pack kernels) */
#define CAMBLAS_PACK_A 1    /* pack A only */
#define CAMBLAS_PACK_B 2    /* pack B only */
#define CAMBLAS_PACK_BOTH 3 /* pack both A and B */

/*
 * Parallel decomposition schemes.
 */
#define CAMBLAS_PARALLEL_NONE 0 /* single-threaded */
#define CAMBLAS_PARALLEL_ROW 1  /* split M dimension across threads */
#define CAMBLAS_PARALLEL_COL 2  /* split N dimension across threads */
#define CAMBLAS_PARALLEL_K 3    /* split K (requires partial reduction) */
#define CAMBLAS_PARALLEL_2D 4   /* 2D M x N decomposition */

/*
 * Rationale codes (for plan trace).
 * Each code maps to an immutable string via camblas_plan_rationale().
 */
#define CAMBLAS_RAT_NONE_DISABLED 0   /* planner disabled */
#define CAMBLAS_RAT_FIXED_POLICY 1    /* fixed policy (ablation baseline) */
#define CAMBLAS_RAT_SMALL_SCALAR 2    /* tiny matrix, scalar optimal */
#define CAMBLAS_RAT_MEDIUM_NOPACK 3   /* medium matrix, no-pack */
#define CAMBLAS_RAT_LARGE_PACKED 4    /* large matrix, packed blocked */
#define CAMBLAS_RAT_BATCH 5           /* batched GEMM path */
#define CAMBLAS_RAT_SINGLE_THREAD 6   /* topology has 1 CPU */
#define CAMBLAS_RAT_REPRO_BITWISE 7   /* bitwise repro forces scalar 1-thread */
#define CAMBLAS_RAT_SVE_SINGLE_CORE 8 /* explicit single-core SVE path */
#define CAMBLAS_RAT_SVE_FALLBACK 9    /* SVE mode, unsupported transpose */

/* ---- Operation descriptor (planner input) ---- */

/*
 * Describes a single GEMM operation for planning.
 * Dimensions m, n, k must be >= 0. trans_a/trans_b must be 'N' or 'T'.
 * dtype must be CAMBLAS_DTYPE_F32 or CAMBLAS_DTYPE_F64.
 * batch_count must be >= 1 (1 for single GEMM).
 */
typedef struct {
    int m, n, k;
    char trans_a, trans_b;
    int dtype;
    int batch_count;
} camblas_op_t;

/* ---- Execution plan (planner output) ---- */

/*
 * The execution plan produced by the planner.
 * All fields are filled by camblas_plan_make(). The plan is a value type:
 * it may be copied, compared with memcmp, and does not own any resources.
 */
typedef struct {
    int kernel_id;       /* CAMBLAS_KERNEL_* */
    int pack;            /* CAMBLAS_PACK_* */
    int num_threads;     /* effective threads (>= 1, capped by topology/policy) */
    int parallel_scheme; /* CAMBLAS_PARALLEL_* */
    int mc, nc, kc;      /* cache block sizes (0 if unused) */
    int mr, nr;          /* register tile sizes (0 if unused) */
    int rationale_id;    /* CAMBLAS_RAT_* (for trace) */
} camblas_plan_t;

/* ---- Planner function ---- */

/*
 * Produce an execution plan for the given operation, topology, and context.
 *
 * Parameters:
 *   op    — operation descriptor (must be non-NULL, dimensions >= 0)
 *   topo  — topology snapshot (may be NULL for "none"/"fixed" modes;
 *           must be non-NULL with no fatal status for "shape-aware"/"ablation")
 *   ctx   — execution context (must be non-NULL)
 *   plan  — output plan (must be non-NULL, filled on success)
 *
 * Returns:
 *    0  on success
 *   -1  on invalid argument (NULL pointer, negative dimension, invalid
 *        trans/dtype/batch, unknown planner mode, or missing/invalid required
 *        topology)
 *   -2  on a fatal allowed-mask error, including an out-of-range
 *        n_allowed_cpus or an inconsistent allowed_cpus representation
 *        (only returned for shape-aware/ablation modes)
 *
 * Thread safety: reentrant. No global state.
 */
int camblas_plan_make(const camblas_op_t *op, const camblas_topology_t *topo,
                      const camblas_ctx_t *ctx, camblas_plan_t *plan);

/* ---- Observability functions ---- */

/*
 * Format a plan as a compact, single-line trace string.
 * Example: "kernel=scalar pack=none threads=1 parallel=none rationale=..."
 * plan and buf must be non-NULL, bufsize > 0.
 * Returns characters written (excluding NUL), or -1 on failure.
 */
int camblas_plan_format(const camblas_plan_t *plan, char *buf, size_t bufsize);

/*
 * Return the rationale string for a plan.
 * Returns an immutable string literal. Never returns NULL.
 */
const char *camblas_plan_rationale(const camblas_plan_t *plan);

/*
 * Return the kernel name string.
 * Returns an immutable string literal for valid IDs, "unknown" otherwise.
 */
const char *camblas_kernel_name(int kernel_id);

/*
 * Return the parallel scheme name string.
 * Returns an immutable string literal for valid IDs, "unknown" otherwise.
 */
const char *camblas_parallel_name(int scheme);

/*
 * Return the pack option name string.
 * Returns an immutable string literal for valid values, "unknown" otherwise.
 */
const char *camblas_pack_name(int pack);

/* ---- Plan-aware GEMM (observable decision interface) ---- */

/*
 * camblas_sgemm_plan / camblas_dgemm_plan: execute a GEMM exactly like
 * camblas_sgemm / camblas_dgemm, but additionally write the execution plan
 * that was computed (and governs execution) to *plan_out when plan_out is
 * non-NULL. If plan_out is NULL, the call is identical to the plain GEMM.
 *
 * On error (return -1), plan_out is NOT written.
 *
 * The interface reports the selected kernel, packing plan, thread count,
 * and decision rationale. The plan written is identical to what camblas_plan_make()
 * returns for the same op/topo/ctx, because the planner is a pure function
 * with no global state — querying the plan separately or obtaining it as a
 * side effect of execution yields the same result.
 *
 * Callers wanting a human-readable trace can pass the plan to
 * camblas_plan_format() and camblas_plan_rationale().
 *
 * Thread safety: reentrant (same as the plain GEMM). No global state.
 */
int camblas_sgemm_plan(const camblas_ctx_t *ctx, char trans_a, char trans_b, int m, int n, int k,
                       float alpha, const float *A, int lda, const float *B, int ldb, float beta,
                       float *C, int ldc, camblas_plan_t *plan_out);

int camblas_dgemm_plan(const camblas_ctx_t *ctx, char trans_a, char trans_b, int m, int n, int k,
                       double alpha, const double *A, int lda, const double *B, int ldb,
                       double beta, double *C, int ldc, camblas_plan_t *plan_out);

#ifdef __cplusplus
}
#endif

#endif /* CAMBLAS_PLANNER_H */
