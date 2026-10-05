/*
 * CAMBLAS uses a caller-owned executor to dispatch independent GEMM tasks.
 * Each camblas_ctx_t references its executor through ctx->executor, so callers
 * can use different pools or reuse an application's pool without changing
 * process-wide BLAS thread settings. The default camblas_executor_serial runs
 * tasks in order on the calling thread.
 *
 * Tasks write disjoint regions of C, with rows [i0,i1) and columns [j0,j1), and
 * only read A/B through gctx. They can run concurrently in any order without
 * racing on C. Partitioning preserves each element's accumulation sequence;
 * serial scalar output therefore matches the preceding monolithic scalar
 * kernel bitwise, and concurrency preserves that per-element sequence.
 *
 * Scalar and packed kernels, including the Grace kernels, use this interface.
 * Packing and compute complete synchronously, and independent calls require
 * distinct mutable workspaces. Scalar execution currently partitions M with
 * the full column range per task; the packed correctness gate uses M x N
 * rectangles. K decomposition and reduction tasks remain future work. The
 * caller sets affinity and NUMA policy; camblas_topology_t describes placement.
 *
 * run() must finish every fn invocation before returning. On success, task
 * writes to C must be visible to the calling thread through a happens-before
 * relationship, for example pthread_join or mutex synchronisation. The caller
 * can then read C immediately.
 *
 * run() must call fn(&tasks[k], gctx) exactly once for each k in [0,n_tasks).
 * It must not call fn outside that range or after returning. Each task pointer
 * remains valid during its fn call, and gctx remains valid throughout run().
 * fn writes only its task's C region and reads A/B through gctx.
 *
 * Return 0 on success and -1 on failure. If all tasks cannot be guaranteed to
 * complete successfully, return -1; GEMM then returns -1 and leaves plan_out
 * untouched. C may contain partial results after an error and is unspecified.
 * Negative n_tasks is invalid and must return -1. For n_tasks == 0, tasks may
 * be NULL, and run() must return 0 without calling fn.
 *
 * Keep the executor handle, tasks and gctx valid throughout run(), and keep
 * ctx->executor valid throughout the GEMM call. The executor must not retain
 * tasks, gctx or fn after returning.
 *
 * The interface has no global mutable state, and the serial executor supports
 * nested GEMM calls. A shared caller-owned executor requires its own
 * synchronisation. A fixed-size pool can deadlock when a task submits a nested
 * GEMM to the same pool without spare workers; provide capacity for nesting or
 * document the restriction.
 *
 * GEMM validates arguments and context before either no-op form. With a valid
 * executor, m == 0 or n == 0 returns 0 without calling run(). A non-NULL executor
 * with a NULL callback is invalid, and plan_out remains untouched on rejection.
 * The same rule applies when positive-output alpha==0 or zero-K calls have
 * beta==1: a valid executor is not invoked, and a malformed one is rejected.
 * Executor implementations should still handle zero-task calls as specified.
 */
#ifndef CAMBLAS_EXECUTOR_H
#define CAMBLAS_EXECUTOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/* ---- Tile task ---- */

/*
 * A single tile task: compute C[i,j] for i in [i0, i1) and j in [j0, j1).
 * Ranges are half-open. i1<=i0 or j1<=j0 means an empty task (no work).
 *
 * Tasks produced by a single GEMM call have DISJOINT (i,j) regions and may be
 * executed concurrently and in any order (see contract above).
 */
typedef struct {
    int i0, i1; /* row range of C, half-open [i0, i1) */
    int j0, j1; /* column range of C, half-open [j0, j1) */
} camblas_task_t;

/*
 * Per-task compute function. The GEMM provides this; the executor calls it
 * exactly once per task. gctx is an opaque pointer to the GEMM's internal
 * work descriptor (arguments + operands). The function reads gctx and writes
 * only the C elements in the task's [i0,i1) x [j0,j1) region.
 */
typedef void (*camblas_task_fn)(const camblas_task_t *task, void *gctx);

/* ---- Executor ---- */

/*
 * Executor dispatch callback.
 *
 * Contract: see CONTRACT SEMANTICS above. Summary:
 *   - Call fn(&tasks[k], gctx) exactly once for every k in [0, n_tasks).
 *   - MAY run the calls concurrently and in any order.
 *   - MUST return only after every fn call has completed (synchronous).
 *   - Return 0 on success, -1 on failure (GEMM fails closed on -1).
 *
 * user_data is the executor's own state (e.g. a thread-pool handle or a
 * worker count); it is opaque to CAMBLAS.
 */
typedef int (*camblas_exec_fn)(camblas_task_fn fn, const camblas_task_t *tasks, int n_tasks,
                               void *gctx, void *user_data);

/*
 * Executor handle, stored by reference in camblas_ctx_t.
 *   run        — the dispatch callback (must be non-NULL for a valid executor).
 *   user_data  — opaque state passed as the last argument to run.
 */
typedef struct {
    camblas_exec_fn run;
    void *user_data;
} camblas_executor_t;

/* ---- Default serial executor ---- */

/*
 * Default serial executor: runs every task in the calling thread, in index
 * order. Synchronous (returns only after all fn calls complete). Never calls
 * fn after returning. Returns 0 for a valid zero-or-more-task call and -1 for
 * a missing callback, a negative task count, or a missing task array when
 * n_tasks is positive. user_data is ignored.
 * Reentrant (plain in-order loop; safe for nested GEMM from within fn).
 *
 * This is used automatically when ctx->executor is NULL, so existing callers
 * get today's single-threaded behavior without changes.
 */
int camblas_executor_serial_fn(camblas_task_fn fn, const camblas_task_t *tasks, int n_tasks,
                               void *gctx, void *user_data);

/*
 * A ready-to-use serial executor. Equivalent to a NULL ctx->executor.
 * (ctx->executor == NULL and ctx->executor == &camblas_executor_serial are
 * treated identically by the GEMM.)
 */
extern const camblas_executor_t camblas_executor_serial;

#ifdef __cplusplus
}
#endif

#endif /* CAMBLAS_EXECUTOR_H */
