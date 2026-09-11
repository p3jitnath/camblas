/*
 * CAMBLAS execution-context executor — caller-owned parallelism contract.
 *
 * The per-call execution context can use a caller-provided executor without
 * introducing process-wide threading state into the native interface.
 *
 * PROBLEM IT SOLVES
 *   Conventional BLAS libraries (OpenBLAS, BLIS, Cray LibSci) drive internal
 *   parallelism through PROCESS-GLOBAL state: openblas_set_num_threads(),
 *   OMP_NUM_THREADS, bli_global_rntm(), etc. In a concurrently threaded
 *   application, every BLAS call contends for the same global knob, so
 *   nested/parallel callers cannot independently control their BLAS threading
 *   without racing on that global state. This is the core composability
 *   failure mode CAMBLAS targets (RQ1/RQ2).
 *
 * CAMBLAS DESIGN
 *   The GEMM execution path partitions the output matrix into independent
 *   TILE TASKS and hands them to an EXECUTOR. The executor is referenced
 *   PER-CALL from the camblas_ctx_t (ctx->executor), NOT from any global. So:
 *     - Two callers can issue GEMM concurrently with DIFFERENT executors
 *       (e.g. different thread pools) without interfering.
 *     - A caller that already owns a thread pool / task system can plug it in
 *       as the executor, so BLAS parallelism COMPOSES with the caller's
 *       parallelism rather than fighting it (no oversubscription).
 *     - The default executor (camblas_executor_serial) runs tasks in the
 *       calling thread, in order. Scalar execution remains the portable
 *       deterministic path; the correctness-gated packed branch also uses
 *       the same interface when a caller supplies an executor.
 *
 * TASK INDEPENDENCE CONTRACT
 *   Every task describes a DISJOINT region of the output C: rows [i0,i1) and
 *   columns [j0,j1). Two different tasks never write the same C element, and
 *   each task only READS its A/B operands. Therefore an executor MAY run tasks
 *   concurrently and in any order without data races. The serial executor
 *   produces bitwise-identical output to the prior monolithic scalar kernel
 *   because the per-element operation sequence is unchanged (only the loop
 *   partitioning differs). A concurrent executor also produces a correct
 *   result; whether it is bitwise-identical depends only on each element's
 *   independent accumulation, which is unchanged.
 *
 * CURRENT SCOPE / NOT CLAIMED
 *   - Scalar and packed execution use this interface, including the tuned
 *     Grace kernels. Packing and compute phases complete synchronously;
 *     callers must not share mutable workspace between independent calls.
 *   - Scalar execution currently splits only M (full column range per task).
 *     The packed correctness gate uses 2D M x N rectangles; K decomposition
 *     and reduction tasks remain future work.
 *   - The executor does not set affinity or NUMA policy; that is the caller's
 *     responsibility (the caller owns the threads). Topology-aware placement is
 *     surfaced separately via camblas_topology_t.
 *
 * CONTRACT SEMANTICS
 *
 *   Synchronous completion:
 *     run() MUST return only after every invocation of fn has fully completed.
 *     When run() returns 0, all writes to C by every task invocation are
 *     visible to the calling thread (the executor establishes a happens-before
 *     relationship between task completion and run() return — e.g. via
 *     pthread_join, mutex release, or equivalent). The caller may immediately
 *     read the result matrix C. An executor that dispatches asynchronously
 *     and returns before tasks finish violates the contract.
 *
 *   Callback ("exactly once") semantics:
 *     run() MUST call fn(&tasks[k], gctx) exactly once for every k in
 *     [0, n_tasks). It MUST NOT call fn for k outside [0, n_tasks). It MUST
 *     NOT call fn after run() has returned. Each task pointer passed to fn
 *     remains valid for the duration of that fn call; gctx remains valid for
 *     the duration of run(). fn writes only the C elements in the task's
 *     [i0,i1) x [j0,j1) region and reads only A/B via gctx.
 *
 *   Error / fail-closed semantics:
 *     run() returns 0 on success, -1 on failure. If run() returns -1, the
 *     GEMM returns -1 to its caller (fail-closed): the operation is treated
 *     as failed and plan_out is not written. The state of C after an error is
 *     UNSPECIFIED — partial computation may have occurred, so the caller must
 *     not rely on C. An executor that cannot guarantee all tasks completed
 *     successfully MUST return -1.
 *     A negative n_tasks value is invalid and must return -1. For n_tasks == 0,
 *     tasks may be NULL and the executor must return 0 without invoking fn.
 *
 *   Lifetime:
 *     The executor handle (camblas_executor_t), the tasks array, and gctx
 *     must remain valid for the entire duration of the run() call. The
 *     executor MUST NOT retain pointers to tasks, gctx, or fn after run()
 *     returns (no asynchronous escape of GEMM-internal state). The executor
 *     handle itself (ctx->executor) must remain valid for the duration of
 *     the GEMM call that reads it.
 *
 *   Reentrancy:
 *     The executor interface uses no global mutable state. The default serial
 *     executor is reentrant (it is a plain in-order loop), so a GEMM call
 *     issued from within a task fn (nested GEMM) is safe with the serial
 *     executor. A caller-provided executor is responsible for its own
 *     synchronization if shared across threads. A fixed-size thread-pool
 *     executor may DEADLOCK if a task fn issues a nested GEMM that dispatches
 *     to the same pool with no spare workers — such executors should either
 *     document this limitation or provide enough workers for nesting.
 *
 *   n_tasks == 0:
 *     After the public arguments and execution context have been validated,
 *     the GEMM does NOT invoke run() when there are no output tasks (m == 0
 *     or n == 0): it returns 0 directly for a valid executor. A non-NULL
 *     caller-owned executor whose run callback is NULL remains invalid and is
 *     rejected before this no-op rule; plan_out is left untouched on that
 *     error. The same executor validity rule applies to positive-output
 *     alpha==0 or zero-K, beta==1 no-read/no-write calls: a valid executor is
 *     not invoked, while a malformed caller-owned executor is rejected.
 *     Executor authors should still handle n_tasks == 0 defensively by
 *     returning 0 without calling fn; a NULL tasks pointer is acceptable in
 *     this zero-task case. Negative counts are invalid.
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
