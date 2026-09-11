/*
 * CAMBLAS executor — implementation of the default serial executor.
 *
 * See include/camblas_executor.h for the full contract. The serial executor is
 * the fallback used when ctx->executor is NULL or points to this exported
 * default; it reproduces today's single-threaded, in-order execution. It is
 * reentrant and uses no global state.
 *
 * Caller-provided executors are NOT defined here; they live in application code
 * (and in the test harness). This file only owns the default.
 */
#include "camblas_executor.h"

int camblas_executor_serial_fn(camblas_task_fn fn, const camblas_task_t *tasks, int n_tasks,
                               void *gctx, void *user_data)
{
    (void)user_data; /* the serial executor carries no state */
    if (!fn || n_tasks < 0 || (n_tasks > 0 && !tasks))
        return -1;
    for (int k = 0; k < n_tasks; k++) {
        fn(&tasks[k], gctx);
    }
    return 0;
}

const camblas_executor_t camblas_executor_serial = {
    .run = camblas_executor_serial_fn,
    .user_data = NULL,
};
