#ifndef CAMBLAS_PTHREAD_H
#define CAMBLAS_PTHREAD_H
#include "camblas_executor.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Optional persistent, caller-owned Linux pthread executor. No global pool.
 * Create count dedicated workers, each pinned to the corresponding distinct
 * CPU in cpus. All CPUs must belong to the creating thread's allowed mask.
 * Returns NULL on invalid input or resource/affinity failure, without leaving
 * workers alive. Link the optional implementation with -pthread.
 *
 * Calls sharing a pool are serialized; separate pools can run concurrently.
 * Recursive dispatch to the same pool from its own callback runs serially
 * on that worker. Cyclic nesting across different pools is unsupported.
 * The submitting thread waits and does not execute tasks. Pool creation and
 * destruction are outside GEMM calls. Destroy only after all users and nested
 * callbacks have completed; concurrent destruction or fork use is unsupported.
 * Callbacks must return normally. Thread cancellation, pthread_exit, and
 * nonlocal exits across executor frames are unsupported.
 * The returned executor is valid until destruction and must not be modified.
 */
typedef struct camblas_pthread_pool camblas_pthread_pool_t;

/**
 * Create a persistent pool with one worker pinned to each requested CPU.
 *
 * @param count Number of workers, in the range 1..CPU_SETSIZE.
 * @param cpus Array of count distinct CPU IDs in the caller's allowed mask.
 *        The array is read during creation and is not retained by the pool.
 * @return Caller-owned pool, or NULL on invalid input or initialisation failure.
 */
camblas_pthread_pool_t *camblas_pthread_pool_create(int count, const int *cpus);

/**
 * Borrow the synchronous executor interface from a live pool.
 *
 * @param pool Pool returned by camblas_pthread_pool_create, or NULL.
 * @return Immutable interface valid until pool destruction, or NULL for NULL.
 */
const camblas_executor_t *camblas_pthread_pool_executor(camblas_pthread_pool_t *pool);

/**
 * Join the workers and release a pool after all submissions have finished.
 *
 * @param pool Pool to destroy, or NULL for a no-op.
 * @note Callers must exclude concurrent users and destruction from callbacks.
 */
void camblas_pthread_pool_destroy(camblas_pthread_pool_t *pool);

#ifdef __cplusplus
}
#endif
#endif
