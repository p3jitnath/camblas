/* Persistent caller-owned executor. Submission is synchronous: task buffers
 * remain borrowed until every worker has finished the current generation. */
#define _GNU_SOURCE
#include "camblas_pthread.h"
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <limits.h>
#include <stdlib.h>
#ifndef CAMBLAS_PTHREAD_SPINS
#define CAMBLAS_PTHREAD_SPINS 0
#endif
#if CAMBLAS_PTHREAD_SPINS < 0 || CAMBLAS_PTHREAD_SPINS > 1000000
#error "CAMBLAS_PTHREAD_SPINS must be between zero and one million"
#endif

struct camblas_pthread_pool {
    camblas_executor_t executor;
    /* submit serialises callers; state protects batch publication, completion
     * and shutdown. next only assigns unique task indices within a batch. */
    pthread_mutex_t submit, state;
    pthread_cond_t work, done;
    pthread_t *threads;
    int count, remaining, stop, n_tasks;
    unsigned long generation;
    atomic_int next;
    camblas_task_fn fn;
    const camblas_task_t *tasks;
    void *gctx;
#if CAMBLAS_PTHREAD_SPINS > 0
    /* Keep polling away from mutable queue/state fields on 64-byte lines
       without requiring over-aligned allocation. */
    char notice_padding[64];
    atomic_ulong notice;
    char notice_tail[64];
#endif
};
static _Thread_local camblas_pthread_pool_t *active_pool;

static void *worker(void *arg)
{
    camblas_pthread_pool_t *p = arg;
    unsigned long seen = 0;
    pthread_mutex_lock(&p->state);
    for (;;) {
        while (!p->stop && seen == p->generation)
            pthread_cond_wait(&p->work, &p->state);
        if (p->stop)
            break;
        seen = p->generation;
        pthread_mutex_unlock(&p->state);
        active_pool = p;
        for (;;) {
            /* The state mutex publishes the task metadata. This atomic only
             * claims an index, so it does not need acquire/release ordering. */
            int index = atomic_fetch_add_explicit(&p->next, 1, memory_order_relaxed);
            if (index >= p->n_tasks)
                break;
            p->fn(&p->tasks[index], p->gctx);
        }
        active_pool = NULL;
        pthread_mutex_lock(&p->state);
        if (--p->remaining == 0)
            pthread_cond_signal(&p->done);
#if CAMBLAS_PTHREAD_SPINS > 0
        pthread_mutex_unlock(&p->state);
        for (unsigned spin = 0; spin < CAMBLAS_PTHREAD_SPINS; ++spin) {
            if (atomic_load_explicit(&p->notice, memory_order_relaxed) != seen)
                break;
#if defined(__aarch64__)
            __asm__ volatile("yield");
#elif defined(__x86_64__)
            __asm__ volatile("pause");
#endif
        }
        /* Metadata and stop are consumed only after taking the mutex.
           A notification racing with sleep is rechecked by the while loop. */
        pthread_mutex_lock(&p->state);
#endif
    }
    pthread_mutex_unlock(&p->state);
    return NULL;
}

static int run(camblas_task_fn fn, const camblas_task_t *tasks, int n_tasks, void *gctx, void *data)
{
    camblas_pthread_pool_t *p = data;
    /* Leave room for each worker's final atomic claim without signed overflow. */
    if (!p || !fn || n_tasks < 0 || n_tasks > INT_MAX - p->count || (n_tasks && !tasks))
        return -1;
    if (!n_tasks)
        return 0;
    /* A worker cannot wait for its own pool to finish a nested submission.
     * Execute that nested batch locally instead of reacquiring submit. */
    if (active_pool == p) {
        for (int i = 0; i < n_tasks; ++i)
            fn(&tasks[i], gctx);
        return 0;
    }
    pthread_mutex_lock(&p->submit);
    pthread_mutex_lock(&p->state);
    p->fn = fn;
    p->tasks = tasks;
    p->gctx = gctx;
    p->n_tasks = n_tasks;
    p->remaining = p->count;
    atomic_store_explicit(&p->next, 0, memory_order_relaxed);
    ++p->generation;
#if CAMBLAS_PTHREAD_SPINS > 0
    atomic_store_explicit(&p->notice, p->generation, memory_order_release);
#endif
    pthread_cond_broadcast(&p->work);
    while (p->remaining)
        pthread_cond_wait(&p->done, &p->state);
    p->fn = NULL;
    p->tasks = NULL;
    p->gctx = NULL;
    pthread_mutex_unlock(&p->state);
    pthread_mutex_unlock(&p->submit);
    return 0;
}

void camblas_pthread_pool_destroy(camblas_pthread_pool_t *p)
{
    if (!p)
        return;
    pthread_mutex_lock(&p->state);
    p->stop = 1;
    pthread_cond_broadcast(&p->work);
    pthread_mutex_unlock(&p->state);
    for (int i = 0; i < p->count; ++i)
        pthread_join(p->threads[i], NULL);
    pthread_cond_destroy(&p->done);
    pthread_cond_destroy(&p->work);
    pthread_mutex_destroy(&p->state);
    pthread_mutex_destroy(&p->submit);
    free(p->threads);
    free(p);
}

camblas_pthread_pool_t *camblas_pthread_pool_create(int count, const int *cpus)
{
    cpu_set_t allowed, used;
    if (count < 1 || count > CPU_SETSIZE || !cpus ||
        sched_getaffinity(0, sizeof(allowed), &allowed))
        return NULL;
    CPU_ZERO(&used);
    for (int i = 0; i < count; ++i) {
        if (cpus[i] < 0 || cpus[i] >= CPU_SETSIZE || !CPU_ISSET(cpus[i], &allowed) ||
            CPU_ISSET(cpus[i], &used))
            return NULL;
        CPU_SET(cpus[i], &used);
    }
    camblas_pthread_pool_t *p = calloc(1, sizeof(*p));
    if (!p)
        return NULL;
    p->threads = calloc((size_t)count, sizeof(*p->threads));
    if (!p->threads) {
        free(p);
        return NULL;
    }
    int stage = 0;
    if (pthread_mutex_init(&p->submit, NULL))
        goto fail;
    stage = 1;
    if (pthread_mutex_init(&p->state, NULL))
        goto fail;
    stage = 2;
    if (pthread_cond_init(&p->work, NULL))
        goto fail;
    stage = 3;
    if (pthread_cond_init(&p->done, NULL))
        goto fail;
    atomic_init(&p->next, 0);
#if CAMBLAS_PTHREAD_SPINS > 0
    atomic_init(&p->notice, 0);
#endif
    for (int i = 0; i < count; ++i) {
        pthread_attr_t attr;
        if (pthread_attr_init(&attr)) {
            camblas_pthread_pool_destroy(p);
            return NULL;
        }
        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET(cpus[i], &mask);
        int rc = pthread_attr_setaffinity_np(&attr, sizeof(mask), &mask);
        if (!rc)
            rc = pthread_create(&p->threads[i], &attr, worker, p);
        pthread_attr_destroy(&attr);
        if (rc) {
            camblas_pthread_pool_destroy(p);
            return NULL;
        }
        ++p->count;
    }
    p->executor = (camblas_executor_t){run, p};
    return p;
fail:
    if (stage >= 3)
        pthread_cond_destroy(&p->work);
    if (stage >= 2)
        pthread_mutex_destroy(&p->state);
    if (stage >= 1)
        pthread_mutex_destroy(&p->submit);
    free(p->threads);
    free(p);
    return NULL;
}

const camblas_executor_t *camblas_pthread_pool_executor(camblas_pthread_pool_t *p)
{
    return p ? &p->executor : NULL;
}
