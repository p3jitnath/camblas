/* Conservative overflow preflight for the experimental Strassen route.
 * The ordinary GEMM path handles operands outside this range. The check is
 * recomputed for every call; no operand values or addresses are cached. */
#ifndef CAMBLAS_RECTANGULAR32_RANGE_H
#define CAMBLAS_RECTANGULAR32_RANGE_H

#include "rectangular32.h"
#include <arm_neon.h>
#include <float.h>
#include <math.h>

typedef struct {
    const float *a, *b;
    int m, n, k, lda, ldb, workers, tb;
    float a_max[64], b_max[64];
} rectangular32_range_work_t;

static float rectangular32_column_max(const float *matrix, int rows, int stride, int first,
                                      int last)
{
    float32x4_t maximum = vdupq_n_f32(0);
    float tail = 0;
    for (int column = first; column < last; ++column) {
        const float *values = matrix + (size_t)column * stride;
        int row = 0;
        for (; row + 4 <= rows; row += 4)
            maximum = vmaxq_f32(maximum, vabsq_f32(vld1q_f32(values + row)));
        for (; row < rows; ++row) {
            float value = fabs(values[row]);
            if (!isfinite(value))
                return INFINITY;
            if (value > tail)
                tail = value;
        }
    }
    float peak = vmaxvq_f32(maximum);
    return isfinite(peak) ? fmax(peak, tail) : INFINITY;
}

static void rectangular32_range_task(const camblas_task_t *task, void *opaque)
{
    rectangular32_range_work_t *work = opaque;
    int worker = task->i0, count = work->workers;
    work->a_max[worker] = rectangular32_column_max(
        work->a, work->m, work->lda, work->k * worker / count, work->k * (worker + 1) / count);
    work->b_max[worker] =
        rectangular32_column_max(work->b, work->tb ? work->n : work->k, work->ldb,
                                 (work->tb ? work->k : work->n) * worker / count,
                                 (work->tb ? work->k : work->n) * (worker + 1) / count);
}

/* Call only after the physical NN/NT matrix spans have been validated. Return one for
 * a conservative finite range, zero for classical fallback, or -1 on an
 * executor failure. This bounds overflow growth, not relative roundoff. */
static int rectangular32_range_safe_op(int tb, const camblas_executor_t *executor, int workers,
                                       int m, int n, int k, const float *a, int lda, const float *b,
                                       int ldb, int levels)
{
    if ((tb != 0 && tb != 1) || !executor || !executor->run || workers < 1 || workers > 64 ||
        (levels != 1 && levels != 2))
        return -1;
    rectangular32_range_work_t work = {.tb = tb,
                                       .a = a,
                                       .b = b,
                                       .m = m,
                                       .n = n,
                                       .k = k,
                                       .lda = lda,
                                       .ldb = ldb,
                                       .workers = workers};
    camblas_task_t tasks[64];
    for (int worker = 0; worker < workers; ++worker)
        tasks[worker] = (camblas_task_t){worker, worker + 1, 0, 1};
    if (executor->run(rectangular32_range_task, tasks, workers, &work, executor->user_data))
        return -1;
    float a_peak = 0, b_peak = 0;
    for (int worker = 0; worker < workers; ++worker) {
        a_peak = fmax(a_peak, work.a_max[worker]);
        b_peak = fmax(b_peak, work.b_max[worker]);
    }
    return rectangular32_range_verdict(a_peak, b_peak, k, levels);
}

/* Compatibility entry point for NN range-oracle checks. */
static inline int rectangular32_range_safe(const camblas_executor_t *executor, int workers, int m,
                                           int n, int k, const float *a, int lda, const float *b,
                                           int ldb, int levels)
{
    return rectangular32_range_safe_op(0, executor, workers, m, n, k, a, lda, b, ldb, levels);
}
#endif
