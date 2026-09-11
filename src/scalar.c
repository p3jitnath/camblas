/*
 * CAMBLAS scalar reference GEMM implementation.
 *
 * Correct, unoptimized triple-loop GEMM for SGEMM and DGEMM. Handles all
 * transpose combinations (N/T), general alpha and beta, and arbitrary leading
 * dimensions. This is the correctness reference, not a performance baseline.
 *
 * EXECUTION MODEL:
 *   Scalar execution partitions output C into DISJOINT row-tile tasks and
 *   dispatches them through the ctx's executor. The correctness-gated packed
 *   branch uses a hidden 2-D task grid with bounded packing phases. The
 *   default serial executor runs in the calling thread; caller-owned
 *   executors may run disjoint tasks concurrently. The Grace build supplies
 *   the tuned microkernels through the same dispatch boundary.
 *
 * Column-major storage: element (i, j) of matrix X is at X[i + j * ldx].
 */
#include "camblas.h"
#include "camblas_workspace.h"
#include "camblas_planner.h"
#include "camblas_executor.h"
#include "kernels.h"
#include "packed.h"
#include "gemm_bounds.h"
#include <string.h>

/* Direct source-only correctness tests intentionally do not need to link the
 * architecture-specific object.  The shared library and the SVE benchmark
 * provide the strong hidden definition; otherwise this weak reference is
 * NULL and execution remains on the scalar reference. */
extern int camblas_sgemm_sve(char trans_a, char trans_b, int m, int n, int k, float alpha,
                             const float *A, int lda, const float *B, int ldb, float beta, float *C,
                             int ldc) CAMBLAS_INTERNAL CAMBLAS_WEAK;
extern int camblas_dgemm_sve(char trans_a, char trans_b, int m, int n, int k, double alpha,
                             const double *A, int lda, const double *B, int ldb, double beta,
                             double *C, int ldc) CAMBLAS_INTERNAL CAMBLAS_WEAK;

extern int camblas_sgemm_packed(char trans_a, char trans_b, int m, int n, int k, float alpha,
                                const float *A, int lda, const float *B, int ldb, float beta,
                                float *C, int ldc, int mc, int nc, int kc, int mr,
                                int nr) CAMBLAS_PACKED_INTERNAL CAMBLAS_PACKED_WEAK;
extern int camblas_dgemm_packed(char trans_a, char trans_b, int m, int n, int k, double alpha,
                                const double *A, int lda, const double *B, int ldb, double beta,
                                double *C, int ldc, int mc, int nc, int kc, int mr,
                                int nr) CAMBLAS_PACKED_INTERNAL CAMBLAS_PACKED_WEAK;
extern int camblas_sgemm_packed_executor(
    char trans_a, char trans_b, int m, int n, int k, float alpha, const float *A, int lda,
    const float *B, int ldb, float beta, float *C, int ldc, int mc, int nc, int kc, int mr, int nr,
    const camblas_executor_t *executor) CAMBLAS_PACKED_INTERNAL CAMBLAS_PACKED_WEAK;
extern int camblas_dgemm_packed_executor(
    char trans_a, char trans_b, int m, int n, int k, double alpha, const double *A, int lda,
    const double *B, int ldb, double beta, double *C, int ldc, int mc, int nc, int kc, int mr,
    int nr, const camblas_executor_t *executor) CAMBLAS_PACKED_INTERNAL CAMBLAS_PACKED_WEAK;

extern int camblas_sgemm_packed_executor_workspace(char, char, int, int, int, float, const float *,
                                                   int, const float *, int, float, float *, int,
                                                   int, int, int, int, int,
                                                   const camblas_executor_t *, void *, size_t)
CAMBLAS_PACKED_INTERNAL CAMBLAS_PACKED_WEAK;
extern int camblas_dgemm_packed_executor_workspace(char, char, int, int, int, double,
                                                   const double *, int, const double *, int, double,
                                                   double *, int, int, int, int, int, int,
                                                   const camblas_executor_t *, void *, size_t)
CAMBLAS_PACKED_INTERNAL CAMBLAS_PACKED_WEAK;

const camblas_ctx_t camblas_ctx_default = {
    .num_threads = 1,
    .reproducibility = CAMBLAS_REPRO_FAST,
    .planner_mode = "none",
    .topo = NULL,
    .executor = NULL, /* NULL = default serial executor */
};

uint32_t camblas_abi_version(void)
{
    return CAMBLAS_ABI_VERSION;
}

/* Validate arguments for a GEMM call. Returns 0 if valid, -1 otherwise. */
static int validate_args(char trans_a, char trans_b, int m, int n, int k, int lda, int ldb, int ldc,
                         const void *A, const void *B, void *C)
{
    if (trans_a != 'N' && trans_a != 'T')
        return -1;
    if (trans_b != 'N' && trans_b != 'T')
        return -1;
    if (m < 0 || n < 0 || k < 0)
        return -1;
    /* No output element is read or written when m or n is zero.  Preserve
       the established no-op contract: all leading dimensions must be
       positive, but unused operands need not satisfy a larger shape minimum. */
    if (m == 0 || n == 0) {
        if (lda < 1 || ldb < 1 || ldc < 1)
            return -1;
        if (A == NULL || B == NULL || C == NULL)
            return -1;
        return 0;
    }
    /* BLAS leading dimensions must cover the physical row count of each
       operand.  Keep a minimum of one for zero-dimensional calls, matching
       the existing pointer/ld contract and avoiding zero-stride ambiguity. */
    int min_lda = (trans_a == 'N') ? m : k;
    int min_ldb = (trans_b == 'N') ? k : n;
    int min_ldc = m;
    if (min_lda < 1)
        min_lda = 1;
    if (min_ldb < 1)
        min_ldb = 1;
    if (min_ldc < 1)
        min_ldc = 1;
    if (lda < min_lda || ldb < min_ldb || ldc < min_ldc)
        return -1;
    if (A == NULL || B == NULL || C == NULL)
        return -1;
    return 0;
}

/*
 * Maximum number of tile tasks a single GEMM call will produce. Bounds the
 * stack-allocated task array. For m larger than this, the per-task row count
 * grows so n_tasks stays <= this cap. 256 tasks * 16 bytes/task = 4 KiB stack.
 */
#define CAMBLAS_EXEC_MAX_TASKS 256

/*
 * Partition the M (row) dimension into at most max_tasks tiles, each covering
 * a full column range [0, n). Tasks have disjoint, contiguous row ranges that
 * together cover [0, m). Returns the number of tasks (0 if m <= 0 or n == 0).
 */
static int split_rows(int m, int n, int max_tasks, camblas_task_t *tasks)
{
    if (m < 0 || n < 0 || max_tasks < 1)
        return -1;
    if (m == 0 || n == 0)
        return 0;
    if (!tasks)
        return -1;

    int tile = 1;
    int ntasks = m;
    if (ntasks > max_tasks) {
        /* Avoid m + divisor - 1: public dimensions are signed int and may
           be close to INT_MAX.  Both quotients are positive, so quotient
           plus the remainder bit is an overflow-free ceil division. */
        tile = m / max_tasks + (m % max_tasks != 0);
        ntasks = m / tile + (m % tile != 0);
    }
    for (int t = 0; t < ntasks; t++) {
        int i0 = t * tile;
        int remaining = m - i0;
        /* Compute the end from the non-negative remaining range so the
           final i0 + tile addition is performed only when it is <= m. */
        int i1 = remaining < tile ? m : i0 + tile;
        tasks[t].i0 = i0;
        tasks[t].i1 = i1;
        tasks[t].j0 = 0;
        tasks[t].j1 = n;
    }
    return ntasks;
}

/*
 * Resolve the effective executor: ctx->executor, or the default serial
 * executor if NULL. Returns NULL if the executor's run callback is NULL
 * (invalid). The caller treats NULL as an error, including before a valid
 * zero-output no-op is allowed to return.
 */
static const camblas_executor_t *resolve_executor(const camblas_ctx_t *ctx)
{
    const camblas_executor_t *ex =
        (ctx && ctx->executor) ? ctx->executor : &camblas_executor_serial;
    return ex->run ? ex : NULL;
}

/* The exported serial executor is the explicit spelling of the NULL default.
 * Keep the packed dispatch choice aligned with that executor contract: only a
 * caller-owned non-serial executor takes the private task-grid entry point. */
static int uses_default_serial_executor(const camblas_ctx_t *ctx)
{
    return !ctx || ctx->executor == NULL || ctx->executor == &camblas_executor_serial;
}

/* The packed boundary has one and only one fallback status.  A zero status
 * completed the operation; CAMBLAS_PACKED_UNAVAILABLE guarantees that C was
 * untouched; every other status is an execution failure and must not be
 * followed by a second scalar GEMM over a potentially partial C. */
static int packed_result_is_execution_error(int packed_rc)
{
    return packed_rc != 0 && packed_rc != CAMBLAS_PACKED_UNAVAILABLE;
}

/* The hidden no-pack SVE backend has one and only one retryable status.  A
 * different nonzero status means that execution may have started, so a scalar
 * retry could apply the GEMM a second time over a partial C. */
static int sve_result_is_execution_error(int sve_rc)
{
    return sve_rc != 0 && sve_rc != CAMBLAS_SVE_UNAVAILABLE;
}

/* ---- SGEMM work descriptor + tile function ---- */

typedef struct {
    char trans_a, trans_b;
    int m, n, k, lda, ldb, ldc;
    float alpha, beta;
    const float *A, *B;
    float *C;
} sgemm_work_t;

/* op(A)[i][l]: element (i, l) of the m×k matrix op(A). */
#define ELEM_A_N(w, i, l) ((w)->A[(size_t)(i) + (size_t)(l) * (size_t)((w)->lda)]) /* trans_a='N' */
#define ELEM_A_T(w, i, l) ((w)->A[(size_t)(l) + (size_t)(i) * (size_t)((w)->lda)]) /* trans_a='T' */
/* op(B)[l][j]: element (l, j) of the k×n matrix op(B). */
#define ELEM_B_N(w, l, j) ((w)->B[(size_t)(l) + (size_t)(j) * (size_t)((w)->ldb)]) /* trans_b='N' */
#define ELEM_B_T(w, l, j) ((w)->B[(size_t)(j) + (size_t)(l) * (size_t)((w)->ldb)]) /* trans_b='T' */

/*
 * Tile compute: beta-scale then alpha*acc for C rows [i0,i1), cols [j0,j1).
 * Bitwise-identical to the prior monolithic loop for the same (i,j): each
 * element is beta-scaled once, then receives alpha * sum_l(a*b) with the same
 * accumulation order (l = 0..k-1). Only the loop partitioning differs.
 */
static void sgemm_tile_fn(const camblas_task_t *task, void *gctx)
{
    sgemm_work_t *w = (sgemm_work_t *)gctx;
    size_t i0 = (size_t)task->i0, i1 = (size_t)task->i1;
    size_t j0 = (size_t)task->j0, j1 = (size_t)task->j1;
    size_t ldc = (size_t)w->ldc;
    float beta = w->beta;

    if (beta == 0.0f) {
        for (size_t j = j0; j < j1; j++)
            for (size_t i = i0; i < i1; i++)
                w->C[i + j * ldc] = 0.0f;
    } else if (beta != 1.0f) {
        for (size_t j = j0; j < j1; j++)
            for (size_t i = i0; i < i1; i++)
                w->C[i + j * ldc] *= beta;
    }

    if (w->alpha == 0.0f || w->k == 0)
        return;

    for (size_t j = j0; j < j1; j++) {
        for (size_t i = i0; i < i1; i++) {
            float acc = 0.0f;
            for (size_t l = 0; l < (size_t)w->k; l++) {
                float a = (w->trans_a == 'N') ? ELEM_A_N(w, i, l) : ELEM_A_T(w, i, l);
                float b = (w->trans_b == 'N') ? ELEM_B_N(w, l, j) : ELEM_B_T(w, l, j);
                acc += a * b;
            }
            w->C[i + j * ldc] += w->alpha * acc;
        }
    }
}

/* ---- DGEMM work descriptor + tile function ---- */

typedef struct {
    char trans_a, trans_b;
    int m, n, k, lda, ldb, ldc;
    double alpha, beta;
    const double *A, *B;
    double *C;
} dgemm_work_t;

static void dgemm_tile_fn(const camblas_task_t *task, void *gctx)
{
    dgemm_work_t *w = (dgemm_work_t *)gctx;
    size_t i0 = (size_t)task->i0, i1 = (size_t)task->i1;
    size_t j0 = (size_t)task->j0, j1 = (size_t)task->j1;
    size_t lda = (size_t)w->lda;
    size_t ldb = (size_t)w->ldb;
    size_t ldc = (size_t)w->ldc;
    double beta = w->beta;

    if (beta == 0.0) {
        for (size_t j = j0; j < j1; j++)
            for (size_t i = i0; i < i1; i++)
                w->C[i + j * ldc] = 0.0;
    } else if (beta != 1.0) {
        for (size_t j = j0; j < j1; j++)
            for (size_t i = i0; i < i1; i++)
                w->C[i + j * ldc] *= beta;
    }

    if (w->alpha == 0.0 || w->k == 0)
        return;

    for (size_t j = j0; j < j1; j++) {
        for (size_t i = i0; i < i1; i++) {
            double acc = 0.0;
            for (size_t l = 0; l < (size_t)w->k; l++) {
                double a = (w->trans_a == 'N') ? w->A[i + l * lda] : w->A[l + i * lda];
                double b = (w->trans_b == 'N') ? w->B[l + j * ldb] : w->B[j + l * ldb];
                acc += a * b;
            }
            w->C[i + j * ldc] += w->alpha * acc;
        }
    }
}

int camblas_workspace_bytes(int n, int k, int dtype, size_t *bytes)
{
    if (!bytes || n < 0 || k < 0 || (dtype != CAMBLAS_DTYPE_F32 && dtype != CAMBLAS_DTYPE_F64))
        return -1;
    size_t np = ((size_t)n + 7) / 8 * 8;
    size_t size = dtype == CAMBLAS_DTYPE_F64 ? sizeof(double) : sizeof(float);
    if (np && (size_t)k > (size_t)PTRDIFF_MAX / np / size)
        return -1;
    *bytes = np * (size_t)k * size;
    return 0;
}

static int workspace_disjoint(void *workspace, size_t bytes, const void *matrix, int rows, int cols,
                              int ld, size_t size)
{
    if (!workspace || !bytes || !rows || !cols)
        return 1;
    size_t span = ((size_t)(cols - 1) * (size_t)ld + (size_t)rows) * size;
    uintptr_t w = (uintptr_t)workspace, p = (uintptr_t)matrix;
    return p <= w ? w - p >= span : p - w >= bytes;
}

static int workspace_valid(void *workspace, size_t bytes, char ta, char tb, int m, int n, int k,
                           int lda, int ldb, int ldc, int product, int output, size_t size,
                           const void *a, const void *b, const void *c)
{
    if (!workspace)
        return bytes == 0;
    if ((uintptr_t)workspace % _Alignof(double) || bytes > PTRDIFF_MAX ||
        bytes > UINTPTR_MAX - (uintptr_t)workspace)
        return 0;
    if (!m || !n)
        return 1;
    if (product &&
        (!workspace_disjoint(workspace, bytes, a, ta == 'N' ? m : k, ta == 'N' ? k : m, lda,
                             size) ||
         !workspace_disjoint(workspace, bytes, b, tb == 'N' ? k : n, tb == 'N' ? n : k, ldb, size)))
        return 0;
    return !output || workspace_disjoint(workspace, bytes, c, m, n, ldc, size);
}

/* ---- Plan-aware GEMM (observable decision interface + executor dispatch) ---- */

static int sgemm_plan_impl(const camblas_ctx_t *ctx, char trans_a, char trans_b, int m, int n,
                           int k, float alpha, const float *A, int lda, const float *B, int ldb,
                           float beta, float *C, int ldc, camblas_plan_t *plan_out, void *workspace,
                           size_t workspace_bytes)
{
    if (validate_args(trans_a, trans_b, m, n, k, lda, ldb, ldc, A, B, C) ||
        camblas_gemm_access_spans_fit(
            trans_a, trans_b, m, n, k, lda, ldb, ldc, alpha != 0.0f && k > 0,
            camblas_gemm_output_access_needed(k, alpha != 0.0f, beta != 1.0f), sizeof(float)) != 0)
        return -1;
    if (!workspace_valid(workspace, workspace_bytes, trans_a, trans_b, m, n, k, lda, ldb, ldc,
                         alpha != 0 && k > 0,
                         camblas_gemm_output_access_needed(k, alpha != 0, beta != 1), sizeof(float),
                         A, B, C))
        return -1;

    /* Call the planner to validate the operation and produce a plan. */
    camblas_op_t op = {
        .m = m,
        .n = n,
        .k = k,
        .trans_a = trans_a,
        .trans_b = trans_b,
        .dtype = CAMBLAS_DTYPE_F32,
        .batch_count = 1,
    };
    camblas_plan_t plan;
    const camblas_ctx_t *eff_ctx = ctx ? ctx : &camblas_ctx_default;
    if (camblas_plan_make(&op, eff_ctx->topo, eff_ctx, &plan) != 0)
        return -1;

    /* Resolve the executor (fail-closed on an invalid executor). This check
       intentionally precedes the no-output no-op below: a malformed
       caller-owned context is still an invalid call even when no task would
       be dispatched. */
    const camblas_executor_t *ex = resolve_executor(eff_ctx);
    if (!ex)
        return -1;

    /* A zero-alpha or zero-K product with beta=1 has no input or output
       access. Keep the public boundary independent of any optimized backend's
       implementation and preserve the already-made plan for observability. */
    if ((alpha == 0.0f || k == 0) && beta == 1.0f) {
        if (plan_out)
            *plan_out = plan;
        return 0;
    }

    /* The explicit SVE mode is single-core and no-pack.  Capability and
       vector-length checks happen inside the hidden backend before any write;
       only UNAVAILABLE means that the scalar reference may execute instead.
       Any other status is a post-dispatch failure and must not be retried. */
    if (plan.kernel_id == CAMBLAS_KERNEL_SVE && camblas_sgemm_sve) {
        int sve_rc =
            camblas_sgemm_sve(trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        if (sve_rc == 0) {
            if (plan_out)
                *plan_out = plan;
            return 0;
        }
        if (sve_result_is_execution_error(sve_rc))
            return -1;
    }

    if (plan.kernel_id == CAMBLAS_KERNEL_PACKED) {
        int packed_rc = CAMBLAS_PACKED_UNAVAILABLE;
        if (workspace) {
            if (!camblas_sgemm_packed_executor_workspace)
                return -1;
            packed_rc = camblas_sgemm_packed_executor_workspace(
                trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, plan.mc, plan.nc,
                plan.kc, plan.mr, plan.nr, ex, workspace, workspace_bytes);
            if (packed_rc != 0)
                return -1;
        } else if (uses_default_serial_executor(eff_ctx) && camblas_sgemm_packed) {
            packed_rc = camblas_sgemm_packed(trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta,
                                             C, ldc, plan.mc, plan.nc, plan.kc, plan.mr, plan.nr);
        } else if (!uses_default_serial_executor(eff_ctx) && camblas_sgemm_packed_executor) {
            packed_rc = camblas_sgemm_packed_executor(trans_a, trans_b, m, n, k, alpha, A, lda, B,
                                                      ldb, beta, C, ldc, plan.mc, plan.nc, plan.kc,
                                                      plan.mr, plan.nr, ex);
        }
        if (packed_rc == 0) {
            if (plan_out)
                *plan_out = plan;
            return 0;
        }
        /* Only UNAVAILABLE permits the scalar fallback. Any later invariant
           failure is fail-closed and must not execute a second GEMM over a
           partial result. */
        if (packed_result_is_execution_error(packed_rc))
            return -1;
    }

    /* Partition C into row-tile tasks and dispatch through the executor. */
    sgemm_work_t w = {
        .trans_a = trans_a,
        .trans_b = trans_b,
        .m = m,
        .n = n,
        .k = k,
        .lda = lda,
        .ldb = ldb,
        .ldc = ldc,
        .alpha = alpha,
        .beta = beta,
        .A = A,
        .B = B,
        .C = C,
    };
    camblas_task_t tasks[CAMBLAS_EXEC_MAX_TASKS];
    int n_tasks = split_rows(m, n, CAMBLAS_EXEC_MAX_TASKS, tasks);
    if (n_tasks > 0) {
        if (ex->run(sgemm_tile_fn, tasks, n_tasks, &w, ex->user_data) != 0)
            return -1;
    }
    /* plan_out is written iff the call succeeds (contract: untouched on error). */
    if (plan_out)
        *plan_out = plan;
    return 0;
}

int camblas_sgemm_plan(const camblas_ctx_t *ctx, char trans_a, char trans_b, int m, int n, int k,
                       float alpha, const float *A, int lda, const float *B, int ldb, float beta,
                       float *C, int ldc, camblas_plan_t *plan_out)
{
    return sgemm_plan_impl(ctx, trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc,
                           plan_out, NULL, 0);
}

int camblas_sgemm_plan_workspace(const camblas_ctx_t *ctx, char trans_a, char trans_b, int m, int n,
                                 int k, float alpha, const float *A, int lda, const float *B,
                                 int ldb, float beta, float *C, int ldc, camblas_plan_t *plan_out,
                                 void *workspace, size_t workspace_bytes)
{
    return sgemm_plan_impl(ctx, trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc,
                           plan_out, workspace, workspace_bytes);
}

int camblas_sgemm(const camblas_ctx_t *ctx, char trans_a, char trans_b, int m, int n, int k,
                  float alpha, const float *A, int lda, const float *B, int ldb, float beta,
                  float *C, int ldc)
{
    return camblas_sgemm_plan(ctx, trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc,
                              NULL);
}

static int dgemm_plan_impl(const camblas_ctx_t *ctx, char trans_a, char trans_b, int m, int n,
                           int k, double alpha, const double *A, int lda, const double *B, int ldb,
                           double beta, double *C, int ldc, camblas_plan_t *plan_out,
                           void *workspace, size_t workspace_bytes)
{
    if (validate_args(trans_a, trans_b, m, n, k, lda, ldb, ldc, A, B, C) ||
        camblas_gemm_access_spans_fit(
            trans_a, trans_b, m, n, k, lda, ldb, ldc, alpha != 0.0 && k > 0,
            camblas_gemm_output_access_needed(k, alpha != 0.0, beta != 1.0), sizeof(double)) != 0)
        return -1;
    if (!workspace_valid(workspace, workspace_bytes, trans_a, trans_b, m, n, k, lda, ldb, ldc,
                         alpha != 0 && k > 0,
                         camblas_gemm_output_access_needed(k, alpha != 0, beta != 1),
                         sizeof(double), A, B, C))
        return -1;

    camblas_op_t op = {
        .m = m,
        .n = n,
        .k = k,
        .trans_a = trans_a,
        .trans_b = trans_b,
        .dtype = CAMBLAS_DTYPE_F64,
        .batch_count = 1,
    };
    camblas_plan_t plan;
    const camblas_ctx_t *eff_ctx = ctx ? ctx : &camblas_ctx_default;
    if (camblas_plan_make(&op, eff_ctx->topo, eff_ctx, &plan) != 0)
        return -1;

    const camblas_executor_t *ex = resolve_executor(eff_ctx);
    if (!ex)
        return -1;

    /* See the SGEMM path: a zero-alpha or zero-K product with beta=1 is a
       no-read/no-write call and must not reach an optimized backend or
       executor. The executor validity check above remains authoritative. */
    if ((alpha == 0.0 || k == 0) && beta == 1.0) {
        if (plan_out)
            *plan_out = plan;
        return 0;
    }

    if (plan.kernel_id == CAMBLAS_KERNEL_SVE && camblas_dgemm_sve) {
        int sve_rc =
            camblas_dgemm_sve(trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        if (sve_rc == 0) {
            if (plan_out)
                *plan_out = plan;
            return 0;
        }
        if (sve_result_is_execution_error(sve_rc))
            return -1;
    }

    if (plan.kernel_id == CAMBLAS_KERNEL_PACKED) {
        int packed_rc = CAMBLAS_PACKED_UNAVAILABLE;
        if (workspace) {
            if (!camblas_dgemm_packed_executor_workspace)
                return -1;
            packed_rc = camblas_dgemm_packed_executor_workspace(
                trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, plan.mc, plan.nc,
                plan.kc, plan.mr, plan.nr, ex, workspace, workspace_bytes);
            if (packed_rc != 0)
                return -1;
        } else if (uses_default_serial_executor(eff_ctx) && camblas_dgemm_packed) {
            packed_rc = camblas_dgemm_packed(trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta,
                                             C, ldc, plan.mc, plan.nc, plan.kc, plan.mr, plan.nr);
        } else if (!uses_default_serial_executor(eff_ctx) && camblas_dgemm_packed_executor) {
            packed_rc = camblas_dgemm_packed_executor(trans_a, trans_b, m, n, k, alpha, A, lda, B,
                                                      ldb, beta, C, ldc, plan.mc, plan.nc, plan.kc,
                                                      plan.mr, plan.nr, ex);
        }
        if (packed_rc == 0) {
            if (plan_out)
                *plan_out = plan;
            return 0;
        }
        if (packed_result_is_execution_error(packed_rc))
            return -1;
    }

    dgemm_work_t w = {
        .trans_a = trans_a,
        .trans_b = trans_b,
        .m = m,
        .n = n,
        .k = k,
        .lda = lda,
        .ldb = ldb,
        .ldc = ldc,
        .alpha = alpha,
        .beta = beta,
        .A = A,
        .B = B,
        .C = C,
    };
    camblas_task_t tasks[CAMBLAS_EXEC_MAX_TASKS];
    int n_tasks = split_rows(m, n, CAMBLAS_EXEC_MAX_TASKS, tasks);
    if (n_tasks > 0) {
        if (ex->run(dgemm_tile_fn, tasks, n_tasks, &w, ex->user_data) != 0)
            return -1;
    }
    /* plan_out is written iff the call succeeds (contract: untouched on error). */
    if (plan_out)
        *plan_out = plan;
    return 0;
}

int camblas_dgemm_plan(const camblas_ctx_t *ctx, char trans_a, char trans_b, int m, int n, int k,
                       double alpha, const double *A, int lda, const double *B, int ldb,
                       double beta, double *C, int ldc, camblas_plan_t *plan_out)
{
    return dgemm_plan_impl(ctx, trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc,
                           plan_out, NULL, 0);
}

int camblas_dgemm_plan_workspace(const camblas_ctx_t *ctx, char trans_a, char trans_b, int m, int n,
                                 int k, double alpha, const double *A, int lda, const double *B,
                                 int ldb, double beta, double *C, int ldc, camblas_plan_t *plan_out,
                                 void *workspace, size_t workspace_bytes)
{
    return dgemm_plan_impl(ctx, trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc,
                           plan_out, workspace, workspace_bytes);
}

int camblas_dgemm(const camblas_ctx_t *ctx, char trans_a, char trans_b, int m, int n, int k,
                  double alpha, const double *A, int lda, const double *B, int ldb, double beta,
                  double *C, int ldc)
{
    return camblas_dgemm_plan(ctx, trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc,
                              NULL);
}
