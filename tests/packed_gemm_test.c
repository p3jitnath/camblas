/* Correctness-only integration test for the hidden packed GEMM executor. */
#define _GNU_SOURCE 1

#include "camblas.h"
#include "camblas_planner.h"

#include <math.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>

static int failures;

#define CHECK(condition, message)                     \
    do {                                              \
        if (!(condition)) {                           \
            fprintf(stderr, "FAIL: %s\n", (message)); \
            failures++;                               \
        }                                             \
    } while (0)

typedef struct {
    int m, n;
    int calls, n_tasks, bad, next;
    size_t area;
    pthread_mutex_t lock;
    camblas_task_fn fn;
    const camblas_task_t *tasks;
    void *gctx;
} packed_parallel_state_t;

static void *packed_worker(void *arg)
{
    packed_parallel_state_t *state = (packed_parallel_state_t *)arg;
    for (;;) {
        int index;
        pthread_mutex_lock(&state->lock);
        index = state->next++;
        pthread_mutex_unlock(&state->lock);
        if (index >= state->n_tasks)
            return NULL;
        state->fn(&state->tasks[index], state->gctx);
    }
}

static int packed_parallel_executor(camblas_task_fn fn, const camblas_task_t *tasks, int n_tasks,
                                    void *gctx, void *user_data)
{
    packed_parallel_state_t *state = (packed_parallel_state_t *)user_data;
    pthread_t workers[4];
    int n_workers = 0;

    if (!state || !fn || n_tasks < 0 || (n_tasks > 0 && !tasks))
        return -1;
    state->calls++;
    state->n_tasks = n_tasks;
    state->area = 0;
    state->next = 0;
    state->fn = fn;
    state->tasks = tasks;
    state->gctx = gctx;
    for (int a = 0; a < n_tasks; a++) {
        const camblas_task_t *x = &tasks[a];
        if (x->i0 < 0 || x->i1 > state->m || x->i0 >= x->i1 || x->j0 < 0 || x->j1 > state->n ||
            x->j0 >= x->j1) {
            state->bad = 1;
            continue;
        }
        state->area += (size_t)(x->i1 - x->i0) * (size_t)(x->j1 - x->j0);
        for (int b = 0; b < a; b++) {
            const camblas_task_t *y = &tasks[b];
            if (x->i0 < y->i1 && y->i0 < x->i1 && x->j0 < y->j1 && y->j0 < x->j1)
                state->bad = 1;
        }
    }
    if (state->bad || n_tasks == 0)
        return state->bad ? -1 : 0;
    if (pthread_mutex_init(&state->lock, NULL) != 0)
        return -1;
    for (int t = 0; t < 4 && t < n_tasks; t++) {
        if (pthread_create(&workers[n_workers], NULL, packed_worker, state) != 0) {
            state->bad = 1;
            break;
        }
        n_workers++;
    }
    for (int t = 0; t < n_workers; t++)
        pthread_join(workers[t], NULL);
    pthread_mutex_destroy(&state->lock);
    return state->bad ? -1 : 0;
}

static int packed_failing_executor(camblas_task_fn fn, const camblas_task_t *tasks, int n_tasks,
                                   void *gctx, void *user_data)
{
    packed_parallel_state_t *state = (packed_parallel_state_t *)user_data;
    (void)fn;
    (void)tasks;
    (void)gctx;
    if (!state || n_tasks < 0)
        return -1;
    state->calls++;
    state->n_tasks = n_tasks;
    return -1;
}

/* Run exactly one packed task, then report executor failure.  This makes a
 * post-dispatch error observable: the public wrapper must not run the scalar
 * GEMM over the remaining rectangles after the packed path has touched C. */
static int packed_partial_failing_executor(camblas_task_fn fn, const camblas_task_t *tasks,
                                           int n_tasks, void *gctx, void *user_data)
{
    packed_parallel_state_t *state = (packed_parallel_state_t *)user_data;
    if (!state || !fn || n_tasks < 1 || !tasks)
        return -1;
    state->calls++;
    state->n_tasks = n_tasks;
    fn(&tasks[0], gctx);
    return -1;
}

static camblas_topology_t test_topology(void)
{
    camblas_topology_t topo;
    memset(&topo, 0, sizeof(topo));
    topo.n_allowed_cpus = 8;
    topo.sve_vl_bits = 128;
    for (int cpu = 0; cpu < topo.n_allowed_cpus; cpu++)
        camblas_cpuset_set(&topo.allowed_cpus, cpu);
    return topo;
}

static long double ref_value(char trans_a, char trans_b, int i, int j, int k, long double alpha,
                             const float *Af, const double *Ad, int lda, const float *Bf,
                             const double *Bd, int ldb, long double beta, const float *Cf,
                             const double *Cd, int ldc, int is_f64)
{
    long double acc = 0.0L;
    for (int l = 0; l < k; l++) {
        long double a = is_f64 ? ((trans_a == 'N') ? Ad[i + l * lda] : Ad[l + i * lda])
                               : ((trans_a == 'N') ? Af[i + l * lda] : Af[l + i * lda]);
        long double b = is_f64 ? ((trans_b == 'N') ? Bd[l + j * ldb] : Bd[j + l * ldb])
                               : ((trans_b == 'N') ? Bf[l + j * ldb] : Bf[j + l * ldb]);
        acc += a * b;
    }
    long double old = is_f64 ? Cd[i + j * ldc] : Cf[i + j * ldc];
    return alpha * acc + beta * old;
}

static void fill_f32(float *x, size_t count, float offset)
{
    for (size_t q = 0; q < count; q++)
        x[q] = (float)((int)(q % 17u) - 8) * 0.0625f + offset;
}

static void fill_f64(double *x, size_t count, double offset)
{
    for (size_t q = 0; q < count; q++)
        x[q] = (double)((int)(q % 17u) - 8) * 0.0625 + offset;
}

static int run_f32_general(char trans_a, char trans_b)
{
    const int m = 131, n = 137, k = 133;
    const int lda = (trans_a == 'N') ? m + 3 : k + 3;
    const int ldb = (trans_b == 'N') ? k + 5 : n + 5;
    const int ldc = m + 7;
    const size_t na = (size_t)lda * (size_t)((trans_a == 'N') ? k : m);
    const size_t nb = (size_t)ldb * (size_t)((trans_b == 'N') ? n : k);
    const size_t nc = (size_t)ldc * (size_t)n;
    float *A = malloc(na * sizeof(*A)), *B = malloc(nb * sizeof(*B));
    float *C = malloc(nc * sizeof(*C)), *before = malloc(nc * sizeof(*before));
    if (!A || !B || !C || !before) {
        free(A);
        free(B);
        free(C);
        free(before);
        return -1;
    }
    fill_f32(A, na, 0.125f);
    fill_f32(B, nb, -0.25f);
    fill_f32(C, nc, 0.375f);
    memcpy(before, C, nc * sizeof(*C));

    camblas_topology_t topo = test_topology();
    camblas_ctx_t ctx = {
        .num_threads = 1,
        .reproducibility = CAMBLAS_REPRO_FAST,
        .planner_mode = "shape-aware",
        .topo = &topo,
        .executor = NULL,
    };
    camblas_plan_t plan = {0};
    int rc = camblas_sgemm_plan(&ctx, trans_a, trans_b, m, n, k, 1.25f, A, lda, B, ldb, -0.5f, C,
                                ldc, &plan);
    int ok = (rc == 0 && plan.kernel_id == CAMBLAS_KERNEL_PACKED);
    float max_err = 0.0f;
    for (int j = 0; ok && j < n; j++) {
        for (int i = 0; i < m; i++) {
            long double expected = ref_value(trans_a, trans_b, i, j, k, 1.25L, A, NULL, lda, B,
                                             NULL, ldb, -0.5L, before, NULL, ldc, 0);
            float err = fabsf(C[i + j * ldc] - (float)expected);
            if (!isfinite(C[i + j * ldc]) || err > max_err)
                max_err = err;
        }
    }
    for (int j = 0; ok && j < n; j++) {
        for (int i = m; i < ldc; i++)
            if (C[i + j * ldc] != before[i + j * ldc])
                ok = 0;
    }
    CHECK(ok && max_err <= 3.0e-3f, "fp32 packed blocked general case");
    if (!(ok && max_err <= 3.0e-3f))
        fprintf(stderr, "  fp32 %c%c rc=%d kernel=%d max_err=%g\n", trans_a, trans_b, rc,
                plan.kernel_id, max_err);
    free(A);
    free(B);
    free(C);
    free(before);
    return ok ? 0 : -1;
}

static int run_f64_general(char trans_a, char trans_b)
{
    const int m = 131, n = 137, k = 133;
    const int lda = (trans_a == 'N') ? m + 3 : k + 3;
    const int ldb = (trans_b == 'N') ? k + 5 : n + 5;
    const int ldc = m + 7;
    const size_t na = (size_t)lda * (size_t)((trans_a == 'N') ? k : m);
    const size_t nb = (size_t)ldb * (size_t)((trans_b == 'N') ? n : k);
    const size_t nc = (size_t)ldc * (size_t)n;
    double *A = malloc(na * sizeof(*A)), *B = malloc(nb * sizeof(*B));
    double *C = malloc(nc * sizeof(*C)), *before = malloc(nc * sizeof(*before));
    if (!A || !B || !C || !before) {
        free(A);
        free(B);
        free(C);
        free(before);
        return -1;
    }
    fill_f64(A, na, 0.125);
    fill_f64(B, nb, -0.25);
    fill_f64(C, nc, 0.375);
    memcpy(before, C, nc * sizeof(*C));

    camblas_topology_t topo = test_topology();
    camblas_ctx_t ctx = {
        .num_threads = 1,
        .reproducibility = CAMBLAS_REPRO_FAST,
        .planner_mode = "shape-aware",
        .topo = &topo,
        .executor = NULL,
    };
    camblas_plan_t plan = {0};
    int rc = camblas_dgemm_plan(&ctx, trans_a, trans_b, m, n, k, 1.25, A, lda, B, ldb, -0.5, C, ldc,
                                &plan);
    int ok = (rc == 0 && plan.kernel_id == CAMBLAS_KERNEL_PACKED);
    double max_err = 0.0;
    for (int j = 0; ok && j < n; j++) {
        for (int i = 0; i < m; i++) {
            long double expected = ref_value(trans_a, trans_b, i, j, k, 1.25L, NULL, A, lda, NULL,
                                             B, ldb, -0.5L, NULL, before, ldc, 1);
            double err = fabs(C[i + j * ldc] - (double)expected);
            if (!isfinite(C[i + j * ldc]) || err > max_err)
                max_err = err;
        }
    }
    for (int j = 0; ok && j < n; j++) {
        for (int i = m; i < ldc; i++)
            if (C[i + j * ldc] != before[i + j * ldc])
                ok = 0;
    }
    CHECK(ok && max_err <= 2.0e-11, "fp64 packed blocked general case");
    if (!(ok && max_err <= 2.0e-11))
        fprintf(stderr, "  fp64 %c%c rc=%d kernel=%d max_err=%g\n", trans_a, trans_b, rc,
                plan.kernel_id, max_err);
    free(A);
    free(B);
    free(C);
    free(before);
    return ok ? 0 : -1;
}

static int run_beta_zero(void)
{
    const int m = 129, n = 130, k = 131, ldc = 133;
    const size_t na = (size_t)m * k, nb = (size_t)k * n, nc = (size_t)ldc * n;
    float *A = malloc(na * sizeof(*A)), *B = malloc(nb * sizeof(*B));
    float *C = malloc(nc * sizeof(*C));
    if (!A || !B || !C) {
        free(A);
        free(B);
        free(C);
        return -1;
    }
    fill_f32(A, na, 0.0f);
    fill_f32(B, nb, 0.25f);
    for (size_t q = 0; q < nc; q++)
        C[q] = NAN;
    camblas_topology_t topo = test_topology();
    camblas_ctx_t ctx = {1, CAMBLAS_REPRO_FAST, "shape-aware", &topo, NULL};
    camblas_plan_t plan = {0};
    int rc = camblas_sgemm_plan(&ctx, 'N', 'N', m, n, k, 1.0f, A, m, B, k, 0.0f, C, ldc, &plan);
    int ok = rc == 0 && plan.kernel_id == CAMBLAS_KERNEL_PACKED;
    for (int j = 0; ok && j < n; j++)
        for (int i = 0; i < m; i++)
            if (!isfinite(C[i + j * ldc]))
                ok = 0;
    CHECK(ok, "beta=0 packed path does not read old C");
    free(A);
    free(B);
    free(C);
    return ok ? 0 : -1;
}

static int run_alpha_zero_no_read(void)
{
    const int m = 129, n = 129, k = 129, ldc = 132;
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size < 1)
        return -1;
    size_t page = (size_t)page_size;
    void *a_page = mmap(NULL, page, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    void *b_page = mmap(NULL, page, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    float *Cf = calloc((size_t)ldc * n, sizeof(*Cf));
    double *Cd = calloc((size_t)ldc * n, sizeof(*Cd));
    if (a_page == MAP_FAILED || b_page == MAP_FAILED || !Cf || !Cd) {
        if (a_page != MAP_FAILED)
            munmap(a_page, page);
        if (b_page != MAP_FAILED)
            munmap(b_page, page);
        free(Cf);
        free(Cd);
        return -1;
    }
    camblas_topology_t topo = test_topology();
    camblas_ctx_t ctx = {1, CAMBLAS_REPRO_FAST, "shape-aware", &topo, NULL};
    camblas_plan_t plan_f = {0}, plan_d = {0};
    int rcf = camblas_sgemm_plan(&ctx, 'N', 'N', m, n, k, 0.0f, a_page, m, b_page, k, 0.0f, Cf, ldc,
                                 &plan_f);
    int rcd = camblas_dgemm_plan(&ctx, 'N', 'N', m, n, k, 0.0, a_page, m, b_page, k, 0.0, Cd, ldc,
                                 &plan_d);
    int ok = rcf == 0 && rcd == 0 && plan_f.kernel_id == CAMBLAS_KERNEL_PACKED &&
             plan_d.kernel_id == CAMBLAS_KERNEL_PACKED;
    for (int j = 0; ok && j < n; j++)
        for (int i = 0; i < m; i++)
            if (Cf[i + j * ldc] != 0.0f || Cd[i + j * ldc] != 0.0)
                ok = 0;
    CHECK(ok, "alpha=0 packed path does not read A or B");
    munmap(a_page, page);
    munmap(b_page, page);
    free(Cf);
    free(Cd);
    return ok ? 0 : -1;
}

static int run_f32_custom_executor(void)
{
    const int m = 257, n = 513, k = 129;
    const int lda = m + 3, ldb = n + 5, ldc = m + 7;
    const size_t na = (size_t)lda * k, nb = (size_t)ldb * k;
    const size_t nc = (size_t)ldc * n;
    float *A = malloc(na * sizeof(*A)), *B = malloc(nb * sizeof(*B));
    float *C = malloc(nc * sizeof(*C)), *before = malloc(nc * sizeof(*before));
    packed_parallel_state_t state = {.m = m, .n = n};
    camblas_executor_t executor = {
        .run = packed_parallel_executor,
        .user_data = &state,
    };
    if (!A || !B || !C || !before) {
        free(A);
        free(B);
        free(C);
        free(before);
        return -1;
    }
    fill_f32(A, na, 0.125f);
    fill_f32(B, nb, -0.25f);
    fill_f32(C, nc, 0.375f);
    memcpy(before, C, nc * sizeof(*before));
    camblas_topology_t topo = test_topology();
    camblas_ctx_t ctx = {4, CAMBLAS_REPRO_FAST, "shape-aware", &topo, &executor};
    camblas_plan_t plan = {0};
    int rc =
        camblas_sgemm_plan(&ctx, 'N', 'T', m, n, k, 1.25f, A, lda, B, ldb, -0.5f, C, ldc, &plan);
    int ok = rc == 0 && plan.kernel_id == CAMBLAS_KERNEL_PACKED &&
             plan.parallel_scheme == CAMBLAS_PARALLEL_2D && state.calls == 1 &&
             state.n_tasks == 9 && !state.bad && state.area == (size_t)m * (size_t)n;
    float max_err = 0.0f;
    for (int j = 0; ok && j < n; j++) {
        for (int i = 0; i < m; i++) {
            long double expected = ref_value('N', 'T', i, j, k, 1.25L, A, NULL, lda, B, NULL, ldb,
                                             -0.5L, before, NULL, ldc, 0);
            float err = fabsf(C[i + j * ldc] - (float)expected);
            if (!isfinite(C[i + j * ldc]) || err > max_err)
                max_err = err;
        }
    }
    for (int j = 0; ok && j < n; j++)
        for (int i = m; i < ldc; i++)
            if (C[i + j * ldc] != before[i + j * ldc])
                ok = 0;
    CHECK(ok && max_err <= 3.0e-3f, "fp32 packed caller-owned executor task grid");
    if (!(ok && max_err <= 3.0e-3f))
        fprintf(stderr, "  custom fp32 rc=%d kernel=%d tasks=%d area=%zu max_err=%g\n", rc,
                plan.kernel_id, state.n_tasks, state.area, max_err);
    free(A);
    free(B);
    free(C);
    free(before);
    return ok ? 0 : -1;
}

static int run_f64_custom_executor(void)
{
    const int m = 257, n = 513, k = 129;
    const int lda = k + 3, ldb = k + 5, ldc = m + 7;
    const size_t na = (size_t)lda * m, nb = (size_t)ldb * n;
    const size_t nc = (size_t)ldc * n;
    double *A = malloc(na * sizeof(*A)), *B = malloc(nb * sizeof(*B));
    double *C = malloc(nc * sizeof(*C)), *before = malloc(nc * sizeof(*before));
    packed_parallel_state_t state = {.m = m, .n = n};
    camblas_executor_t executor = {
        .run = packed_parallel_executor,
        .user_data = &state,
    };
    if (!A || !B || !C || !before) {
        free(A);
        free(B);
        free(C);
        free(before);
        return -1;
    }
    fill_f64(A, na, 0.125);
    fill_f64(B, nb, -0.25);
    fill_f64(C, nc, 0.375);
    memcpy(before, C, nc * sizeof(*before));
    camblas_topology_t topo = test_topology();
    camblas_ctx_t ctx = {4, CAMBLAS_REPRO_FAST, "shape-aware", &topo, &executor};
    camblas_plan_t plan = {0};
    int rc = camblas_dgemm_plan(&ctx, 'T', 'N', m, n, k, 1.25, A, lda, B, ldb, -0.5, C, ldc, &plan);
    int ok = rc == 0 && plan.kernel_id == CAMBLAS_KERNEL_PACKED &&
             plan.parallel_scheme == CAMBLAS_PARALLEL_2D && state.calls == 1 &&
             state.n_tasks == 9 && !state.bad && state.area == (size_t)m * (size_t)n;
    double max_err = 0.0;
    for (int j = 0; ok && j < n; j++) {
        for (int i = 0; i < m; i++) {
            long double expected = ref_value('T', 'N', i, j, k, 1.25L, NULL, A, lda, NULL, B, ldb,
                                             -0.5L, NULL, before, ldc, 1);
            double err = fabs(C[i + j * ldc] - (double)expected);
            if (!isfinite(C[i + j * ldc]) || err > max_err)
                max_err = err;
        }
    }
    for (int j = 0; ok && j < n; j++)
        for (int i = m; i < ldc; i++)
            if (C[i + j * ldc] != before[i + j * ldc])
                ok = 0;
    CHECK(ok && max_err <= 2.0e-11, "fp64 packed caller-owned executor task grid");
    if (!(ok && max_err <= 2.0e-11))
        fprintf(stderr, "  custom fp64 rc=%d kernel=%d tasks=%d area=%zu max_err=%g\n", rc,
                plan.kernel_id, state.n_tasks, state.area, max_err);
    free(A);
    free(B);
    free(C);
    free(before);
    return ok ? 0 : -1;
}

static int run_custom_executor_failure(void)
{
    const int m = 257, n = 513, k = 129;
    const int lda = m, ldb = k, ldc = m;
    const size_t na = (size_t)lda * k, nb = (size_t)ldb * n;
    const size_t nc = (size_t)ldc * n;
    float *A = calloc(na, sizeof(*A)), *B = calloc(nb, sizeof(*B));
    float *C = calloc(nc, sizeof(*C)), *before = calloc(nc, sizeof(*before));
    packed_parallel_state_t state = {.m = m, .n = n};
    camblas_executor_t executor = {
        .run = packed_partial_failing_executor,
        .user_data = &state,
    };
    camblas_plan_t plan, sentinel;
    if (!A || !B || !C || !before) {
        free(A);
        free(B);
        free(C);
        free(before);
        return -1;
    }
    for (size_t q = 0; q < na; q++)
        A[q] = 1.0f;
    for (size_t q = 0; q < nb; q++)
        B[q] = 1.0f;
    memset(&sentinel, 0xA7, sizeof(sentinel));
    plan = sentinel;
    camblas_topology_t topo = test_topology();
    camblas_ctx_t ctx = {4, CAMBLAS_REPRO_FAST, "shape-aware", &topo, &executor};
    int rc = camblas_sgemm_plan(&ctx, 'N', 'N', m, n, k, 1.0f, A, lda, B, ldb, 1.0f, C, ldc, &plan);
    int ok = rc == -1 && state.calls == 1 && state.n_tasks == 9 && C[0] == (float)k &&
             C[256 * ldc] == 0.0f && memcmp(&plan, &sentinel, sizeof(plan)) == 0;
    CHECK(ok, "packed execution failure does not trigger scalar re-execution");
    free(A);
    free(B);
    free(C);
    free(before);
    return ok ? 0 : -1;
}

static int run_required_sve_hook_failure(void)
{
    const int m = 129, n = 129, k = 129;
    const size_t na = (size_t)m * (size_t)k;
    const size_t nb = (size_t)k * (size_t)n;
    const size_t nc = (size_t)m * (size_t)n;
    float *A = calloc(na, sizeof(*A));
    float *B = calloc(nb, sizeof(*B));
    float *C = calloc(nc, sizeof(*C));
    camblas_topology_t topo = test_topology();
    camblas_ctx_t ctx = {1, CAMBLAS_REPRO_FAST, "shape-aware", &topo, NULL};
    camblas_plan_t plan, sentinel;
    int configured, rc, ok;

    if (!A || !B || !C) {
        free(A);
        free(B);
        free(C);
        return -1;
    }
    memset(&sentinel, 0x5C, sizeof(sentinel));
    plan = sentinel;
    configured = setenv("CAMBLAS_PACKED_TRACE", "1", 1) == 0 &&
                 setenv("CAMBLAS_PACKED_REQUIRE_SVE_HOOK", "1", 1) == 0;
    rc = configured
             ? camblas_sgemm_plan(&ctx, 'N', 'N', m, n, k, 1.0f, A, m, B, k, 0.0f, C, m, &plan)
             : 0;
    unsetenv("CAMBLAS_PACKED_REQUIRE_SVE_HOOK");
    unsetenv("CAMBLAS_PACKED_TRACE");
    ok = configured && rc == -1 && memcmp(&plan, &sentinel, sizeof(plan)) == 0;
    CHECK(ok, "required SVE packed hook rejects ordinary-C fallback");
    free(A);
    free(B);
    free(C);
    return ok ? 0 : -1;
}

static int run_unrepresentable_span_rejection(void)
{
    const int m = INT_MAX, n = INT_MAX, k = 1;
    const double *fake_a = (const double *)(uintptr_t)1;
    const double *fake_b = (const double *)(uintptr_t)1;
    double *fake_c = (double *)(uintptr_t)1;
    packed_parallel_state_t state = {.m = m, .n = n};
    camblas_executor_t executor = {
        .run = packed_failing_executor,
        .user_data = &state,
    };
    camblas_topology_t topo = test_topology();
    camblas_ctx_t ctx = {
        4, CAMBLAS_REPRO_FAST, "shape-aware", &topo, &executor,
    };
    camblas_ctx_t no_executor_ctx = {
        1, CAMBLAS_REPRO_FAST, "shape-aware", &topo, NULL,
    };
    camblas_plan_t plan, sentinel;
    int rc;

    memset(&sentinel, 0xD3, sizeof(sentinel));
    plan = sentinel;
    rc = camblas_dgemm_plan(&ctx, 'N', 'N', m, n, k, 1.0, fake_a, m, fake_b, 1, 1.0, fake_c, m,
                            &plan);
    int rejected = rc == -1 && state.calls == 0 && memcmp(&plan, &sentinel, sizeof(plan)) == 0;
    CHECK(rejected, "unrepresentable fp64 GEMM span rejects before packed dispatch");

    /* The no-read/no-write alpha=0,beta=1 rule remains valid even for the
       same synthetic shape: no operand span is checked or accessed. */
    memset(&plan, 0, sizeof(plan));
    rc = camblas_dgemm_plan(&no_executor_ctx, 'N', 'N', m, n, k, 0.0, fake_a, m, fake_b, 1, 1.0,
                            fake_c, m, &plan);
    int no_access = rc == 0 && plan.kernel_id == CAMBLAS_KERNEL_PACKED;
    CHECK(no_access, "extreme alpha=0 beta=1 GEMM preserves no-access contract");

    state.calls = 0;
    memset(&plan, 0xE4, sizeof(plan));
    rc = camblas_dgemm_plan(&ctx, 'N', 'N', 0, n, INT_MAX, 1.0, fake_a, 1, fake_b, INT_MAX, 1.0,
                            fake_c, 1, &plan);
    int zero_no_access = rc == 0 && state.calls == 0 && plan.kernel_id == CAMBLAS_KERNEL_SCALAR;
    CHECK(zero_no_access, "extreme zero-output GEMM ignores unaccessed operand spans");
    return (rejected && no_access && zero_no_access) ? 0 : -1;
}

int main(void)
{
    const char transes[4][2] = {{'N', 'N'}, {'N', 'T'}, {'T', 'N'}, {'T', 'T'}};
    for (int q = 0; q < 4; q++) {
        run_f32_general(transes[q][0], transes[q][1]);
        run_f64_general(transes[q][0], transes[q][1]);
    }
    run_beta_zero();
    run_alpha_zero_no_read();
    run_f32_custom_executor();
    run_f64_custom_executor();
    run_custom_executor_failure();
    run_required_sve_hook_failure();
    run_unrepresentable_span_rejection();
    if (failures != 0) {
        fprintf(stderr, "packed GEMM tests: %d failure(s)\n", failures);
        return 1;
    }
    printf("packed GEMM tests: PASS (blocked fp32/fp64, N/T, padding, alpha/beta gates)\n");
    return 0;
}
