/* Experimental framework BLAS bridge. Native GEMM uses the CAMBLAS core;
 * unimplemented operations remain in an explicitly linked vendor dependency.
 * All three variants use this same bridge for symmetric call diagnostics.
 * Concurrent CAMBLAS GEMMs are serialised around shared execution resources
 * and workspace. Reusing allocation capacity never reuses packed input values.
 * Fork after bridge initialisation is unsupported and fails visibly. */
#define _GNU_SOURCE
#include "camblas.h"
#include "camblas_pthread.h"
#include "camblas_workspace.h"
#include "strassen_one_level.h"
#include <ctype.h>
#include <dlfcn.h>
#include <limits.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#ifndef FRAMEWORK_BACKEND
#error "Select FRAMEWORK_BACKEND: 0 CAMBLAS, 1 OpenBLAS, 2 NVPL"
#endif
#ifndef FRAMEWORK_VENDOR_LIBRARY
#error "Provide the exact compatibility/vendor library path"
#endif
#ifndef CAMBLAS_FRAMEWORK_NATIVE_SYRK
#define CAMBLAS_FRAMEWORK_NATIVE_SYRK 0
#endif
#ifndef CAMBLAS_FRAMEWORK_PRIVATE_SHORT
#define CAMBLAS_FRAMEWORK_PRIVATE_SHORT 0
#endif
#ifndef CAMBLAS_PRIVATE_K_LIMIT
#define CAMBLAS_PRIVATE_K_LIMIT 256
#endif
#ifndef CAMBLAS_FRAMEWORK_SMALL_THREADS
#define CAMBLAS_FRAMEWORK_SMALL_THREADS 0
#endif
#ifndef CAMBLAS_FRAMEWORK_OPENMP
#define CAMBLAS_FRAMEWORK_OPENMP 0
#endif
#ifndef CAMBLAS_FRAMEWORK_TEAM_REUSE
#define CAMBLAS_FRAMEWORK_TEAM_REUSE 0
#endif
#if CAMBLAS_FRAMEWORK_TEAM_REUSE && !CAMBLAS_FRAMEWORK_OPENMP
#error "Team reuse requires the OpenMP executor"
#endif
#ifndef CAMBLAS_FRAMEWORK_SYMMETRIC
#define CAMBLAS_FRAMEWORK_SYMMETRIC 0
#endif
#ifndef CAMBLAS_FRAMEWORK_BILINEAR
#define CAMBLAS_FRAMEWORK_BILINEAR 0
#endif
#ifndef CAMBLAS_FRAMEWORK_COMPACT
#define CAMBLAS_FRAMEWORK_COMPACT 0
#endif
#ifndef CAMBLAS_FRAMEWORK_DOT
#define CAMBLAS_FRAMEWORK_DOT 0
#endif
#if CAMBLAS_FRAMEWORK_DOT
#include "dot_product.h"
#endif
#if CAMBLAS_FRAMEWORK_COMPACT
#include "compact.h"
#endif
#ifndef CAMBLAS_FRAMEWORK_EXEC_PROFILE
#define CAMBLAS_FRAMEWORK_EXEC_PROFILE 0
#endif
#if CAMBLAS_FRAMEWORK_BILINEAR
#include "bilinear.h"
#endif
#ifndef CAMBLAS_FRAMEWORK_PRETOUCH
#define CAMBLAS_FRAMEWORK_PRETOUCH 0
#endif
#ifndef CAMBLAS_FRAMEWORK_TOUCH_WIDE
#define CAMBLAS_FRAMEWORK_TOUCH_WIDE 0
#endif
#ifndef CAMBLAS_FRAMEWORK_TOUCH_MAX_BYTES
#define CAMBLAS_FRAMEWORK_TOUCH_MAX_BYTES (64u * 1024u * 1024u)
#endif
#ifndef CAMBLAS_FRAMEWORK_REUSE_WIDE_A
#define CAMBLAS_FRAMEWORK_REUSE_WIDE_A 0
#endif
#if CAMBLAS_FRAMEWORK_PRETOUCH
#if !CAMBLAS_FRAMEWORK_OPENMP
#error "Output preparation currently requires the OpenMP bridge executor"
#endif
#include "output_page_touch.h"
static framework_output_touch_t pending_output_touch;
static size_t output_page_bytes;
#endif
#if CAMBLAS_FRAMEWORK_SYMMETRIC
#include "symmetric.h"
#endif
#if CAMBLAS_FRAMEWORK_OPENMP
#ifndef _OPENMP
#error "OpenMP executor requires -fopenmp"
#endif
#if CAMBLAS_FRAMEWORK_SMALL_THREADS
#error "Do not combine OpenMP and the experimental auxiliary pthread pool"
#endif
#include <omp.h>
#endif
#if CAMBLAS_FRAMEWORK_SMALL_THREADS < 0 || CAMBLAS_FRAMEWORK_SMALL_THREADS > 64
#error "Invalid small-operation thread cap"
#endif
/* Counter order is also consumed by the Python validators. Keep this sequence
 * stable: GEMM, SYRK, Strassen, native kernel routes, errors, then ABI routes. */
enum {
    SG,
    DG,
    SS,
    DS,
    STRASSEN_S,
    STRASSEN_D,
    SCALAR,
    NOPACK,
    PACKED,
    ERRORS,
    C_INTERFACE,
    F_INTERFACE,
    NCOUNTERS
};
static _Atomic uint64_t counters[NCOUNTERS];
static pthread_once_t once = PTHREAD_ONCE_INIT;
static pthread_mutex_t gemm_mutex = PTHREAD_MUTEX_INITIALIZER;
static int thread_count, trace_calls;
static pid_t owner;
static void *vendor;
typedef void (*sg_t)(int, int, int, int, int, int, float, const float *, int, const float *, int,
                     float, float *, int);
typedef void (*dg_t)(int, int, int, int, int, int, double, const double *, int, const double *, int,
                     double, double *, int);
typedef void (*ss_t)(int, int, int, int, int, float, const float *, int, float, float *, int);
typedef void (*ds_t)(int, int, int, int, int, double, const double *, int, double, double *, int);
static sg_t vendor_sg;
static dg_t vendor_dg;
static ss_t vendor_ss;
static ds_t vendor_ds;
/* NVPL's CBLAS entry calls its interposable Fortran symbol. Call the
 * handle-resolved Fortran implementation directly to avoid bridge recursion. */
#define VENDOR_DIRECT(S, T)                                                                        \
    static void (*raw_##S##g)(const char *, const char *, const int *, const int *, const int *,   \
                              const T *, const T *, const int *, const T *, const int *,           \
                              const T *, T *, const int *);                                        \
    static void (*raw_##S##s)(const char *, const char *, const int *, const int *, const T *,     \
                              const T *, const int *, const T *, T *, const int *);                \
    static void direct_##S##g(int order, int ta, int tb, int m, int n, int k, T alpha, const T *a, \
                              int lda, const T *b, int ldb, T beta, T *c, int ldc)                 \
    {                                                                                              \
        char at = ta == 111 ? 'N' : 'T', bt = tb == 111 ? 'N' : 'T';                               \
        if (order == 101) {                                                                        \
            int x = m;                                                                             \
            m = n;                                                                                 \
            n = x;                                                                                 \
            x = lda;                                                                               \
            lda = ldb;                                                                             \
            ldb = x;                                                                               \
            const T *p = a;                                                                        \
            a = b;                                                                                 \
            b = p;                                                                                 \
            char t = at;                                                                           \
            at = bt;                                                                               \
            bt = t;                                                                                \
        }                                                                                          \
        raw_##S##g(&at, &bt, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);                \
    }                                                                                              \
    static void direct_##S##s(int order, int uplo, int ta, int n, int k, T alpha, const T *a,      \
                              int lda, T beta, T *c, int ldc)                                      \
    {                                                                                              \
        char u = uplo == 121 ? 'U' : 'L', t = ta == 111 ? 'N' : 'T';                               \
        if (order == 101) {                                                                        \
            u = u == 'U' ? 'L' : 'U';                                                              \
            t = t == 'N' ? 'T' : 'N';                                                              \
        }                                                                                          \
        raw_##S##s(&u, &t, &n, &k, &alpha, a, &lda, &beta, c, &ldc);                               \
    }
VENDOR_DIRECT(s, float)
VENDOR_DIRECT(d, double)
#undef VENDOR_DIRECT
static camblas_topology_t topology;
static camblas_ctx_t context;
static camblas_pthread_pool_t *pool;
static camblas_pthread_pool_t *small_pool;
static camblas_ctx_t small_context;
static _Atomic uint64_t small_pool_calls;
static _Atomic uint64_t symmetric_calls;
static _Atomic uint64_t output_touch_calls;
static _Atomic uint64_t output_resident_skips;
static void *symmetric_workspaces[2];
static void *bilinear_workspaces[2];
static void *compact_workspaces[2];
#if CAMBLAS_FRAMEWORK_COMPACT
static size_t compact_capacities[2];
#endif
static _Atomic uint64_t compact_calls;
static _Atomic uint64_t dot_calls;
#if CAMBLAS_FRAMEWORK_BILINEAR
static size_t bilinear_capacities[2];
#endif
static _Atomic uint64_t bilinear_calls;
#if CAMBLAS_FRAMEWORK_SYMMETRIC
static size_t symmetric_capacities[2];
#endif
static void *workspaces[2];
static size_t capacities[2];
static strassen_scratch_t scratches[2];
#if CAMBLAS_FRAMEWORK_OPENMP
#if CAMBLAS_FRAMEWORK_TEAM_REUSE
typedef struct {
    camblas_task_fn fn;
    const camblas_task_t *tasks;
    void *gctx;
    int count, stop, threads;
#if CAMBLAS_FRAMEWORK_PRETOUCH
    framework_output_touch_t touch;
#endif
} framework_team_t;
static _Thread_local framework_team_t *active_team;
static _Thread_local int inside_team_task;
/* Every team member encounters these same barrier regions in the same order.
 * The master publishes one synchronous executor batch before the first
 * barrier; all task writes are visible before the second barrier returns. */
static int framework_team_exchange(framework_team_t *team)
{
#pragma omp barrier
    if (team->stop)
        return 1;
    int t = omp_get_thread_num(), threads = omp_get_num_threads();
#if CAMBLAS_FRAMEWORK_PRETOUCH
    if (team->touch.c) {
        int block = team->touch.n / threads, tail = team->touch.n % threads;
        int begin = t * block + (t < tail ? t : tail), end = begin + block + (t < tail);
        for (int j = begin; j < end; j++)
            framework_touch_column(&team->touch, j);
#pragma omp barrier
    }
#endif
    int block = team->count / threads, tail = team->count % threads;
    int begin = t * block + (t < tail ? t : tail), end = begin + block + (t < tail);
    inside_team_task = 1;
    for (int i = begin; i < end; i++)
        team->fn(team->tasks + i, team->gctx);
    inside_team_task = 0;
#pragma omp barrier
    return 0;
}
static int framework_team_call(void (*fn)(void *), void *data, int m, int n, int k)
{
    if (omp_in_parallel() || thread_count < 16 || m > 2048 || n > 2048 || k > 4096) {
        fn(data);
        return 0;
    }
    framework_team_t team = {0};
#pragma omp parallel num_threads(thread_count)
    {
        if (omp_get_thread_num() == 0) {
            team.threads = omp_get_num_threads();
            active_team = &team;
            fn(data);
            team.stop = 1;
            framework_team_exchange(&team);
            active_team = NULL;
        } else
            while (!framework_team_exchange(&team)) {
            }
    }
    return team.threads == thread_count ? 0 : -1;
}
#endif
/* Reuse the application's OpenMP runtime, without changing its global thread
 * settings. The encountering thread participates in the team. Nested calls
 * execute serially rather than creating an oversubscribed nested team. */
static int framework_omp_run(camblas_task_fn fn, const camblas_task_t *tasks, int count, void *gctx,
                             void *data)
{
    if (!fn || count < 0 || (count && !tasks))
        return -1;
    if (!count)
        return 0;
    struct timespec profile_start = {0};
    if (CAMBLAS_FRAMEWORK_EXEC_PROFILE)
        clock_gettime(CLOCK_MONOTONIC, &profile_start);
#if CAMBLAS_FRAMEWORK_PRETOUCH
    framework_output_touch_t touch = pending_output_touch;
    pending_output_touch.c = NULL;
    if (touch.c)
        atomic_fetch_add_explicit(&output_touch_calls, 1, memory_order_relaxed);
#endif
#if CAMBLAS_FRAMEWORK_TEAM_REUSE
    if (active_team && !inside_team_task) {
        active_team->fn = fn;
        active_team->tasks = tasks;
        active_team->gctx = gctx;
        active_team->count = count;
#if CAMBLAS_FRAMEWORK_PRETOUCH
        active_team->touch = touch;
#endif
        framework_team_exchange(active_team);
        return 0;
    }
#endif
    if (omp_in_parallel()) {
#if CAMBLAS_FRAMEWORK_PRETOUCH
        if (touch.c)
            for (int j = 0; j < touch.n; j++)
                framework_touch_column(&touch, j);
#endif
        return camblas_executor_serial_fn(fn, tasks, count, gctx, NULL);
    }
    int threads = *(const int *)data;
    int actual = 0;
#pragma omp parallel num_threads(threads)
    {
#if CAMBLAS_FRAMEWORK_PRETOUCH
        if (touch.c) {
#pragma omp for schedule(static)
            for (int j = 0; j < touch.n; j++)
                framework_touch_column(&touch, j);
        }
#endif
#pragma omp for schedule(static)
        for (int i = 0; i < count; i++) {
            if (i == 0)
                actual = omp_get_num_threads();
            fn(tasks + i, gctx);
        }
    }
    if (CAMBLAS_FRAMEWORK_EXEC_PROFILE) {
        struct timespec end;
        clock_gettime(CLOCK_MONOTONIC, &end);
        long long ns = (end.tv_sec - profile_start.tv_sec) * 1000000000LL + end.tv_nsec -
                       profile_start.tv_nsec;
        fprintf(stderr, "FRAMEWORK_EXEC_DIAGNOSTIC tasks=%d threads=%d ns=%lld\n", count, threads,
                ns);
    }
    return actual == threads ? 0 : -1;
}
static camblas_executor_t framework_omp_executor = {framework_omp_run, &thread_count};
#endif
#if !CAMBLAS_FRAMEWORK_TEAM_REUSE
static int framework_team_call(void (*fn)(void *), void *data, int m, int n, int k)
{
    (void)m;
    (void)n;
    (void)k;
    fn(data);
    return 0;
}
#endif

static void fail(const char *message)
{
    atomic_fetch_add_explicit(&counters[ERRORS], 1, memory_order_relaxed);
    fprintf(stderr, "FRAMEWORK_BLAS_FATAL %s\n", message);
    abort();
}
static void load_symbol(void *destination, size_t size, const char *name)
{
    void *p = dlsym(vendor, name);
    if (!p || size != sizeof(p))
        fail(name);
    memcpy(destination, &p, size);
}
static void initialize(void)
{
    const char *value = getenv("CAMBLAS_FRAMEWORK_THREADS");
    char *end = NULL;
    long requested = value ? strtol(value, &end, 10) : 1;
    if ((value && (!*value || *end)) || requested < 1 || requested > 64)
        fail("invalid thread count");
    thread_count = (int)requested;
    owner = getpid();
#if CAMBLAS_FRAMEWORK_PRETOUCH
    long page = sysconf(_SC_PAGESIZE);
    if (page < 1 || ((size_t)page & ((size_t)page - 1)))
        fail("unsupported page size");
    output_page_bytes = (size_t)page;
#endif
    value = getenv("CAMBLAS_FRAMEWORK_TRACE");
    trace_calls = value && !strcmp(value, "1");
    vendor = dlopen(FRAMEWORK_VENDOR_LIBRARY, RTLD_NOW | RTLD_LOCAL);
    if (!vendor) {
        fprintf(stderr, "%s\n", dlerror());
        fail("vendor load");
    }
    load_symbol(&raw_sg, sizeof(raw_sg), "sgemm_");
    load_symbol(&raw_dg, sizeof(raw_dg), "dgemm_");
    load_symbol(&raw_ss, sizeof(raw_ss), "ssyrk_");
    load_symbol(&raw_ds, sizeof(raw_ds), "dsyrk_");
    vendor_sg = direct_sg;
    vendor_dg = direct_dg;
    vendor_ss = direct_ss;
    vendor_ds = direct_ds;
    void (*set_threads)(int);
    int (*get_threads)(void);
    load_symbol(&set_threads, sizeof(set_threads),
                FRAMEWORK_BACKEND == 2 ? "nvpl_blas_set_num_threads" : "openblas_set_num_threads");
    load_symbol(&get_threads, sizeof(get_threads),
                FRAMEWORK_BACKEND == 2 ? "nvpl_blas_get_max_threads" : "openblas_get_num_threads");
    set_threads(thread_count);
    if (get_threads() != thread_count)
        fail("vendor thread count mismatch");
}
static void ready(void)
{
    if (pthread_once(&once, initialize))
        fail("pthread_once");
    if (owner != getpid())
        fail("fork after bridge initialization is unsupported; use spawn/exec");
}
const char *framework_blas_backend(void)
{
    return FRAMEWORK_BACKEND == 0   ? "camblas+openblas-compatibility"
           : FRAMEWORK_BACKEND == 1 ? "openblas"
                                    : "nvpl";
}
const char *framework_blas_vendor_library(void)
{
    return FRAMEWORK_VENDOR_LIBRARY;
}
int framework_blas_threads(void)
{
    ready();
    return thread_count;
}
/* Reset between completed measurements, not concurrently with active calls.
 * Relaxed counters diagnose routing; they are not synchronisation primitives. */
void framework_blas_reset_stats(void)
{
    for (int i = 0; i < NCOUNTERS; i++)
        atomic_store_explicit(&counters[i], 0, memory_order_relaxed);
    atomic_store_explicit(&small_pool_calls, 0, memory_order_relaxed);
    atomic_store_explicit(&symmetric_calls, 0, memory_order_relaxed);
    atomic_store_explicit(&output_touch_calls, 0, memory_order_relaxed);
    atomic_store_explicit(&output_resident_skips, 0, memory_order_relaxed);
    atomic_store_explicit(&bilinear_calls, 0, memory_order_relaxed);
    atomic_store_explicit(&compact_calls, 0, memory_order_relaxed);
    atomic_store_explicit(&dot_calls, 0, memory_order_relaxed);
}
uint64_t framework_blas_dot_calls(void)
{
    return atomic_load_explicit(&dot_calls, memory_order_relaxed);
}
uint64_t framework_blas_compact_calls(void)
{
    return atomic_load_explicit(&compact_calls, memory_order_relaxed);
}
uint64_t framework_blas_small_pool_calls(void)
{
    return atomic_load_explicit(&small_pool_calls, memory_order_relaxed);
}
uint64_t framework_blas_symmetric_calls(void)
{
    return atomic_load_explicit(&symmetric_calls, memory_order_relaxed);
}
uint64_t framework_blas_output_touch_calls(void)
{
    return atomic_load_explicit(&output_touch_calls, memory_order_relaxed);
}
uint64_t framework_blas_output_resident_skips(void)
{
    return atomic_load_explicit(&output_resident_skips, memory_order_relaxed);
}
uint64_t framework_blas_bilinear_calls(void)
{
    return atomic_load_explicit(&bilinear_calls, memory_order_relaxed);
}
/**
 * Copy the fixed counter sequence into caller-owned storage.
 *
 * @param out Writable array containing at least NCOUNTERS uint64_t elements.
 * @param count Capacity of out in elements, not bytes.
 * @return NCOUNTERS on success, or -1 without writing for invalid storage.
 * @note Loads are individually atomic, not a coherent concurrent snapshot.
 *       Read after the measured operation completes for per-run statistics.
 */
int framework_blas_stats(uint64_t *out, size_t count)
{
    if (!out || count < NCOUNTERS)
        return -1;
    for (int i = 0; i < NCOUNTERS; i++)
        out[i] = atomic_load_explicit(&counters[i], memory_order_relaxed);
    return NCOUNTERS;
}
static char trans(int value)
{
    if (value == 111)
        return 'N';
    if (value == 112 || value == 113)
        return 'T';
    fail("unsupported transpose");
    return 'N';
}
static int ctrans(const char *value)
{
    if (!value)
        fail("null transpose");
    int c = toupper((unsigned char)*value);
    if (c == 'N')
        return 111;
    if (c == 'T')
        return 112;
    if (c == 'C')
        return 113;
    fail("unsupported Fortran transpose");
    return 111;
}
static void camblas_ready(void)
{
    if (context.executor)
        return;
    if (camblas_abi_version() != CAMBLAS_ABI_VERSION ||
        CAMBLAS_TOPO_IS_FATAL(camblas_topology_discover(&topology)) ||
        topology.sve_vl_bits != 128 || topology.n_allowed_cpus < thread_count)
        fail("CAMBLAS topology/ABI mismatch");
    int cpus[64], count = 0;
    for (int i = 0; i < CAMBLAS_MAX_CPUS && count < thread_count; i++)
        if (camblas_cpuset_test(&topology.allowed_cpus, i))
            cpus[count++] = i;
    if (count != thread_count)
        fail("CAMBLAS CPU count");
#if CAMBLAS_FRAMEWORK_OPENMP
    (void)cpus;
    const camblas_executor_t *executor = &framework_omp_executor;
#else
    pool = camblas_pthread_pool_create(thread_count, cpus);
    if (!pool)
        fail("CAMBLAS pool creation");
    const camblas_executor_t *executor = camblas_pthread_pool_executor(pool);
#endif
    context = (camblas_ctx_t){.num_threads = thread_count,
                              .reproducibility = CAMBLAS_REPRO_FAST,
                              .planner_mode = "shape-aware",
                              .topo = &topology,
                              .executor = executor};
}
static void ensure_workspace(int m, int n, int k, int fp64, int strassen)
{
    /* Called while gemm_mutex is held. Grow each precision's allocation only
     * when necessary; packing still reads the current operands on every call. */
    size_t bytes;
    if (camblas_workspace_bytes(n, k, fp64 ? CAMBLAS_DTYPE_F64 : CAMBLAS_DTYPE_F32, &bytes))
        fail("workspace query");
    size_t factor = strassen ? 1 : 4;
    if (bytes > SIZE_MAX / factor)
        fail("workspace overflow");
    bytes *= factor;
    /* A capacity hint, not a packed-layout contract: the core still checks
     * exact capacity before borrowing storage and can allocate A separately.
     * Keep asymmetric shared-A panels resident between calls, but repack all
     * input values on every call. The existing GEMM mutex owns this storage. */
    if (CAMBLAS_FRAMEWORK_REUSE_WIDE_A && !strassen && thread_count >= 16 && m > n && m <= 8192 &&
        n <= 2048 && k <= 4096) {
        size_t a_envelope, b_minimum;
        if (camblas_workspace_bytes(m + 256, k + 256, fp64 ? CAMBLAS_DTYPE_F64 : CAMBLAS_DTYPE_F32,
                                    &a_envelope) ||
            camblas_workspace_bytes(n, k, fp64 ? CAMBLAS_DTYPE_F64 : CAMBLAS_DTYPE_F32, &b_minimum))
            fail("wide workspace query");
        size_t wanted = b_minimum + a_envelope + a_envelope / 4;
        if (wanted <= 256u * 1024u * 1024u && wanted > bytes)
            bytes = wanted;
    }
    if (bytes > capacities[fp64]) {
        void *next = benchmark_alloc(bytes);
        if (!next)
            fail("workspace allocation");
        free(workspaces[fp64]);
        workspaces[fp64] = next;
        capacities[fp64] = bytes;
    }
}
static camblas_ctx_t *gemm_context(int m, int n, int k)
{
    if (!CAMBLAS_FRAMEWORK_SMALL_THREADS || thread_count < 64 || k > 256 || m > 2048 || n > 2048)
        return &context;
    if (!small_pool) {
        int cpus[64], count = 0;
        for (int i = 0; i < CAMBLAS_MAX_CPUS && count < CAMBLAS_FRAMEWORK_SMALL_THREADS; i++)
            if (camblas_cpuset_test(&topology.allowed_cpus, i))
                cpus[count++] = i;
        if (count != CAMBLAS_FRAMEWORK_SMALL_THREADS)
            fail("small pool CPU count");
        small_pool = camblas_pthread_pool_create(count, cpus);
        if (!small_pool)
            fail("small pool creation");
        small_context = context;
        small_context.num_threads = count;
        small_context.executor = camblas_pthread_pool_executor(small_pool);
    }
    atomic_fetch_add_explicit(&small_pool_calls, 1, memory_order_relaxed);
    return &small_context;
}
static void record_plan(const camblas_plan_t *plan)
{
    int index = plan->kernel_id == CAMBLAS_KERNEL_PACKED   ? PACKED
                : plan->kernel_id == CAMBLAS_KERNEL_NOPACK ? NOPACK
                                                           : SCALAR;
    atomic_fetch_add_explicit(&counters[index], 1, memory_order_relaxed);
}
static int symmetric_policy(int n, int k)
{
    return thread_count >= 16 && n >= 256 && n <= 1024 && k >= 512 && k <= 4096;
}
#if CAMBLAS_FRAMEWORK_SYMMETRIC
static void ensure_symmetric_workspace(int n, int k, int fp64)
{
    size_t bytes;
    if (camblas_symmetric_bytes(n, k, fp64, &bytes))
        fail("symmetric workspace query");
    if (bytes > symmetric_capacities[fp64]) {
        void *next = benchmark_alloc(bytes);
        if (!next)
            fail("symmetric workspace allocation");
        free(symmetric_workspaces[fp64]);
        symmetric_workspaces[fp64] = next;
        symmetric_capacities[fp64] = bytes;
    }
}
#define DEFINE_SYMMETRIC_CALL(S, T, FP64, FN)                                                    \
    static int try_symmetric_##S(char ta, char tb, int m, int n, int k, T alpha, const T *a,     \
                                 int lda, const T *b, int ldb, T beta, T *c, int ldc, int upper, \
                                 int full)                                                       \
    {                                                                                            \
        if (m != n || n < 1 || n > 1024 || k < 1 || k > 65536 || a != b || lda != ldb ||         \
            ta == tb || alpha == (T)0)                                                           \
            return 0;                                                                            \
        if (CAMBLAS_FRAMEWORK_SYMMETRIC != 2 && !symmetric_policy(n, k))                         \
            return 0;                                                                            \
        ensure_symmetric_workspace(n, k, FP64);                                                  \
        if (FN(context.executor, context.num_threads, ta, n, k, alpha, a, lda, beta, c, ldc,     \
               upper, full, symmetric_workspaces[FP64], symmetric_capacities[FP64]))             \
            fail("symmetric product failed");                                                    \
        atomic_fetch_add_explicit(&symmetric_calls, 1, memory_order_relaxed);                    \
        atomic_fetch_add_explicit(&counters[PACKED], 1, memory_order_relaxed);                   \
        return 1;                                                                                \
    }
DEFINE_SYMMETRIC_CALL(s, float, 0, camblas_symmetric_f32)
DEFINE_SYMMETRIC_CALL(d, double, 1, camblas_symmetric_f64)
#undef DEFINE_SYMMETRIC_CALL
#else
#define try_symmetric_s(ta, ...) ((void)(ta), 0)
#define try_symmetric_d(ta, ...) ((void)(ta), 0)
#endif
#if CAMBLAS_FRAMEWORK_COMPACT
#define DEFINE_COMPACT_CALL(S, T, FP64, FN)                                                   \
    static int try_compact_##S(char ta, char tb, int m, int n, int k, T alpha, const T *a,    \
                               int lda, const T *b, int ldb, T beta, T *c, int ldc)           \
    {                                                                                         \
        size_t needed;                                                                        \
        if (camblas_compact_bytes(m, n, k, FP64, &needed))                                    \
            return 0;                                                                         \
        if (CAMBLAS_FRAMEWORK_COMPACT != 2 &&                                                 \
            (thread_count < 32 || m < 256 || n < 256 || k < 128 || k > 1024))                 \
            return 0;                                                                         \
        if (needed > compact_capacities[FP64]) {                                              \
            void *next = benchmark_alloc(needed);                                             \
            if (!next)                                                                        \
                fail("compact allocation");                                                   \
            free(compact_workspaces[FP64]);                                                   \
            compact_workspaces[FP64] = next;                                                  \
            compact_capacities[FP64] = needed;                                                \
        }                                                                                     \
        if (FN(context.executor, context.num_threads, ta, tb, m, n, k, alpha, a, lda, b, ldb, \
               beta, c, ldc, compact_workspaces[FP64], compact_capacities[FP64]))             \
            fail("compact product failed");                                                   \
        atomic_fetch_add_explicit(&compact_calls, 1, memory_order_relaxed);                   \
        atomic_fetch_add_explicit(&counters[PACKED], 1, memory_order_relaxed);                \
        return 1;                                                                             \
    }
DEFINE_COMPACT_CALL(s, float, 0, camblas_compact_f32)
DEFINE_COMPACT_CALL(d, double, 1, camblas_compact_f64)
#undef DEFINE_COMPACT_CALL
#else
#define try_compact_s(ta, ...) ((void)(ta), 0)
#define try_compact_d(ta, ...) ((void)(ta), 0)
#endif
#if CAMBLAS_FRAMEWORK_DOT
#define DEFINE_DOT_CALL(S, T, FN)                                                               \
    static int try_dot_##S(char ta, char tb, int m, int n, int k, T alpha, const T *a, int lda, \
                           const T *b, int ldb, T beta, T *c, int ldc)                          \
    {                                                                                           \
        if (ta != 'T' || tb != 'N' || m > 2048 || n > 2048 || k > 1024)                         \
            return 0;                                                                           \
        if (CAMBLAS_FRAMEWORK_DOT != 2 && (thread_count < 16 || m < 128 || n < 128 || k < 64))  \
            return 0;                                                                           \
        if (FN(context.executor, context.num_threads, m, n, k, alpha, a, lda, b, ldb, beta, c,  \
               ldc))                                                                            \
            fail("dot product failed");                                                         \
        atomic_fetch_add_explicit(&dot_calls, 1, memory_order_relaxed);                         \
        atomic_fetch_add_explicit(&counters[NOPACK], 1, memory_order_relaxed);                  \
        return 1;                                                                               \
    }
DEFINE_DOT_CALL(s, float, camblas_dot_f32)
DEFINE_DOT_CALL(d, double, camblas_dot_f64)
#undef DEFINE_DOT_CALL
#else
#define try_dot_s(ta, ...) ((void)(ta), 0)
#define try_dot_d(ta, ...) ((void)(ta), 0)
#endif
#if CAMBLAS_FRAMEWORK_PRETOUCH
static void prepare_output_touch(int m, int n, int k, int ldc, int fp64, int beta_zero, void *c)
{
    pending_output_touch.c = NULL;
    size_t elements = (size_t)m * n;
    int bounded = CAMBLAS_FRAMEWORK_TOUCH_WIDE
                      ? (m <= 8192 && n <= 8192 &&
                         elements <= CAMBLAS_FRAMEWORK_TOUCH_MAX_BYTES /
                                         (fp64 ? sizeof(double) : sizeof(float)))
                      : (m <= 2048 && n <= 2048);
    if (beta_zero && (CAMBLAS_FRAMEWORK_PRETOUCH == 2 ||
                      (thread_count >= 16 && bounded && k <= 4096 && elements >= 262144))) {
        if (CAMBLAS_FRAMEWORK_PRETOUCH == 3) {
            /* Residency is a performance hint, not a proof of writable
             * private backing (a resident COW page may still fault).
             * Failure or oversized spans retain the safe preparation. */
            uintptr_t first = (uintptr_t)c & ~(output_page_bytes - 1);
            size_t span = ((size_t)(n - 1) * ldc + m) * (fp64 ? sizeof(double) : sizeof(float));
            size_t length = span + ((uintptr_t)c - first);
            size_t pages = (length + output_page_bytes - 1) / output_page_bytes;
            unsigned char residency[1025];
            if (pages <= sizeof(residency) && !mincore((void *)first, length, residency)) {
                int resident = 1;
                for (size_t page = 0; page < pages; page++)
                    if (!(residency[page] & 1)) {
                        resident = 0;
                        break;
                    }
                if (resident) {
                    atomic_fetch_add_explicit(&output_resident_skips, 1, memory_order_relaxed);
                    return;
                }
            }
        }
        pending_output_touch = (framework_output_touch_t){
            c, m, n, ldc, fp64 ? sizeof(double) : sizeof(float), output_page_bytes, 0};
    }
}
static void prepare_triangle_touch(int n, int k, int ldc, int fp64, int beta_zero, void *c,
                                   int upper)
{
    prepare_output_touch(n, n, k, ldc, fp64, beta_zero, c);
    if (pending_output_touch.c)
        pending_output_touch.triangle = upper ? 1 : 2;
}
static void clear_output_touch(void)
{
    pending_output_touch.c = NULL;
}
#else
#define prepare_output_touch(...) ((void)0)
#define prepare_triangle_touch(...) ((void)0)
#define clear_output_touch() ((void)0)
#endif
#if CAMBLAS_FRAMEWORK_BILINEAR
#define DEFINE_BILINEAR_CALL(S, T, FP64, FN, COUNTER)                                       \
    static int try_bilinear_##S(char ta, char tb, int m, int n, int k, T alpha, const T *a, \
                                int lda, const T *b, int ldb, T beta, T *c, int ldc)        \
    {                                                                                       \
        if (ta != 'N' || tb != 'N' || m != n || n != k || n < 2 || n > 2048 || n % 2 ||     \
            lda != n || ldb != n || ldc != n || alpha != (T)1 || beta != (T)0 ||            \
            (CAMBLAS_FRAMEWORK_BILINEAR != 2 && n < 512))                                   \
            return 0;                                                                       \
        size_t bytes;                                                                       \
        if (camblas_bilinear_bytes(n, FP64, &bytes))                                        \
            fail("bilinear workspace query");                                               \
        if (bytes > bilinear_capacities[FP64]) {                                            \
            void *next = benchmark_alloc(bytes);                                            \
            if (!next)                                                                      \
                fail("bilinear workspace allocation");                                      \
            free(bilinear_workspaces[FP64]);                                                \
            bilinear_workspaces[FP64] = next;                                               \
            bilinear_capacities[FP64] = bytes;                                              \
        }                                                                                   \
        if (FN(context.executor, n, a, b, c, bilinear_workspaces[FP64],                     \
               bilinear_capacities[FP64]))                                                  \
            fail("bilinear GEMM failed");                                                   \
        atomic_fetch_add_explicit(&bilinear_calls, 1, memory_order_relaxed);                \
        atomic_fetch_add_explicit(&counters[COUNTER], 1, memory_order_relaxed);             \
        atomic_fetch_add_explicit(&counters[PACKED], 1, memory_order_relaxed);              \
        return 1;                                                                           \
    }
DEFINE_BILINEAR_CALL(s, float, 0, camblas_bilinear_f32, STRASSEN_S)
DEFINE_BILINEAR_CALL(d, double, 1, camblas_bilinear_f64, STRASSEN_D)
#undef DEFINE_BILINEAR_CALL
#else
#define try_bilinear_s(ta, ...) ((void)(ta), 0)
#define try_bilinear_d(ta, ...) ((void)(ta), 0)
#endif
#if defined(CAMBLAS_FRAMEWORK_RECT64) && CAMBLAS_FRAMEWORK_RECT64
#include "rectangular64_policy.h"
#else
#define try_rect64_d(...) 0
#define release_rect64() ((void)0)
#endif
#if defined(CAMBLAS_FRAMEWORK_RECT32) && CAMBLAS_FRAMEWORK_RECT32
#include "rectangular32_policy.h"
#define try_rect64_s(...) try_rect32_s(__VA_ARGS__)
#else
#define try_rect64_s(...) 0
#define release_rect32() ((void)0)
#endif

#define DEFINE_GEMM(SUFFIX, TYPE, FP64, VENDOR, COUNTER, STRASSEN_COUNTER)                                                  \
    typedef struct {                                                                                                        \
        camblas_ctx_t *ctx;                                                                                                 \
        char ta, tb;                                                                                                        \
        int m, n, k, lda, ldb, ldc;                                                                                         \
        TYPE alpha, beta;                                                                                                   \
        const TYPE *a, *b;                                                                                                  \
        TYPE *c;                                                                                                            \
        camblas_plan_t *plan;                                                                                               \
        void *workspace;                                                                                                    \
        size_t capacity;                                                                                                    \
        int rc;                                                                                                             \
    } classical_##SUFFIX##_args;                                                                                            \
    static void classical_##SUFFIX##_call(void *opaque)                                                                     \
    {                                                                                                                       \
        classical_##SUFFIX##_args *w = opaque;                                                                              \
        w->rc = camblas_##SUFFIX##gemm_plan_workspace(                                                                      \
            w->ctx, w->ta, w->tb, w->m, w->n, w->k, w->alpha, w->a, w->lda, w->b, w->ldb, w->beta,                          \
            w->c, w->ldc, w->plan, w->workspace, w->capacity);                                                              \
    }                                                                                                                       \
    static void gemm_##SUFFIX(int order, int ta, int tb, int m, int n, int k, TYPE alpha,                                   \
                              const TYPE *a, int lda, const TYPE *b, int ldb, TYPE beta, TYPE *c,                           \
                              int ldc, int fortran)                                                                         \
    {                                                                                                                       \
        ready();                                                                                                            \
        atomic_fetch_add_explicit(&counters[COUNTER], 1, memory_order_relaxed);                                             \
        atomic_fetch_add_explicit(&counters[fortran ? F_INTERFACE : C_INTERFACE], 1,                                        \
                                  memory_order_relaxed);                                                                    \
        const char *algorithm = "vendor";                                                                                   \
        int kernel = -1;                                                                                                    \
        if (FRAMEWORK_BACKEND != 0)                                                                                         \
            VENDOR(order, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);                                            \
        else {                                                                                                              \
            char at = trans(ta), bt = trans(tb);                                                                            \
            if (order == 101) {                                                                                             \
                int t = m;                                                                                                  \
                m = n;                                                                                                      \
                n = t;                                                                                                      \
                t = lda;                                                                                                    \
                lda = ldb;                                                                                                  \
                ldb = t;                                                                                                    \
                char ct = at;                                                                                               \
                at = bt;                                                                                                    \
                bt = ct;                                                                                                    \
                const TYPE *p = a;                                                                                          \
                a = b;                                                                                                      \
                b = p;                                                                                                      \
            } else if (order != 102)                                                                                        \
                fail("unsupported layout");                                                                                 \
            if (m < 0 || n < 0 || k < 0 ||                                                                                  \
                lda < (at == 'N' ? (m > 0 ? m : 1) : (k > 0 ? k : 1)) ||                                                    \
                ldb < (bt == 'N' ? (k > 0 ? k : 1) : (n > 0 ? n : 1)) || ldc < (m > 0 ? m : 1))                             \
                fail("invalid GEMM dimensions");                                                                            \
            if (!m || !n)                                                                                                   \
                return;                                                                                                     \
            if (!c || ((alpha != (TYPE)0 && k) && (!a || !b)))                                                              \
                fail("null active GEMM buffer");                                                                            \
            if (alpha == (TYPE)0 || !k) {                                                                                   \
                if (beta != (TYPE)1)                                                                                        \
                    for (int j = 0; j < n; j++)                                                                             \
                        for (int i = 0; i < m; i++)                                                                         \
                            c[(size_t)j * ldc + i] =                                                                        \
                                beta == (TYPE)0 ? (TYPE)0 : beta * c[(size_t)j * ldc + i];                                  \
                return;                                                                                                     \
            }                                                                                                               \
            if (pthread_mutex_lock(&gemm_mutex))                                                                            \
                fail("GEMM lock");                                                                                          \
            camblas_ready();                                                                                                \
            prepare_output_touch(m, n, k, ldc, FP64, beta == (TYPE)0, c);                                                   \
            if (try_symmetric_##SUFFIX(at, bt, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, 1,                             \
                                       1)) {                                                                                \
                clear_output_touch();                                                                                       \
                if (pthread_mutex_unlock(&gemm_mutex))                                                                      \
                    fail("symmetric GEMM unlock");                                                                          \
                return;                                                                                                     \
            }                                                                                                               \
            camblas_ctx_t *call_context = gemm_context(m, n, k);                                                            \
            if (try_dot_##SUFFIX(at, bt, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)) {                                   \
                clear_output_touch();                                                                                       \
                if (pthread_mutex_unlock(&gemm_mutex))                                                                      \
                    fail("dot GEMM unlock");                                                                                \
                return;                                                                                                     \
            }                                                                                                               \
            if (try_compact_##SUFFIX(at, bt, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)) {                               \
                clear_output_touch();                                                                                       \
                if (pthread_mutex_unlock(&gemm_mutex))                                                                      \
                    fail("compact GEMM unlock");                                                                            \
                return;                                                                                                     \
            }                                                                                                               \
            if (try_bilinear_##SUFFIX(at, bt, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)) {                              \
                clear_output_touch();                                                                                       \
                if (pthread_mutex_unlock(&gemm_mutex))                                                                      \
                    fail("bilinear GEMM unlock");                                                                           \
                return;                                                                                                     \
            }                                                                                                               \
            int use_strassen = at == 'N' && bt == 'N' && m == n && n == k && lda == n &&                                    \
                               ldb == n && ldc == n && alpha == (TYPE)1 && beta == (TYPE)0 &&                               \
                               (n == 8192 || (n == 4096 && thread_count <= 32));                                            \
            ensure_workspace(m, n, k, FP64, use_strassen);                                                                  \
            camblas_plan_t plan;                                                                                            \
            int rc;                                                                                                         \
            if (try_rect64_##SUFFIX(call_context, at, bt, m, n, k, alpha, a, lda, b, ldb, beta, c,                          \
                                    ldc, &plan)) {                                                                          \
                rc = 0;                                                                                                     \
                algorithm = "batched-strassen";                                                                             \
                atomic_fetch_add_explicit(&counters[STRASSEN_COUNTER], 1, memory_order_relaxed);                            \
            } else if (use_strassen) {                                                                                      \
                if (scratches[FP64].n != n) {                                                                               \
                    free(scratches[FP64].data);                                                                             \
                    memset(&scratches[FP64], 0, sizeof(scratches[FP64]));                                                   \
                    if (strassen_scratch_init(&scratches[FP64], n, FP64))                                                   \
                        fail("Strassen scratch allocation");                                                                \
                }                                                                                                           \
                rc = strassen_one_level(call_context, FP64, n, a, b, c, &plan, workspaces[FP64],                            \
                                        capacities[FP64], &scratches[FP64]);                                                \
                algorithm = "strassen";                                                                                     \
                atomic_fetch_add_explicit(&counters[STRASSEN_COUNTER], 1, memory_order_relaxed);                            \
            } else {                                                                                                        \
                int private_short = CAMBLAS_FRAMEWORK_PRIVATE_SHORT &&                                                      \
                                    k <= CAMBLAS_PRIVATE_K_LIMIT && m <= 2048 && n <= 2048 &&                               \
                                    (k <= 256 || thread_count >= 32);                                                       \
                classical_##SUFFIX##_args call = {                                                                          \
                    .ctx = call_context,                                                                                    \
                    .ta = at,                                                                                               \
                    .tb = bt,                                                                                               \
                    .m = m,                                                                                                 \
                    .n = n,                                                                                                 \
                    .k = k,                                                                                                 \
                    .lda = lda,                                                                                             \
                    .ldb = ldb,                                                                                             \
                    .ldc = ldc,                                                                                             \
                    .alpha = alpha,                                                                                         \
                    .beta = beta,                                                                                           \
                    .a = a,                                                                                                 \
                    .b = b,                                                                                                 \
                    .c = c,                                                                                                 \
                    .plan = &plan,                                                                                          \
                    .workspace = private_short ? NULL : workspaces[FP64],                                                   \
                    .capacity = private_short ? 0 : capacities[FP64]};                                                      \
                int team_rc = framework_team_call(classical_##SUFFIX##_call, &call, m, n, k);                               \
                rc = team_rc ? team_rc : call.rc;                                                                           \
                algorithm = "classical";                                                                                    \
            }                                                                                                               \
            if (rc)                                                                                                         \
                fail("CAMBLAS GEMM failed");                                                                                \
            clear_output_touch();                                                                                           \
            kernel = plan.kernel_id;                                                                                        \
            record_plan(&plan);                                                                                             \
            if (pthread_mutex_unlock(&gemm_mutex))                                                                          \
                fail("GEMM unlock");                                                                                        \
        }                                                                                                                   \
        if (trace_calls)                                                                                                    \
            fprintf(                                                                                                        \
                stderr,                                                                                                     \
                "FRAMEWORK_BLAS backend=%s operation=gemm precision=%s algorithm=%s kernel=%d m=%d n=%d k=%d threads=%d\n", \
                framework_blas_backend(), FP64 ? "fp64" : "fp32", algorithm, kernel, m, n, k,                               \
                thread_count);                                                                                              \
    }                                                                                                                       \
    void cblas_##SUFFIX##gemm(int order, int ta, int tb, int m, int n, int k, TYPE alpha,                                   \
                              const TYPE *a, int lda, const TYPE *b, int ldb, TYPE beta, TYPE *c,                           \
                              int ldc)                                                                                      \
    {                                                                                                                       \
        gemm_##SUFFIX(order, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, 0);                                      \
    }                                                                                                                       \
    void SUFFIX##gemm_(const char *ta, const char *tb, const int *m, const int *n, const int *k,                            \
                       const TYPE *alpha, const TYPE *a, const int *lda, const TYPE *b,                                     \
                       const int *ldb, const TYPE *beta, TYPE *c, const int *ldc)                                           \
    {                                                                                                                       \
        gemm_##SUFFIX(102, ctrans(ta), ctrans(tb), *m, *n, *k, *alpha, a, *lda, b, *ldb, *beta, c,                          \
                      *ldc, 1);                                                                                             \
    }
DEFINE_GEMM(s, float, 0, vendor_sg, SG, STRASSEN_S)
DEFINE_GEMM(d, double, 1, vendor_dg, DG, STRASSEN_D)
#undef DEFINE_GEMM
/* Experimental native compatibility path: optionally use paired triangles;
 * otherwise compute a complete GEMM into
 * private temporary storage, then update ONLY the requested triangle.
 * This deliberately does extra arithmetic; it is not a triangular kernel.
 * Per-call storage avoids adding shared state or nesting the GEMM mutex. */
#define DEFINE_NATIVE_SYRK(SUFFIX, TYPE)                                                           \
    static void native_##SUFFIX##syrk(int order, int uplo, int ta, int n, int k, TYPE alpha,       \
                                      const TYPE *a, int lda, TYPE beta, TYPE *c, int ldc)         \
    {                                                                                              \
        if ((order != 101 && order != 102) || (uplo != 121 && uplo != 122) ||                      \
            (ta != 111 && ta != 112 && ta != 113) || n < 0 || k < 0 || ldc < (n ? n : 1))          \
            fail("invalid SYRK arguments");                                                        \
        int inner = order == 102 ? (ta == 111 ? n : k) : (ta == 111 ? k : n);                      \
        if (lda < (inner ? inner : 1))                                                             \
            fail("invalid SYRK leading dimension");                                                \
        if (!n || ((alpha == (TYPE)0 || !k) && beta == (TYPE)1))                                   \
            return;                                                                                \
        if (!c || (alpha != (TYPE)0 && k && !a))                                                   \
            fail("null active SYRK buffer");                                                       \
        if (CAMBLAS_FRAMEWORK_SYMMETRIC && alpha != (TYPE)0 && k > 0 && n <= 1024 && k <= 65536) { \
            if (pthread_mutex_lock(&gemm_mutex))                                                   \
                fail("SYRK lock");                                                                 \
            camblas_ready();                                                                       \
            char at = (order == 102) == (ta == 111) ? 'N' : 'T';                                   \
            prepare_triangle_touch(n, k, ldc, sizeof(TYPE) == sizeof(double), beta == (TYPE)0, c,  \
                                   order == 102 ? uplo == 121 : uplo == 122);                      \
            int used =                                                                             \
                try_symmetric_##SUFFIX(at, at == 'N' ? 'T' : 'N', n, n, k, alpha, a, lda, a, lda,  \
                                       beta, c, ldc, order == 102 ? uplo == 121 : uplo == 122, 0); \
            clear_output_touch();                                                                  \
            if (pthread_mutex_unlock(&gemm_mutex))                                                 \
                fail("SYRK unlock");                                                               \
            if (used)                                                                              \
                return;                                                                            \
        }                                                                                          \
        TYPE *product = NULL;                                                                      \
        if (alpha != (TYPE)0 && k) {                                                               \
            if ((size_t)n > SIZE_MAX / (size_t)n / sizeof(TYPE) ||                                 \
                (size_t)n * (size_t)n > PTRDIFF_MAX / sizeof(TYPE))                                \
                fail("SYRK allocation overflow");                                                  \
            product = benchmark_alloc((size_t)n * (size_t)n * sizeof(TYPE));                       \
            if (!product)                                                                          \
                fail("SYRK allocation");                                                           \
            gemm_##SUFFIX(order, ta, ta == 111 ? 112 : 111, n, n, k, alpha, a, lda, a, lda,        \
                          (TYPE)0, product, n, 0);                                                 \
        }                                                                                          \
        int upper = order == 102 ? uplo == 121 : uplo == 122;                                      \
        for (int j = 0; j < n; j++) {                                                              \
            int first = upper ? 0 : j, last = upper ? j + 1 : n;                                   \
            if (product && beta == (TYPE)0) {                                                      \
                memcpy(c + (size_t)j * ldc + first, product + (size_t)j * n + first,               \
                       (size_t)(last - first) * sizeof(TYPE));                                     \
                continue;                                                                          \
            }                                                                                      \
            for (int i = first; i < last; i++) {                                                   \
                size_t dst = (size_t)j * ldc + i;                                                  \
                size_t src = (size_t)j * n + i;                                                    \
                TYPE value = product ? product[src] : (TYPE)0;                                     \
                c[dst] = beta == (TYPE)0 ? value : value + beta * c[dst];                          \
            }                                                                                      \
        }                                                                                          \
        free(product);                                                                             \
    }
DEFINE_NATIVE_SYRK(s, float)
DEFINE_NATIVE_SYRK(d, double)
#undef DEFINE_NATIVE_SYRK
#define DEFINE_SYRK(SUFFIX, TYPE, VENDOR, COUNTER)                                              \
    void cblas_##SUFFIX##syrk(int order, int uplo, int ta, int n, int k, TYPE alpha,            \
                              const TYPE *a, int lda, TYPE beta, TYPE *c, int ldc)              \
    {                                                                                           \
        ready();                                                                                \
        atomic_fetch_add_explicit(&counters[COUNTER], 1, memory_order_relaxed);                 \
        int native = FRAMEWORK_BACKEND == 0 &&                                                  \
                     (CAMBLAS_FRAMEWORK_NATIVE_SYRK == 1 ||                                     \
                      (CAMBLAS_FRAMEWORK_NATIVE_SYRK == 2 && thread_count >= 64 && n >= 256 &&  \
                       n <= 1024 && k >= 1024) ||                                               \
                      (CAMBLAS_FRAMEWORK_NATIVE_SYRK == 3 && CAMBLAS_FRAMEWORK_SYMMETRIC &&     \
                       symmetric_policy(n, k)));                                                \
        if (native)                                                                             \
            native_##SUFFIX##syrk(order, uplo, ta, n, k, alpha, a, lda, beta, c, ldc);          \
        else                                                                                    \
            VENDOR(order, uplo, ta, n, k, alpha, a, lda, beta, c, ldc);                         \
        if (trace_calls)                                                                        \
            fprintf(stderr, "FRAMEWORK_BLAS backend=%s operation=syrk provider=%s n=%d k=%d\n", \
                    framework_blas_backend(),                                                   \
                    native                   ? "camblas-native"                                 \
                    : FRAMEWORK_BACKEND == 2 ? "nvpl"                                           \
                                             : "openblas",                                      \
                    n, k);                                                                      \
    }                                                                                           \
    void SUFFIX##syrk_(const char *uplo, const char *ta, const int *n, const int *k,            \
                       const TYPE *alpha, const TYPE *a, const int *lda, const TYPE *beta,      \
                       TYPE *c, const int *ldc)                                                 \
    {                                                                                           \
        int u = toupper((unsigned char)*uplo);                                                  \
        if (u != 'U' && u != 'L')                                                               \
            fail("invalid SYRK triangle");                                                      \
        cblas_##SUFFIX##syrk(102, u == 'U' ? 121 : 122, ctrans(ta), *n, *k, *alpha, a, *lda,    \
                             *beta, c, *ldc);                                                   \
    }
DEFINE_SYRK(s, float, vendor_ss, SS)
DEFINE_SYRK(d, double, vendor_ds, DS)
#undef DEFINE_SYRK
__attribute__((destructor)) static void release_resources(void)
{
    if (owner && owner == getpid()) {
        release_rect64();
        release_rect32();
        if (pool)
            camblas_pthread_pool_destroy(pool);
        if (small_pool)
            camblas_pthread_pool_destroy(small_pool);
        for (int i = 0; i < 2; i++) {
            free(workspaces[i]);
            free(scratches[i].data);
            free(symmetric_workspaces[i]);
            free(bilinear_workspaces[i]);
            free(compact_workspaces[i]);
        }
    }
}
