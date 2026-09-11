/*
 * CAMBLAS guard-page / red-zone memory-safety test.
 *
 * Each operand buffer
 * (A, B, C) is backed by mmap memory whose logical END is placed flush
 * against a PROT_NONE guard page. Any forward overrun by even one byte
 * (i.e. any access to the byte immediately past the declared footprint)
 * triggers SIGSEGV. This regression-guards the exact leading-dimension
 * access pattern of the scalar GEMM — the bug class found during Milestone 8
 * (allocating lda*k instead of lda*a_cols), and any future kernel that
 * respects the same footprint contract.
 *
 * WHAT IS TESTED
 *   - SGEMM and DGEMM (float and double).
 *   - All four transpose combinations (NN, NT, TN, TT).
 *   - Minimal leading dimensions (lda == required minimum) and padded
 *     leading dimensions (lda == minimum + pad), so the code is verified
 *     to use ld as the stride rather than assuming ld == rows.
 *   - General alpha/beta, alpha=0 (early-return path), beta=0 (zeroing path).
 *   - Zero-dimension cases (k=0, m=0, n=0).
 *   - A representative subset under the shape-aware planner with a discovered
 *     topology, to confirm the integrated call path is also memory-safe.
 *
 * WHAT IS NOT CLAIMED
 *   - This is a correctness-and-memory-safety test on a login node. It makes
 *     NO performance claim.
 *   - The guard page catches FORWARD overruns past the declared footprint.
 *     Backward underrun (negative indexing) is structurally impossible in the
 *     scalar code (all indices are non-negative) and is separately covered by
 *     ASAN; it is not what this test targets.
 *   - This target checks reference-policy execution. The tuned Grace paths
 *     require the separate framework and packing correctness checks.
 *
 * Exit codes: 0 = all configs completed with no overrun and correct output;
 *             nonzero = crash (SIGSEGV/SIGBUS) or a correctness mismatch.
 */
#include "camblas.h"
#include "camblas_planner.h"
#include "camblas_topology.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <signal.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/mman.h>

/* ---- Crash diagnostics ---- */
/* Recorded so a SIGSEGV handler can report which configuration crashed. */
static volatile sig_atomic_t g_cur_index = -1;
static volatile sig_atomic_t g_cur_m = 0, g_cur_n = 0, g_cur_k = 0;
static volatile sig_atomic_t g_cur_dtype = 0;            /* 0=f32, 1=f64 */
static volatile sig_atomic_t g_cur_ta = 0, g_cur_tb = 0; /* 'N'/'T' */
static volatile sig_atomic_t g_cur_pad = 0;

static void sig_handler(int signum, siginfo_t *si, void *ctx)
{
    (void)si;
    (void)ctx;
    const char *signame = (signum == SIGSEGV)  ? "SIGSEGV"
                          : (signum == SIGBUS) ? "SIGBUS"
                                               : "SIGNAL";
    fprintf(stderr, "\n*** FATAL: %s during guard-page test ***\n", signame);
    fprintf(stderr, "  config index=%d  dtype=%s  trans=%c%c  m=%d n=%d k=%d  ld_pad=%d\n",
            (int)g_cur_index, g_cur_dtype ? "f64" : "f32", (char)g_cur_ta, (char)g_cur_tb,
            (int)g_cur_m, (int)g_cur_n, (int)g_cur_k, (int)g_cur_pad);
    fprintf(stderr, "  -> memory overrun (or underrun) detected: GEMM accessed\n");
    fprintf(stderr, "     memory outside the declared operand footprint.\n");
    _exit(1);
}

/* ---- Guard-page allocator ---- */
/*
 * Allocate a writable region of exactly want_bytes, with a PROT_NONE guard
 * page placed immediately AFTER the last byte, so an access to byte
 * want_bytes (a 1-byte forward overrun) traps. Returns the usable pointer
 * via *out_ptr and an opaque handle via *out_handle for guard_free().
 *
 * want_bytes is floored at 1 so a zero-footprint operand still receives a
 * valid, writable pointer with a guard page after it (the GEMM must not
 * touch it; if it does, the guard catches the overrun).
 */
typedef struct {
    void *base;      /* mmap base */
    size_t map_size; /* total mmap size */
} guard_handle_t;

static long g_pagesize = 0;

static void *guard_alloc(size_t want_bytes, guard_handle_t *handle)
{
    if (g_pagesize == 0) {
        long ps = sysconf(_SC_PAGESIZE);
        if (ps <= 0)
            ps = 4096;
        g_pagesize = ps;
    }
    if (want_bytes == 0)
        want_bytes = 1; /* floor: valid pointer + guard */

    size_t ps = (size_t)g_pagesize;
    size_t usable_pages = (want_bytes + ps - 1) / ps;
    size_t map_size = ps + usable_pages * ps + ps; /* lead guard + usable + trail guard */

    void *base = mmap(NULL, map_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED) {
        return NULL;
    }

    /* Leading guard page. */
    if (mprotect(base, ps, PROT_NONE) != 0) {
        munmap(base, map_size);
        return NULL;
    }
    /* Trailing guard page. */
    char *trail = (char *)base + ps + usable_pages * ps;
    if (mprotect(trail, ps, PROT_NONE) != 0) {
        munmap(base, map_size);
        return NULL;
    }

    /* Place the buffer so its END is exactly at the trailing guard page. */
    char *usable = trail - want_bytes; /* usable[want_bytes] == trail (PROT_NONE) */

    handle->base = base;
    handle->map_size = map_size;
    return usable;
}

static void guard_free(guard_handle_t *handle)
{
    if (handle->base && handle->map_size) {
        munmap(handle->base, handle->map_size);
        handle->base = NULL;
        handle->map_size = 0;
    }
}

/* ---- Naive reference (independent loop order: i, j, l) ---- */
/*
 * Different loop order from src/scalar.c (which uses j, i, l) so this is a
 * genuine cross-check rather than a copy. Computes C = beta*C + alpha*op(A)*op(B).
 */
static void ref_sgemm(char ta, char tb, int m, int n, int k, float alpha, const float *A, int lda,
                      const float *B, int ldb, float beta, float *C, int ldc)
{
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
            C[i + j * ldc] = (beta == 0.0f) ? 0.0f : beta * C[i + j * ldc];
    if (alpha == 0.0f)
        return;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0.0f;
            for (int l = 0; l < k; l++) {
                float a = (ta == 'N') ? A[i + l * lda] : A[l + i * lda];
                float b = (tb == 'N') ? B[l + j * ldb] : B[j + l * ldb];
                acc += a * b;
            }
            C[i + j * ldc] += alpha * acc;
        }
    }
}

static void ref_dgemm(char ta, char tb, int m, int n, int k, double alpha, const double *A, int lda,
                      const double *B, int ldb, double beta, double *C, int ldc)
{
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
            C[i + j * ldc] = (beta == 0.0) ? 0.0 : beta * C[i + j * ldc];
    if (alpha == 0.0)
        return;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double acc = 0.0;
            for (int l = 0; l < k; l++) {
                double a = (ta == 'N') ? A[i + l * lda] : A[l + i * lda];
                double b = (tb == 'N') ? B[l + j * ldb] : B[j + l * ldb];
                acc += a * b;
            }
            C[i + j * ldc] += alpha * acc;
        }
    }
}

/* ---- Test a single configuration ---- */
/* Returns 0 on success, nonzero on correctness mismatch. A memory overrun
 * does not return — it crashes via the guard page. */
static int run_one(int dtype, char ta, char tb, int m, int n, int k, int pad, double alpha_d,
                   double beta_d, const camblas_ctx_t *ctx)
{
    int a_cols = (ta == 'N') ? k : m;
    int b_cols = (tb == 'N') ? n : k;
    int min_lda = (ta == 'N') ? m : k;
    int min_ldb = (tb == 'N') ? k : n;
    int lda = min_lda + pad;
    if (lda < 1)
        lda = 1;
    int ldb = min_ldb + pad;
    if (ldb < 1)
        ldb = 1;
    int ldc = m + pad;
    if (ldc < 1)
        ldc = 1;

    size_t elem = (dtype == 0) ? sizeof(float) : sizeof(double);
    size_t a_bytes = (size_t)lda * (size_t)a_cols * elem;
    size_t b_bytes = (size_t)ldb * (size_t)b_cols * elem;
    size_t c_bytes = (size_t)ldc * (size_t)n * elem;

    guard_handle_t ha, hb, hc, hrc; /* hrc: reference C (plain, separate) */
    void *A = guard_alloc(a_bytes, &ha);
    void *B = guard_alloc(b_bytes, &hb);
    void *C = guard_alloc(c_bytes, &hc);
    if (!A || !B || !C) {
        guard_free(&ha);
        guard_free(&hb);
        guard_free(&hc);
        fprintf(stderr, "ERROR: guard_alloc failed for %c%c m=%d n=%d k=%d\n", ta, tb, m, n, k);
        return 2;
    }

    /* Reference C uses a separate guard allocation too, so the reference is
     * also checked for footprint. Fill inputs deterministically. */
    void *Cref = guard_alloc(c_bytes, &hrc);
    if (!Cref) {
        guard_free(&ha);
        guard_free(&hb);
        guard_free(&hc);
        fprintf(stderr, "ERROR: guard_alloc(Cref) failed\n");
        return 2;
    }

    /* Fill A, B with a deterministic pattern; fill C and Cref identically. */
    if (dtype == 0) {
        float *Af = A, *Bf = B, *Cf = C, *Crf = Cref;
        for (size_t idx = 0; idx < a_bytes / elem; idx++)
            Af[idx] = (float)((int)((idx * 7 + 3) % 97) - 48) / 10.0f;
        for (size_t idx = 0; idx < b_bytes / elem; idx++)
            Bf[idx] = (float)((int)((idx * 11 + 5) % 89) - 44) / 10.0f;
        for (size_t idx = 0; idx < c_bytes / elem; idx++) {
            float v = (float)((int)((idx * 13 + 1) % 101) - 50) / 10.0f;
            Cf[idx] = v;
            Crf[idx] = v;
        }
    } else {
        double *Ad = A, *Bd = B, *Cd = C, *Crd = Cref;
        for (size_t idx = 0; idx < a_bytes / elem; idx++)
            Ad[idx] = (double)((int)((idx * 7 + 3) % 97) - 48) / 10.0;
        for (size_t idx = 0; idx < b_bytes / elem; idx++)
            Bd[idx] = (double)((int)((idx * 11 + 5) % 89) - 44) / 10.0;
        for (size_t idx = 0; idx < c_bytes / elem; idx++) {
            double v = (double)((int)((idx * 13 + 1) % 101) - 50) / 10.0;
            Cd[idx] = v;
            Crd[idx] = v;
        }
    }

    int rc_camblas, rc_ref_unused;
    (void)rc_ref_unused;
    if (dtype == 0) {
        float af = (float)alpha_d, bf = (float)beta_d;
        rc_camblas = camblas_sgemm(ctx, ta, tb, m, n, k, af, A, lda, B, ldb, bf, C, ldc);
        ref_sgemm(ta, tb, m, n, k, af, A, lda, B, ldb, bf, Cref, ldc);
    } else {
        rc_camblas = camblas_dgemm(ctx, ta, tb, m, n, k, alpha_d, A, lda, B, ldb, beta_d, C, ldc);
        ref_dgemm(ta, tb, m, n, k, alpha_d, A, lda, B, ldb, beta_d, Cref, ldc);
    }

    /* If CAMBLAS rejected the op (e.g. invalid for shape-aware w/o topo),
     * the caller sets up a valid ctx, so this should not happen. Treat as
     * failure if it does. */
    if (rc_camblas != 0) {
        fprintf(stderr, "  FAIL: camblas_*gemm returned %d for %c%c m=%d n=%d k=%d pad=%d\n",
                rc_camblas, ta, tb, m, n, k, pad);
        guard_free(&ha);
        guard_free(&hb);
        guard_free(&hc);
        guard_free(&hrc);
        return 3;
    }

    /* Compare C (CAMBLAS) vs Cref (reference). Both must be bitwise-identical
     * in accumulation only if loop order matched; they don't, so use a
     * tolerance. fp32: 1e-4 abs; fp64: 1e-9 abs. */
    int mismatch = 0;
    double tol = (dtype == 0) ? 1e-4 : 1e-9;
    double worst = 0.0;
    if (dtype == 0) {
        float *Cf = C, *Crf = Cref;
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++) {
                double d = (double)Cf[i + j * ldc] - (double)Crf[i + j * ldc];
                if (d < 0)
                    d = -d;
                if (d > worst)
                    worst = d;
                if (!isfinite(d) || d > tol)
                    mismatch = 1;
            }
    } else {
        double *Cd = C, *Crd = Cref;
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++) {
                double d = Cd[i + j * ldc] - Crd[i + j * ldc];
                if (d < 0)
                    d = -d;
                if (d > worst)
                    worst = d;
                if (!isfinite(d) || d > tol)
                    mismatch = 1;
            }
    }

    if (mismatch) {
        fprintf(stderr,
                "  FAIL: correctness mismatch %c%c m=%d n=%d k=%d pad=%d "
                "alpha=%.3g beta=%.3g worst=%.3e (tol=%.0e)\n",
                ta, tb, m, n, k, pad, alpha_d, beta_d, worst, tol);
    }

    guard_free(&ha);
    guard_free(&hb);
    guard_free(&hc);
    guard_free(&hrc);
    return mismatch ? 1 : 0;
}

/* ---- A negative control: confirm the guard actually traps an overrun ---- */
/*
 * This is run ONLY when --self-check is passed. It deliberately writes one
 * element past a guarded buffer to confirm the guard page traps (process
 * exits with SIGSEGV, which we convert to a documented exit code). This
 * validates that the test machinery can detect an overrun — i.e. a negative
 * result here would mean the guard-page mechanism itself is broken, so a
 * passing GEMM run would be meaningless.
 *
 * Implementation: fork(); the child writes past the guard and must die with
 * SIGSEGV/SIGBUS; the parent checks the child was killed by a signal.
 */
static int self_check_guard_traps(void)
{
    int ok = 1;

    /* (1) Positive control: an IN-BOUNDS access must NOT trap. If this
     * traps, the guard machinery is too aggressive and a passing main run
     * would be meaningless. */
    pid_t pid_ok = fork();
    if (pid_ok < 0) {
        fprintf(stderr, "self-check: fork failed\n");
        return 1;
    }
    if (pid_ok == 0) {
        signal(SIGSEGV, SIG_DFL);
        signal(SIGBUS, SIG_DFL);
        guard_handle_t h;
        char *p = guard_alloc(1, &h); /* 1 usable byte */
        if (!p)
            _exit(2);
        p[0] = 'x'; /* in-bounds: must NOT trap */
        _exit(0);
    }
    int st_ok = 0;
    waitpid(pid_ok, &st_ok, 0);
    if (WIFEXITED(st_ok) && WEXITSTATUS(st_ok) == 0) {
        printf("[self-check] positive control: in-bounds access did NOT trap (OK)\n");
    } else {
        fprintf(stderr, "[self-check] FAIL: in-bounds access trapped or failed (status 0x%x)\n",
                st_ok);
        ok = 0;
    }

    /* (2) Negative control: a 1-byte OVERRUN MUST trap. */
    pid_t pid_bad = fork();
    if (pid_bad < 0) {
        fprintf(stderr, "self-check: fork failed\n");
        return 1;
    }
    if (pid_bad == 0) {
        signal(SIGSEGV, SIG_DFL);
        signal(SIGBUS, SIG_DFL);
        guard_handle_t h;
        char *p = guard_alloc(1, &h); /* 1 usable byte, guard right after */
        if (!p)
            _exit(2);
        p[0] = 'x'; /* in-bounds */
        p[1] = 'y'; /* 1-byte overrun: MUST trap */
        _exit(0);   /* should never reach here */
    }
    int st_bad = 0;
    waitpid(pid_bad, &st_bad, 0);
    if (WIFSIGNALED(st_bad) && (WTERMSIG(st_bad) == SIGSEGV || WTERMSIG(st_bad) == SIGBUS)) {
        printf("[self-check] negative control: 1-byte overrun correctly trapped "
               "(child killed by SIG%s)\n",
               WTERMSIG(st_bad) == SIGSEGV ? "SEGV" : "BUS");
    } else {
        fprintf(stderr,
                "[self-check] FAIL: overrun was NOT trapped (status 0x%x) "
                "— guard mechanism broken\n",
                st_bad);
        ok = 0;
    }

    return ok ? 0 : 1;
}

int main(int argc, char **argv)
{
    int self_check = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--self-check") == 0)
            self_check = 1;
        else {
            fprintf(stderr, "unknown option: %s\n", argv[i]);
            fprintf(stderr, "usage: %s [--self-check]\n", argv[0]);
            return 1;
        }
    }

    /* Install SIGSEGV/SIGBUS handler for diagnostics. */
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = sig_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_SIGINFO;
    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGBUS, &sa, NULL);

    if (self_check) {
        printf("=== Guard-page self-check ===\n");
        return self_check_guard_traps();
    }

    printf("=== CAMBLAS Guard-Page Memory-Safety Test ===\n");
    printf("page size: %ld bytes\n", g_pagesize ? g_pagesize : sysconf(_SC_PAGESIZE));
    printf("Each operand buffer ends flush against a PROT_NONE guard page;\n");
    printf("a 1-byte forward overrun traps as SIGSEGV.\n\n");

    /* Size cases (m, n, k): cover 1x1x1, square, rectangular, odd, small-k. */
    struct {
        int m, n, k;
    } sizes[] = {
        {1, 1, 1},  {16, 16, 16}, {7, 5, 11},  {33, 17, 3},
        {5, 9, 13}, {13, 5, 9},   {17, 33, 3}, {3, 3, 33},
    };
    int n_sizes = (int)(sizeof(sizes) / sizeof(sizes[0]));

    char trans_list[4][2] = {{'N', 'N'}, {'N', 'T'}, {'T', 'N'}, {'T', 'T'}};

    /* (alpha, beta) cases: general, beta=0 zeroing, alpha=0 early-return, both unit. */
    struct {
        double a, b;
    } ab[] = {
        {1.0, 0.0}, {2.5, 0.5}, {0.0, 1.0}, {1.7, 0.0}, {1.0, 1.0},
    };
    int n_ab = (int)(sizeof(ab) / sizeof(ab[0]));

    int pads[] = {0, 4}; /* minimal ld and padded ld */
    int n_pads = (int)(sizeof(pads) / sizeof(pads[0]));

    const camblas_ctx_t *ctx = &camblas_ctx_default;

    int n_run = 0, n_fail = 0;
    int idx = 0;

    for (int dt = 0; dt < 2; dt++) {
        for (int ti = 0; ti < 4; ti++) {
            char ta = trans_list[ti][0], tb = trans_list[ti][1];
            for (int si = 0; si < n_sizes; si++) {
                int m = sizes[si].m, n = sizes[si].n, k = sizes[si].k;
                for (int pi = 0; pi < n_pads; pi++) {
                    int pad = pads[pi];
                    for (int abi = 0; abi < n_ab; abi++) {
                        g_cur_index = idx++;
                        g_cur_dtype = dt;
                        g_cur_ta = ta;
                        g_cur_tb = tb;
                        g_cur_m = m;
                        g_cur_n = n;
                        g_cur_k = k;
                        g_cur_pad = pad;

                        int rc = run_one(dt, ta, tb, m, n, k, pad, ab[abi].a, ab[abi].b, ctx);
                        n_run++;
                        if (rc != 0)
                            n_fail++;
                    }
                }
            }
        }
    }

    /* Zero-dimension cases (k=0, m=0, n=0) with general alpha/beta. */
    {
        struct {
            int m, n, k;
        } zcases[] = {
            {5, 7, 0}, /* k=0: inner loop empty, beta scaling runs */
            {0, 7, 5}, /* m=0: no C elements */
            {5, 0, 7}, /* n=0: no C elements */
            {0, 0, 0}, /* all zero */
        };
        int nz = (int)(sizeof(zcases) / sizeof(zcases[0]));
        for (int dt = 0; dt < 2; dt++) {
            for (int zi = 0; zi < nz; zi++) {
                for (int ti = 0; ti < 4; ti++) {
                    char ta = trans_list[ti][0], tb = trans_list[ti][1];
                    g_cur_index = idx++;
                    g_cur_dtype = dt;
                    g_cur_ta = ta;
                    g_cur_tb = tb;
                    g_cur_m = zcases[zi].m;
                    g_cur_n = zcases[zi].n;
                    g_cur_k = zcases[zi].k;
                    g_cur_pad = 0;
                    int rc = run_one(dt, ta, tb, zcases[zi].m, zcases[zi].n, zcases[zi].k, 0, 2.5,
                                     0.5, ctx);
                    n_run++;
                    if (rc != 0)
                        n_fail++;
                }
            }
        }
    }

    /* Representative subset under the shape-aware planner with a discovered
     * topology, to confirm the integrated planner+execution path is also
     * memory-safe. (Execution is scalar regardless of plan; this verifies the
     * integration did not introduce an access-pattern regression.) */
    camblas_topology_t topo;
    int topo_rc = camblas_topology_discover(&topo);
    if (topo_rc == 0 && topo.n_allowed_cpus > 0) {
        camblas_ctx_t sa_ctx = {
            .num_threads = 2,
            .reproducibility = CAMBLAS_REPRO_FAST,
            .planner_mode = "shape-aware",
            .topo = &topo,
        };
        printf("\nShape-aware planner path (topology: %d CPUs, SVE VL=%d bits):\n",
               topo.n_allowed_cpus, topo.sve_vl_bits);
        int sa_sizes[][3] = {{1, 1, 1}, {16, 16, 16}, {7, 5, 11}, {200, 200, 200}};
        int n_sa = (int)(sizeof(sa_sizes) / sizeof(sa_sizes[0]));
        for (int dt = 0; dt < 2; dt++) {
            for (int si = 0; si < n_sa; si++) {
                for (int ti = 0; ti < 4; ti++) {
                    char ta = trans_list[ti][0], tb = trans_list[ti][1];
                    g_cur_index = idx++;
                    g_cur_dtype = dt;
                    g_cur_ta = ta;
                    g_cur_tb = tb;
                    g_cur_m = sa_sizes[si][0];
                    g_cur_n = sa_sizes[si][1];
                    g_cur_k = sa_sizes[si][2];
                    g_cur_pad = 0;
                    int rc = run_one(dt, ta, tb, sa_sizes[si][0], sa_sizes[si][1], sa_sizes[si][2],
                                     0, 1.0, 0.0, &sa_ctx);
                    n_run++;
                    if (rc != 0)
                        n_fail++;
                }
            }
        }
    } else {
        printf("\nSkipped shape-aware subset: topology discover returned %d "
               "(n_cpus=%d)\n",
               topo_rc, topo.n_allowed_cpus);
    }

    printf("\n=== Summary ===\n");
    printf("Configurations run: %d\n", n_run);
    printf("Failures: %d\n", n_fail);
    if (n_fail == 0) {
        printf("Result: ALL CONFIGS PASSED (no guard-page trap, no mismatch)\n");
        return 0;
    }
    printf("Result: FAILURES DETECTED\n");
    return 1;
}
