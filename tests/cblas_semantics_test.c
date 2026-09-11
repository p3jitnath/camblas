/*
 * Bounded CBLAS ABI-0 semantic contract test.
 *
 * The CBLAS entry points are void-valued, so this test exercises the current
 * observable error mechanism: invalid arguments are rejected before native
 * dispatch, cblas_xerbla receives the 1-based offending parameter, and
 * camblas_cblas_last_error() exposes that value to the calling thread.
 * ABI major 0 remains explicitly unstable; this is not an ABI-1 promise.
 */
#include "camblas_cblas.h"

#include <dlfcn.h>
#include <limits.h>
#include <stdint.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>

static int pass_count;
static int fail_count;

#define CHECK(condition, label)          \
    do {                                 \
        if (condition)                   \
            pass_count++;                \
        else {                           \
            fail_count++;                \
            printf("FAIL: %s\n", label); \
        }                                \
    } while (0)

/* Strong executable definition: -Wl,-E makes the shared-library call resolve
   here, proving the documented xerbla interposition boundary. */
static _Thread_local int hook_calls;
static _Thread_local int hook_parameter;
static _Thread_local char hook_routine[32];

void cblas_xerbla(int p, const char *rout, const char *form, ...)
{
    size_t i = 0;
    (void)form;
    hook_calls++;
    hook_parameter = p;
    for (; rout != NULL && i + 1 < sizeof(hook_routine) && rout[i] != '\0'; ++i)
        hook_routine[i] = rout[i];
    hook_routine[i] = '\0';
}

static void reset_hook(void)
{
    hook_calls = 0;
    hook_parameter = 0;
    hook_routine[0] = '\0';
}

static void check_hook(const char *label, int expected_parameter, const char *expected_routine)
{
    char text[160];
    snprintf(text, sizeof(text), "%s invokes exactly one xerbla hook", label);
    CHECK(hook_calls == 1, text);
    snprintf(text, sizeof(text), "%s reports xerbla parameter %d", label, expected_parameter);
    CHECK(hook_parameter == expected_parameter, text);
    snprintf(text, sizeof(text), "%s reports xerbla routine %s", label, expected_routine);
    CHECK(strcmp(hook_routine, expected_routine) == 0, text);
}

static void check_success_state(const char *label)
{
    char text[160];
    snprintf(text, sizeof(text), "%s leaves no CBLAS error", label);
    CHECK(camblas_cblas_last_error() == CAMBLAS_CBLAS_NO_ERROR, text);
    snprintf(text, sizeof(text), "%s does not invoke xerbla", label);
    CHECK(hook_calls == 0, text);
}

static void fill_f(float *values, int count)
{
    for (int i = 0; i < count; ++i)
        values[i] = (float)(i + 1);
}

static void fill_d(double *values, int count)
{
    for (int i = 0; i < count; ++i)
        values[i] = (double)(i + 1);
}

static void check_invalid_s(const char *label, camblas_cblas_order_t order,
                            camblas_cblas_transpose_t trans_a, camblas_cblas_transpose_t trans_b,
                            int m, int n, int k, const float *A, int lda, const float *B, int ldb,
                            float *C, int ldc, int expected_error)
{
    float before[64];
    if (C != NULL)
        memcpy(before, C, sizeof(before));
    reset_hook();
    cblas_sgemm(order, trans_a, trans_b, m, n, k, 1.0f, A, lda, B, ldb, 0.0f, C, ldc);
    char status_label[128];
    snprintf(status_label, sizeof(status_label), "%s reports parameter %d", label, expected_error);
    CHECK(camblas_cblas_last_error() == expected_error, status_label);
    check_hook(label, expected_error, "cblas_sgemm");
    if (C != NULL) {
        char unchanged_label[128];
        snprintf(unchanged_label, sizeof(unchanged_label), "%s leaves C unchanged", label);
        CHECK(memcmp(C, before, sizeof(before)) == 0, unchanged_label);
    }
}

static void check_invalid_d(const char *label, camblas_cblas_order_t order,
                            camblas_cblas_transpose_t trans_a, camblas_cblas_transpose_t trans_b,
                            int m, int n, int k, const double *A, int lda, const double *B, int ldb,
                            double *C, int ldc, int expected_error)
{
    double before[64];
    if (C != NULL)
        memcpy(before, C, sizeof(before));
    reset_hook();
    cblas_dgemm(order, trans_a, trans_b, m, n, k, 1.0, A, lda, B, ldb, 0.0, C, ldc);
    char status_label[128];
    snprintf(status_label, sizeof(status_label), "%s reports parameter %d", label, expected_error);
    CHECK(camblas_cblas_last_error() == expected_error, status_label);
    check_hook(label, expected_error, "cblas_dgemm");
    if (C != NULL) {
        char unchanged_label[128];
        snprintf(unchanged_label, sizeof(unchanged_label), "%s leaves C unchanged", label);
        CHECK(memcmp(C, before, sizeof(before)) == 0, unchanged_label);
    }
}

static void check_valid_s(void)
{
    /* Column-major 2x2 result: [23 31; 34 46]. */
    const float A_col[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float B_col[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    const float expected_col[4] = {23.0f, 34.0f, 31.0f, 46.0f};
    float C[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    reset_hook();
    cblas_sgemm(CAMBLAS_CBLAS_COL_MAJOR, CAMBLAS_CBLAS_NO_TRANS, CAMBLAS_CBLAS_NO_TRANS, 2, 2, 2,
                1.0f, A_col, 2, B_col, 2, 0.0f, C, 2);
    check_success_state("valid column-major SGEMM");
    CHECK(memcmp(C, expected_col, sizeof(C)) == 0, "valid column-major SGEMM result");

    /* Row-major 2x2 result: [19 22; 43 50]. */
    const float A_row[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float B_row[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    const float expected_row[4] = {19.0f, 22.0f, 43.0f, 50.0f};
    memset(C, 0, sizeof(C));
    reset_hook();
    cblas_sgemm(CAMBLAS_CBLAS_ROW_MAJOR, CAMBLAS_CBLAS_NO_TRANS, CAMBLAS_CBLAS_NO_TRANS, 2, 2, 2,
                1.0f, A_row, 2, B_row, 2, 0.0f, C, 2);
    check_success_state("valid row-major SGEMM");
    CHECK(memcmp(C, expected_row, sizeof(C)) == 0, "valid row-major SGEMM result");

    /* Real CONJ_TRANS is defined as TRANS; compare the observable bytes. */
    float C_trans[4], C_conj[4];
    reset_hook();
    cblas_sgemm(CAMBLAS_CBLAS_COL_MAJOR, CAMBLAS_CBLAS_TRANS, CAMBLAS_CBLAS_TRANS, 2, 2, 2, 1.0f,
                A_col, 2, B_col, 2, 0.0f, C_trans, 2);
    check_success_state("valid SGEMM TRANS");
    reset_hook();
    cblas_sgemm(CAMBLAS_CBLAS_COL_MAJOR, CAMBLAS_CBLAS_CONJ_TRANS, CAMBLAS_CBLAS_CONJ_TRANS, 2, 2,
                2, 1.0f, A_col, 2, B_col, 2, 0.0f, C_conj, 2);
    check_success_state("valid SGEMM CONJ_TRANS");
    CHECK(memcmp(C_trans, C_conj, sizeof(C_trans)) == 0, "real SGEMM CONJ_TRANS equals TRANS");
}

static void check_valid_d(void)
{
    const double A_col[4] = {1.0, 2.0, 3.0, 4.0};
    const double B_col[4] = {5.0, 6.0, 7.0, 8.0};
    const double expected_col[4] = {23.0, 34.0, 31.0, 46.0};
    double C[4] = {0.0, 0.0, 0.0, 0.0};

    reset_hook();
    cblas_dgemm(CAMBLAS_CBLAS_COL_MAJOR, CAMBLAS_CBLAS_NO_TRANS, CAMBLAS_CBLAS_NO_TRANS, 2, 2, 2,
                1.0, A_col, 2, B_col, 2, 0.0, C, 2);
    check_success_state("valid column-major DGEMM");
    CHECK(memcmp(C, expected_col, sizeof(C)) == 0, "valid column-major DGEMM result");

    const double A_row[4] = {1.0, 2.0, 3.0, 4.0};
    const double B_row[4] = {5.0, 6.0, 7.0, 8.0};
    const double expected_row[4] = {19.0, 22.0, 43.0, 50.0};
    memset(C, 0, sizeof(C));
    reset_hook();
    cblas_dgemm(CAMBLAS_CBLAS_ROW_MAJOR, CAMBLAS_CBLAS_NO_TRANS, CAMBLAS_CBLAS_NO_TRANS, 2, 2, 2,
                1.0, A_row, 2, B_row, 2, 0.0, C, 2);
    check_success_state("valid row-major DGEMM");
    CHECK(memcmp(C, expected_row, sizeof(C)) == 0, "valid row-major DGEMM result");

    double C_trans[4], C_conj[4];
    reset_hook();
    cblas_dgemm(CAMBLAS_CBLAS_COL_MAJOR, CAMBLAS_CBLAS_TRANS, CAMBLAS_CBLAS_TRANS, 2, 2, 2, 1.0,
                A_col, 2, B_col, 2, 0.0, C_trans, 2);
    check_success_state("valid DGEMM TRANS");
    reset_hook();
    cblas_dgemm(CAMBLAS_CBLAS_COL_MAJOR, CAMBLAS_CBLAS_CONJ_TRANS, CAMBLAS_CBLAS_CONJ_TRANS, 2, 2,
                2, 1.0, A_col, 2, B_col, 2, 0.0, C_conj, 2);
    check_success_state("valid DGEMM CONJ_TRANS");
    CHECK(memcmp(C_trans, C_conj, sizeof(C_trans)) == 0, "real DGEMM CONJ_TRANS equals TRANS");
}

static void check_ld_boundaries_s(const float *A, const float *B, float *C)
{
    const camblas_cblas_order_t orders[] = {CAMBLAS_CBLAS_COL_MAJOR, CAMBLAS_CBLAS_ROW_MAJOR};
    const camblas_cblas_transpose_t trans[] = {CAMBLAS_CBLAS_NO_TRANS, CAMBLAS_CBLAS_TRANS};

    for (unsigned int oi = 0; oi < 2; ++oi) {
        for (unsigned int ai = 0; ai < 2; ++ai) {
            for (unsigned int bi = 0; bi < 2; ++bi) {
                int lda_min =
                    (orders[oi] == CAMBLAS_CBLAS_COL_MAJOR) ? (ai == 0 ? 3 : 4) : (ai == 0 ? 4 : 3);
                int ldb_min =
                    (orders[oi] == CAMBLAS_CBLAS_COL_MAJOR) ? (bi == 0 ? 4 : 2) : (bi == 0 ? 2 : 4);
                int ldc_min = (orders[oi] == CAMBLAS_CBLAS_COL_MAJOR) ? 3 : 2;
                char label[128];

                snprintf(label, sizeof(label), "SGEMM %s/%s/%s exact LD minima",
                         orders[oi] == CAMBLAS_CBLAS_COL_MAJOR ? "col" : "row", ai == 0 ? "N" : "T",
                         bi == 0 ? "N" : "T");
                reset_hook();
                cblas_sgemm(orders[oi], trans[ai], trans[bi], 3, 2, 4, 1.0f, A, lda_min, B, ldb_min,
                            0.0f, C, ldc_min);
                check_success_state(label);

                snprintf(label, sizeof(label), "SGEMM %s/%s/%s lda boundary",
                         orders[oi] == CAMBLAS_CBLAS_COL_MAJOR ? "col" : "row", ai == 0 ? "N" : "T",
                         bi == 0 ? "N" : "T");
                check_invalid_s(label, orders[oi], trans[ai], trans[bi], 3, 2, 4, A, lda_min - 1, B,
                                ldb_min, C, ldc_min, 9);

                snprintf(label, sizeof(label), "SGEMM %s/%s/%s ldb boundary",
                         orders[oi] == CAMBLAS_CBLAS_COL_MAJOR ? "col" : "row", ai == 0 ? "N" : "T",
                         bi == 0 ? "N" : "T");
                check_invalid_s(label, orders[oi], trans[ai], trans[bi], 3, 2, 4, A, lda_min, B,
                                ldb_min - 1, C, ldc_min, 11);

                snprintf(label, sizeof(label), "SGEMM %s/%s/%s ldc boundary",
                         orders[oi] == CAMBLAS_CBLAS_COL_MAJOR ? "col" : "row", ai == 0 ? "N" : "T",
                         bi == 0 ? "N" : "T");
                check_invalid_s(label, orders[oi], trans[ai], trans[bi], 3, 2, 4, A, lda_min, B,
                                ldb_min, C, ldc_min - 1, 14);
            }
        }
    }
}

static void check_ld_boundaries_d(const double *A, const double *B, double *C)
{
    const camblas_cblas_order_t orders[] = {CAMBLAS_CBLAS_COL_MAJOR, CAMBLAS_CBLAS_ROW_MAJOR};
    const camblas_cblas_transpose_t trans[] = {CAMBLAS_CBLAS_NO_TRANS, CAMBLAS_CBLAS_TRANS};

    for (unsigned int oi = 0; oi < 2; ++oi) {
        for (unsigned int ai = 0; ai < 2; ++ai) {
            for (unsigned int bi = 0; bi < 2; ++bi) {
                int lda_min =
                    (orders[oi] == CAMBLAS_CBLAS_COL_MAJOR) ? (ai == 0 ? 3 : 4) : (ai == 0 ? 4 : 3);
                int ldb_min =
                    (orders[oi] == CAMBLAS_CBLAS_COL_MAJOR) ? (bi == 0 ? 4 : 2) : (bi == 0 ? 2 : 4);
                int ldc_min = (orders[oi] == CAMBLAS_CBLAS_COL_MAJOR) ? 3 : 2;
                char label[128];

                snprintf(label, sizeof(label), "DGEMM %s/%s/%s exact LD minima",
                         orders[oi] == CAMBLAS_CBLAS_COL_MAJOR ? "col" : "row", ai == 0 ? "N" : "T",
                         bi == 0 ? "N" : "T");
                reset_hook();
                cblas_dgemm(orders[oi], trans[ai], trans[bi], 3, 2, 4, 1.0, A, lda_min, B, ldb_min,
                            0.0, C, ldc_min);
                check_success_state(label);

                snprintf(label, sizeof(label), "DGEMM %s/%s/%s lda boundary",
                         orders[oi] == CAMBLAS_CBLAS_COL_MAJOR ? "col" : "row", ai == 0 ? "N" : "T",
                         bi == 0 ? "N" : "T");
                check_invalid_d(label, orders[oi], trans[ai], trans[bi], 3, 2, 4, A, lda_min - 1, B,
                                ldb_min, C, ldc_min, 9);

                snprintf(label, sizeof(label), "DGEMM %s/%s/%s ldb boundary",
                         orders[oi] == CAMBLAS_CBLAS_COL_MAJOR ? "col" : "row", ai == 0 ? "N" : "T",
                         bi == 0 ? "N" : "T");
                check_invalid_d(label, orders[oi], trans[ai], trans[bi], 3, 2, 4, A, lda_min, B,
                                ldb_min - 1, C, ldc_min, 11);

                snprintf(label, sizeof(label), "DGEMM %s/%s/%s ldc boundary",
                         orders[oi] == CAMBLAS_CBLAS_COL_MAJOR ? "col" : "row", ai == 0 ? "N" : "T",
                         bi == 0 ? "N" : "T");
                check_invalid_d(label, orders[oi], trans[ai], trans[bi], 3, 2, 4, A, lda_min, B,
                                ldb_min, C, ldc_min - 1, 14);
            }
        }
    }
}

static void check_zero_output_s(const float *A, const float *B, float *C)
{
    float before[64];
    fill_f(C, 64);
    memcpy(before, C, sizeof(before));
    reset_hook();
    cblas_sgemm(CAMBLAS_CBLAS_COL_MAJOR, CAMBLAS_CBLAS_NO_TRANS, CAMBLAS_CBLAS_NO_TRANS, 0, 2, 4,
                1.0f, A, 1, B, 1, 0.0f, C, 1);
    check_success_state("SGEMM column-major m=0 no-op");
    CHECK(memcmp(C, before, sizeof(before)) == 0, "SGEMM column-major m=0 leaves C unchanged");

    fill_f(C, 64);
    memcpy(before, C, sizeof(before));
    reset_hook();
    cblas_sgemm(CAMBLAS_CBLAS_ROW_MAJOR, CAMBLAS_CBLAS_TRANS, CAMBLAS_CBLAS_NO_TRANS, 3, 0, 4, 1.0f,
                A, 1, B, 1, 0.0f, C, 1);
    check_success_state("SGEMM row-major n=0 no-op");
    CHECK(memcmp(C, before, sizeof(before)) == 0, "SGEMM row-major n=0 leaves C unchanged");
}

static void check_zero_output_d(const double *A, const double *B, double *C)
{
    double before[64];
    fill_d(C, 64);
    memcpy(before, C, sizeof(before));
    reset_hook();
    cblas_dgemm(CAMBLAS_CBLAS_COL_MAJOR, CAMBLAS_CBLAS_NO_TRANS, CAMBLAS_CBLAS_NO_TRANS, 0, 2, 4,
                1.0, A, 1, B, 1, 0.0, C, 1);
    check_success_state("DGEMM column-major m=0 no-op");
    CHECK(memcmp(C, before, sizeof(before)) == 0, "DGEMM column-major m=0 leaves C unchanged");

    fill_d(C, 64);
    memcpy(before, C, sizeof(before));
    reset_hook();
    cblas_dgemm(CAMBLAS_CBLAS_ROW_MAJOR, CAMBLAS_CBLAS_TRANS, CAMBLAS_CBLAS_NO_TRANS, 3, 0, 4, 1.0,
                A, 1, B, 1, 0.0, C, 1);
    check_success_state("DGEMM row-major n=0 no-op");
    CHECK(memcmp(C, before, sizeof(before)) == 0, "DGEMM row-major n=0 leaves C unchanged");
}

static void check_extreme_dispatch_boundaries(void)
{
    float Af = 3.0f, Bf = 5.0f, Cf[64], Cfbefore[64];
    double Ad = 3.0, Bd = 5.0, Cd[64], Cdbefore[64];

    /* An alpha=0,beta=1 row-major call must remain a bounded no-op even at
       the largest LP64 dimensions. The span checker must skip all operands
       because this call reads neither inputs nor output. */
    fill_f(Cf, 64);
    memcpy(Cfbefore, Cf, sizeof(Cf));
    reset_hook();
    cblas_sgemm(CAMBLAS_CBLAS_ROW_MAJOR, CAMBLAS_CBLAS_NO_TRANS, CAMBLAS_CBLAS_NO_TRANS, INT_MAX,
                INT_MAX, 0, 0.0f, &Af, 1, &Bf, INT_MAX, 1.0f, Cf, INT_MAX);
    check_success_state("extreme row-major SGEMM no-access boundary");
    CHECK(memcmp(Cf, Cfbefore, sizeof(Cf)) == 0,
          "extreme row-major SGEMM no-access leaves C unchanged");

    /* The fp64 output span is not representable as a byte extent at this
       shape. It must fail before native dispatch and before any C access. */
    fill_d(Cd, 64);
    memcpy(Cdbefore, Cd, sizeof(Cd));
    reset_hook();
    cblas_dgemm(CAMBLAS_CBLAS_COL_MAJOR, CAMBLAS_CBLAS_NO_TRANS, CAMBLAS_CBLAS_NO_TRANS, INT_MAX,
                INT_MAX, 0, 0.0, &Ad, INT_MAX, &Bd, 1, 0.0, Cd, INT_MAX);
    CHECK(camblas_cblas_last_error() == CAMBLAS_CBLAS_INTERNAL_ERROR,
          "extreme column-major DGEMM reports internal span failure");
    check_hook("extreme column-major DGEMM span failure", CAMBLAS_CBLAS_INTERNAL_ERROR,
               "cblas_dgemm");
    CHECK(memcmp(Cd, Cdbefore, sizeof(Cd)) == 0,
          "extreme column-major DGEMM span failure leaves C unchanged");
}

typedef struct {
    int kind;
    int expected_parameter;
    const char *expected_routine;
} thread_case_t;

static void *thread_local_error_worker(void *opaque)
{
    const thread_case_t *test = (const thread_case_t *)opaque;
    float Af[64], Bf[64], Cf[64];
    double Ad[64], Bd[64], Cd[64];
    fill_f(Af, 64);
    fill_f(Bf, 64);
    fill_f(Cf, 64);
    fill_d(Ad, 64);
    fill_d(Bd, 64);
    fill_d(Cd, 64);
    reset_hook();
    if (test->kind == 0) {
        cblas_sgemm(CAMBLAS_CBLAS_COL_MAJOR, (camblas_cblas_transpose_t)999, CAMBLAS_CBLAS_NO_TRANS,
                    3, 2, 4, 1.0f, Af, 3, Bf, 4, 0.0f, Cf, 3);
    } else {
        cblas_dgemm(CAMBLAS_CBLAS_ROW_MAJOR, CAMBLAS_CBLAS_NO_TRANS, (camblas_cblas_transpose_t)999,
                    3, 2, 4, 1.0, Ad, 4, Bd, 2, 0.0, Cd, 2);
    }
    return (camblas_cblas_last_error() == test->expected_parameter && hook_calls == 1 &&
            hook_parameter == test->expected_parameter &&
            strcmp(hook_routine, test->expected_routine) == 0)
               ? (void *)(intptr_t)0
               : (void *)(intptr_t)1;
}

static void check_thread_local_errors(void)
{
    pthread_t threads[2];
    thread_case_t cases[2] = {
        {0, 2, "cblas_sgemm"},
        {1, 3, "cblas_dgemm"},
    };
    int create0, create1;
    void *result0 = (void *)(intptr_t)1;
    void *result1 = (void *)(intptr_t)1;
    int join0, join1;

    camblas_cblas_clear_error();
    reset_hook();
    create0 = pthread_create(&threads[0], NULL, thread_local_error_worker, &cases[0]);
    create1 = pthread_create(&threads[1], NULL, thread_local_error_worker, &cases[1]);
    CHECK(create0 == 0, "thread-local SGEMM error worker created");
    CHECK(create1 == 0, "thread-local DGEMM error worker created");
    join0 = (create0 == 0) ? pthread_join(threads[0], &result0) : -1;
    join1 = (create1 == 0) ? pthread_join(threads[1], &result1) : -1;
    CHECK(join0 == 0, "thread-local SGEMM error worker joined");
    CHECK(join1 == 0, "thread-local DGEMM error worker joined");
    CHECK((create0 != 0) || ((intptr_t)result0 == 0), "thread-local SGEMM error state is isolated");
    CHECK((create1 != 0) || ((intptr_t)result1 == 0), "thread-local DGEMM error state is isolated");
    CHECK(camblas_cblas_last_error() == CAMBLAS_CBLAS_NO_ERROR,
          "main thread CBLAS error remains clear");
    CHECK(hook_calls == 0, "main thread xerbla state remains clear");
}

static void check_default_xerbla(void)
{
    typedef void (*xerbla_fn)(int, const char *, const char *, ...);
    void *handle = dlopen("libcamblas.so", RTLD_NOW | RTLD_LOCAL);
    void *symbol = NULL;
    xerbla_fn fn = NULL;
    int size_ok = (sizeof(fn) == sizeof(symbol));

    CHECK(handle != NULL, "default xerbla library reopens");
    if (handle != NULL) {
        (void)dlerror();
        symbol = dlsym(handle, "cblas_xerbla");
    }
    CHECK(symbol != NULL, "default xerbla symbol resolves directly");
    CHECK(size_ok, "default xerbla function pointer sizes match");
    if (symbol != NULL && size_ok) {
        memcpy(&fn, &symbol, sizeof(fn));
        camblas_cblas_clear_error();
        fn(42, "default-test", "synthetic");
        CHECK(camblas_cblas_last_error() == 42, "default xerbla records the parameter index");
    } else {
        CHECK(0, "default xerbla records the parameter index");
    }
    if (handle != NULL) {
        CHECK(dlclose(handle) == 0, "default xerbla library closes");
    } else {
        CHECK(0, "default xerbla library closes");
    }
}

int main(void)
{
    float Af[64], Bf[64], Cf[64];
    double Ad[64], Bd[64], Cd[64];
    fill_f(Af, 64);
    fill_f(Bf, 64);
    fill_f(Cf, 64);
    fill_d(Ad, 64);
    fill_d(Bd, 64);
    fill_d(Cd, 64);

    puts("=== CAMBLAS CBLAS ABI-0 semantic contract test ===");

    /* Column-major invalid arguments: m=3, n=2, k=4, lda=3, ldb=4, ldc=3. */
    check_invalid_s("SGEMM invalid order", (camblas_cblas_order_t)999, CAMBLAS_CBLAS_NO_TRANS,
                    CAMBLAS_CBLAS_NO_TRANS, 3, 2, 4, Af, 3, Bf, 4, Cf, 3, 1);
    check_invalid_s("SGEMM invalid trans_a", CAMBLAS_CBLAS_COL_MAJOR,
                    (camblas_cblas_transpose_t)999, CAMBLAS_CBLAS_NO_TRANS, 3, 2, 4, Af, 3, Bf, 4,
                    Cf, 3, 2);
    check_invalid_s("SGEMM invalid trans_b", CAMBLAS_CBLAS_COL_MAJOR, CAMBLAS_CBLAS_NO_TRANS,
                    (camblas_cblas_transpose_t)999, 3, 2, 4, Af, 3, Bf, 4, Cf, 3, 3);
    check_invalid_s("SGEMM negative m", CAMBLAS_CBLAS_COL_MAJOR, CAMBLAS_CBLAS_NO_TRANS,
                    CAMBLAS_CBLAS_NO_TRANS, -1, 2, 4, Af, 3, Bf, 4, Cf, 3, 4);
    check_invalid_s("SGEMM negative n", CAMBLAS_CBLAS_COL_MAJOR, CAMBLAS_CBLAS_NO_TRANS,
                    CAMBLAS_CBLAS_NO_TRANS, 3, -2, 4, Af, 3, Bf, 4, Cf, 3, 5);
    check_invalid_s("SGEMM negative k", CAMBLAS_CBLAS_COL_MAJOR, CAMBLAS_CBLAS_NO_TRANS,
                    CAMBLAS_CBLAS_NO_TRANS, 3, 2, -4, Af, 3, Bf, 4, Cf, 3, 6);
    check_invalid_s("SGEMM lda too small", CAMBLAS_CBLAS_COL_MAJOR, CAMBLAS_CBLAS_NO_TRANS,
                    CAMBLAS_CBLAS_NO_TRANS, 3, 2, 4, Af, 2, Bf, 4, Cf, 3, 9);
    check_invalid_s("SGEMM ldb too small", CAMBLAS_CBLAS_COL_MAJOR, CAMBLAS_CBLAS_NO_TRANS,
                    CAMBLAS_CBLAS_NO_TRANS, 3, 2, 4, Af, 3, Bf, 3, Cf, 3, 11);
    check_invalid_s("SGEMM ldc too small", CAMBLAS_CBLAS_COL_MAJOR, CAMBLAS_CBLAS_NO_TRANS,
                    CAMBLAS_CBLAS_NO_TRANS, 3, 2, 4, Af, 3, Bf, 4, Cf, 2, 14);
    check_invalid_s("SGEMM NULL A", CAMBLAS_CBLAS_COL_MAJOR, CAMBLAS_CBLAS_NO_TRANS,
                    CAMBLAS_CBLAS_NO_TRANS, 3, 2, 4, NULL, 3, Bf, 4, Cf, 3, 8);
    check_invalid_s("SGEMM NULL B", CAMBLAS_CBLAS_COL_MAJOR, CAMBLAS_CBLAS_NO_TRANS,
                    CAMBLAS_CBLAS_NO_TRANS, 3, 2, 4, Af, 3, NULL, 4, Cf, 3, 10);
    check_invalid_s("SGEMM NULL C", CAMBLAS_CBLAS_COL_MAJOR, CAMBLAS_CBLAS_NO_TRANS,
                    CAMBLAS_CBLAS_NO_TRANS, 3, 2, 4, Af, 3, Bf, 4, NULL, 3, 13);

    /* Row-major leading dimensions use physical row-major column counts. */
    check_invalid_s("SGEMM row-major lda too small", CAMBLAS_CBLAS_ROW_MAJOR,
                    CAMBLAS_CBLAS_NO_TRANS, CAMBLAS_CBLAS_NO_TRANS, 3, 2, 4, Af, 3, Bf, 2, Cf, 2,
                    9);
    check_invalid_s("SGEMM row-major ldb too small", CAMBLAS_CBLAS_ROW_MAJOR,
                    CAMBLAS_CBLAS_NO_TRANS, CAMBLAS_CBLAS_NO_TRANS, 3, 2, 4, Af, 4, Bf, 1, Cf, 2,
                    11);
    check_invalid_s("SGEMM row-major ldc too small", CAMBLAS_CBLAS_ROW_MAJOR,
                    CAMBLAS_CBLAS_NO_TRANS, CAMBLAS_CBLAS_NO_TRANS, 3, 2, 4, Af, 4, Bf, 2, Cf, 1,
                    14);

    check_invalid_d("DGEMM invalid order", (camblas_cblas_order_t)999, CAMBLAS_CBLAS_NO_TRANS,
                    CAMBLAS_CBLAS_NO_TRANS, 3, 2, 4, Ad, 3, Bd, 4, Cd, 3, 1);
    check_invalid_d("DGEMM invalid trans_a", CAMBLAS_CBLAS_COL_MAJOR,
                    (camblas_cblas_transpose_t)999, CAMBLAS_CBLAS_NO_TRANS, 3, 2, 4, Ad, 3, Bd, 4,
                    Cd, 3, 2);
    check_invalid_d("DGEMM invalid trans_b", CAMBLAS_CBLAS_COL_MAJOR, CAMBLAS_CBLAS_NO_TRANS,
                    (camblas_cblas_transpose_t)999, 3, 2, 4, Ad, 3, Bd, 4, Cd, 3, 3);
    check_invalid_d("DGEMM negative m", CAMBLAS_CBLAS_COL_MAJOR, CAMBLAS_CBLAS_NO_TRANS,
                    CAMBLAS_CBLAS_NO_TRANS, -1, 2, 4, Ad, 3, Bd, 4, Cd, 3, 4);
    check_invalid_d("DGEMM negative n", CAMBLAS_CBLAS_COL_MAJOR, CAMBLAS_CBLAS_NO_TRANS,
                    CAMBLAS_CBLAS_NO_TRANS, 3, -2, 4, Ad, 3, Bd, 4, Cd, 3, 5);
    check_invalid_d("DGEMM negative k", CAMBLAS_CBLAS_COL_MAJOR, CAMBLAS_CBLAS_NO_TRANS,
                    CAMBLAS_CBLAS_NO_TRANS, 3, 2, -4, Ad, 3, Bd, 4, Cd, 3, 6);
    check_invalid_d("DGEMM lda too small", CAMBLAS_CBLAS_COL_MAJOR, CAMBLAS_CBLAS_NO_TRANS,
                    CAMBLAS_CBLAS_NO_TRANS, 3, 2, 4, Ad, 2, Bd, 4, Cd, 3, 9);
    check_invalid_d("DGEMM ldb too small", CAMBLAS_CBLAS_COL_MAJOR, CAMBLAS_CBLAS_NO_TRANS,
                    CAMBLAS_CBLAS_NO_TRANS, 3, 2, 4, Ad, 3, Bd, 3, Cd, 3, 11);
    check_invalid_d("DGEMM ldc too small", CAMBLAS_CBLAS_COL_MAJOR, CAMBLAS_CBLAS_NO_TRANS,
                    CAMBLAS_CBLAS_NO_TRANS, 3, 2, 4, Ad, 3, Bd, 4, Cd, 2, 14);
    check_invalid_d("DGEMM NULL A", CAMBLAS_CBLAS_COL_MAJOR, CAMBLAS_CBLAS_NO_TRANS,
                    CAMBLAS_CBLAS_NO_TRANS, 3, 2, 4, NULL, 3, Bd, 4, Cd, 3, 8);
    check_invalid_d("DGEMM NULL B", CAMBLAS_CBLAS_COL_MAJOR, CAMBLAS_CBLAS_NO_TRANS,
                    CAMBLAS_CBLAS_NO_TRANS, 3, 2, 4, Ad, 3, NULL, 4, Cd, 3, 10);
    check_invalid_d("DGEMM NULL C", CAMBLAS_CBLAS_COL_MAJOR, CAMBLAS_CBLAS_NO_TRANS,
                    CAMBLAS_CBLAS_NO_TRANS, 3, 2, 4, Ad, 3, Bd, 4, NULL, 3, 13);
    check_invalid_d("DGEMM row-major lda too small", CAMBLAS_CBLAS_ROW_MAJOR,
                    CAMBLAS_CBLAS_NO_TRANS, CAMBLAS_CBLAS_NO_TRANS, 3, 2, 4, Ad, 3, Bd, 2, Cd, 2,
                    9);
    check_invalid_d("DGEMM row-major ldb too small", CAMBLAS_CBLAS_ROW_MAJOR,
                    CAMBLAS_CBLAS_NO_TRANS, CAMBLAS_CBLAS_NO_TRANS, 3, 2, 4, Ad, 4, Bd, 1, Cd, 2,
                    11);
    check_invalid_d("DGEMM row-major ldc too small", CAMBLAS_CBLAS_ROW_MAJOR,
                    CAMBLAS_CBLAS_NO_TRANS, CAMBLAS_CBLAS_NO_TRANS, 3, 2, 4, Ad, 4, Bd, 2, Cd, 1,
                    14);

    check_ld_boundaries_s(Af, Bf, Cf);
    check_ld_boundaries_d(Ad, Bd, Cd);
    check_zero_output_s(Af, Bf, Cf);
    check_zero_output_d(Ad, Bd, Cd);
    check_extreme_dispatch_boundaries();
    check_valid_s();
    check_valid_d();
    check_thread_local_errors();
    check_default_xerbla();
    camblas_cblas_clear_error();
    CHECK(camblas_cblas_last_error() == CAMBLAS_CBLAS_NO_ERROR,
          "explicit CBLAS error clear returns zero");

    printf("Checks: %d passed, %d failed\n", pass_count, fail_count);
    if (fail_count != 0)
        return 1;
    puts("CBLAS SEMANTICS CHECK PASSED (development ABI 0 is unstable)");
    return 0;
}
