/* Independent scalar-oracle and boundary checks for the FP64 one-level
 * rectangular prototype. Test data include changed inputs, cancellation, padded
 * strides, partial micro-panels and depth-block tails. This executable is never
 * timed as BLAS. */
#include "rectangular64.h"
#include "rectangular64_range.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int execute_case(int ta, int tb, const camblas_executor_t *executor, int workers, int m,
                        int n, int k, const double *a, int lda, const double *b, int ldb, double *c,
                        int ldc, void *scratch, size_t bytes, int levels)
{
    if (ta)
        return camblas_experimental_rectangular64_f64_ops_checked(
            ta, tb, executor, workers, m, n, k, a, lda, b, ldb, c, ldc, scratch, bytes, levels);
    return camblas_experimental_rectangular64_f64_op_checked(
        tb, executor, workers, m, n, k, a, lda, b, ldb, c, ldc, scratch, bytes, levels);
}

static uint32_t random_state = 812371;

static void check_range(void)
{
    /* Check vector lanes and scalar tails against a long-double bound. */
    const double values[] = {0,        1,           -3,      0x1p-1000, 0x1p500,
                             0x1p1000, DBL_MAX / 8, DBL_MAX, INFINITY,  NAN};
    double a[35 * 6], b[8 * 6];
    const int workers[] = {1, 16, 64};
    int cases = 0;
    for (int tb = 0; tb < 2; ++tb)
        for (size_t ai = 0; ai < sizeof(values) / sizeof(*values); ++ai)
            for (size_t bi = 0; bi < sizeof(values) / sizeof(*values); ++bi)
                for (int position = 0; position < 2; ++position)
                    for (size_t t = 0; t < sizeof(workers) / sizeof(*workers); ++t) {
                        memset(a, 0, sizeof(a));
                        memset(b, 0, sizeof(b));
                        a[position ? 30 : 0] = values[ai];
                        b[position ? (tb ? 2 : 4) : 0] = values[bi];
                        long double ma = fabsl((long double)values[ai]);
                        long double mb = fabsl((long double)values[bi]);
                        int expected = isfinite(values[ai]) && isfinite(values[bi]) &&
                                       ma <= DBL_MAX / 4 && mb <= DBL_MAX / 4 &&
                                       ma * mb * 5 * 16 <= (long double)DBL_MAX;
                        int actual = rectangular64_range_safe_op(
                            tb, &camblas_executor_serial, workers[t], 31, 3, 5, a, 35, b, 8, 1);
                        assert(actual == expected);
                        ++cases;
                    }
    printf("%d FP64 finite-range and overflow-preflight checks passed\n", cases);
}
static double sample(void)
{
    /* Two independent LCG outputs provide a full binary64 mantissa. */
    random_state = 1664525u * random_state + 1013904223u;
    uint64_t high = random_state >> 5;
    random_state = 1664525u * random_state + 1013904223u;
    uint64_t low = random_state >> 6;
    return (double)(high * 67108864u + low) * 0x1p-52 - 1.0;
}

static double check_case(int m, int n, int k, int workers, int tb, int ta)
{
    size_t bytes = 0;
    assert(!camblas_experimental_rectangular64_bytes(m, n, k, &bytes));
    int lda = (ta ? k : m) + 3, ldb = (tb ? n : k) + 5, ldc = m + 7;
    size_t na = (size_t)lda * (ta ? m : k), nb = (size_t)ldb * (tb ? k : n), nc = (size_t)ldc * n;
    double *a = malloc(na * sizeof(double)), *b = malloc(nb * sizeof(double));
    double *c = malloc((nc + 16) * sizeof(double));
    double *saved_a = malloc(na * sizeof(double)), *saved_b = malloc(nb * sizeof(double));
    unsigned char *scratch = malloc(bytes + 128);
    assert(a && b && c && saved_a && saved_b && scratch);
    double peak = 0;
    for (int change = 0; change < 3; ++change) {
        for (size_t i = 0; i < na; ++i)
            a[i] = change == 1 ? (i % 2 ? -sample() : sample()) : sample();
        for (size_t i = 0; i < nb; ++i)
            b[i] = change == 2 ? sample() * 128.0f : sample();
        for (size_t i = 0; i < nc + 16; ++i)
            c[i] = -731.0f;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i)
                c[8 + i + (size_t)j * ldc] = NAN;
        memset(scratch, 0x6b, bytes + 128);
        memcpy(saved_a, a, na * sizeof(double));
        memcpy(saved_b, b, nb * sizeof(double));
        assert(!execute_case(ta, tb, &camblas_executor_serial, workers, m, n, k, a, lda, b, ldb,
                             c + 8, ldc, scratch + 64, bytes, 0));
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) {
                double expected = 0, correction = 0, magnitude = 0;
                for (int q = 0; q < k; ++q) {
                    double term = (double)a[ta ? q + (size_t)i * lda : i + (size_t)q * lda] *
                                  b[tb ? j + (size_t)q * ldb : q + (size_t)j * ldb];
                    /* Compensated scalar accumulation is independent of BLAS. */
                    double adjusted = term - correction;
                    double next = expected + adjusted;
                    correction = (next - expected) - adjusted;
                    expected = next;
                    magnitude += fabs(term);
                }
                double actual = c[8 + i + (size_t)j * ldc];
                double error = fabs(actual - expected) / fmax(1.0, magnitude);
                assert(isfinite(actual) && error < 1e-13);
                if (error > peak)
                    peak = error;
            }
        for (int j = 0; j < n; ++j)
            for (int i = m; i < ldc; ++i)
                assert(c[8 + i + (size_t)j * ldc] == -731.0f);
        for (int i = 0; i < 8; ++i)
            assert(c[i] == -731.0f && c[nc + 8 + i] == -731.0f);
        for (int i = 0; i < 64; ++i)
            assert(scratch[i] == 0x6b && scratch[bytes + 64 + i] == 0x6b);
        assert(!memcmp(saved_a, a, na * sizeof(double)));
        assert(!memcmp(saved_b, b, nb * sizeof(double)));
    }
    assert(execute_case(ta, tb, &camblas_executor_serial, workers, m, n, k, a, lda, b, ldb, c + 8,
                        ldc, scratch + 64, bytes - 1, 0) == -1);
    assert(execute_case(ta, tb, &camblas_executor_serial, workers, m, n, k, a, lda, b, ldb, c + 8,
                        ldc, scratch + 65, bytes, 0) == -1);
    free(a);
    free(b);
    free(c);
    free(saved_a);
    free(saved_b);
    free(scratch);
    return peak;
}

static void check_stream_preflight(void)
{
    /* The unsafe entry is in the final depth slab. Earlier private products
     * may have completed, but declining the route must leave all of C intact. */
    enum { M = 256, N = 64, K = 1030 };
    size_t bytes = 0;
    assert(!camblas_experimental_rectangular64_bytes(M, N, K, &bytes));
    double *a = calloc((size_t)M * K, sizeof(*a));
    double *b = calloc((size_t)K * N, sizeof(*b));
    double *c = malloc((size_t)M * N * sizeof(*c));
    unsigned char *scratch = malloc(bytes + 128);
    assert(a && b && c && scratch);
    for (int ta = 0; ta < 2; ++ta)
        for (int tb = 0; tb < 2; ++tb)
            for (int scenario = 0; scenario < 5; ++scenario) {
                memset(a, 0, (size_t)M * K * sizeof(*a));
                memset(b, 0, (size_t)K * N * sizeof(*b));
                memset(scratch, 0x6b, bytes + 128);
                for (int i = 0; i < M * N; ++i)
                    c[i] = -731;
                size_t late_a = (size_t)M * K - 1;
                size_t late_b = (size_t)K * N - 1;
                if (scenario == 1)
                    a[late_a] = DBL_MAX / 2;
                else if (scenario == 2)
                    b[late_b] = DBL_MAX / 2;
                else if (scenario == 3)
                    a[late_a] = NAN;
                else if (scenario == 4)
                    b[late_b] = INFINITY;
                int status = execute_case(ta, tb, &camblas_executor_serial, 16, M, N, K, a,
                                          ta ? K : M, b, tb ? N : K, c, M, scratch + 64, bytes, 1);
                assert(status == (scenario != 0));
                for (int i = 0; i < M * N; ++i)
                    assert(c[i] == (scenario ? -731 : 0));
                for (int i = 0; i < 64; ++i)
                    assert(scratch[i] == 0x6b && scratch[bytes + 64 + i] == 0x6b);
            }
    free(a);
    free(b);
    free(c);
    free(scratch);
    puts("20 late-depth FP64 range-preflight and untouched-output checks passed");
}

int main(void)
{
    check_range();
    check_stream_preflight();
    const int shapes[][3] = {{2, 2, 2},      {14, 18, 30},     {50, 34, 66},
                             {256, 96, 512}, {260, 132, 1026}, {258, 1026, 1026}};
    const int workers[] = {1, 16, 64};
    /* Selected application shapes are large; one worker count each keeps the
     * compensated serial oracle affordable while still exercising deep
     * depth-block tails, incomplete twelve-row panels and wide columns. */
    const int application_shapes[][3] = {{4096, 512, 2048}, {1024, 512, 4096}};
    const int application_workers[] = {16};
    const int square_depth_shapes[][3] = {{512, 512, 514}, {514, 514, 1026}};
    double peak = 0;
    long long cases = 0;
    for (int ta = 0; ta < 2; ++ta) {
        for (int tb = 0; tb < 2; ++tb)
            for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); ++i)
                for (size_t j = 0; j < sizeof(workers) / sizeof(workers[0]); ++j) {
                    peak = fmax(peak, check_case(shapes[i][0], shapes[i][1], shapes[i][2],
                                                 workers[j], tb, ta));
                    ++cases;
                }
        for (size_t i = 0; i < sizeof(square_depth_shapes) / sizeof(square_depth_shapes[0]); ++i)
            for (int tb = 0; tb < 2; ++tb) {
                peak = fmax(peak, check_case(square_depth_shapes[i][0], square_depth_shapes[i][1],
                                             square_depth_shapes[i][2], 64, tb, ta));
                ++cases;
            }
    }
    for (size_t i = 0; i < sizeof(application_shapes) / sizeof(application_shapes[0]); ++i)
        for (size_t j = 0; j < sizeof(application_workers) / sizeof(application_workers[0]); ++j) {
            peak = fmax(peak, check_case(application_shapes[i][0], application_shapes[i][1],
                                         application_shapes[i][2], application_workers[j], 0, 0));
            ++cases;
        }
    const int short_transposed_shapes[][3] = {{1024, 1024, 256}, {1032, 1032, 258}};
    for (size_t i = 0; i < sizeof(short_transposed_shapes) / sizeof(short_transposed_shapes[0]);
         ++i)
        for (int tb = 0; tb < 2; ++tb) {
            peak =
                fmax(peak, check_case(short_transposed_shapes[i][0], short_transposed_shapes[i][1],
                                      short_transposed_shapes[i][2], 16, tb, 1));
            ++cases;
        }
    size_t sentinel = 123;
    assert(camblas_experimental_rectangular64_bytes(5, 4, 4, &sentinel) == -1 && sentinel == 123);
    assert(camblas_experimental_rectangular64_bytes(4, 0, 4, &sentinel) == -1 && sentinel == 123);
    assert(camblas_experimental_rectangular64_bytes(8196, 4, 4, &sentinel) == -1 &&
           sentinel == 123);
    printf("%lld full-output scalar-oracle cases passed; max "
           "absolute-product-scaled error %.9g\n",
           cases, peak);
    return 0;
}
