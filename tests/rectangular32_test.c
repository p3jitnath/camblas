/* Independent scalar-oracle and boundary checks for the FP32 one-level
 * rectangular prototype. Test data include changed inputs, cancellation, padded
 * strides, partial micro-panels and depth-block tails. This executable is never
 * timed as BLAS. */
#include "rectangular32.h"
#include "rectangular32_range.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t random_state = 812371;

static void check_range(void)
{
    /* Check vector lanes and scalar tails against a long-double bound. */
    const float values[] = {0,        1,           -3,      0x1p-120f, 0x1p60f,
                            0x1p120f, FLT_MAX / 8, FLT_MAX, INFINITY,  NAN};
    float a[35 * 6], b[8 * 6];
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
                                       ma <= FLT_MAX / 4 && mb <= FLT_MAX / 4 &&
                                       ma * mb * 5 * 16 <= (long double)FLT_MAX;
                        int actual = rectangular32_range_safe_op(
                            tb, &camblas_executor_serial, workers[t], 31, 3, 5, a, 35, b, 8, 1);
                        assert(actual == expected);
                        ++cases;
                    }
    printf("%d FP32 finite-range and overflow-preflight checks passed\n", cases);
}
static double sample(void)
{
    /* Generate binary64 samples, rounded once when stored in binary32 operands. */
    random_state = 1664525u * random_state + 1013904223u;
    uint64_t high = random_state >> 5;
    random_state = 1664525u * random_state + 1013904223u;
    uint64_t low = random_state >> 6;
    return (double)(high * 67108864u + low) * 0x1p-52 - 1.0;
}

static double check_case(int m, int n, int k, int workers, int tb)
{
    size_t bytes = 0;
    assert(!camblas_experimental_rectangular32_bytes(m, n, k, &bytes));
    int lda = m + 3, ldb = (tb ? n : k) + 5, ldc = m + 7;
    size_t na = (size_t)lda * k, nb = (size_t)ldb * (tb ? k : n), nc = (size_t)ldc * n;
    float *a = malloc(na * sizeof(float)), *b = malloc(nb * sizeof(float));
    float *c = malloc((nc + 16) * sizeof(float));
    float *saved_a = malloc(na * sizeof(float)), *saved_b = malloc(nb * sizeof(float));
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
        memcpy(saved_a, a, na * sizeof(float));
        memcpy(saved_b, b, nb * sizeof(float));
        assert(!camblas_experimental_rectangular32_f32_op(tb, &camblas_executor_serial, workers, m,
                                                          n, k, a, lda, b, ldb, c + 8, ldc,
                                                          scratch + 64, bytes));
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) {
                double expected = 0, correction = 0, magnitude = 0;
                for (int q = 0; q < k; ++q) {
                    double term = (double)a[i + (size_t)q * lda] *
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
                assert(isfinite(actual) && error < 2e-5);
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
        assert(!memcmp(saved_a, a, na * sizeof(float)));
        assert(!memcmp(saved_b, b, nb * sizeof(float)));
    }
    assert(camblas_experimental_rectangular32_f32(&camblas_executor_serial, workers, m, n, k, a,
                                                  lda, b, ldb, c + 8, ldc, scratch + 64,
                                                  bytes - 1) == -1);
    assert(camblas_experimental_rectangular32_f32(&camblas_executor_serial, workers, m, n, k, a,
                                                  lda, b, ldb, c + 8, ldc, scratch + 65,
                                                  bytes) == -1);
    free(a);
    free(b);
    free(c);
    free(saved_a);
    free(saved_b);
    free(scratch);
    return peak;
}

int main(void)
{
    check_range();
    const int shapes[][3] = {
        {2, 2, 2}, {14, 18, 30}, {50, 34, 66}, {256, 96, 512}, {260, 132, 1026}};
    const int workers[] = {1, 16, 64};
    double peak = 0;
    for (int tb = 0; tb < 2; ++tb)
        for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); ++i)
            for (size_t j = 0; j < sizeof(workers) / sizeof(workers[0]); ++j)
                peak = fmax(peak,
                            check_case(shapes[i][0], shapes[i][1], shapes[i][2], workers[j], tb));
    size_t sentinel = 123;
    assert(camblas_experimental_rectangular32_bytes(5, 4, 4, &sentinel) == -1 && sentinel == 123);
    assert(camblas_experimental_rectangular32_bytes(4, 0, 4, &sentinel) == -1 && sentinel == 123);
    assert(camblas_experimental_rectangular32_bytes(8196, 4, 4, &sentinel) == -1 &&
           sentinel == 123);
    printf("90 full-output scalar-oracle cases passed; max "
           "absolute-product-scaled error %.9g\n",
           peak);
    return 0;
}
