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

static int segmented_mode;

/* Guard independent A/B/product buffers and verify every invalid capacity or
 * alignment is rejected before C changes. Valid results use the same scalar
 * oracle below as the historical contiguous scratch interface. */
static int execute_case(int tb, const camblas_executor_t *executor, int workers, int m, int n,
                        int k, const float *a, int lda, const float *b, int ldb, float *c, int ldc,
                        void *scratch, size_t bytes, int levels)
{
    if (!segmented_mode)
        return camblas_experimental_rectangular32_f32_op_checked(
            tb, executor, workers, m, n, k, a, lda, b, ldb, c, ldc, scratch, bytes, levels);
    camblas_rectangular32_segments_t segments;
    size_t sizes[3];
    assert(!camblas_experimental_rectangular32_segment_bytes(workers, m, n, k, sizes));
    unsigned char *buffers[3];
    for (int t = 0; t < 3; ++t) {
        buffers[t] = malloc(sizes[t] + 128);
        assert(buffers[t]);
        memset(buffers[t], 0x6b, sizes[t] + 128);
        segments.data[t] = buffers[t] + 64;
        segments.bytes[t] = sizes[t];
    }
    size_t c_bytes = (size_t)ldc * n * sizeof(float);
    void *saved = malloc(c_bytes);
    assert(saved);
    memcpy(saved, c, c_bytes);
    for (int t = 0; t < 3; ++t) {
        segments.bytes[t]--;
        assert(camblas_experimental_rectangular32_f32_segments_checked(tb, executor, workers, m, n,
                                                                       k, a, lda, b, ldb, c, ldc,
                                                                       &segments, levels) == -1);
        assert(!memcmp(saved, c, c_bytes));
        segments.bytes[t]++;
        segments.data[t] = buffers[t] + 65;
        assert(camblas_experimental_rectangular32_f32_segments_checked(tb, executor, workers, m, n,
                                                                       k, a, lda, b, ldb, c, ldc,
                                                                       &segments, levels) == -1);
        assert(!memcmp(saved, c, c_bytes));
        segments.data[t] = buffers[t] + 64;
    }
    free(saved);
    int status = camblas_experimental_rectangular32_f32_segments_checked(
        tb, executor, workers, m, n, k, a, lda, b, ldb, c, ldc, &segments, levels);
    for (int t = 0; t < 3; ++t) {
        for (int i = 0; i < 64; ++i)
            assert(buffers[t][i] == 0x6b && buffers[t][sizes[t] + 64 + i] == 0x6b);
        free(buffers[t]);
    }
    return status;
}

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
        assert(!execute_case(tb, &camblas_executor_serial, workers, m, n, k, a, lda, b, ldb, c + 8,
                             ldc, scratch + 64, bytes, 0));
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

static void check_stream_preflight(void)
{
    /* The unsafe entry is in the final depth slab. Earlier private products
     * may have completed, but declining the route must leave all of C intact. */
    enum { M = 256, N = 64, K = 1030 };
    size_t bytes = 0;
    assert(!camblas_experimental_rectangular32_bytes(M, N, K, &bytes));
    float *a = calloc((size_t)M * K, sizeof(*a));
    float *b = calloc((size_t)K * N, sizeof(*b));
    float *c = malloc((size_t)M * N * sizeof(*c));
    unsigned char *scratch = malloc(bytes + 128);
    assert(a && b && c && scratch);
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
                a[late_a] = FLT_MAX / 2;
            else if (scenario == 2)
                b[late_b] = FLT_MAX / 2;
            else if (scenario == 3)
                a[late_a] = NAN;
            else if (scenario == 4)
                b[late_b] = INFINITY;
            int status = execute_case(tb, &camblas_executor_serial, 16, M, N, K, a, M, b,
                                      tb ? N : K, c, M, scratch + 64, bytes, 1);
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
    puts("10 late-depth FP32 range-preflight and untouched-output checks passed");
}

int main(void)
{
    check_range();
    const int shapes[][3] = {{2, 2, 2},      {14, 18, 30},     {50, 34, 66},
                             {256, 96, 512}, {260, 132, 1026}, {258, 1026, 1026}};
    const int workers[] = {1, 16, 64};
    /* Selected application shapes are large; one worker count each keeps the
     * compensated serial oracle affordable while still exercising deep
     * depth-block tails, incomplete twelve-row panels and wide columns. */
    const int application_shapes[][3] = {{4096, 512, 2048}, {1024, 512, 4096}};
    const int application_workers[] = {16};
    double peak = 0;
    long long cases = 0;
    for (segmented_mode = 0; segmented_mode < 2; ++segmented_mode) {
        check_stream_preflight();
        for (int tb = 0; tb < 2; ++tb)
            for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); ++i)
                for (size_t j = 0; j < sizeof(workers) / sizeof(workers[0]); ++j) {
                    peak = fmax(
                        peak, check_case(shapes[i][0], shapes[i][1], shapes[i][2], workers[j], tb));
                    ++cases;
                }
        for (size_t i = 0; i < sizeof(application_shapes) / sizeof(application_shapes[0]); ++i)
            for (size_t j = 0; j < sizeof(application_workers) / sizeof(application_workers[0]);
                 ++j) {
                peak = fmax(peak, check_case(application_shapes[i][0], application_shapes[i][1],
                                             application_shapes[i][2], application_workers[j], 0));
                ++cases;
            }
    }
    size_t sentinel = 123;
    assert(camblas_experimental_rectangular32_bytes(5, 4, 4, &sentinel) == -1 && sentinel == 123);
    assert(camblas_experimental_rectangular32_bytes(4, 0, 4, &sentinel) == -1 && sentinel == 123);
    assert(camblas_experimental_rectangular32_bytes(8196, 4, 4, &sentinel) == -1 &&
           sentinel == 123);
    size_t sizes[3] = {123, 456, 789};
    assert(camblas_experimental_rectangular32_segment_bytes(0, 4, 4, 4, sizes) == -1);
    assert(camblas_experimental_rectangular32_segment_bytes(65, 4, 4, 4, sizes) == -1);
    assert(camblas_experimental_rectangular32_segment_bytes(16, 5, 4, 4, sizes) == -1);
    assert(sizes[0] == 123 && sizes[1] == 456 && sizes[2] == 789);
    assert(camblas_experimental_rectangular32_segment_bytes(16, 4, 4, 4, NULL) == -1);
    printf("%lld contiguous/segmented full-output scalar-oracle cases passed; max "
           "absolute-product-scaled error %.9g\n",
           cases, peak);
    return 0;
}
