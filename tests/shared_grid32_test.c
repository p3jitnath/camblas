/* Independent scalar checks for the bounded 64-thread FP32 packed grid.
 * Exercise partial depth blocks, four/eight-column tails, uneven row groups,
 * transposed B, padding, misalignment, beta-zero NaNs and changed inputs. */
#include <dlfcn.h>
#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void (*gemm_t)(int, int, int, int, int, int, float, const float *, int, const float *, int,
                       float, float *, int);
typedef uint64_t (*counter_t)(void);
typedef void (*reset_t)(void);
typedef int (*stats_t)(uint64_t *, size_t);

static int transpose_a;
static stats_t route_stats;

static uint64_t rectangular_counter(void)
{
    uint64_t stats[12];
    if (route_stats(stats, 12) != 12)
        abort();
    return stats[4];
}

static int check(gemm_t gemm, counter_t counter, reset_t reset, int m, int n, int k, int transpose,
                 int eligible, int trial)
{
    int lda = (transpose_a ? k : m) + 3, ldb = (transpose ? n : k) + 5, ldc = m + 7;
    size_t a_count = (size_t)lda * (transpose_a ? m : k);
    size_t b_count = (size_t)ldb * (transpose ? k : n);
    size_t c_count = (size_t)ldc * n;
    float *a_base = malloc((a_count + 8) * sizeof(float));
    float *b_base = malloc((b_count + 8) * sizeof(float));
    float *c_base = malloc((c_count + 8) * sizeof(float));
    if (!a_base || !b_base || !c_base)
        abort();
    float *a = a_base + 1, *b = b_base + 3, *c = c_base + 2;
    for (size_t i = 0; i < a_count + 8; ++i)
        a_base[i] = NAN;
    for (size_t i = 0; i < b_count + 8; ++i)
        b_base[i] = NAN;
    for (size_t i = 0; i < c_count + 8; ++i)
        c_base[i] = -731.25f;
    for (int p = 0; p < k; ++p)
        for (int row = 0; row < m; ++row)
            a[transpose_a ? p + (size_t)row * lda : row + (size_t)p * lda] =
                (float)(((row * 17 + p * 31 + trial * 7) % 79) - 39) / 37.0f;
    for (int col = 0; col < n; ++col)
        for (int p = 0; p < k; ++p) {
            size_t index = transpose ? col + (size_t)p * ldb : p + (size_t)col * ldb;
            b[index] = (float)(((p * 13 + col * 23 + trial * 11) % 83) - 41) / 43.0f;
        }
    for (int col = 0; col < n; ++col)
        for (int row = 0; row < m; ++row)
            c[row + (size_t)col * ldc] = NAN;
    reset();
    gemm(102, transpose_a ? 112 : 111, transpose ? 112 : 111, m, n, k, 1.0f, a, lda, b, ldb, 0.0f,
         c, ldc);
    int failures = counter() != (uint64_t)eligible;
    if (failures)
        fprintf(stderr, "Unexpected route for %dx%dx%d: %llu\n", m, n, k,
                (unsigned long long)counter());
#pragma omp parallel for num_threads(16) reduction(+ : failures) schedule(static)
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < m; ++row) {
            double sum = 0.0, correction = 0.0, magnitude = 0.0;
            for (int p = 0; p < k; ++p) {
                size_t index = transpose ? col + (size_t)p * ldb : p + (size_t)col * ldb;
                double product =
                    (double)a[transpose_a ? p + (size_t)row * lda : row + (size_t)p * lda] *
                    b[index];
                double corrected = product - correction;
                double next = sum + corrected;
                correction = (next - sum) - corrected;
                sum = next;
                magnitude += fabs(product);
            }
            double actual = c[row + (size_t)col * ldc];
            failures +=
                !isfinite(actual) || fabs(actual - sum) > 32.0 * FLT_EPSILON * (magnitude + 1.0);
        }
        for (int row = m; row < ldc; ++row)
            failures += c[row + (size_t)col * ldc] != -731.25f;
    }
    for (int i = 0; i < 2; ++i)
        failures += c_base[i] != -731.25f;
    for (size_t i = c_count + 2; i < c_count + 8; ++i)
        failures += c_base[i] != -731.25f;
    free(a_base);
    free(b_base);
    free(c_base);
    if (failures)
        fprintf(stderr, "Incorrect values or padding: %dx%dx%d TB=%d trial=%d: %d\n", m, n, k,
                transpose, trial, failures);
    return failures;
}

int shared_grid32_check(const char *library_path, int deep)
{
    transpose_a = deep == 2;
    void *library = dlopen(library_path, RTLD_NOW | RTLD_LOCAL);
    if (!library) {
        fprintf(stderr, "%s\n", dlerror());
        return 2;
    }
    void *g = dlsym(library, "cblas_sgemm");
    void *s =
        dlsym(library, transpose_a ? "framework_blas_stats" : "framework_blas_shared_grid32_calls");
    void *r = dlsym(library, "framework_blas_reset_stats");
    if (!g || !s || !r)
        return 2;
    gemm_t gemm;
    counter_t counter;
    reset_t reset;
    memcpy(&gemm, &g, sizeof(gemm));
    if (transpose_a) {
        memcpy(&route_stats, &s, sizeof(route_stats));
        counter = rectangular_counter;
    } else {
        memcpy(&counter, &s, sizeof(counter));
    }
    memcpy(&reset, &r, sizeof(reset));
    if (transpose_a) {
        const int shapes[][3] = {{512, 512, 256}, {528, 528, 260}, {1032, 1032, 260}};
        int failures = 0;
        for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); ++i)
            for (int trial = 0; trial < 2; ++trial)
                failures += check(gemm, counter, reset, shapes[i][0], shapes[i][1], shapes[i][2], 0,
                                  1, trial);
        failures += check(gemm, counter, reset, 512, 512, 258, 0, 0, 0);
        failures += check(gemm, counter, reset, 514, 514, 260, 0, 0, 0);
        dlclose(library);
        if (failures)
            return 1;
        puts(
            "8 complete transposed-A scalar-oracle FP32 products passed with route, tails and padding checks");
        return 0;
    }
    if (deep == 3) {
        /* NN squares exercise the eight-row worker grid, all output-column
         * tails, and the existing grid when the column split is ineligible. */
        const int dimensions[] = {128, 256, 512, 528, 544, 576, 1024};
        int failures = 0;
        for (size_t i = 0; i < sizeof(dimensions) / sizeof(dimensions[0]); ++i)
            for (int trial = 0; trial < 2; ++trial)
                failures += check(gemm, counter, reset, dimensions[i], dimensions[i], dimensions[i],
                                  0, 1, trial);
        dlclose(library);
        if (failures)
            return 1;
        puts(
            "14 complete NN square scalar-oracle products passed with grid, tails, changed inputs and padding checks");
        return 0;
    }
    if (deep == 1) {
        const int shapes[][3] = {{264, 128, 1025}, {520, 256, 1537}, {1032, 512, 2049}};
        int failures = 0;
        for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); ++i)
            for (int trial = 0; trial < 2; ++trial)
                failures += check(gemm, counter, reset, shapes[i][0], shapes[i][1], shapes[i][2], 0,
                                  1, trial);
        dlclose(library);
        if (failures)
            return 1;
        puts(
            "6 complete deep scalar-oracle FP32 products passed with depth-panel and padding checks");
        return 0;
    }
    const int shapes[][3] = {
        {256, 128, 129}, {264, 256, 257}, {512, 256, 513}, {1024, 512, 1024}, {2048, 1024, 512}};
    int failures = 0, products = 0;
    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); ++i)
        for (int transpose = 0; transpose < 2; ++transpose) {
            failures += check(gemm, counter, reset, shapes[i][0], shapes[i][1], shapes[i][2],
                              transpose, 1, 0);
            ++products;
            if (i < 3) {
                failures += check(gemm, counter, reset, shapes[i][0], shapes[i][1], shapes[i][2],
                                  transpose, 1, 1);
                ++products;
            }
        }
    failures += check(gemm, counter, reset, 263, 256, 257, 0, 0, 0);
    failures += check(gemm, counter, reset, 256, 512, 257, 0, 0, 0);
    products += 2;
    dlclose(library);
    if (failures)
        return 1;
    printf("%d complete scalar-oracle FP32 products passed with route, tail and padding checks\n",
           products);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc == 2)
        return shared_grid32_check(argv[1], 0);
    if (argc == 3 && !strcmp(argv[2], "--deep"))
        return shared_grid32_check(argv[1], 1);
    if (argc == 3 && !strcmp(argv[2], "--square"))
        return shared_grid32_check(argv[1], 3);
    if (argc == 3 && !strcmp(argv[2], "--transpose-a"))
        return shared_grid32_check(argv[1], 2);
    return 2;
}
