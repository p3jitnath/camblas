/* Exercise eligible batched products from concurrent outer OpenMP callers.
 * GEMM owns its existing mutex; nested calls must not create another team. */
#include <dlfcn.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEFINE_CHECK(S, T)                                                                        \
    typedef void (*gemm_##S##_t)(int, int, int, int, int, int, T, const T *, int, const T *, int, \
                                 T, T *, int);                                                    \
    static int check_##S(gemm_##S##_t gemm, int trial, int transpose)                             \
    {                                                                                             \
        enum { N = 512 };                                                                         \
        size_t count = (size_t)N * N;                                                             \
        T *a = malloc(count * sizeof(T)), *b = malloc(count * sizeof(T));                         \
        T *c = malloc(count * sizeof(T));                                                         \
        if (!a || !b || !c)                                                                       \
            abort();                                                                              \
        for (size_t i = 0; i < count; ++i) {                                                      \
            a[i] = 1;                                                                             \
            b[i] = (T)(trial + 1);                                                                \
            c[i] = NAN;                                                                           \
        }                                                                                         \
        gemm(102, 111, transpose ? 112 : 111, N, N, N, 1, a, N, b, N, 0, c, N);                   \
        int failures = 0;                                                                         \
        for (size_t i = 0; i < count; ++i) {                                                      \
            failures += c[i] != N * (trial + 1);                                                  \
            b[i] = 2;                                                                             \
        }                                                                                         \
        /* Nonzero beta must use the ordinary route and preserve changed inputs. */               \
        gemm(102, 111, transpose ? 112 : 111, N, N, N, 1, a, N, b, N, (T)0.5, c, N);              \
        for (size_t i = 0; i < count; ++i)                                                        \
            failures += c[i] != 2 * N + (T)0.5 * N * (trial + 1);                                 \
        free(a);                                                                                  \
        free(b);                                                                                  \
        free(c);                                                                                  \
        return failures;                                                                          \
    }

DEFINE_CHECK(s, float)
DEFINE_CHECK(d, double)

int main(int argc, char **argv)
{
    if (argc != 2)
        return 2;
    void *library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!library)
        return 2;
    void *symbol_s = dlsym(library, "cblas_sgemm");
    void *symbol_d = dlsym(library, "cblas_dgemm");
    if (!symbol_s || !symbol_d)
        return 2;
    gemm_s_t sgemm;
    gemm_d_t dgemm;
    memcpy(&sgemm, &symbol_s, sizeof(sgemm));
    memcpy(&dgemm, &symbol_d, sizeof(dgemm));
    float one = 1, result = 0;
    sgemm(102, 111, 111, 1, 1, 1, 1, &one, 1, &one, 1, 0, &result, 1);
    if (result != 1)
        return 1;
    int failures = 0;
#pragma omp parallel for num_threads(4) reduction(+ : failures) schedule(static)
    for (int trial = 0; trial < 8; ++trial) {
        failures += check_s(sgemm, trial, trial % 2);
        failures += check_d(dgemm, trial, trial % 2);
    }
    dlclose(library);
    if (failures) {
        fprintf(stderr, "Incorrect values: %d\n", failures);
        return 1;
    }
    puts("32 full-output FP32/FP64 products passed from concurrent outer callers");
    return 0;
}
