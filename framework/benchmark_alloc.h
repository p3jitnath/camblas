#ifndef CAMBLAS_BENCHMARK_ALLOC_H
#define CAMBLAS_BENCHMARK_ALLOC_H
/* Process-local allocation experiment; never changes global THP policy.
 * THP advice is not a guarantee of huge backing. Report observed smaps data. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#ifndef CAMBLAS_BENCH_MEMORY_POLICY
#define CAMBLAS_BENCH_MEMORY_POLICY 0
#endif
#ifndef CAMBLAS_BENCH_WORKSPACE_FACTOR
#define CAMBLAS_BENCH_WORKSPACE_FACTOR 1
#endif
#if CAMBLAS_BENCH_WORKSPACE_FACTOR != 1 && CAMBLAS_BENCH_WORKSPACE_FACTOR != 2 && \
    CAMBLAS_BENCH_WORKSPACE_FACTOR != 4
#error "Workspace factor must be one, two or four"
#endif
#if CAMBLAS_BENCH_MEMORY_POLICY < 0 || CAMBLAS_BENCH_MEMORY_POLICY > 2
#error "Memory policy must be malloc, base64k or thp512m"
#endif
static inline size_t benchmark_reserved_bytes(size_t bytes)
{
    size_t alignment = CAMBLAS_BENCH_MEMORY_POLICY == 1 ? 65536 : 536870912;
    if (!CAMBLAS_BENCH_MEMORY_POLICY || (CAMBLAS_BENCH_MEMORY_POLICY == 2 && bytes < 67108864))
        return bytes;
    if (bytes > SIZE_MAX - (alignment - 1))
        return 0;
    return (bytes + alignment - 1) / alignment * alignment;
}
static inline void *benchmark_alloc(size_t bytes)
{
    if (!CAMBLAS_BENCH_MEMORY_POLICY || (CAMBLAS_BENCH_MEMORY_POLICY == 2 && bytes < 67108864))
        return malloc(bytes);
    size_t reserved = benchmark_reserved_bytes(bytes);
    size_t alignment = CAMBLAS_BENCH_MEMORY_POLICY == 1 ? 65536 : 536870912;
    void *result = NULL;
    if (!reserved || posix_memalign(&result, alignment, reserved))
        return NULL;
    if (madvise(result, reserved,
                CAMBLAS_BENCH_MEMORY_POLICY == 1 ? MADV_NOHUGEPAGE : MADV_HUGEPAGE)) {
        free(result);
        return NULL;
    }
    return result;
}
static inline long long benchmark_anon_huge_kib(void)
{
    FILE *f = fopen("/proc/self/smaps", "r");
    if (!f)
        return -1;
    char line[512];
    long long total = 0, value;
    while (fgets(line, sizeof(line), f))
        if (sscanf(line, "AnonHugePages: %lld kB", &value) == 1)
            total += value;
    int failed = ferror(f);
    fclose(f);
    return failed ? -1 : total;
}
#endif
