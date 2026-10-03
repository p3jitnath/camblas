/* Test-only fault injection. Arm after CUDA and classical GEMM are warmed up. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>

static int armed;

void camblas_test_fail_next_allocation(void)
{
    __atomic_store_n(&armed, 1, __ATOMIC_RELEASE);
}

int camblas_test_allocation_pending(void)
{
    return __atomic_load_n(&armed, __ATOMIC_ACQUIRE);
}

/* Match the CUDA runtime ABI without requiring toolkit headers for this shim. */
int cudaMalloc(void **pointer, size_t bytes)
{
    if (__atomic_exchange_n(&armed, 0, __ATOMIC_ACQ_REL)) {
        *pointer = NULL;
        return 2; /* cudaErrorMemoryAllocation */
    }
    int (*allocate)(void **, size_t) = dlsym(RTLD_NEXT, "cudaMalloc");
    return allocate ? allocate(pointer, bytes) : 999; /* cudaErrorUnknown */
}
