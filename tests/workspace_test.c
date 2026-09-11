#include "camblas_workspace.h"
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void)
{
    enum { M = 131, N = 137, K = 133 };
    float af[M * K], bf[K * N], cf[M * N];
    double ad[M * K], bd[K * N], cd[M * N];
    for (int i = 0; i < M * K; ++i)
        af[i] = ad[i] = 1;
    camblas_topology_t topo;
    assert(!CAMBLAS_TOPO_IS_FATAL(camblas_topology_discover(&topo)));
    assert(topo.n_allowed_cpus >= 2);
    camblas_ctx_t ctx = {.num_threads = 2,
                         .reproducibility = CAMBLAS_REPRO_FAST,
                         .planner_mode = "shape-aware",
                         .topo = &topo,
                         .executor = &camblas_executor_serial};
    size_t untouched = 123;
    assert(camblas_workspace_bytes(-1, K, CAMBLAS_DTYPE_F64, &untouched) == -1 && untouched == 123);
    assert(camblas_workspace_bytes(N, K, -99, &untouched) == -1 && untouched == 123);
    assert(camblas_workspace_bytes(INT_MAX, INT_MAX, CAMBLAS_DTYPE_F64, &untouched) == -1 &&
           untouched == 123);
    assert(camblas_workspace_bytes(0, K, CAMBLAS_DTYPE_F32, &untouched) == 0 && untouched == 0);
    int cases = 0;
    for (int precision = 0; precision < 2; ++precision) {
        size_t bytes = 0;
        assert(!camblas_workspace_bytes(N, K, precision ? CAMBLAS_DTYPE_F64 : CAMBLAS_DTYPE_F32,
                                        &bytes));
        unsigned char *memory = malloc(bytes + 128);
        assert(memory);
        memset(memory, 0xa5, bytes + 128);
        void *workspace = memory + 64;
        for (int repeat = 1; repeat <= 3; ++repeat) {
            for (int i = 0; i < K * N; ++i)
                bf[i] = bd[i] = repeat;
            for (int i = 0; i < M * N; ++i)
                cf[i] = cd[i] = NAN;
            camblas_plan_t plan;
            int rc = precision ? camblas_dgemm_plan_workspace(&ctx, 'N', 'N', M, N, K, 1, ad, M, bd,
                                                              K, 0, cd, M, &plan, workspace, bytes)
                               : camblas_sgemm_plan_workspace(&ctx, 'N', 'N', M, N, K, 1, af, M, bf,
                                                              K, 0, cf, M, &plan, workspace, bytes);
            assert(!rc && plan.kernel_id == CAMBLAS_KERNEL_PACKED);
            for (int i = 0; i < M * N; ++i)
                assert((precision ? cd[i] : cf[i]) == K * repeat);
            for (int i = 0; i < 64; ++i)
                assert(memory[i] == 0xa5 && memory[bytes + 64 + i] == 0xa5);
            ++cases;
        }
        for (int invalid = 0; invalid < 4; ++invalid) {
            void *ptr = invalid == 0   ? NULL
                        : invalid == 1 ? memory + 65
                        : invalid == 2 ? workspace
                        : precision    ? (void *)cd
                                       : (void *)cf;
            size_t capacity = invalid == 2 ? bytes - 1 : bytes;
            camblas_plan_t plan, saved;
            memset(&plan, 0x5a, sizeof(plan));
            saved = plan;
            for (int i = 0; i < M * N; ++i)
                cf[i] = cd[i] = 7;
            int rc = precision ? camblas_dgemm_plan_workspace(&ctx, 'N', 'N', M, N, K, 1, ad, M, bd,
                                                              K, 0, cd, M, &plan, ptr, capacity)
                               : camblas_sgemm_plan_workspace(&ctx, 'N', 'N', M, N, K, 1, af, M, bf,
                                                              K, 0, cf, M, &plan, ptr, capacity);
            assert(rc == -1 && !memcmp(&plan, &saved, sizeof(plan)));
            for (int i = 0; i < M * N; ++i)
                assert((precision ? cd[i] : cf[i]) == 7);
            ++cases;
        }
        free(memory);
    }
    printf(
        "PASS %d workspace reuse/invalid-input cases, guards, changed operands, untouched C/plan on rejection\n",
        cases);
}
