/*
 * CAMBLAS planner unit tests.
 *
 * Tests:
 *   1. NULL handling (op, ctx, plan, topo)
 *   2. Invalid operation descriptors (negative dims, bad trans, bad dtype, bad batch)
 *   3. Planner mode "none" (always scalar, 1 thread)
 *   4. Planner mode "fixed" (scalar, clamped threads)
 *   5. Planner mode "shape-aware" (decision table: batch, 1-cpu, zero-dim, small, medium, large)
 *   6. Bitwise reproducibility override
 *   7. Thread clamping (requested > n_cpus, requested < 1)
 *   8. Block sizes (F32 vs F64 for packed path)
 *   9. Format function (valid, NULL, buffer too small)
 *  10. Name functions (valid and invalid IDs)
 *  11. Unknown planner mode (fail closed)
 *  12. Overflow handling (very large dimensions)
 *  13. Reentrancy (4 pthreads, concurrent planning)
 *  14. Topology NULL for shape-aware (fail closed)
 *  15. Topology fatal (n_allowed_cpus < 1) for shape-aware
 *  16. Ablation mode (same decision table as shape-aware)
 *  17. Transpose variants (all 4 combinations)
 *  18. Rationale function
 */
#include "camblas_planner.h"
#include "camblas_topology.h"

#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>

static int tests_run = 0;
static int tests_pass = 0;

#define CHECK(cond)                                            \
    do {                                                       \
        if (!(cond)) {                                         \
            printf("  FAIL: %s (line %d)\n", #cond, __LINE__); \
            return 1;                                          \
        }                                                      \
    } while (0)

/* ---- Helpers ---- */

static void set_allowed_mask(camblas_topology_t *topo, int n_cpus)
{
    camblas_cpuset_zero(&topo->allowed_cpus);
    topo->n_allowed_cpus = n_cpus;
    for (int i = 0; i < n_cpus && i < CAMBLAS_MAX_CPUS; i++)
        camblas_cpuset_set(&topo->allowed_cpus, i);
}

/* Build a synthetic topology with n_cpus CPUs. */
static camblas_topology_t make_topo(int n_cpus)
{
    camblas_topology_t topo;
    memset(&topo, 0, sizeof(topo));
    set_allowed_mask(&topo, n_cpus);
    topo.n_online_cpus = n_cpus;
    topo.sve_vl_bits = 128;
    return topo;
}

static camblas_op_t make_op(int m, int n, int k, int dtype)
{
    camblas_op_t op;
    op.m = m;
    op.n = n;
    op.k = k;
    op.trans_a = 'N';
    op.trans_b = 'N';
    op.dtype = dtype;
    op.batch_count = 1;
    return op;
}

static camblas_ctx_t make_ctx(const char *mode, int threads, int repro)
{
    camblas_ctx_t ctx = {0}; /* zero-init so new fields (executor) are NULL */
    ctx.num_threads = threads;
    ctx.reproducibility = repro;
    ctx.planner_mode = mode;
    ctx.topo = NULL; /* planner tests pass topology separately to camblas_plan_make */
    return ctx;
}

#define RUN_TEST(fn)                 \
    do {                             \
        tests_run++;                 \
        printf("--- %s ---\n", #fn); \
        if (fn() == 0) {             \
            tests_pass++;            \
            printf("  PASS\n\n");    \
        } else {                     \
            printf("  FAIL\n\n");    \
        }                            \
    } while (0)

/* ---- Test functions (return 0=pass, 1=fail) ---- */

static int test_null_args(void)
{
    camblas_op_t op = make_op(64, 64, 64, CAMBLAS_DTYPE_F32);
    camblas_topology_t topo = make_topo(8);
    camblas_ctx_t ctx = make_ctx("shape-aware", 4, CAMBLAS_REPRO_FAST);
    camblas_plan_t plan;

    CHECK(camblas_plan_make(NULL, &topo, &ctx, &plan) == -1);
    CHECK(camblas_plan_make(&op, &topo, NULL, &plan) == -1);
    CHECK(camblas_plan_make(&op, &topo, &ctx, NULL) == -1);

    /* topo NULL is OK for "none" and "fixed" */
    ctx.planner_mode = "none";
    CHECK(camblas_plan_make(&op, NULL, &ctx, &plan) == 0);
    CHECK(plan.kernel_id == CAMBLAS_KERNEL_SCALAR);

    ctx.planner_mode = "fixed";
    CHECK(camblas_plan_make(&op, NULL, &ctx, &plan) == 0);
    CHECK(plan.kernel_id == CAMBLAS_KERNEL_SCALAR);

    /* topo NULL is NOT OK for "shape-aware" */
    ctx.planner_mode = "shape-aware";
    CHECK(camblas_plan_make(&op, NULL, &ctx, &plan) == -1);
    return 0;
}

static int test_invalid_op(void)
{
    camblas_topology_t topo = make_topo(8);
    camblas_ctx_t ctx = make_ctx("shape-aware", 4, CAMBLAS_REPRO_FAST);
    camblas_plan_t plan;

    /* Negative dimensions */
    camblas_op_t op = make_op(-1, 64, 64, CAMBLAS_DTYPE_F32);
    CHECK(camblas_plan_make(&op, &topo, &ctx, &plan) == -1);

    op = make_op(64, -1, 64, CAMBLAS_DTYPE_F32);
    CHECK(camblas_plan_make(&op, &topo, &ctx, &plan) == -1);

    op = make_op(64, 64, -1, CAMBLAS_DTYPE_F32);
    CHECK(camblas_plan_make(&op, &topo, &ctx, &plan) == -1);

    /* Invalid transpose */
    op = make_op(64, 64, 64, CAMBLAS_DTYPE_F32);
    op.trans_a = 'X';
    CHECK(camblas_plan_make(&op, &topo, &ctx, &plan) == -1);

    op.trans_a = 'N';
    op.trans_b = 'X';
    CHECK(camblas_plan_make(&op, &topo, &ctx, &plan) == -1);

    /* Invalid dtype */
    op.trans_b = 'N';
    op.dtype = 99;
    CHECK(camblas_plan_make(&op, &topo, &ctx, &plan) == -1);

    /* Invalid batch_count */
    op.dtype = CAMBLAS_DTYPE_F32;
    op.batch_count = 0;
    CHECK(camblas_plan_make(&op, &topo, &ctx, &plan) == -1);

    op.batch_count = -1;
    CHECK(camblas_plan_make(&op, &topo, &ctx, &plan) == -1);
    return 0;
}

static int test_mode_none(void)
{
    camblas_topology_t topo = make_topo(8);
    camblas_op_t op = make_op(1024, 1024, 1024, CAMBLAS_DTYPE_F32);
    camblas_ctx_t ctx = make_ctx("none", 8, CAMBLAS_REPRO_FAST);
    camblas_plan_t plan;

    int rc = camblas_plan_make(&op, &topo, &ctx, &plan);
    CHECK(rc == 0);
    CHECK(plan.kernel_id == CAMBLAS_KERNEL_SCALAR);
    CHECK(plan.pack == CAMBLAS_PACK_NONE);
    CHECK(plan.num_threads == 1);
    CHECK(plan.parallel_scheme == CAMBLAS_PARALLEL_NONE);
    CHECK(plan.mc == 0 && plan.nc == 0 && plan.kc == 0);
    CHECK(plan.mr == 0 && plan.nr == 0);
    CHECK(plan.rationale_id == CAMBLAS_RAT_NONE_DISABLED);

    /* "none" mode ignores the shape — always scalar 1-thread */
    op = make_op(4, 4, 4, CAMBLAS_DTYPE_F32);
    rc = camblas_plan_make(&op, &topo, &ctx, &plan);
    CHECK(rc == 0);
    CHECK(plan.kernel_id == CAMBLAS_KERNEL_SCALAR);
    CHECK(plan.num_threads == 1);

    /* NULL mode is treated as "none" */
    ctx.planner_mode = NULL;
    rc = camblas_plan_make(&op, &topo, &ctx, &plan);
    CHECK(rc == 0);
    CHECK(plan.kernel_id == CAMBLAS_KERNEL_SCALAR);
    CHECK(plan.num_threads == 1);
    return 0;
}

static int test_mode_fixed(void)
{
    camblas_topology_t topo = make_topo(8);
    camblas_op_t op = make_op(64, 64, 64, CAMBLAS_DTYPE_F32);
    camblas_ctx_t ctx = make_ctx("fixed", 4, CAMBLAS_REPRO_FAST);
    camblas_plan_t plan;

    int rc = camblas_plan_make(&op, &topo, &ctx, &plan);
    CHECK(rc == 0);
    CHECK(plan.kernel_id == CAMBLAS_KERNEL_SCALAR);
    CHECK(plan.pack == CAMBLAS_PACK_NONE);
    CHECK(plan.num_threads == 4);
    CHECK(plan.parallel_scheme == CAMBLAS_PARALLEL_ROW);
    CHECK(plan.rationale_id == CAMBLAS_RAT_FIXED_POLICY);

    /* Requested > n_cpus → clamped */
    ctx.num_threads = 16;
    rc = camblas_plan_make(&op, &topo, &ctx, &plan);
    CHECK(rc == 0);
    CHECK(plan.num_threads == 8);

    /* Requested < 1 → treated as 1 */
    ctx.num_threads = 0;
    rc = camblas_plan_make(&op, &topo, &ctx, &plan);
    CHECK(rc == 0);
    CHECK(plan.num_threads == 1);
    CHECK(plan.parallel_scheme == CAMBLAS_PARALLEL_NONE);

    /* Requested = 1 → single-threaded */
    ctx.num_threads = 1;
    rc = camblas_plan_make(&op, &topo, &ctx, &plan);
    CHECK(rc == 0);
    CHECK(plan.num_threads == 1);
    CHECK(plan.parallel_scheme == CAMBLAS_PARALLEL_NONE);
    return 0;
}

static int test_thread_request_bounds(void)
{
    camblas_topology_t topo = make_topo(8);
    camblas_op_t medium = make_op(32, 32, 32, CAMBLAS_DTYPE_F32);
    camblas_op_t large = make_op(256, 256, 256, CAMBLAS_DTYPE_F32);
    camblas_ctx_t ctx = make_ctx("fixed", 0, CAMBLAS_REPRO_FAST);
    camblas_plan_t plan;
    const int low_requests[] = {0, -1, INT_MIN};

    /* Values below one are a valid request that normalizes to one, including
       the integer extreme.  This is the fixed-policy path used by GEMM. */
    for (size_t i = 0; i < sizeof(low_requests) / sizeof(low_requests[0]); i++) {
        ctx.num_threads = low_requests[i];
        CHECK(camblas_plan_make(&medium, &topo, &ctx, &plan) == 0);
        CHECK(plan.num_threads == 1);
        CHECK(plan.parallel_scheme == CAMBLAS_PARALLEL_NONE);
    }

    /* Positive requests are capped by the selected policy and topology;
       INT_MAX must not overflow or escape the effective-count contract. */
    ctx.num_threads = INT_MAX;
    CHECK(camblas_plan_make(&medium, &topo, &ctx, &plan) == 0);
    CHECK(plan.num_threads == 8); /* fixed-mode topology cap */
    CHECK(plan.parallel_scheme == CAMBLAS_PARALLEL_ROW);
    CHECK(camblas_plan_make(&large, &topo, &ctx, &plan) == 0);
    CHECK(plan.num_threads == 8); /* topology cap */
    CHECK(plan.parallel_scheme == CAMBLAS_PARALLEL_ROW);

    /* Shape-aware planning uses the same request normalization on its
       topology-dependent large path. */
    ctx.planner_mode = "shape-aware";
    for (size_t i = 0; i < sizeof(low_requests) / sizeof(low_requests[0]); i++) {
        ctx.num_threads = low_requests[i];
        CHECK(camblas_plan_make(&large, &topo, &ctx, &plan) == 0);
        CHECK(plan.num_threads == 1);
        CHECK(plan.parallel_scheme == CAMBLAS_PARALLEL_NONE);
    }
    ctx.num_threads = INT_MAX;
    CHECK(camblas_plan_make(&large, &topo, &ctx, &plan) == 0);
    CHECK(plan.num_threads == 8);
    CHECK(plan.parallel_scheme == CAMBLAS_PARALLEL_2D);

    /* Serial-by-policy modes intentionally report one regardless of the
       request; this also covers no-topology public default-style planning. */
    ctx.planner_mode = "none";
    ctx.topo = NULL;
    ctx.num_threads = INT_MIN;
    CHECK(camblas_plan_make(&large, NULL, &ctx, &plan) == 0);
    CHECK(plan.num_threads == 1);
    ctx.num_threads = INT_MAX;
    CHECK(camblas_plan_make(&large, NULL, &ctx, &plan) == 0);
    CHECK(plan.num_threads == 1);
    return 0;
}

static int test_shape_aware_small(void)
{
    camblas_topology_t topo = make_topo(8);
    camblas_ctx_t ctx = make_ctx("shape-aware", 4, CAMBLAS_REPRO_FAST);
    camblas_plan_t plan;

    /* 4x4x4: 128 FLOPs, well below SMALL_CUTOFF (8192) */
    camblas_op_t op = make_op(4, 4, 4, CAMBLAS_DTYPE_F32);
    int rc = camblas_plan_make(&op, &topo, &ctx, &plan);
    CHECK(rc == 0);
    CHECK(plan.kernel_id == CAMBLAS_KERNEL_SCALAR);
    CHECK(plan.num_threads == 1);
    CHECK(plan.parallel_scheme == CAMBLAS_PARALLEL_NONE);
    CHECK(plan.rationale_id == CAMBLAS_RAT_SMALL_SCALAR);

    /* 16x16x16: 8192 FLOPs, exactly at SMALL_CUTOFF → medium */
    op = make_op(16, 16, 16, CAMBLAS_DTYPE_F32);
    rc = camblas_plan_make(&op, &topo, &ctx, &plan);
    CHECK(rc == 0);
    CHECK(plan.kernel_id == CAMBLAS_KERNEL_NOPACK);
    CHECK(plan.rationale_id == CAMBLAS_RAT_MEDIUM_NOPACK);

    /* Zero-dimension: scalar 1-thread */
    op = make_op(0, 64, 64, CAMBLAS_DTYPE_F32);
    rc = camblas_plan_make(&op, &topo, &ctx, &plan);
    CHECK(rc == 0);
    CHECK(plan.kernel_id == CAMBLAS_KERNEL_SCALAR);
    CHECK(plan.num_threads == 1);
    CHECK(plan.rationale_id == CAMBLAS_RAT_SMALL_SCALAR);

    op = make_op(64, 0, 64, CAMBLAS_DTYPE_F32);
    rc = camblas_plan_make(&op, &topo, &ctx, &plan);
    CHECK(rc == 0);
    CHECK(plan.kernel_id == CAMBLAS_KERNEL_SCALAR);

    op = make_op(64, 64, 0, CAMBLAS_DTYPE_F32);
    rc = camblas_plan_make(&op, &topo, &ctx, &plan);
    CHECK(rc == 0);
    CHECK(plan.kernel_id == CAMBLAS_KERNEL_SCALAR);
    return 0;
}

static int test_shape_aware_medium(void)
{
    camblas_topology_t topo = make_topo(8);
    camblas_ctx_t ctx = make_ctx("shape-aware", 4, CAMBLAS_REPRO_FAST);
    camblas_plan_t plan;

    /* 32x32x32: 65536 FLOPs, medium range */
    camblas_op_t op = make_op(32, 32, 32, CAMBLAS_DTYPE_F32);
    int rc = camblas_plan_make(&op, &topo, &ctx, &plan);
    CHECK(rc == 0);
    CHECK(plan.kernel_id == CAMBLAS_KERNEL_NOPACK);
    CHECK(plan.pack == CAMBLAS_PACK_NONE);
    CHECK(plan.num_threads == 4);
    CHECK(plan.parallel_scheme == CAMBLAS_PARALLEL_ROW);
    CHECK(plan.rationale_id == CAMBLAS_RAT_MEDIUM_NOPACK);

    /* Medium with more threads requested → capped at MEDIUM_MAX_THREADS (4) */
    ctx.num_threads = 8;
    rc = camblas_plan_make(&op, &topo, &ctx, &plan);
    CHECK(rc == 0);
    CHECK(plan.num_threads == 4);

    /* Medium with fewer threads requested → uses requested */
    ctx.num_threads = 2;
    rc = camblas_plan_make(&op, &topo, &ctx, &plan);
    CHECK(rc == 0);
    CHECK(plan.num_threads == 2);
    return 0;
}

static int test_shape_aware_large(void)
{
    camblas_topology_t topo = make_topo(8);
    camblas_ctx_t ctx = make_ctx("shape-aware", 8, CAMBLAS_REPRO_FAST);
    camblas_plan_t plan;

    /* 256x256x256: ~33M FLOPs, large */
    camblas_op_t op = make_op(256, 256, 256, CAMBLAS_DTYPE_F32);
    int rc = camblas_plan_make(&op, &topo, &ctx, &plan);
    CHECK(rc == 0);
    CHECK(plan.kernel_id == CAMBLAS_KERNEL_PACKED);
    CHECK(plan.pack == CAMBLAS_PACK_BOTH);
    CHECK(plan.num_threads == 8);
    CHECK(plan.parallel_scheme == CAMBLAS_PARALLEL_2D);
    CHECK(plan.mc == 128 && plan.nc == 256 && plan.kc == 128);
    CHECK(plan.mr == 8 && plan.nr == 8); /* F32, two VL rows */
    CHECK(plan.rationale_id == CAMBLAS_RAT_LARGE_PACKED);

    /* Large DGEMM: different register tile */
    op = make_op(256, 256, 256, CAMBLAS_DTYPE_F64);
    rc = camblas_plan_make(&op, &topo, &ctx, &plan);
    CHECK(rc == 0);
    CHECK(plan.kernel_id == CAMBLAS_KERNEL_PACKED);
    CHECK(plan.mr == 4 && plan.nr == 8); /* F64, two VL rows */

    /* Large with 1 thread → parallel none */
    ctx.num_threads = 1;
    rc = camblas_plan_make(&op, &topo, &ctx, &plan);
    CHECK(rc == 0);
    CHECK(plan.num_threads == 1);
    CHECK(plan.parallel_scheme == CAMBLAS_PARALLEL_NONE);
    return 0;
}

static int test_shape_aware_batch(void)
{
    camblas_topology_t topo = make_topo(8);
    camblas_ctx_t ctx = make_ctx("shape-aware", 4, CAMBLAS_REPRO_FAST);
    camblas_plan_t plan;

    /* Batched GEMM: batch_count > 1 → batched kernel */
    camblas_op_t op = make_op(8, 8, 8, CAMBLAS_DTYPE_F32);
    op.batch_count = 16;
    int rc = camblas_plan_make(&op, &topo, &ctx, &plan);
    CHECK(rc == 0);
    CHECK(plan.kernel_id == CAMBLAS_KERNEL_BATCHED);
    CHECK(plan.num_threads == 4);
    CHECK(plan.parallel_scheme == CAMBLAS_PARALLEL_2D);
    CHECK(plan.rationale_id == CAMBLAS_RAT_BATCH);

    /* Batch with 1 thread → parallel none */
    ctx.num_threads = 1;
    rc = camblas_plan_make(&op, &topo, &ctx, &plan);
    CHECK(rc == 0);
    CHECK(plan.parallel_scheme == CAMBLAS_PARALLEL_NONE);
    return 0;
}

static int test_single_cpu(void)
{
    camblas_topology_t topo = make_topo(1);
    camblas_ctx_t ctx = make_ctx("shape-aware", 8, CAMBLAS_REPRO_FAST);
    camblas_plan_t plan;

    /* Large matrix but only 1 CPU → scalar 1-thread */
    camblas_op_t op = make_op(1024, 1024, 1024, CAMBLAS_DTYPE_F32);
    int rc = camblas_plan_make(&op, &topo, &ctx, &plan);
    CHECK(rc == 0);
    CHECK(plan.kernel_id == CAMBLAS_KERNEL_SCALAR);
    CHECK(plan.num_threads == 1);
    CHECK(plan.rationale_id == CAMBLAS_RAT_SINGLE_THREAD);
    return 0;
}

static int test_bitwise_repro(void)
{
    camblas_topology_t topo = make_topo(8);
    camblas_op_t op = make_op(1024, 1024, 1024, CAMBLAS_DTYPE_F32);
    camblas_plan_t plan;

    /* Bitwise repro forces scalar 1-thread regardless of mode */
    camblas_ctx_t ctx = make_ctx("shape-aware", 8, CAMBLAS_REPRO_BITWISE);
    int rc = camblas_plan_make(&op, &topo, &ctx, &plan);
    CHECK(rc == 0);
    CHECK(plan.kernel_id == CAMBLAS_KERNEL_SCALAR);
    CHECK(plan.num_threads == 1);
    CHECK(plan.parallel_scheme == CAMBLAS_PARALLEL_NONE);
    CHECK(plan.rationale_id == CAMBLAS_RAT_REPRO_BITWISE);

    /* Same for "fixed" mode */
    ctx.planner_mode = "fixed";
    rc = camblas_plan_make(&op, &topo, &ctx, &plan);
    CHECK(rc == 0);
    CHECK(plan.kernel_id == CAMBLAS_KERNEL_SCALAR);
    CHECK(plan.num_threads == 1);
    CHECK(plan.rationale_id == CAMBLAS_RAT_REPRO_BITWISE);

    /* Reproducible (not bitwise) does NOT force scalar */
    ctx.reproducibility = CAMBLAS_REPRO_REPRODUCIBLE;
    ctx.planner_mode = "shape-aware";
    ctx.num_threads = 4;
    rc = camblas_plan_make(&op, &topo, &ctx, &plan);
    CHECK(rc == 0);
    CHECK(plan.kernel_id != CAMBLAS_KERNEL_SCALAR);

    /* Reproducibility is a closed selector. Out-of-range values must fail
       closed even in an otherwise valid context and must not modify plan. */
    {
        const int invalid_repro[] = {
            -1,
            CAMBLAS_REPRO_BITWISE + 1,
            INT_MIN,
            INT_MAX,
        };
        camblas_plan_t sentinel;
        memset(&sentinel, 0xC3, sizeof(sentinel));
        ctx.planner_mode = "none";
        ctx.topo = NULL;
        for (size_t i = 0; i < sizeof(invalid_repro) / sizeof(invalid_repro[0]); i++) {
            ctx.reproducibility = invalid_repro[i];
            plan = sentinel;
            CHECK(camblas_plan_make(&op, NULL, &ctx, &plan) == -1);
            CHECK(memcmp(&plan, &sentinel, sizeof(plan)) == 0);
        }
    }

    /* Bitwise selection cannot sanitize malformed planner context. These
       checks protect the same fail-closed boundary used by public GEMM. */
    ctx.reproducibility = CAMBLAS_REPRO_BITWISE;
    ctx.planner_mode = "bogus-mode";
    CHECK(camblas_plan_make(&op, &topo, &ctx, &plan) == -1);

    ctx.planner_mode = "shape-aware";
    CHECK(camblas_plan_make(&op, NULL, &ctx, &plan) == -1);
    ctx.planner_mode = "ablation";
    CHECK(camblas_plan_make(&op, NULL, &ctx, &plan) == -1);
    ctx.planner_mode = "sve";
    CHECK(camblas_plan_make(&op, NULL, &ctx, &plan) == -1);

    camblas_topology_t fatal = make_topo(0);
    ctx.planner_mode = "shape-aware";
    CHECK(camblas_plan_make(&op, &fatal, &ctx, &plan) == -2);
    ctx.planner_mode = "ablation";
    CHECK(camblas_plan_make(&op, &fatal, &ctx, &plan) == -2);

    camblas_topology_t multi = make_topo(2);
    ctx.planner_mode = "sve";
    CHECK(camblas_plan_make(&op, &multi, &ctx, &plan) == -1);
    return 0;
}

static int test_topo_fatal(void)
{
    camblas_topology_t topo;
    memset(&topo, 0, sizeof(topo));
    topo.n_allowed_cpus = 0; /* invalid */
    topo.sve_vl_bits = 128;

    camblas_op_t op = make_op(64, 64, 64, CAMBLAS_DTYPE_F32);
    camblas_ctx_t ctx = make_ctx("shape-aware", 4, CAMBLAS_REPRO_FAST);
    camblas_plan_t plan;

    /* n_allowed_cpus < 1 → fatal for shape-aware */
    int rc = camblas_plan_make(&op, &topo, &ctx, &plan);
    CHECK(rc == -2);

    /* But OK for "none" (topo not needed) */
    ctx.planner_mode = "none";
    rc = camblas_plan_make(&op, &topo, &ctx, &plan);
    CHECK(rc == 0);
    return 0;
}

static int test_topo_scalar_metadata(void)
{
    camblas_op_t op = make_op(64, 64, 64, CAMBLAS_DTYPE_F32);
    camblas_ctx_t ctx = make_ctx("shape-aware", INT_MAX, CAMBLAS_REPRO_FAST);
    camblas_topology_t topo = make_topo(8);
    camblas_plan_t plan, sentinel;
    memset(&sentinel, 0xA7, sizeof(sentinel));

    /* An allowed mask outside the public representation is fatal for the
       topology-dependent planner, just like a zero/negative count. */
    topo.n_allowed_cpus = CAMBLAS_MAX_CPUS + 1;
    plan = sentinel;
    CHECK(camblas_plan_make(&op, &topo, &ctx, &plan) == -2);
    CHECK(memcmp(&plan, &sentinel, sizeof(plan)) == 0);
    topo.n_allowed_cpus = INT_MAX;
    plan = sentinel;
    CHECK(camblas_plan_make(&op, &topo, &ctx, &plan) == -2);
    CHECK(memcmp(&plan, &sentinel, sizeof(plan)) == 0);

    /* Zero is the optional NO_SVE value for non-SVE planning.  Strictly
       malformed nonzero values must not be treated as usable metadata. */
    const int bad_vls[] = {-1, 1, 127, 129, 65537, INT_MAX};
    topo.n_allowed_cpus = 8;
    topo.sve_vl_bits = 0;
    CHECK(camblas_plan_make(&op, &topo, &ctx, &plan) == 0);
    for (size_t i = 0; i < sizeof(bad_vls) / sizeof(bad_vls[0]); i++) {
        topo.sve_vl_bits = bad_vls[i];
        plan = sentinel;
        CHECK(camblas_plan_make(&op, &topo, &ctx, &plan) == -1);
        CHECK(memcmp(&plan, &sentinel, sizeof(plan)) == 0);
    }

    /* NUMA and online counts are optional/reference metadata, but impossible
       scalar values are malformed even though planning does not use them. */
    topo.sve_vl_bits = 0;
    topo.n_numa_nodes = -1;
    plan = sentinel;
    CHECK(camblas_plan_make(&op, &topo, &ctx, &plan) == -1);
    topo.n_numa_nodes = CAMBLAS_MAX_NUMA_NODES + 1;
    plan = sentinel;
    CHECK(camblas_plan_make(&op, &topo, &ctx, &plan) == -1);
    topo.n_numa_nodes = 0;
    topo.n_online_cpus = -1;
    plan = sentinel;
    CHECK(camblas_plan_make(&op, &topo, &ctx, &plan) == -1);
    topo.n_online_cpus = 0;
    CHECK(camblas_plan_make(&op, &topo, &ctx, &plan) == 0);
    topo.n_online_cpus = INT_MAX;
    CHECK(camblas_plan_make(&op, &topo, &ctx, &plan) == 0);

    /* SVE mode requires a decoded, usable VL; the same scalar checks happen
       before the bitwise override. */
    topo.n_online_cpus = 0;
    set_allowed_mask(&topo, 1);
    ctx.planner_mode = "sve";
    topo.sve_vl_bits = 128;
    CHECK(camblas_plan_make(&op, &topo, &ctx, &plan) == 0);
    topo.sve_vl_bits = 65536;
    CHECK(camblas_plan_make(&op, &topo, &ctx, &plan) == 0);
    topo.sve_vl_bits = 0;
    CHECK(camblas_plan_make(&op, &topo, &ctx, &plan) == -1);
    topo.sve_vl_bits = 1;
    CHECK(camblas_plan_make(&op, &topo, &ctx, &plan) == -1);
    ctx.reproducibility = CAMBLAS_REPRO_BITWISE;
    CHECK(camblas_plan_make(&op, &topo, &ctx, &plan) == -1);

    return 0;
}

static int test_unknown_mode(void)
{
    camblas_topology_t topo = make_topo(8);
    camblas_op_t op = make_op(64, 64, 64, CAMBLAS_DTYPE_F32);
    camblas_ctx_t ctx = make_ctx("bogus-mode", 4, CAMBLAS_REPRO_FAST);
    camblas_plan_t plan;

    int rc = camblas_plan_make(&op, &topo, &ctx, &plan);
    CHECK(rc == -1);
    return 0;
}

static int test_transpose_variants(void)
{
    camblas_topology_t topo = make_topo(8);
    camblas_ctx_t ctx = make_ctx("shape-aware", 4, CAMBLAS_REPRO_FAST);
    camblas_plan_t plan;

    /* All transpose combinations should produce valid plans */
    char combos[4][2] = {{'N', 'N'}, {'N', 'T'}, {'T', 'N'}, {'T', 'T'}};
    for (int i = 0; i < 4; i++) {
        camblas_op_t op = make_op(64, 64, 64, CAMBLAS_DTYPE_F32);
        op.trans_a = combos[i][0];
        op.trans_b = combos[i][1];
        int rc = camblas_plan_make(&op, &topo, &ctx, &plan);
        CHECK(rc == 0);
        CHECK(plan.kernel_id == CAMBLAS_KERNEL_NOPACK);
    }
    return 0;
}

static int test_overflow(void)
{
    camblas_topology_t topo = make_topo(8);
    camblas_ctx_t ctx = make_ctx("shape-aware", 4, CAMBLAS_REPRO_FAST);
    camblas_plan_t plan;

    /* Very large dimensions that would overflow int product. */
    /* 2*100000^3 = 2e15, fits in int64_t but overflows int. */
    /* compute_flops should detect via __builtin_mul_overflow. */
    /* Result > MEDIUM_CUTOFF, so this should select packed. */
    camblas_op_t op = make_op(100000, 100000, 100000, CAMBLAS_DTYPE_F32);
    int rc = camblas_plan_make(&op, &topo, &ctx, &plan);
    CHECK(rc == 0);
    CHECK(plan.kernel_id == CAMBLAS_KERNEL_PACKED);
    CHECK(plan.rationale_id == CAMBLAS_RAT_LARGE_PACKED);
    return 0;
}

static int test_format(void)
{
    camblas_topology_t topo = make_topo(8);
    camblas_ctx_t ctx = make_ctx("shape-aware", 4, CAMBLAS_REPRO_FAST);
    camblas_plan_t plan;

    camblas_op_t op = make_op(256, 256, 256, CAMBLAS_DTYPE_F32);
    int rc = camblas_plan_make(&op, &topo, &ctx, &plan);
    CHECK(rc == 0);

    /* Valid format */
    char buf[512];
    int n = camblas_plan_format(&plan, buf, sizeof(buf));
    CHECK(n > 0);
    CHECK(strstr(buf, "kernel=packed") != NULL);
    CHECK(strstr(buf, "threads=4") != NULL);
    CHECK(strstr(buf, "parallel=2d") != NULL);
    CHECK(strstr(buf, "pack=both") != NULL);
    CHECK(strstr(buf, "mc=128") != NULL);
    CHECK(strstr(buf, "rationale=large") != NULL);

    /* Buffer too small */
    char tiny[8];
    n = camblas_plan_format(&plan, tiny, sizeof(tiny));
    CHECK(n == -1);

    /* NULL args */
    CHECK(camblas_plan_format(NULL, buf, sizeof(buf)) == -1);
    CHECK(camblas_plan_format(&plan, NULL, sizeof(buf)) == -1);
    CHECK(camblas_plan_format(&plan, buf, 0) == -1);
    return 0;
}

static int test_name_functions(void)
{
    /* Kernel names */
    CHECK(strcmp(camblas_kernel_name(CAMBLAS_KERNEL_SCALAR), "scalar") == 0);
    CHECK(strcmp(camblas_kernel_name(CAMBLAS_KERNEL_NOPACK), "nopack") == 0);
    CHECK(strcmp(camblas_kernel_name(CAMBLAS_KERNEL_PACKED), "packed") == 0);
    CHECK(strcmp(camblas_kernel_name(CAMBLAS_KERNEL_BATCHED), "batched") == 0);
    CHECK(strcmp(camblas_kernel_name(-1), "unknown") == 0);
    CHECK(strcmp(camblas_kernel_name(99), "unknown") == 0);

    /* Parallel names */
    CHECK(strcmp(camblas_parallel_name(CAMBLAS_PARALLEL_NONE), "none") == 0);
    CHECK(strcmp(camblas_parallel_name(CAMBLAS_PARALLEL_ROW), "row") == 0);
    CHECK(strcmp(camblas_parallel_name(CAMBLAS_PARALLEL_COL), "col") == 0);
    CHECK(strcmp(camblas_parallel_name(CAMBLAS_PARALLEL_K), "k") == 0);
    CHECK(strcmp(camblas_parallel_name(CAMBLAS_PARALLEL_2D), "2d") == 0);
    CHECK(strcmp(camblas_parallel_name(-1), "unknown") == 0);

    /* Pack names */
    CHECK(strcmp(camblas_pack_name(CAMBLAS_PACK_NONE), "none") == 0);
    CHECK(strcmp(camblas_pack_name(CAMBLAS_PACK_A), "a") == 0);
    CHECK(strcmp(camblas_pack_name(CAMBLAS_PACK_B), "b") == 0);
    CHECK(strcmp(camblas_pack_name(CAMBLAS_PACK_BOTH), "both") == 0);
    CHECK(strcmp(camblas_pack_name(-1), "unknown") == 0);
    return 0;
}

static int test_rationale(void)
{
    camblas_plan_t plan;
    plan.rationale_id = CAMBLAS_RAT_LARGE_PACKED;
    CHECK(strstr(camblas_plan_rationale(&plan), "large") != NULL);

    plan.rationale_id = CAMBLAS_RAT_SMALL_SCALAR;
    CHECK(strstr(camblas_plan_rationale(&plan), "small") != NULL);

    plan.rationale_id = 99;
    CHECK(strcmp(camblas_plan_rationale(&plan), "unknown") == 0);

    CHECK(strcmp(camblas_plan_rationale(NULL), "unknown") == 0);
    return 0;
}

/* ---- Reentrancy test ---- */

struct reentrancy_arg {
    camblas_topology_t topo;
    camblas_op_t op;
    camblas_ctx_t ctx;
    camblas_plan_t plan;
    int rc;
};

static void *reentrancy_thread(void *p)
{
    struct reentrancy_arg *a = (struct reentrancy_arg *)p;
    a->rc = camblas_plan_make(&a->op, &a->topo, &a->ctx, &a->plan);
    return NULL;
}

static int test_reentrancy(void)
{
    camblas_topology_t topo = make_topo(8);

    struct reentrancy_arg args[4];
    pthread_t threads[4];

    /* Each thread uses a different shape and mode. */
    args[0].topo = topo;
    args[0].op = make_op(4, 4, 4, CAMBLAS_DTYPE_F32);
    args[0].ctx = make_ctx("shape-aware", 4, CAMBLAS_REPRO_FAST);

    args[1].topo = topo;
    args[1].op = make_op(64, 64, 64, CAMBLAS_DTYPE_F64);
    args[1].ctx = make_ctx("shape-aware", 4, CAMBLAS_REPRO_FAST);

    args[2].topo = topo;
    args[2].op = make_op(1024, 1024, 1024, CAMBLAS_DTYPE_F32);
    args[2].ctx = make_ctx("shape-aware", 8, CAMBLAS_REPRO_FAST);

    args[3].topo = topo;
    args[3].op = make_op(8, 8, 8, CAMBLAS_DTYPE_F32);
    args[3].op.batch_count = 16;
    args[3].ctx = make_ctx("shape-aware", 4, CAMBLAS_REPRO_FAST);

    for (int i = 0; i < 4; i++)
        pthread_create(&threads[i], NULL, reentrancy_thread, &args[i]);
    for (int i = 0; i < 4; i++)
        pthread_join(threads[i], NULL);

    /* Thread 0: small → scalar 1-thread */
    CHECK(args[0].rc == 0);
    CHECK(args[0].plan.kernel_id == CAMBLAS_KERNEL_SCALAR);
    CHECK(args[0].plan.num_threads == 1);

    /* Thread 1: medium → no-pack 4 threads */
    CHECK(args[1].rc == 0);
    CHECK(args[1].plan.kernel_id == CAMBLAS_KERNEL_NOPACK);
    CHECK(args[1].plan.num_threads == 4);

    /* Thread 2: large → packed 8 threads */
    CHECK(args[2].rc == 0);
    CHECK(args[2].plan.kernel_id == CAMBLAS_KERNEL_PACKED);
    CHECK(args[2].plan.num_threads == 8);

    /* Thread 3: batched → batched 4 threads */
    CHECK(args[3].rc == 0);
    CHECK(args[3].plan.kernel_id == CAMBLAS_KERNEL_BATCHED);
    CHECK(args[3].plan.num_threads == 4);
    return 0;
}

static int test_ablation_mode(void)
{
    /* "ablation" mode uses the same decision table as "shape-aware". */
    camblas_topology_t topo = make_topo(8);
    camblas_ctx_t ctx = make_ctx("ablation", 4, CAMBLAS_REPRO_FAST);
    camblas_plan_t plan;

    /* Small */
    camblas_op_t op = make_op(4, 4, 4, CAMBLAS_DTYPE_F32);
    int rc = camblas_plan_make(&op, &topo, &ctx, &plan);
    CHECK(rc == 0);
    CHECK(plan.kernel_id == CAMBLAS_KERNEL_SCALAR);

    /* Large */
    op = make_op(256, 256, 256, CAMBLAS_DTYPE_F32);
    rc = camblas_plan_make(&op, &topo, &ctx, &plan);
    CHECK(rc == 0);
    CHECK(plan.kernel_id == CAMBLAS_KERNEL_PACKED);

    /* Ablation also requires topology */
    CHECK(camblas_plan_make(&op, NULL, &ctx, &plan) == -1);
    return 0;
}

static int test_sve_mode(void)
{
    camblas_topology_t topo = make_topo(1);
    camblas_ctx_t ctx = make_ctx("sve", 8, CAMBLAS_REPRO_REPRODUCIBLE);
    camblas_plan_t plan;
    camblas_op_t op = make_op(64, 32, 48, CAMBLAS_DTYPE_F32);

    /* CPU model is optional metadata.  A missing NO_MODEL value must not
       block a valid SVE plan; benchmark provenance applies its own stricter
       model requirement at the emission boundary. */
    CHECK(topo.cpu_model[0] == '\0');

    /* The first SVE path is explicitly single-core and derives its row tile
       from the runtime vector length rather than a fixed public ABI value. */
    topo.sve_vl_bits = 256;
    CHECK(camblas_plan_make(&op, &topo, &ctx, &plan) == 0);
    CHECK(plan.kernel_id == CAMBLAS_KERNEL_SVE);
    CHECK(plan.pack == CAMBLAS_PACK_NONE);
    CHECK(plan.num_threads == 1);
    CHECK(plan.parallel_scheme == CAMBLAS_PARALLEL_NONE);
    CHECK(plan.mr == 8 && plan.nr == 2);
    CHECK(plan.rationale_id == CAMBLAS_RAT_SVE_SINGLE_CORE);

    /* N? A with transposed B remains in the vector path.  T? A is an
       intentional scalar fallback until a transposed-A kernel exists. */
    op.trans_b = 'T';
    CHECK(camblas_plan_make(&op, &topo, &ctx, &plan) == 0);
    CHECK(plan.kernel_id == CAMBLAS_KERNEL_SVE);
    op.trans_a = 'T';
    CHECK(camblas_plan_make(&op, &topo, &ctx, &plan) == 0);
    CHECK(plan.kernel_id == CAMBLAS_KERNEL_SCALAR);
    CHECK(plan.rationale_id == CAMBLAS_RAT_SVE_FALLBACK);

    /* The planner refuses an ambiguous multi-core or unavailable-VL target;
       it must not silently relabel either case as a single-core SVE plan. */
    topo.n_allowed_cpus = 2;
    set_allowed_mask(&topo, 2);
    CHECK(camblas_plan_make(&op, &topo, &ctx, &plan) == -1);
    set_allowed_mask(&topo, 1);
    topo.sve_vl_bits = 0;
    CHECK(camblas_plan_make(&op, &topo, &ctx, &plan) == -1);

    /* Bitwise reproducibility retains the scalar override in SVE mode. */
    topo.sve_vl_bits = 128;
    ctx.reproducibility = CAMBLAS_REPRO_BITWISE;
    CHECK(camblas_plan_make(&op, &topo, &ctx, &plan) == 0);
    CHECK(plan.kernel_id == CAMBLAS_KERNEL_SCALAR);
    CHECK(plan.rationale_id == CAMBLAS_RAT_REPRO_BITWISE);
    return 0;
}

static int test_topo_set_and_numa_consistency(void)
{
    camblas_op_t op = make_op(64, 64, 64, CAMBLAS_DTYPE_F32);
    camblas_ctx_t ctx = make_ctx("shape-aware", 4, CAMBLAS_REPRO_FAST);
    camblas_topology_t topo = make_topo(4);
    camblas_plan_t plan, sentinel;
    memset(&sentinel, 0xD4, sizeof(sentinel));

    /* A positive online count smaller than the allowed set is an
       inconsistent optional snapshot and must fail closed. */
    topo.n_online_cpus = 3;
    plan = sentinel;
    CHECK(camblas_plan_make(&op, &topo, &ctx, &plan) == -1);
    CHECK(memcmp(&plan, &sentinel, sizeof(plan)) == 0);
    topo.n_online_cpus = 4;

    /* A scalar count cannot stand in for the allowed mask. */
    topo.n_allowed_cpus = 3;
    plan = sentinel;
    CHECK(camblas_plan_make(&op, &topo, &ctx, &plan) == -2);
    CHECK(memcmp(&plan, &sentinel, sizeof(plan)) == 0);

    set_allowed_mask(&topo, 4);
    topo.allowed_cpus.nbits = 3; /* stale canonical bound */
    plan = sentinel;
    CHECK(camblas_plan_make(&op, &topo, &ctx, &plan) == -2);
    CHECK(memcmp(&plan, &sentinel, sizeof(plan)) == 0);

    set_allowed_mask(&topo, 4);
    camblas_cpuset_zero(&topo.allowed_cpus); /* positive scalar, empty mask */
    plan = sentinel;
    CHECK(camblas_plan_make(&op, &topo, &ctx, &plan) == -2);
    CHECK(memcmp(&plan, &sentinel, sizeof(plan)) == 0);

    /* none/fixed remain topology-agnostic by contract. */
    ctx.planner_mode = "none";
    CHECK(camblas_plan_make(&op, &topo, &ctx, &plan) == 0);
    ctx.planner_mode = "fixed";
    CHECK(camblas_plan_make(&op, &topo, &ctx, &plan) == 0);

    /* Two valid disjoint NUMA intersections are accepted. */
    topo = make_topo(4);
    topo.n_numa_nodes = 2;
    topo.numa_nodes[0].node_id = 0;
    topo.numa_nodes[0].n_cpus = 2;
    topo.numa_nodes[0].n_cpus_physical = 2;
    camblas_cpuset_set(&topo.numa_nodes[0].cpus, 0);
    camblas_cpuset_set(&topo.numa_nodes[0].cpus, 1);
    topo.numa_nodes[1].node_id = 1;
    topo.numa_nodes[1].n_cpus = 2;
    topo.numa_nodes[1].n_cpus_physical = 2;
    camblas_cpuset_set(&topo.numa_nodes[1].cpus, 2);
    camblas_cpuset_set(&topo.numa_nodes[1].cpus, 3);
    ctx.planner_mode = "shape-aware";
    CHECK(camblas_plan_make(&op, &topo, &ctx, &plan) == 0);

    /* Scalar/bit-count mismatch, out-of-mask CPU, and overlap are optional
       NUMA metadata errors and must fail closed without publishing a plan. */
    topo.numa_nodes[0].n_cpus = 1;
    plan = sentinel;
    CHECK(camblas_plan_make(&op, &topo, &ctx, &plan) == -1);
    CHECK(memcmp(&plan, &sentinel, sizeof(plan)) == 0);

    topo.numa_nodes[0].n_cpus = 2;
    camblas_cpuset_set(&topo.numa_nodes[0].cpus, 4);
    topo.numa_nodes[0].n_cpus = 3;
    plan = sentinel;
    CHECK(camblas_plan_make(&op, &topo, &ctx, &plan) == -1);
    CHECK(memcmp(&plan, &sentinel, sizeof(plan)) == 0);

    topo.numa_nodes[0].n_cpus = 2;
    camblas_cpuset_zero(&topo.numa_nodes[0].cpus);
    camblas_cpuset_set(&topo.numa_nodes[0].cpus, 0);
    camblas_cpuset_set(&topo.numa_nodes[0].cpus, 1);
    camblas_cpuset_zero(&topo.numa_nodes[1].cpus);
    camblas_cpuset_set(&topo.numa_nodes[1].cpus, 1);
    camblas_cpuset_set(&topo.numa_nodes[1].cpus, 3);
    plan = sentinel;
    CHECK(camblas_plan_make(&op, &topo, &ctx, &plan) == -1);
    CHECK(memcmp(&plan, &sentinel, sizeof(plan)) == 0);

    /* NUMA-unavailable is the valid optional zero-node representation. */
    topo.n_numa_nodes = 0;
    CHECK(camblas_plan_make(&op, &topo, &ctx, &plan) == 0);

    /* Bitwise reproducibility cannot sanitize malformed optional metadata. */
    topo.n_numa_nodes = 1;
    topo.numa_nodes[0].n_cpus = 0;
    ctx.reproducibility = CAMBLAS_REPRO_BITWISE;
    plan = sentinel;
    CHECK(camblas_plan_make(&op, &topo, &ctx, &plan) == -1);
    CHECK(memcmp(&plan, &sentinel, sizeof(plan)) == 0);
    return 0;
}

/* ---- Main ---- */

int main(void)
{
    printf("=== CAMBLAS planner unit tests ===\n\n");

    RUN_TEST(test_null_args);
    RUN_TEST(test_invalid_op);
    RUN_TEST(test_mode_none);
    RUN_TEST(test_mode_fixed);
    RUN_TEST(test_thread_request_bounds);
    RUN_TEST(test_shape_aware_small);
    RUN_TEST(test_shape_aware_medium);
    RUN_TEST(test_shape_aware_large);
    RUN_TEST(test_shape_aware_batch);
    RUN_TEST(test_single_cpu);
    RUN_TEST(test_bitwise_repro);
    RUN_TEST(test_topo_fatal);
    RUN_TEST(test_topo_scalar_metadata);
    RUN_TEST(test_topo_set_and_numa_consistency);
    RUN_TEST(test_unknown_mode);
    RUN_TEST(test_transpose_variants);
    RUN_TEST(test_overflow);
    RUN_TEST(test_format);
    RUN_TEST(test_name_functions);
    RUN_TEST(test_rationale);
    RUN_TEST(test_ablation_mode);
    RUN_TEST(test_sve_mode);
    RUN_TEST(test_reentrancy);

    printf("=== Summary: %d/%d passed ===\n", tests_pass, tests_run);
    return (tests_pass == tests_run) ? 0 : 1;
}
