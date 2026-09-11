/*
 * CAMBLAS runtime planner — implementation.
 *
 * Pure function: no global mutable state, no allocation, no side effects.
 * The planner uses explicit shape rules and bounded cache-capacity estimates.
 * Conservative reference defaults can be overridden by the retained Grace
 * build configuration without introducing mutable runtime tuning state.
 *
 * See include/camblas_planner.h for the full API contract.
 */
#include "camblas_planner.h"
#include "kernels.h"
#include "cpuset_bits.h"
#include "packed_grid.h"

#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <limits.h>

/* ---- Input validation ---- */

/*
 * Validate an operation descriptor.
 * Returns 0 on success, -1 on invalid.
 */
static int validate_op(const camblas_op_t *op)
{
    if (!op)
        return -1;
    if (op->m < 0 || op->n < 0 || op->k < 0)
        return -1;
    if (op->trans_a != 'N' && op->trans_a != 'T')
        return -1;
    if (op->trans_b != 'N' && op->trans_b != 'T')
        return -1;
    if (op->dtype != CAMBLAS_DTYPE_F32 && op->dtype != CAMBLAS_DTYPE_F64)
        return -1;
    if (op->batch_count < 1)
        return -1;
    return 0;
}

/*
 * Compare planner_mode string safely.
 * NULL is treated as "none".
 */
static int mode_is(const char *mode, const char *want)
{
    if (!mode)
        mode = "none";
    return strcmp(mode, want) == 0;
}

/*
 * Reproducibility is a closed public selector.  Do not let an unknown value
 * silently fall through to the fast/reproducible planner behavior: callers
 * must receive the documented invalid-call result before a plan is written.
 */
static int validate_reproducibility(int reproducibility)
{
    switch (reproducibility) {
    case CAMBLAS_REPRO_FAST:
    case CAMBLAS_REPRO_REPRODUCIBLE:
    case CAMBLAS_REPRO_BITWISE:
        return 0;
    default:
        return -1;
    }
}

/*
 * Validate the representation invariants maintained by camblas_cpuset_*.
 * Return the set-bit count on success, or -1 for malformed metadata.  A
 * positive n_allowed_cpus with an empty/stale bitset is an unusable allowed
 * mask, even when its scalar count is otherwise in range.
 */
static int validate_cpuset_representation(const camblas_cpuset_t *set)
{
    return camblas_cpuset_validated_count(set);
}

/*
 * The allowed mask is the fatal part of a topology snapshot.  Discovery
 * cannot return a usable snapshot with an empty, noncanonical, or scalar/
 * bit-count-mismatched mask, so topology-dependent planning preserves the
 * public -2 fatal split for every such inconsistency.
 */
static int validate_allowed_cpu_metadata(const camblas_topology_t *topo)
{
    if (!topo || topo->n_allowed_cpus < 1 || topo->n_allowed_cpus > CAMBLAS_MAX_CPUS)
        return -2;

    int count = validate_cpuset_representation(&topo->allowed_cpus);
    if (count < 1 || count != topo->n_allowed_cpus)
        return -2;
    return 0;
}

/*
 * Validate the optional NUMA portion of a snapshot.  A discovered node is a
 * nonempty intersection with the allowed mask; its scalar count must equal
 * its canonical CPU set, its physical count must cover that intersection,
 * and distinct physical nodes must not overlap.  NUMA is optional, so zero
 * nodes remains valid; malformed nonempty metadata is an invalid context.
 */
static int validate_numa_metadata(const camblas_topology_t *topo)
{
    if (!topo || topo->n_numa_nodes < 0 || topo->n_numa_nodes > CAMBLAS_MAX_NUMA_NODES)
        return -1;
    if (topo->n_numa_nodes == 0)
        return 0;

    camblas_cpuset_t seen;
    camblas_cpuset_zero(&seen);
    int total_cpus = 0;

    for (int i = 0; i < topo->n_numa_nodes; i++) {
        const camblas_numa_node_t *node = &topo->numa_nodes[i];
        if (node->node_id < 0 || node->node_id >= INT_MAX || node->n_cpus <= 0 ||
            node->n_cpus_physical < node->n_cpus || node->n_cpus_physical > CAMBLAS_MAX_CPUS)
            return -1;

        for (int previous = 0; previous < i; previous++) {
            if (topo->numa_nodes[previous].node_id == node->node_id)
                return -1;
        }

        int set_count = validate_cpuset_representation(&node->cpus);
        if (set_count != node->n_cpus)
            return -1;

        for (size_t word = 0; word < sizeof(seen.bits) / sizeof(seen.bits[0]); ++word) {
            uint64_t mask = node->cpus.bits[word];
            if ((mask & ~topo->allowed_cpus.bits[word]) || (mask & seen.bits[word]))
                return -1;
            seen.bits[word] |= mask;
        }

        if (total_cpus > CAMBLAS_MAX_CPUS - node->n_cpus)
            return -1;
        total_cpus += node->n_cpus;
    }

    if (total_cpus > topo->n_allowed_cpus)
        return -1;
    return 0;
}

/*
 * Validate the scalar metadata that a planner can safely consume from a
 * topology snapshot.  Discovery returns zero for unavailable optional
 * metadata, so zero NUMA nodes, zero online CPUs, and zero SVE VL remain
 * valid outside consumers that specifically require that metadata.  A
 * nonzero SVE VL, however, must be one of the Linux UAPI values accepted by
 * the private PR_SVE_GET_VL decoder; accepting an arbitrary positive value
 * would let the planner publish a fabricated vector tile.
 *
 * The allowed-CPU count has a separate fatal/invalid split because the
 * public planner contract reports an unusable allowed mask as -2 for
 * shape-aware and ablation modes.  The caller supplies no discovery status
 * to camblas_plan_make(), so the representable scalar bounds are the
 * fail-closed checks available at this boundary.
 */
static int validate_topology_metadata(const camblas_topology_t *topo)
{
    if (!topo)
        return -1;

    int allowed_rc = validate_allowed_cpu_metadata(topo);
    if (allowed_rc != 0)
        return allowed_rc;

    /* Zero is the optional NO_SVE result; nonzero values are strict. */
    if (topo->sve_vl_bits != 0 &&
        (topo->sve_vl_bits < 128 || topo->sve_vl_bits > 65536 || topo->sve_vl_bits % 128 != 0))
        return -1;

    /* Zero means no NUMA nodes were available or the optional scan was
       cleared fail-closed.  Negative/capacity-overflow counts are malformed. */
    if (topo->n_numa_nodes < 0 || topo->n_numa_nodes > CAMBLAS_MAX_NUMA_NODES)
        return -1;

    /* Online CPU count is reference-only.  It may be unavailable (zero) and
       may exceed the public allowed-mask capacity on a larger host, but a
       positive value must still cover the process's allowed CPUs.  A smaller
       positive value is an inconsistent caller-owned snapshot, not a reason
       to cap allocation-visible planning. */
    if (topo->n_online_cpus < 0 ||
        (topo->n_online_cpus > 0 && topo->n_online_cpus < topo->n_allowed_cpus))
        return -1;

    if (validate_numa_metadata(topo) != 0)
        return -1;

    return 0;
}

/*
 * Validate the planner context before applying any policy override. In
 * particular, an unknown reproducibility level and bitwise reproducibility
 * cannot make an unknown mode or a topology-dependent mode without its
 * required topology valid. Keep the public error split used by the decision
 * table: malformed/missing context is -1, while an explicitly supplied
 * topology with an unusable allowed mask is the shape-aware fatal condition
 * (-2).
 */
static int validate_mode_context(const char *mode, const camblas_topology_t *topo)
{
    if (mode_is(mode, "none") || mode_is(mode, "fixed"))
        return 0;

    if (mode_is(mode, "sve")) {
        if (!topo || topo->n_allowed_cpus != 1 || topo->sve_vl_bits < 128 ||
            topo->sve_vl_bits > 65536 || topo->sve_vl_bits % 128 != 0 ||
            validate_topology_metadata(topo) != 0)
            return -1;
        return 0;
    }

    if (mode_is(mode, "shape-aware") || mode_is(mode, "ablation")) {
        if (!topo)
            return -1;
        if (topo->n_allowed_cpus < 1 || topo->n_allowed_cpus > CAMBLAS_MAX_CPUS)
            return -2;
        int topo_rc = validate_topology_metadata(topo);
        if (topo_rc != 0)
            return topo_rc;
        return 0;
    }

    return -1;
}

/*
 * Clamp a thread count to [1, max].
 */
static int clamp_threads(int requested, int max_cpus)
{
    if (requested < 1)
        requested = 1;
    if (max_cpus < 1)
        max_cpus = 1;
    if (requested > max_cpus)
        return max_cpus;
    return requested;
}

/*
 * Compute 2*m*n*k as int64_t with overflow check.
 * Returns 0 on success, -1 on overflow.
 * Sets *flops to the result (0 if any dimension is zero).
 */
static int compute_flops(const camblas_op_t *op, int64_t *flops)
{
    int64_t m = op->m;
    int64_t n = op->n;
    int64_t k = op->k;
    int64_t b = op->batch_count;

    /* For planning, we only need relative magnitude, not exact FLOPs. */
    /* If any dimension is 0, flops = 0. */
    if (m == 0 || n == 0 || k == 0 || b == 0) {
        *flops = 0;
        return 0;
    }

    /* Compute 2*m*n*k with overflow detection. */
    int64_t result;
    if (__builtin_mul_overflow(m, n, &result))
        goto overflow;
    if (__builtin_mul_overflow(result, k, &result))
        goto overflow;
    if (__builtin_mul_overflow(result, (int64_t)2, &result))
        goto overflow;
    if (__builtin_mul_overflow(result, b, &result))
        goto overflow;
    *flops = result;
    return 0;

overflow:
    /* Overflow: treat as very large. Use INT64_MAX as sentinel. */
    *flops = INT64_MAX;
    return 0;
}

/* ---- Decision table thresholds (UNTUNED PLACEHOLDERS) ---- */

/*
 * Below SMALL_CUTOFF FLOPs, the overhead of any parallelism or packing
 * dominates. Scalar single-threaded is optimal.
 * Threshold: 2*16*16*16 = 8192 FLOPs.
 */
#define SMALL_CUTOFF 8192

/*
 * Below MEDIUM_CUTOFF FLOPs, the matrix fits in L1/L2 cache and packing
 * overhead may not be amortized. No-pack kernel is preferred.
 * Threshold: 2*128*128*128 = 4,194,304 FLOPs (~4M).
 */
#define MEDIUM_CUTOFF 4194304

/*
 * Maximum threads for medium-sized matrices. Over-threading small/medium
 * matrices increases overhead without benefit.
 */
#define MEDIUM_MAX_THREADS 4

/* ---- Conservative defaults and build-time blocking parameters ---- */

/*
 * The reference defaults are mc=128, nc=256 and kc=128. The Grace build
 * overrides these cache blocks and enables guarded shape-dependent choices.
 * SVE vector lengths are measured in bits: 128 bits hold four FP32 lanes or
 * two FP64 lanes. Register tile dimensions are a separate kernel property.
 */
#ifndef DEFAULT_MC
#define DEFAULT_MC 128
#endif
#ifndef CAMBLAS_DEEP_KC_FP32
#define CAMBLAS_DEEP_KC_FP32 0
#endif
#ifndef CAMBLAS_DEEP_KC_CACHE_BYTES
#define CAMBLAS_DEEP_KC_CACHE_BYTES 0
#endif
#ifndef CAMBLAS_DEEP_KC_MIN_THREADS
#define CAMBLAS_DEEP_KC_MIN_THREADS 16
#endif
#ifndef DEFAULT_NC
#define DEFAULT_NC 256
#endif
#ifndef DEFAULT_KC
#define DEFAULT_KC 128
#endif
#define DEFAULT_MR_F32 4
#define DEFAULT_NR_F32 8
#define DEFAULT_MR_F64 2
#define DEFAULT_NR_F64 4
#define PAIRED_PACKED_NR 8
#ifndef CAMBLAS_SMALL_CACHEBLOCK
#define CAMBLAS_SMALL_CACHEBLOCK 0
#endif
#ifndef CAMBLAS_DEEP_CACHEBLOCK
#define CAMBLAS_DEEP_CACHEBLOCK 0
#endif
#ifndef CAMBLAS_DEEP_ROW_BLOCK
#define CAMBLAS_DEEP_ROW_BLOCK 0
#endif
#ifndef CAMBLAS_DEEP_ROW_BLOCK_FP64
#define CAMBLAS_DEEP_ROW_BLOCK_FP64 0
#endif
#ifndef CAMBLAS_DEEP_ROW_BLOCK_MAX_THREADS
#define CAMBLAS_DEEP_ROW_BLOCK_MAX_THREADS 32
#endif
#ifndef CAMBLAS_SHALLOW_ASPECT_GRID_FP32
#define CAMBLAS_SHALLOW_ASPECT_GRID_FP32 0
#endif

/* ---- Planner function ---- */

int camblas_plan_make(const camblas_op_t *op, const camblas_topology_t *topo,
                      const camblas_ctx_t *ctx, camblas_plan_t *plan)
{
    if (!op || !ctx || !plan)
        return -1;
    if (validate_op(op) != 0)
        return -1;

    /* Every context selector is validated before any policy shortcut. */
    if (validate_reproducibility(ctx->reproducibility) != 0)
        return -1;

    /* Mode/topology validation also precedes bitwise reproducibility; a
       reproducibility policy cannot sanitize malformed planner state. */
    int mode_rc = validate_mode_context(ctx->planner_mode, topo);
    if (mode_rc != 0)
        return mode_rc;

    /* Determine available CPU count. */
    int n_cpus = 1;
    if (topo) {
        n_cpus = topo->n_allowed_cpus;
        if (n_cpus < 1)
            n_cpus = 1;
    }

    /* Normalize the request before any plan/no-op publication.  This is
       deliberate valid-context handling: zero, negative, and INT_MIN
       requests become one, while positive requests are capped below by the
       selected topology/policy. */
    int req_threads = ctx->num_threads;
    if (req_threads < 1)
        req_threads = 1;

    /* ---- Bitwise reproducibility override ---- */
    /* Bitwise repro forces scalar, 1 thread, no pack, regardless of mode. */
    if (ctx->reproducibility == CAMBLAS_REPRO_BITWISE) {
        plan->kernel_id = CAMBLAS_KERNEL_SCALAR;
        plan->pack = CAMBLAS_PACK_NONE;
        plan->num_threads = 1;
        plan->parallel_scheme = CAMBLAS_PARALLEL_NONE;
        plan->mc = plan->nc = plan->kc = 0;
        plan->mr = plan->nr = 0;
        plan->rationale_id = CAMBLAS_RAT_REPRO_BITWISE;
        return 0;
    }

    /* ---- Planner mode: "none" ---- */
    if (mode_is(ctx->planner_mode, "none")) {
        plan->kernel_id = CAMBLAS_KERNEL_SCALAR;
        plan->pack = CAMBLAS_PACK_NONE;
        plan->num_threads = 1;
        plan->parallel_scheme = CAMBLAS_PARALLEL_NONE;
        plan->mc = plan->nc = plan->kc = 0;
        plan->mr = plan->nr = 0;
        plan->rationale_id = CAMBLAS_RAT_NONE_DISABLED;
        return 0;
    }

    /* ---- Planner mode: "fixed" ---- */
    /* Fixed policy: scalar kernel, no packing, requested threads clamped. */
    if (mode_is(ctx->planner_mode, "fixed")) {
        plan->kernel_id = CAMBLAS_KERNEL_SCALAR;
        plan->pack = CAMBLAS_PACK_NONE;
        plan->num_threads = clamp_threads(req_threads, n_cpus);
        plan->parallel_scheme =
            (plan->num_threads > 1) ? CAMBLAS_PARALLEL_ROW : CAMBLAS_PARALLEL_NONE;
        plan->mc = plan->nc = plan->kc = 0;
        plan->mr = plan->nr = 0;
        plan->rationale_id = CAMBLAS_RAT_FIXED_POLICY;
        return 0;
    }

    /* ---- Planner mode: explicit single-core SVE path ---- */
    /*
     * This mode is intentionally opt-in.  It keeps the scalar reference and
     * the existing Phase-1 decision-table modes unchanged while providing a
     * correctness-gated first optimized path for one allowed CPU.  The
     * execution translation unit performs a second runtime HWCAP/VL check;
     * the topology check here makes the requested scope explicit and keeps a
     * multi-core call from being mislabeled as a single-core benchmark.
     */
    if (mode_is(ctx->planner_mode, "sve")) {
        if (!topo || topo->n_allowed_cpus != 1)
            return -1;
        if (topo->sve_vl_bits <= 0)
            return -1;

        plan->kernel_id = CAMBLAS_KERNEL_SCALAR;
        plan->pack = CAMBLAS_PACK_NONE;
        plan->num_threads = 1;
        plan->parallel_scheme = CAMBLAS_PARALLEL_NONE;
        plan->mc = plan->nc = plan->kc = 0;
        plan->mr = plan->nr = 0;
        plan->rationale_id = CAMBLAS_RAT_SVE_FALLBACK;

        /* The first vector kernel covers N? with the build-selected output
           tile; an odd final column is handled by the kernel tail. T? A
           remains on the scalar reference until a transposed-A kernel exists. */
        if (op->batch_count == 1 && op->m > 0 && op->n > 0 && op->k >= 0 && op->trans_a == 'N') {
            plan->kernel_id = CAMBLAS_KERNEL_SVE;
            plan->mr =
                (op->dtype == CAMBLAS_DTYPE_F32) ? topo->sve_vl_bits / 32 : topo->sve_vl_bits / 64;
            if (plan->mr < 1)
                plan->mr = 1;
            plan->nr = CAMBLAS_SVE_NR;
            plan->rationale_id = CAMBLAS_RAT_SVE_SINGLE_CORE;
        }
        return 0;
    }

    /* ---- Planner modes requiring topology: "shape-aware", "ablation" ---- */
    if (mode_is(ctx->planner_mode, "shape-aware") || mode_is(ctx->planner_mode, "ablation")) {

        /* Topology is required for shape-aware planning. */
        if (!topo)
            return -1;

        /* Check topology for fatal errors. */
        /* We cannot call camblas_topology_discover's status here, but we can */
        /* sanity-check the topology struct. */
        if (topo->n_allowed_cpus < 1)
            return -2;

        /* Compute FLOPs for shape classification. */
        int64_t flops;
        if (compute_flops(op, &flops) != 0)
            return -1;

        /* ---- Batched path ---- */
        if (op->batch_count > 1) {
            plan->kernel_id = CAMBLAS_KERNEL_BATCHED;
            plan->pack = CAMBLAS_PACK_NONE; /* batched may pack internally */
            plan->num_threads = clamp_threads(req_threads, n_cpus);
            plan->parallel_scheme =
                (plan->num_threads > 1) ? CAMBLAS_PARALLEL_2D : CAMBLAS_PARALLEL_NONE;
            plan->mc = plan->nc = plan->kc = 0;
            plan->mr = plan->nr = 0;
            plan->rationale_id = CAMBLAS_RAT_BATCH;
            return 0;
        }

        /* ---- Single CPU: force single-threaded ---- */
        if (n_cpus == 1) {
            plan->kernel_id = CAMBLAS_KERNEL_SCALAR;
            plan->pack = CAMBLAS_PACK_NONE;
            plan->num_threads = 1;
            plan->parallel_scheme = CAMBLAS_PARALLEL_NONE;
            plan->mc = plan->nc = plan->kc = 0;
            plan->mr = plan->nr = 0;
            plan->rationale_id = CAMBLAS_RAT_SINGLE_THREAD;
            return 0;
        }

        /* ---- Zero-dimension GEMM: scalar, 1 thread ---- */
        if (flops == 0) {
            plan->kernel_id = CAMBLAS_KERNEL_SCALAR;
            plan->pack = CAMBLAS_PACK_NONE;
            plan->num_threads = 1;
            plan->parallel_scheme = CAMBLAS_PARALLEL_NONE;
            plan->mc = plan->nc = plan->kc = 0;
            plan->mr = plan->nr = 0;
            plan->rationale_id = CAMBLAS_RAT_SMALL_SCALAR;
            return 0;
        }

        /* ---- Small matrix: scalar, 1 thread ---- */
        if (flops < SMALL_CUTOFF) {
            plan->kernel_id = CAMBLAS_KERNEL_SCALAR;
            plan->pack = CAMBLAS_PACK_NONE;
            plan->num_threads = 1;
            plan->parallel_scheme = CAMBLAS_PARALLEL_NONE;
            plan->mc = plan->nc = plan->kc = 0;
            plan->mr = plan->nr = 0;
            plan->rationale_id = CAMBLAS_RAT_SMALL_SCALAR;
            return 0;
        }

        /* ---- Medium matrix: no-pack, limited threads ---- */
        if (flops < MEDIUM_CUTOFF) {
            plan->kernel_id = CAMBLAS_KERNEL_NOPACK;
            plan->pack = CAMBLAS_PACK_NONE;
            plan->num_threads = clamp_threads(
                req_threads < MEDIUM_MAX_THREADS ? req_threads : MEDIUM_MAX_THREADS, n_cpus);
            plan->parallel_scheme =
                (plan->num_threads > 1) ? CAMBLAS_PARALLEL_ROW : CAMBLAS_PARALLEL_NONE;
            plan->mc = plan->nc = plan->kc = 0;
            plan->mr = plan->nr = 0;
            plan->rationale_id = CAMBLAS_RAT_MEDIUM_NOPACK;
            return 0;
        }

        /* ---- Large matrix: packed blocked, up to n_cpus threads ---- */
        plan->kernel_id = CAMBLAS_KERNEL_PACKED;
        plan->pack = CAMBLAS_PACK_BOTH;
        plan->num_threads = clamp_threads(req_threads, n_cpus);
        plan->parallel_scheme =
            (plan->num_threads > 1) ? CAMBLAS_PARALLEL_2D : CAMBLAS_PARALLEL_NONE;
        plan->mc = DEFAULT_MC;
        plan->nc = DEFAULT_NC;
        plan->kc = DEFAULT_KC;
        if (CAMBLAS_DEEP_KC_FP32 > 0 && op->dtype == CAMBLAS_DTYPE_F32 && op->trans_a == 'N' &&
            plan->num_threads >= 16 && op->m > op->n && op->m <= 8192 && op->n <= 2048 &&
            op->k > 1024 && op->k <= 4096)
            plan->kc = CAMBLAS_DEEP_KC_FP32;
        if (CAMBLAS_SMALL_CACHEBLOCK && plan->num_threads >= 16 && op->m <= 2048 && op->n <= 2048 &&
            op->k <= 1024) {
            plan->mc = 64;
            plan->nc = 64;
            plan->kc = (op->dtype == CAMBLAS_DTYPE_F32 ? 128 : 64) *
                       (CAMBLAS_SMALL_CACHEBLOCK == 2 ? 2 : 1);
        }
        /* Opt-in deep rectangular products: reduce the live A/B working set
         * without changing the output-task grid or arithmetic precision. */
        if (CAMBLAS_DEEP_CACHEBLOCK && plan->num_threads >= 16 && op->m > op->n && op->m <= 8192 &&
            op->n <= 2048 && op->k > 1024 && op->k <= 4096)
            plan->kc = CAMBLAS_DEEP_CACHEBLOCK == 3 ? (op->dtype == CAMBLAS_DTYPE_F32 ? 512 : 256)
                       : CAMBLAS_DEEP_CACHEBLOCK == 4
                           ? (op->dtype == CAMBLAS_DTYPE_F32 ? 384 : 192)
                           : (op->dtype == CAMBLAS_DTYPE_F32 ? 128 : 64) /
                                 (CAMBLAS_DEEP_CACHEBLOCK == 2 ? 2 : 1);
        /* The packed SVE body consumes two vector-height rows at once.  Keep
           a positive legacy MR when topology has no usable VL so the ordinary
           C packed path retains its existing allocation/edge contract. */
        int fallback_mr = (op->dtype == CAMBLAS_DTYPE_F32) ? DEFAULT_MR_F32 : DEFAULT_MR_F64;
        int scalar_bits = (op->dtype == CAMBLAS_DTYPE_F32) ? 32 : 64;
        int vector_lanes = 0;
        if (topo->sve_vl_bits >= 128 && topo->sve_vl_bits % 128 == 0)
            vector_lanes = topo->sve_vl_bits / scalar_bits;
        plan->mr = vector_lanes > 0 ? 2 * vector_lanes : fallback_mr;
        plan->nr = PAIRED_PACKED_NR;
#if defined(CAMBLAS_PACKED_FILL_GRID) && CAMBLAS_PACKED_FILL_GRID
        if (ctx->executor)
            camblas_fill_packed_grid(op->m, op->n, plan->num_threads, plan->mr, plan->nr, &plan->mc,
                                     &plan->nc);
#endif
        if (CAMBLAS_DEEP_KC_CACHE_BYTES > 0 && op->dtype == CAMBLAS_DTYPE_F32 &&
            op->trans_a == 'N' && plan->num_threads >= CAMBLAS_DEEP_KC_MIN_THREADS &&
            op->m > op->n && op->m <= 8192 && op->n <= 2048 && op->k > 1024 && op->k <= 4096) {
            uint64_t ma = ((uint64_t)plan->mc + 11) / 12 * 12,
                     nb = ((uint64_t)plan->nc + 7) / 8 * 8;
            uint64_t working =
                (ma * 512 + nb * 512 + (uint64_t)plan->mc * plan->nc) * sizeof(float);
            if (working <= CAMBLAS_DEEP_KC_CACHE_BYTES)
                plan->kc = 512;
        }
        /* For shallow NN products, balance the row groups against aspect
         * ratio. Exact divisibility keeps one disjoint task wave and retains
         * the standard allocation, alignment and edge contracts. */
        if (CAMBLAS_SHALLOW_ASPECT_GRID_FP32 && ctx->executor && op->dtype == CAMBLAS_DTYPE_F32 &&
            op->trans_a == 'N' && op->trans_b == 'N' && plan->num_threads >= 16 &&
            plan->num_threads <= 32 && op->m <= op->n && op->m >= 128 && op->m <= 2048 &&
            op->n >= 128 && op->n <= 2048 && op->k >= 128 && op->k <= 1024) {
            int rows = (uint64_t)op->m * 2 >= (uint64_t)op->n ? 4 : 2;
            if (plan->num_threads % rows == 0 && op->m % rows == 0) {
                int cols = plan->num_threads / rows;
                if (cols >= 1 && cols <= 16 && op->n % cols == 0 &&
                    (op->m / rows) % plan->mr == 0 && (op->n / cols) % plan->nr == 0) {
                    plan->mc = op->m / rows;
                    plan->nc = op->n / cols;
                }
            }
        }
        /* Bounded deep-rectangle policy: preserve one complete wave
         * of disjoint macro tiles, with no partial packed micro-panels. */
        int row_block =
            op->dtype == CAMBLAS_DTYPE_F32 ? CAMBLAS_DEEP_ROW_BLOCK : CAMBLAS_DEEP_ROW_BLOCK_FP64;
        if (row_block > 0 && ctx->executor && op->trans_a == 'N' && op->trans_b == 'N' &&
            plan->num_threads >= 16 && plan->num_threads <= CAMBLAS_DEEP_ROW_BLOCK_MAX_THREADS &&
            op->m > op->n && op->m <= 8192 && op->n <= 2048 && op->k > 1024 && op->k <= 4096) {
            int cm = row_block;
            if (cm > 0 && cm % plan->mr == 0 && op->m % cm == 0) {
                int rows = op->m / cm;
                if (rows >= 1 && rows <= 16 && plan->num_threads % rows == 0) {
                    int cols = plan->num_threads / rows;
                    if (cols >= 1 && cols <= 16 && op->n % cols == 0 &&
                        (op->n / cols) % plan->nr == 0) {
                        plan->mc = cm;
                        plan->nc = op->n / cols;
                    }
                }
            }
        }
        plan->rationale_id = CAMBLAS_RAT_LARGE_PACKED;
        return 0;
    }

    /* Unknown planner mode: fail closed. */
    return -1;
}

/* ---- Observability functions ---- */

int camblas_plan_format(const camblas_plan_t *plan, char *buf, size_t bufsize)
{
    if (!plan || !buf || bufsize == 0)
        return -1;

    int n = snprintf(buf, bufsize,
                     "kernel=%s pack=%s threads=%d parallel=%s "
                     "mc=%d nc=%d kc=%d mr=%d nr=%d rationale=%s",
                     camblas_kernel_name(plan->kernel_id), camblas_pack_name(plan->pack),
                     plan->num_threads, camblas_parallel_name(plan->parallel_scheme), plan->mc,
                     plan->nc, plan->kc, plan->mr, plan->nr, camblas_plan_rationale(plan));

    if (n < 0 || (size_t)n >= bufsize)
        return -1;
    return n;
}

const char *camblas_plan_rationale(const camblas_plan_t *plan)
{
    if (!plan)
        return "unknown";
    switch (plan->rationale_id) {
    case CAMBLAS_RAT_NONE_DISABLED:
        return "planner disabled, scalar reference";
    case CAMBLAS_RAT_FIXED_POLICY:
        return "fixed policy, scalar kernel";
    case CAMBLAS_RAT_SMALL_SCALAR:
        return "small matrix, scalar optimal";
    case CAMBLAS_RAT_MEDIUM_NOPACK:
        return "medium matrix, no-pack kernel";
    case CAMBLAS_RAT_LARGE_PACKED:
        return "large matrix, packed blocked kernel";
    case CAMBLAS_RAT_BATCH:
        return "batched GEMM path";
    case CAMBLAS_RAT_SINGLE_THREAD:
        return "single CPU available";
    case CAMBLAS_RAT_REPRO_BITWISE:
        return "bitwise repro forces scalar 1-thread";
    case CAMBLAS_RAT_SVE_SINGLE_CORE:
        return "explicit single-core SVE path";
    case CAMBLAS_RAT_SVE_FALLBACK:
        return "SVE mode falls back to scalar reference";
    default:
        return "unknown";
    }
}

const char *camblas_kernel_name(int kernel_id)
{
    switch (kernel_id) {
    case CAMBLAS_KERNEL_SCALAR:
        return "scalar";
    case CAMBLAS_KERNEL_NOPACK:
        return "nopack";
    case CAMBLAS_KERNEL_PACKED:
        return "packed";
    case CAMBLAS_KERNEL_BATCHED:
        return "batched";
    case CAMBLAS_KERNEL_SVE:
        return "sve";
    default:
        return "unknown";
    }
}

const char *camblas_parallel_name(int scheme)
{
    switch (scheme) {
    case CAMBLAS_PARALLEL_NONE:
        return "none";
    case CAMBLAS_PARALLEL_ROW:
        return "row";
    case CAMBLAS_PARALLEL_COL:
        return "col";
    case CAMBLAS_PARALLEL_K:
        return "k";
    case CAMBLAS_PARALLEL_2D:
        return "2d";
    default:
        return "unknown";
    }
}

const char *camblas_pack_name(int pack)
{
    switch (pack) {
    case CAMBLAS_PACK_NONE:
        return "none";
    case CAMBLAS_PACK_A:
        return "a";
    case CAMBLAS_PACK_B:
        return "b";
    case CAMBLAS_PACK_BOTH:
        return "both";
    default:
        return "unknown";
    }
}
