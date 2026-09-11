/*
 * CAMBLAS topology discovery — hardened API (v2).
 *
 * Discovers the process's allowed CPU set, NUMA topology, and SVE vector
 * length at runtime using POSIX/Linux APIs only (no libnuma, hwloc, or
 * popen). Foundation for allocation-aware planning.
 *
 * Discovery only — no performance measurement, no affinity setting.
 *
 * RETURN STATUS SEMANTICS:
 *   camblas_topology_discover() returns a bitfield status:
 *     0                         = fully successful
 *     CAMBLAS_TOPO_FATAL        = fatal: could not determine allowed CPU mask
 *     CAMBLAS_TOPO_OVERFLOW     = fatal: allowed CPU index exceeds CAMBLAS_MAX_CPUS
 *     CAMBLAS_TOPO_INDETERMINATE= fatal: absolute CPU cap reached, mask completeness unknown
 *     CAMBLAS_TOPO_NO_NUMA      = optional: NUMA topology unavailable
 *     CAMBLAS_TOPO_NUMA_INCOMPLETE = optional: NUMA discovery encountered errors, nodes cleared
 *     CAMBLAS_TOPO_NO_SVE       = optional: strict SVE VL query unavailable
 *     CAMBLAS_TOPO_NO_MODEL     = optional: CPU model name unavailable
 *   Fatal errors mean the topology struct is not safe for planning; the
 *   snapshot remains zeroed/cleared and optional fields are not populated.
 *   Optional errors mean some metadata is missing but the allowed CPU set
 *   and (if available) NUMA node intersections are valid. NO_SVE gates only
 *   SVE-specific consumers; NO_MODEL does not make planning unsafe and is
 *   enforced only by consumers that require model provenance.
 *   In a discovered snapshot, n_allowed_cpus is at least one whenever no
 *   fatal allowed-mask status is returned, and it equals the set-bit count of
 *   allowed_cpus. The allowed_cpus.nbits field is canonical (zero for an
 *   empty set, otherwise max-set-CPU plus one). sve_vl_bits is zero for the
 *   optional NO_SVE result, otherwise it is a Linux UAPI value from 128
 *   through 65536 bits in 128-bit increments. n_online_cpus is reference
 *   metadata: zero means unavailable and it is not a planning input; when it
 *   is positive, it must be at least n_allowed_cpus because the allowed set
 *   is a subset of the system-wide online CPUs. An inconsistent positive
 *   discovery value is cleared to zero, while a caller-supplied inconsistent
 *   snapshot is rejected by topology-dependent planning.
 *   CPU model metadata is optional: it is reported only when an exact bounded
 *   model-name field or a complete, consistent AArch64 MIDR field tuple is
 *   available. Malformed, truncated, overflowing, or ambiguous model input
 *   leaves cpu_model empty and contributes CAMBLAS_TOPO_NO_MODEL.
 *
 * NUMA NODE SEMANTICS:
 *   Each camblas_numa_node_t exposes the INTERSECTION of the node's physical
 *   CPU list with the process's allowed CPU set. Nodes whose intersection
 *   is empty are NOT included. Physical totals (n_cpus_physical,
 *   memory_bytes_physical) are preserved separately for reference.
 *   The planner should use the intersection CPUs, not physical totals.
 *   If any NUMA structural/CPU discovery error occurs, ALL NUMA nodes are
 *   cleared and CAMBLAS_TOPO_NUMA_INCOMPLETE is returned (fail-closed).
 *   When n_numa_nodes is nonzero, each node has a nonempty canonical cpuset,
 *   n_cpus equals its set-bit count, that set is a disjoint subset of
 *   allowed_cpus, and n_cpus_physical is at least n_cpus. Zero nodes means
 *   NUMA metadata is unavailable or optional and is valid.
 *   Memory is reference-only; an unavailable or malformed MemTotal field is
 *   represented as zero without exposing a partial value.
 *
 * REENTRANCY:
 *   All functions are reentrant. No function uses mutable global/static
 *   state. camblas_topo_status_str() returns immutable string literals only.
 *
 * PRECONDITIONS:
 *   All cpuset functions require non-NULL set pointers (except
 *   camblas_cpuset_test which also accepts NULL, returning 0).
 *   camblas_cpuset_set/test: cpu must be in [0, CAMBLAS_MAX_CPUS-1];
 *   out-of-range indices are silently ignored by set, return 0 by test.
 */
#ifndef CAMBLAS_TOPOLOGY_H
#define CAMBLAS_TOPOLOGY_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdio.h>
#include <stdint.h>

/*
 * Maximum number of CPUs representable in the public API.
 * Large enough for 288-core Grace; bounded for stack safety.
 * If the allowed mask contains a CPU index >= this, CAMBLAS_TOPO_OVERFLOW
 * is returned.
 */
#define CAMBLAS_MAX_CPUS 1024

/*
 * Maximum number of NUMA nodes representable in the public API.
 * This bounds the node array capacity, NOT valid node IDs.
 * Node IDs >= this are stored as-is if capacity allows; node IDs >= INT_MAX
 * are rejected as malformed.
 */
#define CAMBLAS_MAX_NUMA_NODES 128

/*
 * Absolute cap for sched_getaffinity probing. If the kernel reports a mask
 * requiring more than this many CPUs, we return CAMBLAS_TOPO_INDETERMINATE
 * rather than pretending we checked for overflow. This cap is far above any
 * current hardware (65536 CPUs) but prevents unbounded allocation.
 */
#define CAMBLAS_TOPO_AFFINITY_ABSOLUTE_CAP 65536

/*
 * Topology discovery status codes (bitfield, combinable for optional errors).
 */
#define CAMBLAS_TOPO_OK 0
#define CAMBLAS_TOPO_FATAL (1 << 0)           /* cannot determine allowed mask */
#define CAMBLAS_TOPO_OVERFLOW (1 << 1)        /* CPU index exceeds CAMBLAS_MAX_CPUS */
#define CAMBLAS_TOPO_INDETERMINATE (1 << 2)   /* absolute cap reached, mask unknown */
#define CAMBLAS_TOPO_NO_NUMA (1 << 3)         /* NUMA topology unavailable (no /sys/node) */
#define CAMBLAS_TOPO_NUMA_INCOMPLETE (1 << 4) /* NUMA error: nodes cleared (fail-closed) */
#define CAMBLAS_TOPO_NO_SVE (1 << 5)          /* SVE VL unavailable */
#define CAMBLAS_TOPO_NO_MODEL (1 << 6)        /* CPU model name unavailable */

/*
 * Check if status has a fatal error (planning unsafe).
 */
#define CAMBLAS_TOPO_IS_FATAL(s) \
    ((s) & (CAMBLAS_TOPO_FATAL | CAMBLAS_TOPO_OVERFLOW | CAMBLAS_TOPO_INDETERMINATE))

/*
 * CPU set: a bitmask of allowed CPUs.
 */
typedef struct {
    uint64_t bits[(CAMBLAS_MAX_CPUS + 63) / 64];
    int nbits; /* max CPU index + 1, or 0 if empty */
} camblas_cpuset_t;

/*
 * NUMA node info.
 * cpus = intersection(node physical cpus, process allowed cpus).
 * Physical totals are for reference only.
 */
typedef struct {
    int node_id;                    /* NUMA node number from sysfs */
    int n_cpus;                     /* count of CPUs in intersection */
    camblas_cpuset_t cpus;          /* intersection with allowed_cpus */
    int n_cpus_physical;            /* physical CPU count on this node */
    uint64_t memory_bytes_physical; /* physical memory (0 if unavailable or malformed) */
} camblas_numa_node_t;

/*
 * System topology.
 */
typedef struct {
    camblas_cpuset_t allowed_cpus;
    int n_allowed_cpus;

    int n_numa_nodes; /* 0 when NUMA is unavailable/incomplete */
    camblas_numa_node_t numa_nodes[CAMBLAS_MAX_NUMA_NODES];

    int sve_vl_bits; /* 0 for optional NO_SVE; otherwise 128..65536, step 128 */
    char cpu_model[256];
    int n_online_cpus; /* reference-only; 0 unavailable; positive >= allowed */
} camblas_topology_t;

/* ---- Status string (reentrant: returns immutable literal) ---- */

/*
 * Return a human-readable description of a topology status code.
 * Returns an immutable string literal. No mutable global state.
 * For combined statuses, returns the first matching label.
 */
const char *camblas_topo_status_str(int status);

/*
 * Format a status code into a caller-owned buffer.
 * Returns characters written, or -1 if buffer too small.
 * This is the preferred API for combined statuses.
 */
int camblas_topo_status_format(int status, char *buf, size_t bufsize);

/* ---- CPU set operations ---- */

/*
 * Zero a CPU set. set must be non-NULL.
 */
void camblas_cpuset_zero(camblas_cpuset_t *set);

/*
 * Set a CPU bit. set must be non-NULL.
 * cpu outside [0, CAMBLAS_MAX_CPUS-1] is silently ignored.
 */
void camblas_cpuset_set(camblas_cpuset_t *set, int cpu);

/*
 * Test if a CPU is in the set.
 * set may be NULL (returns 0).
 * cpu outside [0, CAMBLAS_MAX_CPUS-1] returns 0.
 */
int camblas_cpuset_test(const camblas_cpuset_t *set, int cpu);

/*
 * Count set bits. set must be non-NULL.
 */
int camblas_cpuset_count(const camblas_cpuset_t *set);

/*
 * Format a CPU set as a compact range string (e.g., "0-71,144-215").
 * set and buf must be non-NULL, bufsize > 0.
 * Returns characters written, or -1 if buffer too small or invalid args.
 */
int camblas_cpuset_format(const camblas_cpuset_t *set, char *buf, size_t bufsize);

/*
 * Compute the intersection of two CPU sets: dst = a & b.
 * All pointers must be non-NULL. dst may alias a or b.
 */
void camblas_cpuset_intersect(camblas_cpuset_t *dst, const camblas_cpuset_t *a,
                              const camblas_cpuset_t *b);

/* ---- Cpulist parser (exposed for testing) ---- */

/*
 * Parse a CPU list string (e.g., "0-71,144-215") into a cpuset.
 *
 * Grammar: [whitespace] cpu_spec {"," cpu_spec} [whitespace/newline]
 *          cpu_spec := number | number "-" number
 *          number := digit {digit}
 * Surrounding whitespace and a trailing newline are allowed.
 * Leading/trailing/repeated commas, repeated dashes, junk suffixes,
 * and ambiguous whitespace between entries are rejected.
 *
 * Returns:
 *   0  on success
 *  -1  on parse error (malformed grammar)
 *  -2  on overflow (CPU index >= CAMBLAS_MAX_CPUS)
 *  -3  on empty input (no CPUs parsed)
 */
int camblas_parse_cpulist(const char *str, camblas_cpuset_t *set);

/* ---- Topology discovery ---- */

/*
 * Discover the system topology.
 * topo must be non-NULL.
 * Returns a status bitfield (see CAMBLAS_TOPO_* above).
 */
int camblas_topology_discover(camblas_topology_t *topo);

/*
 * Print a human-readable topology summary to the given stream.
 * topo and stream must be non-NULL. This diagnostic consumer bounds the
 * caller-supplied model string and NUMA-node count, so a malformed optional
 * snapshot cannot make the printer read beyond the fixed topology object.
 */
void camblas_topology_print(const camblas_topology_t *topo, FILE *stream);

#ifdef __cplusplus
}
#endif

#endif /* CAMBLAS_TOPOLOGY_H */
