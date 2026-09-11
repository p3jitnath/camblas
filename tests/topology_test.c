/*
 * Topology discovery test + cpulist parser unit tests (v2).
 *
 * Tests:
 *   1. cpulist parser: valid, boundary, malformed, overflow, empty, strict grammar
 *   2. cpuset operations: set, test, count, format, intersect, null/boundary
 *   3. status string: immutable literals, format API, reentrancy
 *   4. topology discovery: allowed CPUs, NUMA intersections, SVE VL, status
 */
#define _GNU_SOURCE 1
#include "camblas_topology.h"
#include "../src/sve_runtime.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sched.h>
#include <pthread.h>

#ifdef CAMBLAS_TOPOLOGY_TESTING
extern int camblas_topology_test_cpu_model_from_stream(FILE *stream, char *buf, size_t bufsize);
#endif

static int tests_run = 0;
static int tests_pass = 0;
static int tests_fail = 0;

#define TEST(name)                         \
    do {                                   \
        tests_run++;                       \
        if (name()) {                      \
            tests_pass++;                  \
        } else {                           \
            tests_fail++;                  \
            printf("  FAIL: %s\n", #name); \
        }                                  \
    } while (0)

#define ASSERT(cond, msg)                           \
    do {                                            \
        if (!(cond)) {                              \
            printf("    ASSERT failed: %s\n", msg); \
            return 0;                               \
        }                                           \
    } while (0)

/* ============================================================
 * Cpulist parser tests — strict grammar
 * ============================================================ */

static int test_parse_simple(void)
{
    camblas_cpuset_t set;
    ASSERT(camblas_parse_cpulist("0-3", &set) == 0, "parse 0-3");
    ASSERT(camblas_cpuset_count(&set) == 4, "count 4");
    ASSERT(camblas_cpuset_test(&set, 0), "cpu 0");
    ASSERT(camblas_cpuset_test(&set, 3), "cpu 3");
    ASSERT(!camblas_cpuset_test(&set, 4), "not cpu 4");
    return 1;
}

static int test_parse_multi_range(void)
{
    camblas_cpuset_t set;
    ASSERT(camblas_parse_cpulist("0-71,144-215", &set) == 0, "parse multi-range");
    ASSERT(camblas_cpuset_count(&set) == 144, "count 144");
    return 1;
}

static int test_parse_single(void)
{
    camblas_cpuset_t set;
    ASSERT(camblas_parse_cpulist("5", &set) == 0, "parse single");
    ASSERT(camblas_cpuset_count(&set) == 1, "count 1");
    return 1;
}

static int test_parse_single_comma(void)
{
    camblas_cpuset_t set;
    ASSERT(camblas_parse_cpulist("0,5,10", &set) == 0, "parse 0,5,10");
    ASSERT(camblas_cpuset_count(&set) == 3, "count 3");
    return 1;
}

static int test_parse_surrounding_ws(void)
{
    camblas_cpuset_t set;
    ASSERT(camblas_parse_cpulist("  0-3  ", &set) == 0, "surrounding ws ok");
    ASSERT(camblas_cpuset_count(&set) == 4, "count 4");
    return 1;
}

static int test_parse_trailing_newline(void)
{
    camblas_cpuset_t set;
    ASSERT(camblas_parse_cpulist("0-3\n", &set) == 0, "trailing newline ok");
    ASSERT(camblas_cpuset_count(&set) == 4, "count 4");
    return 1;
}

static int test_parse_boundary_max(void)
{
    camblas_cpuset_t set;
    char buf[32];
    snprintf(buf, sizeof(buf), "%d", CAMBLAS_MAX_CPUS - 1);
    ASSERT(camblas_parse_cpulist(buf, &set) == 0, "max cpu ok");
    ASSERT(camblas_cpuset_count(&set) == 1, "count 1");
    return 1;
}

static int test_parse_overflow_single(void)
{
    camblas_cpuset_t set;
    char buf[32];
    snprintf(buf, sizeof(buf), "%d", CAMBLAS_MAX_CPUS);
    ASSERT(camblas_parse_cpulist(buf, &set) == -2, "overflow single");
    return 1;
}

static int test_parse_overflow_range(void)
{
    camblas_cpuset_t set;
    char buf[64];
    snprintf(buf, sizeof(buf), "0-%d", CAMBLAS_MAX_CPUS);
    ASSERT(camblas_parse_cpulist(buf, &set) == -2, "overflow range");
    return 1;
}

/* Strict grammar: rejected cases */

static int test_parse_leading_comma(void)
{
    camblas_cpuset_t set;
    ASSERT(camblas_parse_cpulist(",0-3", &set) == -1, "leading comma rejected");
    return 1;
}

static int test_parse_trailing_comma(void)
{
    camblas_cpuset_t set;
    ASSERT(camblas_parse_cpulist("0-3,", &set) == -1, "trailing comma rejected");
    return 1;
}

static int test_parse_repeated_comma(void)
{
    camblas_cpuset_t set;
    ASSERT(camblas_parse_cpulist("0,,3", &set) == -1, "repeated comma rejected");
    return 1;
}

static int test_parse_repeated_dash(void)
{
    camblas_cpuset_t set;
    ASSERT(camblas_parse_cpulist("0--3", &set) == -1, "repeated dash rejected");
    return 1;
}

static int test_parse_missing_comma(void)
{
    camblas_cpuset_t set;
    ASSERT(camblas_parse_cpulist("0-3 4-7", &set) == -1, "missing comma rejected");
    return 1;
}

static int test_parse_junk_suffix(void)
{
    camblas_cpuset_t set;
    ASSERT(camblas_parse_cpulist("0-3abc", &set) == -1, "junk suffix rejected");
    return 1;
}

static int test_parse_junk_prefix(void)
{
    camblas_cpuset_t set;
    ASSERT(camblas_parse_cpulist("abc0-3", &set) == -1, "junk prefix rejected");
    return 1;
}

static int test_parse_negative(void)
{
    camblas_cpuset_t set;
    ASSERT(camblas_parse_cpulist("-5", &set) == -1, "negative rejected");
    return 1;
}

static int test_parse_reversed(void)
{
    camblas_cpuset_t set;
    ASSERT(camblas_parse_cpulist("5-3", &set) == -1, "reversed range rejected");
    return 1;
}

static int test_parse_letters(void)
{
    camblas_cpuset_t set;
    ASSERT(camblas_parse_cpulist("abc", &set) == -1, "letters rejected");
    return 1;
}

static int test_parse_trailing_dash(void)
{
    camblas_cpuset_t set;
    ASSERT(camblas_parse_cpulist("0-", &set) == -1, "trailing dash rejected");
    return 1;
}

static int test_parse_empty(void)
{
    camblas_cpuset_t set;
    ASSERT(camblas_parse_cpulist("", &set) == -3, "empty -> -3");
    ASSERT(camblas_parse_cpulist("   ", &set) == -3, "whitespace -> -3");
    ASSERT(camblas_parse_cpulist(NULL, &set) == -3, "NULL -> -3");
    return 1;
}

static int test_parse_null_set(void)
{
    /* NULL set pointer should not crash */
    int rc = camblas_parse_cpulist("0-3", NULL);
    ASSERT(rc == -3, "NULL set -> -3");
    return 1;
}

/* ============================================================
 * Cpuset operation tests — null/boundary
 * ============================================================ */

static int test_cpuset_format(void)
{
    camblas_cpuset_t set;
    camblas_cpuset_zero(&set);
    camblas_cpuset_set(&set, 0);
    camblas_cpuset_set(&set, 1);
    camblas_cpuset_set(&set, 2);
    camblas_cpuset_set(&set, 5);
    camblas_cpuset_set(&set, 6);
    char buf[256];
    int rc = camblas_cpuset_format(&set, buf, sizeof(buf));
    ASSERT(rc > 0, "format ok");
    ASSERT(strcmp(buf, "0-2,5-6") == 0, "format '0-2,5-6'");
    return 1;
}

static int test_cpuset_format_empty(void)
{
    camblas_cpuset_t set;
    camblas_cpuset_zero(&set);
    char buf[256];
    int rc = camblas_cpuset_format(&set, buf, sizeof(buf));
    ASSERT(rc > 0, "format ok");
    ASSERT(strcmp(buf, "(empty)") == 0, "format '(empty)'");
    return 1;
}

static int test_cpuset_format_null(void)
{
    char buf[256];
    ASSERT(camblas_cpuset_format(NULL, buf, sizeof(buf)) == -1, "null set -> -1");
    ASSERT(camblas_cpuset_format(NULL, NULL, 0) == -1, "null buf -> -1");
    camblas_cpuset_t set;
    camblas_cpuset_zero(&set);
    ASSERT(camblas_cpuset_format(&set, buf, 0) == -1, "zero bufsize -> -1");
    return 1;
}

static int test_cpuset_null_ops(void)
{
    camblas_cpuset_zero(NULL);   /* should not crash */
    camblas_cpuset_set(NULL, 0); /* should not crash */
    ASSERT(camblas_cpuset_test(NULL, 0) == 0, "null test -> 0");
    ASSERT(camblas_cpuset_count(NULL) == 0, "null count -> 0");
    return 1;
}

static int test_cpuset_boundary(void)
{
    camblas_cpuset_t set;
    camblas_cpuset_zero(&set);
    /* Out-of-range CPU indices silently ignored */
    camblas_cpuset_set(&set, -1);
    camblas_cpuset_set(&set, CAMBLAS_MAX_CPUS);
    camblas_cpuset_set(&set, CAMBLAS_MAX_CPUS + 100);
    ASSERT(camblas_cpuset_count(&set) == 0, "no bits set from OOB");
    ASSERT(camblas_cpuset_test(&set, -1) == 0, "test -1 -> 0");
    ASSERT(camblas_cpuset_test(&set, CAMBLAS_MAX_CPUS) == 0, "test MAX -> 0");
    return 1;
}

static int test_cpuset_intersect(void)
{
    camblas_cpuset_t a, b, c;
    camblas_parse_cpulist("0-71,144-215", &a);
    camblas_parse_cpulist("0-143", &b);
    camblas_cpuset_intersect(&c, &a, &b);
    ASSERT(camblas_cpuset_count(&c) == 72, "intersect count 72");
    ASSERT(camblas_cpuset_test(&c, 0), "cpu 0 in intersection");
    ASSERT(camblas_cpuset_test(&c, 71), "cpu 71 in intersection");
    ASSERT(!camblas_cpuset_test(&c, 72), "cpu 72 not in intersection");
    ASSERT(!camblas_cpuset_test(&c, 144), "cpu 144 not in intersection");
    return 1;
}

static int test_cpuset_intersect_disjoint(void)
{
    camblas_cpuset_t a, b, c;
    camblas_parse_cpulist("0-71", &a);
    camblas_parse_cpulist("72-143", &b);
    camblas_cpuset_intersect(&c, &a, &b);
    ASSERT(camblas_cpuset_count(&c) == 0, "disjoint -> empty");
    return 1;
}

static int test_cpuset_intersect_aliasing(void)
{
    /* dst aliases a */
    camblas_cpuset_t a, b;
    camblas_parse_cpulist("0-71", &a);
    camblas_parse_cpulist("0-35", &b);
    camblas_cpuset_intersect(&a, &a, &b);
    ASSERT(camblas_cpuset_count(&a) == 36, "aliased dst=a count 36");
    return 1;
}

/* ============================================================
 * Status string tests — reentrancy, immutability
 * ============================================================ */

static int test_status_str_immutable(void)
{
    /* All return values should be immutable literals (no mutable global) */
    const char *s0 = camblas_topo_status_str(CAMBLAS_TOPO_OK);
    const char *s1 = camblas_topo_status_str(CAMBLAS_TOPO_FATAL);
    const char *s2 = camblas_topo_status_str(CAMBLAS_TOPO_OVERFLOW);
    const char *s3 = camblas_topo_status_str(CAMBLAS_TOPO_NO_NUMA);
    ASSERT(s0 && strcmp(s0, "OK") == 0, "OK string");
    ASSERT(s1 && strcmp(s1, "FATAL(allowed-mask)") == 0, "FATAL string");
    ASSERT(s2 && strcmp(s2, "OVERFLOW(cpu-index)") == 0, "OVERFLOW string");
    ASSERT(s3 && strcmp(s3, "NO_NUMA") == 0, "NO_NUMA string");
    /* Calling again should return the same pointer (immutable) */
    ASSERT(camblas_topo_status_str(CAMBLAS_TOPO_OK) == s0, "same pointer for OK");
    return 1;
}

static int test_status_format(void)
{
    char buf[256];
    int rc = camblas_topo_status_format(CAMBLAS_TOPO_OK, buf, sizeof(buf));
    ASSERT(rc > 0, "format OK");
    ASSERT(strcmp(buf, "OK") == 0, "OK");

    int combined = CAMBLAS_TOPO_NO_NUMA | CAMBLAS_TOPO_NO_SVE;
    rc = camblas_topo_status_format(combined, buf, sizeof(buf));
    ASSERT(rc > 0, "format combined");
    ASSERT(strstr(buf, "NO_NUMA") != NULL, "contains NO_NUMA");
    ASSERT(strstr(buf, "NO_SVE") != NULL, "contains NO_SVE");
    ASSERT(strstr(buf, "|") != NULL, "contains separator");

    /* NULL buffer */
    ASSERT(camblas_topo_status_format(0, NULL, 0) == -1, "null buf -> -1");
    return 1;
}

static int test_sve_vl_decoder(void)
{
#ifdef __aarch64__
    ASSERT(camblas_sve_vl_bits_from_prctl(16) == 128, "minimum SVE VL accepted");
    ASSERT(camblas_sve_vl_bits_from_prctl(128 | PR_SVE_VL_INHERIT) == 1024,
           "GET inherit flag accepted");
    ASSERT(camblas_sve_vl_bits_from_prctl(CAMBLAS_SVE_VL_MAX_BYTES) == CAMBLAS_SVE_VL_MAX_BYTES * 8,
           "maximum UAPI SVE VL accepted");
    ASSERT(camblas_sve_vl_bits_from_prctl(0) == 0, "zero length rejected");
    ASSERT(camblas_sve_vl_bits_from_prctl(-1) == 0, "negative prctl error rejected");
    ASSERT(camblas_sve_vl_bits_from_prctl(15) == 0, "below-minimum length rejected");
    ASSERT(camblas_sve_vl_bits_from_prctl(17) == 0, "non-granular length rejected");
    ASSERT(camblas_sve_vl_bits_from_prctl(CAMBLAS_SVE_VL_MAX_BYTES + 16) == 0,
           "above-maximum length rejected");
    ASSERT(camblas_sve_vl_bits_from_prctl(16 | (1 << 16)) == 0, "unknown flag rejected");
    ASSERT(camblas_sve_vl_bits_from_prctl(16 | (1 << 18)) == 0,
           "SET on-exec flag rejected for GET");
    ASSERT(camblas_sve_vl_bits_from_prctl(16 | (1U << 30)) == 0, "high unknown flag rejected");
#else
    ASSERT(camblas_sve_vl_bits_from_prctl(16) == 0, "non-AArch64 has no SVE VL");
#endif
    return 1;
}

/* Reentrancy test: call status_str from multiple threads */
static void *thread_status_fn(void *arg)
{
    (void)arg;
    for (int i = 0; i < 10000; i++) {
        const char *s = camblas_topo_status_str(CAMBLAS_TOPO_OK);
        if (!s || strcmp(s, "OK") != 0)
            return (void *)1;
        s = camblas_topo_status_str(CAMBLAS_TOPO_FATAL);
        if (!s || strcmp(s, "FATAL(allowed-mask)") != 0)
            return (void *)1;
    }
    return NULL;
}

static int test_status_str_reentrant(void)
{
    pthread_t threads[4];
    for (int i = 0; i < 4; i++)
        pthread_create(&threads[i], NULL, thread_status_fn, NULL);
    int ok = 1;
    for (int i = 0; i < 4; i++) {
        void *rc;
        pthread_join(threads[i], &rc);
        if (rc != NULL)
            ok = 0;
    }
    ASSERT(ok, "all threads got correct immutable strings");
    return 1;
}

#ifdef CAMBLAS_TOPOLOGY_TESTING
/* ============================================================
 * CPU model parser tests — exact fields and fail-closed MIDR values
 * ============================================================ */

static int cpu_model_from_text(const void *data, size_t length, char *out, size_t out_size)
{
    FILE *stream = tmpfile();
    if (!stream)
        return -2;
    if (fwrite(data, 1, length, stream) != length || fflush(stream) != 0 ||
        fseek(stream, 0, SEEK_SET) != 0) {
        fclose(stream);
        return -2;
    }
    int rc = camblas_topology_test_cpu_model_from_stream(stream, out, out_size);
    if (fclose(stream) != 0)
        return -1;
    return rc;
}

static int test_cpu_model_valid_records(void)
{
    static const char arm_cpuinfo[] = "CPU implementer : 0x41\n"
                                      "CPU architecture : 8\n"
                                      "CPU part : 0xd4f\n"
                                      "CPU implementer : 0x41\n"
                                      "CPU architecture : 8\n"
                                      "CPU part : 0xd4f\n";
    static const char x86_cpuinfo[] = "model name : Synthetic CPU 9000\n"
                                      "model name : Synthetic CPU 9000\n";
    char model[256];

    ASSERT(cpu_model_from_text(arm_cpuinfo, sizeof(arm_cpuinfo) - 1, model, sizeof(model)) == 0,
           "valid repeated MIDR records accepted");
    ASSERT(strcmp(model, "ARM Neoverse-V2") == 0, "known MIDR mapping preserved");
    ASSERT(cpu_model_from_text(x86_cpuinfo, sizeof(x86_cpuinfo) - 1, model, sizeof(model)) == 0,
           "valid exact model-name records accepted");
    ASSERT(strcmp(model, "Synthetic CPU 9000") == 0, "exact model-name value preserved");
    return 1;
}

static int test_cpu_model_rejects_ambiguous_numeric_metadata(void)
{
    static const char *const invalid_records[] = {
        "CPU implementer : 0x41junk\nCPU architecture : 8\nCPU part : 0xd4f\n",
        "CPU implementer : 0x100\nCPU architecture : 8\nCPU part : 0xd4f\n",
        "CPU implementer : -1\nCPU architecture : 8\nCPU part : 0xd4f\n",
        "CPU implementer : 0x41\nCPU architecture : 8\n",
        "CPU implementer : 0x41\nCPU architecture : 8x\nCPU part : 0xd4f\n",
        "CPU implementer : 0x41\nCPU architecture : 8\nCPU part : 0xd4fgarbage\n",
        "CPU implementer : 0x41\nCPU architecture : 8\nCPU part : 0xd4f\n"
        "CPU part : 0xd49\n",
        "CPU implementer-extra : 0x41\nCPU architecture : 8\n"
        "CPU part : 0xd4f\n",
        "model name-extra : misleading\n"};
    char model[256];

    for (size_t i = 0; i < sizeof(invalid_records) / sizeof(invalid_records[0]); i++) {
        memset(model, 0xA5, sizeof(model));
        ASSERT(cpu_model_from_text(invalid_records[i], strlen(invalid_records[i]), model,
                                   sizeof(model)) != 0,
               "malformed/ambiguous model metadata rejected");
        ASSERT(model[0] == '\0', "rejected metadata leaves no partial model");
    }

    static const unsigned char embedded_nul[] =
        "CPU implementer : 0x41\0junk\nCPU architecture : 8\n"
        "CPU part : 0xd4f\n";
    memset(model, 0xA5, sizeof(model));
    ASSERT(cpu_model_from_text(embedded_nul, sizeof(embedded_nul) - 1, model, sizeof(model)) != 0,
           "embedded NUL rejected");
    ASSERT(model[0] == '\0', "embedded NUL leaves no partial model");
    return 1;
}

static int test_cpu_model_requires_complete_architecture_tuple(void)
{
    static const char incomplete[] = "CPU implementer : 0x41\n"
                                     "CPU part : 0xd4f\n";
    static const char wrong_architecture[] = "CPU implementer : 0x41\n"
                                             "CPU architecture : 7\n"
                                             "CPU part : 0xd4f\n";
    char model[256];

    memset(model, 0xA5, sizeof(model));
    ASSERT(cpu_model_from_text(incomplete, sizeof(incomplete) - 1, model, sizeof(model)) != 0,
           "missing architecture rejected");
    ASSERT(model[0] == '\0', "missing architecture leaves no model");
    memset(model, 0xA5, sizeof(model));
    ASSERT(cpu_model_from_text(wrong_architecture, sizeof(wrong_architecture) - 1, model,
                               sizeof(model)) != 0,
           "non-AArch64 architecture rejected");
    ASSERT(model[0] == '\0', "non-AArch64 metadata leaves no model");
    return 1;
}
#endif

/* ============================================================
 * Topology discovery test
 * ============================================================ */

static int test_topology_discover(void)
{
    camblas_topology_t topo;
    int status = camblas_topology_discover(&topo);

    printf("\n  --- Topology discovery ---\n");
    printf("  Status: %d (%s)\n", status, camblas_topo_status_str(status));
    camblas_topology_print(&topo, stdout);

    ASSERT(!CAMBLAS_TOPO_IS_FATAL(status), "no fatal errors");

    ASSERT(((status & CAMBLAS_TOPO_NO_SVE) != 0) == (topo.sve_vl_bits == 0),
           "NO_SVE agrees with zero VL");
    ASSERT(((status & CAMBLAS_TOPO_NO_MODEL) != 0) == (topo.cpu_model[0] == '\0'),
           "NO_MODEL agrees with empty model");

    ASSERT(topo.n_allowed_cpus > 0, "allowed > 0");
    ASSERT(topo.n_online_cpus >= 0, "online nonnegative");
    ASSERT(topo.n_online_cpus == 0 || topo.n_allowed_cpus <= topo.n_online_cpus,
           "positive online covers allowed");
    ASSERT(camblas_cpuset_count(&topo.allowed_cpus) == topo.n_allowed_cpus,
           "allowed scalar equals bit count");
    int allowed_nbits = 0;
    for (int cpu = 0; cpu < CAMBLAS_MAX_CPUS; cpu++)
        if (camblas_cpuset_test(&topo.allowed_cpus, cpu))
            allowed_nbits = cpu + 1;
    ASSERT(topo.allowed_cpus.nbits == allowed_nbits, "allowed nbits is canonical");

    if (topo.n_numa_nodes > 0) {
        camblas_cpuset_t seen;
        camblas_cpuset_zero(&seen);
        for (int i = 0; i < topo.n_numa_nodes; i++) {
            ASSERT(topo.numa_nodes[i].n_cpus > 0, "node non-empty intersection");
            ASSERT(topo.numa_nodes[i].n_cpus == camblas_cpuset_count(&topo.numa_nodes[i].cpus),
                   "node scalar equals bit count");
            ASSERT(topo.numa_nodes[i].n_cpus_physical >= topo.numa_nodes[i].n_cpus,
                   "physical count covers intersection");
            for (int cpu = 0; cpu < topo.numa_nodes[i].cpus.nbits; cpu++) {
                if (camblas_cpuset_test(&topo.numa_nodes[i].cpus, cpu)) {
                    ASSERT(camblas_cpuset_test(&topo.allowed_cpus, cpu), "node cpu in allowed");
                    ASSERT(!camblas_cpuset_test(&seen, cpu), "node intersections are disjoint");
                    camblas_cpuset_set(&seen, cpu);
                }
            }
        }
        for (int i = 1; i < topo.n_numa_nodes; i++) {
            ASSERT(topo.numa_nodes[i].node_id > topo.numa_nodes[i - 1].node_id,
                   "sorted by node_id");
        }
        int total = 0;
        for (int i = 0; i < topo.n_numa_nodes; i++)
            total += topo.numa_nodes[i].n_cpus;
        ASSERT(total <= topo.n_allowed_cpus, "sum <= allowed");
    }

    return 1;
}

static int test_topology_null(void)
{
    ASSERT(camblas_topology_discover(NULL) == CAMBLAS_TOPO_FATAL, "null -> FATAL");
    camblas_topology_print(NULL, stdout); /* should not crash */
    camblas_topology_print(NULL, NULL);   /* should not crash */
    return 1;
}

static int test_topology_print_malformed_snapshot(void)
{
    camblas_topology_t topo;
    memset(&topo, 0, sizeof(topo));
    memset(topo.cpu_model, 'X', sizeof(topo.cpu_model));
    topo.n_online_cpus = 0x5a5a5a5a;
    topo.n_numa_nodes = CAMBLAS_MAX_NUMA_NODES + 1;

    FILE *stream = tmpfile();
    ASSERT(stream != NULL, "tmpfile for malformed snapshot");
    camblas_topology_print(&topo, stream);
    ASSERT(fclose(stream) == 0, "close malformed snapshot stream");
    return 1;
}

/* ============================================================
 * Main
 * ============================================================ */

int main(void)
{
    printf("=== CAMBLAS Topology Unit Tests (v2) ===\n\n");

    printf("--- Cpulist parser: valid ---\n");
    TEST(test_parse_simple);
    TEST(test_parse_multi_range);
    TEST(test_parse_single);
    TEST(test_parse_single_comma);
    TEST(test_parse_surrounding_ws);
    TEST(test_parse_trailing_newline);
    TEST(test_parse_boundary_max);

    printf("\n--- Cpulist parser: overflow ---\n");
    TEST(test_parse_overflow_single);
    TEST(test_parse_overflow_range);

    printf("\n--- Cpulist parser: strict grammar rejections ---\n");
    TEST(test_parse_leading_comma);
    TEST(test_parse_trailing_comma);
    TEST(test_parse_repeated_comma);
    TEST(test_parse_repeated_dash);
    TEST(test_parse_missing_comma);
    TEST(test_parse_junk_suffix);
    TEST(test_parse_junk_prefix);
    TEST(test_parse_negative);
    TEST(test_parse_reversed);
    TEST(test_parse_letters);
    TEST(test_parse_trailing_dash);

    printf("\n--- Cpulist parser: empty/null ---\n");
    TEST(test_parse_empty);
    TEST(test_parse_null_set);

    printf("\n--- Cpuset operations ---\n");
    TEST(test_cpuset_format);
    TEST(test_cpuset_format_empty);
    TEST(test_cpuset_format_null);
    TEST(test_cpuset_null_ops);
    TEST(test_cpuset_boundary);
    TEST(test_cpuset_intersect);
    TEST(test_cpuset_intersect_disjoint);
    TEST(test_cpuset_intersect_aliasing);

    printf("\n--- Status string (immutable/reentrant) ---\n");
    TEST(test_status_str_immutable);
    TEST(test_status_format);
    TEST(test_sve_vl_decoder);
    TEST(test_status_str_reentrant);

#ifdef CAMBLAS_TOPOLOGY_TESTING
    printf("\n--- CPU model parser (strict optional metadata) ---\n");
    TEST(test_cpu_model_valid_records);
    TEST(test_cpu_model_rejects_ambiguous_numeric_metadata);
    TEST(test_cpu_model_requires_complete_architecture_tuple);
#endif

    printf("\n--- Topology discovery ---\n");
    TEST(test_topology_discover);
    TEST(test_topology_null);
    TEST(test_topology_print_malformed_snapshot);

    printf("\n=== Summary: %d run, %d pass, %d fail ===\n", tests_run, tests_pass, tests_fail);
    return tests_fail == 0 ? 0 : 1;
}
