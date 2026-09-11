/*
 * CAMBLAS topology discovery — hardened implementation (v2).
 *
 * Hardening (v2):
 *   - No mutable global/static state; status_str returns immutable literals.
 *   - Dynamic sched_getaffinity: grow until success, checked allocation,
 *     absolute cap (CAMBLAS_TOPO_AFFINITY_ABSOLUTE_CAP), distinct
 *     CAMBLAS_TOPO_INDETERMINATE status if cap reached.
 *   - NUMA fail-closed: structural/CPU metadata errors clear ALL nodes and
 *     return CAMBLAS_TOPO_NUMA_INCOMPLETE. Node ID vs capacity distinguished.
 *   - Strict cpulist grammar: reject malformed commas/dashes/junk.
 *   - File reading: single-descriptor read-through, no reopen/lseek race,
 *     and fail-closed read/close results.
 *   - meminfo: bounded exact-field search, validate numeric+kB, strtoull,
 *     checked uint64; unavailable or malformed optional memory remains zero.
 *   - CPU model: exact /proc/cpuinfo field boundaries, checked numeric MIDR
 *     values, complete/consistent field sets, and NO_MODEL on ambiguity.
 *   - Public cpuset functions: documented preconditions, null-safe where stated.
 */
#define _GNU_SOURCE 1
#include "camblas_topology.h"
#include "sve_runtime.h"

#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <dirent.h>
#include <unistd.h>
#include <fcntl.h>
#include <sched.h>
#include <ctype.h>
#include <limits.h>

#ifdef __aarch64__
#include <sys/prctl.h>
#endif

/* ============================================================
 * Status string (reentrant: immutable literals or caller buffer)
 * ============================================================ */

const char *camblas_topo_status_str(int status)
{
    if (status == CAMBLAS_TOPO_OK)
        return "OK";
    if (status & CAMBLAS_TOPO_FATAL)
        return "FATAL(allowed-mask)";
    if (status & CAMBLAS_TOPO_OVERFLOW)
        return "OVERFLOW(cpu-index)";
    if (status & CAMBLAS_TOPO_INDETERMINATE)
        return "INDETERMINATE(mask-cap)";
    if (status & CAMBLAS_TOPO_NUMA_INCOMPLETE)
        return "NUMA_INCOMPLETE";
    if (status & CAMBLAS_TOPO_NO_NUMA)
        return "NO_NUMA";
    if (status & CAMBLAS_TOPO_NO_SVE)
        return "NO_SVE";
    if (status & CAMBLAS_TOPO_NO_MODEL)
        return "NO_MODEL";
    return "UNKNOWN";
}

int camblas_topo_status_format(int status, char *buf, size_t bufsize)
{
    if (!buf || bufsize == 0)
        return -1;
    if (status == CAMBLAS_TOPO_OK) {
        return snprintf(buf, bufsize, "OK");
    }
    int pos = 0;
    buf[0] = '\0';

    struct {
        int bit;
        const char *name;
    } flags[] = {
        {CAMBLAS_TOPO_FATAL, "FATAL"},
        {CAMBLAS_TOPO_OVERFLOW, "OVERFLOW"},
        {CAMBLAS_TOPO_INDETERMINATE, "INDETERMINATE"},
        {CAMBLAS_TOPO_NO_NUMA, "NO_NUMA"},
        {CAMBLAS_TOPO_NUMA_INCOMPLETE, "NUMA_INCOMPLETE"},
        {CAMBLAS_TOPO_NO_SVE, "NO_SVE"},
        {CAMBLAS_TOPO_NO_MODEL, "NO_MODEL"},
    };
    int first = 1;
    for (size_t i = 0; i < sizeof(flags) / sizeof(flags[0]); i++) {
        if (status & flags[i].bit) {
            int n = snprintf(buf + pos, bufsize - pos, "%s%s", first ? "" : "|", flags[i].name);
            if (n < 0 || (size_t)(pos + n) >= bufsize)
                return -1;
            pos += n;
            first = 0;
        }
    }
    if (pos == 0) {
        int n = snprintf(buf, bufsize, "UNKNOWN");
        if (n < 0 || (size_t)n >= bufsize)
            return -1;
        return n;
    }
    return pos;
}

/* ============================================================
 * CPU set operations (documented preconditions)
 * ============================================================ */

void camblas_cpuset_zero(camblas_cpuset_t *set)
{
    if (!set)
        return;
    memset(set->bits, 0, sizeof(set->bits));
    set->nbits = 0;
}

void camblas_cpuset_set(camblas_cpuset_t *set, int cpu)
{
    if (!set || cpu < 0 || cpu >= CAMBLAS_MAX_CPUS)
        return;
    set->bits[cpu / 64] |= (1ULL << (cpu % 64));
    if (cpu + 1 > set->nbits)
        set->nbits = cpu + 1;
}

int camblas_cpuset_test(const camblas_cpuset_t *set, int cpu)
{
    if (!set || cpu < 0 || cpu >= CAMBLAS_MAX_CPUS)
        return 0;
    return (set->bits[cpu / 64] >> (cpu % 64)) & 1;
}

int camblas_cpuset_count(const camblas_cpuset_t *set)
{
    if (!set)
        return 0;
    int count = 0;
    int nwords = (int)(sizeof(set->bits) / sizeof(set->bits[0]));
    for (int i = 0; i < nwords; i++) {
        count += __builtin_popcountll(set->bits[i]);
    }
    return count;
}

int camblas_cpuset_format(const camblas_cpuset_t *set, char *buf, size_t bufsize)
{
    if (!set || !buf || bufsize == 0)
        return -1;
    buf[0] = '\0';
    int pos = 0;
    int i = 0;
    while (i < set->nbits && i < CAMBLAS_MAX_CPUS) {
        if (!camblas_cpuset_test(set, i)) {
            i++;
            continue;
        }
        int start = i;
        while (i < set->nbits && camblas_cpuset_test(set, i))
            i++;
        int end = i - 1;
        int n;
        if (start == end)
            n = snprintf(buf + pos, bufsize - pos, pos == 0 ? "%d" : ",%d", start);
        else
            n = snprintf(buf + pos, bufsize - pos, pos == 0 ? "%d-%d" : ",%d-%d", start, end);
        if (n < 0 || (size_t)(pos + n) >= bufsize)
            return -1;
        pos += n;
    }
    if (pos == 0) {
        int n = snprintf(buf, bufsize, "(empty)");
        if (n < 0 || (size_t)n >= bufsize)
            return -1;
        return n;
    }
    return pos;
}

void camblas_cpuset_intersect(camblas_cpuset_t *dst, const camblas_cpuset_t *a,
                              const camblas_cpuset_t *b)
{
    if (!dst || !a || !b)
        return;
    /* Use temp in case dst aliases a or b */
    camblas_cpuset_t tmp;
    memset(&tmp, 0, sizeof(tmp));
    int nwords = (int)(sizeof(tmp.bits) / sizeof(tmp.bits[0]));
    for (int i = 0; i < nwords; i++)
        tmp.bits[i] = a->bits[i] & b->bits[i];
    tmp.nbits = 0;
    for (int i = CAMBLAS_MAX_CPUS - 1; i >= 0; i--) {
        if ((tmp.bits[i / 64] >> (i % 64)) & 1) {
            tmp.nbits = i + 1;
            break;
        }
    }
    *dst = tmp;
}

/* ============================================================
 * Cpulist parser (strict grammar)
 * ============================================================ */

/*
 * Strict grammar:
 *   input := [ws] spec { "," spec } [ws] [newline]
 *   spec  := number [ "-" number ]
 *   number := digit { digit }
 *   ws    := " " | "\t"
 *
 * Rejected: leading comma, trailing comma, repeated comma, repeated dash,
 * missing comma between specs, junk after number, whitespace inside spec,
 * negative, reversed range, empty range.
 */
int camblas_parse_cpulist(const char *str, camblas_cpuset_t *set)
{
    if (!str || !set)
        return -3;
    camblas_cpuset_zero(set);

    const char *p = str;

    /* Skip leading whitespace */
    while (*p == ' ' || *p == '\t')
        p++;

    /* Empty input */
    if (*p == '\0' || *p == '\n' || *p == '\r')
        return -3;

    int parsed_any = 0;

    while (*p && *p != '\n' && *p != '\r') {
        /* Must be at a digit (start of a spec) */
        if (*p == ',')
            return -1; /* leading/repeated comma */

        if (!isdigit((unsigned char)*p))
            return -1; /* junk where number expected */

        /* Parse first number */
        errno = 0;
        char *endptr;
        long start = strtol(p, &endptr, 10);
        if (errno == ERANGE)
            return -1;
        if (endptr == p)
            return -1;
        p = endptr;

        long end = start;
        if (*p == '-') {
            p++;
            if (*p == '-')
                return -1; /* repeated dash */
            if (!isdigit((unsigned char)*p))
                return -1; /* no digit after dash */
            errno = 0;
            end = strtol(p, &endptr, 10);
            if (errno == ERANGE)
                return -1;
            if (endptr == p)
                return -1;
            p = endptr;
        }

        /* Validate range */
        if (start < 0 || end < 0)
            return -1;
        if (start > end)
            return -1;
        if (end >= CAMBLAS_MAX_CPUS)
            return -2;

        for (long c = start; c <= end; c++)
            camblas_cpuset_set(set, (int)c);
        parsed_any = 1;

        /* Skip trailing whitespace before comma or end */
        while (*p == ' ' || *p == '\t')
            p++;

        if (*p == '\0' || *p == '\n' || *p == '\r')
            break;

        if (*p != ',')
            return -1; /* junk after number */

        /* Consume comma */
        p++;

        /* Skip whitespace after comma before next number */
        while (*p == ' ' || *p == '\t')
            p++;

        /* Must be a digit or end (trailing comma check happens below) */
        if (*p == '\0' || *p == '\n' || *p == '\r')
            return -1; /* trailing comma */
        if (*p == ',')
            return -1; /* repeated comma */
        if (!isdigit((unsigned char)*p))
            return -1; /* junk after comma */
    }

    if (!parsed_any)
        return -3;
    return 0;
}

/* ============================================================
 * File reading (single-descriptor, no reopen/lseek race)
 * ============================================================ */

/*
 * Read a file into a buffer with a single descriptor. Detects truncation
 * by checking if the read filled the buffer and more bytes remain.
 * Handles EINTR and empty files.
 *
 * Returns:
 *   0   on success (buf is null-terminated, *out_len set)
 *  -1   on invalid arguments, open/read error, or close error
 *  -2   on truncation (file larger than bufsize-1; buf contains partial data)
 */
static int read_file_full(const char *path, char *buf, size_t bufsize, size_t *out_len)
{
    if (!buf || bufsize == 0)
        return -1;
    buf[0] = '\0';
    if (out_len)
        *out_len = 0;
    if (!path)
        return -1;

    int fd = open(path, O_RDONLY);
    if (fd < 0)
        return -1;

    size_t total = 0;
    int rc = 0;
    while (total < bufsize - 1) {
        ssize_t n = read(fd, buf + total, bufsize - 1 - total);
        if (n < 0) {
            if (errno == EINTR)
                continue;
            rc = -1;
            break;
        }
        if (n == 0)
            break; /* EOF */
        total += (size_t)n;
    }

    /* Check if there's more data (truncation detection without reopen) */
    if (rc == 0 && total == bufsize - 1) {
        /* Buffer is full — try reading one more byte to check */
        char extra;
        ssize_t n = 0;
        do {
            n = read(fd, &extra, 1);
        } while (n < 0 && errno == EINTR);

        if (n < 0) {
            rc = -1;
        } else if (n > 0) {
            rc = -2; /* truncated */
        }
        /* n == 0: exact fit, no truncation */
    }

    if (close(fd) != 0)
        rc = -1;
    buf[total] = '\0';
    if (rc == -1) {
        /* Never expose a partial value after a read or close error. */
        buf[0] = '\0';
        if (out_len)
            *out_len = 0;
    } else if (out_len) {
        *out_len = total;
    }
    return rc;
}

/* ============================================================
 * Allowed CPU set discovery (dynamic, no truncation)
 * ============================================================ */

/*
 * Discover allowed CPUs using dynamically-sized sched_getaffinity.
 *
 * Algorithm:
 *   1. Start with ncpus = CPU_SETSIZE (typically 1024).
 *   2. Loop: allocate mask, call sched_getaffinity.
 *      - On EINVAL (mask too small), free and double ncpus.
 *      - On success, break.
 *      - On other error, return FATAL.
 *   3. Check ncpus against absolute cap. If exceeded, return INDETERMINATE.
 *   4. Iterate all CPUs in the mask. If any CPU >= CAMBLAS_MAX_CPUS, return OVERFLOW.
 *   5. Populate the public cpuset; on overflow, clear the partial mask and
 *      count so the fatal result cannot masquerade as a usable snapshot.
 *
 * Returns: CAMBLAS_TOPO_OK, CAMBLAS_TOPO_OVERFLOW, CAMBLAS_TOPO_INDETERMINATE,
 *          or CAMBLAS_TOPO_FATAL.
 */
static int discover_allowed_cpus(camblas_cpuset_t *out_set, int *out_count)
{
    camblas_cpuset_zero(out_set);
    *out_count = 0;

    int ncpus = CPU_SETSIZE; /* Start with the libc default (typically 1024) */
    cpu_set_t *mask = NULL;

    for (;;) {
        /* Check against absolute cap before allocating */
        if (ncpus > CAMBLAS_TOPO_AFFINITY_ABSOLUTE_CAP) {
            return CAMBLAS_TOPO_INDETERMINATE;
        }

        /* Allocate with overflow check */
        if (ncpus > CAMBLAS_TOPO_AFFINITY_ABSOLUTE_CAP) {
            return CAMBLAS_TOPO_INDETERMINATE;
        }

        mask = CPU_ALLOC(ncpus);
        if (!mask) {
            return CAMBLAS_TOPO_FATAL;
        }
        size_t size = CPU_ALLOC_SIZE(ncpus);

        if (sched_getaffinity(0, size, mask) == 0) {
            break; /* success */
        }

        CPU_FREE(mask);
        mask = NULL;

        if (errno != EINVAL) {
            return CAMBLAS_TOPO_FATAL;
        }

        /* Mask too small; double with overflow check */
        if (ncpus > CAMBLAS_TOPO_AFFINITY_ABSOLUTE_CAP / 2) {
            return CAMBLAS_TOPO_INDETERMINATE;
        }
        ncpus *= 2;
    }

    /* Count CPUs and check for overflow against public capacity */
    int count = 0;
    int overflow = 0;
    size_t size = CPU_ALLOC_SIZE(ncpus);

    for (int i = 0; i < ncpus; i++) {
        if (CPU_ISSET_S(i, size, mask)) {
            if (i >= CAMBLAS_MAX_CPUS) {
                overflow = 1;
            } else {
                camblas_cpuset_set(out_set, i);
                count++;
            }
        }
    }
    CPU_FREE(mask);

    if (overflow) {
        /* The returned status is fatal.  Do not leave a partial positive
         * count that a caller could mistake for a complete planning mask if
         * it fails to propagate the separate status bitfield. */
        camblas_cpuset_zero(out_set);
        *out_count = 0;
        return CAMBLAS_TOPO_OVERFLOW;
    }

    if (count == 0) {
        return CAMBLAS_TOPO_FATAL;
    }

    *out_count = count;
    return CAMBLAS_TOPO_OK;
}

/* ============================================================
 * NUMA discovery (fail-closed)
 * ============================================================ */

static int compare_node_id(const void *a, const void *b)
{
    const camblas_numa_node_t *na = (const camblas_numa_node_t *)a;
    const camblas_numa_node_t *nb = (const camblas_numa_node_t *)b;
    if (na->node_id < nb->node_id)
        return -1;
    if (na->node_id > nb->node_id)
        return 1;
    return 0;
}

/*
 * Parse one NUL-terminated /proc- or NUMA-sysfs-style MemTotal line.
 *
 * Returns:
 *   1   line is MemTotal and was parsed successfully
 *   0   line is not MemTotal
 *  -1   line claims to be MemTotal but is malformed or overflows
 */
static int parse_memtotal_line(char *line, int expected_node_id, uint64_t *out_bytes)
{
    static const char key[] = "MemTotal:";
    if (!line || !out_bytes)
        return -1;

    char *p = line;
    if (strncmp(p, key, sizeof(key) - 1) == 0) {
        p += sizeof(key) - 1;
    } else if (strncmp(p, "Node ", 5) == 0) {
        /* Linux NUMA meminfo prefixes each field with its node number. */
        p += 5;
        if (!isdigit((unsigned char)*p))
            return 0;
        errno = 0;
        char *node_end;
        long node_id = strtol(p, &node_end, 10);
        if (errno == ERANGE || node_end == p || *node_end != ' ' || node_id != expected_node_id) {
            return 0;
        }
        p = node_end + 1;
        if (strncmp(p, key, sizeof(key) - 1) != 0)
            return 0;
        p += sizeof(key) - 1;
    } else {
        return 0;
    }

    while (*p == ' ' || *p == '\t')
        p++;
    if (!isdigit((unsigned char)*p))
        return -1;

    errno = 0;
    char *endptr;
    unsigned long long mem_kb = strtoull(p, &endptr, 10);
    if (errno == ERANGE || endptr == p)
        return -1;

    while (*endptr == ' ' || *endptr == '\t')
        endptr++;
    if ((endptr[0] != 'k' && endptr[0] != 'K') || (endptr[1] != 'b' && endptr[1] != 'B')) {
        return -1;
    }
    endptr += 2;
    while (*endptr == ' ' || *endptr == '\t')
        endptr++;
    if (*endptr != '\0')
        return -1;

    if (mem_kb > UINT64_MAX / 1024ULL)
        return -1;
    *out_bytes = (uint64_t)mem_kb * 1024ULL;
    return 1;
}

/*
 * Parse the complete bounded contents of a NUMA meminfo file.
 *
 * The file is optional metadata: any read, truncation, embedded-NUL,
 * missing, duplicate, malformed, or overflowing MemTotal value returns -1;
 * callers retain the documented zero value in that case.  A successful
 * parse requires exactly one well-formed MemTotal field and rejects trailing
 * non-whitespace text after the kB unit.
 */
static int parse_memtotal_bytes(char *buf, size_t len, int expected_node_id, uint64_t *out_bytes)
{
    if (!buf || !out_bytes || len == 0)
        return -1;
    *out_bytes = 0;

    char *line = buf;
    char *buf_end = buf + len;
    int found = 0;
    uint64_t parsed_bytes = 0;

    while (line < buf_end) {
        char *eol = memchr(line, '\n', (size_t)(buf_end - line));
        char *line_end = eol ? eol : buf_end;
        size_t line_len = (size_t)(line_end - line);

        /* Do not let a NUL hide a suffix or a second field in this line. */
        if (memchr(line, '\0', line_len) != NULL)
            return -1;

        if (eol)
            *eol = '\0';
        uint64_t line_bytes = 0;
        int line_rc = parse_memtotal_line(line, expected_node_id, &line_bytes);
        if (eol)
            *eol = '\n';

        if (line_rc < 0)
            return -1;
        if (line_rc > 0) {
            if (found)
                return -1;
            found = 1;
            parsed_bytes = line_bytes;
        }

        line = eol ? eol + 1 : buf_end;
    }

    if (!found)
        return -1;
    *out_bytes = parsed_bytes;
    return 0;
}

/*
 * Discover NUMA topology. Fail-closed: any error clears ALL nodes.
 *
 * Errors that trigger fail-closed:
 *   - cpulist read truncation
 *   - cpulist parse error or overflow
 *   - duplicate node ID
 *   - malformed node directory entry (nodeN where N is not a valid number)
 *   - capacity overflow (more nodes than CAMBLAS_MAX_NUMA_NODES)
 *
 * Returns: 0 on success, CAMBLAS_TOPO_NO_NUMA (no /sys/node),
 *          or CAMBLAS_TOPO_NUMA_INCOMPLETE (fail-closed).
 */
static int discover_numa(camblas_topology_t *topo, const camblas_cpuset_t *allowed_cpus)
{
    topo->n_numa_nodes = 0;

    DIR *dir = opendir("/sys/devices/system/node");
    if (!dir) {
        if (errno == ENOENT || errno == ENOTDIR)
            return CAMBLAS_TOPO_NO_NUMA;
        return CAMBLAS_TOPO_NUMA_INCOMPLETE;
    }

    /* Temporary storage for all discovered nodes */
    camblas_numa_node_t temp_nodes[CAMBLAS_MAX_NUMA_NODES];
    int n_temp = 0;
    int fail_closed = 0;

    struct dirent *entry;
    for (;;) {
        /* POSIX leaves errno unchanged at end-of-directory. */
        errno = 0;
        entry = readdir(dir);
        if (!entry) {
            if (errno != 0)
                fail_closed = 1;
            break;
        }

        /* Check for "node" prefix */
        if (strncmp(entry->d_name, "node", 4) != 0)
            continue;

        /* Parse node ID — must be a valid non-negative integer */
        const char *node_id_text = entry->d_name + 4;
        if (*node_id_text == '\0') {
            fail_closed = 1;
            break;
        }
        for (const char *p = node_id_text; *p != '\0'; p++) {
            if (!isdigit((unsigned char)*p)) {
                fail_closed = 1;
                break;
            }
        }
        if (fail_closed)
            break;

        char *endptr;
        errno = 0;
        long node_id_l = strtol(node_id_text, &endptr, 10);
        if (errno == ERANGE || endptr == node_id_text || *endptr != '\0') {
            /* Malformed node directory entry — fail-closed */
            fail_closed = 1;
            break;
        }
        if (node_id_l < 0 || node_id_l >= INT_MAX) {
            /* Invalid node ID — fail-closed */
            fail_closed = 1;
            break;
        }
        int node_id = (int)node_id_l;

        /* Check capacity (node ID vs capacity are separate concepts,
         * but we can only store CAMBLAS_MAX_NUMA_NODES entries) */
        if (n_temp >= CAMBLAS_MAX_NUMA_NODES) {
            /* Too many nodes for our capacity — fail-closed */
            fail_closed = 1;
            break;
        }

        /* Check for duplicate node ID */
        for (int i = 0; i < n_temp; i++) {
            if (temp_nodes[i].node_id == node_id) {
                fail_closed = 1;
                break;
            }
        }
        if (fail_closed)
            break;

        /* Read physical CPU list */
        char path[512];
        int path_len =
            snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist", node_id);
        if (path_len < 0 || (size_t)path_len >= sizeof(path)) {
            fail_closed = 1;
            break;
        }
        char buf[8192];
        size_t flen = 0;
        int file_rc = read_file_full(path, buf, sizeof(buf), &flen);

        if (file_rc == -2) {
            /* Truncation — fail-closed */
            fail_closed = 1;
            break;
        }
        if (file_rc == -1) {
            /* Any read or close error makes the NUMA snapshot incomplete. */
            fail_closed = 1;
            break;
        }

        /* Parse CPU list */
        camblas_cpuset_t physical_cpus;
        int pr = camblas_parse_cpulist(buf, &physical_cpus);
        if (pr == -2) {
            /* Overflow — fail-closed */
            fail_closed = 1;
            break;
        }
        if (pr == -1) {
            /* Parse error — fail-closed */
            fail_closed = 1;
            break;
        }
        if (pr == -3) {
            /* Empty cpulist — skip node (no CPUs) */
            continue;
        }

        /* Compute intersection with allowed CPUs */
        camblas_cpuset_t intersection;
        camblas_cpuset_intersect(&intersection, &physical_cpus, allowed_cpus);
        int n_intersect = camblas_cpuset_count(&intersection);

        /* Skip nodes with empty intersection (not planner nodes) */
        if (n_intersect == 0)
            continue;

        /* Populate node entry */
        camblas_numa_node_t *node = &temp_nodes[n_temp];
        memset(node, 0, sizeof(*node));
        node->node_id = node_id;
        node->cpus = intersection;
        node->n_cpus = n_intersect;
        node->n_cpus_physical = camblas_cpuset_count(&physical_cpus);

        /* Read memory size (physical) */
        path_len = snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/meminfo", node_id);
        if (path_len < 0 || (size_t)path_len >= sizeof(path)) {
            fail_closed = 1;
            break;
        }
        char membuf[8192];
        size_t memlen = 0;
        node->memory_bytes_physical = 0;

        if (read_file_full(path, membuf, sizeof(membuf), &memlen) == 0) {
            /* Memory is reference-only; reject, rather than partially parse,
             * malformed or ambiguous optional metadata and retain zero. */
            (void)parse_memtotal_bytes(membuf, memlen, node_id, &node->memory_bytes_physical);
        }

        n_temp++;
    }
    if (closedir(dir) != 0)
        fail_closed = 1;

    if (fail_closed) {
        /* Clear all nodes — fail-closed */
        topo->n_numa_nodes = 0;
        return CAMBLAS_TOPO_NUMA_INCOMPLETE;
    }

    if (n_temp == 0) {
        return CAMBLAS_TOPO_NO_NUMA;
    }

    /* Sort by node_id for deterministic output */
    qsort(temp_nodes, n_temp, sizeof(camblas_numa_node_t), compare_node_id);

    /* Copy sorted nodes into topology */
    for (int i = 0; i < n_temp; i++) {
        topo->numa_nodes[i] = temp_nodes[i];
    }
    topo->n_numa_nodes = n_temp;

    return CAMBLAS_TOPO_OK;
}

/* ============================================================
 * SVE vector length
 * ============================================================ */

static int discover_sve_vl(void)
{
#ifdef __aarch64__
    return camblas_sve_vl_bits_from_prctl((int)prctl(PR_SVE_GET_VL));
#else
    return 0;
#endif
}

/* ============================================================
 * CPU model (no popen/lscpu; /proc/cpuinfo MIDR or unknown)
 * ============================================================ */

/*
 * fgets() returns a successful prefix when a line is larger than its
 * destination.  The sentinel bytes let this check also reject embedded NUL
 * bytes, which would otherwise make strlen() hide the remainder of a line.
 * A final line without a newline is accepted only when fgets() has observed
 * EOF; a full buffer with no newline is therefore never treated as complete.
 */
static int cpuinfo_line_complete(const char *line, size_t linebuf_size, int at_eof)
{
    if (!line || linebuf_size == 0)
        return 0;

    size_t first_nul = 0;
    while (first_nul < linebuf_size && line[first_nul] != '\0')
        first_nul++;
    if (first_nul == linebuf_size)
        return 0;

    /* A second NUL before the sentinel is an embedded input NUL. */
    for (size_t i = first_nul + 1; i < linebuf_size; i++) {
        if (line[i] == '\0')
            return 0;
    }

    if (memchr(line, '\n', first_nul) != NULL)
        return 1;
    return at_eof != 0;
}

/*
 * Return the exact value span for one /proc/cpuinfo field.
 *
 * The field name must be followed by ':' or horizontal whitespace followed
 * by ':'. This avoids treating "CPU part-extra" as "CPU part". The span
 * excludes the line ending and surrounding horizontal whitespace, but does
 * not otherwise interpret the value.
 *
 * Returns 1 for a matching well-formed field boundary, 0 for a different
 * field, and -1 for a matching field with a malformed/empty value.
 */
static int cpuinfo_field_span(const char *line, const char *field, const char **value_begin,
                              const char **value_end)
{
    if (!line || !field || !value_begin || !value_end)
        return -1;

    size_t field_len = strlen(field);
    if (strncmp(line, field, field_len) != 0)
        return 0;

    const char *p = line + field_len;
    if (*p != ':' && *p != ' ' && *p != '\t')
        return 0;
    while (*p == ' ' || *p == '\t')
        p++;
    if (*p != ':')
        return -1;
    p++;
    while (*p == ' ' || *p == '\t')
        p++;

    const char *end = strchr(p, '\n');
    if (!end)
        end = p + strlen(p);
    while (end > p && (end[-1] == ' ' || end[-1] == '\t' || end[-1] == '\r'))
        end--;
    if (p == end)
        return -1;

    *value_begin = p;
    *value_end = end;
    return 1;
}

/* Parse one unsigned, fully-consumed /proc/cpuinfo numeric token. */
static int cpuinfo_parse_uint(const char *begin, const char *end, int allow_hex_prefix,
                              unsigned max_value, unsigned *out_value)
{
    if (!begin || !end || !out_value || begin >= end)
        return -1;

    int base = 10;
    if (allow_hex_prefix && (size_t)(end - begin) >= 2 && begin[0] == '0' &&
        (begin[1] == 'x' || begin[1] == 'X')) {
        base = 16;
        begin += 2;
        if (begin >= end)
            return -1;
    }

    uint64_t value = 0;
    for (const char *p = begin; p < end; p++) {
        unsigned digit;
        unsigned char c = (unsigned char)*p;
        if (c >= '0' && c <= '9') {
            digit = (unsigned)(c - '0');
        } else if (base == 16 && c >= 'a' && c <= 'f') {
            digit = (unsigned)(c - 'a') + 10U;
        } else if (base == 16 && c >= 'A' && c <= 'F') {
            digit = (unsigned)(c - 'A') + 10U;
        } else {
            return -1;
        }
        if (digit >= (unsigned)base || value > ((uint64_t)max_value - digit) / (unsigned)base)
            return -1;
        value = value * (unsigned)base + digit;
    }

    *out_value = (unsigned)value;
    return 1;
}

static int cpuinfo_parse_uint_field(const char *line, const char *field, int allow_hex_prefix,
                                    unsigned max_value, unsigned *out_value)
{
    const char *begin;
    const char *end;
    int rc = cpuinfo_field_span(line, field, &begin, &end);
    if (rc <= 0)
        return rc;
    return cpuinfo_parse_uint(begin, end, allow_hex_prefix, max_value, out_value);
}

/* Repeated CPU records must agree; a heterogeneous/ambiguous snapshot is not
 * safe to turn into one model string. */
static int cpuinfo_update_uint_field(const char *line, const char *field, int allow_hex_prefix,
                                     unsigned max_value, int *seen, unsigned *value)
{
    unsigned parsed;
    int rc;

    if (!seen || !value)
        return -1;
    rc = cpuinfo_parse_uint_field(line, field, allow_hex_prefix, max_value, &parsed);
    if (rc <= 0)
        return rc;
    if (*seen && *value != parsed)
        return -1;
    *seen = 1;
    *value = parsed;
    return 1;
}

/* Parse a complete /proc/cpuinfo stream without owning or closing it. */
static int discover_cpu_model_from_stream(FILE *f, char *buf, size_t bufsize)
{
    enum {
        CPUINFO_IMPLEMENTER_MAX = 0xffU,
        CPUINFO_ARCHITECTURE_MAX = 0xffU,
        CPUINFO_PART_MAX = 0xfffU
    };

    if (!f || !buf || bufsize == 0)
        return -1;
    buf[0] = '\0';

    char line[512];
    char model_candidate[sizeof(((camblas_topology_t *)0)->cpu_model)];
    memset(model_candidate, 0, sizeof(model_candidate));
    size_t model_len = 0;
    int have_model = 0;
    int invalid_input = 0;
    int impl_seen = 0, part_seen = 0, arch_seen = 0;
    unsigned impl = 0, part = 0, arch = 0;

    for (;;) {
        memset(line, 0xA5, sizeof(line));
        if (!fgets(line, sizeof(line), f)) {
            if (ferror(f))
                invalid_input = 1;
            break;
        }
        if (!cpuinfo_line_complete(line, sizeof(line), feof(f) != 0)) {
            invalid_input = 1;
            break;
        }

        int rc = cpuinfo_update_uint_field(line, "CPU implementer", 1, CPUINFO_IMPLEMENTER_MAX,
                                           &impl_seen, &impl);
        if (rc < 0) {
            invalid_input = 1;
            break;
        }
        rc = cpuinfo_update_uint_field(line, "CPU architecture", 0, CPUINFO_ARCHITECTURE_MAX,
                                       &arch_seen, &arch);
        if (rc < 0) {
            invalid_input = 1;
            break;
        }
        rc = cpuinfo_update_uint_field(line, "CPU part", 1, CPUINFO_PART_MAX, &part_seen, &part);
        if (rc < 0) {
            invalid_input = 1;
            break;
        }

        /* Check for an exact x86-style "model name" field. */
        const char *model_begin;
        const char *model_end;
        int model_rc = cpuinfo_field_span(line, "model name", &model_begin, &model_end);
        if (model_rc == 0)
            model_rc = cpuinfo_field_span(line, "Model name", &model_begin, &model_end);
        if (model_rc < 0) {
            invalid_input = 1;
            break;
        }
        if (model_rc > 0) {
            size_t len = (size_t)(model_end - model_begin);
            if (len == 0 || len >= sizeof(model_candidate)) {
                invalid_input = 1;
                break;
            }
            if (!have_model) {
                memcpy(model_candidate, model_begin, len);
                model_candidate[len] = '\0';
                model_len = len;
                have_model = 1;
            } else if (model_len != len || memcmp(model_candidate, model_begin, len) != 0) {
                invalid_input = 1;
                break;
            }
        }
    }

    if (invalid_input || ferror(f)) {
        buf[0] = '\0';
        return -1;
    }

    if (have_model) {
        if (model_len >= bufsize)
            return -1;
        memcpy(buf, model_candidate, model_len + 1);
        return 0;
    }

    /* A MIDR-derived model requires one complete, consistent field tuple. */
    if (!impl_seen || !part_seen || !arch_seen || arch != 8U)
        return -1;

    if (impl == 0x41U) {
        const char *known_model = NULL;
        switch (part) {
        case 0xd4fU:
            known_model = "ARM Neoverse-V2";
            break;
        case 0xd49U:
            known_model = "ARM Neoverse-N2";
            break;
        case 0xd0cU:
            known_model = "ARM Neoverse-N1";
            break;
        case 0xd4aU:
            known_model = "ARM Neoverse-E1";
            break;
        default:
            break;
        }
        int n;
        if (known_model) {
            n = snprintf(buf, bufsize, "%s", known_model);
        } else {
            n = snprintf(buf, bufsize, "ARM (impl=0x%x part=0x%x)", impl, part);
        }
        if (n < 0 || (size_t)n >= bufsize) {
            buf[0] = '\0';
            return -1;
        }
        return 0;
    }

    int n = snprintf(buf, bufsize, "AArch64 (impl=0x%x part=0x%x)", impl, part);
    if (n < 0 || (size_t)n >= bufsize) {
        buf[0] = '\0';
        return -1;
    }
    return 0;
}

/*
 * Discover the optional model and make read/close failures indistinguishable
 * from other unavailable model metadata to the public status path.
 */
static int discover_cpu_model(char *buf, size_t bufsize)
{
    if (!buf || bufsize == 0)
        return -1;
    buf[0] = '\0';

    FILE *f = fopen("/proc/cpuinfo", "r");
    if (!f)
        return -1;

    int rc = discover_cpu_model_from_stream(f, buf, bufsize);
    int read_error = ferror(f) != 0;
    int close_error = fclose(f) != 0;
    if (read_error || close_error) {
        buf[0] = '\0';
        return -1;
    }
    return rc;
}

#ifdef CAMBLAS_TOPOLOGY_TESTING
/* Test-only seam; it adds no public ABI symbol in production builds. */
int camblas_topology_test_cpu_model_from_stream(FILE *stream, char *buf, size_t bufsize)
{
    return discover_cpu_model_from_stream(stream, buf, bufsize);
}
#endif

/* ============================================================
 * Online CPU count
 * ============================================================ */

static int discover_online_cpus_from_sysconf(void)
{
    errno = 0;
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n <= 0 || n > INT_MAX)
        return 0;
    return (int)n;
}

static int discover_online_cpus(void)
{
    char buf[8192];
    size_t len = 0;
    int file_rc = read_file_full("/sys/devices/system/cpu/online", buf, sizeof(buf), &len);
    if (file_rc == 0) {
        camblas_cpuset_t online;
        if (camblas_parse_cpulist(buf, &online) == 0) {
            int count = camblas_cpuset_count(&online);
            if (count > 0)
                return count;
        }
        /* A present, fully-read but malformed/empty/overflowing sysfs value
         * must not be hidden by a less-specific fallback. */
        return 0;
    }
    if (file_rc == -2)
        return 0;

    /* Only an unavailable/failed sysfs read uses the POSIX fallback. */
    return discover_online_cpus_from_sysconf();
}

/* ============================================================
 * Main discovery function
 * ============================================================ */

int camblas_topology_discover(camblas_topology_t *topo)
{
    if (!topo)
        return CAMBLAS_TOPO_FATAL;
    memset(topo, 0, sizeof(*topo));

    int status = CAMBLAS_TOPO_OK;

    /* 1. Allowed CPU set (fatal if fails) */
    int allowed_rc = discover_allowed_cpus(&topo->allowed_cpus, &topo->n_allowed_cpus);
    if (allowed_rc == CAMBLAS_TOPO_OVERFLOW) {
        status |= CAMBLAS_TOPO_OVERFLOW;
    } else if (allowed_rc == CAMBLAS_TOPO_INDETERMINATE) {
        status |= CAMBLAS_TOPO_INDETERMINATE;
    } else if (allowed_rc == CAMBLAS_TOPO_FATAL) {
        status |= CAMBLAS_TOPO_FATAL;
        return status; /* fatal: return early */
    }

    /* A fatal allowed-mask result leaves the zeroed snapshot unsafe.  Do not
       populate unrelated optional fields after the fatal boundary: callers
       that accidentally inspect the struct without its status cannot mistake
       a partial result for a usable discovery snapshot. */
    if (CAMBLAS_TOPO_IS_FATAL(status))
        return status;

    /* 2. Online CPU count (reference only).  The system-wide online set must
       contain the process's allowed set.  If a race or inconsistent source
       reports a smaller positive count, discard it as unavailable rather than
       publishing contradictory metadata. */
    topo->n_online_cpus = discover_online_cpus();
    if (topo->n_online_cpus < 0 ||
        (topo->n_online_cpus > 0 && topo->n_online_cpus < topo->n_allowed_cpus))
        topo->n_online_cpus = 0;

    /* 3. NUMA topology (optional, but fail-closed within) */
    int numa_rc = discover_numa(topo, &topo->allowed_cpus);
    if (numa_rc != 0) {
        status |= numa_rc;
    }

    /* 4. SVE vector length (optional) */
    topo->sve_vl_bits = discover_sve_vl();
    if (topo->sve_vl_bits == 0) {
        status |= CAMBLAS_TOPO_NO_SVE;
    }

    /* 5. CPU model (optional) */
    if (discover_cpu_model(topo->cpu_model, sizeof(topo->cpu_model)) != 0) {
        status |= CAMBLAS_TOPO_NO_MODEL;
    }

    return status;
}

/* ============================================================
 * Print
 * ============================================================ */

void camblas_topology_print(const camblas_topology_t *topo, FILE *stream)
{
    if (!topo || !stream)
        return;

    char cpustr[8192];
    camblas_cpuset_format(&topo->allowed_cpus, cpustr, sizeof(cpustr));

    /* This is a diagnostic consumer, not a planner validation boundary.  It
       must remain bounded when a caller supplies a partially initialized or
       otherwise malformed snapshot. */
    int n_nodes = topo->n_numa_nodes;
    if (n_nodes < 0)
        n_nodes = 0;
    if (n_nodes > CAMBLAS_MAX_NUMA_NODES)
        n_nodes = CAMBLAS_MAX_NUMA_NODES;

    fprintf(stream, "=== CAMBLAS Topology ===\n");
    fprintf(stream, "CPU model:           %.*s\n", (int)sizeof(topo->cpu_model),
            topo->cpu_model[0] ? topo->cpu_model : "(unknown)");
    fprintf(stream, "Online CPUs:         %d\n", topo->n_online_cpus);
    fprintf(stream, "Allowed CPUs:        %d  [%s]\n", topo->n_allowed_cpus, cpustr);
    fprintf(stream, "SVE vector length:   %d bits\n", topo->sve_vl_bits);
    fprintf(stream, "NUMA nodes (allowed):%d\n", n_nodes);

    for (int i = 0; i < n_nodes; i++) {
        const camblas_numa_node_t *node = &topo->numa_nodes[i];
        char nodecpus[8192];
        camblas_cpuset_format(&node->cpus, nodecpus, sizeof(nodecpus));
        double mem_gib = (double)node->memory_bytes_physical / (1024.0 * 1024.0 * 1024.0);
        fprintf(stream, "  Node %d: %d allowed CPUs [%s] (physical: %d CPUs, %.1f GiB)\n",
                node->node_id, node->n_cpus, nodecpus, node->n_cpus_physical, mem_gib);
    }
}
