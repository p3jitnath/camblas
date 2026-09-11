#ifndef CAMBLAS_CPUSET_BITS_H
#define CAMBLAS_CPUSET_BITS_H
#include "camblas_topology.h"

/* Inspect every storage word: caller-supplied nbits is not trusted to bound
 * the scan. clz is only applied to nonzero words. */
static inline int camblas_cpuset_validated_count(const camblas_cpuset_t *set)
{
    if (!set || set->nbits < 0 || set->nbits > CAMBLAS_MAX_CPUS)
        return -1;
    int count = 0, width = 0;
    for (size_t word = 0; word < sizeof(set->bits) / sizeof(set->bits[0]); ++word) {
        uint64_t value = set->bits[word];
        count += __builtin_popcountll(value);
        if (value)
            width = (int)(word * 64) + 64 - __builtin_clzll(value);
    }
    return width == set->nbits ? count : -1;
}
#endif
