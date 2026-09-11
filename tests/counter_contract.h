#ifndef CAMBLAS_COUNTER_CONTRACT_H
#define CAMBLAS_COUNTER_CONTRACT_H

#include <stdint.h>

/*
 * Counter totals are normalized by the number of timed GEMM calls before
 * entering an observation record. Keep the positive-cycle/instruction
 * invariant explicit at that boundary: a positive raw total can otherwise
 * floor to zero when integer-divided by repetitions.
 */
static inline int camblas_counter_normalized_positive(uint64_t cycles, uint64_t instructions,
                                                      int repetitions)
{
    uint64_t divisor;
    if (repetitions < 1)
        return 0;
    divisor = (uint64_t)repetitions;
    return cycles / divisor > 0 && instructions / divisor > 0;
}

#endif /* CAMBLAS_COUNTER_CONTRACT_H */
