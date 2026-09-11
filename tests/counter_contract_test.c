#include "counter_contract.h"

#include <stdint.h>
#include <stdio.h>

static int expect(const char *name, int actual, int wanted)
{
    if (actual == wanted)
        return 0;
    fprintf(stderr, "FAIL %s: got %d, expected %d\n", name, actual, wanted);
    return 1;
}

int main(void)
{
    int failures = 0;

    failures +=
        expect("ordinary positive totals",
               camblas_counter_normalized_positive(UINT64_C(458388), UINT64_C(2829234), 3), 1);
    failures += expect("exact one-per-call totals",
                       camblas_counter_normalized_positive(UINT64_C(3), UINT64_C(3), 3), 1);
    failures += expect("cycles floor to zero",
                       camblas_counter_normalized_positive(UINT64_C(2), UINT64_C(3), 3), 0);
    failures += expect("instructions floor to zero",
                       camblas_counter_normalized_positive(UINT64_C(3), UINT64_C(2), 3), 0);
    failures += expect("zero repetitions",
                       camblas_counter_normalized_positive(UINT64_C(3), UINT64_C(3), 0), 0);
    failures += expect("large totals",
                       camblas_counter_normalized_positive(UINT64_MAX, UINT64_MAX, 1000), 1);

    if (failures != 0)
        return 1;
    puts("counter normalization contract PASS: positive per-call cycles/instructions");
    return 0;
}
