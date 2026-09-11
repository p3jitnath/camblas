/* Bit-exact reference for the transposed twelve-row A packer. */
#include "amicro_transpose.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t element_bits(const float *value)
{
    uint32_t result;
    memcpy(&result, value, sizeof(result));
    return result;
}

static void fill_bits(float *values, size_t count, uint32_t bits)
{
    for (size_t index = 0; index < count; ++index)
        memcpy(values + index, &bits, sizeof(bits));
}

int main(void)
{
    const uint32_t sentinel = UINT32_C(0xc1234567);
    const uint32_t patterns[] = {0,
                                 UINT32_C(0x80000000),
                                 UINT32_C(0x7f800000),
                                 UINT32_C(0xff800000),
                                 UINT32_C(0x7fc12345),
                                 UINT32_C(0x3f123456),
                                 UINT32_C(0xbf765432)};
    size_t cases = 0;
    for (size_t rows = 0; rows <= 37; ++rows)
        for (size_t depth = 0; depth <= 65; ++depth) {
            size_t lda = depth + 13, row0 = 3, depth0 = 5;
            size_t source_count = (rows + row0 + 3) * lda;
            size_t padded_rows = (rows + 11) / 12 * 12;
            size_t destination_count = padded_rows * depth + 32;
            float *source = malloc(source_count * sizeof(*source));
            float *before = malloc(source_count * sizeof(*before));
            float *destination = malloc(destination_count * sizeof(*destination));
            assert(source && before && destination);
            fill_bits(source, source_count, sentinel);
            fill_bits(destination, destination_count, sentinel);
            for (size_t row = 0; row < rows; ++row)
                for (size_t k = 0; k < depth; ++k) {
                    uint32_t bits = patterns[(row * 3 + k) % 7];
                    memcpy(source + (row0 + row) * lda + depth0 + k, &bits, sizeof(bits));
                }
            memcpy(before, source, source_count * sizeof(*source));
            float *output = destination + 7;
            camblas_pack_transposed_amicro12(source, lda, row0, depth0, rows, depth, output);
            for (size_t row = 0; row < padded_rows; ++row)
                for (size_t k = 0; k < depth; ++k) {
                    size_t index = (row / 12 * 12) * depth + k * 12 + row % 12;
                    uint32_t expected =
                        row < rows ? element_bits(source + (row0 + row) * lda + depth0 + k) : 0;
                    assert(element_bits(output + index) == expected);
                }
            for (size_t index = 0; index < 7; ++index)
                assert(element_bits(destination + index) == sentinel);
            for (size_t index = 7 + padded_rows * depth; index < destination_count; ++index)
                assert(element_bits(destination + index) == sentinel);
            assert(memcmp(source, before, source_count * sizeof(*source)) == 0);
            free(destination);
            free(before);
            free(source);
            ++cases;
        }
    printf("{\"status\":\"passed\",\"cases\":%zu}\n", cases);
    return 0;
}
