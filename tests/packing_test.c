/* Bounded correctness and failure-contract tests for the internal panel packer. */
#include "../src/packing.h"

#include <limits.h>
#include <stdio.h>
#include <string.h>

static int failures;

#define CHECK(condition, message)                     \
    do {                                              \
        if (!(condition)) {                           \
            fprintf(stderr, "FAIL: %s\n", (message)); \
            failures++;                               \
        }                                             \
    } while (0)

static void fill_f32(float *x, size_t count)
{
    for (size_t q = 0; q < count; q++)
        x[q] = (float)(q * 13u + 7u) * 0.125f;
}

static void fill_f64(double *x, size_t count)
{
    for (size_t q = 0; q < count; q++)
        x[q] = (double)(q * 13u + 7u) * 0.125;
}

static void test_f32_no_transpose(void)
{
    const int rows = 5, cols = 4, lda = 8;
    float src[8 * 4], before[8 * 4], dst[4 * 3];
    camblas_packed_panel_t shape;

    fill_f32(src, sizeof(src) / sizeof(src[0]));
    memcpy(before, src, sizeof(src));
    for (size_t q = 0; q < sizeof(dst) / sizeof(dst[0]); q++)
        dst[q] = -99.0f;
    CHECK(camblas_pack_f32_panel('N', rows, cols, lda, 1, 1, 3, 2, 4, 3, src, dst, 12, &shape) == 0,
          "fp32 N panel pack succeeds");
    CHECK(shape.logical_rows == 3 && shape.logical_cols == 2 && shape.padded_rows == 4 &&
              shape.padded_cols == 3 && shape.ld == 4 && shape.elements == 12,
          "fp32 N packed shape is explicit");
    for (size_t j = 0; j < shape.padded_cols; j++) {
        for (size_t i = 0; i < shape.padded_rows; i++) {
            float expected = (i < 3 && j < 2) ? src[(1 + i) + (1 + j) * (size_t)lda] : 0.0f;
            CHECK(dst[i + j * shape.ld] == expected, "fp32 N values and zero padding match");
        }
    }
    CHECK(memcmp(src, before, sizeof(src)) == 0, "fp32 N source is unchanged");
}

static void test_f64_transpose(void)
{
    const int rows = 4, cols = 5, lda = 7;
    double src[7 * 4], before[7 * 4], dst[3 * 4];
    camblas_packed_panel_t shape;

    fill_f64(src, sizeof(src) / sizeof(src[0]));
    memcpy(before, src, sizeof(src));
    for (size_t q = 0; q < sizeof(dst) / sizeof(dst[0]); q++)
        dst[q] = -99.0;
    CHECK(camblas_pack_f64_panel('T', rows, cols, lda, 1, 2, 2, 2, 3, 4, src, dst, 12, &shape) == 0,
          "fp64 T panel pack succeeds");
    CHECK(shape.logical_rows == 2 && shape.logical_cols == 2 && shape.padded_rows == 3 &&
              shape.padded_cols == 4 && shape.ld == 3 && shape.elements == 12,
          "fp64 T packed shape is explicit");
    for (size_t j = 0; j < shape.padded_cols; j++) {
        for (size_t i = 0; i < shape.padded_rows; i++) {
            double expected = (i < 2 && j < 2) ? src[(2 + j) + (1 + i) * (size_t)lda] : 0.0;
            CHECK(dst[i + j * shape.ld] == expected, "fp64 T values and zero padding match");
        }
    }
    CHECK(memcmp(src, before, sizeof(src)) == 0, "fp64 T source is unchanged");
}

static void test_fail_closed(void)
{
    float src[4 * 3], dst[16], before_dst[16];
    camblas_packed_panel_t shape, before_shape;

    fill_f32(src, sizeof(src) / sizeof(src[0]));
    for (size_t q = 0; q < sizeof(dst) / sizeof(dst[0]); q++)
        dst[q] = 23.0f;
    memcpy(before_dst, dst, sizeof(dst));
    memset(&before_shape, 0x5a, sizeof(before_shape));
    shape = before_shape;
    CHECK(camblas_packed_panel_shape(3, 4, 2, 3, 2, 1, 4, 1, &shape) == -1,
          "out-of-range panel shape is rejected");
    CHECK(memcmp(&shape, &before_shape, sizeof(shape)) == 0,
          "failed shape query leaves output untouched");

    shape = before_shape;
    CHECK(camblas_pack_f32_panel('N', 3, 4, 4, 0, 0, 3, 2, 4, 2, src, dst, 7, &shape) == -1,
          "insufficient destination capacity is rejected");
    CHECK(memcmp(dst, before_dst, sizeof(dst)) == 0 &&
              memcmp(&shape, &before_shape, sizeof(shape)) == 0,
          "capacity failure writes neither destination nor shape");

    CHECK(camblas_pack_f32_panel('X', 3, 4, 4, 0, 0, 1, 1, 1, 1, src, dst, 16, &shape) == -1,
          "invalid transpose is rejected");
    CHECK(memcmp(dst, before_dst, sizeof(dst)) == 0,
          "argument failure leaves destination untouched");

    CHECK(camblas_pack_f32_panel('N', 3, 4, 2, 0, 0, 1, 1, 1, 1, src, dst, 16, &shape) == -1,
          "leading dimension below logical rows is rejected");

    CHECK(camblas_pack_f32_panel('N', 3, 4, 3, 3, 0, 0, 2, 4, 2, NULL, NULL, 0, &shape) == 0 &&
              shape.logical_rows == 0 && shape.logical_cols == 2 && shape.elements == 0,
          "empty panel is a checked no-op");
}

static void test_extreme_empty_panel(void)
{
    camblas_packed_panel_t shape;
    float f32_src[1] = {3.0f}, f32_dst[1] = {0.0f};
    double f64_src[1] = {3.0}, f64_dst[1] = {0.0};

    CHECK(camblas_pack_f32_panel('N', INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, 0, 0, 1, 1, NULL,
                                 NULL, 0, &shape) == 0 &&
              shape.logical_rows == 0 && shape.logical_cols == 0 && shape.elements == 0,
          "fp32 extreme empty panel has no source span");
    CHECK(camblas_pack_f64_panel('T', INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, 0, 0, 1, 1, NULL,
                                 NULL, 0, &shape) == 0 &&
              shape.logical_rows == 0 && shape.logical_cols == 0 && shape.elements == 0,
          "fp64 extreme empty panel has no source span");

    /* A one-element low panel must not require the unaccessed full matrix
       extent to fit in size_t.  The real source/destination accesses are
       bounded to these one-element objects. */
    CHECK(camblas_pack_f32_panel('N', INT_MAX, 1, INT_MAX, 0, 0, 1, 1, 1, 1, f32_src, f32_dst, 1,
                                 &shape) == 0 &&
              f32_dst[0] == 3.0f,
          "fp32 low panel span is checked locally");
    CHECK(camblas_pack_f64_panel('T', 1, INT_MAX, INT_MAX, 0, 0, 1, 1, 1, 1, f64_src, f64_dst, 1,
                                 &shape) == 0 &&
              f64_dst[0] == 3.0,
          "fp64 low transposed panel span is checked locally");
}

static void test_null_shape_is_fail_closed(void)
{
    float f32_src[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float f32_dst[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float f32_src_before[4], f32_dst_before[4];
    double f64_src[4] = {1.0, 2.0, 3.0, 4.0};
    double f64_dst[4] = {5.0, 6.0, 7.0, 8.0};
    double f64_src_before[4], f64_dst_before[4];

    memcpy(f32_src_before, f32_src, sizeof(f32_src));
    memcpy(f32_dst_before, f32_dst, sizeof(f32_dst));
    CHECK(camblas_pack_f32_panel('N', 2, 2, 2, 0, 0, 1, 1, 1, 1, f32_src, f32_dst, 4, NULL) == -1,
          "fp32 NULL shape output is rejected");
    CHECK(memcmp(f32_src, f32_src_before, sizeof(f32_src)) == 0 &&
              memcmp(f32_dst, f32_dst_before, sizeof(f32_dst)) == 0,
          "fp32 NULL shape failure does not access source or destination");

    memcpy(f64_src_before, f64_src, sizeof(f64_src));
    memcpy(f64_dst_before, f64_dst, sizeof(f64_dst));
    CHECK(camblas_pack_f64_panel('N', 2, 2, 2, 0, 0, 1, 1, 1, 1, f64_src, f64_dst, 4, NULL) == -1,
          "fp64 NULL shape output is rejected");
    CHECK(memcmp(f64_src, f64_src_before, sizeof(f64_src)) == 0 &&
              memcmp(f64_dst, f64_dst_before, sizeof(f64_dst)) == 0,
          "fp64 NULL shape failure does not access source or destination");
}

int main(void)
{
    test_f32_no_transpose();
    test_f64_transpose();
    test_fail_closed();
    test_extreme_empty_panel();
    test_null_shape_is_fail_closed();
    if (failures != 0) {
        fprintf(stderr, "packed-panel tests: %d failure(s)\n", failures);
        return 1;
    }
    printf("packed-panel tests: PASS (fp32/fp64, N/T, padding, failure gates)\n");
    return 0;
}
