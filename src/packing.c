/*
 * CAMBLAS checked packed-panel primitives.
 *
 * The shape-aware planner's correctness-gated packed executor consumes these
 * routines. The pack boundary remains independent of the ordinary-C blocked
 * compute loop and any future SVE/NEON microkernel, so shape, transpose,
 * padding, and failure behavior stay testable in isolation.
 */
#include "packing.h"
#ifndef CAMBLAS_PACK_TRANSPOSE_TILED
#define CAMBLAS_PACK_TRANSPOSE_TILED 0
#endif

/* Transposed source columns are contiguous along logical j. Small tiles keep
 * both source and destination cache lines live instead of repeatedly walking
 * a large source stride. Padding is still cleared by the checked caller. */
#define DEFINE_TRANSPOSE_COPY(SUFFIX, TYPE)                                                        \
    static void transpose_copy_##SUFFIX(const TYPE *src, TYPE *dst, size_t lda, size_t ld,         \
                                        size_t row0, size_t col0, size_t rows, size_t cols)        \
    {                                                                                              \
        for (size_t ii = 0; ii < rows; ii += 16)                                                   \
            for (size_t jj = 0; jj < cols; jj += 16) {                                             \
                size_t ie = rows - ii < 16 ? rows : ii + 16, je = cols - jj < 16 ? cols : jj + 16; \
                for (size_t i = ii; i < ie; ++i)                                                   \
                    for (size_t j = jj; j < je; ++j)                                               \
                        dst[i + j * ld] = src[col0 + j + (row0 + i) * lda];                        \
            }                                                                                      \
    }
#if CAMBLAS_PACK_TRANSPOSE_TILED
DEFINE_TRANSPOSE_COPY(f32, float)
DEFINE_TRANSPOSE_COPY(f64, double)
#endif
#undef DEFINE_TRANSPOSE_COPY

#include <string.h>
#include <stdint.h>

static int round_up_size(size_t value, size_t multiple, size_t *out)
{
    size_t remainder;
    size_t add;

    if (!out || multiple == 0)
        return -1;
    remainder = value % multiple;
    if (remainder == 0) {
        *out = value;
        return 0;
    }
    add = multiple - remainder;
    if (value > SIZE_MAX - add)
        return -1;
    *out = value + add;
    return 0;
}

/* Check only the source elements this particular panel will read. */
static int source_panel_span_fits(char trans, int lda, int row0, int col0, int panel_rows,
                                  int panel_cols, size_t element_size)
{
    size_t source_row_last, source_col_last, last_index;

    if ((trans != 'N' && trans != 'T') || lda < 1 || row0 < 0 || col0 < 0 || panel_rows < 0 ||
        panel_cols < 0 || element_size == 0)
        return -1;
    if (panel_rows == 0 || panel_cols == 0)
        return 0;

    if (trans == 'N') {
        source_row_last = (size_t)row0 + (size_t)panel_rows - 1;
        source_col_last = (size_t)col0 + (size_t)panel_cols - 1;
    } else {
        source_row_last = (size_t)col0 + (size_t)panel_cols - 1;
        source_col_last = (size_t)row0 + (size_t)panel_rows - 1;
    }
    if (source_col_last > (SIZE_MAX - source_row_last) / (size_t)lda)
        return -1;
    last_index = source_row_last + source_col_last * (size_t)lda;
    if (last_index == SIZE_MAX || last_index + 1 > SIZE_MAX / element_size)
        return -1;
    return 0;
}

int camblas_packed_panel_shape(int rows, int cols, int row0, int col0, int panel_rows,
                               int panel_cols, int row_multiple, int col_multiple,
                               camblas_packed_panel_t *shape)
{
    camblas_packed_panel_t result = {0};

    if (!shape || rows < 0 || cols < 0 || row0 < 0 || col0 < 0 || panel_rows < 0 ||
        panel_cols < 0 || row_multiple < 1 || col_multiple < 1 || row0 > rows || col0 > cols ||
        panel_rows > rows - row0 || panel_cols > cols - col0)
        return -1;

    if (round_up_size((size_t)panel_rows, (size_t)row_multiple, &result.padded_rows) != 0 ||
        round_up_size((size_t)panel_cols, (size_t)col_multiple, &result.padded_cols) != 0)
        return -1;

    if (result.padded_rows != 0 && result.padded_cols > SIZE_MAX / result.padded_rows)
        return -1;

    result.logical_rows = (size_t)panel_rows;
    result.logical_cols = (size_t)panel_cols;
    result.ld = result.padded_rows;
    result.elements = result.padded_rows * result.padded_cols;
    result.layout = CAMBLAS_PANEL_COLUMN_MAJOR;
    *shape = result;
    return 0;
}

static int validate_pack_args(char trans, int rows, int cols, int lda, int row0, int col0,
                              int panel_rows, int panel_cols, int row_multiple, int col_multiple,
                              size_t element_size, const void *src, void *dst, size_t dst_elems,
                              camblas_packed_panel_t *shape)
{
    int min_lda;

    if (trans != 'N' && trans != 'T')
        return -1;
    if (rows < 0 || cols < 0 || lda < 1 || element_size == 0)
        return -1;
    min_lda = (trans == 'N') ? rows : cols;
    if (min_lda < 1)
        min_lda = 1;
    if (lda < min_lda)
        return -1;
    if (camblas_packed_panel_shape(rows, cols, row0, col0, panel_rows, panel_cols, row_multiple,
                                   col_multiple, shape) != 0)
        return -1;
    if (source_panel_span_fits(trans, lda, row0, col0, panel_rows, panel_cols, element_size) != 0 ||
        shape->elements > dst_elems || shape->elements > SIZE_MAX / element_size)
        return -1;
    if (shape->elements != 0 && (!src || !dst))
        return -1;
    return 0;
}

static size_t source_index(char trans, int lda, size_t row, size_t col)
{
    if (trans == 'N')
        return row + col * (size_t)lda;
    return col + row * (size_t)lda;
}

int camblas_pack_f32_bpanel_interleaved(char trans, int rows, int cols, int lda, int row0, int col0,
                                        int panel_rows, int panel_cols, const float *src,
                                        float *dst, size_t dst_elems, camblas_packed_panel_t *shape)
{
    camblas_packed_panel_t result;
    if (!shape || validate_pack_args(trans, rows, cols, lda, row0, col0, panel_rows, panel_cols, 1,
                                     8, sizeof(*dst), src, dst, dst_elems, &result) != 0)
        return -1;
    result.layout = CAMBLAS_PANEL_B_MICRO8;
    if (!result.elements) {
        *shape = result;
        return 0;
    }
    size_t j0 = 0;
#if defined(CAMBLAS_PACK_B_FULL8) && CAMBLAS_PACK_B_FULL8
    /* Full groups need neither edge predicates nor per-element transpose tests. */
    for (; j0 + 8 <= result.logical_cols; j0 += 8) {
        float *out = dst + j0 * result.ld;
        if (trans == 'T') {
            for (size_t k = 0; k < result.logical_rows; ++k)
                memcpy(out + k * 8, src + (size_t)col0 + j0 + ((size_t)row0 + k) * (size_t)lda,
                       8 * sizeof(*out));
        } else {
            const float *s0 = src + (size_t)row0 + ((size_t)col0 + j0 + 0) * (size_t)lda;
            const float *s1 = src + (size_t)row0 + ((size_t)col0 + j0 + 1) * (size_t)lda;
            const float *s2 = src + (size_t)row0 + ((size_t)col0 + j0 + 2) * (size_t)lda;
            const float *s3 = src + (size_t)row0 + ((size_t)col0 + j0 + 3) * (size_t)lda;
            const float *s4 = src + (size_t)row0 + ((size_t)col0 + j0 + 4) * (size_t)lda;
            const float *s5 = src + (size_t)row0 + ((size_t)col0 + j0 + 5) * (size_t)lda;
            const float *s6 = src + (size_t)row0 + ((size_t)col0 + j0 + 6) * (size_t)lda;
            const float *s7 = src + (size_t)row0 + ((size_t)col0 + j0 + 7) * (size_t)lda;
            for (size_t k = 0; k < result.logical_rows; ++k) {
                out[k * 8 + 0] = s0[k];
                out[k * 8 + 1] = s1[k];
                out[k * 8 + 2] = s2[k];
                out[k * 8 + 3] = s3[k];
                out[k * 8 + 4] = s4[k];
                out[k * 8 + 5] = s5[k];
                out[k * 8 + 6] = s6[k];
                out[k * 8 + 7] = s7[k];
            }
        }
    }
#endif
    for (; j0 < result.padded_cols; j0 += 8) {
        for (size_t k = 0; k < result.logical_rows; k++) {
            for (size_t q = 0; q < 8; q++) {
                size_t j = j0 + q;
                dst[j0 * result.ld + k * 8 + q] =
                    j < result.logical_cols
                        ? src[source_index(trans, lda, (size_t)row0 + k, (size_t)col0 + j)]
                        : 0;
            }
        }
    }
    *shape = result;
    return 0;
}

int camblas_pack_f64_bpanel_interleaved(char trans, int rows, int cols, int lda, int row0, int col0,
                                        int panel_rows, int panel_cols, const double *src,
                                        double *dst, size_t dst_elems,
                                        camblas_packed_panel_t *shape)
{
    camblas_packed_panel_t result;
    if (!shape || validate_pack_args(trans, rows, cols, lda, row0, col0, panel_rows, panel_cols, 1,
                                     8, sizeof(*dst), src, dst, dst_elems, &result) != 0)
        return -1;
    result.layout = CAMBLAS_PANEL_B_MICRO8;
    if (!result.elements) {
        *shape = result;
        return 0;
    }
    size_t j0 = 0;
#if defined(CAMBLAS_PACK_B_FULL8) && CAMBLAS_PACK_B_FULL8
    /* Full groups need neither edge predicates nor per-element transpose tests. */
    for (; j0 + 8 <= result.logical_cols; j0 += 8) {
        double *out = dst + j0 * result.ld;
        if (trans == 'T') {
            for (size_t k = 0; k < result.logical_rows; ++k)
                memcpy(out + k * 8, src + (size_t)col0 + j0 + ((size_t)row0 + k) * (size_t)lda,
                       8 * sizeof(*out));
        } else {
            const double *s0 = src + (size_t)row0 + ((size_t)col0 + j0 + 0) * (size_t)lda;
            const double *s1 = src + (size_t)row0 + ((size_t)col0 + j0 + 1) * (size_t)lda;
            const double *s2 = src + (size_t)row0 + ((size_t)col0 + j0 + 2) * (size_t)lda;
            const double *s3 = src + (size_t)row0 + ((size_t)col0 + j0 + 3) * (size_t)lda;
            const double *s4 = src + (size_t)row0 + ((size_t)col0 + j0 + 4) * (size_t)lda;
            const double *s5 = src + (size_t)row0 + ((size_t)col0 + j0 + 5) * (size_t)lda;
            const double *s6 = src + (size_t)row0 + ((size_t)col0 + j0 + 6) * (size_t)lda;
            const double *s7 = src + (size_t)row0 + ((size_t)col0 + j0 + 7) * (size_t)lda;
            for (size_t k = 0; k < result.logical_rows; ++k) {
                out[k * 8 + 0] = s0[k];
                out[k * 8 + 1] = s1[k];
                out[k * 8 + 2] = s2[k];
                out[k * 8 + 3] = s3[k];
                out[k * 8 + 4] = s4[k];
                out[k * 8 + 5] = s5[k];
                out[k * 8 + 6] = s6[k];
                out[k * 8 + 7] = s7[k];
            }
        }
    }
#endif
    for (; j0 < result.padded_cols; j0 += 8) {
        for (size_t k = 0; k < result.logical_rows; k++) {
            for (size_t q = 0; q < 8; q++) {
                size_t j = j0 + q;
                dst[j0 * result.ld + k * 8 + q] =
                    j < result.logical_cols
                        ? src[source_index(trans, lda, (size_t)row0 + k, (size_t)col0 + j)]
                        : 0;
            }
        }
    }
    *shape = result;
    return 0;
}

int camblas_pack_f32_bpanel_micro6(char trans, int rows, int cols, int lda, int row0, int col0,
                                   int panel_rows, int panel_cols, const float *src, float *dst,
                                   size_t dst_elems, camblas_packed_panel_t *shape)
{
    camblas_packed_panel_t result;
    if (!shape || validate_pack_args(trans, rows, cols, lda, row0, col0, panel_rows, panel_cols, 1,
                                     6, sizeof(*dst), src, dst, dst_elems, &result) != 0)
        return -1;
    result.layout = CAMBLAS_PANEL_B_MICRO6;
    for (size_t j0 = 0; j0 < result.padded_cols; j0 += 6) {
        for (size_t k = 0; k < result.logical_rows; k++) {
            for (size_t q = 0; q < 6; q++) {
                size_t j = j0 + q;
                dst[j0 * result.ld + k * 6 + q] =
                    j < result.logical_cols
                        ? src[source_index(trans, lda, (size_t)row0 + k, (size_t)col0 + j)]
                        : 0;
            }
        }
    }
    *shape = result;
    return 0;
}

int camblas_pack_f64_bpanel_micro6(char trans, int rows, int cols, int lda, int row0, int col0,
                                   int panel_rows, int panel_cols, const double *src, double *dst,
                                   size_t dst_elems, camblas_packed_panel_t *shape)
{
    camblas_packed_panel_t result;
    if (!shape || validate_pack_args(trans, rows, cols, lda, row0, col0, panel_rows, panel_cols, 1,
                                     6, sizeof(*dst), src, dst, dst_elems, &result) != 0)
        return -1;
    result.layout = CAMBLAS_PANEL_B_MICRO6;
    for (size_t j0 = 0; j0 < result.padded_cols; j0 += 6) {
        for (size_t k = 0; k < result.logical_rows; k++) {
            for (size_t q = 0; q < 6; q++) {
                size_t j = j0 + q;
                dst[j0 * result.ld + k * 6 + q] =
                    j < result.logical_cols
                        ? src[source_index(trans, lda, (size_t)row0 + k, (size_t)col0 + j)]
                        : 0;
            }
        }
    }
    *shape = result;
    return 0;
}

int camblas_pack_f32_bpanel_micro6_padded(char trans, int rows, int cols, int lda, int row0,
                                          int col0, int panel_rows, int panel_cols,
                                          const float *src, float *dst, size_t dst_elems,
                                          camblas_packed_panel_t *shape)
{
    camblas_packed_panel_t result;
    if (!shape || validate_pack_args(trans, rows, cols, lda, row0, col0, panel_rows, panel_cols, 1,
                                     6, sizeof(*dst), src, dst, dst_elems, &result) != 0)
        return -1;
    size_t slices = result.elements / 6;
    if (slices > SIZE_MAX / 8 / sizeof(*dst) || slices > dst_elems / 8)
        return -1;
    result.elements = slices * 8;
    result.layout = CAMBLAS_PANEL_B_MICRO6_PAD8;
    for (size_t j0 = 0; j0 < result.padded_cols; j0 += 6)
        for (size_t l = 0; l < result.logical_rows; l++)
            for (size_t q = 0; q < 8; q++)
                dst[(j0 / 6 * 8) * result.ld + l * 8 + q] =
                    q < 6 && j0 + q < result.logical_cols
                        ? src[source_index(trans, lda, (size_t)row0 + l, (size_t)col0 + j0 + q)]
                        : 0;
    *shape = result;
    return 0;
}

int camblas_pack_f64_bpanel_micro6_padded(char trans, int rows, int cols, int lda, int row0,
                                          int col0, int panel_rows, int panel_cols,
                                          const double *src, double *dst, size_t dst_elems,
                                          camblas_packed_panel_t *shape)
{
    camblas_packed_panel_t result;
    if (!shape || validate_pack_args(trans, rows, cols, lda, row0, col0, panel_rows, panel_cols, 1,
                                     6, sizeof(*dst), src, dst, dst_elems, &result) != 0)
        return -1;
    size_t slices = result.elements / 6;
    if (slices > SIZE_MAX / 8 / sizeof(*dst) || slices > dst_elems / 8)
        return -1;
    result.elements = slices * 8;
    result.layout = CAMBLAS_PANEL_B_MICRO6_PAD8;
    for (size_t j0 = 0; j0 < result.padded_cols; j0 += 6)
        for (size_t l = 0; l < result.logical_rows; l++)
            for (size_t q = 0; q < 8; q++)
                dst[(j0 / 6 * 8) * result.ld + l * 8 + q] =
                    q < 6 && j0 + q < result.logical_cols
                        ? src[source_index(trans, lda, (size_t)row0 + l, (size_t)col0 + j0 + q)]
                        : 0;
    *shape = result;
    return 0;
}

int camblas_pack_f32_apanel_micro(char trans, int rows, int cols, int lda, int row0, int col0,
                                  int panel_rows, int panel_cols, int group, const float *src,
                                  float *dst, size_t dst_elems, camblas_packed_panel_t *shape)
{
    camblas_packed_panel_t result;
    if (!shape || validate_pack_args(trans, rows, cols, lda, row0, col0, panel_rows, panel_cols,
                                     group, 1, sizeof(*dst), src, dst, dst_elems, &result) != 0)
        return -1;
    result.layout = CAMBLAS_PANEL_A_MICRO;
    result.row_group = (size_t)group;
    result.ld = result.logical_cols;
    for (size_t i0 = 0; i0 < result.padded_rows; i0 += (size_t)group)
        for (size_t l = 0; l < result.logical_cols; l++)
            for (size_t q = 0; q < (size_t)group; q++)
                dst[i0 * result.ld + l * (size_t)group + q] =
                    i0 + q < result.logical_rows
                        ? src[source_index(trans, lda, (size_t)row0 + i0 + q, (size_t)col0 + l)]
                        : 0;
    *shape = result;
    return 0;
}

int camblas_pack_f64_apanel_micro(char trans, int rows, int cols, int lda, int row0, int col0,
                                  int panel_rows, int panel_cols, int group, const double *src,
                                  double *dst, size_t dst_elems, camblas_packed_panel_t *shape)
{
    camblas_packed_panel_t result;
    if (!shape || validate_pack_args(trans, rows, cols, lda, row0, col0, panel_rows, panel_cols,
                                     group, 1, sizeof(*dst), src, dst, dst_elems, &result) != 0)
        return -1;
    result.layout = CAMBLAS_PANEL_A_MICRO;
    result.row_group = (size_t)group;
    result.ld = result.logical_cols;
    size_t i0 = 0;
    /* The Grace FP64 microkernel consumes six doubles. Copy complete
       groups without a per-element edge predicate; keep the generic path
       for transpose views, other group sizes, and the final partial group. */
    if (trans == 'N' && group == 6) {
        for (; i0 + 6 <= result.logical_rows; i0 += 6)
            for (size_t l = 0; l < result.logical_cols; l++)
                memcpy(dst + i0 * result.ld + l * 6,
                       src + (size_t)row0 + i0 + ((size_t)col0 + l) * (size_t)lda,
                       6 * sizeof(*dst));
    }
    for (; i0 < result.padded_rows; i0 += (size_t)group)
        for (size_t l = 0; l < result.logical_cols; l++)
            for (size_t q = 0; q < (size_t)group; q++)
                dst[i0 * result.ld + l * (size_t)group + q] =
                    i0 + q < result.logical_rows
                        ? src[source_index(trans, lda, (size_t)row0 + i0 + q, (size_t)col0 + l)]
                        : 0;
    *shape = result;
    return 0;
}

int camblas_pack_f32_panel(char trans, int rows, int cols, int lda, int row0, int col0,
                           int panel_rows, int panel_cols, int row_multiple, int col_multiple,
                           const float *src, float *dst, size_t dst_elems,
                           camblas_packed_panel_t *shape)
{
    camblas_packed_panel_t result;

    if (!shape ||
        validate_pack_args(trans, rows, cols, lda, row0, col0, panel_rows, panel_cols, row_multiple,
                           col_multiple, sizeof(float), src, dst, dst_elems, &result) != 0)
        return -1;

    /* An exact, non-transposed panel has no padding to clear.  Copying one
       source column at a time preserves the packed layout while avoiding a
       full-panel zero pass and the per-element source-index arithmetic. */
    if (trans == 'N' && result.elements != 0 && result.padded_rows == result.logical_rows &&
        result.padded_cols == result.logical_cols) {
        for (size_t j = 0; j < result.logical_cols; j++)
            memcpy(dst + j * result.ld, src + (size_t)row0 + ((size_t)col0 + j) * (size_t)lda,
                   result.logical_rows * sizeof(*dst));
        *shape = result;
        return 0;
    }

    for (size_t q = 0; q < result.elements; q++)
        dst[q] = 0.0f;
#if CAMBLAS_PACK_TRANSPOSE_TILED
    if (trans != 'N') {
        transpose_copy_f32(src, dst, (size_t)lda, result.ld, (size_t)row0, (size_t)col0,
                           result.logical_rows, result.logical_cols);
        *shape = result;
        return 0;
    }
#endif
    for (size_t j = 0; j < result.logical_cols; j++) {
        for (size_t i = 0; i < result.logical_rows; i++) {
            size_t src_row = (size_t)row0 + i;
            size_t src_col = (size_t)col0 + j;
            dst[i + j * result.ld] = src[source_index(trans, lda, src_row, src_col)];
        }
    }
    *shape = result;
    return 0;
}

int camblas_pack_f64_panel(char trans, int rows, int cols, int lda, int row0, int col0,
                           int panel_rows, int panel_cols, int row_multiple, int col_multiple,
                           const double *src, double *dst, size_t dst_elems,
                           camblas_packed_panel_t *shape)
{
    camblas_packed_panel_t result;

    if (!shape ||
        validate_pack_args(trans, rows, cols, lda, row0, col0, panel_rows, panel_cols, row_multiple,
                           col_multiple, sizeof(double), src, dst, dst_elems, &result) != 0)
        return -1;

    /* Match the fp32 fast path while leaving padding and transpose handling
       on the established checked element-copy path. */
    if (trans == 'N' && result.elements != 0 && result.padded_rows == result.logical_rows &&
        result.padded_cols == result.logical_cols) {
        for (size_t j = 0; j < result.logical_cols; j++)
            memcpy(dst + j * result.ld, src + (size_t)row0 + ((size_t)col0 + j) * (size_t)lda,
                   result.logical_rows * sizeof(*dst));
        *shape = result;
        return 0;
    }

    for (size_t q = 0; q < result.elements; q++)
        dst[q] = 0.0;
#if CAMBLAS_PACK_TRANSPOSE_TILED
    if (trans != 'N') {
        transpose_copy_f64(src, dst, (size_t)lda, result.ld, (size_t)row0, (size_t)col0,
                           result.logical_rows, result.logical_cols);
        *shape = result;
        return 0;
    }
#endif
    for (size_t j = 0; j < result.logical_cols; j++) {
        for (size_t i = 0; i < result.logical_rows; i++) {
            size_t src_row = (size_t)row0 + i;
            size_t src_col = (size_t)col0 + j;
            dst[i + j * result.ld] = src[source_index(trans, lda, src_row, src_col)];
        }
    }
    *shape = result;
    return 0;
}
