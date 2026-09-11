/*
 * CAMBLAS internal packed-panel contract.
 *
 * This header is deliberately outside include/ and is not part of the public
 * ABI. Ordinary panels are column-major in caller-owned memory. Their row
 * and column counts are rounded up to row_multiple and col_multiple,
 * respectively; every padded element is written as zero. The logical panel
 * coordinates are in op(X) space, so trans='T' reads the transposed view of
 * the source matrix without changing the packed layout.
 *
 * The dedicated B packers also support explicitly tagged private layouts
 * with six- or eight-column interleaving. Packed execution must inspect the tag;
 * column-major kernels must never consume interleaved panels.
 */
#ifndef CAMBLAS_INTERNAL_PACKING_H
#define CAMBLAS_INTERNAL_PACKING_H

#include <stddef.h>

#if defined(__GNUC__) || defined(__clang__)
#define CAMBLAS_PACK_INTERNAL __attribute__((visibility("hidden")))
#else
#define CAMBLAS_PACK_INTERNAL
#endif

typedef enum {
    CAMBLAS_PANEL_COLUMN_MAJOR = 0,
    CAMBLAS_PANEL_B_MICRO8 = 1,
    CAMBLAS_PANEL_B_MICRO6 = 2,
    CAMBLAS_PANEL_B_MICRO6_PAD8 = 3,
    CAMBLAS_PANEL_A_MICRO = 4
} camblas_panel_layout_t;

/* Logical sizes describe the requested panel; padded sizes include zero-filled
 * tails. ld is layout-dependent: a column stride for ordinary panels, or a K
 * stride for interleaved groups. row_group identifies the A-micro row grouping. */
typedef struct {
    size_t logical_rows;
    size_t logical_cols;
    size_t padded_rows;
    size_t padded_cols;
    size_t ld;
    size_t elements;
    camblas_panel_layout_t layout;
    size_t row_group;
} camblas_packed_panel_t;

/* Private B layout: (j/8*8)*ld + k*8 + j%8. K is not padded;
 * N is padded to eight, with zero-filled missing columns. Failure leaves
 * destination and descriptor unchanged, as for the ordinary packers. */
CAMBLAS_PACK_INTERNAL int
camblas_pack_f32_bpanel_interleaved(char trans, int rows, int cols, int lda, int row0, int col0,
                                    int panel_rows, int panel_cols, const float *src, float *dst,
                                    size_t dst_elems, camblas_packed_panel_t *shape);
CAMBLAS_PACK_INTERNAL int
camblas_pack_f64_bpanel_interleaved(char trans, int rows, int cols, int lda, int row0, int col0,
                                    int panel_rows, int panel_cols, const double *src, double *dst,
                                    size_t dst_elems, camblas_packed_panel_t *shape);

/* Six-column variant: (j/6*6)*ld + k*6 + j%6, N padded to six. */
CAMBLAS_PACK_INTERNAL int camblas_pack_f32_bpanel_micro6(char trans, int rows, int cols, int lda,
                                                         int row0, int col0, int panel_rows,
                                                         int panel_cols, const float *src,
                                                         float *dst, size_t dst_elems,
                                                         camblas_packed_panel_t *shape);

/* Six-column variant: (j/6*6)*ld + k*6 + j%6, N padded to six. */
CAMBLAS_PACK_INTERNAL int camblas_pack_f64_bpanel_micro6(char trans, int rows, int cols, int lda,
                                                         int row0, int col0, int panel_rows,
                                                         int panel_cols, const double *src,
                                                         double *dst, size_t dst_elems,
                                                         camblas_packed_panel_t *shape);

/* Six logical columns per group, eight allocated slots per K slice. */
CAMBLAS_PACK_INTERNAL int
camblas_pack_f32_bpanel_micro6_padded(char trans, int rows, int cols, int lda, int row0, int col0,
                                      int panel_rows, int panel_cols, const float *src, float *dst,
                                      size_t dst_elems, camblas_packed_panel_t *shape);

/* Six logical columns per group, eight allocated slots per K slice. */
CAMBLAS_PACK_INTERNAL int
camblas_pack_f64_bpanel_micro6_padded(char trans, int rows, int cols, int lda, int row0, int col0,
                                      int panel_rows, int panel_cols, const double *src,
                                      double *dst, size_t dst_elems, camblas_packed_panel_t *shape);

/* A row groups: (i/group*group)*ld + k*group + i%group; ld is K. */
CAMBLAS_PACK_INTERNAL int camblas_pack_f32_apanel_micro(char trans, int rows, int cols, int lda,
                                                        int row0, int col0, int panel_rows,
                                                        int panel_cols, int group, const float *src,
                                                        float *dst, size_t dst_elems,
                                                        camblas_packed_panel_t *shape);

/* A row groups: (i/group*group)*ld + k*group + i%group; ld is K. */
CAMBLAS_PACK_INTERNAL int
camblas_pack_f64_apanel_micro(char trans, int rows, int cols, int lda, int row0, int col0,
                              int panel_rows, int panel_cols, int group, const double *src,
                              double *dst, size_t dst_elems, camblas_packed_panel_t *shape);

/*
 * Compute the packed shape for a logical rows x cols matrix panel starting at
 * (row0, col0). panel_rows/panel_cols must fit within the logical matrix.
 * Returns 0 on success and -1 on invalid dimensions, coordinates, tile
 * multiples, or size overflow. The output is unchanged on failure.
 */
CAMBLAS_PACK_INTERNAL int camblas_packed_panel_shape(int rows, int cols, int row0, int col0,
                                                     int panel_rows, int panel_cols,
                                                     int row_multiple, int col_multiple,
                                                     camblas_packed_panel_t *shape);

/*
 * Pack a column-major panel from the logical op(X) view of src. For trans='N',
 * src has physical dimensions rows x cols; for trans='T', its dimensions
 * are cols x rows. In both cases lda is the physical column stride, measured
 * in scalar elements. dst_elems is the number of scalar elements available in dst.
 *
 * Sources and destination must not overlap. shape must be non-NULL. No source
 * or destination access is made on failure, and shape is written only after a
 * successful pack.
 */
CAMBLAS_PACK_INTERNAL int camblas_pack_f32_panel(char trans, int rows, int cols, int lda, int row0,
                                                 int col0, int panel_rows, int panel_cols,
                                                 int row_multiple, int col_multiple,
                                                 const float *src, float *dst, size_t dst_elems,
                                                 camblas_packed_panel_t *shape);

CAMBLAS_PACK_INTERNAL int camblas_pack_f64_panel(char trans, int rows, int cols, int lda, int row0,
                                                 int col0, int panel_rows, int panel_cols,
                                                 int row_multiple, int col_multiple,
                                                 const double *src, double *dst, size_t dst_elems,
                                                 camblas_packed_panel_t *shape);

#endif /* CAMBLAS_INTERNAL_PACKING_H */
