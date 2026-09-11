/*
 * CAMBLAS private GEMM address-arithmetic checks.
 *
 * Public dimensions and leading dimensions are LP64 C int values, while
 * matrix indexing must not be performed in signed int.  These helpers check
 * the largest element index that a path will form before it touches an
 * operand.  They are private implementation infrastructure and do not change
 * the public ABI or its no-read rules.
 */
#ifndef CAMBLAS_INTERNAL_GEMM_BOUNDS_H
#define CAMBLAS_INTERNAL_GEMM_BOUNDS_H

#include <stddef.h>
#include <stdint.h>

/*
 * Check the addressable span of a column-major rows x cols matrix.  The
 * element-size check keeps a valid element index from describing an object
 * whose byte extent cannot be represented by size_t.  Empty matrices have no
 * accessed element and therefore have no span to check.
 */
static inline int camblas_matrix_span_fits(int rows, int cols, int ld, size_t element_size)
{
    size_t last_row, last_col, last_index, element_count;

    if (rows < 0 || cols < 0 || ld < 1 || element_size == 0)
        return -1;
    if (rows == 0 || cols == 0)
        return 0;

    last_row = (size_t)rows - 1;
    last_col = (size_t)cols - 1;
    if (last_col > (SIZE_MAX - last_row) / (size_t)ld)
        return -1;
    last_index = last_row + last_col * (size_t)ld;
    if (last_index == SIZE_MAX)
        return -1;
    element_count = last_index + 1;
    return element_count > SIZE_MAX / element_size ? -1 : 0;
}

/*
 * Check only the spans that the requested operation will access.  In
 * particular, alpha==0 callers can skip A/B, and C is needed only when beta
 * changes it or a nonzero-alpha product has positive K. This preserves the
 * no-read/no-write behavior even for synthetic or protected operands.
 */
static inline int camblas_gemm_output_access_needed(int k, int alpha_nonzero, int beta_not_one)
{
    return beta_not_one || (alpha_nonzero && k > 0);
}

static inline int camblas_gemm_access_spans_fit(char trans_a, char trans_b, int m, int n, int k,
                                                int lda, int ldb, int ldc, int read_inputs,
                                                int write_output, size_t element_size)
{
    int a_rows, a_cols, b_rows, b_cols;

    if ((trans_a != 'N' && trans_a != 'T') || (trans_b != 'N' && trans_b != 'T') || m < 0 ||
        n < 0 || k < 0)
        return -1;
    if (m == 0 || n == 0)
        return 0;

    if (read_inputs) {
        a_rows = (trans_a == 'N') ? m : k;
        a_cols = (trans_a == 'N') ? k : m;
        b_rows = (trans_b == 'N') ? k : n;
        b_cols = (trans_b == 'N') ? n : k;
        if (camblas_matrix_span_fits(a_rows, a_cols, lda, element_size) != 0 ||
            camblas_matrix_span_fits(b_rows, b_cols, ldb, element_size) != 0)
            return -1;
    }
    if (write_output && camblas_matrix_span_fits(m, n, ldc, element_size) != 0)
        return -1;
    return 0;
}

#endif /* CAMBLAS_INTERNAL_GEMM_BOUNDS_H */
