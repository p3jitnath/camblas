/* FP64 six-row A-micro kernel in a separate translation unit so that its
 * register blocking can be selected independently of the other SVE kernels. */
#include "packed.h"
#include "kernels.h"
#include "gemm_bounds.h"
#include <stdint.h>
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#include "sve_interleaved.h"
#endif
int camblas_dgemm_sve_amicro6_tile(int m, int n, int k, double alpha, const double *A, int lda,
                                   const double *B, int ldb, double *C, int ldc, int initialize)
{
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
    if (m < 0 || n < 0 || k < 0 || !A || !B || !C || lda < 1 || ldb < 1 || ldc < 1 ||
        (initialize != 0 && initialize != 1))
        return CAMBLAS_PACKED_UNAVAILABLE;
    /* Three two-lane FP64 vectors cover six rows. Refuse other runtime
     * vector lengths before the fixed-layout packed panels can be read. */
    camblas_kernel_runtime_t runtime;
    camblas_kernel_runtime_query(&runtime);
    if (!runtime.compiled_sve || !runtime.hw_sve || runtime.vl_bits != 128 ||
        CAMBLAS_MICRO8_ROWS != 3)
        return CAMBLAS_PACKED_UNAVAILABLE;
    if (m == 0 || n == 0)
        return 0;
    if (ldc < m || ldb < k || (k > 0 && lda != k))
        return CAMBLAS_PACKED_UNAVAILABLE;
    if (alpha == 0 || k == 0)
        return 0;
    /* A is packed in six-row groups and B in eight-column groups.
     * Validate the padded spans, not only the logical matrix dimensions. */
    size_t mp = ((size_t)m + 5) / 6 * 6, np = ((size_t)n + 7) / 8 * 8;
    if (mp > PTRDIFF_MAX / sizeof(*A) / (size_t)lda ||
        np > PTRDIFF_MAX / sizeof(*B) / (size_t)ldb ||
        camblas_matrix_span_fits(m, n, ldc, sizeof(*C)))
        return CAMBLAS_PACKED_UNAVAILABLE;
    if (initialize)
        return camblas_micro8_layout_f64(m, n, k, alpha, A, lda, B, ldb, C, ldc, 1, 1);
    return camblas_micro8_layout_f64(m, n, k, alpha, A, lda, B, ldb, C, ldc, 1, 0);
#else
    (void)m;
    (void)n;
    (void)k;
    (void)alpha;
    (void)A;
    (void)lda;
    (void)B;
    (void)ldb;
    (void)C;
    (void)ldc;
    (void)initialize;
    return CAMBLAS_PACKED_UNAVAILABLE;
#endif
}
