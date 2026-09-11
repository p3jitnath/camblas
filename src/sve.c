/*
 * CAMBLAS first AArch64 SVE kernel path.
 *
 * The translation unit is compiled separately with
 * -march=armv8.2-a+sve -msve-vector-bits=scalable when the selected compiler
 * supports it.  The implementation is vector-length agnostic: svcntw()/
 * svcntd() and whilelt predicates handle the current process VL and the final
 * partial row tile.
 *
 * Scope of this first path:
 *   - single-core, no-pack GEMM with a build-selected one-, two-, or
 *     four-column output microtile (the accepted default is nr=2; nr=4 is
 *     a bounded Phase-4 candidate);
 *   - trans_a='N';
 *   - trans_b='N' or 'T';
 *   - fp32 and fp64.
 *
 * CAMBLAS_SVE_UNAVAILABLE means that the caller must use the scalar reference.
 * Any other nonzero status is an execution failure and must not be retried.
 * All capability and no-access checks happen before the first output access,
 * so the unavailable path is safe and a zero-alpha or zero-K beta-one no-op
 * does not touch C.
 */
#define _GNU_SOURCE 1

#include "kernels.h"
#include "packed.h"
#include "gemm_bounds.h"
#include "sve_runtime.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <sys/auxv.h>

#if defined(__aarch64__)
#include <asm/hwcap.h>
#include <sys/prctl.h>
#endif

#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#define CAMBLAS_SVE_ACLE 1
#else
#define CAMBLAS_SVE_ACLE 0
#endif

void camblas_kernel_runtime_query(camblas_kernel_runtime_t *info)
{
    if (!info)
        return;
    memset(info, 0, sizeof(*info));
    info->vl_bits = -1;
    info->compiled_sve = CAMBLAS_SVE_ACLE;

#if defined(__aarch64__)
    {
        unsigned long hwcap = getauxval(AT_HWCAP);
        unsigned long hwcap2 = getauxval(AT_HWCAP2);
#ifdef HWCAP_SVE
        info->hw_sve = (hwcap & HWCAP_SVE) != 0;
#endif
#ifdef HWCAP2_SVE2
        info->hw_sve2 = (hwcap2 & HWCAP2_SVE2) != 0;
#endif
    }

    {
        int vl_bits = camblas_sve_vl_bits_from_prctl(prctl(PR_SVE_GET_VL));
        if (vl_bits > 0)
            info->vl_bits = vl_bits;
    }
#endif
}

const char *camblas_kernel_runtime_name(const camblas_kernel_runtime_t *info)
{
    if (!info || !info->compiled_sve || !info->hw_sve || info->vl_bits <= 0)
        return "scalar-fallback";
    return info->hw_sve2 ? "sve2" : "sve";
}

#if CAMBLAS_SVE_ACLE

static int sve_runtime_ready(void)
{
    camblas_kernel_runtime_t info;
    camblas_kernel_runtime_query(&info);
#if defined(__ARM_FEATURE_SVE_BITS) && __ARM_FEATURE_SVE_BITS > 0
    /* A vector-length-specialized research build must refuse another runtime
     * VL rather than executing compiler-constant lane arithmetic incorrectly. */
    if (info.vl_bits != __ARM_FEATURE_SVE_BITS)
        return 0;
#endif
    return info.compiled_sve && info.hw_sve && info.vl_bits > 0;
}

/* Keep Neoverse-V2's scalar-replicate load form while allowing one full
   predicate to be shared by the four-column fast body. */
static inline __attribute__((always_inline)) svfloat32_t
camblas_sve_load_replicate_f32(svbool_t pg, const float *address)
{
    svfloat32_t value;
    __asm__ volatile("ld1rw {%0.s}, %1/z, [%2]" : "=w"(value) : "Upl"(pg), "r"(address) : "memory");
    return value;
}

static inline __attribute__((always_inline)) svfloat64_t
camblas_sve_load_replicate_f64(svbool_t pg, const double *address)
{
    svfloat64_t value;
    __asm__ volatile("ld1rd {%0.d}, %1/z, [%2]" : "=w"(value) : "Upl"(pg), "r"(address) : "memory");
    return value;
}

#ifndef CAMBLAS_SVE_BALANCED_COLS
#define CAMBLAS_SVE_BALANCED_COLS 0
#endif
#ifndef CAMBLAS_SVE_BLOCK6
#define CAMBLAS_SVE_BLOCK6 0
#endif
#ifndef CAMBLAS_SVE_TIGHT_N
#define CAMBLAS_SVE_TIGHT_N 0
#endif
#ifndef CAMBLAS_SVE_COLS6
#define CAMBLAS_SVE_COLS6 0
#endif
#if CAMBLAS_SVE_COLS6 && !CAMBLAS_SVE_BLOCK6
#error "CAMBLAS_SVE_COLS6 requires CAMBLAS_SVE_BLOCK6"
#endif
#if CAMBLAS_SVE_BALANCED_COLS && !CAMBLAS_SVE_BLOCK6
#error "CAMBLAS_SVE_BALANCED_COLS requires CAMBLAS_SVE_BLOCK6"
#endif
#if CAMBLAS_SVE_BLOCK6
#include "sve_register_block.h"
#include "sve_block4x6.h"
#include "sve_block_balanced.h"
#endif

/*
 * Keep the hidden entry point fail-closed even when it is called outside the
 * public scalar wrapper.  The public API performs the same checks, but this
 * boundary is also used by the opt-in SVE library and must never turn a
 * malformed leading dimension into an out-of-bounds vector load/store. The
 * public API uses int dimensions, so the shared span check keeps the hidden
 * backend's index arithmetic representable in size_t; the caller still owns
 * the actual object-size/lifetime contract.
 */
static int sve_valid_gemm_args(char trans_a, char trans_b, int m, int n, int k, int lda, int ldb,
                               int ldc, const void *A, const void *B, const void *C,
                               int read_inputs, int write_output, size_t element_size)
{
    if ((trans_a != 'N' && trans_a != 'T') || (trans_b != 'N' && trans_b != 'T') || m < 0 ||
        n < 0 || k < 0 || !A || !B || !C)
        return 0;

    /* Match the no-op contract: positive LDs are required, but a call with
       no output elements does not need the larger shape-dependent minima. */
    if (m == 0 || n == 0)
        return lda >= 1 && ldb >= 1 && ldc >= 1;

    int min_lda = (trans_a == 'N') ? m : k;
    int min_ldb = (trans_b == 'N') ? k : n;
    if (min_lda < 1)
        min_lda = 1;
    if (min_ldb < 1)
        min_ldb = 1;
    if (lda < min_lda || ldb < min_ldb || ldc < m)
        return 0;

    /* Keep the direct hidden entry point aligned with the public access-aware
       span gate.  In particular, alpha=0 and k=0 do not require an A/B span,
       while beta=1 with no product does not require a C span because it is a
       true no-read/no-write operation. */
    return camblas_gemm_access_spans_fit(trans_a, trans_b, m, n, k, lda, ldb, ldc, read_inputs,
                                         write_output, element_size) == 0;
}

#if CAMBLAS_SVE_NR == 4

/*
 * The wider candidate is intentionally a separate body.  Keeping the four
 * vector accumulators explicit avoids arrays of sizeless SVE types and makes
 * the selected register tile visible in the generated code.  The scalar
 * accumulation order for each output column is unchanged.
 */
static int sgemm_sve_body_nr4(char trans_b, int m, int n, int k, float alpha, const float *A,
                              int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
    size_t lanes = (size_t)svcntw();
    size_t m_size = (size_t)m;
    size_t n_size = (size_t)n;
    size_t k_size = (size_t)k;
    size_t lda_size = (size_t)lda;
    size_t ldb_size = (size_t)ldb;
    size_t ldc_size = (size_t)ldc;

    if (lanes == 0)
        return CAMBLAS_SVE_UNAVAILABLE;
    for (size_t j = 0; j < n_size; j += 4) {
        size_t j1 = j + 1;
        size_t j2 = j + 2;
        size_t j3 = j + 3;
        int have_j1 = j1 < n_size;
        int have_j2 = j2 < n_size;
        int have_j3 = j3 < n_size;
        for (size_t i = 0; i < m_size; i += lanes) {
            svbool_t pg = svwhilelt_b32((uint64_t)i, (uint64_t)m_size);
            float *c0 = C + i + j * ldc_size;
            float *c1 = have_j1 ? C + i + j1 * ldc_size : NULL;
            float *c2 = have_j2 ? C + i + j2 * ldc_size : NULL;
            float *c3 = have_j3 ? C + i + j3 * ldc_size : NULL;
            svfloat32_t cv0 = svdup_f32(0.0f);
            svfloat32_t cv1 = svdup_f32(0.0f);
            svfloat32_t cv2 = svdup_f32(0.0f);
            svfloat32_t cv3 = svdup_f32(0.0f);

            if (beta == 0.0f) {
                cv0 = svdup_f32(0.0f);
                if (have_j1)
                    cv1 = svdup_f32(0.0f);
                if (have_j2)
                    cv2 = svdup_f32(0.0f);
                if (have_j3)
                    cv3 = svdup_f32(0.0f);
            } else {
                cv0 = svld1_f32(pg, c0);
                if (beta != 1.0f)
                    cv0 = svmul_n_f32_x(pg, cv0, beta);
                if (have_j1) {
                    cv1 = svld1_f32(pg, c1);
                    if (beta != 1.0f)
                        cv1 = svmul_n_f32_x(pg, cv1, beta);
                }
                if (have_j2) {
                    cv2 = svld1_f32(pg, c2);
                    if (beta != 1.0f)
                        cv2 = svmul_n_f32_x(pg, cv2, beta);
                }
                if (have_j3) {
                    cv3 = svld1_f32(pg, c3);
                    if (beta != 1.0f)
                        cv3 = svmul_n_f32_x(pg, cv3, beta);
                }
            }

            if (alpha != 0.0f) {
                svfloat32_t acc0 = svdup_f32(0.0f);
                svfloat32_t acc1 = svdup_f32(0.0f);
                svfloat32_t acc2 = svdup_f32(0.0f);
                svfloat32_t acc3 = svdup_f32(0.0f);
                for (size_t l = 0; l < k_size; l++) {
                    const float *a = A + i + l * lda_size;
                    svfloat32_t av = svld1_f32(pg, a);
                    float b0 = (trans_b == 'N') ? B[l + j * ldb_size] : B[j + l * ldb_size];
                    acc0 = svmla_n_f32_x(pg, acc0, av, b0);
                    if (have_j1) {
                        float b1 = (trans_b == 'N') ? B[l + j1 * ldb_size] : B[j1 + l * ldb_size];
                        acc1 = svmla_n_f32_x(pg, acc1, av, b1);
                    }
                    if (have_j2) {
                        float b2 = (trans_b == 'N') ? B[l + j2 * ldb_size] : B[j2 + l * ldb_size];
                        acc2 = svmla_n_f32_x(pg, acc2, av, b2);
                    }
                    if (have_j3) {
                        float b3 = (trans_b == 'N') ? B[l + j3 * ldb_size] : B[j3 + l * ldb_size];
                        acc3 = svmla_n_f32_x(pg, acc3, av, b3);
                    }
                }
                if (alpha != 1.0f) {
                    acc0 = svmul_n_f32_x(pg, acc0, alpha);
                    if (have_j1)
                        acc1 = svmul_n_f32_x(pg, acc1, alpha);
                    if (have_j2)
                        acc2 = svmul_n_f32_x(pg, acc2, alpha);
                    if (have_j3)
                        acc3 = svmul_n_f32_x(pg, acc3, alpha);
                }
                cv0 = svadd_f32_x(pg, cv0, acc0);
                if (have_j1)
                    cv1 = svadd_f32_x(pg, cv1, acc1);
                if (have_j2)
                    cv2 = svadd_f32_x(pg, cv2, acc2);
                if (have_j3)
                    cv3 = svadd_f32_x(pg, cv3, acc3);
            }

            svst1_f32(pg, c0, cv0);
            if (have_j1)
                svst1_f32(pg, c1, cv1);
            if (have_j2)
                svst1_f32(pg, c2, cv2);
            if (have_j3)
                svst1_f32(pg, c3, cv3);
        }
    }
    return 0;
}

static int dgemm_sve_body_nr4(char trans_b, int m, int n, int k, double alpha, const double *A,
                              int lda, const double *B, int ldb, double beta, double *C, int ldc)
{
    size_t lanes = (size_t)svcntd();
    size_t m_size = (size_t)m;
    size_t n_size = (size_t)n;
    size_t k_size = (size_t)k;
    size_t lda_size = (size_t)lda;
    size_t ldb_size = (size_t)ldb;
    size_t ldc_size = (size_t)ldc;

    if (lanes == 0)
        return CAMBLAS_SVE_UNAVAILABLE;
    for (size_t j = 0; j < n_size; j += 4) {
        size_t j1 = j + 1;
        size_t j2 = j + 2;
        size_t j3 = j + 3;
        int have_j1 = j1 < n_size;
        int have_j2 = j2 < n_size;
        int have_j3 = j3 < n_size;
        for (size_t i = 0; i < m_size; i += lanes) {
            svbool_t pg = svwhilelt_b64((uint64_t)i, (uint64_t)m_size);
            double *c0 = C + i + j * ldc_size;
            double *c1 = have_j1 ? C + i + j1 * ldc_size : NULL;
            double *c2 = have_j2 ? C + i + j2 * ldc_size : NULL;
            double *c3 = have_j3 ? C + i + j3 * ldc_size : NULL;
            svfloat64_t cv0 = svdup_f64(0.0);
            svfloat64_t cv1 = svdup_f64(0.0);
            svfloat64_t cv2 = svdup_f64(0.0);
            svfloat64_t cv3 = svdup_f64(0.0);

            if (beta == 0.0) {
                cv0 = svdup_f64(0.0);
                if (have_j1)
                    cv1 = svdup_f64(0.0);
                if (have_j2)
                    cv2 = svdup_f64(0.0);
                if (have_j3)
                    cv3 = svdup_f64(0.0);
            } else {
                cv0 = svld1_f64(pg, c0);
                if (beta != 1.0)
                    cv0 = svmul_n_f64_x(pg, cv0, beta);
                if (have_j1) {
                    cv1 = svld1_f64(pg, c1);
                    if (beta != 1.0)
                        cv1 = svmul_n_f64_x(pg, cv1, beta);
                }
                if (have_j2) {
                    cv2 = svld1_f64(pg, c2);
                    if (beta != 1.0)
                        cv2 = svmul_n_f64_x(pg, cv2, beta);
                }
                if (have_j3) {
                    cv3 = svld1_f64(pg, c3);
                    if (beta != 1.0)
                        cv3 = svmul_n_f64_x(pg, cv3, beta);
                }
            }

            if (alpha != 0.0) {
                svfloat64_t acc0 = svdup_f64(0.0);
                svfloat64_t acc1 = svdup_f64(0.0);
                svfloat64_t acc2 = svdup_f64(0.0);
                svfloat64_t acc3 = svdup_f64(0.0);
                for (size_t l = 0; l < k_size; l++) {
                    const double *a = A + i + l * lda_size;
                    svfloat64_t av = svld1_f64(pg, a);
                    double b0 = (trans_b == 'N') ? B[l + j * ldb_size] : B[j + l * ldb_size];
                    acc0 = svmla_n_f64_x(pg, acc0, av, b0);
                    if (have_j1) {
                        double b1 = (trans_b == 'N') ? B[l + j1 * ldb_size] : B[j1 + l * ldb_size];
                        acc1 = svmla_n_f64_x(pg, acc1, av, b1);
                    }
                    if (have_j2) {
                        double b2 = (trans_b == 'N') ? B[l + j2 * ldb_size] : B[j2 + l * ldb_size];
                        acc2 = svmla_n_f64_x(pg, acc2, av, b2);
                    }
                    if (have_j3) {
                        double b3 = (trans_b == 'N') ? B[l + j3 * ldb_size] : B[j3 + l * ldb_size];
                        acc3 = svmla_n_f64_x(pg, acc3, av, b3);
                    }
                }
                if (alpha != 1.0) {
                    acc0 = svmul_n_f64_x(pg, acc0, alpha);
                    if (have_j1)
                        acc1 = svmul_n_f64_x(pg, acc1, alpha);
                    if (have_j2)
                        acc2 = svmul_n_f64_x(pg, acc2, alpha);
                    if (have_j3)
                        acc3 = svmul_n_f64_x(pg, acc3, alpha);
                }
                cv0 = svadd_f64_x(pg, cv0, acc0);
                if (have_j1)
                    cv1 = svadd_f64_x(pg, cv1, acc1);
                if (have_j2)
                    cv2 = svadd_f64_x(pg, cv2, acc2);
                if (have_j3)
                    cv3 = svadd_f64_x(pg, cv3, acc3);
            }

            svst1_f64(pg, c0, cv0);
            if (have_j1)
                svst1_f64(pg, c1, cv1);
            if (have_j2)
                svst1_f64(pg, c2, cv2);
            if (have_j3)
                svst1_f64(pg, c3, cv3);
        }
    }
    return 0;
}

#else

static int sgemm_sve_body(char trans_b, int m, int n, int k, float alpha, const float *A, int lda,
                          const float *B, int ldb, float beta, float *C, int ldc)
{
    size_t lanes = (size_t)svcntw();
    size_t m_size = (size_t)m;
    size_t n_size = (size_t)n;
    size_t k_size = (size_t)k;
    size_t lda_size = (size_t)lda;
    size_t ldb_size = (size_t)ldb;
    size_t ldc_size = (size_t)ldc;

    if (lanes == 0)
        return CAMBLAS_SVE_UNAVAILABLE;
    /* Reuse the A vector across the configured output columns. The final
       odd column is handled by the same loop as a one-column tail. */
    for (size_t j = 0; j < n_size; j += (size_t)CAMBLAS_SVE_NR) {
        size_t j1 = j + 1;
        int have_j1 = CAMBLAS_SVE_NR == 2 && j1 < n_size;
        for (size_t i = 0; i < m_size; i += lanes) {
            svbool_t pg = svwhilelt_b32((uint64_t)i, (uint64_t)m_size);
            float *c0 = C + i + j * ldc_size;
            float *c1 = have_j1 ? C + i + j1 * ldc_size : NULL;
            svfloat32_t cv0 = svdup_f32(0.0f);
            svfloat32_t cv1 = svdup_f32(0.0f);

            /* Match the scalar contract: beta=0 does not read C. */
            if (beta == 0.0f) {
                cv0 = svdup_f32(0.0f);
                if (have_j1)
                    cv1 = svdup_f32(0.0f);
            } else {
                cv0 = svld1_f32(pg, c0);
                if (beta != 1.0f)
                    cv0 = svmul_n_f32_x(pg, cv0, beta);
                if (have_j1) {
                    cv1 = svld1_f32(pg, c1);
                    if (beta != 1.0f)
                        cv1 = svmul_n_f32_x(pg, cv1, beta);
                }
            }

            /* Match the scalar contract: alpha=0 does not read A or B. */
            if (alpha != 0.0f) {
                svfloat32_t acc0 = svdup_f32(0.0f);
                svfloat32_t acc1 = svdup_f32(0.0f);
                for (size_t l = 0; l < k_size; l++) {
                    const float *a = A + i + l * lda_size;
                    svfloat32_t av = svld1_f32(pg, a);
                    float b0 = (trans_b == 'N') ? B[l + j * ldb_size] : B[j + l * ldb_size];
                    acc0 = svmla_n_f32_x(pg, acc0, av, b0);
                    if (have_j1) {
                        float b1 = (trans_b == 'N') ? B[l + j1 * ldb_size] : B[j1 + l * ldb_size];
                        acc1 = svmla_n_f32_x(pg, acc1, av, b1);
                    }
                }
                if (alpha != 1.0f)
                    acc0 = svmul_n_f32_x(pg, acc0, alpha);
                cv0 = svadd_f32_x(pg, cv0, acc0);
                if (have_j1) {
                    if (alpha != 1.0f)
                        acc1 = svmul_n_f32_x(pg, acc1, alpha);
                    cv1 = svadd_f32_x(pg, cv1, acc1);
                }
            }

            svst1_f32(pg, c0, cv0);
            if (have_j1)
                svst1_f32(pg, c1, cv1);
        }
    }
    return 0;
}

static int dgemm_sve_body(char trans_b, int m, int n, int k, double alpha, const double *A, int lda,
                          const double *B, int ldb, double beta, double *C, int ldc)
{
    size_t lanes = (size_t)svcntd();
    size_t m_size = (size_t)m;
    size_t n_size = (size_t)n;
    size_t k_size = (size_t)k;
    size_t lda_size = (size_t)lda;
    size_t ldb_size = (size_t)ldb;
    size_t ldc_size = (size_t)ldc;

    if (lanes == 0)
        return CAMBLAS_SVE_UNAVAILABLE;
    /* Reuse the A vector across the configured output columns. The final
       odd column is handled by the same loop as a one-column tail. */
    for (size_t j = 0; j < n_size; j += (size_t)CAMBLAS_SVE_NR) {
        size_t j1 = j + 1;
        int have_j1 = CAMBLAS_SVE_NR == 2 && j1 < n_size;
        for (size_t i = 0; i < m_size; i += lanes) {
            svbool_t pg = svwhilelt_b64((uint64_t)i, (uint64_t)m_size);
            double *c0 = C + i + j * ldc_size;
            double *c1 = have_j1 ? C + i + j1 * ldc_size : NULL;
            svfloat64_t cv0 = svdup_f64(0.0);
            svfloat64_t cv1 = svdup_f64(0.0);

            if (beta == 0.0) {
                cv0 = svdup_f64(0.0);
                if (have_j1)
                    cv1 = svdup_f64(0.0);
            } else {
                cv0 = svld1_f64(pg, c0);
                if (beta != 1.0)
                    cv0 = svmul_n_f64_x(pg, cv0, beta);
                if (have_j1) {
                    cv1 = svld1_f64(pg, c1);
                    if (beta != 1.0)
                        cv1 = svmul_n_f64_x(pg, cv1, beta);
                }
            }

            if (alpha != 0.0) {
                svfloat64_t acc0 = svdup_f64(0.0);
                svfloat64_t acc1 = svdup_f64(0.0);
                for (size_t l = 0; l < k_size; l++) {
                    const double *a = A + i + l * lda_size;
                    svfloat64_t av = svld1_f64(pg, a);
                    double b0 = (trans_b == 'N') ? B[l + j * ldb_size] : B[j + l * ldb_size];
                    acc0 = svmla_n_f64_x(pg, acc0, av, b0);
                    if (have_j1) {
                        double b1 = (trans_b == 'N') ? B[l + j1 * ldb_size] : B[j1 + l * ldb_size];
                        acc1 = svmla_n_f64_x(pg, acc1, av, b1);
                    }
                }
                if (alpha != 1.0)
                    acc0 = svmul_n_f64_x(pg, acc0, alpha);
                cv0 = svadd_f64_x(pg, cv0, acc0);
                if (have_j1) {
                    if (alpha != 1.0)
                        acc1 = svmul_n_f64_x(pg, acc1, alpha);
                    cv1 = svadd_f64_x(pg, cv1, acc1);
                }
            }

            svst1_f64(pg, c0, cv0);
            if (have_j1)
                svst1_f64(pg, c1, cv1);
        }
    }
    return 0;
}

#endif /* CAMBLAS_SVE_NR == 4 */

/*
 * Packed-panel tile hooks.  The caller has already scaled C by beta and
 * packs both operands into ordinary column-major panels, so these functions
 * only accumulate alpha*A*B.  Vectorizing the packed A rows keeps the
 * implementation valid for the planner's conservative packed nr values and
 * for any runtime VL; the svwhilelt predicate handles edge tiles.
 */
static int sve_valid_packed_tile(int m, int n, int k, int lda, int ldb, int ldc, const void *A,
                                 const void *B, const void *C, int read_inputs, int write_output,
                                 size_t element_size)
{
    if (m < 0 || n < 0 || k < 0 || !A || !B || !C)
        return 0;
    if (m == 0 || n == 0)
        return lda >= 1 && ldb >= 1 && ldc >= 1;
    if (lda < m || ldb < 1 || ldb < k || ldc < m)
        return 0;
    /* The hook only accumulates alpha*A*B.  A zero-alpha or zero-K tile is a
       complete no-op: retain the panel pointer/positive-LD contract, but do
       not require an unaccessed A/B/C span to be representable. */
    return camblas_gemm_access_spans_fit('N', 'N', m, n, k, lda, ldb, ldc, read_inputs,
                                         write_output, element_size) == 0;
}

#if CAMBLAS_SVE_NR == 4

/* The nr=4 packed candidate reuses one packed-A vector across four columns. */
static inline __attribute__((always_inline)) int sgemm_sve_packed_body(int m, int n, int k,
                                                                       float alpha, const float *A,
                                                                       int lda, const float *B,
                                                                       int ldb, float *C, int ldc)
{
    size_t lanes = (size_t)svcntw();
    size_t m_size = (size_t)m;
    size_t n_size = (size_t)n;
    size_t k_size = (size_t)k;
    size_t lda_size = (size_t)lda;
    size_t ldb_size = (size_t)ldb;
    size_t ldc_size = (size_t)ldc;

    if (lanes == 0)
        return CAMBLAS_PACKED_UNAVAILABLE;
    size_t fast_rows =
        (CAMBLAS_SVE_BLOCK6 && !CAMBLAS_SVE_COLS6 && !CAMBLAS_SVE_BALANCED_COLS ? 6U : 4U) * lanes;
    size_t fast_m = m_size - (m_size % fast_rows);
    size_t fast_cols =
        CAMBLAS_SVE_BALANCED_COLS ? CAMBLAS_SVE_BALANCED_COLS : (CAMBLAS_SVE_COLS6 ? 6U : 4U);
    size_t column_group =
        CAMBLAS_SVE_TIGHT_N ? fast_cols : (fast_cols == 5 ? 40U : (fast_cols == 6 ? 24U : 8U));
    size_t fast_n = n_size - n_size % column_group;
    const svbool_t bpg = svptrue_b32();
#if defined(CAMBLAS_SVE_ROW_FIRST) && CAMBLAS_SVE_ROW_FIRST
    for (size_t i = 0; i < fast_m; i += fast_rows) {
        for (size_t j = 0; j < fast_n; j += fast_cols) {
#else
    for (size_t j = 0; j < fast_n; j += fast_cols) {
        for (size_t i = 0; i < fast_m; i += fast_rows) {
#endif
#if CAMBLAS_SVE_BLOCK6
#if CAMBLAS_SVE_BALANCED_COLS
            camblas_balanced_f32(k, alpha, A + i, lda_size, B + j * ldb_size, ldb_size,
                                 C + i + j * ldc_size, ldc_size);
#elif CAMBLAS_SVE_COLS6
            camblas_block4x6_f32(k, alpha, A + i, lda_size, B + j * ldb_size, ldb_size,
                                 C + i + j * ldc_size, ldc_size);
#else
            camblas_block6_f32(k, alpha, A + i, lda_size, B + j * ldb_size, ldb_size,
                               C + i + j * ldc_size, ldc_size);
#endif
            continue;
#endif
            svbool_t pg0 = svwhilelt_b32((uint64_t)i, (uint64_t)m_size);
            svbool_t pg1 = svwhilelt_b32((uint64_t)(i + lanes), (uint64_t)m_size);
            svbool_t pg2 = svwhilelt_b32((uint64_t)(i + 2 * lanes), (uint64_t)m_size);
            svbool_t pg3 = svwhilelt_b32((uint64_t)(i + 3 * lanes), (uint64_t)m_size);
            svfloat32_t acc00 = svdup_f32(0.0f);
            svfloat32_t acc01 = svdup_f32(0.0f);
            svfloat32_t acc02 = svdup_f32(0.0f);
            svfloat32_t acc03 = svdup_f32(0.0f);
            svfloat32_t acc10 = svdup_f32(0.0f);
            svfloat32_t acc11 = svdup_f32(0.0f);
            svfloat32_t acc12 = svdup_f32(0.0f);
            svfloat32_t acc13 = svdup_f32(0.0f);
            svfloat32_t acc20 = svdup_f32(0.0f);
            svfloat32_t acc21 = svdup_f32(0.0f);
            svfloat32_t acc22 = svdup_f32(0.0f);
            svfloat32_t acc23 = svdup_f32(0.0f);
            svfloat32_t acc30 = svdup_f32(0.0f);
            svfloat32_t acc31 = svdup_f32(0.0f);
            svfloat32_t acc32 = svdup_f32(0.0f);
            svfloat32_t acc33 = svdup_f32(0.0f);
            const float *bp0 = B + j * ldb_size;
            const float *bp1 = bp0 + ldb_size;
            const float *bp2 = bp1 + ldb_size;
            const float *bp3 = bp2 + ldb_size;
            for (size_t l = 0; l < k_size; l++) {
                svfloat32_t b0 = camblas_sve_load_replicate_f32(bpg, bp0);
                svfloat32_t b1 = camblas_sve_load_replicate_f32(bpg, bp1);
                svfloat32_t b2 = camblas_sve_load_replicate_f32(bpg, bp2);
                svfloat32_t b3 = camblas_sve_load_replicate_f32(bpg, bp3);
                bp0++;
                bp1++;
                bp2++;
                bp3++;
                svfloat32_t av0 = svld1_f32(pg0, A + i + l * lda_size);
                acc00 = svmla_f32_x(pg0, acc00, av0, b0);
                acc01 = svmla_f32_x(pg0, acc01, av0, b1);
                acc02 = svmla_f32_x(pg0, acc02, av0, b2);
                acc03 = svmla_f32_x(pg0, acc03, av0, b3);
                svfloat32_t av1 = svld1_f32(pg1, A + i + lanes + l * lda_size);
                acc10 = svmla_f32_x(pg1, acc10, av1, b0);
                acc11 = svmla_f32_x(pg1, acc11, av1, b1);
                acc12 = svmla_f32_x(pg1, acc12, av1, b2);
                acc13 = svmla_f32_x(pg1, acc13, av1, b3);
                svfloat32_t av2 = svld1_f32(pg2, A + i + 2 * lanes + l * lda_size);
                acc20 = svmla_f32_x(pg2, acc20, av2, b0);
                acc21 = svmla_f32_x(pg2, acc21, av2, b1);
                acc22 = svmla_f32_x(pg2, acc22, av2, b2);
                acc23 = svmla_f32_x(pg2, acc23, av2, b3);
                svfloat32_t av3 = svld1_f32(pg3, A + i + 3 * lanes + l * lda_size);
                acc30 = svmla_f32_x(pg3, acc30, av3, b0);
                acc31 = svmla_f32_x(pg3, acc31, av3, b1);
                acc32 = svmla_f32_x(pg3, acc32, av3, b2);
                acc33 = svmla_f32_x(pg3, acc33, av3, b3);
            }
            if (alpha != 1.0f) {
                acc00 = svmul_n_f32_x(pg0, acc00, alpha);
                acc01 = svmul_n_f32_x(pg0, acc01, alpha);
                acc02 = svmul_n_f32_x(pg0, acc02, alpha);
                acc03 = svmul_n_f32_x(pg0, acc03, alpha);
                acc10 = svmul_n_f32_x(pg1, acc10, alpha);
                acc11 = svmul_n_f32_x(pg1, acc11, alpha);
                acc12 = svmul_n_f32_x(pg1, acc12, alpha);
                acc13 = svmul_n_f32_x(pg1, acc13, alpha);
                acc20 = svmul_n_f32_x(pg2, acc20, alpha);
                acc21 = svmul_n_f32_x(pg2, acc21, alpha);
                acc22 = svmul_n_f32_x(pg2, acc22, alpha);
                acc23 = svmul_n_f32_x(pg2, acc23, alpha);
                acc30 = svmul_n_f32_x(pg3, acc30, alpha);
                acc31 = svmul_n_f32_x(pg3, acc31, alpha);
                acc32 = svmul_n_f32_x(pg3, acc32, alpha);
                acc33 = svmul_n_f32_x(pg3, acc33, alpha);
            }
            float *c0 = C + i + j * ldc_size;
            svfloat32_t cv0 = svld1_f32(pg0, c0);
            cv0 = svadd_f32_x(pg0, cv0, acc00);
            svst1_f32(pg0, c0, cv0);
            float *c1 = C + i + (j + 1) * ldc_size;
            svfloat32_t cv1 = svld1_f32(pg0, c1);
            cv1 = svadd_f32_x(pg0, cv1, acc01);
            svst1_f32(pg0, c1, cv1);
            float *c2 = C + i + (j + 2) * ldc_size;
            svfloat32_t cv2 = svld1_f32(pg0, c2);
            cv2 = svadd_f32_x(pg0, cv2, acc02);
            svst1_f32(pg0, c2, cv2);
            float *c3 = C + i + (j + 3) * ldc_size;
            svfloat32_t cv3 = svld1_f32(pg0, c3);
            cv3 = svadd_f32_x(pg0, cv3, acc03);
            svst1_f32(pg0, c3, cv3);
            float *c10 = C + i + lanes + j * ldc_size;
            svfloat32_t cv10 = svld1_f32(pg1, c10);
            cv10 = svadd_f32_x(pg1, cv10, acc10);
            svst1_f32(pg1, c10, cv10);
            float *c11 = C + i + lanes + (j + 1) * ldc_size;
            svfloat32_t cv11 = svld1_f32(pg1, c11);
            cv11 = svadd_f32_x(pg1, cv11, acc11);
            svst1_f32(pg1, c11, cv11);
            float *c12 = C + i + lanes + (j + 2) * ldc_size;
            svfloat32_t cv12 = svld1_f32(pg1, c12);
            cv12 = svadd_f32_x(pg1, cv12, acc12);
            svst1_f32(pg1, c12, cv12);
            float *c13 = C + i + lanes + (j + 3) * ldc_size;
            svfloat32_t cv13 = svld1_f32(pg1, c13);
            cv13 = svadd_f32_x(pg1, cv13, acc13);
            svst1_f32(pg1, c13, cv13);
            float *c20 = C + i + 2 * lanes + j * ldc_size;
            svfloat32_t cv20 = svld1_f32(pg2, c20);
            cv20 = svadd_f32_x(pg2, cv20, acc20);
            svst1_f32(pg2, c20, cv20);
            float *c21 = C + i + 2 * lanes + (j + 1) * ldc_size;
            svfloat32_t cv21 = svld1_f32(pg2, c21);
            cv21 = svadd_f32_x(pg2, cv21, acc21);
            svst1_f32(pg2, c21, cv21);
            float *c22 = C + i + 2 * lanes + (j + 2) * ldc_size;
            svfloat32_t cv22 = svld1_f32(pg2, c22);
            cv22 = svadd_f32_x(pg2, cv22, acc22);
            svst1_f32(pg2, c22, cv22);
            float *c23 = C + i + 2 * lanes + (j + 3) * ldc_size;
            svfloat32_t cv23 = svld1_f32(pg2, c23);
            cv23 = svadd_f32_x(pg2, cv23, acc23);
            svst1_f32(pg2, c23, cv23);
            float *c30 = C + i + 3 * lanes + j * ldc_size;
            svfloat32_t cv30 = svld1_f32(pg3, c30);
            cv30 = svadd_f32_x(pg3, cv30, acc30);
            svst1_f32(pg3, c30, cv30);
            float *c31 = C + i + 3 * lanes + (j + 1) * ldc_size;
            svfloat32_t cv31 = svld1_f32(pg3, c31);
            cv31 = svadd_f32_x(pg3, cv31, acc31);
            svst1_f32(pg3, c31, cv31);
            float *c32 = C + i + 3 * lanes + (j + 2) * ldc_size;
            svfloat32_t cv32 = svld1_f32(pg3, c32);
            cv32 = svadd_f32_x(pg3, cv32, acc32);
            svst1_f32(pg3, c32, cv32);
            float *c33 = C + i + 3 * lanes + (j + 3) * ldc_size;
            svfloat32_t cv33 = svld1_f32(pg3, c33);
            cv33 = svadd_f32_x(pg3, cv33, acc33);
            svst1_f32(pg3, c33, cv33);
        }
    }
    /* Split edge traversal at fast_n: bottom rows of full columns first,
       then all rows of the remaining columns, without double updates. */
#if CAMBLAS_SVE_TIGHT_N
    for (size_t j = 0; j < n_size;) {
        size_t edge_end = j < fast_n ? fast_n : n_size;
#else
    for (size_t j = 0; j < n_size; j += 8) {
        size_t edge_end = n_size;
#endif
        size_t j1 = j + 1;
        size_t j2 = j + 2;
        size_t j3 = j + 3;
        size_t j4 = j + 4;
        size_t j5 = j + 5;
        size_t j6 = j + 6;
        size_t j7 = j + 7;
        int have_j1 = j1 < edge_end;
        int have_j2 = j2 < edge_end;
        int have_j3 = j3 < edge_end;
        int have_j4 = j4 < edge_end;
        int have_j5 = j5 < edge_end;
        int have_j6 = j6 < edge_end;
        int have_j7 = j7 < edge_end;
        for (size_t i = j < fast_n ? fast_m : 0; i < m_size; i += 2 * lanes) {
            svbool_t pg0 = svwhilelt_b32((uint64_t)i, (uint64_t)m_size);
            int have_i1 = i + lanes < m_size;
            svbool_t pg1 = svwhilelt_b32((uint64_t)(i + lanes), (uint64_t)m_size);
            svfloat32_t acc00 = svdup_f32(0.0f);
            svfloat32_t acc01 = svdup_f32(0.0f);
            svfloat32_t acc02 = svdup_f32(0.0f);
            svfloat32_t acc03 = svdup_f32(0.0f);
            svfloat32_t acc04 = svdup_f32(0.0f);
            svfloat32_t acc05 = svdup_f32(0.0f);
            svfloat32_t acc06 = svdup_f32(0.0f);
            svfloat32_t acc07 = svdup_f32(0.0f);
            svfloat32_t acc10 = svdup_f32(0.0f);
            svfloat32_t acc11 = svdup_f32(0.0f);
            svfloat32_t acc12 = svdup_f32(0.0f);
            svfloat32_t acc13 = svdup_f32(0.0f);
            svfloat32_t acc14 = svdup_f32(0.0f);
            svfloat32_t acc15 = svdup_f32(0.0f);
            svfloat32_t acc16 = svdup_f32(0.0f);
            svfloat32_t acc17 = svdup_f32(0.0f);
            for (size_t l = 0; l < k_size; l++) {
                const float *bp = B + l + j * ldb_size;
                float b0 = bp[0];
                float b1 = have_j1 ? bp[ldb_size] : 0.0f;
                float b2 = have_j2 ? bp[2 * ldb_size] : 0.0f;
                float b3 = have_j3 ? bp[3 * ldb_size] : 0.0f;
                float b4 = have_j4 ? bp[4 * ldb_size] : 0.0f;
                float b5 = have_j5 ? bp[5 * ldb_size] : 0.0f;
                float b6 = have_j6 ? bp[6 * ldb_size] : 0.0f;
                float b7 = have_j7 ? bp[7 * ldb_size] : 0.0f;
                svfloat32_t av0 = svld1_f32(pg0, A + i + l * lda_size);
                acc00 = svmla_n_f32_x(pg0, acc00, av0, b0);
                if (have_j1)
                    acc01 = svmla_n_f32_x(pg0, acc01, av0, b1);
                if (have_j2)
                    acc02 = svmla_n_f32_x(pg0, acc02, av0, b2);
                if (have_j3)
                    acc03 = svmla_n_f32_x(pg0, acc03, av0, b3);
                if (have_j4)
                    acc04 = svmla_n_f32_x(pg0, acc04, av0, b4);
                if (have_j5)
                    acc05 = svmla_n_f32_x(pg0, acc05, av0, b5);
                if (have_j6)
                    acc06 = svmla_n_f32_x(pg0, acc06, av0, b6);
                if (have_j7)
                    acc07 = svmla_n_f32_x(pg0, acc07, av0, b7);
                if (have_i1) {
                    svfloat32_t av1 = svld1_f32(pg1, A + i + lanes + l * lda_size);
                    acc10 = svmla_n_f32_x(pg1, acc10, av1, b0);
                    if (have_j1)
                        acc11 = svmla_n_f32_x(pg1, acc11, av1, b1);
                    if (have_j2)
                        acc12 = svmla_n_f32_x(pg1, acc12, av1, b2);
                    if (have_j3)
                        acc13 = svmla_n_f32_x(pg1, acc13, av1, b3);
                    if (have_j4)
                        acc14 = svmla_n_f32_x(pg1, acc14, av1, b4);
                    if (have_j5)
                        acc15 = svmla_n_f32_x(pg1, acc15, av1, b5);
                    if (have_j6)
                        acc16 = svmla_n_f32_x(pg1, acc16, av1, b6);
                    if (have_j7)
                        acc17 = svmla_n_f32_x(pg1, acc17, av1, b7);
                }
            }
            if (alpha != 1.0f) {
                acc00 = svmul_n_f32_x(pg0, acc00, alpha);
                if (have_j1)
                    acc01 = svmul_n_f32_x(pg0, acc01, alpha);
                if (have_j2)
                    acc02 = svmul_n_f32_x(pg0, acc02, alpha);
                if (have_j3)
                    acc03 = svmul_n_f32_x(pg0, acc03, alpha);
                if (have_j4)
                    acc04 = svmul_n_f32_x(pg0, acc04, alpha);
                if (have_j5)
                    acc05 = svmul_n_f32_x(pg0, acc05, alpha);
                if (have_j6)
                    acc06 = svmul_n_f32_x(pg0, acc06, alpha);
                if (have_j7)
                    acc07 = svmul_n_f32_x(pg0, acc07, alpha);
                if (have_i1) {
                    if (have_j1)
                        acc11 = svmul_n_f32_x(pg1, acc11, alpha);
                    if (have_j2)
                        acc12 = svmul_n_f32_x(pg1, acc12, alpha);
                    if (have_j3)
                        acc13 = svmul_n_f32_x(pg1, acc13, alpha);
                    if (have_j4)
                        acc14 = svmul_n_f32_x(pg1, acc14, alpha);
                    if (have_j5)
                        acc15 = svmul_n_f32_x(pg1, acc15, alpha);
                    if (have_j6)
                        acc16 = svmul_n_f32_x(pg1, acc16, alpha);
                    if (have_j7)
                        acc17 = svmul_n_f32_x(pg1, acc17, alpha);
                    acc10 = svmul_n_f32_x(pg1, acc10, alpha);
                }
            }
            float *c0 = C + i + j * ldc_size;
            svfloat32_t cv0 = svld1_f32(pg0, c0);
            cv0 = svadd_f32_x(pg0, cv0, acc00);
            svst1_f32(pg0, c0, cv0);
            if (have_j1) {
                float *c1 = C + i + j1 * ldc_size;
                svfloat32_t cv1 = svld1_f32(pg0, c1);
                cv1 = svadd_f32_x(pg0, cv1, acc01);
                svst1_f32(pg0, c1, cv1);
            }
            if (have_j2) {
                float *c2 = C + i + j2 * ldc_size;
                svfloat32_t cv2 = svld1_f32(pg0, c2);
                cv2 = svadd_f32_x(pg0, cv2, acc02);
                svst1_f32(pg0, c2, cv2);
            }
            if (have_j3) {
                float *c3 = C + i + j3 * ldc_size;
                svfloat32_t cv3 = svld1_f32(pg0, c3);
                cv3 = svadd_f32_x(pg0, cv3, acc03);
                svst1_f32(pg0, c3, cv3);
            }
            if (have_j4) {
                float *c4 = C + i + j4 * ldc_size;
                svfloat32_t cv4 = svld1_f32(pg0, c4);
                cv4 = svadd_f32_x(pg0, cv4, acc04);
                svst1_f32(pg0, c4, cv4);
            }
            if (have_j5) {
                float *c5 = C + i + j5 * ldc_size;
                svfloat32_t cv5 = svld1_f32(pg0, c5);
                cv5 = svadd_f32_x(pg0, cv5, acc05);
                svst1_f32(pg0, c5, cv5);
            }
            if (have_j6) {
                float *c6 = C + i + j6 * ldc_size;
                svfloat32_t cv6 = svld1_f32(pg0, c6);
                cv6 = svadd_f32_x(pg0, cv6, acc06);
                svst1_f32(pg0, c6, cv6);
            }
            if (have_j7) {
                float *c7 = C + i + j7 * ldc_size;
                svfloat32_t cv7 = svld1_f32(pg0, c7);
                cv7 = svadd_f32_x(pg0, cv7, acc07);
                svst1_f32(pg0, c7, cv7);
            }
            if (have_i1) {
                float *c10 = C + i + lanes + j * ldc_size;
                svfloat32_t cv10 = svld1_f32(pg1, c10);
                cv10 = svadd_f32_x(pg1, cv10, acc10);
                svst1_f32(pg1, c10, cv10);
                if (have_j1) {
                    float *c11 = C + i + lanes + j1 * ldc_size;
                    svfloat32_t cv11 = svld1_f32(pg1, c11);
                    cv11 = svadd_f32_x(pg1, cv11, acc11);
                    svst1_f32(pg1, c11, cv11);
                }
                if (have_j2) {
                    float *c12 = C + i + lanes + j2 * ldc_size;
                    svfloat32_t cv12 = svld1_f32(pg1, c12);
                    cv12 = svadd_f32_x(pg1, cv12, acc12);
                    svst1_f32(pg1, c12, cv12);
                }
                if (have_j3) {
                    float *c13 = C + i + lanes + j3 * ldc_size;
                    svfloat32_t cv13 = svld1_f32(pg1, c13);
                    cv13 = svadd_f32_x(pg1, cv13, acc13);
                    svst1_f32(pg1, c13, cv13);
                }
                if (have_j4) {
                    float *c14 = C + i + lanes + j4 * ldc_size;
                    svfloat32_t cv14 = svld1_f32(pg1, c14);
                    cv14 = svadd_f32_x(pg1, cv14, acc14);
                    svst1_f32(pg1, c14, cv14);
                }
                if (have_j5) {
                    float *c15 = C + i + lanes + j5 * ldc_size;
                    svfloat32_t cv15 = svld1_f32(pg1, c15);
                    cv15 = svadd_f32_x(pg1, cv15, acc15);
                    svst1_f32(pg1, c15, cv15);
                }
                if (have_j6) {
                    float *c16 = C + i + lanes + j6 * ldc_size;
                    svfloat32_t cv16 = svld1_f32(pg1, c16);
                    cv16 = svadd_f32_x(pg1, cv16, acc16);
                    svst1_f32(pg1, c16, cv16);
                }
                if (have_j7) {
                    float *c17 = C + i + lanes + j7 * ldc_size;
                    svfloat32_t cv17 = svld1_f32(pg1, c17);
                    cv17 = svadd_f32_x(pg1, cv17, acc17);
                    svst1_f32(pg1, c17, cv17);
                }
            }
        }
#if CAMBLAS_SVE_TIGHT_N
        j += edge_end - j < 8 ? edge_end - j : 8;
#endif
    }
    return 0;
}

static inline __attribute__((always_inline)) int
dgemm_sve_packed_body(int m, int n, int k, double alpha, const double *A, int lda, const double *B,
                      int ldb, double *C, int ldc)
{
    size_t lanes = (size_t)svcntd();
    size_t m_size = (size_t)m;
    size_t n_size = (size_t)n;
    size_t k_size = (size_t)k;
    size_t lda_size = (size_t)lda;
    size_t ldb_size = (size_t)ldb;
    size_t ldc_size = (size_t)ldc;

    if (lanes == 0)
        return CAMBLAS_PACKED_UNAVAILABLE;
    size_t fast_rows =
        (CAMBLAS_SVE_BLOCK6 && !CAMBLAS_SVE_COLS6 && !CAMBLAS_SVE_BALANCED_COLS ? 6U : 4U) * lanes;
    size_t fast_m = m_size - (m_size % fast_rows);
    size_t fast_cols =
        CAMBLAS_SVE_BALANCED_COLS ? CAMBLAS_SVE_BALANCED_COLS : (CAMBLAS_SVE_COLS6 ? 6U : 4U);
    size_t column_group =
        CAMBLAS_SVE_TIGHT_N ? fast_cols : (fast_cols == 5 ? 40U : (fast_cols == 6 ? 24U : 8U));
    size_t fast_n = n_size - n_size % column_group;
    const svbool_t bpg = svptrue_b64();
#if defined(CAMBLAS_SVE_ROW_FIRST) && CAMBLAS_SVE_ROW_FIRST
    for (size_t i = 0; i < fast_m; i += fast_rows) {
        for (size_t j = 0; j < fast_n; j += fast_cols) {
#else
    for (size_t j = 0; j < fast_n; j += fast_cols) {
        for (size_t i = 0; i < fast_m; i += fast_rows) {
#endif
#if CAMBLAS_SVE_BLOCK6
#if CAMBLAS_SVE_BALANCED_COLS
            camblas_balanced_f64(k, alpha, A + i, lda_size, B + j * ldb_size, ldb_size,
                                 C + i + j * ldc_size, ldc_size);
#elif CAMBLAS_SVE_COLS6
            camblas_block4x6_f64(k, alpha, A + i, lda_size, B + j * ldb_size, ldb_size,
                                 C + i + j * ldc_size, ldc_size);
#else
            camblas_block6_f64(k, alpha, A + i, lda_size, B + j * ldb_size, ldb_size,
                               C + i + j * ldc_size, ldc_size);
#endif
            continue;
#endif
            svbool_t pg0 = svwhilelt_b64((uint64_t)i, (uint64_t)m_size);
            svbool_t pg1 = svwhilelt_b64((uint64_t)(i + lanes), (uint64_t)m_size);
            svbool_t pg2 = svwhilelt_b64((uint64_t)(i + 2 * lanes), (uint64_t)m_size);
            svbool_t pg3 = svwhilelt_b64((uint64_t)(i + 3 * lanes), (uint64_t)m_size);
            svfloat64_t acc00 = svdup_f64(0.0);
            svfloat64_t acc01 = svdup_f64(0.0);
            svfloat64_t acc02 = svdup_f64(0.0);
            svfloat64_t acc03 = svdup_f64(0.0);
            svfloat64_t acc10 = svdup_f64(0.0);
            svfloat64_t acc11 = svdup_f64(0.0);
            svfloat64_t acc12 = svdup_f64(0.0);
            svfloat64_t acc13 = svdup_f64(0.0);
            svfloat64_t acc20 = svdup_f64(0.0);
            svfloat64_t acc21 = svdup_f64(0.0);
            svfloat64_t acc22 = svdup_f64(0.0);
            svfloat64_t acc23 = svdup_f64(0.0);
            svfloat64_t acc30 = svdup_f64(0.0);
            svfloat64_t acc31 = svdup_f64(0.0);
            svfloat64_t acc32 = svdup_f64(0.0);
            svfloat64_t acc33 = svdup_f64(0.0);
            const double *bp0 = B + j * ldb_size;
            const double *bp1 = bp0 + ldb_size;
            const double *bp2 = bp1 + ldb_size;
            const double *bp3 = bp2 + ldb_size;
            for (size_t l = 0; l < k_size; l++) {
                svfloat64_t b0 = camblas_sve_load_replicate_f64(bpg, bp0);
                svfloat64_t b1 = camblas_sve_load_replicate_f64(bpg, bp1);
                svfloat64_t b2 = camblas_sve_load_replicate_f64(bpg, bp2);
                svfloat64_t b3 = camblas_sve_load_replicate_f64(bpg, bp3);
                bp0++;
                bp1++;
                bp2++;
                bp3++;
                svfloat64_t av0 = svld1_f64(pg0, A + i + l * lda_size);
                acc00 = svmla_f64_x(pg0, acc00, av0, b0);
                acc01 = svmla_f64_x(pg0, acc01, av0, b1);
                acc02 = svmla_f64_x(pg0, acc02, av0, b2);
                acc03 = svmla_f64_x(pg0, acc03, av0, b3);
                svfloat64_t av1 = svld1_f64(pg1, A + i + lanes + l * lda_size);
                acc10 = svmla_f64_x(pg1, acc10, av1, b0);
                acc11 = svmla_f64_x(pg1, acc11, av1, b1);
                acc12 = svmla_f64_x(pg1, acc12, av1, b2);
                acc13 = svmla_f64_x(pg1, acc13, av1, b3);
                svfloat64_t av2 = svld1_f64(pg2, A + i + 2 * lanes + l * lda_size);
                acc20 = svmla_f64_x(pg2, acc20, av2, b0);
                acc21 = svmla_f64_x(pg2, acc21, av2, b1);
                acc22 = svmla_f64_x(pg2, acc22, av2, b2);
                acc23 = svmla_f64_x(pg2, acc23, av2, b3);
                svfloat64_t av3 = svld1_f64(pg3, A + i + 3 * lanes + l * lda_size);
                acc30 = svmla_f64_x(pg3, acc30, av3, b0);
                acc31 = svmla_f64_x(pg3, acc31, av3, b1);
                acc32 = svmla_f64_x(pg3, acc32, av3, b2);
                acc33 = svmla_f64_x(pg3, acc33, av3, b3);
            }
            if (alpha != 1.0) {
                acc00 = svmul_n_f64_x(pg0, acc00, alpha);
                acc01 = svmul_n_f64_x(pg0, acc01, alpha);
                acc02 = svmul_n_f64_x(pg0, acc02, alpha);
                acc03 = svmul_n_f64_x(pg0, acc03, alpha);
                acc10 = svmul_n_f64_x(pg1, acc10, alpha);
                acc11 = svmul_n_f64_x(pg1, acc11, alpha);
                acc12 = svmul_n_f64_x(pg1, acc12, alpha);
                acc13 = svmul_n_f64_x(pg1, acc13, alpha);
                acc20 = svmul_n_f64_x(pg2, acc20, alpha);
                acc21 = svmul_n_f64_x(pg2, acc21, alpha);
                acc22 = svmul_n_f64_x(pg2, acc22, alpha);
                acc23 = svmul_n_f64_x(pg2, acc23, alpha);
                acc30 = svmul_n_f64_x(pg3, acc30, alpha);
                acc31 = svmul_n_f64_x(pg3, acc31, alpha);
                acc32 = svmul_n_f64_x(pg3, acc32, alpha);
                acc33 = svmul_n_f64_x(pg3, acc33, alpha);
            }
            double *c0 = C + i + j * ldc_size;
            svfloat64_t cv0 = svld1_f64(pg0, c0);
            cv0 = svadd_f64_x(pg0, cv0, acc00);
            svst1_f64(pg0, c0, cv0);
            double *c1 = C + i + (j + 1) * ldc_size;
            svfloat64_t cv1 = svld1_f64(pg0, c1);
            cv1 = svadd_f64_x(pg0, cv1, acc01);
            svst1_f64(pg0, c1, cv1);
            double *c2 = C + i + (j + 2) * ldc_size;
            svfloat64_t cv2 = svld1_f64(pg0, c2);
            cv2 = svadd_f64_x(pg0, cv2, acc02);
            svst1_f64(pg0, c2, cv2);
            double *c3 = C + i + (j + 3) * ldc_size;
            svfloat64_t cv3 = svld1_f64(pg0, c3);
            cv3 = svadd_f64_x(pg0, cv3, acc03);
            svst1_f64(pg0, c3, cv3);
            double *c10 = C + i + lanes + j * ldc_size;
            svfloat64_t cv10 = svld1_f64(pg1, c10);
            cv10 = svadd_f64_x(pg1, cv10, acc10);
            svst1_f64(pg1, c10, cv10);
            double *c11 = C + i + lanes + (j + 1) * ldc_size;
            svfloat64_t cv11 = svld1_f64(pg1, c11);
            cv11 = svadd_f64_x(pg1, cv11, acc11);
            svst1_f64(pg1, c11, cv11);
            double *c12 = C + i + lanes + (j + 2) * ldc_size;
            svfloat64_t cv12 = svld1_f64(pg1, c12);
            cv12 = svadd_f64_x(pg1, cv12, acc12);
            svst1_f64(pg1, c12, cv12);
            double *c13 = C + i + lanes + (j + 3) * ldc_size;
            svfloat64_t cv13 = svld1_f64(pg1, c13);
            cv13 = svadd_f64_x(pg1, cv13, acc13);
            svst1_f64(pg1, c13, cv13);
            double *c20 = C + i + 2 * lanes + j * ldc_size;
            svfloat64_t cv20 = svld1_f64(pg2, c20);
            cv20 = svadd_f64_x(pg2, cv20, acc20);
            svst1_f64(pg2, c20, cv20);
            double *c21 = C + i + 2 * lanes + (j + 1) * ldc_size;
            svfloat64_t cv21 = svld1_f64(pg2, c21);
            cv21 = svadd_f64_x(pg2, cv21, acc21);
            svst1_f64(pg2, c21, cv21);
            double *c22 = C + i + 2 * lanes + (j + 2) * ldc_size;
            svfloat64_t cv22 = svld1_f64(pg2, c22);
            cv22 = svadd_f64_x(pg2, cv22, acc22);
            svst1_f64(pg2, c22, cv22);
            double *c23 = C + i + 2 * lanes + (j + 3) * ldc_size;
            svfloat64_t cv23 = svld1_f64(pg2, c23);
            cv23 = svadd_f64_x(pg2, cv23, acc23);
            svst1_f64(pg2, c23, cv23);
            double *c30 = C + i + 3 * lanes + j * ldc_size;
            svfloat64_t cv30 = svld1_f64(pg3, c30);
            cv30 = svadd_f64_x(pg3, cv30, acc30);
            svst1_f64(pg3, c30, cv30);
            double *c31 = C + i + 3 * lanes + (j + 1) * ldc_size;
            svfloat64_t cv31 = svld1_f64(pg3, c31);
            cv31 = svadd_f64_x(pg3, cv31, acc31);
            svst1_f64(pg3, c31, cv31);
            double *c32 = C + i + 3 * lanes + (j + 2) * ldc_size;
            svfloat64_t cv32 = svld1_f64(pg3, c32);
            cv32 = svadd_f64_x(pg3, cv32, acc32);
            svst1_f64(pg3, c32, cv32);
            double *c33 = C + i + 3 * lanes + (j + 3) * ldc_size;
            svfloat64_t cv33 = svld1_f64(pg3, c33);
            cv33 = svadd_f64_x(pg3, cv33, acc33);
            svst1_f64(pg3, c33, cv33);
        }
    }
    /* Split edge traversal at fast_n: bottom rows of full columns first,
       then all rows of the remaining columns, without double updates. */
#if CAMBLAS_SVE_TIGHT_N
    for (size_t j = 0; j < n_size;) {
        size_t edge_end = j < fast_n ? fast_n : n_size;
#else
    for (size_t j = 0; j < n_size; j += 8) {
        size_t edge_end = n_size;
#endif
        size_t j1 = j + 1;
        size_t j2 = j + 2;
        size_t j3 = j + 3;
        size_t j4 = j + 4;
        size_t j5 = j + 5;
        size_t j6 = j + 6;
        size_t j7 = j + 7;
        int have_j1 = j1 < edge_end;
        int have_j2 = j2 < edge_end;
        int have_j3 = j3 < edge_end;
        int have_j4 = j4 < edge_end;
        int have_j5 = j5 < edge_end;
        int have_j6 = j6 < edge_end;
        int have_j7 = j7 < edge_end;
        for (size_t i = j < fast_n ? fast_m : 0; i < m_size; i += 2 * lanes) {
            svbool_t pg0 = svwhilelt_b64((uint64_t)i, (uint64_t)m_size);
            int have_i1 = i + lanes < m_size;
            svbool_t pg1 = svwhilelt_b64((uint64_t)(i + lanes), (uint64_t)m_size);
            svfloat64_t acc00 = svdup_f64(0.0);
            svfloat64_t acc01 = svdup_f64(0.0);
            svfloat64_t acc02 = svdup_f64(0.0);
            svfloat64_t acc03 = svdup_f64(0.0);
            svfloat64_t acc04 = svdup_f64(0.0);
            svfloat64_t acc05 = svdup_f64(0.0);
            svfloat64_t acc06 = svdup_f64(0.0);
            svfloat64_t acc07 = svdup_f64(0.0);
            svfloat64_t acc10 = svdup_f64(0.0);
            svfloat64_t acc11 = svdup_f64(0.0);
            svfloat64_t acc12 = svdup_f64(0.0);
            svfloat64_t acc13 = svdup_f64(0.0);
            svfloat64_t acc14 = svdup_f64(0.0);
            svfloat64_t acc15 = svdup_f64(0.0);
            svfloat64_t acc16 = svdup_f64(0.0);
            svfloat64_t acc17 = svdup_f64(0.0);
            for (size_t l = 0; l < k_size; l++) {
                const double *bp = B + l + j * ldb_size;
                double b0 = bp[0];
                double b1 = have_j1 ? bp[ldb_size] : 0.0;
                double b2 = have_j2 ? bp[2 * ldb_size] : 0.0;
                double b3 = have_j3 ? bp[3 * ldb_size] : 0.0;
                double b4 = have_j4 ? bp[4 * ldb_size] : 0.0;
                double b5 = have_j5 ? bp[5 * ldb_size] : 0.0;
                double b6 = have_j6 ? bp[6 * ldb_size] : 0.0;
                double b7 = have_j7 ? bp[7 * ldb_size] : 0.0;
                svfloat64_t av0 = svld1_f64(pg0, A + i + l * lda_size);
                acc00 = svmla_n_f64_x(pg0, acc00, av0, b0);
                if (have_j1)
                    acc01 = svmla_n_f64_x(pg0, acc01, av0, b1);
                if (have_j2)
                    acc02 = svmla_n_f64_x(pg0, acc02, av0, b2);
                if (have_j3)
                    acc03 = svmla_n_f64_x(pg0, acc03, av0, b3);
                if (have_j4)
                    acc04 = svmla_n_f64_x(pg0, acc04, av0, b4);
                if (have_j5)
                    acc05 = svmla_n_f64_x(pg0, acc05, av0, b5);
                if (have_j6)
                    acc06 = svmla_n_f64_x(pg0, acc06, av0, b6);
                if (have_j7)
                    acc07 = svmla_n_f64_x(pg0, acc07, av0, b7);
                if (have_i1) {
                    svfloat64_t av1 = svld1_f64(pg1, A + i + lanes + l * lda_size);
                    acc10 = svmla_n_f64_x(pg1, acc10, av1, b0);
                    if (have_j1)
                        acc11 = svmla_n_f64_x(pg1, acc11, av1, b1);
                    if (have_j2)
                        acc12 = svmla_n_f64_x(pg1, acc12, av1, b2);
                    if (have_j3)
                        acc13 = svmla_n_f64_x(pg1, acc13, av1, b3);
                    if (have_j4)
                        acc14 = svmla_n_f64_x(pg1, acc14, av1, b4);
                    if (have_j5)
                        acc15 = svmla_n_f64_x(pg1, acc15, av1, b5);
                    if (have_j6)
                        acc16 = svmla_n_f64_x(pg1, acc16, av1, b6);
                    if (have_j7)
                        acc17 = svmla_n_f64_x(pg1, acc17, av1, b7);
                }
            }
            if (alpha != 1.0) {
                acc00 = svmul_n_f64_x(pg0, acc00, alpha);
                if (have_j1)
                    acc01 = svmul_n_f64_x(pg0, acc01, alpha);
                if (have_j2)
                    acc02 = svmul_n_f64_x(pg0, acc02, alpha);
                if (have_j3)
                    acc03 = svmul_n_f64_x(pg0, acc03, alpha);
                if (have_j4)
                    acc04 = svmul_n_f64_x(pg0, acc04, alpha);
                if (have_j5)
                    acc05 = svmul_n_f64_x(pg0, acc05, alpha);
                if (have_j6)
                    acc06 = svmul_n_f64_x(pg0, acc06, alpha);
                if (have_j7)
                    acc07 = svmul_n_f64_x(pg0, acc07, alpha);
                if (have_i1) {
                    if (have_j1)
                        acc11 = svmul_n_f64_x(pg1, acc11, alpha);
                    if (have_j2)
                        acc12 = svmul_n_f64_x(pg1, acc12, alpha);
                    if (have_j3)
                        acc13 = svmul_n_f64_x(pg1, acc13, alpha);
                    if (have_j4)
                        acc14 = svmul_n_f64_x(pg1, acc14, alpha);
                    if (have_j5)
                        acc15 = svmul_n_f64_x(pg1, acc15, alpha);
                    if (have_j6)
                        acc16 = svmul_n_f64_x(pg1, acc16, alpha);
                    if (have_j7)
                        acc17 = svmul_n_f64_x(pg1, acc17, alpha);
                    acc10 = svmul_n_f64_x(pg1, acc10, alpha);
                }
            }
            double *c0 = C + i + j * ldc_size;
            svfloat64_t cv0 = svld1_f64(pg0, c0);
            cv0 = svadd_f64_x(pg0, cv0, acc00);
            svst1_f64(pg0, c0, cv0);
            if (have_j1) {
                double *c1 = C + i + j1 * ldc_size;
                svfloat64_t cv1 = svld1_f64(pg0, c1);
                cv1 = svadd_f64_x(pg0, cv1, acc01);
                svst1_f64(pg0, c1, cv1);
            }
            if (have_j2) {
                double *c2 = C + i + j2 * ldc_size;
                svfloat64_t cv2 = svld1_f64(pg0, c2);
                cv2 = svadd_f64_x(pg0, cv2, acc02);
                svst1_f64(pg0, c2, cv2);
            }
            if (have_j3) {
                double *c3 = C + i + j3 * ldc_size;
                svfloat64_t cv3 = svld1_f64(pg0, c3);
                cv3 = svadd_f64_x(pg0, cv3, acc03);
                svst1_f64(pg0, c3, cv3);
            }
            if (have_j4) {
                double *c4 = C + i + j4 * ldc_size;
                svfloat64_t cv4 = svld1_f64(pg0, c4);
                cv4 = svadd_f64_x(pg0, cv4, acc04);
                svst1_f64(pg0, c4, cv4);
            }
            if (have_j5) {
                double *c5 = C + i + j5 * ldc_size;
                svfloat64_t cv5 = svld1_f64(pg0, c5);
                cv5 = svadd_f64_x(pg0, cv5, acc05);
                svst1_f64(pg0, c5, cv5);
            }
            if (have_j6) {
                double *c6 = C + i + j6 * ldc_size;
                svfloat64_t cv6 = svld1_f64(pg0, c6);
                cv6 = svadd_f64_x(pg0, cv6, acc06);
                svst1_f64(pg0, c6, cv6);
            }
            if (have_j7) {
                double *c7 = C + i + j7 * ldc_size;
                svfloat64_t cv7 = svld1_f64(pg0, c7);
                cv7 = svadd_f64_x(pg0, cv7, acc07);
                svst1_f64(pg0, c7, cv7);
            }
            if (have_i1) {
                double *c10 = C + i + lanes + j * ldc_size;
                svfloat64_t cv10 = svld1_f64(pg1, c10);
                cv10 = svadd_f64_x(pg1, cv10, acc10);
                svst1_f64(pg1, c10, cv10);
                if (have_j1) {
                    double *c11 = C + i + lanes + j1 * ldc_size;
                    svfloat64_t cv11 = svld1_f64(pg1, c11);
                    cv11 = svadd_f64_x(pg1, cv11, acc11);
                    svst1_f64(pg1, c11, cv11);
                }
                if (have_j2) {
                    double *c12 = C + i + lanes + j2 * ldc_size;
                    svfloat64_t cv12 = svld1_f64(pg1, c12);
                    cv12 = svadd_f64_x(pg1, cv12, acc12);
                    svst1_f64(pg1, c12, cv12);
                }
                if (have_j3) {
                    double *c13 = C + i + lanes + j3 * ldc_size;
                    svfloat64_t cv13 = svld1_f64(pg1, c13);
                    cv13 = svadd_f64_x(pg1, cv13, acc13);
                    svst1_f64(pg1, c13, cv13);
                }
                if (have_j4) {
                    double *c14 = C + i + lanes + j4 * ldc_size;
                    svfloat64_t cv14 = svld1_f64(pg1, c14);
                    cv14 = svadd_f64_x(pg1, cv14, acc14);
                    svst1_f64(pg1, c14, cv14);
                }
                if (have_j5) {
                    double *c15 = C + i + lanes + j5 * ldc_size;
                    svfloat64_t cv15 = svld1_f64(pg1, c15);
                    cv15 = svadd_f64_x(pg1, cv15, acc15);
                    svst1_f64(pg1, c15, cv15);
                }
                if (have_j6) {
                    double *c16 = C + i + lanes + j6 * ldc_size;
                    svfloat64_t cv16 = svld1_f64(pg1, c16);
                    cv16 = svadd_f64_x(pg1, cv16, acc16);
                    svst1_f64(pg1, c16, cv16);
                }
                if (have_j7) {
                    double *c17 = C + i + lanes + j7 * ldc_size;
                    svfloat64_t cv17 = svld1_f64(pg1, c17);
                    cv17 = svadd_f64_x(pg1, cv17, acc17);
                    svst1_f64(pg1, c17, cv17);
                }
            }
        }
#if CAMBLAS_SVE_TIGHT_N
        j += edge_end - j < 8 ? edge_end - j : 8;
#endif
    }
    return 0;
}

#else

static int sgemm_sve_packed_body(int m, int n, int k, float alpha, const float *A, int lda,
                                 const float *B, int ldb, float *C, int ldc)
{
    size_t lanes = (size_t)svcntw();
    size_t m_size = (size_t)m;
    size_t n_size = (size_t)n;
    size_t k_size = (size_t)k;
    size_t lda_size = (size_t)lda;
    size_t ldb_size = (size_t)ldb;
    size_t ldc_size = (size_t)ldc;

    if (lanes == 0)
        return CAMBLAS_PACKED_UNAVAILABLE;
    for (size_t j = 0; j < n_size; j++) {
        for (size_t i = 0; i < m_size; i += lanes) {
            svbool_t pg = svwhilelt_b32((uint64_t)i, (uint64_t)m_size);
            svfloat32_t acc = svdup_f32(0.0f);
            for (size_t l = 0; l < k_size; l++) {
                svfloat32_t av = svld1_f32(pg, A + i + l * lda_size);
                acc = svmla_n_f32_x(pg, acc, av, B[l + j * ldb_size]);
            }
            if (alpha != 1.0f)
                acc = svmul_n_f32_x(pg, acc, alpha);
            float *c = C + i + j * ldc_size;
            svfloat32_t cv = svld1_f32(pg, c);
            cv = svadd_f32_x(pg, cv, acc);
            svst1_f32(pg, c, cv);
        }
    }
    return 0;
}

static int dgemm_sve_packed_body(int m, int n, int k, double alpha, const double *A, int lda,
                                 const double *B, int ldb, double *C, int ldc)
{
    size_t lanes = (size_t)svcntd();
    size_t m_size = (size_t)m;
    size_t n_size = (size_t)n;
    size_t k_size = (size_t)k;
    size_t lda_size = (size_t)lda;
    size_t ldb_size = (size_t)ldb;
    size_t ldc_size = (size_t)ldc;

    if (lanes == 0)
        return CAMBLAS_PACKED_UNAVAILABLE;
    for (size_t j = 0; j < n_size; j++) {
        for (size_t i = 0; i < m_size; i += lanes) {
            svbool_t pg = svwhilelt_b64((uint64_t)i, (uint64_t)m_size);
            svfloat64_t acc = svdup_f64(0.0);
            for (size_t l = 0; l < k_size; l++) {
                svfloat64_t av = svld1_f64(pg, A + i + l * lda_size);
                acc = svmla_n_f64_x(pg, acc, av, B[l + j * ldb_size]);
            }
            if (alpha != 1.0)
                acc = svmul_n_f64_x(pg, acc, alpha);
            double *c = C + i + j * ldc_size;
            svfloat64_t cv = svld1_f64(pg, c);
            cv = svadd_f64_x(pg, cv, acc);
            svst1_f64(pg, c, cv);
        }
    }
    return 0;
}

#endif /* CAMBLAS_SVE_NR == 4 */

#endif /* CAMBLAS_SVE_ACLE */

int camblas_sgemm_sve(char trans_a, char trans_b, int m, int n, int k, float alpha, const float *A,
                      int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
#if CAMBLAS_SVE_ACLE
    if (!sve_valid_gemm_args(
            trans_a, trans_b, m, n, k, lda, ldb, ldc, A, B, C, alpha != 0.0f && k > 0,
            camblas_gemm_output_access_needed(k, alpha != 0.0f, beta != 1.0f), sizeof(float)) ||
        !sve_runtime_ready() || trans_a != 'N' || (trans_b != 'N' && trans_b != 'T'))
        return CAMBLAS_SVE_UNAVAILABLE;
    if ((alpha == 0.0f || k == 0) && beta == 1.0f)
        return 0;
#if CAMBLAS_SVE_NR == 4
    /* The register-blocked body uses ordinary column-major strides. It can
       therefore consume NN inputs directly, without panel allocation/copy.
       Keep no-product cases in the original access-aware body. */
    if (trans_b == 'N' && alpha != 0.0f && k > 0 && (size_t)m >= 4 * svcntw() && n >= 8) {
        if (beta != 1.0f) {
            for (size_t j = 0; j < (size_t)n; ++j) {
                for (size_t i = 0; i < (size_t)m; i += svcntw()) {
                    svbool_t pg = svwhilelt_b32((uint64_t)i, (uint64_t)m);
                    float *c = C + i + j * (size_t)ldc;
                    svfloat32_t v = svdup_f32(0.0f);
                    if (beta != 0.0f)
                        v = svmul_n_f32_x(pg, svld1_f32(pg, c), beta);
                    svst1_f32(pg, c, v);
                }
            }
        }
        return sgemm_sve_packed_body(m, n, k, alpha, A, lda, B, ldb, C, ldc);
    }
    return sgemm_sve_body_nr4(trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
    return sgemm_sve_body(trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
#else
    (void)trans_a;
    (void)trans_b;
    (void)m;
    (void)n;
    (void)k;
    (void)alpha;
    (void)A;
    (void)lda;
    (void)B;
    (void)ldb;
    (void)beta;
    (void)C;
    (void)ldc;
    return CAMBLAS_SVE_UNAVAILABLE;
#endif
}

int camblas_dgemm_sve(char trans_a, char trans_b, int m, int n, int k, double alpha,
                      const double *A, int lda, const double *B, int ldb, double beta, double *C,
                      int ldc)
{
#if CAMBLAS_SVE_ACLE
    if (!sve_valid_gemm_args(
            trans_a, trans_b, m, n, k, lda, ldb, ldc, A, B, C, alpha != 0.0 && k > 0,
            camblas_gemm_output_access_needed(k, alpha != 0.0, beta != 1.0), sizeof(double)) ||
        !sve_runtime_ready() || trans_a != 'N' || (trans_b != 'N' && trans_b != 'T'))
        return CAMBLAS_SVE_UNAVAILABLE;
    if ((alpha == 0.0 || k == 0) && beta == 1.0)
        return 0;
#if CAMBLAS_SVE_NR == 4
    if (trans_b == 'N' && alpha != 0.0 && k > 0 && (size_t)m >= 4 * svcntd() && n >= 8) {
        if (beta != 1.0) {
            for (size_t j = 0; j < (size_t)n; ++j) {
                for (size_t i = 0; i < (size_t)m; i += svcntd()) {
                    svbool_t pg = svwhilelt_b64((uint64_t)i, (uint64_t)m);
                    double *c = C + i + j * (size_t)ldc;
                    svfloat64_t v = svdup_f64(0.0);
                    if (beta != 0.0)
                        v = svmul_n_f64_x(pg, svld1_f64(pg, c), beta);
                    svst1_f64(pg, c, v);
                }
            }
        }
        return dgemm_sve_packed_body(m, n, k, alpha, A, lda, B, ldb, C, ldc);
    }
    return dgemm_sve_body_nr4(trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
    return dgemm_sve_body(trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
#else
    (void)trans_a;
    (void)trans_b;
    (void)m;
    (void)n;
    (void)k;
    (void)alpha;
    (void)A;
    (void)lda;
    (void)B;
    (void)ldb;
    (void)beta;
    (void)C;
    (void)ldc;
    return CAMBLAS_SVE_UNAVAILABLE;
#endif
}

#if CAMBLAS_SVE_ACLE
#include "sve_interleaved.h"
#include "sve_micro6.h"
#endif

int camblas_sgemm_sve_interleaved_tile(int m, int n, int k, float alpha, const float *A, int lda,
                                       const float *B, int ldb, float *C, int ldc)
{
#if CAMBLAS_SVE_ACLE
    if (!sve_valid_packed_tile(m, n, k, lda, ldb, ldc, A, B, C, alpha != 0 && k > 0,
                               alpha != 0 && k > 0, sizeof(*C)) ||
        !sve_runtime_ready())
        return CAMBLAS_PACKED_UNAVAILABLE;
    if (alpha == 0 || m == 0 || n == 0 || k == 0)
        return 0;
    /* The last B microgroup reads all eight allocated columns. */
    size_t np = ((size_t)n + 7) / 8 * 8;
    if (np > SIZE_MAX / sizeof(*B) / (size_t)ldb)
        return CAMBLAS_PACKED_UNAVAILABLE;
#if defined(CAMBLAS_MICRO8_ALPHA_ONE) && CAMBLAS_MICRO8_ALPHA_ONE
    if (alpha == 1.0f)
        return camblas_micro8_f32(m, n, k, 1.0f, A, lda, B, ldb, C, ldc);
#endif
    return camblas_micro8_f32(m, n, k, alpha, A, lda, B, ldb, C, ldc);
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
    return CAMBLAS_PACKED_UNAVAILABLE;
#endif
}

int camblas_dgemm_sve_interleaved_tile(int m, int n, int k, double alpha, const double *A, int lda,
                                       const double *B, int ldb, double *C, int ldc)
{
#if CAMBLAS_SVE_ACLE
    if (!sve_valid_packed_tile(m, n, k, lda, ldb, ldc, A, B, C, alpha != 0 && k > 0,
                               alpha != 0 && k > 0, sizeof(*C)) ||
        !sve_runtime_ready())
        return CAMBLAS_PACKED_UNAVAILABLE;
    if (alpha == 0 || m == 0 || n == 0 || k == 0)
        return 0;
    /* The last B microgroup reads all eight allocated columns. */
    size_t np = ((size_t)n + 7) / 8 * 8;
    if (np > SIZE_MAX / sizeof(*B) / (size_t)ldb)
        return CAMBLAS_PACKED_UNAVAILABLE;
#if defined(CAMBLAS_MICRO8_ALPHA_ONE) && CAMBLAS_MICRO8_ALPHA_ONE
    if (alpha == 1.0)
        return camblas_micro8_f64(m, n, k, 1.0, A, lda, B, ldb, C, ldc);
#endif
    return camblas_micro8_f64(m, n, k, alpha, A, lda, B, ldb, C, ldc);
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
    return CAMBLAS_PACKED_UNAVAILABLE;
#endif
}

int camblas_sgemm_sve_interleaved_init_tile(int m, int n, int k, float alpha, const float *A,
                                            int lda, const float *B, int ldb, float *C, int ldc)
{
#if CAMBLAS_SVE_ACLE
    if (!sve_valid_packed_tile(m, n, k, lda, ldb, ldc, A, B, C, alpha != 0 && k > 0,
                               alpha != 0 && k > 0, sizeof(*C)) ||
        !sve_runtime_ready())
        return CAMBLAS_PACKED_UNAVAILABLE;
    if (alpha == 0 || m == 0 || n == 0 || k == 0)
        return 0;
    /* The last B microgroup reads all eight allocated columns. */
    size_t np = ((size_t)n + 7) / 8 * 8;
    if (np > SIZE_MAX / sizeof(*B) / (size_t)ldb)
        return CAMBLAS_PACKED_UNAVAILABLE;
#if defined(CAMBLAS_MICRO8_ALPHA_ONE) && CAMBLAS_MICRO8_ALPHA_ONE
    if (alpha == 1.0f)
        return camblas_micro8_init_f32(m, n, k, 1.0f, A, lda, B, ldb, C, ldc);
#endif
    return camblas_micro8_init_f32(m, n, k, alpha, A, lda, B, ldb, C, ldc);
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
    return CAMBLAS_PACKED_UNAVAILABLE;
#endif
}

int camblas_dgemm_sve_interleaved_init_tile(int m, int n, int k, double alpha, const double *A,
                                            int lda, const double *B, int ldb, double *C, int ldc)
{
#if CAMBLAS_SVE_ACLE
    if (!sve_valid_packed_tile(m, n, k, lda, ldb, ldc, A, B, C, alpha != 0 && k > 0,
                               alpha != 0 && k > 0, sizeof(*C)) ||
        !sve_runtime_ready())
        return CAMBLAS_PACKED_UNAVAILABLE;
    if (alpha == 0 || m == 0 || n == 0 || k == 0)
        return 0;
    /* The last B microgroup reads all eight allocated columns. */
    size_t np = ((size_t)n + 7) / 8 * 8;
    if (np > SIZE_MAX / sizeof(*B) / (size_t)ldb)
        return CAMBLAS_PACKED_UNAVAILABLE;
#if defined(CAMBLAS_MICRO8_ALPHA_ONE) && CAMBLAS_MICRO8_ALPHA_ONE
    if (alpha == 1.0)
        return camblas_micro8_init_f64(m, n, k, 1.0, A, lda, B, ldb, C, ldc);
#endif
    return camblas_micro8_init_f64(m, n, k, alpha, A, lda, B, ldb, C, ldc);
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
    return CAMBLAS_PACKED_UNAVAILABLE;
#endif
}

int camblas_dgemm_sve_amicro_tile(int m, int n, int k, double alpha, const double *A, int lda,
                                  const double *B, int ldb, double *C, int ldc)
{
#if CAMBLAS_SVE_ACLE
    if (m < 0 || n < 0 || k < 0 || !A || !B || !C || lda < 1 || ldb < 1 || ldc < 1 ||
        !sve_runtime_ready() || CAMBLAS_MICRO8_ROWS * svcntd() != 6)
        return CAMBLAS_PACKED_UNAVAILABLE;
    if (m == 0 || n == 0)
        return 0;
    if (ldc < m || ldb < k || (k > 0 && lda != k))
        return CAMBLAS_PACKED_UNAVAILABLE;
    if (alpha == 0 || k == 0)
        return 0;
    size_t mp = ((size_t)m + 5) / 6 * 6, np = ((size_t)n + 7) / 8 * 8;
    if (mp > SIZE_MAX / sizeof(*A) / (size_t)lda || np > SIZE_MAX / sizeof(*B) / (size_t)ldb ||
        camblas_matrix_span_fits(m, n, ldc, sizeof(*C)) != 0)
        return CAMBLAS_PACKED_UNAVAILABLE;
    return camblas_micro8_amicro_f64(m, n, k, alpha, A, lda, B, ldb, C, ldc);
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
    return CAMBLAS_PACKED_UNAVAILABLE;
#endif
}

int camblas_sgemm_sve_micro6_tile(int m, int n, int k, float alpha, const float *A, int lda,
                                  const float *B, int ldb, float *C, int ldc)
{
#if CAMBLAS_SVE_ACLE
    if (!sve_valid_packed_tile(m, n, k, lda, ldb, ldc, A, B, C, alpha != 0 && k > 0,
                               alpha != 0 && k > 0, sizeof(*C)) ||
        !sve_runtime_ready())
        return CAMBLAS_PACKED_UNAVAILABLE;
    if (alpha == 0 || m == 0 || n == 0 || k == 0)
        return 0;
    /* The last B microgroup reads all six allocated columns. */
    size_t np = ((size_t)n + 5) / 6 * 6;
    if (np > SIZE_MAX / sizeof(*B) / (size_t)ldb)
        return CAMBLAS_PACKED_UNAVAILABLE;
    return camblas_micro6_f32(m, n, k, alpha, A, lda, B, ldb, C, ldc);
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
    return CAMBLAS_PACKED_UNAVAILABLE;
#endif
}

int camblas_dgemm_sve_micro6_tile(int m, int n, int k, double alpha, const double *A, int lda,
                                  const double *B, int ldb, double *C, int ldc)
{
#if CAMBLAS_SVE_ACLE
    if (!sve_valid_packed_tile(m, n, k, lda, ldb, ldc, A, B, C, alpha != 0 && k > 0,
                               alpha != 0 && k > 0, sizeof(*C)) ||
        !sve_runtime_ready())
        return CAMBLAS_PACKED_UNAVAILABLE;
    if (alpha == 0 || m == 0 || n == 0 || k == 0)
        return 0;
    /* The last B microgroup reads all six allocated columns. */
    size_t np = ((size_t)n + 5) / 6 * 6;
    if (np > SIZE_MAX / sizeof(*B) / (size_t)ldb)
        return CAMBLAS_PACKED_UNAVAILABLE;
    return camblas_micro6_f64(m, n, k, alpha, A, lda, B, ldb, C, ldc);
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
    return CAMBLAS_PACKED_UNAVAILABLE;
#endif
}

int camblas_sgemm_sve_micro6_padded_tile(int m, int n, int k, float alpha, const float *A, int lda,
                                         const float *B, int ldb, float *C, int ldc)
{
#if CAMBLAS_SVE_ACLE
    if (!sve_valid_packed_tile(m, n, k, lda, ldb, ldc, A, B, C, alpha != 0 && k > 0,
                               alpha != 0 && k > 0, sizeof(*C)) ||
        !sve_runtime_ready())
        return CAMBLAS_PACKED_UNAVAILABLE;
    if (alpha == 0 || m == 0 || n == 0 || k == 0)
        return 0;
    /* The last B microgroup reads six logical columns in eight allocated slots. */
    size_t np = ((size_t)n + 5) / 6 * 8;
    if (np > SIZE_MAX / sizeof(*B) / (size_t)ldb)
        return CAMBLAS_PACKED_UNAVAILABLE;
    return camblas_micro6_f32_padded(m, n, k, alpha, A, lda, B, ldb, C, ldc);
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
    return CAMBLAS_PACKED_UNAVAILABLE;
#endif
}

int camblas_dgemm_sve_micro6_padded_tile(int m, int n, int k, double alpha, const double *A,
                                         int lda, const double *B, int ldb, double *C, int ldc)
{
#if CAMBLAS_SVE_ACLE
    if (!sve_valid_packed_tile(m, n, k, lda, ldb, ldc, A, B, C, alpha != 0 && k > 0,
                               alpha != 0 && k > 0, sizeof(*C)) ||
        !sve_runtime_ready())
        return CAMBLAS_PACKED_UNAVAILABLE;
    if (alpha == 0 || m == 0 || n == 0 || k == 0)
        return 0;
    /* The last B microgroup reads six logical columns in eight allocated slots. */
    size_t np = ((size_t)n + 5) / 6 * 8;
    if (np > SIZE_MAX / sizeof(*B) / (size_t)ldb)
        return CAMBLAS_PACKED_UNAVAILABLE;
    return camblas_micro6_f64_padded(m, n, k, alpha, A, lda, B, ldb, C, ldc);
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
    return CAMBLAS_PACKED_UNAVAILABLE;
#endif
}

int camblas_sgemm_sve_packed_tile(int m, int n, int k, float alpha, const float *A, int lda,
                                  const float *B, int ldb, float *C, int ldc)
{
#if CAMBLAS_SVE_ACLE
    if (!sve_valid_packed_tile(m, n, k, lda, ldb, ldc, A, B, C, alpha != 0.0f && k > 0,
                               alpha != 0.0f && k > 0, sizeof(float)) ||
        !sve_runtime_ready())
        return CAMBLAS_PACKED_UNAVAILABLE;
    /* The packed caller handles alpha=0 before entering the tile loop. Keep
       the direct hidden boundary no-read safe as well. */
    if (alpha == 0.0f || m == 0 || n == 0 || k == 0)
        return 0;
    return sgemm_sve_packed_body(m, n, k, alpha, A, lda, B, ldb, C, ldc);
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
    return CAMBLAS_PACKED_UNAVAILABLE;
#endif
}

int camblas_dgemm_sve_packed_tile(int m, int n, int k, double alpha, const double *A, int lda,
                                  const double *B, int ldb, double *C, int ldc)
{
#if CAMBLAS_SVE_ACLE
    if (!sve_valid_packed_tile(m, n, k, lda, ldb, ldc, A, B, C, alpha != 0.0 && k > 0,
                               alpha != 0.0 && k > 0, sizeof(double)) ||
        !sve_runtime_ready())
        return CAMBLAS_PACKED_UNAVAILABLE;
    if (alpha == 0.0 || m == 0 || n == 0 || k == 0)
        return 0;
    return dgemm_sve_packed_body(m, n, k, alpha, A, lda, B, ldb, C, ldc);
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
    return CAMBLAS_PACKED_UNAVAILABLE;
#endif
}
