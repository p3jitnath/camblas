#ifndef CAMBLAS_BENCH_FUSED_SUM_API_H
#define CAMBLAS_BENCH_FUSED_SUM_API_H
/* Research-only exports, present only with CAMBLAS_EXPERIMENTAL_SUM_PACK=1.
 * Not part of the public BLAS ABI. Caller provides disjoint valid matrices,
 * workspace and a synchronous executor. Only NN sum operands are supported. */
#include "camblas_executor.h"
#include <stddef.h>
int camblas_experimental_sgemm_sum_workspace(char trans_a, char trans_b, int m, int n, int k,
                                             float alpha, const float *A, int lda, const float *B,
                                             int ldb, float beta, float *C, int ldc, int mc, int nc,
                                             int kc, int mr, int nr,
                                             const camblas_executor_t *executor, void *workspace,
                                             size_t bytes, const float *A2, int lda2, int sign_a,
                                             const float *B2, int ldb2, int sign_b);
int camblas_experimental_dgemm_sum_workspace(char trans_a, char trans_b, int m, int n, int k,
                                             double alpha, const double *A, int lda,
                                             const double *B, int ldb, double beta, double *C,
                                             int ldc, int mc, int nc, int kc, int mr, int nr,
                                             const camblas_executor_t *executor, void *workspace,
                                             size_t bytes, const double *A2, int lda2, int sign_a,
                                             const double *B2, int ldb2, int sign_b);
#endif
