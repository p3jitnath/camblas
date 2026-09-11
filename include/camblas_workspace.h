#ifndef CAMBLAS_WORKSPACE_H
#define CAMBLAS_WORKSPACE_H
#include "camblas_planner.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Experimental additive API. Existing context layout and entry points are
 * unchanged. The caller owns workspace and must keep it alive, writable,
 * double-aligned, and disjoint from accessed A/B/C storage for the full call.
 * Concurrent or nested calls need distinct workspaces. Sequential reuse is
 * allowed, including after an error; B is packed anew on every call.
 *
 * NULL with zero bytes is equivalent to the existing plan API. Non-NULL
 * workspace is currently consumed by monolithic shared-B packed builds.
 * Other selected kernels may leave it unused. An unavailable workspace
 * backend, insufficient capacity, or incompatible packed layout fails before
 * computation rather than silently allocating a replacement B buffer.
 * Private A packing and executor state may still allocate. Workspace contents
 * are unspecified after every call; the library never retains or frees it.
 */
int camblas_workspace_bytes(int n, int k, int dtype, size_t *bytes);
int camblas_sgemm_plan_workspace(const camblas_ctx_t *ctx, char trans_a, char trans_b, int m, int n,
                                 int k, float alpha, const float *A, int lda, const float *B,
                                 int ldb, float beta, float *C, int ldc, camblas_plan_t *plan_out,
                                 void *workspace, size_t bytes);
int camblas_dgemm_plan_workspace(const camblas_ctx_t *ctx, char trans_a, char trans_b, int m, int n,
                                 int k, double alpha, const double *A, int lda, const double *B,
                                 int ldb, double beta, double *C, int ldc, camblas_plan_t *plan_out,
                                 void *workspace, size_t bytes);

#ifdef __cplusplus
}
#endif
#endif
