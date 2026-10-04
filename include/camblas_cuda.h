/* Experimental CUDA backend: real FP32/FP64, column-major GEMM, device-accessible pointers. */
#ifndef CAMBLAS_CUDA_H
#define CAMBLAS_CUDA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct camblas_cuda_context camblas_cuda_context;

enum camblas_cuda_algorithm {
    CAMBLAS_CUDA_AUTO = 0,
    CAMBLAS_CUDA_CLASSICAL = 1,
    CAMBLAS_CUDA_STRASSEN = 2,
    CAMBLAS_CUDA_LT = 3,
    CAMBLAS_CUDA_SYMMETRIC = 4,
    CAMBLAS_CUDA_STRASSEN_TWO = 5,
    CAMBLAS_CUDA_STRASSEN_THREE = 6,
    CAMBLAS_CUDA_STRASSEN_FOUR = 7
};

/*
 * One context owns one device and one CUDA stream, cuBLAS handles and scratch.
 * stream is a cudaStream_t converted to void*. Calls enqueue work asynchronously
 * except guarded Strassen, which checks its packed-input range before GEMM.
 * The caller keeps device-accessible operands alive until that stream completes. A context
 * must not be called concurrently or used after fork. Use distinct contexts for
 * different streams or host threads. Destruction waits for the context stream.
 * Warm up scratch-using operations before graph capture. Replay on the context
 * stream, or explicitly serialise replay with all context work. Captured scratch
 * addresses survive later growth until context destruction; destroy graphs before
 * their context. Low-memory third-level scratch allocation tries two levels, then classical GEMM.
 * Four-level scratch allocation failure falls back to classical GEMM.
 * Return codes: 0 success; negative failure, described by camblas_cuda_error().
 */
int camblas_cuda_create(int device, void *stream, camblas_cuda_context **out);
int camblas_cuda_destroy(camblas_cuda_context *context);
const char *camblas_cuda_error(const camblas_cuda_context *context);
int camblas_cuda_set_algorithm(camblas_cuda_context *context, int algorithm);

/*
 * C = alpha op(A) op(B) + beta C, LP64 dimensions and column-major leading
 * dimensions. trans_a/trans_b are 'N' or 'T'. Output may not overlap the
 * input operands. beta==0 does not read C. All pointers must be non-NULL for
 * positive output dimensions; A/B may be NULL when alpha==0 or k==0.
 * SYMMETRIC exploits exact Gram symmetry when A==B, transpose flags differ,
 * leading dimensions match and beta==0. STRASSEN modes only apply to eligible
 * even NN squares; bounded-input checks retain a classical fallback. It changes
 * rounding and does not guarantee small relative error for cancellation.
 * Ordinary CPU allocations are valid only on systems whose CUDA device reports
 * pageableMemoryAccess and pageableMemoryAccessUsesHostPageTables. CPU callers
 * must wait for the stream before reading outputs or modifying inputs.
 */
int camblas_cuda_sgemm(camblas_cuda_context *context, char trans_a, char trans_b, int m, int n,
                       int k, float alpha, const float *a, int lda, const float *b, int ldb,
                       float beta, float *c, int ldc);
int camblas_cuda_dgemm(camblas_cuda_context *context, char trans_a, char trans_b, int m, int n,
                       int k, double alpha, const double *a, int lda, const double *b, int ldb,
                       double beta, double *c, int ldc);

/* BF16 input/output storage with FP32 accumulation and alpha/beta. Device
 * pointers address BF16 values; layouts and ownership follow GEMM above.
 * BF16 uses cuBLAS/cuBLASLt, including for explicitly requested Strassen modes. */
int camblas_cuda_bgemm(camblas_cuda_context *context, char trans_a, char trans_b, int m, int n,
                       int k, float alpha, const void *a, int lda, const void *b, int ldb,
                       float beta, void *c, int ldc);

/* dtype: 0 FP32, 1 FP64. Row-major affine result = X W + bias, optionally ReLU.
 * Output and hidden arrays are caller-owned contiguous device storage. */
int camblas_cuda_affine(camblas_cuda_context *context, int dtype, int rows, int inner, int columns,
                        const void *x, const void *weight, const void *bias, int relu,
                        void *output);
int camblas_cuda_mlp(camblas_cuda_context *context, int dtype, int rows, int inputs, int hidden,
                     int outputs, const void *x, const void *w1, const void *b1, const void *w2,
                     const void *b2, void *hidden_output, void *output);

/* Dense single-head attention: softmax(scale Q K^T) V. Row-major device
 * operands; no dropout or mask. Queries and keys may have different row counts.
 * Scratch belongs to the context. The scale is caller-specified. */
int camblas_cuda_attention(camblas_cuda_context *context, int dtype, int queries, int keys,
                           int depth, int values, double scale, const void *q, const void *k,
                           const void *v, void *output);

/* MLP first derivatives. Every non-NULL gradient output is filled; NULL means
 * that gradient was not requested. grad_hidden is caller-owned rows*hidden
 * scratch when dx, dw1 or db1 is requested. All arrays are row-major. */
int camblas_cuda_mlp_backward(camblas_cuda_context *context, int dtype, int rows, int inputs,
                              int hidden, int outputs, const void *x, const void *w1,
                              const void *w2, const void *hidden_output, const void *grad_output,
                              void *grad_hidden, void *dx, void *dw1, void *db1, void *dw2,
                              void *db2);

/* Inference-only elementwise SiLU(gate)*up with FP32 arithmetic and an
 * intermediate conversion to the storage type. dtype: 0 FP32, 2 BF16.
 * Exact in-place output==gate/up is supported; partial overlap is not. */
int camblas_cuda_silu_multiply(camblas_cuda_context *context, int dtype, uint64_t count,
                               const void *gate, const void *up, void *output);
/* SiLU(gate)*up, optionally multiplied by one FP32 routing value per row.
 * dtype: 0 FP32, 2 BF16; FP32 intermediates round only at the final output.
 * limit=0 disables clipping; positive finite limit caps gate above and up on
 * both sides. width>0 divides count. routing may be null. Exact in-place
 * output==gate/up is supported; partial overlap and output==routing are not.
 * Buffers remain caller-owned and execution uses the context stream. */
int camblas_cuda_swiglu(camblas_cuda_context *context, int dtype, uint64_t count, int width,
                        float limit, const void *gate, const void *up, const float *routing,
                        void *output);
/* Row-major FP8 E4M3 input [rows,inner] times weight [outputs,inner]^T,
 * accumulating in FP32 and writing BF16 [rows,outputs]. packed=1 stores E2M1
 * nibbles in low/high order [outputs,inner/2]; packed=0 stores FP8 bytes.
 * E8M0 input scales are [rows,inner/activation_block]; weight scales are
 * [outputs,inner/32] for FP4 or [ceil(outputs/32),inner/32] for FP8.
 * activation_block is 32 (both types) or 128 (FP4 only). SM90+ binary required;
 * outputs is a positive multiple of eight, inner a positive multiple of 32,
 * rows is 0..65535. Buffers must not overlap and must belong to this device.
 * Input is four-byte aligned; weight alignment is two/four bytes for FP4/FP8.
 * Outputs and operand lifetimes remain caller-owned; execution is asynchronous. */
int camblas_cuda_quantized_matmul(camblas_cuda_context *context, int packed, int rows, int outputs,
                                  int inner, int activation_block, const void *input,
                                  const void *input_scale, const void *weight,
                                  const void *weight_scale, void *output);
/* Batched quantized products: input [groups,rows,inner], output [groups,rows,outputs].
 * metadata is device uint64 [weight_groups,2] containing weight/scale addresses
 * in the preceding operation's layouts. active/counts are device int32 [groups].
 * Invalid active indices produce zeros; counts are clipped to [0,rows]. All
 * padding is zeroed. groups/weight_groups <=65535. Operands, metadata and the
 * referenced weight storages remain caller-owned, non-overlapping and alive
 * until the context stream finishes. No activation or product data is cached. */
int camblas_cuda_grouped_quantized_matmul(camblas_cuda_context *context, int packed, int groups,
                                          int weight_groups, int rows, int outputs, int inner,
                                          int activation_block, const void *input,
                                          const void *input_scale, const void *metadata,
                                          const void *active, const void *counts, void *output);
/* Route int64 expert IDs [tokens,choices] to local int32 active IDs [groups].
 * first_expert converts local IDs to global IDs; IDs outside [0,INT_MAX) are ignored.
 * Initialises counts [groups], slots [groups,rows] and reverse [tokens,choices].
 * Unassigned slots/reverse are -1; excess rows are discarded. Duplicate active
 * IDs select their first group. All buffers are device-resident, disjoint and
 * caller-owned. groups/rows <=65535; tokens*choices and groups*rows <=INT_MAX. */
int camblas_cuda_route_groups(camblas_cuda_context *context, int tokens, int choices,
                              int first_expert, int groups, int rows, const void *experts,
                              const void *active, void *counts, void *slots, void *reverse);
/* Sum BF16 grouped rows in ascending expert-ID/choice order into FP32 [tokens,width].
 * reverse maps int64 expert IDs [tokens,choices] to input [grouped_rows,width].
 * Negative/out-of-range expert IDs or reverse rows are ignored. choices is 1..64.
 * Buffers remain caller-owned, disjoint and alive until the stream completes. */
int camblas_cuda_reduce_groups(camblas_cuda_context *context, int tokens, int choices, int width,
                               int grouped_rows, const void *experts, const void *reverse,
                               const void *input, void *output);
/* Row-major RMS normalisation in FP32, rounding normalised values to the
 * storage type before multiplying by weight. dtype: 0 FP32, 2 BF16.
 * Weight has width elements. Inputs/output must not partially overlap. */
int camblas_cuda_rms_norm(camblas_cuda_context *context, int dtype, int rows, int width,
                          float epsilon, const void *input, const void *weight, void *output);
/* Single-row RMS with explicit storage-rounding and warp-reduction policies.
 * dtype/weight_dtype: 0 FP32, 2 BF16; FP32 input requires FP32 weight.
 * round_before_weight and descending are boolean. width: 2048..65536, divisible
 * by four. All buffers use their type's alignment and must not overlap. */
int camblas_cuda_rms_norm_decode(camblas_cuda_context *context, int dtype, int weight_dtype,
                                 int width, float epsilon, int round_before_weight, int descending,
                                 const void *input, const void *weight, void *output);
/* Single-row residual sum and RMS norm. dtype: 0 FP32, 2 BF16. Both outputs
 * must be distinct caller-owned buffers that do not alias inputs; width is
 * 2048..65536, divisible by four. Sum is rounded to storage dtype first. */
int camblas_cuda_add_rms_norm(camblas_cuda_context *context, int dtype, int width, float epsilon,
                              const void *input, const void *residual, const void *weight,
                              void *added, void *output);
/* BF16-to-FP32 squares and final RMS scaling allow the tensor binding to retain
 * PyTorch's FP32 mean reduction order. Means contains count/width FP32 entries. */
int camblas_cuda_bfloat16_square(camblas_cuda_context *context, uint64_t count, const void *input,
                                 float *output);
int camblas_cuda_rms_scale(camblas_cuda_context *context, int dtype, uint64_t count, int width,
                           float epsilon, const void *input, const void *weight, const float *means,
                           void *output);

/* Host launch counters: classical GEMM, Strassen, symmetric Gram, affine, LT,
 * guarded classical fallback, attention, MLP backward. length is 6 or 8.
 * Counters describe launched work, not completion. */
int camblas_cuda_stats(const camblas_cuda_context *context, uint64_t *counts, int length);
int camblas_cuda_reset_stats(camblas_cuda_context *context);

#ifdef __cplusplus
}
#endif
#endif
