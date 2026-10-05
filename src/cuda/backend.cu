/* CAMBLAS CUDA backend. Device packing and fusion are independently implemented. */
#include "camblas_cuda.h"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <mma.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <tuple>
#include <type_traits>
#include <vector>

namespace
{
thread_local char creation_error[256] = "";
using PlanKey = std::tuple<int, int, int, int, int, int, int, int, int, int, int, int, int, int>;

struct LtPlan {
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr, d = nullptr;
    cublasLtMatmulAlgo_t algorithm{};
    bool available = false;
    bool tuned = false;
    bool prefer_unfused = false;
    bool broadcast_bias = false;
    std::vector<cublasLtMatmulHeuristicResult_t> candidates;
    ~LtPlan()
    {
        if (operation)
            cublasLtMatmulDescDestroy(operation);
        if (a)
            cublasLtMatrixLayoutDestroy(a);
        if (b)
            cublasLtMatrixLayoutDestroy(b);
        if (c)
            cublasLtMatrixLayoutDestroy(c);
        if (d)
            cublasLtMatrixLayoutDestroy(d);
    }
};
} // namespace

struct camblas_cuda_context {
    int device = 0;
    int algorithm = CAMBLAS_CUDA_AUTO;
    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;
    cublasLtHandle_t lt = nullptr;
    void *lt_workspace = nullptr;
    size_t lt_workspace_bytes = 32u * 1024u * 1024u;
    void *scratch = nullptr;
    size_t scratch_bytes = 0;
    bool scratch_captured = false;
    bool short_attention_supported = false;
    bool quantized_supported = false;
    bool fp8_decode_supported = false;
    std::vector<void *> retired_allocations;
    unsigned *guard = nullptr;
    unsigned *host_guard = nullptr;
    std::atomic<uint64_t> counts[8] = {};
    char error[256] = "";
    std::map<PlanKey, std::unique_ptr<LtPlan>> plans;
};

namespace
{
int fail(camblas_cuda_context *context, const char *operation, int code)
{
    char *message = context ? context->error : creation_error;
    std::snprintf(message, 256, "%s failed (status %d)", operation, code);
    return code > 0 ? -code : -1;
}

template <typename T> __global__ void scale_output(int m, int n, T beta, T *c, int ldc)
{
    size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= size_t(m) * n)
        return;
    size_t offset = index % m + (index / m) * ldc;
    c[offset] = beta == T(0) ? T(0) : beta * c[offset];
}

template <typename T> __global__ void mirror_triangle(int n, T *c, int ldc)
{
    size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= size_t(n) * n)
        return;
    int row = int(index % n), column = int(index / n);
    if (row < column)
        c[row + size_t(column) * ldc] = c[column + size_t(row) * ldc];
}

template <typename T>
__global__ void pack_strassen(const T *input, int ld, int h, T *packed, T limit, unsigned *unsafe,
                              bool right)
{
    size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t length = size_t(h) * h;
    if (index >= length)
        return;
    input += size_t(blockIdx.y) * (2 * h) * ld;
    packed += size_t(blockIdx.y) * 7 * length;
    int row = int(index % h), column = int(index / h);
    T q11 = input[row + size_t(column) * ld];
    T q12 = input[row + size_t(column + h) * ld];
    T q21 = input[row + h + size_t(column) * ld];
    T q22 = input[row + h + size_t(column + h) * ld];
    if (unsafe) {
        bool bad =
            !(fabs(q11) <= limit && fabs(q12) <= limit && fabs(q21) <= limit && fabs(q22) <= limit);
        if (__any_sync(__activemask(), bad) && (threadIdx.x & 31) == 0)
            atomicOr(unsafe, 1u);
    }
    if (!right) {
        packed[index] = q11 + q22;
        packed[length + index] = q21 + q22;
        packed[2 * length + index] = q11;
        packed[3 * length + index] = q22;
        packed[4 * length + index] = q11 + q12;
        packed[5 * length + index] = q21 - q11;
        packed[6 * length + index] = q12 - q22;
    } else {
        packed[index] = q11 + q22;
        packed[length + index] = q11;
        packed[2 * length + index] = q12 - q22;
        packed[3 * length + index] = q21 - q11;
        packed[4 * length + index] = q22;
        packed[5 * length + index] = q11 + q12;
        packed[6 * length + index] = q21 + q22;
    }
}

template <typename T>
__global__ void recombine_strassen(const T *products, int h, T alpha, T beta, T *c, int ldc)
{
    size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t length = size_t(h) * h;
    if (index >= length)
        return;
    products += size_t(blockIdx.y) * 7 * length;
    c += size_t(blockIdx.y) * (2 * h) * ldc;
    T p1 = products[index], p2 = products[length + index];
    T p3 = products[2 * length + index], p4 = products[3 * length + index];
    T p5 = products[4 * length + index], p6 = products[5 * length + index];
    T p7 = products[6 * length + index];
    int row = int(index % h), column = int(index / h);
    size_t offsets[4] = {row + size_t(column) * ldc, row + size_t(column + h) * ldc,
                         row + h + size_t(column) * ldc, row + h + size_t(column + h) * ldc};
    T values[4] = {((p1 + p4) - p5) + p7, p3 + p5, p2 + p4, ((p1 - p2) + p3) + p6};
    for (int q = 0; q < 4; ++q)
        c[offsets[q]] = beta == T(0) ? alpha * values[q] : alpha * values[q] + beta * c[offsets[q]];
}

template <bool Right, typename T>
__device__ __forceinline__ T packed_term(T q11, T q12, T q21, T q22, int product)
{
    if constexpr (Right) {
        switch (product) {
        case 0:
            return q11 + q22;
        case 1:
            return q11;
        case 2:
            return q12 - q22;
        case 3:
            return q21 - q11;
        case 4:
            return q22;
        case 5:
            return q11 + q12;
        default:
            return q21 + q22;
        }
    } else {
        switch (product) {
        case 0:
            return q11 + q22;
        case 1:
            return q21 + q22;
        case 2:
            return q11;
        case 3:
            return q22;
        case 4:
            return q11 + q12;
        case 5:
            return q21 - q11;
        default:
            return q12 - q22;
        }
    }
}

/* Form the 49 quarter-size operands directly, preserving both levels' sum order.
 * No half-size operand buffers are written or read between packing stages. */
template <typename T, bool Right>
__global__ void pack_strassen_two(const T *input, int ld, int quarter, T *packed, T limit,
                                  unsigned *unsafe)
{
    size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t length = size_t(quarter) * quarter;
    if (index >= length)
        return;
    int row = int(index % quarter), column = int(index / quarter);
    T original[4][4];
    bool bad = false;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        int r = row + (q / 2) * quarter, col = column + (q % 2) * quarter;
        original[q][0] = input[r + size_t(col) * ld];
        original[q][1] = input[r + size_t(col + 2 * quarter) * ld];
        original[q][2] = input[r + 2 * quarter + size_t(col) * ld];
        original[q][3] = input[r + 2 * quarter + size_t(col + 2 * quarter) * ld];
#pragma unroll
        for (int outer = 0; outer < 4; ++outer)
            bad |= !(fabs(original[q][outer]) <= limit);
    }
    if (__any_sync(__activemask(), bad) && (threadIdx.x & 31) == 0)
        atomicOr(unsafe, 1u);
#pragma unroll
    for (int parent = 0; parent < 7; ++parent) {
        T inner[4];
#pragma unroll
        for (int q = 0; q < 4; ++q)
            inner[q] = packed_term<Right>(original[q][0], original[q][1], original[q][2],
                                          original[q][3], parent);
#pragma unroll
        for (int product = 0; product < 7; ++product)
            packed[size_t(parent * 7 + product) * length + index] =
                packed_term<Right>(inner[0], inner[1], inner[2], inner[3], product);
    }
}

template <typename T>
__device__ __forceinline__ T child_result(const T *products, size_t length, int quadrant)
{
    switch (quadrant) {
    case 0:
        return ((products[0] + products[3 * length]) - products[4 * length]) + products[6 * length];
    case 1:
        return products[2 * length] + products[4 * length];
    case 2:
        return products[length] + products[3 * length];
    default:
        return ((products[0] - products[length]) + products[2 * length]) + products[5 * length];
    }
}

/* Neighbouring blocks handle the four inner quadrants, so their shared product
 * reads can reuse L2. Half-size intermediate products are never stored. */
template <typename T>
__global__ void recombine_strassen_two(const T *products, int quarter, T alpha, T beta, T *c,
                                       int ldc)
{
    int quadrant = int(blockIdx.x % 4);
    size_t index = size_t(blockIdx.x / 4) * blockDim.x + threadIdx.x;
    size_t length = size_t(quarter) * quarter;
    if (index >= length)
        return;
    T p[7];
#pragma unroll
    for (int parent = 0; parent < 7; ++parent)
        p[parent] = child_result(products + size_t(parent) * 7 * length + index, length, quadrant);
    T values[4] = {((p[0] + p[3]) - p[4]) + p[6], p[2] + p[4], p[1] + p[3],
                   ((p[0] - p[1]) + p[2]) + p[5]};
    int row = int(index % quarter) + (quadrant / 2) * quarter;
    int column = int(index / quarter) + (quadrant % 2) * quarter;
    int half = 2 * quarter;
    size_t offsets[4] = {row + size_t(column) * ldc, row + size_t(column + half) * ldc,
                         row + half + size_t(column) * ldc,
                         row + half + size_t(column + half) * ldc};
#pragma unroll
    for (int q = 0; q < 4; ++q)
        c[offsets[q]] = beta == T(0) ? alpha * values[q] : alpha * values[q] + beta * c[offsets[q]];
}

/* Form all third-level operands without materialising either parent level. */
template <typename T, bool Right>
__global__ void pack_strassen_three(const T *input, int ld, int eighth, T *packed, T limit,
                                    unsigned *unsafe)
{
    size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t length = size_t(eighth) * eighth;
    if (index >= length)
        return;
    int row = int(index % eighth), column = int(index / eighth);
    T original[4][4][4];
    bool bad = false;
#pragma unroll
    for (int low = 0; low < 4; ++low)
#pragma unroll
        for (int middle = 0; middle < 4; ++middle)
#pragma unroll
            for (int outer = 0; outer < 4; ++outer) {
                int r = row + (low / 2 + 2 * (middle / 2) + 4 * (outer / 2)) * eighth;
                int col = column + (low % 2 + 2 * (middle % 2) + 4 * (outer % 2)) * eighth;
                T value = input[r + size_t(col) * ld];
                original[low][middle][outer] = value;
                bad |= !(fabs(value) <= limit);
            }
    if (__any_sync(__activemask(), bad) && (threadIdx.x & 31) == 0)
        atomicOr(unsafe, 1u);
#pragma unroll
    for (int parent = 0; parent < 7; ++parent) {
        T middle_values[4][4];
#pragma unroll
        for (int low = 0; low < 4; ++low)
#pragma unroll
            for (int middle = 0; middle < 4; ++middle)
                middle_values[low][middle] =
                    packed_term<Right>(original[low][middle][0], original[low][middle][1],
                                       original[low][middle][2], original[low][middle][3], parent);
#pragma unroll
        for (int middle = 0; middle < 7; ++middle) {
            T inner[4];
#pragma unroll
            for (int low = 0; low < 4; ++low)
                inner[low] =
                    packed_term<Right>(middle_values[low][0], middle_values[low][1],
                                       middle_values[low][2], middle_values[low][3], middle);
#pragma unroll
            for (int product = 0; product < 7; ++product)
                packed[size_t((parent * 7 + middle) * 7 + product) * length + index] =
                    packed_term<Right>(inner[0], inner[1], inner[2], inner[3], product);
        }
    }
}

template <typename T>
__global__ void recombine_strassen_three(const T *products, int eighth, T alpha, T beta, T *c,
                                         int ldc)
{
    int quadrant = int(blockIdx.x % 16), low = quadrant % 4, middle = quadrant / 4;
    size_t index = size_t(blockIdx.x / 16) * blockDim.x + threadIdx.x;
    size_t length = size_t(eighth) * eighth;
    if (index >= length)
        return;
    T outer_values[7];
#pragma unroll
    for (int parent = 0; parent < 7; ++parent) {
        T inner[7];
#pragma unroll
        for (int child = 0; child < 7; ++child)
            inner[child] = child_result(products + size_t(parent * 7 + child) * 7 * length + index,
                                        length, low);
        outer_values[parent] = child_result(inner, 1, middle);
    }
    T values[4] = {child_result(outer_values, 1, 0), child_result(outer_values, 1, 1),
                   child_result(outer_values, 1, 2), child_result(outer_values, 1, 3)};
    int row = int(index % eighth) + (low / 2 + 2 * (middle / 2)) * eighth;
    int column = int(index / eighth) + (low % 2 + 2 * (middle % 2)) * eighth;
    int half = 4 * eighth;
    size_t offsets[4] = {row + size_t(column) * ldc, row + size_t(column + half) * ldc,
                         row + half + size_t(column) * ldc,
                         row + half + size_t(column + half) * ldc};
#pragma unroll
    for (int q = 0; q < 4; ++q)
        c[offsets[q]] = beta == T(0) ? alpha * values[q] : alpha * values[q] + beta * c[offsets[q]];
}

template <typename T>
__global__ void bias_activation(T *values, const T *bias, size_t length, int columns, bool relu)
{
    size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= length)
        return;
    T value = values[index] + (bias ? bias[index % columns] : T(0));
    values[index] = relu && value < T(0) ? T(0) : value;
}

template <typename T>
cublasStatus_t classical(camblas_cuda_context *context, cublasOperation_t ta, cublasOperation_t tb,
                         int m, int n, int k, T alpha, const T *a, int lda, const T *b, int ldb,
                         T beta, T *c, int ldc)
{
    context->counts[0]++;
    if constexpr (std::is_same_v<T, float>)
        return cublasSgemm(context->blas, ta, tb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
    else
        return cublasDgemm(context->blas, ta, tb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
}

int reserve_scratch(camblas_cuda_context *context, size_t required)
{
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    cudaError_t error = cudaStreamIsCapturing(context->stream, &capture);
    if (error != cudaSuccess)
        return fail(context, "scratch capture status", error);
    if (capture != cudaStreamCaptureStatusNone) {
        if (required > context->scratch_bytes)
            return fail(context, "warm up scratch storage before CUDA graph capture", 1);
        context->scratch_captured = true;
        return 0;
    }
    if (required > context->scratch_bytes) {
        error = cudaStreamSynchronize(context->stream);
        if (error != cudaSuccess)
            return fail(context, "scratch synchronisation", error);
        void *replacement = nullptr;
        error = cudaMalloc(&replacement, required);
        if (error != cudaSuccess) {
            cudaGetLastError();
            return fail(context, "scratch allocation", error);
        }
        if (context->scratch_captured) {
            try {
                context->retired_allocations.push_back(context->scratch);
            } catch (...) {
                cudaFree(replacement);
                throw;
            }
        } else if (context->scratch) {
            cudaFree(context->scratch);
        }
        context->scratch = replacement;
        context->scratch_bytes = required;
        context->scratch_captured = false;
    }
    return 0;
}

#include "strassen_four.cuh"
#include "decode_float.cuh"
#include "quantized.cuh"
#include "fp8_decode.cuh"
#include "routing.cuh"

template <typename T>
int strassen(camblas_cuda_context *context, int n, T alpha, const T *a, int lda, const T *b,
             int ldb, T beta, T *c, int ldc)
{
    if (n % 16 == 0 && (context->algorithm == CAMBLAS_CUDA_STRASSEN_FOUR ||
                        (context->algorithm == CAMBLAS_CUDA_AUTO && n % 256 == 0 &&
                         n >= (std::is_same_v<T, float> ? 12288 : 24576))))
        return strassen_four_streamed(context, n, alpha, a, lda, b, ldb, beta, c, ldc);
    int h = n / 2;
    bool two = n % 4 == 0 && (context->algorithm == CAMBLAS_CUDA_STRASSEN_TWO ||
                              context->algorithm == CAMBLAS_CUDA_STRASSEN_THREE ||
                              context->algorithm == CAMBLAS_CUDA_STRASSEN_FOUR ||
                              (context->algorithm == CAMBLAS_CUDA_AUTO && n >= 8192));
    bool three = two && n % 8 == 0 &&
                 (context->algorithm == CAMBLAS_CUDA_STRASSEN_THREE ||
                  context->algorithm == CAMBLAS_CUDA_STRASSEN_FOUR ||
                  (context->algorithm == CAMBLAS_CUDA_AUTO && n % 128 == 0 &&
                   n >= (std::is_same_v<T, double> ? 16384 : 32768)));
    int size, batches, allocation;
    size_t stride;
    bool downgraded = false;
    for (;;) {
        size = three ? h / 4 : (two ? h / 2 : h);
        batches = three ? 343 : (two ? 49 : 7);
        stride = size_t(size) * size;
        if (stride > std::numeric_limits<size_t>::max() / (3 * size_t(batches) * sizeof(T)))
            return fail(context, "Strassen scratch size overflow", 2);
        allocation = reserve_scratch(context, 3 * size_t(batches) * stride * sizeof(T));
        if (allocation == -int(cudaErrorMemoryAllocation) && three) {
            // Retain the existing two-level route when a larger arena cannot fit.
            three = false;
            downgraded = true;
            context->counts[5]++;
            continue;
        }
        break;
    }
    if (allocation == -int(cudaErrorMemoryAllocation)) {
        if (!downgraded)
            context->counts[5]++;
        cublasStatus_t status = classical(context, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, alpha, a, lda,
                                          b, ldb, beta, c, ldc);
        return status == CUBLAS_STATUS_SUCCESS ? 0 : fail(context, "low-memory GEMM", status);
    }
    if (allocation)
        return allocation;
    T *pa = static_cast<T *>(context->scratch), *pb = pa + batches * stride,
      *products = pb + batches * stride;
    T limit =
        T(std::sqrt(std::numeric_limits<T>::max() / ((three ? 1024.0 : (two ? 256.0 : 64.0)) * n *
                                                     std::max(1.0, std::fabs(double(alpha))))));
    cudaError_t error = cudaMemsetAsync(context->guard, 0, sizeof(unsigned), context->stream);
    if (error != cudaSuccess)
        return fail(context, "range-flag reset", error);
    unsigned blocks = unsigned((stride + 255) / 256);
    if (three) {
        pack_strassen_three<T, false>
            <<<blocks, 256, 0, context->stream>>>(a, lda, size, pa, limit, context->guard);
        pack_strassen_three<T, true>
            <<<blocks, 256, 0, context->stream>>>(b, ldb, size, pb, limit, context->guard);
    } else if (two) {
        pack_strassen_two<T, false>
            <<<blocks, 256, 0, context->stream>>>(a, lda, size, pa, limit, context->guard);
        pack_strassen_two<T, true>
            <<<blocks, 256, 0, context->stream>>>(b, ldb, size, pb, limit, context->guard);
    } else {
        pack_strassen<<<blocks, 256, 0, context->stream>>>(a, lda, h, pa, limit, context->guard,
                                                           false);
        pack_strassen<<<blocks, 256, 0, context->stream>>>(b, ldb, h, pb, limit, context->guard,
                                                           true);
    }
    error = cudaMemcpyAsync(context->host_guard, context->guard, sizeof(unsigned),
                            cudaMemcpyDeviceToHost, context->stream);
    if (error != cudaSuccess)
        return fail(context, "range-flag copy", error);
    error = cudaStreamSynchronize(context->stream);
    if (error != cudaSuccess)
        return fail(context, "packed-input range check", error);
    if (*context->host_guard) {
        if (!downgraded)
            context->counts[5]++;
        cublasStatus_t status = classical(context, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, alpha, a, lda,
                                          b, ldb, beta, c, ldc);
        return status == CUBLAS_STATUS_SUCCESS ? 0 : fail(context, "guarded GEMM", status);
    }
    T one = 1, zero = 0;
    T *gemm_a = pa, *gemm_b = pb, *gemm_c = products;
    cublasStatus_t status;
    if constexpr (std::is_same_v<T, float>)
        status = cublasSgemmStridedBatched(context->blas, CUBLAS_OP_N, CUBLAS_OP_N, size, size,
                                           size, &one, gemm_a, size, stride, gemm_b, size, stride,
                                           &zero, gemm_c, size, stride, batches);
    else
        status = cublasDgemmStridedBatched(context->blas, CUBLAS_OP_N, CUBLAS_OP_N, size, size,
                                           size, &one, gemm_a, size, stride, gemm_b, size, stride,
                                           &zero, gemm_c, size, stride, batches);
    if (status != CUBLAS_STATUS_SUCCESS)
        return fail(context, "packed batched GEMM", status);
    context->counts[1]++;
    if (three)
        recombine_strassen_three<<<blocks * 16, 256, 0, context->stream>>>(products, size, alpha,
                                                                           beta, c, ldc);
    else if (two)
        recombine_strassen_two<<<blocks * 4, 256, 0, context->stream>>>(products, size, alpha, beta,
                                                                        c, ldc);
    else
        recombine_strassen<<<blocks, 256, 0, context->stream>>>(products, h, alpha, beta, c, ldc);
    error = cudaGetLastError();
    return error == cudaSuccess ? 0 : fail(context, "Strassen recombination", error);
}

unsigned pointer_alignment(const void *pointer)
{
    uintptr_t address = reinterpret_cast<uintptr_t>(pointer);
    return unsigned(std::min<uintptr_t>(256, address & (uintptr_t(0) - address)));
}

template <typename T>
LtPlan *get_plan(camblas_cuda_context *context, cublasOperation_t ta, cublasOperation_t tb, int m,
                 int n, int k, int lda, int ldb, int ldc, const T *bias, const T *a_pointer,
                 const T *b_pointer, const T *c_pointer, bool relu)
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                  std::is_same_v<T, __nv_bfloat16>);
    int type = std::is_same_v<T, float> ? 0 : (std::is_same_v<T, double> ? 1 : 2);
    unsigned aa = pointer_alignment(a_pointer), ab = pointer_alignment(b_pointer);
    unsigned ac = pointer_alignment(c_pointer), bias_alignment = pointer_alignment(bias);
    PlanKey key(type, int(ta), int(tb), m, n, k, lda, ldb, ldc, bias ? (relu ? 2 : 1) : 0, int(aa),
                int(ab), int(ac), int(bias_alignment));
    auto existing = context->plans.find(key);
    if (existing != context->plans.end())
        return existing->second.get();
    // Wide FP32 panels benefit from a larger search/workspace. Keep short
    // panels and other precisions on the established allocation budget.
    bool wide_panel =
        type == 0 && m >= 1024 && k >= 4096 && n >= 256 && n <= 1024 && (m >= 16384 || k >= 16384);
    if (wide_panel && context->lt_workspace_bytes < 128u * 1024u * 1024u) {
        cudaStreamCaptureStatus capture;
        if (cudaStreamIsCapturing(context->stream, &capture) == cudaSuccess &&
            capture == cudaStreamCaptureStatusNone) {
            context->retired_allocations.reserve(context->retired_allocations.size() + 1);
            void *grown = nullptr;
            size_t bytes = 128u * 1024u * 1024u;
            cudaError_t error = cudaMalloc(&grown, bytes);
            if (error == cudaSuccess) {
                if (cublasSetWorkspace(context->blas, grown, bytes) == CUBLAS_STATUS_SUCCESS) {
                    // Earlier graphs may still reference the original workspace.
                    context->retired_allocations.push_back(context->lt_workspace);
                    context->lt_workspace = grown;
                    context->lt_workspace_bytes = bytes;
                } else
                    cudaFree(grown);
            } else
                cudaGetLastError();
        }
    }
    auto owned = std::make_unique<LtPlan>();
    LtPlan *plan = owned.get();
    plan->broadcast_bias = bias && std::is_same_v<T, double>;
    context->plans.emplace(key, std::move(owned));
    cudaDataType_t datatype = type == 0 ? CUDA_R_32F : (type == 1 ? CUDA_R_64F : CUDA_R_16BF);
    cudaDataType_t scale_type = type == 1 ? CUDA_R_64F : CUDA_R_32F;
    cublasComputeType_t compute = type == 1 ? CUBLAS_COMPUTE_64F : CUBLAS_COMPUTE_32F;
    if (cublasLtMatmulDescCreate(&plan->operation, compute, scale_type) != CUBLAS_STATUS_SUCCESS)
        return plan;
    if (cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_TRANSA, &ta,
                                       sizeof(ta)) != CUBLAS_STATUS_SUCCESS)
        return plan;
    if (cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_TRANSB, &tb,
                                       sizeof(tb)) != CUBLAS_STATUS_SUCCESS)
        return plan;
    if (bias) {
        auto epilogue = plan->broadcast_bias
                            ? (relu ? CUBLASLT_EPILOGUE_RELU : CUBLASLT_EPILOGUE_DEFAULT)
                            : (relu ? CUBLASLT_EPILOGUE_RELU_BIAS : CUBLASLT_EPILOGUE_BIAS);
        if (cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                           &epilogue, sizeof(epilogue)) != CUBLAS_STATUS_SUCCESS)
            return plan;
        if (!plan->broadcast_bias &&
            cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                           &bias, sizeof(bias)) != CUBLAS_STATUS_SUCCESS)
            return plan;
    }
    if (cublasLtMatrixLayoutCreate(&plan->a, datatype, ta == CUBLAS_OP_N ? m : k,
                                   ta == CUBLAS_OP_N ? k : m, lda) != CUBLAS_STATUS_SUCCESS)
        return plan;
    if (cublasLtMatrixLayoutCreate(&plan->b, datatype, tb == CUBLAS_OP_N ? k : n,
                                   tb == CUBLAS_OP_N ? n : k, ldb) != CUBLAS_STATUS_SUCCESS)
        return plan;
    if (cublasLtMatrixLayoutCreate(&plan->c, datatype, m, n, ldc) != CUBLAS_STATUS_SUCCESS)
        return plan;
    if (plan->broadcast_bias) {
        int64_t zero = 0;
        if (cublasLtMatrixLayoutSetAttribute(plan->c, CUBLASLT_MATRIX_LAYOUT_LD, &zero,
                                             sizeof(zero)) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatrixLayoutCreate(&plan->d, datatype, m, n, ldc) != CUBLAS_STATUS_SUCCESS)
            return plan;
    }
    cublasLtMatmulPreference_t preference;
    if (cublasLtMatmulPreferenceCreate(&preference) != CUBLAS_STATUS_SUCCESS)
        return plan;
    cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                         &context->lt_workspace_bytes, sizeof(size_t));
    cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
                                         &aa, sizeof(aa));
    cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                                         &ab, sizeof(ab));
    cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
                                         plan->broadcast_bias ? &bias_alignment : &ac, sizeof(ac));
    cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,
                                         &ac, sizeof(ac));
    if (type == 2) {
        uint32_t reductions = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
        cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
                                             &reductions, sizeof(reductions));
    }
    cublasLtMatmulHeuristicResult_t candidates[64];
    int requested = wide_panel ? 64 : 12;
    int count = 0;
    cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
        context->lt, plan->operation, plan->a, plan->b, plan->c, plan->d ? plan->d : plan->c,
        preference, requested, candidates, &count);
    cublasLtMatmulPreferenceDestroy(preference);
    if (status == CUBLAS_STATUS_SUCCESS && count > 0) {
        plan->algorithm = candidates[0].algo;
        plan->available = true;
        for (int i = 0; i < count; ++i)
            if (candidates[i].state == CUBLAS_STATUS_SUCCESS)
                plan->candidates.push_back(candidates[i]);
    }
    return plan;
}

template <typename T, typename Scalar = T>
cublasStatus_t lt_call(camblas_cuda_context *context, LtPlan *plan,
                       const cublasLtMatmulAlgo_t *algorithm, Scalar alpha, const T *a, const T *b,
                       Scalar beta, T *c, const T *bias = nullptr)
{
    Scalar broadcast_beta = Scalar(1);
    return cublasLtMatmul(context->lt, plan->operation, &alpha, a, plan->a, b, plan->b,
                          plan->broadcast_bias ? &broadcast_beta : &beta,
                          plan->broadcast_bias ? bias : c, plan->c, c, plan->d ? plan->d : plan->c,
                          algorithm, context->lt_workspace, context->lt_workspace_bytes,
                          context->stream);
}

template <typename T>
cublasStatus_t lt_gemm(camblas_cuda_context *context, cublasOperation_t ta, cublasOperation_t tb,
                       int m, int n, int k, T alpha, const T *a, int lda, const T *b, int ldb,
                       T beta, T *c, int ldc, const T *bias = nullptr, bool relu = false)
{
    LtPlan *plan = get_plan(context, ta, tb, m, n, k, lda, ldb, ldc, bias, a, b, c, relu);
    if (!plan->available)
        return CUBLAS_STATUS_NOT_SUPPORTED;
    if (bias && !plan->broadcast_bias) {
        cublasStatus_t status = cublasLtMatmulDescSetAttribute(
            plan->operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
        if (status != CUBLAS_STATUS_SUCCESS)
            return status;
    }
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (!plan->tuned && beta == T(0))
        cudaStreamIsCapturing(context->stream, &capture);
    auto unfused = [&]() {
        cublasStatus_t status =
            bias ? lt_gemm(context, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
                 : classical(context, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        if (status == CUBLAS_STATUS_SUCCESS && bias) {
            bias_activation<<<unsigned((size_t(m) * n + 255) / 256), 256, 0, context->stream>>>(
                c, bias, size_t(m) * n, m, relu);
            if (cudaGetLastError() != cudaSuccess)
                return CUBLAS_STATUS_EXECUTION_FAILED;
        }
        return status;
    };
    // Tuning only overwrites beta-zero output; no input or caller-owned C is
    // changed for beta != 0. All tuning and allocations belong to warm-up.
    if (!plan->tuned && beta == T(0) && capture == cudaStreamCaptureStatusNone) {
        cudaEvent_t begin = nullptr, end = nullptr;
        if (cudaEventCreate(&begin) == cudaSuccess && cudaEventCreate(&end) == cudaSuccess) {
            float fastest = std::numeric_limits<float>::infinity();
            for (const auto &candidate : plan->candidates) {
                cublasStatus_t status =
                    lt_call(context, plan, &candidate.algo, alpha, a, b, beta, c, bias);
                if (status != CUBLAS_STATUS_SUCCESS)
                    continue;
                cudaEventRecord(begin, context->stream);
                for (int repeat = 0; repeat < 3; ++repeat)
                    status = lt_call(context, plan, &candidate.algo, alpha, a, b, beta, c, bias);
                cudaEventRecord(end, context->stream);
                cudaError_t error = cudaEventSynchronize(end);
                float milliseconds = 0;
                cudaEventElapsedTime(&milliseconds, begin, end);
                if (error == cudaSuccess && status == CUBLAS_STATUS_SUCCESS &&
                    milliseconds < fastest) {
                    fastest = milliseconds;
                    plan->algorithm = candidate.algo;
                }
            }
            if (!bias || ldc == m) {
                cublasStatus_t status = unfused();
                cudaEventRecord(begin, context->stream);
                for (int repeat = 0; repeat < 3; ++repeat)
                    status = unfused();
                cudaEventRecord(end, context->stream);
                cudaError_t error = cudaEventSynchronize(end);
                float milliseconds = 0;
                cudaEventElapsedTime(&milliseconds, begin, end);
                if (error == cudaSuccess && status == CUBLAS_STATUS_SUCCESS) {
                    // GPU-only batches hide dispatch overhead. Compare completed
                    // calls with the selected Lt algorithm before choosing a tie.
                    auto latency = [&](auto operation) {
                        double total = 0;
                        for (int repeat = 0; repeat < 3; ++repeat) {
                            auto start = std::chrono::steady_clock::now();
                            cublasStatus_t result = operation();
                            cudaError_t finished = cudaStreamSynchronize(context->stream);
                            if (result != CUBLAS_STATUS_SUCCESS || finished != cudaSuccess)
                                return std::numeric_limits<double>::infinity();
                            total += std::chrono::duration<double>(
                                         std::chrono::steady_clock::now() - start)
                                         .count();
                        }
                        return total;
                    };
                    double fused = latency([&] {
                        return lt_call(context, plan, &plan->algorithm, alpha, a, b, beta, c, bias);
                    });
                    plan->prefer_unfused = latency(unfused) < fused;
                }
            }
        }
        if (begin)
            cudaEventDestroy(begin);
        if (end)
            cudaEventDestroy(end);
        plan->tuned = true;
    }
    if (plan->prefer_unfused && context->algorithm == CAMBLAS_CUDA_AUTO)
        return unfused();
    context->counts[4]++;
    return lt_call(context, plan, &plan->algorithm, alpha, a, b, beta, c, bias);
}

template <typename T>
int gemm(camblas_cuda_context *context, char trans_a, char trans_b, int m, int n, int k, T alpha,
         const T *a, int lda, const T *b, int ldb, T beta, T *c, int ldc)
{
    if (!context)
        return fail(nullptr, "missing context", 1);
    bool na = trans_a == 'N' || trans_a == 'n', ta = trans_a == 'T' || trans_a == 't';
    bool nb = trans_b == 'N' || trans_b == 'n', tb = trans_b == 'T' || trans_b == 't';
    if ((!na && !ta) || (!nb && !tb) || m < 0 || n < 0 || k < 0 || lda < 1 || ldb < 1 || ldc < 1)
        return fail(context, "invalid GEMM arguments", 1);
    if (m == 0 || n == 0)
        return 0;
    if (!c || lda < std::max(1, na ? m : k) || ldb < std::max(1, nb ? k : n) || ldc < m)
        return fail(context, "invalid device pointer or leading dimension", 1);
    if (alpha == T(0) || k == 0) {
        if (beta == T(1))
            return 0;
        scale_output<<<unsigned((size_t(m) * n + 255) / 256), 256, 0, context->stream>>>(m, n, beta,
                                                                                         c, ldc);
        cudaError_t error = cudaGetLastError();
        return error == cudaSuccess ? 0 : fail(context, "scale output", error);
    }
    if (!a || !b || a == c || b == c)
        return fail(context, "missing or aliased GEMM operand", 1);
    if constexpr (std::is_same_v<T, float>) {
        if (context->algorithm == CAMBLAS_CUDA_AUTO && n == 1 && ta && nb && m >= 1024 &&
            k >= 4096 && k % 4 == 0 && lda % 4 == 0 && reinterpret_cast<uintptr_t>(a) % 16 == 0 &&
            reinterpret_cast<uintptr_t>(b) % 16 == 0)
            return native_float_gemv(context, m, k, alpha, a, lda, b, beta, c);
    }
    cublasOperation_t opa = na ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t opb = nb ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasStatus_t status;
    if (context->algorithm == CAMBLAS_CUDA_SYMMETRIC && a == b && opa != opb && m == n &&
        lda == ldb && beta == T(0)) {
        context->counts[2]++;
        if constexpr (std::is_same_v<T, float>)
            status = cublasSsyrk(context->blas, CUBLAS_FILL_MODE_LOWER, opa, m, k, &alpha, a, lda,
                                 &beta, c, ldc);
        else
            status = cublasDsyrk(context->blas, CUBLAS_FILL_MODE_LOWER, opa, m, k, &alpha, a, lda,
                                 &beta, c, ldc);
        if (status != CUBLAS_STATUS_SUCCESS)
            return fail(context, "symmetric Gram", status);
        mirror_triangle<<<unsigned((size_t(n) * n + 255) / 256), 256, 0, context->stream>>>(n, c,
                                                                                            ldc);
        cudaError_t error = cudaGetLastError();
        return error == cudaSuccess ? 0 : fail(context, "Gram mirroring", error);
    }
    // Attention retains its scores in this arena across its second GEMM. Packing
    // into that same arena would overwrite a live operand or invalidate it on growth.
    auto uses_scratch = [context](const void *pointer) {
        uintptr_t address = reinterpret_cast<uintptr_t>(pointer);
        uintptr_t base = reinterpret_cast<uintptr_t>(context->scratch);
        return address >= base && address - base < context->scratch_bytes;
    };
    bool eligible = na && nb && m == n && n == k && (n % 2 == 0) && n >= 256 && !uses_scratch(a) &&
                    !uses_scratch(b) && !uses_scratch(c);
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(context->stream, &capture);
    if (eligible && std::isfinite(double(alpha)) && capture == cudaStreamCaptureStatusNone &&
        (context->algorithm == CAMBLAS_CUDA_STRASSEN ||
         context->algorithm == CAMBLAS_CUDA_STRASSEN_TWO ||
         context->algorithm == CAMBLAS_CUDA_STRASSEN_THREE ||
         context->algorithm == CAMBLAS_CUDA_STRASSEN_FOUR ||
         (context->algorithm == CAMBLAS_CUDA_AUTO &&
          n >= (std::is_same_v<T, float> ? 4096 : 8192))))
        return strassen(context, n, alpha, a, lda, b, ldb, beta, c, ldc);
    bool prefer_dgemm = std::is_same_v<T, double> && na && nb && m == n && n == k && n >= 2048;
    if ((context->algorithm == CAMBLAS_CUDA_AUTO && !prefer_dgemm &&
         capture == cudaStreamCaptureStatusNone) ||
        context->algorithm == CAMBLAS_CUDA_LT) {
        status = lt_gemm(context, opa, opb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        if (status == CUBLAS_STATUS_SUCCESS)
            return 0;
        if (status != CUBLAS_STATUS_NOT_SUPPORTED)
            return fail(context, "cuBLASLt GEMM", status);
    }
    status = classical(context, opa, opb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    return status == CUBLAS_STATUS_SUCCESS ? 0 : fail(context, "classical GEMM", status);
}

#include "bfloat16.cuh"

template <typename T>
int affine(camblas_cuda_context *context, int rows, int inner, int columns, const T *x,
           const T *weight, const T *bias, bool relu, T *output)
{
    if (!context || rows < 0 || inner < 0 || columns < 0)
        return fail(context, "invalid affine dimensions", 1);
    if (rows == 0 || columns == 0)
        return 0;
    if (!output || !bias || (inner > 0 && (!x || !weight)))
        return fail(context, "missing affine operand", 1);
    context->counts[3]++;
    bool fused_bias = false;
    int result = 0;
    if (inner > 0 &&
        (context->algorithm == CAMBLAS_CUDA_AUTO || context->algorithm == CAMBLAS_CUDA_LT)) {
        cublasStatus_t status =
            lt_gemm(context, CUBLAS_OP_N, CUBLAS_OP_N, columns, rows, inner, T(1), weight, columns,
                    x, inner, T(0), output, columns, bias, relu);
        if (status == CUBLAS_STATUS_SUCCESS)
            fused_bias = true;
        else if (status != CUBLAS_STATUS_NOT_SUPPORTED)
            return fail(context, "fused affine", status);
    }
    if (!fused_bias)
        result = gemm(context, 'N', 'N', columns, rows, inner, T(1), weight, std::max(1, columns),
                      x, std::max(1, inner), T(0), output, columns);
    if (result != 0)
        return result;
    if (!fused_bias) {
        bias_activation<<<unsigned((size_t(rows) * columns + 255) / 256), 256, 0,
                          context->stream>>>(output, bias, size_t(rows) * columns, columns, relu);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
            return fail(context, "bias/activation", error);
    }
    return 0;
}
#include "fusion.cuh"
#include "inference_fusion.cuh"
} // namespace

namespace
{
class DeviceScope
{
    int previous = -1;
    bool changed = false;

  public:
    cudaError_t status;
    explicit DeviceScope(int device) : status(cudaGetDevice(&previous))
    {
        if (status == cudaSuccess && previous != device) {
            status = cudaSetDevice(device);
            changed = status == cudaSuccess;
        }
    }
    ~DeviceScope()
    {
        if (changed)
            cudaSetDevice(previous);
    }
};

template <typename Function> int execute(camblas_cuda_context *context, Function function)
{
    if (!context)
        return fail(nullptr, "missing context", 1);
    DeviceScope device(context->device);
    if (device.status != cudaSuccess)
        return fail(context, "activate device", device.status);
    try {
        return function();
    } catch (const std::bad_alloc &) {
        return fail(context, "host allocation", 2);
    } catch (const std::exception &error) {
        std::snprintf(context->error, sizeof(context->error), "host exception: %s", error.what());
        return -1;
    } catch (...) {
        return fail(context, "unexpected host exception", 1);
    }
}
} // namespace

extern "C" int camblas_cuda_create(int device, void *stream, camblas_cuda_context **out)
{
    if (!out)
        return fail(nullptr, "missing context output", 1);
    *out = nullptr;
    DeviceScope scope(device);
    if (scope.status != cudaSuccess)
        return fail(nullptr, "activate device", scope.status);
    auto *context = new (std::nothrow) camblas_cuda_context;
    if (!context)
        return fail(nullptr, "context allocation", 2);
    context->device = device;
    context->stream = static_cast<cudaStream_t>(stream);
    int status = int(cublasCreate(&context->blas));
    const char *operation = "cuBLAS create";
    if (status == 0) {
        operation = "cuBLAS stream";
        status = int(cublasSetStream(context->blas, context->stream));
    }
    if (status == 0) {
        operation = "cuBLAS pointer mode";
        status = int(cublasSetPointerMode(context->blas, CUBLAS_POINTER_MODE_HOST));
    }
    if (status == 0) {
        operation = "cuBLAS math mode";
        status = int(cublasSetMathMode(
            context->blas,
            cublasMath_t(CUBLAS_DEFAULT_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
    }
    if (status == 0) {
        operation = "cuBLASLt create";
        status = int(cublasLtCreate(&context->lt));
    }
    if (status == 0) {
        operation = "LT workspace";
        status = int(cudaMalloc(&context->lt_workspace, context->lt_workspace_bytes));
    }
    if (status == 0) {
        operation = "cuBLAS workspace";
        status = int(
            cublasSetWorkspace(context->blas, context->lt_workspace, context->lt_workspace_bytes));
    }
    if (status == 0) {
        operation = "range flag allocation";
        status = int(cudaMalloc(reinterpret_cast<void **>(&context->guard), sizeof(unsigned)));
    }
    if (status == 0) {
        operation = "host range flag allocation";
        status =
            int(cudaMallocHost(reinterpret_cast<void **>(&context->host_guard), sizeof(unsigned)));
    }
    if (status != 0) {
        int result = fail(nullptr, operation, status);
        camblas_cuda_destroy(context);
        return result;
    }
    context->short_attention_supported = initialise_short_attention(device);
    context->quantized_supported = initialise_quantized();
    context->fp8_decode_supported = initialise_fp8_decode(device);
    *out = context;
    return 0;
}

extern "C" int camblas_cuda_destroy(camblas_cuda_context *context)
{
    if (!context)
        return 0;
    DeviceScope scope(context->device);
    if (scope.status != cudaSuccess)
        return fail(context, "activate device for destruction", scope.status);
    cudaError_t error = cudaStreamSynchronize(context->stream);
    context->plans.clear();
    if (context->blas)
        cublasDestroy(context->blas);
    if (context->lt)
        cublasLtDestroy(context->lt);
    if (context->scratch)
        cudaFree(context->scratch);
    for (void *allocation : context->retired_allocations)
        cudaFree(allocation);
    if (context->lt_workspace)
        cudaFree(context->lt_workspace);
    if (context->guard)
        cudaFree(context->guard);
    if (context->host_guard)
        cudaFreeHost(context->host_guard);
    delete context;
    return error == cudaSuccess ? 0 : fail(nullptr, "stream completion during destruction", error);
}

extern "C" const char *camblas_cuda_error(const camblas_cuda_context *context)
{
    return context ? context->error : creation_error;
}

extern "C" int camblas_cuda_set_algorithm(camblas_cuda_context *context, int algorithm)
{
    if (!context || algorithm < CAMBLAS_CUDA_AUTO || algorithm > CAMBLAS_CUDA_STRASSEN_FOUR)
        return fail(context, "invalid algorithm", 1);
    context->algorithm = algorithm;
    return 0;
}

extern "C" int camblas_cuda_sgemm(camblas_cuda_context *context, char ta, char tb, int m, int n,
                                  int k, float alpha, const float *a, int lda, const float *b,
                                  int ldb, float beta, float *c, int ldc)
{
    return execute(context, [&] {
        return gemm(context, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    });
}

extern "C" int camblas_cuda_dgemm(camblas_cuda_context *context, char ta, char tb, int m, int n,
                                  int k, double alpha, const double *a, int lda, const double *b,
                                  int ldb, double beta, double *c, int ldc)
{
    return execute(context, [&] {
        return gemm(context, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    });
}

extern "C" int camblas_cuda_bgemm(camblas_cuda_context *context, char ta, char tb, int m, int n,
                                  int k, float alpha, const void *a, int lda, const void *b,
                                  int ldb, float beta, void *c, int ldc)
{
    return execute(context, [&] {
        return bfloat16_gemm(context, ta, tb, m, n, k, alpha, static_cast<const __nv_bfloat16 *>(a),
                             lda, static_cast<const __nv_bfloat16 *>(b), ldb, beta,
                             static_cast<__nv_bfloat16 *>(c), ldc);
    });
}

extern "C" int camblas_cuda_affine(camblas_cuda_context *context, int dtype, int rows, int inner,
                                   int columns, const void *x, const void *weight, const void *bias,
                                   int relu, void *output)
{
    return execute(context, [&] {
        if (dtype == 0)
            return affine(context, rows, inner, columns, static_cast<const float *>(x),
                          static_cast<const float *>(weight), static_cast<const float *>(bias),
                          relu != 0, static_cast<float *>(output));
        if (dtype == 1)
            return affine(context, rows, inner, columns, static_cast<const double *>(x),
                          static_cast<const double *>(weight), static_cast<const double *>(bias),
                          relu != 0, static_cast<double *>(output));
        return fail(context, "unsupported affine dtype", 1);
    });
}

extern "C" int camblas_cuda_mlp(camblas_cuda_context *context, int dtype, int rows, int inputs,
                                int hidden, int outputs, const void *x, const void *w1,
                                const void *b1, const void *w2, const void *b2, void *hidden_output,
                                void *output)
{
    if (rows < 0 || inputs < 0 || hidden < 0 || outputs < 0 ||
        (rows > 0 && hidden > 0 && hidden_output == output))
        return fail(context, "invalid MLP dimensions or aliased outputs", 1);
    int result =
        camblas_cuda_affine(context, dtype, rows, inputs, hidden, x, w1, b1, 1, hidden_output);
    if (result != 0)
        return result;
    return camblas_cuda_affine(context, dtype, rows, hidden, outputs, hidden_output, w2, b2, 0,
                               output);
}

extern "C" int camblas_cuda_silu_multiply(camblas_cuda_context *context, int dtype, uint64_t count,
                                          const void *gate, const void *up, void *output)
{
    return execute(context, [&] {
        if (dtype != 0 && dtype != 2)
            return fail(context, "unsupported SiLU dtype", 1);
        if (!count)
            return 0;
        if (!gate || !up || !output || count > SIZE_MAX / (dtype == 0 ? 4 : 2))
            return fail(context, "invalid SiLU operands or length", 1);
        if (dtype == 0)
            return launch_silu_multiply<float>(context, count, gate, up, output);
        return launch_silu_multiply<__nv_bfloat16>(context, count, gate, up, output);
    });
}

extern "C" int camblas_cuda_swiglu(camblas_cuda_context *context, int dtype, uint64_t count,
                                   int width, float limit, const void *gate, const void *up,
                                   const float *routing, void *output)
{
    return execute(context, [&] {
        if ((dtype != 0 && dtype != 2) || width <= 0 || count % unsigned(width) ||
            !std::isfinite(limit) || limit < 0 || count > SIZE_MAX / (dtype == 0 ? 4 : 2))
            return fail(context, "invalid SwiGLU dimensions or clipping limit", 1);
        if (!count)
            return 0;
        unsigned alignment = dtype == 0 ? 4 : 2;
        if (!gate || !up || !output || uintptr_t(gate) % alignment || uintptr_t(up) % alignment ||
            uintptr_t(output) % alignment ||
            (routing && (uintptr_t(routing) % 4 || output == routing)))
            return fail(context, "invalid SwiGLU buffers", 1);
        unsigned blocks = unsigned(std::min<uint64_t>((count + 255) / 256, 65535));
        if (dtype == 0)
            swiglu_kernel<float><<<blocks, 256, 0, context->stream>>>(
                count, width, limit, static_cast<const float *>(gate),
                static_cast<const float *>(up), routing, static_cast<float *>(output));
        else
            swiglu_kernel<__nv_bfloat16><<<blocks, 256, 0, context->stream>>>(
                count, width, limit, static_cast<const __nv_bfloat16 *>(gate),
                static_cast<const __nv_bfloat16 *>(up), routing,
                static_cast<__nv_bfloat16 *>(output));
        cudaError_t error = cudaGetLastError();
        return error == cudaSuccess ? 0 : fail(context, "SwiGLU", error);
    });
}

extern "C" int camblas_cuda_quantized_matmul(camblas_cuda_context *context, int packed, int rows,
                                             int outputs, int inner, int activation_block,
                                             const void *input, const void *input_scale,
                                             const void *weight, const void *weight_scale,
                                             void *output)
{
    return execute(context, [&] {
        if ((packed != 0 && packed != 1) || rows < 0 || rows > 65535 || outputs < 8 ||
            outputs % 8 || inner < 32 || inner % 32 ||
            (activation_block != 32 && activation_block != 128) || inner % activation_block ||
            (!packed && activation_block != 32))
            return fail(context, "invalid quantized matrix dimensions", 1);
        if (!rows)
            return 0;
        if (!context->quantized_supported)
            return fail(context, "quantized matrix multiplication requires an SM90+ binary", 1);
        if (!input || !input_scale || !weight || !weight_scale || !output ||
            (uintptr_t(input) & 3) || (uintptr_t(weight) & (packed ? 1 : 3)) ||
            (uintptr_t(output) & 1) || output == input || output == input_scale ||
            output == weight || output == weight_scale)
            return fail(context, "invalid quantized matrix buffers", 1);
        return packed ? launch_quantized<true>(context, rows, outputs, inner, activation_block,
                                               input, input_scale, weight, weight_scale, output)
                      : launch_quantized<false>(context, rows, outputs, inner, activation_block,
                                                input, input_scale, weight, weight_scale, output);
    });
}

extern "C" int camblas_cuda_fp8_decode_supported(const camblas_cuda_context *context)
{
    return context && context->fp8_decode_supported;
}

extern "C" int camblas_cuda_fp8_decode(camblas_cuda_context *context, int outputs, int inner,
                                       const void *input, const float *input_scale,
                                       const void *weight, const float *weight_scale,
                                       int input_scale_stride, int weight_scale_row_stride,
                                       int weight_scale_column_stride, void *workspace,
                                       size_t workspace_bytes, void *output)
{
    return execute(context, [&] {
        if (outputs < 16 || outputs % 16 || inner < 32 || inner > 8192 || inner % 32 ||
            input_scale_stride < 1 || weight_scale_row_stride < 1 || weight_scale_column_stride < 1)
            return fail(context, "invalid FP8 decode dimensions or scale strides", 1);
        if (!context->fp8_decode_supported)
            return fail(context, "FP8 decode requires an SM90a binary on an SM90 device", 1);
        if (!input || !input_scale || !weight || !weight_scale || !output ||
            (uintptr_t(input) & 3) || (uintptr_t(weight) & 3) || (uintptr_t(input_scale) & 3) ||
            (uintptr_t(weight_scale) & 3) || (uintptr_t(output) & 1) || output == input ||
            output == weight || output == input_scale || output == weight_scale)
            return fail(context, "invalid FP8 decode buffers", 1);
        size_t required = fp8_workspace_size(0, outputs, inner);
        if (required && (!workspace || workspace_bytes < required || (uintptr_t(workspace) & 3) ||
                         workspace == input || workspace == input_scale || workspace == weight ||
                         workspace == weight_scale || workspace == output))
            return fail(context, "invalid FP8 decode workspace", 1);
        return launch_fp8_product<false>(context, outputs, inner, input, input_scale, weight,
                                         weight_scale, input_scale_stride, weight_scale_row_stride,
                                         weight_scale_column_stride, workspace, output);
    });
}
extern "C" size_t camblas_cuda_fp8_workspace_size(int quantise, int outputs, int inner)
{
    return fp8_workspace_size(quantise, outputs, inner);
}

extern "C" int camblas_cuda_fp8_linear(camblas_cuda_context *context, int outputs, int inner,
                                       const void *input, const void *weight,
                                       const float *weight_scale, int weight_scale_row_stride,
                                       int weight_scale_column_stride, void *workspace,
                                       size_t workspace_bytes, void *output)
{
    return execute(context, [&] {
        if (outputs < 16 || outputs % 16 || inner < 32 || inner > 8192 || inner % 32 ||
            weight_scale_row_stride < 1 || weight_scale_column_stride < 1)
            return fail(context, "invalid FP8 decode dimensions or scale strides", 1);
        if (!context->fp8_decode_supported)
            return fail(context, "FP8 decode requires an SM90a binary on an SM90 device", 1);
        if (!input || !weight || !weight_scale || !output || (uintptr_t(input) & 3) ||
            (uintptr_t(weight) & 3) || (uintptr_t(weight_scale) & 3) || (uintptr_t(output) & 1) ||
            output == input || output == weight || output == weight_scale)
            return fail(context, "invalid FP8 decode buffers", 1);
        size_t required = fp8_workspace_size(1, outputs, inner);
        if (required && (!workspace || workspace_bytes < required || (uintptr_t(workspace) & 3) ||
                         workspace == input || workspace == weight || workspace == weight_scale ||
                         workspace == output))
            return fail(context, "invalid FP8 linear workspace", 1);
        return launch_fp8_product<true>(context, outputs, inner, input, nullptr, weight,
                                        weight_scale, 1, weight_scale_row_stride,
                                        weight_scale_column_stride, workspace, output);
    });
}

extern "C" int camblas_cuda_grouped_quantized_matmul(camblas_cuda_context *context, int packed,
                                                     int groups, int weight_groups, int rows,
                                                     int outputs, int inner, int activation_block,
                                                     const void *input, const void *input_scale,
                                                     const void *metadata, const void *active,
                                                     const void *counts, void *output)
{
    return execute(context, [&] {
        if ((packed != 0 && packed != 1) || groups < 0 || groups > 65535 || weight_groups < 1 ||
            weight_groups > 65535 || rows < 0 || rows > 65535 || outputs < 8 || outputs % 8 ||
            inner < 32 || inner % 32 ||
            (activation_block != 32 && !(packed && activation_block == 128)) ||
            inner % activation_block ||
            uint64_t(groups) * rows > SIZE_MAX / (uint64_t(outputs) * 2))
            return fail(context, "invalid grouped matrix dimensions", 1);
        if (!groups || !rows)
            return 0;
        if (!context->quantized_supported)
            return fail(context, "grouped multiplication requires an SM90+ binary", 1);
        if (!input || !input_scale || !metadata || !active || !counts || !output ||
            (uintptr_t(input) & 3) || (uintptr_t(metadata) & 7) || (uintptr_t(active) & 3) ||
            (uintptr_t(counts) & 3) || (uintptr_t(output) & 1) || output == input ||
            output == input_scale || output == metadata || output == active || output == counts)
            return fail(context, "invalid grouped matrix buffers", 1);
        cudaError_t error =
            cudaMemsetAsync(output, 0, size_t(groups) * rows * outputs * 2, context->stream);
        if (error != cudaSuccess)
            return fail(context, "grouped output initialisation", error);
        dim3 grid((unsigned(outputs) + 15) / 16, (unsigned(rows) + 15) / 16, groups);
        if (packed)
            quantized_tile<true, true><<<grid, 64, 0, context->stream>>>(
                rows, outputs, inner, activation_block, static_cast<const uint8_t *>(input),
                static_cast<const uint8_t *>(input_scale), nullptr, nullptr,
                static_cast<__nv_bfloat16 *>(output), static_cast<const uint64_t *>(metadata),
                static_cast<const int32_t *>(active), static_cast<const int32_t *>(counts),
                weight_groups);
        else
            quantized_tile<false, true><<<grid, 64, 0, context->stream>>>(
                rows, outputs, inner, activation_block, static_cast<const uint8_t *>(input),
                static_cast<const uint8_t *>(input_scale), nullptr, nullptr,
                static_cast<__nv_bfloat16 *>(output), static_cast<const uint64_t *>(metadata),
                static_cast<const int32_t *>(active), static_cast<const int32_t *>(counts),
                weight_groups);
        error = cudaGetLastError();
        return error == cudaSuccess ? 0 : fail(context, "grouped multiplication", error);
    });
}

extern "C" int camblas_cuda_route_groups(camblas_cuda_context *context, int tokens, int choices,
                                         int first_expert, int groups, int rows,
                                         const void *experts, const void *active, void *counts,
                                         void *slots, void *reverse)
{
    return execute(context, [&] {
        if (tokens < 0 || choices < 1 || choices > 64 || first_expert < 0 || groups < 1 ||
            groups > 65535 || rows < 1 || rows > 65535 || int64_t(tokens) * choices > INT_MAX ||
            int64_t(groups) * rows > INT_MAX || !counts || !slots ||
            (tokens && (!experts || !reverse)) || !active || (uintptr_t(experts) & 7) ||
            (uintptr_t(active) & 3) || (uintptr_t(counts) & 3) || (uintptr_t(slots) & 3) ||
            (uintptr_t(reverse) & 3) || counts == active || slots == active || reverse == active ||
            counts == experts || slots == experts || (reverse && reverse == experts) ||
            counts == slots || counts == reverse || slots == reverse)
            return fail(context, "invalid routing dimensions or buffers", 1);
        cudaError_t error = cudaMemsetAsync(counts, 0, size_t(groups) * 4, context->stream);
        if (error == cudaSuccess)
            error = cudaMemsetAsync(slots, 255, size_t(groups) * rows * 4, context->stream);
        if (error == cudaSuccess && tokens)
            error = cudaMemsetAsync(reverse, 255, size_t(tokens) * choices * 4, context->stream);
        if (error != cudaSuccess)
            return fail(context, "routing initialisation", error);
        if (tokens)
            assign_groups_kernel<<<std::min((unsigned(tokens) * choices + 255) / 256, 65535u), 256,
                                   0, context->stream>>>(
                tokens * choices, first_expert, groups, rows, static_cast<const int64_t *>(experts),
                static_cast<const int32_t *>(active), static_cast<int32_t *>(counts),
                static_cast<int32_t *>(slots), static_cast<int32_t *>(reverse));
        error = cudaGetLastError();
        return error == cudaSuccess ? 0 : fail(context, "expert routing", error);
    });
}

extern "C" int camblas_cuda_reduce_groups(camblas_cuda_context *context, int tokens, int choices,
                                          int width, int grouped_rows, const void *experts,
                                          const void *reverse, const void *input, void *output)
{
    return execute(context, [&] {
        if (tokens < 0 || choices < 1 || choices > 64 || width < 1 || grouped_rows < 0 ||
            int64_t(tokens) * choices > INT_MAX || (grouped_rows && !input) ||
            (tokens && (!experts || !reverse || !output)) || (uintptr_t(experts) & 7) ||
            (uintptr_t(reverse) & 3) || (uintptr_t(input) & 1) || (uintptr_t(output) & 3) ||
            (tokens && (output == experts || output == reverse || output == input)))
            return fail(context, "invalid grouped reduction arguments", 1);
        if (!tokens)
            return 0;
        unsigned blocks =
            unsigned(std::min<uint64_t>((uint64_t(tokens) * width + 255) / 256, 65535));
        reduce_groups_kernel<<<blocks, 256, 0, context->stream>>>(
            tokens, choices, width, grouped_rows, static_cast<const int64_t *>(experts),
            static_cast<const int32_t *>(reverse), static_cast<const __nv_bfloat16 *>(input),
            static_cast<float *>(output));
        cudaError_t error = cudaGetLastError();
        return error == cudaSuccess ? 0 : fail(context, "grouped reduction", error);
    });
}

extern "C" int camblas_cuda_rms_norm_decode(camblas_cuda_context *context, int dtype,
                                            int weight_dtype, int width, float epsilon,
                                            int round_before_weight, int descending,
                                            const void *input, const void *weight, void *output)
{
    return execute(context, [&] {
        if ((dtype != 0 && dtype != 2) || (weight_dtype != 0 && weight_dtype != 2) ||
            (dtype == 0 && weight_dtype != 0) || width < 2048 || width > 65536 || width % 4 ||
            (round_before_weight != 0 && round_before_weight != 1) ||
            (descending != 0 && descending != 1) || !std::isfinite(epsilon) || epsilon < 0 ||
            !input || !weight || !output || output == input || output == weight ||
            (uintptr_t(input) % (dtype == 0 ? 4 : 2)) ||
            (uintptr_t(weight) % (weight_dtype == 0 ? 4 : 2)) ||
            (uintptr_t(output) % (dtype == 0 ? 4 : 2)))
            return fail(context, "invalid decode RMS arguments", 1);
        if (dtype == 0)
            rms_decode_kernel<float><<<1, 512, 0, context->stream>>>(
                width, epsilon, static_cast<const float *>(input),
                static_cast<const float *>(weight), static_cast<float *>(output), nullptr, nullptr,
                round_before_weight, descending);
        else if (weight_dtype == 0)
            rms_decode_kernel<__nv_bfloat16, false, float><<<1, 512, 0, context->stream>>>(
                width, epsilon, static_cast<const __nv_bfloat16 *>(input),
                static_cast<const float *>(weight), static_cast<__nv_bfloat16 *>(output), nullptr,
                nullptr, round_before_weight, descending);
        else
            rms_decode_kernel<__nv_bfloat16><<<1, 512, 0, context->stream>>>(
                width, epsilon, static_cast<const __nv_bfloat16 *>(input),
                static_cast<const __nv_bfloat16 *>(weight), static_cast<__nv_bfloat16 *>(output),
                nullptr, nullptr, round_before_weight, descending);
        cudaError_t error = cudaGetLastError();
        return error == cudaSuccess ? 0 : fail(context, "decode RMS normalisation", error);
    });
}

extern "C" int camblas_cuda_add_rms_norm(camblas_cuda_context *context, int dtype, int width,
                                         float epsilon, const void *input, const void *residual,
                                         const void *weight, void *added, void *output)
{
    return execute(context, [&] {
        if ((dtype != 0 && dtype != 2) || width < 2048 || width > 65536 || width % 4 ||
            !std::isfinite(epsilon) || epsilon < 0 || !input || !residual || !weight || !added ||
            !output || added == output || added == input || added == residual || added == weight ||
            output == input || output == residual || output == weight)
            return fail(context, "invalid residual RMS arguments", 1);
        if (dtype == 0)
            rms_decode_kernel<float, true><<<1, 512, 0, context->stream>>>(
                width, epsilon, static_cast<const float *>(input),
                static_cast<const float *>(weight), static_cast<float *>(output),
                static_cast<const float *>(residual), static_cast<float *>(added));
        else
            rms_decode_kernel<__nv_bfloat16, true><<<1, 512, 0, context->stream>>>(
                width, epsilon, static_cast<const __nv_bfloat16 *>(input),
                static_cast<const __nv_bfloat16 *>(weight), static_cast<__nv_bfloat16 *>(output),
                static_cast<const __nv_bfloat16 *>(residual), static_cast<__nv_bfloat16 *>(added));
        cudaError_t error = cudaGetLastError();
        return error == cudaSuccess ? 0 : fail(context, "residual RMS normalisation", error);
    });
}

extern "C" int camblas_cuda_rms_norm(camblas_cuda_context *context, int dtype, int rows, int width,
                                     float epsilon, const void *input, const void *weight,
                                     void *output)
{
    return execute(context, [&] {
        if ((dtype != 0 && dtype != 2) || rows < 0 || width < 0 || !std::isfinite(epsilon) ||
            epsilon < 0)
            return fail(context, "invalid RMS normalisation dimensions, dtype or epsilon", 1);
        if (!rows || !width)
            return 0;
        if (!input || !weight || !output || output == weight)
            return fail(context, "invalid RMS normalisation operands", 1);
        if (dtype == 0)
            return launch_rms_norm<float>(context, rows, width, epsilon, input, weight, output);
        return launch_rms_norm<__nv_bfloat16>(context, rows, width, epsilon, input, weight, output);
    });
}

extern "C" int camblas_cuda_bfloat16_square(camblas_cuda_context *context, uint64_t count,
                                            const void *input, float *output)
{
    return execute(context, [&] {
        if (!count)
            return 0;
        if (!input || !output || count > SIZE_MAX / sizeof(float))
            return fail(context, "invalid BF16 square operands", 1);
        unsigned blocks = unsigned(std::min<uint64_t>((count + 255) / 256, 65535));
        bfloat16_square_kernel<<<blocks, 256, 0, context->stream>>>(
            count, static_cast<const __nv_bfloat16 *>(input), output);
        cudaError_t error = cudaGetLastError();
        return error == cudaSuccess ? 0 : fail(context, "BF16 squares", error);
    });
}

extern "C" int camblas_cuda_rms_scale(camblas_cuda_context *context, int dtype, uint64_t count,
                                      int width, float epsilon, const void *input,
                                      const void *weight, const float *means, void *output)
{
    return execute(context, [&] {
        if ((dtype != 0 && dtype != 2) || width < 0 || !std::isfinite(epsilon) || epsilon < 0)
            return fail(context, "invalid RMS scale dimensions, dtype or epsilon", 1);
        if (!count)
            return 0;
        if (!width || count % width || count > SIZE_MAX / sizeof(float) || !input || !weight ||
            !means || !output || output == weight)
            return fail(context, "invalid RMS scale operands", 1);
        if (dtype == 0)
            return launch_rms_scale<float>(context, count, width, epsilon, input, weight, means,
                                           output);
        return launch_rms_scale<__nv_bfloat16>(context, count, width, epsilon, input, weight, means,
                                               output);
    });
}

extern "C" int camblas_cuda_stats(const camblas_cuda_context *context, uint64_t *counts, int length)
{
    if (!context || !counts || (length != 6 && length != 8))
        return fail(nullptr, "invalid stats output", 1);
    for (int i = 0; i < length; ++i)
        counts[i] = context->counts[i].load(std::memory_order_relaxed);
    return 0;
}

extern "C" int camblas_cuda_reset_stats(camblas_cuda_context *context)
{
    if (!context)
        return fail(nullptr, "missing context", 1);
    for (auto &count : context->counts)
        count.store(0, std::memory_order_relaxed);
    return 0;
}

extern "C" int camblas_cuda_attention(camblas_cuda_context *context, int dtype, int queries,
                                      int keys, int depth, int values, double scale, const void *q,
                                      const void *k, const void *v, void *output)
{
    return execute(context, [&] {
        if (dtype == 0)
            return attention(context, queries, keys, depth, values, float(scale),
                             static_cast<const float *>(q), static_cast<const float *>(k),
                             static_cast<const float *>(v), static_cast<float *>(output));
        if (dtype == 1)
            return attention(context, queries, keys, depth, values, scale,
                             static_cast<const double *>(q), static_cast<const double *>(k),
                             static_cast<const double *>(v), static_cast<double *>(output));
        return fail(context, "unsupported attention dtype", 1);
    });
}

extern "C" int camblas_cuda_mlp_backward(camblas_cuda_context *context, int dtype, int rows,
                                         int inputs, int hidden, int outputs, const void *x,
                                         const void *w1, const void *w2, const void *h,
                                         const void *grad, void *dh, void *dx, void *dw1, void *db1,
                                         void *dw2, void *db2)
{
    return execute(context, [&] {
        if (dtype == 0)
            return mlp_backward(
                context, rows, inputs, hidden, outputs, static_cast<const float *>(x),
                static_cast<const float *>(w1), static_cast<const float *>(w2),
                static_cast<const float *>(h), static_cast<const float *>(grad),
                static_cast<float *>(dh), static_cast<float *>(dx), static_cast<float *>(dw1),
                static_cast<float *>(db1), static_cast<float *>(dw2), static_cast<float *>(db2));
        if (dtype == 1)
            return mlp_backward(
                context, rows, inputs, hidden, outputs, static_cast<const double *>(x),
                static_cast<const double *>(w1), static_cast<const double *>(w2),
                static_cast<const double *>(h), static_cast<const double *>(grad),
                static_cast<double *>(dh), static_cast<double *>(dx), static_cast<double *>(dw1),
                static_cast<double *>(db1), static_cast<double *>(dw2), static_cast<double *>(db2));
        return fail(context, "unsupported backward dtype", 1);
    });
}
