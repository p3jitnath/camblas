/* CAMBLAS CUDA backend. Device packing and fusion are independently implemented. */
#include "camblas_cuda.h"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <atomic>
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
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulAlgo_t algorithm{};
    bool available = false;
    bool tuned = false;
    bool prefer_classical = false;
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
    std::vector<void *> captured_scratch;
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
                context->captured_scratch.push_back(context->scratch);
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

template <typename T>
int strassen(camblas_cuda_context *context, int n, T alpha, const T *a, int lda, const T *b,
             int ldb, T beta, T *c, int ldc)
{
    int h = n / 2;
    size_t length = size_t(h) * h;
    bool two = n % 4 == 0 && (context->algorithm == CAMBLAS_CUDA_STRASSEN_TWO ||
                              (context->algorithm == CAMBLAS_CUDA_AUTO && n >= 8192));
    if (length > std::numeric_limits<size_t>::max() / (64 * sizeof(T)))
        return fail(context, "Strassen scratch size overflow", 2);
    int size = two ? h / 2 : h, batches = two ? 49 : 7;
    size_t stride = size_t(size) * size;
    size_t elements = 3 * batches * stride;
    int allocation = reserve_scratch(context, elements * sizeof(T));
    if (allocation == -int(cudaErrorMemoryAllocation)) {
        context->counts[5]++;
        cublasStatus_t status = classical(context, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, alpha, a, lda,
                                          b, ldb, beta, c, ldc);
        return status == CUBLAS_STATUS_SUCCESS ? 0 : fail(context, "low-memory GEMM", status);
    }
    if (allocation)
        return allocation;
    T *pa = static_cast<T *>(context->scratch), *pb = pa + batches * stride,
      *products = pb + batches * stride;
    T limit = T(std::sqrt(std::numeric_limits<T>::max() /
                          ((two ? 256.0 : 64.0) * n * std::max(1.0, std::fabs(double(alpha))))));
    cudaError_t error = cudaMemsetAsync(context->guard, 0, sizeof(unsigned), context->stream);
    if (error != cudaSuccess)
        return fail(context, "range-flag reset", error);
    unsigned blocks = unsigned((stride + 255) / 256);
    if (two) {
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
    if (two)
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
                 const T *b_pointer, const T *c_pointer)
{
    int type = std::is_same_v<T, float> ? 0 : 1;
    unsigned aa = pointer_alignment(a_pointer), ab = pointer_alignment(b_pointer);
    unsigned ac = pointer_alignment(c_pointer), bias_alignment = pointer_alignment(bias);
    PlanKey key(type, int(ta), int(tb), m, n, k, lda, ldb, ldc, bias ? 1 : 0, int(aa), int(ab),
                int(ac), int(bias_alignment));
    auto existing = context->plans.find(key);
    if (existing != context->plans.end())
        return existing->second.get();
    auto owned = std::make_unique<LtPlan>();
    LtPlan *plan = owned.get();
    context->plans.emplace(key, std::move(owned));
    cudaDataType_t datatype = type == 0 ? CUDA_R_32F : CUDA_R_64F;
    cublasComputeType_t compute = type == 0 ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_64F;
    if (cublasLtMatmulDescCreate(&plan->operation, compute, datatype) != CUBLAS_STATUS_SUCCESS)
        return plan;
    if (cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_TRANSA, &ta,
                                       sizeof(ta)) != CUBLAS_STATUS_SUCCESS)
        return plan;
    if (cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_TRANSB, &tb,
                                       sizeof(tb)) != CUBLAS_STATUS_SUCCESS)
        return plan;
    if (bias) {
        auto epilogue = CUBLASLT_EPILOGUE_BIAS;
        if (cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                           &epilogue, sizeof(epilogue)) != CUBLAS_STATUS_SUCCESS)
            return plan;
        if (cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
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
                                         &ac, sizeof(ac));
    cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,
                                         &ac, sizeof(ac));
    cublasLtMatmulHeuristicResult_t candidates[12];
    int count = 0;
    cublasStatus_t status =
        cublasLtMatmulAlgoGetHeuristic(context->lt, plan->operation, plan->a, plan->b, plan->c,
                                       plan->c, preference, 12, candidates, &count);
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

template <typename T>
cublasStatus_t lt_call(camblas_cuda_context *context, LtPlan *plan,
                       const cublasLtMatmulAlgo_t *algorithm, T alpha, const T *a, const T *b,
                       T beta, T *c)
{
    return cublasLtMatmul(context->lt, plan->operation, &alpha, a, plan->a, b, plan->b, &beta, c,
                          plan->c, c, plan->c, algorithm, context->lt_workspace,
                          context->lt_workspace_bytes, context->stream);
}

template <typename T>
cublasStatus_t lt_gemm(camblas_cuda_context *context, cublasOperation_t ta, cublasOperation_t tb,
                       int m, int n, int k, T alpha, const T *a, int lda, const T *b, int ldb,
                       T beta, T *c, int ldc, const T *bias = nullptr)
{
    LtPlan *plan = get_plan(context, ta, tb, m, n, k, lda, ldb, ldc, bias, a, b, c);
    if (!plan->available)
        return CUBLAS_STATUS_NOT_SUPPORTED;
    if (bias) {
        cublasStatus_t status = cublasLtMatmulDescSetAttribute(
            plan->operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
        if (status != CUBLAS_STATUS_SUCCESS)
            return status;
    }
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(context->stream, &capture);
    // Tuning only overwrites beta-zero output; no input or caller-owned C is
    // changed for beta != 0. All tuning and allocations belong to warm-up.
    if (!plan->tuned && beta == T(0) && capture == cudaStreamCaptureStatusNone) {
        cudaEvent_t begin = nullptr, end = nullptr;
        if (cudaEventCreate(&begin) == cudaSuccess && cudaEventCreate(&end) == cudaSuccess) {
            float fastest = std::numeric_limits<float>::infinity();
            for (const auto &candidate : plan->candidates) {
                cublasStatus_t status =
                    lt_call(context, plan, &candidate.algo, alpha, a, b, beta, c);
                if (status != CUBLAS_STATUS_SUCCESS)
                    continue;
                cudaEventRecord(begin, context->stream);
                for (int repeat = 0; repeat < 3; ++repeat)
                    status = lt_call(context, plan, &candidate.algo, alpha, a, b, beta, c);
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
            if (!bias) {
                cublasStatus_t status =
                    classical(context, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
                cudaEventRecord(begin, context->stream);
                for (int repeat = 0; repeat < 3; ++repeat)
                    status =
                        classical(context, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
                cudaEventRecord(end, context->stream);
                cudaError_t error = cudaEventSynchronize(end);
                float milliseconds = 0;
                cudaEventElapsedTime(&milliseconds, begin, end);
                plan->prefer_classical = error == cudaSuccess && status == CUBLAS_STATUS_SUCCESS &&
                                         milliseconds < fastest;
            }
        }
        if (begin)
            cudaEventDestroy(begin);
        if (end)
            cudaEventDestroy(end);
        plan->tuned = true;
    }
    if (plan->prefer_classical && context->algorithm == CAMBLAS_CUDA_AUTO)
        return classical(context, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    context->counts[4]++;
    return lt_call(context, plan, &plan->algorithm, alpha, a, b, beta, c);
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
    if constexpr (std::is_same_v<T, float>) {
        if (inner > 0 &&
            (context->algorithm == CAMBLAS_CUDA_AUTO || context->algorithm == CAMBLAS_CUDA_LT)) {
            cublasStatus_t status =
                lt_gemm(context, CUBLAS_OP_N, CUBLAS_OP_N, columns, rows, inner, T(1), weight,
                        columns, x, inner, T(0), output, columns, bias);
            if (status == CUBLAS_STATUS_SUCCESS)
                fused_bias = true;
            else if (status != CUBLAS_STATUS_NOT_SUPPORTED)
                return fail(context, "fused affine", status);
        }
    }
    if (!fused_bias)
        result = gemm(context, 'N', 'N', columns, rows, inner, T(1), weight, std::max(1, columns),
                      x, std::max(1, inner), T(0), output, columns);
    if (result != 0)
        return result;
    if (!fused_bias || relu) {
        bias_activation<<<unsigned((size_t(rows) * columns + 255) / 256), 256, 0,
                          context->stream>>>(output, fused_bias ? nullptr : bias,
                                             size_t(rows) * columns, columns, relu);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
            return fail(context, "bias/activation", error);
    }
    return 0;
}
#include "fusion.cuh"
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
        status = int(cublasSetMathMode(context->blas, CUBLAS_DEFAULT_MATH));
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
    for (void *allocation : context->captured_scratch)
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
    if (!context || algorithm < CAMBLAS_CUDA_AUTO || algorithm > CAMBLAS_CUDA_STRASSEN_TWO)
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
