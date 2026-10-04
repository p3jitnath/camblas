/* Stream one outer product through a fused three-level batch. The original
 * operands are reread on every call; only private leaf and result storage is reused. */
template <typename T, bool Right, int Outer>
__global__ void pack_strassen_four_outer(const T *input, int ld, int eighth, T *packed)
{
    size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t length = size_t(eighth) * eighth;
    if (index >= length)
        return;
    int row = int(index % eighth), column = int(index / eighth);
    T original[4][4][4];
#pragma unroll
    for (int low = 0; low < 4; ++low)
#pragma unroll
        for (int middle = 0; middle < 4; ++middle)
#pragma unroll
            for (int outer = 0; outer < 4; ++outer) {
                int r = row + (low / 2 + 2 * (middle / 2) + 4 * (outer / 2)) * eighth;
                int col = column + (low % 2 + 2 * (middle % 2) + 4 * (outer % 2)) * eighth;
                int half = 8 * eighth;
                original[low][middle][outer] = packed_term<Right>(
                    input[r + size_t(col) * ld], input[r + size_t(col + half) * ld],
                    input[r + half + size_t(col) * ld], input[r + half + size_t(col + half) * ld],
                    Outer);
            }
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

template <typename T, int Outer>
__global__ void recombine_strassen_four_outer(const T *products, int eighth, T *c, int ldc)
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
    int parent_size = 8 * eighth;
    constexpr int coefficients[7][4] = {{1, 0, 0, 1},  {0, 0, 1, -1}, {0, 1, 0, 1}, {1, 0, 1, 0},
                                        {-1, 1, 0, 0}, {0, 0, 0, 1},  {1, 0, 0, 0}};
#pragma unroll
    for (int inner = 0; inner < 4; ++inner) {
        int r = row + (inner / 2) * 4 * eighth, column_index = column + (inner % 2) * 4 * eighth;
        T contribution = values[inner];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            int sign = coefficients[Outer][q];
            if (sign) {
                size_t offset =
                    r + (q / 2) * parent_size + size_t(column_index + (q % 2) * parent_size) * ldc;
                c[offset] = sign > 0 ? c[offset] + contribution : c[offset] - contribution;
            }
        }
    }
}

template <typename T>
__global__ void check_strassen_four_range(const T *a, int lda, const T *b, int ldb, int n, T limit,
                                          unsigned *unsafe)
{
    size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= size_t(n) * n)
        return;
    int row = int(index % n), column = int(index / n);
    bool bad = !(fabs(a[row + size_t(column) * lda]) <= limit) ||
               !(fabs(b[row + size_t(column) * ldb]) <= limit);
    if (__any_sync(__activemask(), bad) && (threadIdx.x & 31) == 0)
        atomicOr(unsafe, 1u);
}

template <typename T>
__global__ void finish_strassen_four(const T *product, int n, T alpha, T beta, T *output, int ld)
{
    size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= size_t(n) * n)
        return;
    size_t offset = index % n + (index / n) * ld;
    output[offset] =
        beta == T(0) ? alpha * product[index] : alpha * product[index] + beta * output[offset];
}

template <typename T, int Parent>
int strassen_four_pass(camblas_cuda_context *context, int size, size_t stride, T *pa, T *pb,
                       T *products, T *accumulated, int n, const T *a, int lda, const T *b, int ldb)
{
    unsigned blocks = unsigned((stride + 255) / 256);
    pack_strassen_four_outer<T, false, Parent>
        <<<blocks, 256, 0, context->stream>>>(a, lda, size, pa);
    pack_strassen_four_outer<T, true, Parent>
        <<<blocks, 256, 0, context->stream>>>(b, ldb, size, pb);
    T one = 1, zero = 0;
    cublasStatus_t result;
    if constexpr (std::is_same_v<T, float>)
        result = cublasSgemmStridedBatched(context->blas, CUBLAS_OP_N, CUBLAS_OP_N, size, size,
                                           size, &one, pa, size, stride, pb, size, stride, &zero,
                                           products, size, stride, 343);
    else
        result = cublasDgemmStridedBatched(context->blas, CUBLAS_OP_N, CUBLAS_OP_N, size, size,
                                           size, &one, pa, size, stride, pb, size, stride, &zero,
                                           products, size, stride, 343);
    if (result != CUBLAS_STATUS_SUCCESS)
        return fail(context, "four-level batched GEMM", result);
    recombine_strassen_four_outer<T, Parent>
        <<<blocks * 16, 256, 0, context->stream>>>(products, size, accumulated, n);
    return 0;
}

template <typename T>
int strassen_four_streamed(camblas_cuda_context *context, int n, T alpha, const T *a, int lda,
                           const T *b, int ldb, T beta, T *c, int ldc)
{
    int size = n / 16, batches = 343;
    size_t stride = size_t(size) * size;
    if (stride > std::numeric_limits<size_t>::max() / (3 * size_t(batches) * sizeof(T)))
        return fail(context, "four-level scratch overflow", 2);
    // Beta-zero calls accumulate the unscaled product directly in the output.
    // Otherwise keep C intact until the final alpha*product + beta*C update.
    size_t accumulation = beta == T(0) ? 0 : size_t(n) * n;
    if (accumulation >
        std::numeric_limits<size_t>::max() / sizeof(T) - 3 * size_t(batches) * stride)
        return fail(context, "four-level accumulation overflow", 2);
    int status =
        reserve_scratch(context, (3 * size_t(batches) * stride + accumulation) * sizeof(T));
    if (status == -int(cudaErrorMemoryAllocation)) {
        context->counts[5]++;
        auto result = classical(context, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, alpha, a, lda, b, ldb,
                                beta, c, ldc);
        return result == CUBLAS_STATUS_SUCCESS
                   ? 0
                   : fail(context, "four-level low-memory GEMM", result);
    }
    if (status)
        return status;
    T *pa = static_cast<T *>(context->scratch), *pb = pa + batches * stride,
      *products = pb + batches * stride,
      *accumulated = beta == T(0) ? c : products + batches * stride;
    int accumulation_ld = beta == T(0) ? ldc : n;
    // Four levels amplify packed operands by at most 16. Leaf products and
    // all recombination prefixes are bounded by 8192*n*max(1,abs(alpha))*limit^2.
    // This avoids overflow; it does not guarantee relative accuracy on cancellation.
    T limit = T(std::sqrt(std::numeric_limits<T>::max() /
                          (8192.0 * n * std::max(1.0, std::fabs(double(alpha))))));
    auto error = cudaMemsetAsync(context->guard, 0, sizeof(unsigned), context->stream);
    if (error != cudaSuccess)
        return fail(context, "four-level range reset", error);
    check_strassen_four_range<<<unsigned((size_t(n) * n + 255) / 256), 256, 0, context->stream>>>(
        a, lda, b, ldb, n, limit, context->guard);
    error = cudaMemcpyAsync(context->host_guard, context->guard, sizeof(unsigned),
                            cudaMemcpyDeviceToHost, context->stream);
    if (error == cudaSuccess)
        error = cudaStreamSynchronize(context->stream);
    if (error != cudaSuccess)
        return fail(context, "four-level range check", error);
    if (*context->host_guard) {
        context->counts[5]++;
        auto result = classical(context, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, alpha, a, lda, b, ldb,
                                beta, c, ldc);
        return result == CUBLAS_STATUS_SUCCESS ? 0
                                               : fail(context, "four-level guarded GEMM", result);
    }
    if (beta == T(0)) {
        scale_output<<<unsigned((size_t(n) * n + 255) / 256), 256, 0, context->stream>>>(n, n, T(0),
                                                                                         c, ldc);
    } else {
        error = cudaMemsetAsync(accumulated, 0, accumulation * sizeof(T), context->stream);
        if (error != cudaSuccess)
            return fail(context, "four-level accumulation reset", error);
    }
    for (int parent = 0; parent < 7; ++parent) {
        switch (parent) {
        case 0:
            status = strassen_four_pass<T, 0>(context, size, stride, pa, pb, products, accumulated,
                                              accumulation_ld, a, lda, b, ldb);
            break;
        case 1:
            status = strassen_four_pass<T, 1>(context, size, stride, pa, pb, products, accumulated,
                                              accumulation_ld, a, lda, b, ldb);
            break;
        case 2:
            status = strassen_four_pass<T, 2>(context, size, stride, pa, pb, products, accumulated,
                                              accumulation_ld, a, lda, b, ldb);
            break;
        case 3:
            status = strassen_four_pass<T, 3>(context, size, stride, pa, pb, products, accumulated,
                                              accumulation_ld, a, lda, b, ldb);
            break;
        case 4:
            status = strassen_four_pass<T, 4>(context, size, stride, pa, pb, products, accumulated,
                                              accumulation_ld, a, lda, b, ldb);
            break;
        case 5:
            status = strassen_four_pass<T, 5>(context, size, stride, pa, pb, products, accumulated,
                                              accumulation_ld, a, lda, b, ldb);
            break;
        case 6:
            status = strassen_four_pass<T, 6>(context, size, stride, pa, pb, products, accumulated,
                                              accumulation_ld, a, lda, b, ldb);
            break;
        }
        if (status)
            return status;
    }
    context->counts[1]++;
    if (beta != T(0))
        finish_strassen_four<<<unsigned((accumulation + 255) / 256), 256, 0, context->stream>>>(
            accumulated, n, alpha, beta, c, ldc);
    else if (alpha != T(1))
        scale_output<<<unsigned((size_t(n) * n + 255) / 256), 256, 0, context->stream>>>(
            n, n, alpha, c, ldc);
    error = cudaGetLastError();
    return error == cudaSuccess ? 0 : fail(context, "four-level recombination", error);
}
