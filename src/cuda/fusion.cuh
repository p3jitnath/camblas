/* Included inside backend.cu's private namespace, after the GEMM implementation. */

template <typename T, bool Maximum> __device__ T warp_reduce(T value)
{
    for (int offset = 16; offset > 0; offset /= 2) {
        T other = __shfl_down_sync(0xffffffffu, value, offset);
        if constexpr (Maximum)
            value = fmax(value, other);
        else
            value += other;
    }
    return value;
}

template <typename T, bool Maximum> __device__ T block_reduce(T value, T *shared)
{
    int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    value = warp_reduce<T, Maximum>(value);
    if (lane == 0)
        shared[warp] = value;
    __syncthreads();
    if (warp == 0) {
        T neutral = Maximum ? -T(CUDART_INF_F) : T(0);
        value = lane < int(blockDim.x / 32) ? shared[lane] : neutral;
        value = warp_reduce<T, Maximum>(value);
        if (lane == 0)
            shared[0] = value;
    }
    __syncthreads();
    T result = shared[0];
    __syncthreads();
    return result;
}

template <typename T> __device__ T exponential(T value)
{
    if constexpr (std::is_same_v<T, float>)
        return expf(value);
    else
        return exp(value);
}

template <typename T> __global__ void softmax_rows(T *scores, int columns, T scale)
{
    __shared__ T shared[8];
    T *row = scores + size_t(blockIdx.x) * columns;
    T maximum = -T(CUDART_INF_F);
    for (int64_t column = threadIdx.x; column < columns; column += blockDim.x)
        maximum = fmax(maximum, row[column] * scale);
    maximum = block_reduce<T, true>(maximum, shared);
    T sum = 0;
    for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
        T value = exponential(row[column] * scale - maximum);
        row[column] = value;
        sum += value;
    }
    sum = block_reduce<T, false>(sum, shared);
    for (int64_t column = threadIdx.x; column < columns; column += blockDim.x)
        row[column] /= sum;
}

template <typename T>
int attention(camblas_cuda_context *context, int queries, int keys, int depth, int values, T scale,
              const T *q, const T *k, const T *v, T *output)
{
    if (queries < 0 || keys < 0 || depth < 0 || values < 0)
        return fail(context, "invalid attention dimensions", 1);
    if (queries == 0 || values == 0)
        return 0;
    if (!output || (queries > 0 && depth > 0 && !q) || (keys > 0 && depth > 0 && !k) ||
        (keys > 0 && !v))
        return fail(context, "missing attention operand", 1);
    if (keys > 0 && size_t(queries) > std::numeric_limits<size_t>::max() / sizeof(T) / size_t(keys))
        return fail(context, "attention scratch size overflow", 2);
    context->counts[6]++;
    int status = reserve_scratch(context, size_t(queries) * keys * sizeof(T));
    if (status)
        return status;
    T *scores = static_cast<T *>(context->scratch);
    status = gemm(context, 'T', 'N', keys, queries, depth, T(1), k, std::max(1, depth), q,
                  std::max(1, depth), T(0), scores, std::max(1, keys));
    if (status)
        return status;
    if (keys > 0) {
        softmax_rows<<<queries, 256, 0, context->stream>>>(scores, keys, scale);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
            return fail(context, "attention softmax", error);
    }
    return gemm(context, 'N', 'N', values, queries, keys, T(1), v, std::max(1, values), scores,
                std::max(1, keys), T(0), output, std::max(1, values));
}

template <typename T, bool Mask>
__global__ void backward_bias(T *grad, const T *activation, int rows, int columns, T *bias,
                              int chunk_rows)
{
    __shared__ T partial[8][32];
    int64_t column = int64_t(blockIdx.x) * 32 + int(threadIdx.x);
    int64_t begin = Mask ? int64_t(blockIdx.y) * chunk_rows : 0;
    int64_t end = begin + chunk_rows < rows ? begin + chunk_rows : rows;
    if constexpr (!Mask)
        end = rows;
    T sum = 0;
    if (column < columns) {
        for (int64_t row = begin + threadIdx.y; row < end; row += 8) {
            size_t offset = size_t(row) * columns + column;
            T value = grad[offset];
            if constexpr (Mask) {
                if (activation[offset] <= T(0))
                    value = T(0);
                grad[offset] = value;
            }
            sum += value;
        }
    }
    partial[threadIdx.y][threadIdx.x] = sum;
    __syncthreads();
    if (threadIdx.y == 0 && column < columns && bias) {
        T value = 0;
        for (int row = 0; row < 8; ++row)
            value += partial[row][threadIdx.x];
        bias[(Mask ? size_t(blockIdx.y) * columns : 0) + column] = value;
    }
}

template <typename T>
__global__ void finish_bias(const T *partial, int chunks, int columns, T *bias)
{
    int64_t column = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (column >= columns)
        return;
    T value = 0;
    for (int chunk = 0; chunk < chunks; ++chunk)
        value += partial[size_t(chunk) * columns + column];
    bias[column] = value;
}

template <typename T>
int mlp_backward(camblas_cuda_context *context, int rows, int inputs, int hidden, int outputs,
                 const T *x, const T *w1, const T *w2, const T *h, const T *grad, T *dh, T *dx,
                 T *dw1, T *db1, T *dw2, T *db2)
{
    if (rows < 0 || inputs < 0 || hidden < 0 || outputs < 0)
        return fail(context, "invalid backward dimensions", 1);
    if ((rows > 0 && outputs > 0 && !grad) ||
        (rows > 0 && hidden > 0 && (dx || dw1 || db1) && (!dh || !h)))
        return fail(context, "missing backward operand", 1);
    context->counts[7]++;
    int chunks = int(std::min<size_t>(65535, std::max<size_t>(1, (size_t(rows) + 127) / 128)));
    int chunk_rows = std::max(1, int((int64_t(rows) + chunks - 1) / chunks));
    T *partial = db1;
    if (db1 && chunks > 1 && hidden > 0) {
        int allocation = reserve_scratch(context, size_t(chunks) * hidden * sizeof(T));
        if (allocation)
            return allocation;
        partial = static_cast<T *>(context->scratch);
    }
    int result = 0;
    if (dw2) {
        result = gemm(context, 'N', 'T', outputs, hidden, rows, T(1), grad, std::max(1, outputs), h,
                      std::max(1, hidden), T(0), dw2, std::max(1, outputs));
        if (result)
            return result;
    }
    if (db2 && outputs > 0)
        backward_bias<T, false>
            <<<unsigned((size_t(outputs) + 31) / 32), dim3(32, 8), 0, context->stream>>>(
                const_cast<T *>(grad), nullptr, rows, outputs, db2, rows);
    if (dx || dw1 || db1) {
        result = gemm(context, 'T', 'N', hidden, rows, outputs, T(1), w2, std::max(1, outputs),
                      grad, std::max(1, outputs), T(0), dh, std::max(1, hidden));
        if (result)
            return result;
        if (hidden > 0)
            backward_bias<T, true>
                <<<dim3(unsigned((size_t(hidden) + 31) / 32), chunks), dim3(32, 8), 0,
                   context->stream>>>(dh, h, rows, hidden, partial, chunk_rows);
        if (db1 && chunks > 1 && hidden > 0)
            finish_bias<<<unsigned((size_t(hidden) + 255) / 256), 256, 0, context->stream>>>(
                partial, chunks, hidden, db1);
        if (dw1) {
            result = gemm(context, 'N', 'T', hidden, inputs, rows, T(1), dh, std::max(1, hidden), x,
                          std::max(1, inputs), T(0), dw1, std::max(1, hidden));
            if (result)
                return result;
        }
        if (dx) {
            result = gemm(context, 'T', 'N', inputs, rows, hidden, T(1), w1, std::max(1, hidden),
                          dh, std::max(1, hidden), T(0), dx, std::max(1, inputs));
            if (result)
                return result;
        }
    }
    cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? 0 : fail(context, "backward bias/activation", error);
}
