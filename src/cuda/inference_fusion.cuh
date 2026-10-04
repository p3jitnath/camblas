/* Inference fusion with the same intermediate storage rounding as PyTorch. */
template <typename T> __device__ float storage_float(T value)
{
    return float(value);
}
template <> __device__ float storage_float(__nv_bfloat16 value)
{
    return __bfloat162float(value);
}
template <typename T> __device__ T storage_round(float value)
{
    return T(value);
}
template <> __device__ __nv_bfloat16 storage_round(float value)
{
    return __float2bfloat16_rn(value);
}

template <typename T>
__global__ void silu_multiply_kernel(size_t count, const T *gate, const T *up, T *output)
{
    for (size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += size_t(gridDim.x) * blockDim.x) {
        float value = storage_float(gate[index]);
        T activated = storage_round<T>(value / (1.f + expf(-value)));
        output[index] = storage_round<T>(storage_float(activated) * storage_float(up[index]));
    }
}

template <typename T>
__global__ void swiglu_kernel(size_t count, int width, float limit, const T *gate, const T *up,
                              const float *routing, T *output)
{
    for (size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += size_t(gridDim.x) * blockDim.x) {
        float g = storage_float(gate[index]), u = storage_float(up[index]);
        if (limit > 0) {
            // Comparisons preserve NaNs, matching clamp rather than fmin/fmax.
            if (g > limit)
                g = limit;
            if (u > limit)
                u = limit;
            if (u < -limit)
                u = -limit;
        }
        float value = __fmul_rn(g / (1.f + expf(-g)), u);
        if (routing)
            value = __fmul_rn(routing[index / width], value);
        output[index] = storage_round<T>(value);
    }
}

template <typename T>
__global__ void rms_norm_kernel(int rows, int width, float epsilon, const T *input, const T *weight,
                                T *output)
{
    __shared__ float partial[8];
    for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const T *source = input + size_t(row) * width;
        float sum = 0;
        for (int64_t column = threadIdx.x; column < width; column += blockDim.x) {
            float value = storage_float(source[column]);
            sum = __fadd_rn(sum, __fmul_rn(value, value));
        }
        for (int offset = 16; offset; offset /= 2)
            sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, offset));
        if (threadIdx.x % 32 == 0)
            partial[threadIdx.x / 32] = sum;
        __syncthreads();
        if (threadIdx.x < 32) {
            sum = threadIdx.x < 8 ? partial[threadIdx.x] : 0;
            for (int offset = 16; offset; offset /= 2)
                sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, offset));
            if (threadIdx.x == 0)
                partial[0] = rsqrtf(sum / float(width) + epsilon);
        }
        __syncthreads();
        float scale = partial[0];
        for (int64_t column = threadIdx.x; column < width; column += blockDim.x) {
            T normalised = storage_round<T>(__fmul_rn(storage_float(source[column]), scale));
            output[size_t(row) * width + column] = storage_round<T>(
                __fmul_rn(storage_float(normalised), storage_float(weight[column])));
        }
        __syncthreads();
    }
}

template <typename T>
int launch_silu_multiply(camblas_cuda_context *context, size_t count, const void *gate,
                         const void *up, void *output)
{
    unsigned blocks = unsigned(std::min<size_t>((count + 255) / 256, 65535));
    silu_multiply_kernel<T><<<blocks, 256, 0, context->stream>>>(
        count, static_cast<const T *>(gate), static_cast<const T *>(up), static_cast<T *>(output));
    cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? 0 : fail(context, "SiLU multiply", error);
}

__global__ void bfloat16_square_kernel(size_t count, const __nv_bfloat16 *input, float *output)
{
    for (size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += size_t(gridDim.x) * blockDim.x) {
        float value = __bfloat162float(input[index]);
        output[index] = __fmul_rn(value, value);
    }
}

template <typename T>
__global__ void rms_scale_kernel(size_t count, int width, float epsilon, const T *input,
                                 const T *weight, const float *means, T *output)
{
    for (size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += size_t(gridDim.x) * blockDim.x) {
        float scale = rsqrtf(__fadd_rn(means[index / width], epsilon));
        T normalised = storage_round<T>(__fmul_rn(storage_float(input[index]), scale));
        output[index] = storage_round<T>(
            __fmul_rn(storage_float(normalised), storage_float(weight[index % width])));
    }
}

template <typename T>
int launch_rms_scale(camblas_cuda_context *context, size_t count, int width, float epsilon,
                     const void *input, const void *weight, const float *means, void *output)
{
    unsigned blocks = unsigned(std::min<size_t>((count + 255) / 256, 65535));
    rms_scale_kernel<T><<<blocks, 256, 0, context->stream>>>(
        count, width, epsilon, static_cast<const T *>(input), static_cast<const T *>(weight), means,
        static_cast<T *>(output));
    cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? 0 : fail(context, "RMS scaling", error);
}

// A single-row mean in PyTorch 2.8 uses four independent accumulators per
// thread, 512 threads, a halving shared-memory tree, then ascending shuffle
// offsets. Retain that arithmetic order while fusing square and scale.
template <typename T, bool Add = false, typename W = T>
__global__ void rms_decode_kernel(int width, float epsilon, const T *input, const W *weight,
                                  T *output, const T *residual = nullptr, T *added = nullptr,
                                  bool round_before_weight = true, bool descending = false)
{
    __shared__ float partial[512];
    __shared__ float scale;
    float sums[4] = {0, 0, 0, 0};
    for (int64_t column = int64_t(threadIdx.x) * 4; column < width; column += 512 * 4) {
#pragma unroll
        for (int offset = 0; offset < 4; ++offset) {
            float value = storage_float(input[column + offset]);
            if constexpr (Add)
                value = storage_float(
                    storage_round<T>(__fadd_rn(value, storage_float(residual[column + offset]))));
            sums[offset] = __fadd_rn(sums[offset], __fmul_rn(value, value));
        }
    }
    float sum = sums[0];
#pragma unroll
    for (int offset = 1; offset < 4; ++offset)
        sum = __fadd_rn(sum, sums[offset]);
    partial[threadIdx.x] = sum;
    for (int offset = 256; offset >= 32; offset /= 2) {
        __syncthreads();
        if (threadIdx.x < unsigned(offset)) {
            sum = __fadd_rn(sum, partial[threadIdx.x + offset]);
            partial[threadIdx.x] = sum;
        }
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        if (descending) {
            for (int offset = 16; offset; offset /= 2)
                sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, offset));
        } else {
            for (int offset = 1; offset < 32; offset *= 2)
                sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, offset));
        }
        if (threadIdx.x == 0)
            scale = rsqrtf(__fadd_rn(__fmul_rn(sum, 1.f / float(width)), epsilon));
    }
    __syncthreads();
    for (int64_t column = threadIdx.x; column < width; column += 512) {
        T value = input[column];
        if constexpr (Add) {
            value =
                storage_round<T>(__fadd_rn(storage_float(value), storage_float(residual[column])));
            added[column] = value;
        }
        float normalised = __fmul_rn(storage_float(value), scale);
        if (round_before_weight)
            normalised = storage_float(storage_round<T>(normalised));
        output[column] = storage_round<T>(__fmul_rn(normalised, storage_float(weight[column])));
    }
}

template <typename T>
int launch_rms_norm(camblas_cuda_context *context, int rows, int width, float epsilon,
                    const void *input, const void *weight, void *output)
{
    if (rows == 1 && width >= 2048 && width <= 65536 && width % 4 == 0)
        rms_decode_kernel<T><<<1, 512, 0, context->stream>>>(
            width, epsilon, static_cast<const T *>(input), static_cast<const T *>(weight),
            static_cast<T *>(output));
    else
        rms_norm_kernel<T><<<std::min(rows, 65535), 256, 0, context->stream>>>(
            rows, width, epsilon, static_cast<const T *>(input), static_cast<const T *>(weight),
            static_cast<T *>(output));
    cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? 0 : fail(context, "RMS normalisation", error);
}
