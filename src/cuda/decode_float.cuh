/* Full FP32 single-vector projection; no TF32 or cached operand values. */
__global__ void float_gemv_kernel(int rows, int inner, float alpha, const float *weight, int ld,
                                  const float *input, float beta, float *output)
{
    constexpr int Warps = 4, Rows = 2;
    unsigned warp = threadIdx.x / 32, lane = threadIdx.x % 32, part = warp % Warps;
    size_t row = size_t(blockIdx.x) * Rows + warp / Warps;
    float sum = 0;
    if (row < size_t(rows)) {
        const float *values = weight + row * ld;
        for (size_t column = (part * 32 + lane) * 4; column < size_t(inner);
             column += Warps * 32 * 4) {
            float4 w = __ldg(reinterpret_cast<const float4 *>(values + column));
            float4 x = __ldg(reinterpret_cast<const float4 *>(input + column));
            sum = fmaf(w.x, x.x, sum);
            sum = fmaf(w.y, x.y, sum);
            sum = fmaf(w.z, x.z, sum);
            sum = fmaf(w.w, x.w, sum);
        }
    }
    for (int offset = 16; offset; offset /= 2)
        sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, offset));
    {
        __shared__ float partial[8];
        if (lane == 0)
            partial[warp] = sum;
        __syncthreads();
        if (part == 0 && lane == 0) {
            sum = partial[warp];
#pragma unroll
            for (int index = 1; index < Warps; ++index)
                sum = __fadd_rn(sum, partial[warp + index]);
        }
    }
    if (row < size_t(rows) && part == 0 && lane == 0) {
        float value = alpha * sum;
        if (beta != 0)
            value = fmaf(beta, output[row], value);
        output[row] = value;
    }
}

int native_float_gemv(camblas_cuda_context *context, int rows, int inner, float alpha,
                      const float *weight, int ld, const float *input, float beta, float *output)
{
    context->counts[0]++;
    constexpr int Rows = 2;
    float_gemv_kernel<<<unsigned((size_t(rows) + Rows - 1) / Rows), 256, 0, context->stream>>>(
        rows, inner, alpha, weight, ld, input, beta, output);
    cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? 0 : fail(context, "FP32 native GEMV", error);
}
