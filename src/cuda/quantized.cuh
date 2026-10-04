/* FP8 activations with FP8 or packed E2M1 weights, block scales and FP32 accumulation.
 * MMA fragment coordinates follow the NVIDIA PTX m16n8k32 specification. */
__device__ float exponent_scale(uint8_t value)
{
    return __uint_as_float(value == 255 ? 0x7fc00000u
                           : value      ? unsigned(value) << 23
                                        : 0x00400000u);
}

__device__ uint32_t fp4_bytes(uint16_t packed)
{
    constexpr uint8_t codes[8] = {0, 0x30, 0x38, 0x3c, 0x40, 0x44, 0x48, 0x4c};
    uint32_t result = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        unsigned nibble = (packed >> (i * 4)) & 15;
        result |= uint32_t(codes[nibble & 7] | ((nibble & 8) << 4)) << (i * 8);
    }
    return result;
}

template <bool Packed>
__global__ void quantized_decode(int outputs, int inner, int activation_block, const uint8_t *input,
                                 const uint8_t *input_scale, const uint8_t *weight,
                                 const uint8_t *weight_scale, __nv_bfloat16 *output)
{
    unsigned lane = threadIdx.x % 32, group = lane / 4, part = lane % 4;
    size_t tile = (size_t(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32) * 8;
    if (tile >= size_t(outputs))
        return; // Uniform for a whole warp.
    int blocks = inner / 32;
    input += size_t(blockIdx.y) * inner;
    input_scale += size_t(blockIdx.y) * (inner / activation_block);
    float total0 = 0, total1 = 0;
    for (int k = 0; k < blocks; ++k) {
        unsigned offset = k * 32 + part * 4;
        uint32_t a0 = group == 0 ? *reinterpret_cast<const uint32_t *>(input + offset) : 0;
        uint32_t a2 = group == 0 ? *reinterpret_cast<const uint32_t *>(input + offset + 16) : 0;
        uint32_t b0, b1;
        if constexpr (Packed) {
            const uint8_t *row = weight + (tile + group) * (inner / 2);
            b0 = fp4_bytes(*reinterpret_cast<const uint16_t *>(row + offset / 2));
            b1 = fp4_bytes(*reinterpret_cast<const uint16_t *>(row + (offset + 16) / 2));
        } else {
            const uint8_t *row = weight + (tile + group) * inner;
            b0 = *reinterpret_cast<const uint32_t *>(row + offset);
            b1 = *reinterpret_cast<const uint32_t *>(row + offset + 16);
        }
        float c0, c1, c2, c3;
#if __CUDA_ARCH__ >= 900
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
                     : "=f"(c0), "=f"(c1), "=f"(c2), "=f"(c3)
                     : "r"(a0), "r"(0), "r"(a2), "r"(0), "r"(b0), "r"(b1), "f"(0.f));
#else
        c0 = c1 = 0;
#endif
        if (group == 0) {
            float a_scale = exponent_scale(input_scale[(k * 32) / activation_block]);
            size_t row0 = tile + part * 2, row1 = row0 + 1;
            float b_scale0 = exponent_scale(weight_scale[(Packed ? row0 : row0 / 32) * blocks + k]);
            float b_scale1 = exponent_scale(weight_scale[(Packed ? row1 : row1 / 32) * blocks + k]);
            total0 = __fadd_rn(total0, __fmul_rn(__fmul_rn(c0, a_scale), b_scale0));
            total1 = __fadd_rn(total1, __fmul_rn(__fmul_rn(c1, a_scale), b_scale1));
        }
    }
    if (group == 0) {
        size_t index = size_t(blockIdx.y) * outputs + tile + part * 2;
        output[index] = __float2bfloat16_rn(total0);
        output[index + 1] = __float2bfloat16_rn(total1);
    }
}

template <bool Packed, bool Grouped = false>
__global__ void quantized_tile(int rows, int outputs, int inner, int activation_block,
                               const uint8_t *input, const uint8_t *input_scale,
                               const uint8_t *weight, const uint8_t *weight_scale,
                               __nv_bfloat16 *output, const uint64_t *metadata = nullptr,
                               const int32_t *active = nullptr, const int32_t *counts = nullptr,
                               int weight_groups = 0)
{
    if constexpr (Grouped) {
        int batch = int(blockIdx.z), selected = active[batch];
        if (selected < 0 || selected >= weight_groups)
            return;
        weight = reinterpret_cast<const uint8_t *>(metadata[size_t(selected) * 2]);
        weight_scale = reinterpret_cast<const uint8_t *>(metadata[size_t(selected) * 2 + 1]);
        input += size_t(batch) * rows * inner;
        input_scale += size_t(batch) * rows * (inner / activation_block);
        output += size_t(batch) * rows * outputs;
        rows = max(0, min(rows, counts[batch]));
        if (blockIdx.y * 16 >= unsigned(rows))
            return;
    }
    unsigned lane = threadIdx.x % 32, group = lane / 4, part = lane % 4;
    size_t tile = (size_t(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32) * 8;
    if (tile >= size_t(outputs))
        return;
    size_t row0 = size_t(blockIdx.y) * 16 + group, row1 = row0 + 8;
    int blocks = inner / 32, scale_blocks = inner / activation_block;
    float total[4] = {0, 0, 0, 0};
    for (int k = 0; k < blocks; ++k) {
        unsigned offset = k * 32 + part * 4;
        uint32_t a[4] = {0, 0, 0, 0}, b0, b1;
        if (row0 < size_t(rows)) {
            a[0] = *reinterpret_cast<const uint32_t *>(input + row0 * inner + offset);
            a[2] = *reinterpret_cast<const uint32_t *>(input + row0 * inner + offset + 16);
        }
        if (row1 < size_t(rows)) {
            a[1] = *reinterpret_cast<const uint32_t *>(input + row1 * inner + offset);
            a[3] = *reinterpret_cast<const uint32_t *>(input + row1 * inner + offset + 16);
        }
        if constexpr (Packed) {
            const uint8_t *w = weight + (tile + group) * (inner / 2);
            b0 = fp4_bytes(*reinterpret_cast<const uint16_t *>(w + offset / 2));
            b1 = fp4_bytes(*reinterpret_cast<const uint16_t *>(w + (offset + 16) / 2));
        } else {
            const uint8_t *w = weight + (tile + group) * inner;
            b0 = *reinterpret_cast<const uint32_t *>(w + offset);
            b1 = *reinterpret_cast<const uint32_t *>(w + offset + 16);
        }
        float c[4];
#if __CUDA_ARCH__ >= 900
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
                     : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(0.f));
#else
        c[0] = c[1] = c[2] = c[3] = 0;
#endif
        float sa0 =
            row0 < size_t(rows)
                ? exponent_scale(input_scale[row0 * scale_blocks + (k * 32) / activation_block])
                : 1.f;
        float sa1 =
            row1 < size_t(rows)
                ? exponent_scale(input_scale[row1 * scale_blocks + (k * 32) / activation_block])
                : 1.f;
        size_t col0 = tile + part * 2, col1 = col0 + 1;
        float sb0 = exponent_scale(weight_scale[(Packed ? col0 : col0 / 32) * blocks + k]);
        float sb1 = exponent_scale(weight_scale[(Packed ? col1 : col1 / 32) * blocks + k]);
        total[0] = __fadd_rn(total[0], __fmul_rn(__fmul_rn(c[0], sa0), sb0));
        total[1] = __fadd_rn(total[1], __fmul_rn(__fmul_rn(c[1], sa0), sb1));
        total[2] = __fadd_rn(total[2], __fmul_rn(__fmul_rn(c[2], sa1), sb0));
        total[3] = __fadd_rn(total[3], __fmul_rn(__fmul_rn(c[3], sa1), sb1));
    }
    size_t column = tile + part * 2;
    if (row0 < size_t(rows)) {
        output[row0 * outputs + column] = __float2bfloat16_rn(total[0]);
        output[row0 * outputs + column + 1] = __float2bfloat16_rn(total[1]);
    }
    if (row1 < size_t(rows)) {
        output[row1 * outputs + column] = __float2bfloat16_rn(total[2]);
        output[row1 * outputs + column + 1] = __float2bfloat16_rn(total[3]);
    }
}

template <bool Packed>
int launch_quantized(camblas_cuda_context *context, int rows, int outputs, int inner,
                     int activation_block, const void *input, const void *input_scale,
                     const void *weight, const void *weight_scale, void *output)
{
    dim3 grid((unsigned(outputs) + 15) / 16,
              rows < 8 ? unsigned(rows) : (unsigned(rows) + 15) / 16);
    if (rows < 8)
        quantized_decode<Packed><<<grid, 64, 0, context->stream>>>(
            outputs, inner, activation_block, static_cast<const uint8_t *>(input),
            static_cast<const uint8_t *>(input_scale), static_cast<const uint8_t *>(weight),
            static_cast<const uint8_t *>(weight_scale), static_cast<__nv_bfloat16 *>(output));
    else
        quantized_tile<Packed><<<grid, 64, 0, context->stream>>>(
            rows, outputs, inner, activation_block, static_cast<const uint8_t *>(input),
            static_cast<const uint8_t *>(input_scale), static_cast<const uint8_t *>(weight),
            static_cast<const uint8_t *>(weight_scale), static_cast<__nv_bfloat16 *>(output));
    cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? 0 : fail(context, "quantized matrix multiplication", error);
}

bool initialise_quantized()
{
    cudaFuncAttributes attributes{};
    return cudaFuncGetAttributes(&attributes, quantized_decode<true>) == cudaSuccess &&
           attributes.binaryVersion >= 90 && attributes.ptxVersion >= 90;
}
