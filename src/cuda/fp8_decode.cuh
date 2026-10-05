/* Single-token block32 FP8 products. Weight tiles remain in registers; the
 * shared input is reused across independent WGMMA products. Each 32-element
 * contribution is scaled and accumulated in its original FP32 order.
 * Fragment/descriptor layout follows NVIDIA's PTX ISA, section 9.7.17.5. */
__device__ __forceinline__ void
fp8_decode_products(const uint32_t a[8][4], const uint64_t *descriptors, float *low, float *high)
{
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    [[maybe_unused]] float c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15,
        c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31;
    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p, 0, 0; wgmma.fence.sync.aligned; "
        "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0,%1,%2,%3}, {%32,%33,%34,%35}, %64, p, 1, 1; "
        "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%4,%5,%6,%7}, {%36,%37,%38,%39}, %65, p, 1, 1; "
        "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%8,%9,%10,%11}, {%40,%41,%42,%43}, %66, p, 1, 1; "
        "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%12,%13,%14,%15}, {%44,%45,%46,%47}, %67, p, 1, 1; "
        "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%16,%17,%18,%19}, {%48,%49,%50,%51}, %68, p, 1, 1; "
        "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%20,%21,%22,%23}, {%52,%53,%54,%55}, %69, p, 1, 1; "
        "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%24,%25,%26,%27}, {%56,%57,%58,%59}, %70, p, 1, 1; "
        "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%28,%29,%30,%31}, {%60,%61,%62,%63}, %71, p, 1, 1; "
        "wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0; }"
        : "=&f"(c0), "=&f"(c1), "=&f"(c2), "=&f"(c3), "=&f"(c4), "=&f"(c5), "=&f"(c6), "=&f"(c7),
          "=&f"(c8), "=&f"(c9), "=&f"(c10), "=&f"(c11), "=&f"(c12), "=&f"(c13), "=&f"(c14),
          "=&f"(c15), "=&f"(c16), "=&f"(c17), "=&f"(c18), "=&f"(c19), "=&f"(c20), "=&f"(c21),
          "=&f"(c22), "=&f"(c23), "=&f"(c24), "=&f"(c25), "=&f"(c26), "=&f"(c27), "=&f"(c28),
          "=&f"(c29), "=&f"(c30), "=&f"(c31)
        : "r"(a[0][0]), "r"(a[0][1]), "r"(a[0][2]), "r"(a[0][3]), "r"(a[1][0]), "r"(a[1][1]),
          "r"(a[1][2]), "r"(a[1][3]), "r"(a[2][0]), "r"(a[2][1]), "r"(a[2][2]), "r"(a[2][3]),
          "r"(a[3][0]), "r"(a[3][1]), "r"(a[3][2]), "r"(a[3][3]), "r"(a[4][0]), "r"(a[4][1]),
          "r"(a[4][2]), "r"(a[4][3]), "r"(a[5][0]), "r"(a[5][1]), "r"(a[5][2]), "r"(a[5][3]),
          "r"(a[6][0]), "r"(a[6][1]), "r"(a[6][2]), "r"(a[6][3]), "r"(a[7][0]), "r"(a[7][1]),
          "r"(a[7][2]), "r"(a[7][3]), "l"(descriptors[0]), "l"(descriptors[1]), "l"(descriptors[2]),
          "l"(descriptors[3]), "l"(descriptors[4]), "l"(descriptors[5]), "l"(descriptors[6]),
          "l"(descriptors[7])
        : "memory");
    low[0] = c0;
    high[0] = c2;
    low[1] = c4;
    high[1] = c6;
    low[2] = c8;
    high[2] = c10;
    low[3] = c12;
    high[3] = c14;
    low[4] = c16;
    high[4] = c18;
    low[5] = c20;
    high[5] = c22;
    low[6] = c24;
    high[6] = c26;
    low[7] = c28;
    high[7] = c30;

#else
    asm volatile("trap;");
#endif
}

/* Round the block32 activation scale up to an exact power of two, using
 * the same FP32 reciprocal and finite E4M3 conversion as the reference. */
__device__ __forceinline__ uint8_t fp8_quantise_bf16(const __nv_bfloat16 *x, int index,
                                                     float &scale)
{
    float value = __bfloat162float(x[index]);
    unsigned magnitude = __float_as_uint(fmaxf(fabsf(value), 1.e-10f));
    float maximum = __uint_as_float(__reduce_max_sync(0xffffffffu, magnitude));
    unsigned bits = __float_as_uint(__fmul_rn(maximum, 1.f / 448.f));
    int exponent = int((bits >> 23) & 255) - 127 + int((bits & 0x7fffff) != 0);
    scale = __uint_as_float(unsigned(exponent + 127) << 23);
    float multiplier = __uint_as_float(unsigned(127 - exponent) << 23);
    return __nv_cvt_float_to_fp8(fminf(__fmul_rn(value, multiplier), 448.f), __NV_SATFINITE,
                                 __NV_E4M3);
}

template <bool Quantise>
__global__ void fp8_decode_kernel(int n, int k, const uint8_t *x, const float *sx, const uint8_t *w,
                                  const float *sw, __nv_bfloat16 *out, int sx_stride, int sw_row,
                                  int sw_col)
{
    constexpr int Columns = 8, Unroll = 8;
    extern __shared__ __align__(16) uint8_t shared[];
    if constexpr (Quantise) {
        float *scales = reinterpret_cast<float *>(shared + Columns * k);
        unsigned lane = threadIdx.x & 31;
        for (int g = threadIdx.x / 32; g < k / 32; g += blockDim.x / 32) {
            float scale;
            uint8_t byte =
                fp8_quantise_bf16(reinterpret_cast<const __nv_bfloat16 *>(x), g * 32 + lane, scale);
#pragma unroll
            for (int column = 0; column < Columns; ++column) {
                int offset = g * 256 + (lane / 16) * 128 + column * 16 + (lane % 16);
                shared[offset] = byte;
            }
            if (lane == 0)
                scales[g] = scale;
        }
        sx = scales;
        sx_stride = 1;
    } else {
        // Columns=8 and k%32=0 bound i/32 below k/16, without a modulo.
        for (int i = threadIdx.x; i < Columns * k / 4; i += blockDim.x) {
            int feature = (i / 32) * 16 + (i & 3) * 4;
            reinterpret_cast<uint32_t *>(shared)[i] =
                *reinterpret_cast<const uint32_t *>(x + feature);
        }
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    uint64_t address = uint64_t(__cvta_generic_to_shared(shared));
    uint64_t descriptor_base = (address >> 4) | (uint64_t(8) << 16) | (uint64_t(k / 2) << 32);
    const unsigned lane = threadIdx.x & 31;
    const unsigned group = lane >> 2, part = lane & 3;
    const unsigned row =
        (blockIdx.x * (blockDim.x / 128) + threadIdx.x / 128) * 64 + (threadIdx.x % 128) / 32 * 16;
    float sum0 = 0.f, sum1 = 0.f;
    const int blocks = k / 32;
    for (int base = 0; base < blocks; base += Unroll) {
        uint32_t a[Unroll][4];
        float as[Unroll], bs[Unroll];
#pragma unroll
        for (int u = 0; u < Unroll; ++u) {
            int g = base + u;
            if (g >= blocks) {
                for (int i = 0; i < 4; ++i)
                    a[u][i] = 0;
                as[u] = bs[u] = 0.f;
                continue;
            }
            unsigned offset = g * 32 + part * 4;
            const uint8_t *low = w + size_t(row + group) * k + offset;
            const uint8_t *high = low + size_t(8) * k;
            a[u][0] = row + group < unsigned(n) ? *reinterpret_cast<const uint32_t *>(low) : 0;
            a[u][1] = row + group + 8 < unsigned(n) ? *reinterpret_cast<const uint32_t *>(high) : 0;
            a[u][2] = row + group < unsigned(n) ? *reinterpret_cast<const uint32_t *>(low + 16) : 0;
            a[u][3] =
                row + group + 8 < unsigned(n) ? *reinterpret_cast<const uint32_t *>(high + 16) : 0;
            if (part == 0) {
                as[u] = sx[size_t(g) * sx_stride];
                bs[u] =
                    row < unsigned(n) ? sw[size_t(row / 32) * sw_row + size_t(g) * sw_col] : 0.f;
            }
        }

        uint64_t descriptors[Unroll];
        float low[Unroll], high[Unroll];
#pragma unroll
        for (int u = 0; u < Unroll; ++u)
            descriptors[u] = descriptor_base + uint64_t(min(base + u, blocks - 1)) * 16;
        fp8_decode_products(a, descriptors, low, high);
#pragma unroll
        for (int u = 0; u < Unroll; ++u) {
            if (base + u >= blocks)
                break;
            if (part == 0) {
                sum0 = __fmaf_rn(__fmul_rn(low[u], bs[u]), as[u], sum0);
                sum1 = __fmaf_rn(__fmul_rn(high[u], bs[u]), as[u], sum1);
            }
        }
    }
    if (part == 0) {
        if (row + group < unsigned(n))
            out[row + group] = __float2bfloat16_rn(sum0);
        if (row + group + 8 < unsigned(n))
            out[row + group + 8] = __float2bfloat16_rn(sum1);
    }
}

/* Keep independent K32 products in scratch, then apply the original ordered
 * FP32 FMAs. This exposes parallelism for small, deep single-token weights. */
template <bool Quantise>
__global__ void fp8_split_products(int n, int k, const uint8_t *x, const float *sx, int sx_stride,
                                   const uint8_t *w, const float *sw, float *scratch, int sw_row,
                                   int sw_col)
{
    constexpr int chunk = 8;
    __shared__ __align__(16) uint8_t shared[8 * chunk * 32];
    int first = int(blockIdx.y) * chunk;
    int groups = min(chunk, k / 32 - first);
    int tile_k = groups * 32;
    float *scales = scratch + size_t(k / 32) * n;
    unsigned lane = threadIdx.x & 31;
    if constexpr (Quantise) {
        for (int local = int(threadIdx.x) / 32; local < groups; local += 4) {
            float scale;
            uint8_t byte = fp8_quantise_bf16(reinterpret_cast<const __nv_bfloat16 *>(x),
                                             (first + local) * 32 + lane, scale);
#pragma unroll
            for (int column = 0; column < 8; ++column)
                shared[local * 256 + (lane / 16) * 128 + column * 16 + (lane % 16)] = byte;
            if (blockIdx.x == 0 && lane == 0)
                scales[first + local] = scale;
        }
    } else {
        for (int i = threadIdx.x; i < 8 * tile_k / 4; i += blockDim.x) {
            int feature = (i / 32) * 16 + (i & 3) * 4;
            reinterpret_cast<uint32_t *>(shared)[i] =
                *reinterpret_cast<const uint32_t *>(x + first * 32 + feature);
        }
        if (blockIdx.x == 0)
            for (int local = threadIdx.x; local < groups; local += blockDim.x)
                scales[first + local] = sx[size_t(first + local) * sx_stride];
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    uint64_t address = uint64_t(__cvta_generic_to_shared(shared));
    uint64_t descriptor_base = (address >> 4) | (uint64_t(8) << 16) | (uint64_t(tile_k / 2) << 32);
    unsigned group = lane >> 2, part = lane & 3;
    unsigned row = blockIdx.x * 64 + (threadIdx.x / 32) * 16;
    for (int base = 0; base < groups; base += 8) {
        uint32_t a[8][4];
        float bs[8];
        uint64_t descriptors[8];
        float low[8], high[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            int local = base + u;
            descriptors[u] = descriptor_base + uint64_t(min(local, groups - 1)) * 16;
            if (local >= groups) {
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    a[u][i] = 0;
                bs[u] = 0;
                continue;
            }
            int g = first + local;
            const uint8_t *bottom = w + size_t(row + group) * k + g * 32 + part * 4;
            const uint8_t *top = bottom + size_t(8) * k;
            a[u][0] = row + group < unsigned(n) ? *reinterpret_cast<const uint32_t *>(bottom) : 0;
            a[u][1] = row + group + 8 < unsigned(n) ? *reinterpret_cast<const uint32_t *>(top) : 0;
            a[u][2] =
                row + group < unsigned(n) ? *reinterpret_cast<const uint32_t *>(bottom + 16) : 0;
            a[u][3] =
                row + group + 8 < unsigned(n) ? *reinterpret_cast<const uint32_t *>(top + 16) : 0;
            if (part == 0)
                bs[u] = row < unsigned(n) ? sw[size_t(row / 32) * sw_row + size_t(g) * sw_col] : 0;
        }
        fp8_decode_products(a, descriptors, low, high);
        if (part == 0) {
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                if (base + u >= groups)
                    break;
                size_t offset = size_t(first + base + u) * n;
                if (row + group < unsigned(n))
                    scratch[offset + row + group] = __fmul_rn(low[u], bs[u]);
                if (row + group + 8 < unsigned(n))
                    scratch[offset + row + group + 8] = __fmul_rn(high[u], bs[u]);
            }
        }
    }
}
template <int Batch>
__global__ void fp8_ordered_reduction(int n, int groups, const float *scratch, __nv_bfloat16 *out)
{
    int row = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= n)
        return;
    const float *scales = scratch + size_t(groups) * n;
    float sum = 0;
    for (int base = 0; base < groups; base += Batch) {
        float values[Batch], as[Batch];
#pragma unroll
        for (int u = 0; u < Batch; ++u) {
            if (base + u < groups) {
                values[u] = scratch[size_t(base + u) * n + row];
                as[u] = scales[base + u];
            }
        }
#pragma unroll
        for (int u = 0; u < Batch; ++u) {
            if (base + u >= groups)
                break;
            sum = __fmaf_rn(values[u], as[u], sum);
        }
    }
    out[row] = __float2bfloat16_rn(sum);
}

bool initialise_fp8_decode(int device)
{
#ifdef CAMBLAS_CUDA_WGMMA
    cudaDeviceProp properties{};
    return cudaGetDeviceProperties(&properties, device) == cudaSuccess && properties.major == 9 &&
           properties.minor == 0 &&
           cudaFuncSetAttribute(fp8_decode_kernel<false>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                65536) == cudaSuccess &&
           cudaFuncSetAttribute(fp8_decode_kernel<true>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, 66560) == cudaSuccess;
#else
    (void)device;
    return false;
#endif
}

template <bool Quantise>
int launch_fp8_decode(camblas_cuda_context *context, int outputs, int inner, const void *input,
                      const float *input_scale, const void *weight, const float *weight_scale,
                      int input_scale_stride, int weight_scale_row_stride,
                      int weight_scale_column_stride, void *output)
{
    int threads = outputs >= 16384 ? 512 : (inner <= 1024 ? 256 : 128);
    unsigned rows_per_block = unsigned(threads / 128) * 64;
    unsigned blocks = (unsigned(outputs) + rows_per_block - 1) / rows_per_block;
    int shared_bytes = 8 * inner + (Quantise ? int(sizeof(float)) * (inner / 32) : 0);
    fp8_decode_kernel<Quantise><<<blocks, threads, shared_bytes, context->stream>>>(
        outputs, inner, static_cast<const uint8_t *>(input), input_scale,
        static_cast<const uint8_t *>(weight), weight_scale, static_cast<__nv_bfloat16 *>(output),
        input_scale_stride, weight_scale_row_stride, weight_scale_column_stride);
    cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? 0 : fail(context, "FP8 decode", error);
}

size_t fp8_workspace_size(int quantise, int outputs, int inner)
{
    bool split = ((outputs == 1152 || outputs == 1792) && inner == 5120) ||
                 ((outputs == 4096 || outputs == 8192) && inner == 1280) ||
                 (outputs == 5120 && (inner == 2048 || (quantise == 1 && inner == 576)));
    return split ? size_t(inner / 32) * (size_t(outputs) + 1) * sizeof(float) : 0;
}

template <bool Quantise>
int launch_fp8_product(camblas_cuda_context *context, int outputs, int inner, const void *input,
                       const float *input_scale, const void *weight, const float *weight_scale,
                       int input_scale_stride, int weight_scale_row_stride,
                       int weight_scale_column_stride, void *workspace, void *output)
{
    if (!fp8_workspace_size(Quantise, outputs, inner))
        return launch_fp8_decode<Quantise>(
            context, outputs, inner, input, input_scale, weight, weight_scale, input_scale_stride,
            weight_scale_row_stride, weight_scale_column_stride, output);
    fp8_split_products<Quantise>
        <<<dim3((outputs + 63) / 64, (inner + 255) / 256), 128, 0, context->stream>>>(
            outputs, inner, static_cast<const uint8_t *>(input), input_scale, input_scale_stride,
            static_cast<const uint8_t *>(weight), weight_scale, static_cast<float *>(workspace),
            weight_scale_row_stride, weight_scale_column_stride);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        return fail(context, "FP8 split products", error);
    int threads = outputs < 8192 ? 64 : 128;
    unsigned blocks = (unsigned(outputs) + unsigned(threads) - 1) / unsigned(threads);
    if (inner >= 2048)
        fp8_ordered_reduction<32><<<blocks, threads, 0, context->stream>>>(
            outputs, inner / 32, static_cast<const float *>(workspace),
            static_cast<__nv_bfloat16 *>(output));
    else
        fp8_ordered_reduction<16><<<blocks, threads, 0, context->stream>>>(
            outputs, inner / 32, static_cast<const float *>(workspace),
            static_cast<__nv_bfloat16 *>(output));
    error = cudaGetLastError();
    return error == cudaSuccess ? 0 : fail(context, "FP8 ordered reduction", error);
}
