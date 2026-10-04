namespace wm = nvcuda::wmma;
template <bool Maximum> __device__ double warp_reduce(double value)
{
    for (int offset = 16; offset > 0; offset /= 2) {
        double other = __shfl_down_sync(0xffffffffu, value, offset);
        if constexpr (Maximum)
            value = fmax(value, other);
        else
            value += other;
    }
    return __shfl_sync(0xffffffffu, value, 0);
}
__global__ void attention_short_fp64(const double *q, const double *k, const double *v, int keys,
                                     double scale, double *output)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    extern __shared__ __align__(32) double storage[];
    constexpr int ld = 260, score_ld = 36;
    double *queries = storage;
    double *tile = queries + 8 * ld;
    double *accumulator = tile + 32 * ld;
    double *scores = accumulator + 8 * ld;
    double *state = scores + 2 * 8 * score_ld;
    int lane = int(threadIdx.x) % 32, warp = int(threadIdx.x) / 32;
    size_t row = size_t(blockIdx.x) * 8;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        queries[r * ld + threadIdx.x] = q[(row + r) * 256 + threadIdx.x];
        accumulator[r * ld + threadIdx.x] = 0;
    }
    double maximum = -CUDART_INF, normaliser = 0;
    for (int base = 0; base < keys; base += 32) {
#pragma unroll
        for (int r = 0; r < 32; ++r)
            tile[r * ld + threadIdx.x] = k[size_t(base + r) * 256 + threadIdx.x];
        __syncthreads();
        wm::fragment<wm::matrix_a, 8, 8, 4, double, wm::row_major> a;
        wm::fragment<wm::matrix_b, 8, 8, 4, double, wm::col_major> b;
        wm::fragment<wm::accumulator, 8, 8, 4, double> c;
        wm::fill_fragment(c, 0.0);
        int split = warp / 4, key = 8 * (warp % 4);
#pragma unroll
        for (int d = 0; d < 128; d += 4) {
            wm::load_matrix_sync(a, queries + split * 128 + d, ld);
            wm::load_matrix_sync(b, tile + key * ld + split * 128 + d, ld);
            wm::mma_sync(c, a, b, c);
        }
        wm::store_matrix_sync(scores + split * 8 * score_ld + key, c, score_ld, wm::mem_row_major);
        __syncthreads();
        double score =
            (scores[warp * score_ld + lane] + scores[(8 + warp) * score_ld + lane]) * scale;
        double next = fmax(maximum, warp_reduce<true>(score));
        double keep = maximum == -CUDART_INF ? 0.0 : exp(maximum - next);
        double weight = next == -CUDART_INF && !isnan(score) ? 0.0 : exp(score - next);
        // Store a normalised running result. Unnormalised weighted sums can
        // overflow for finite V even when its weighted average is representable.
        double next_normaliser = normaliser * keep + warp_reduce<false>(weight);
        double rescale = (normaliser * keep) / next_normaliser;
        scores[warp * score_ld + lane] = weight / next_normaliser;
        normaliser = next_normaliser;
        maximum = next;
        if (lane == 0) {
            state[3 * warp] = rescale;
            state[3 * warp + 1] = normaliser;
            state[3 * warp + 2] = maximum;
        }
        __syncthreads();
#pragma unroll
        for (int r = 0; r < 8; ++r)
            accumulator[r * ld + threadIdx.x] *= state[3 * r];
#pragma unroll
        for (int r = 0; r < 32; ++r)
            tile[r * ld + threadIdx.x] = v[size_t(base + r) * 256 + threadIdx.x];
        __syncthreads();
#pragma unroll
        for (int group = 0; group < 4; ++group) {
            int column = warp * 32 + group * 8;
            wm::fragment<wm::matrix_a, 8, 8, 4, double, wm::row_major> weights;
            wm::fragment<wm::matrix_b, 8, 8, 4, double, wm::row_major> values;
            wm::fragment<wm::accumulator, 8, 8, 4, double> result;
            wm::load_matrix_sync(result, accumulator + column, ld, wm::mem_row_major);
#pragma unroll
            for (int d = 0; d < 32; d += 4) {
                wm::load_matrix_sync(weights, scores + d, score_ld);
                wm::load_matrix_sync(values, tile + d * ld + column, ld);
                wm::mma_sync(result, weights, values, result);
            }
            wm::store_matrix_sync(accumulator + column, result, ld, wm::mem_row_major);
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < 8; ++r)
        output[(row + r) * 256 + threadIdx.x] = accumulator[r * ld + threadIdx.x];
#endif
}
constexpr int short_attention_shared_bytes =
    (8 * 260 + 32 * 260 + 8 * 260 + 2 * 8 * 36 + 8 * 3) * sizeof(double);
bool initialise_short_attention(int device)
{
#if defined(__CUDA_ARCH_LIST__)
    constexpr int architectures[] = {__CUDA_ARCH_LIST__};
    for (int architecture : architectures)
        if (architecture < 800)
            return false;
    int major = 0, capacity = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
        cudaDeviceGetAttribute(&capacity, cudaDevAttrMaxSharedMemoryPerBlockOptin, device) !=
            cudaSuccess ||
        major < 8 || capacity < short_attention_shared_bytes)
        return false;
    cudaError_t status =
        cudaFuncSetAttribute(attention_short_fp64, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             short_attention_shared_bytes);
    if (status != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return true;
#else
    (void)device;
    return false;
#endif
}
