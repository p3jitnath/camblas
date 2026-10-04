/* BF16 device storage with FP32 accumulation. Included inside backend namespace. */
__global__ void scale_bfloat16(int m, int n, float beta, __nv_bfloat16 *c, int ldc)
{
    size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= size_t(m) * n)
        return;
    size_t offset = index % m + (index / m) * ldc;
    c[offset] = __float2bfloat16_rn(beta == 0 ? 0 : beta * __bfloat162float(c[offset]));
}

int bfloat16_gemm(camblas_cuda_context *context, char trans_a, char trans_b, int m, int n, int k,
                  float alpha, const __nv_bfloat16 *a, int lda, const __nv_bfloat16 *b, int ldb,
                  float beta, __nv_bfloat16 *c, int ldc)
{
    if (!context)
        return fail(nullptr, "missing context", 1);
    bool na = trans_a == 'N' || trans_a == 'n', ta = trans_a == 'T' || trans_a == 't';
    bool nb = trans_b == 'N' || trans_b == 'n', tb = trans_b == 'T' || trans_b == 't';
    if ((!na && !ta) || (!nb && !tb) || m < 0 || n < 0 || k < 0 || lda < 1 || ldb < 1 || ldc < 1)
        return fail(context, "invalid BF16 GEMM arguments", 1);
    if (m == 0 || n == 0)
        return 0;
    if (!c || lda < std::max(1, na ? m : k) || ldb < std::max(1, nb ? k : n) || ldc < m)
        return fail(context, "invalid BF16 pointer or leading dimension", 1);
    if (alpha == 0 || k == 0) {
        if (beta == 1)
            return 0;
        scale_bfloat16<<<unsigned((size_t(m) * n + 255) / 256), 256, 0, context->stream>>>(
            m, n, beta, c, ldc);
        cudaError_t error = cudaGetLastError();
        return error == cudaSuccess ? 0 : fail(context, "scale BF16 output", error);
    }
    if (!a || !b || a == c || b == c)
        return fail(context, "missing or aliased BF16 GEMM operand", 1);
    cublasOperation_t opa = na ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t opb = nb ? CUBLAS_OP_N : CUBLAS_OP_T;
    if (context->algorithm == CAMBLAS_CUDA_LT) {
        const __nv_bfloat16 *bias = nullptr;
        LtPlan *plan = get_plan(context, opa, opb, m, n, k, lda, ldb, ldc, bias, a, b, c, false);
        if (plan->available) {
            cublasStatus_t status = lt_call(context, plan, &plan->algorithm, alpha, a, b, beta, c);
            if (status == CUBLAS_STATUS_SUCCESS) {
                context->counts[4]++;
                return 0;
            }
            if (status != CUBLAS_STATUS_NOT_SUPPORTED)
                return fail(context, "BF16 cuBLASLt GEMM", status);
        }
    }
    // The default mirrors PyTorch's BF16 GEMM: float scalars/accumulation and
    // native BF16 operands/results. It never casts the full operands to FP32.
    context->counts[0]++;
    cublasStatus_t status = cublasGemmEx(context->blas, opa, opb, m, n, k, &alpha, a, CUDA_R_16BF,
                                         lda, b, CUDA_R_16BF, ldb, &beta, c, CUDA_R_16BF, ldc,
                                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    return status == CUBLAS_STATUS_SUCCESS ? 0 : fail(context, "BF16 GEMM", status);
}
