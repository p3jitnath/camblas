// Keep the application on ATen operators and use the original CUDA kernels for
// unsupported inputs. The at::cuda entry points bypass the dispatcher, so
// falling back from these registrations cannot call CAMBLAS recursively.
std::unique_ptr<torch::Library> torch_backend;
int torch_algorithm = CAMBLAS_CUDA_LT;

bool torch_matrix_supported(const Tensor &a, const Tensor &b)
{
    if (!a.is_cuda() || !b.is_cuda() || a.layout() != at::kStrided || b.layout() != at::kStrided ||
        a.dim() != 2 || b.dim() != 2 || a.device() != b.device() ||
        a.scalar_type() != b.scalar_type() || a.size(1) != b.size(0) || a.is_conj() ||
        b.is_conj() || a.is_neg() || b.is_neg())
        return false;
    const auto dtype = a.scalar_type();
    if (dtype != at::kFloat && dtype != at::kDouble && dtype != at::kBFloat16)
        return false;
    if (dtype == at::kFloat && at::globalContext().allowTF32CuBLAS())
        return false;
    for (const auto &value : {a, b}) {
        for (int64_t size : value.sizes())
            if (size == 0 || size > INT_MAX)
                return false;
        for (int64_t stride : value.strides())
            if (stride > INT_MAX || stride < 0)
                return false;
    }
    return true;
}

Tensor torch_mm(const Tensor &a, const Tensor &b)
{
    if (!torch_matrix_supported(a, b))
        return at::cuda::mm(a, b);
    return matmul(a, b, std::nullopt, 1., 0., torch_algorithm, false);
}

Tensor &torch_mm_out(const Tensor &a, const Tensor &b, Tensor &out)
{
    if (!torch_matrix_supported(a, b) || out.device() != a.device() ||
        out.scalar_type() != a.scalar_type() || out.layout() != at::kStrided || out.dim() != 2 ||
        out.size(0) != a.size(0) || out.size(1) != b.size(1) || !out.is_contiguous() ||
        out.is_conj() || out.is_neg() || out.is_alias_of(a) || out.is_alias_of(b))
        return at::cuda::mm_out(out, a, b);
    // ADInplaceOrView owns the version increment for ordinary ATen out calls.
    matmul(a, b, out, 1., 0., torch_algorithm, false);
    return out;
}

bool torch_bias_supported(const Tensor &bias, const Tensor &a, const Tensor &b)
{
    if (bias.device() != a.device() || bias.scalar_type() != a.scalar_type() ||
        bias.layout() != at::kStrided || bias.dim() > 2 || bias.is_conj() || bias.is_neg())
        return false;
    if (bias.dim() >= 1 && bias.size(-1) != 1 && bias.size(-1) != b.size(1))
        return false;
    return bias.dim() < 2 || bias.size(0) == 1 || bias.size(0) == a.size(0);
}

Tensor torch_addmm(const Tensor &bias, const Tensor &a, const Tensor &b, const at::Scalar &beta,
                   const at::Scalar &alpha)
{
    if (!torch_matrix_supported(a, b) || !torch_bias_supported(bias, a, b) || beta.isComplex() ||
        alpha.isComplex())
        return at::cuda::addmm(bias, a, b, beta, alpha);
    c10::cuda::CUDAGuard device(a.device());
    const double beta_value = beta.toDouble();
    Tensor out = beta_value == 0
                     ? at::empty({a.size(0), b.size(1)}, a.options())
                     : bias.expand({a.size(0), b.size(1)}).clone(at::MemoryFormat::Contiguous);
    return matmul(a, b, out, alpha.toDouble(), beta_value, torch_algorithm, false);
}

void install_torch_backend(int algorithm)
{
    static std::mutex install_mutex;
    std::lock_guard<std::mutex> lock(install_mutex);
    if (torch_backend)
        return;
    TORCH_CHECK(algorithm >= CAMBLAS_CUDA_AUTO && algorithm <= CAMBLAS_CUDA_STRASSEN_FOUR,
                "Invalid CAMBLAS CUDA algorithm");
    torch_algorithm = algorithm;
    auto registration = std::make_unique<torch::Library>(
        torch::Library::IMPL, "aten", c10::DispatchKey::CUDA, __FILE__, __LINE__);
    registration->impl("mm", TORCH_FN(torch_mm));
    registration->impl("mm.out", TORCH_FN(torch_mm_out));
    registration->impl("addmm", TORCH_FN(torch_addmm));
    torch_backend = std::move(registration);
}
