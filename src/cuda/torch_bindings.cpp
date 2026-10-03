/* PyTorch interoperability: tensor metadata stays in native code. */
#include "camblas_cuda.h"

#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/softmax.h>
#include <ATen/ops/where.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/utils/pybind.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>
#include <unistd.h>

namespace
{
using at::Tensor;
const pid_t owner_process = getpid();
thread_local int default_algorithm = CAMBLAS_CUDA_AUTO;

void require(bool condition, const char *message)
{
    if (!condition)
        throw py::value_error(message);
}

void check_process()
{
    require(owner_process == getpid(), "CUDA bindings cannot be reused after fork; use spawn");
}

void check(int status, camblas_cuda_context *context)
{
    if (status != 0)
        throw std::runtime_error(camblas_cuda_error(context));
}

struct Handle;
struct Entry {
    int device;
    Handle *owner;
};

struct Registry {
    std::mutex mutex;
    std::map<camblas_cuda_context *, Entry> contexts;
};

Registry &registry()
{
    // Autograd worker TLS may be destroyed after Python and C++ static cleanup.
    static Registry *value = new Registry;
    return *value;
}

struct Handle {
    camblas_cuda_context *context = nullptr;
    Handle(int device, cudaStream_t stream)
    {
        check(camblas_cuda_create(device, static_cast<void *>(stream), &context), nullptr);
        try {
            std::lock_guard<std::mutex> lock(registry().mutex);
            registry().contexts.emplace(context, Entry{device, this});
        } catch (...) {
            camblas_cuda_destroy(context);
            throw;
        }
    }
    ~Handle()
    {
        if (!context || owner_process != getpid())
            return;
        {
            std::lock_guard<std::mutex> lock(registry().mutex);
            registry().contexts.erase(context);
        }
        camblas_cuda_destroy(context);
    }
};

thread_local std::map<std::tuple<int, cudaStream_t, bool>, std::unique_ptr<Handle>> handles;

camblas_cuda_context *get_context(int device, int algorithm, bool host_memory = false)
{
    check_process();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device).stream();
    auto key = std::make_tuple(device, stream, host_memory);
    auto iterator = handles.find(key);
    if (iterator == handles.end() || !iterator->second->context) {
        cudaStreamCaptureStatus capture;
        cudaError_t error = cudaStreamIsCapturing(stream, &capture);
        require(error == cudaSuccess && capture == cudaStreamCaptureStatusNone,
                "Warm up CAMBLAS on this stream before CUDA graph capture");
        if (iterator == handles.end())
            iterator = handles.emplace(key, std::make_unique<Handle>(device, stream)).first;
        else
            iterator->second = std::make_unique<Handle>(device, stream);
    }
    auto *context = iterator->second->context;
    check(camblas_cuda_set_algorithm(context, algorithm), context);
    return context;
}

void validate(const Tensor &tensor, const Tensor &reference)
{
    check_process();
    require(tensor.is_cuda() &&
                (tensor.scalar_type() == at::kFloat || tensor.scalar_type() == at::kDouble),
            "CAMBLAS CUDA requires CUDA FP32 or FP64 tensors");
    require(tensor.device() == reference.device() &&
                tensor.scalar_type() == reference.scalar_type(),
            "All operands must have the same device and dtype");
    for (int64_t size : tensor.sizes())
        require(size <= INT_MAX, "LP64 dimension overflow");
    for (int64_t stride : tensor.strides())
        require(stride <= INT_MAX, "LP64 stride overflow");
}

struct Operand {
    Tensor tensor;
    char transpose;
    int ld;
};

Operand operand(const Tensor &tensor)
{
    if (tensor.stride(1) == 1 && tensor.stride(0) >= std::max<int64_t>(1, tensor.size(1)))
        return {tensor, 'N', int(tensor.stride(0))};
    if (tensor.stride(0) == 1 && tensor.stride(1) >= std::max<int64_t>(1, tensor.size(0)))
        return {tensor, 'T', int(tensor.stride(1))};
    return {tensor.contiguous(), 'N', int(std::max<int64_t>(1, tensor.size(1)))};
}

Tensor matmul(const Tensor &a, const Tensor &b, const std::optional<Tensor> &output, double alpha,
              double beta, int algorithm)
{
    validate(a, a);
    validate(b, a);
    require(a.dim() == 2 && b.dim() == 2 && a.size(1) == b.size(0),
            "matmul requires compatible rank-two operands");
    c10::cuda::CUDAGuard device(a.device());
    int m = int(a.size(0)), k = int(a.size(1)), n = int(b.size(1));
    Tensor c;
    if (output) {
        c = *output;
        validate(c, a);
        require(!c10::GradMode::is_enabled() || !c.requires_grad(),
                "Autograd matmul does not support an out tensor requiring gradients");
        require(c.dim() == 2 && c.size(0) == m && c.size(1) == n && c.is_contiguous(),
                "Output must have the product shape and contiguous row storage");
        require(!c.is_alias_of(a) && !c.is_alias_of(b), "GEMM output cannot alias an input");
    } else {
        require(beta == 0, "beta requires an existing output tensor");
        c = at::empty({m, n}, a.options());
    }
    Operand left = operand(b), right = operand(a);
    auto *context = get_context(a.get_device(), algorithm);
    int status;
    if (a.scalar_type() == at::kFloat)
        status = camblas_cuda_sgemm(context, left.transpose, right.transpose, n, m, k, float(alpha),
                                    left.tensor.const_data_ptr<float>(), left.ld,
                                    right.tensor.const_data_ptr<float>(), right.ld, float(beta),
                                    c.mutable_data_ptr<float>(), std::max(1, n));
    else
        status = camblas_cuda_dgemm(context, left.transpose, right.transpose, n, m, k, alpha,
                                    left.tensor.const_data_ptr<double>(), left.ld,
                                    right.tensor.const_data_ptr<double>(), right.ld, beta,
                                    c.mutable_data_ptr<double>(), std::max(1, n));
    check(status, context);
    if (output && !c.is_inference())
        torch::autograd::impl::bump_version(c);
    return c;
}

Tensor matmul_public(const Tensor &a, const Tensor &b, const std::optional<Tensor> &output,
                     double alpha, double beta, int algorithm);

class MatmulAutograd : public torch::autograd::Function<MatmulAutograd>
{
  public:
    static Tensor forward(torch::autograd::AutogradContext *context, const Tensor &a,
                          const Tensor &b, int algorithm)
    {
        context->save_for_backward({a, b});
        context->saved_data["algorithm"] = algorithm;
        return matmul(a, b, std::nullopt, 1, 0, algorithm);
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext *context,
                                                   torch::autograd::variable_list gradients)
    {
        auto inputs = context->get_saved_variables();
        int algorithm = int(context->saved_data["algorithm"].toInt());
        Tensor da, db;
        // These calls create a graph when higher derivatives are requested.
        if (context->needs_input_grad(0))
            da = matmul_public(gradients[0], inputs[1].t(), std::nullopt, 1, 0, algorithm);
        if (context->needs_input_grad(1))
            db = matmul_public(inputs[0].t(), gradients[0], std::nullopt, 1, 0, algorithm);
        return {da, db, Tensor()};
    }
};

Tensor matmul_public(const Tensor &a, const Tensor &b, const std::optional<Tensor> &output,
                     double alpha, double beta, int algorithm)
{
    check_process();
    if (algorithm < 0)
        algorithm = default_algorithm;
    if (c10::GradMode::is_enabled() && (a.requires_grad() || b.requires_grad())) {
        require(!output && alpha == 1 && beta == 0,
                "Autograd matmul requires alpha=1, beta=0 and no out");
        return MatmulAutograd::apply(a, b, algorithm);
    }
    return matmul(a, b, output, alpha, beta, algorithm);
}

Tensor affine(const Tensor &x, const Tensor &weight, const Tensor &bias, bool relu,
              const std::optional<Tensor> &output, int algorithm)
{
    validate(x, x);
    validate(weight, x);
    validate(bias, x);
    require(x.dim() == 2 && weight.dim() == 2 && bias.dim() == 1 && x.size(1) == weight.size(0) &&
                weight.size(1) == bias.size(0),
            "Invalid affine operand shapes");
    c10::cuda::CUDAGuard device(x.device());
    int rows = int(x.size(0)), inner = int(x.size(1)), columns = int(weight.size(1));
    Tensor c;
    if (output) {
        c = *output;
        validate(c, x);
        require(!c10::GradMode::is_enabled() || !c.requires_grad(),
                "Autograd affine does not support an out tensor requiring gradients");
        require(c.dim() == 2 && c.size(0) == rows && c.size(1) == columns && c.is_contiguous(),
                "Invalid affine output shape or strides");
        require(!c.is_alias_of(x) && !c.is_alias_of(weight) && !c.is_alias_of(bias),
                "Affine output cannot alias an input");
    } else
        c = at::empty({rows, columns}, x.options());
    Tensor xc = x.contiguous(), wc = weight.contiguous(), bc = bias.contiguous();
    auto *context = get_context(x.get_device(), algorithm);
    check(camblas_cuda_affine(context, x.scalar_type() == at::kDouble, rows, inner, columns,
                              xc.const_data_ptr(), wc.const_data_ptr(), bc.const_data_ptr(), relu,
                              c.mutable_data_ptr()),
          context);
    if (output && !c.is_inference())
        torch::autograd::impl::bump_version(c);
    return c;
}

class AffineAutograd : public torch::autograd::Function<AffineAutograd>
{
  public:
    static Tensor forward(torch::autograd::AutogradContext *context, const Tensor &x,
                          const Tensor &weight, const Tensor &bias, bool relu, int algorithm)
    {
        Tensor output = affine(x, weight, bias, relu, std::nullopt, algorithm);
        context->save_for_backward({x, weight, output});
        context->saved_data["relu"] = relu;
        context->saved_data["algorithm"] = algorithm;
        return output;
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext *context,
                                                   torch::autograd::variable_list gradients)
    {
        auto inputs = context->get_saved_variables();
        int algorithm = int(context->saved_data["algorithm"].toInt());
        Tensor gradient = gradients[0], dx, dw, db;
        if (context->saved_data["relu"].toBool())
            gradient = at::where(inputs[2].le(0), 0, gradient);
        if (context->needs_input_grad(0))
            dx = matmul_public(gradient, inputs[1].t(), std::nullopt, 1, 0, algorithm);
        if (context->needs_input_grad(1))
            dw = matmul_public(inputs[0].t(), gradient, std::nullopt, 1, 0, algorithm);
        if (context->needs_input_grad(2))
            db = gradient.sum(at::IntArrayRef{0});
        return {dx, dw, db, Tensor(), Tensor()};
    }
};

Tensor affine_public(const Tensor &x, const Tensor &weight, const Tensor &bias, bool relu,
                     int algorithm)
{
    check_process();
    if (algorithm < 0)
        algorithm = default_algorithm;
    if (c10::GradMode::is_enabled() &&
        (x.requires_grad() || weight.requires_grad() || bias.requires_grad()))
        return AffineAutograd::apply(x, weight, bias, relu, algorithm);
    return affine(x, weight, bias, relu, std::nullopt, algorithm);
}

std::pair<Tensor, Tensor> mlp(const Tensor &x, const Tensor &w1, const Tensor &b1, const Tensor &w2,
                              const Tensor &b2, int algorithm)
{
    validate(x, x);
    validate(w1, x);
    validate(b1, x);
    validate(w2, x);
    validate(b2, x);
    require(x.dim() == 2 && w1.dim() == 2 && w2.dim() == 2 && b1.dim() == 1 && b2.dim() == 1 &&
                x.size(1) == w1.size(0) && w1.size(1) == w2.size(0) && b1.size(0) == w1.size(1) &&
                b2.size(0) == w2.size(1),
            "Invalid MLP operand shapes");
    c10::cuda::CUDAGuard device(x.device());
    int rows = int(x.size(0)), inputs = int(x.size(1));
    int hidden = int(w1.size(1)), outputs = int(w2.size(1));
    Tensor h = at::empty({rows, hidden}, x.options()), c = at::empty({rows, outputs}, x.options());
    Tensor xc = x.contiguous(), w1c = w1.contiguous(), b1c = b1.contiguous();
    Tensor w2c = w2.contiguous(), b2c = b2.contiguous();
    auto *context = get_context(x.get_device(), algorithm);
    check(camblas_cuda_mlp(context, x.scalar_type() == at::kDouble, rows, inputs, hidden, outputs,
                           xc.const_data_ptr(), w1c.const_data_ptr(), b1c.const_data_ptr(),
                           w2c.const_data_ptr(), b2c.const_data_ptr(), h.mutable_data_ptr(),
                           c.mutable_data_ptr()),
          context);
    return {c, h};
}

py::dict stats(int device, bool reset, int algorithm)
{
    check_process();
    c10::cuda::CUDAGuard guard(device);
    auto *context = get_context(device, algorithm);
    uint64_t counts[8];
    check(camblas_cuda_stats(context, counts, 8), context);
    const char *names[] = {"classical", "strassen",       "gram",      "affine",
                           "lt",        "guard_fallback", "attention", "backward"};
    py::dict result;
    for (int i = 0; i < 8; ++i)
        result[names[i]] = counts[i];
    if (reset)
        check(camblas_cuda_reset_stats(context), context);
    return result;
}

py::dict stats_all(int device, bool reset)
{
    check_process();
    uint64_t totals[8] = {};
    std::lock_guard<std::mutex> lock(registry().mutex);
    for (const auto &entry : registry().contexts) {
        if (entry.second.device != device)
            continue;
        uint64_t counts[8];
        check(camblas_cuda_stats(entry.first, counts, 8), entry.first);
        for (int i = 0; i < 8; ++i)
            totals[i] += counts[i];
        if (reset)
            check(camblas_cuda_reset_stats(entry.first), entry.first);
    }
    const char *names[] = {"classical", "strassen",       "gram",      "affine",
                           "lt",        "guard_fallback", "attention", "backward"};
    py::dict result;
    for (int i = 0; i < 8; ++i)
        result[names[i]] = totals[i];
    return result;
}

Tensor attention(const Tensor &q, const Tensor &k, const Tensor &v, double scale, int algorithm)
{
    validate(q, q);
    validate(k, q);
    validate(v, q);
    require(q.dim() == 2 && k.dim() == 2 && v.dim() == 2 && q.size(1) == k.size(1) &&
                k.size(0) == v.size(0),
            "Invalid attention operand shapes");
    c10::cuda::CUDAGuard device(q.device());
    int queries = int(q.size(0)), keys = int(k.size(0)), depth = int(q.size(1)),
        values = int(v.size(1));
    Tensor qc = q.contiguous(), kc = k.contiguous(), vc = v.contiguous();
    Tensor output = at::empty({queries, values}, q.options());
    auto *context = get_context(q.get_device(), algorithm);
    check(camblas_cuda_attention(context, q.scalar_type() == at::kDouble, queries, keys, depth,
                                 values, scale, qc.const_data_ptr(), kc.const_data_ptr(),
                                 vc.const_data_ptr(), output.mutable_data_ptr()),
          context);
    return output;
}

Tensor attention_public(const Tensor &q, const Tensor &k, const Tensor &v,
                        const std::optional<double> &scale, int algorithm)
{
    check_process();
    require(q.dim() == 2 && (scale || q.size(1) > 0),
            "Attention requires a rank-two query and positive depth for the default scale");
    double value = scale ? *scale : 1 / std::sqrt(double(q.size(1)));
    if (algorithm < 0)
        algorithm = default_algorithm;
    if (c10::GradMode::is_enabled() &&
        (q.requires_grad() || k.requires_grad() || v.requires_grad())) {
        Tensor scores = matmul_public(q, k.t(), std::nullopt, 1, 0, algorithm).mul(value);
        return matmul_public(at::softmax(scores, -1, std::nullopt), v, std::nullopt, 1, 0,
                             algorithm);
    }
    return attention(q, k, v, value, algorithm);
}

std::array<Tensor, 5> mlp_backward_tensors(const Tensor &x, const Tensor &w1, const Tensor &w2,
                                           const Tensor &h, const Tensor &gradient,
                                           const std::array<bool, 5> &needed, int algorithm)
{
    validate(x, x);
    validate(w1, x);
    validate(w2, x);
    validate(h, x);
    validate(gradient, x);
    require(x.dim() == 2 && w1.dim() == 2 && w2.dim() == 2 && h.dim() == 2 && gradient.dim() == 2 &&
                x.size(1) == w1.size(0) && w1.size(1) == w2.size(0) && h.size(0) == x.size(0) &&
                h.size(1) == w1.size(1) && gradient.size(0) == x.size(0) &&
                gradient.size(1) == w2.size(1),
            "Invalid backward operand shapes");
    c10::cuda::CUDAGuard device(x.device());
    int rows = int(x.size(0)), inputs = int(x.size(1));
    int hidden = int(w1.size(1)), outputs = int(w2.size(1));
    Tensor xc = x.contiguous(), w1c = w1.contiguous(), w2c = w2.contiguous();
    Tensor hc = h.contiguous(), gc = gradient.contiguous();
    Tensor dh, dx, dw1, db1, dw2, db2;
    if (needed[0] || needed[1] || needed[2])
        dh = at::empty({rows, hidden}, x.options());
    if (needed[0])
        dx = at::empty({rows, inputs}, x.options());
    if (needed[1])
        dw1 = at::empty({inputs, hidden}, x.options());
    if (needed[2])
        db1 = at::empty({hidden}, x.options());
    if (needed[3])
        dw2 = at::empty({hidden, outputs}, x.options());
    if (needed[4])
        db2 = at::empty({outputs}, x.options());
    auto pointer = [](Tensor &t) { return t.defined() ? t.mutable_data_ptr() : nullptr; };
    auto *context = get_context(x.get_device(), algorithm);
    check(camblas_cuda_mlp_backward(context, x.scalar_type() == at::kDouble, rows, inputs, hidden,
                                    outputs, xc.const_data_ptr(), w1c.const_data_ptr(),
                                    w2c.const_data_ptr(), hc.const_data_ptr(), gc.const_data_ptr(),
                                    pointer(dh), pointer(dx), pointer(dw1), pointer(db1),
                                    pointer(dw2), pointer(db2)),
          context);
    return {dx, dw1, db1, dw2, db2};
}

py::tuple mlp_backward(const Tensor &x, const Tensor &w1, const Tensor &w2, const Tensor &h,
                       const Tensor &gradient, const std::array<bool, 5> &needed, int algorithm)
{
    auto gradients = mlp_backward_tensors(x, w1, w2, h, gradient, needed, algorithm);
    py::tuple result(5);
    for (int i = 0; i < 5; ++i)
        result[i] = needed[i] ? py::cast(gradients[i]) : py::none();
    return result;
}

class MlpAutograd : public torch::autograd::Function<MlpAutograd>
{
  public:
    static Tensor forward(torch::autograd::AutogradContext *context, const Tensor &x,
                          const Tensor &w1, const Tensor &b1, const Tensor &w2, const Tensor &b2,
                          int algorithm)
    {
        auto output = mlp(x, w1, b1, w2, b2, algorithm);
        context->save_for_backward({x, w1, w2, output.second});
        context->saved_data["algorithm"] = algorithm;
        return output.first;
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext *context,
                                                   torch::autograd::variable_list gradients)
    {
        TORCH_CHECK(!c10::GradMode::is_enabled(),
                    "MLP supports first derivatives; use two affine calls for higher derivatives");
        auto inputs = context->get_saved_variables();
        int algorithm = int(context->saved_data["algorithm"].toInt());
        std::array<bool, 5> needed;
        for (int i = 0; i < 5; ++i)
            needed[i] = context->needs_input_grad(i);
        auto result = mlp_backward_tensors(inputs[0], inputs[1], inputs[2], inputs[3], gradients[0],
                                           needed, algorithm);
        return {result[0], result[1], result[2], result[3], result[4], Tensor()};
    }
};

Tensor mlp_public(const Tensor &x, const Tensor &w1, const Tensor &b1, const Tensor &w2,
                  const Tensor &b2, int algorithm)
{
    check_process();
    if (algorithm < 0)
        algorithm = default_algorithm;
    if (c10::GradMode::is_enabled() &&
        (x.requires_grad() || w1.requires_grad() || b1.requires_grad() || w2.requires_grad() ||
         b2.requires_grad()))
        return MlpAutograd::apply(x, w1, b1, w2, b2, algorithm);
    return mlp(x, w1, b1, w2, b2, algorithm).first;
}

void validate_host(const Tensor &tensor, const Tensor &reference)
{
    check_process();
    require(tensor.device().is_cpu() &&
                (tensor.scalar_type() == at::kFloat || tensor.scalar_type() == at::kDouble),
            "Host operations require CPU FP32 or FP64 tensors");
    require(tensor.scalar_type() == reference.scalar_type(),
            "All operands must have the same dtype");
    require(!c10::GradMode::is_enabled() || !tensor.requires_grad(),
            "Coherent host operations support inference; use CUDA tensors for autograd");
    for (int64_t size : tensor.sizes())
        require(size <= INT_MAX, "LP64 dimension overflow");
    for (int64_t stride : tensor.strides())
        require(stride <= INT_MAX, "LP64 stride overflow");
}

camblas_cuda_context *host_context(int device)
{
    check_process();
    // ATS allows GPU kernels to use ordinary system allocations on Grace Hopper.
    // Checking both properties excludes systems with software page migration.
    thread_local std::map<int, bool> support;
    if (!support.count(device)) {
        int pageable = 0, tables = 0;
        bool supported =
            cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, device) ==
                cudaSuccess &&
            cudaDeviceGetAttribute(&tables, cudaDevAttrPageableMemoryAccessUsesHostPageTables,
                                   device) == cudaSuccess &&
            pageable && tables;
        require(supported, "GPU must support coherent pageable memory through host page tables");
        support.emplace(device, true);
    }
    cudaStreamCaptureStatus capture;
    auto stream = c10::cuda::getCurrentCUDAStream(device).stream();
    require(cudaStreamIsCapturing(stream, &capture) == cudaSuccess &&
                capture == cudaStreamCaptureStatusNone,
            "Synchronous host operations cannot be captured in a CUDA graph");
    // Algorithms tuned for HBM pointers may be poor for coherent CPU storage.
    return get_context(device, default_algorithm, true);
}

struct HostCompletion {
    cudaStream_t stream;
    bool completed = false;
    explicit HostCompletion(int device) : stream(c10::cuda::getCurrentCUDAStream(device).stream())
    {
    }
    ~HostCompletion()
    {
        // CPU temporaries must survive queued work, including exception paths.
        if (!completed)
            cudaStreamSynchronize(stream);
    }
    void finish()
    {
        cudaError_t error = cudaStreamSynchronize(stream);
        completed = true;
        if (error != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(error));
    }
};

Tensor matmul_host(const Tensor &a, const Tensor &b, const std::optional<Tensor> &output,
                   double alpha, double beta, int device)
{
    validate_host(a, a);
    validate_host(b, a);
    require(a.dim() == 2 && b.dim() == 2 && a.size(1) == b.size(0),
            "matmul requires compatible rank-two operands");
    auto *context = host_context(device);
    c10::cuda::CUDAGuard guard(device);
    int m = int(a.size(0)), k = int(a.size(1)), n = int(b.size(1));
    Tensor c;
    if (output) {
        c = *output;
        validate_host(c, a);
        require(c.dim() == 2 && c.size(0) == m && c.size(1) == n && c.is_contiguous(),
                "Output must have the product shape and contiguous row storage");
        require(!c.is_alias_of(a) && !c.is_alias_of(b), "GEMM output cannot alias an input");
    } else {
        require(beta == 0, "beta requires an existing output tensor");
        c = at::empty({m, n}, a.options());
    }
    Operand left = operand(b), right = operand(a);
    HostCompletion completion(device);
    int status;
    if (a.scalar_type() == at::kFloat)
        status = camblas_cuda_sgemm(context, left.transpose, right.transpose, n, m, k, float(alpha),
                                    left.tensor.const_data_ptr<float>(), left.ld,
                                    right.tensor.const_data_ptr<float>(), right.ld, float(beta),
                                    c.mutable_data_ptr<float>(), std::max(1, n));
    else
        status = camblas_cuda_dgemm(context, left.transpose, right.transpose, n, m, k, alpha,
                                    left.tensor.const_data_ptr<double>(), left.ld,
                                    right.tensor.const_data_ptr<double>(), right.ld, beta,
                                    c.mutable_data_ptr<double>(), std::max(1, n));
    check(status, context);
    completion.finish();
    if (output && !c.is_inference())
        torch::autograd::impl::bump_version(c);
    return c;
}

Tensor mlp_host(const Tensor &x, const Tensor &w1, const Tensor &b1, const Tensor &w2,
                const Tensor &b2, int device)
{
    for (const Tensor &input : {x, w1, b1, w2, b2})
        validate_host(input, x);
    require(x.dim() == 2 && w1.dim() == 2 && w2.dim() == 2 && b1.dim() == 1 && b2.dim() == 1 &&
                x.size(1) == w1.size(0) && w1.size(1) == w2.size(0) && b1.size(0) == w1.size(1) &&
                b2.size(0) == w2.size(1),
            "Invalid MLP operand shapes");
    auto *context = host_context(device);
    c10::cuda::CUDAGuard guard(device);
    int rows = int(x.size(0)), inputs = int(x.size(1));
    int hidden = int(w1.size(1)), outputs = int(w2.size(1));
    Tensor h = at::empty({rows, hidden}, x.options().device(c10::Device(c10::kCUDA, device)));
    Tensor c = at::empty({rows, outputs}, x.options());
    Tensor xc = x.contiguous(), w1c = w1.contiguous(), b1c = b1.contiguous();
    Tensor w2c = w2.contiguous(), b2c = b2.contiguous();
    HostCompletion completion(device);
    check(camblas_cuda_mlp(context, x.scalar_type() == at::kDouble, rows, inputs, hidden, outputs,
                           xc.const_data_ptr(), w1c.const_data_ptr(), b1c.const_data_ptr(),
                           w2c.const_data_ptr(), b2c.const_data_ptr(), h.mutable_data_ptr(),
                           c.mutable_data_ptr()),
          context);
    completion.finish();
    return c;
}

Tensor attention_host(const Tensor &q, const Tensor &k, const Tensor &v,
                      const std::optional<double> &scale, int device)
{
    for (const Tensor &input : {q, k, v})
        validate_host(input, q);
    require(q.dim() == 2 && k.dim() == 2 && v.dim() == 2 && q.size(1) == k.size(1) &&
                k.size(0) == v.size(0) && (scale || q.size(1) > 0),
            "Invalid attention shapes or default scale");
    auto *context = host_context(device);
    c10::cuda::CUDAGuard guard(device);
    int queries = int(q.size(0)), keys = int(k.size(0)), depth = int(q.size(1)),
        values = int(v.size(1));
    double value = scale ? *scale : 1 / std::sqrt(double(depth));
    Tensor qc = q.contiguous(), kc = k.contiguous(), vc = v.contiguous();
    Tensor output = at::empty({queries, values}, q.options());
    HostCompletion completion(device);
    check(camblas_cuda_attention(context, q.scalar_type() == at::kDouble, queries, keys, depth,
                                 values, value, qc.const_data_ptr(), kc.const_data_ptr(),
                                 vc.const_data_ptr(), output.mutable_data_ptr()),
          context);
    completion.finish();
    return output;
}
} // namespace

void close_all()
{
    if (owner_process != getpid())
        return;
    // Python calls this while CUDA is still alive and host callers are idle.
    // Worker TLS destructors then observe null handles during engine shutdown.
    std::vector<camblas_cuda_context *> pending;
    {
        std::lock_guard<std::mutex> lock(registry().mutex);
        pending.reserve(registry().contexts.size());
        for (const auto &entry : registry().contexts) {
            pending.push_back(entry.first);
            entry.second.owner->context = nullptr;
        }
        registry().contexts.clear();
    }
    for (auto *context : pending)
        camblas_cuda_destroy(context);
    handles.clear();
}

PYBIND11_MODULE(_camblas_cuda_torch, module)
{
    module.def("matmul", &matmul);
    module.def("matmul_public", &matmul_public, py::arg("a"), py::arg("b"), py::kw_only(),
               py::arg("out") = py::none(), py::arg("alpha") = 1., py::arg("beta") = 0.,
               py::arg("algorithm") = -1,
               "Multiply two CUDA matrices with transpose views and autograd. Strassen changes "
               "rounding; use algorithm('classical') for classical cuBLAS multiplication.");
    module.def("set_default_algorithm", [](int algorithm) { default_algorithm = algorithm; });
    module.def("affine", &affine);
    module.def("affine_public", &affine_public, py::arg("x"), py::arg("weight"), py::arg("bias"),
               py::kw_only(), py::arg("relu") = false, py::arg("algorithm") = -1,
               "Compute x @ weight + bias, optionally followed by ReLU, with autograd.");
    module.def("mlp", &mlp);
    module.def("mlp_public", &mlp_public, py::arg("x"), py::arg("w1"), py::arg("b1"), py::arg("w2"),
               py::arg("b2"), py::kw_only(), py::arg("algorithm") = -1,
               "Compute relu(x @ w1 + b1) @ w2 + b2. First derivatives are supported; "
               "use two affine calls when higher derivatives are required.");
    module.def("attention", &attention);
    module.def("attention_public", &attention_public, py::arg("q"), py::arg("k"), py::arg("v"),
               py::kw_only(), py::arg("scale") = py::none(), py::arg("algorithm") = -1,
               "Compute dense single-head softmax(scale * q @ k.T) @ v without mask or dropout. "
               "The default scale is the inverse square root of the query depth.");
    module.def("mlp_backward", &mlp_backward);
    module.def("matmul_host", &matmul_host, py::arg("a"), py::arg("b"), py::kw_only(),
               py::arg("out") = py::none(), py::arg("alpha") = 1., py::arg("beta") = 0.,
               py::arg("device") = 0,
               "Synchronously multiply CPU matrices on a GPU with coherent host page tables. "
               "No explicit tensor copies; inference only, CPU output.");
    module.def("mlp_host", &mlp_host, py::arg("x"), py::arg("w1"), py::arg("b1"), py::arg("w2"),
               py::arg("b2"), py::kw_only(), py::arg("device") = 0,
               "Synchronously compute relu(x @ w1 + b1) @ w2 + b2 using CPU inputs and output "
               "on a GPU with coherent host page tables. Inference only.");
    module.def("attention_host", &attention_host, py::arg("q"), py::arg("k"), py::arg("v"),
               py::kw_only(), py::arg("scale") = py::none(), py::arg("device") = 0,
               "Synchronously compute dense attention from CPU inputs into CPU output on "
               "a GPU with coherent host page tables. Inference only.");
    module.def("stats", &stats);
    module.def("stats_all", &stats_all);
    module.def("close", &close_all);
}
