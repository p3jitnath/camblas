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
#include <torch/version.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cfloat>
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

void validate(const Tensor &tensor, const Tensor &reference, bool allow_bfloat16 = false)
{
    check_process();
    require(tensor.is_cuda() &&
                (tensor.scalar_type() == at::kFloat || tensor.scalar_type() == at::kDouble ||
                 (allow_bfloat16 && tensor.scalar_type() == at::kBFloat16)),
            allow_bfloat16 ? "CAMBLAS matmul requires CUDA FP32, FP64 or BF16 tensors"
                           : "CAMBLAS CUDA requires CUDA FP32 or FP64 tensors");
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
    validate(a, a, true);
    validate(b, a, true);
    require(a.dim() == 2 && b.dim() == 2 && a.size(1) == b.size(0),
            "matmul requires compatible rank-two operands");
    c10::cuda::CUDAGuard device(a.device());
    int m = int(a.size(0)), k = int(a.size(1)), n = int(b.size(1));
    Tensor c;
    if (output) {
        c = *output;
        validate(c, a, true);
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
    else if (a.scalar_type() == at::kDouble)
        status = camblas_cuda_dgemm(context, left.transpose, right.transpose, n, m, k, alpha,
                                    left.tensor.const_data_ptr<double>(), left.ld,
                                    right.tensor.const_data_ptr<double>(), right.ld, beta,
                                    c.mutable_data_ptr<double>(), std::max(1, n));
    else
        status =
            camblas_cuda_bgemm(context, left.transpose, right.transpose, n, m, k, float(alpha),
                               left.tensor.const_data_ptr(), left.ld, right.tensor.const_data_ptr(),
                               right.ld, float(beta), c.mutable_data_ptr(), std::max(1, n));
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

Tensor linear_public(const Tensor &x, const Tensor &weight, const std::optional<Tensor> &bias,
                     int algorithm)
{
    validate(x, x, true);
    validate(weight, x, true);
    require(x.dim() >= 1 && weight.dim() == 2 && x.size(-1) == weight.size(1),
            "linear requires matching input and [outputs, inputs] weight dimensions");
    int64_t rows = 1;
    for (int64_t index = 0; index + 1 < x.dim(); ++index) {
        int64_t size = x.size(index);
        require(rows <= INT_MAX / std::max<int64_t>(1, size), "LP64 flattened row overflow");
        rows *= size;
    }
    if (bias) {
        validate(*bias, x, true);
        require(bias->dim() == 1 && bias->size(0) == weight.size(0),
                "linear bias must have the output width");
    }
    Tensor result =
        matmul_public(x.reshape({rows, x.size(-1)}), weight.t(), std::nullopt, 1., 0., algorithm);
    if (bias) {
        result = result.add(*bias);
    }
    auto shape = x.sizes().vec();
    shape.back() = weight.size(0);
    return result.reshape(shape);
}

void validate_fusion_inference(const Tensor &value, const Tensor &reference)
{
    validate(value, reference, true);
    require(value.scalar_type() != at::kDouble, "Native fusion requires FP32 or BF16 storage");
    require(!c10::GradMode::is_enabled() || !value.requires_grad(),
            "Native fusion supports inference only");
}

Tensor silu_multiply_public(const Tensor &gate, const Tensor &up)
{
    validate_fusion_inference(gate, gate);
    validate_fusion_inference(up, gate);
    require(gate.sizes() == up.sizes(), "SiLU operands must have matching shapes");
    c10::cuda::CUDAGuard device(gate.device());
    Tensor gc = gate.contiguous(), uc = up.contiguous();
    Tensor result = at::empty(gate.sizes(), gate.options());
    auto *context = get_context(gate.get_device(), default_algorithm);
    check(camblas_cuda_silu_multiply(context, gate.scalar_type() == at::kFloat ? 0 : 2,
                                     gate.numel(), gc.const_data_ptr(), uc.const_data_ptr(),
                                     result.mutable_data_ptr()),
          context);
    return result;
}

Tensor swiglu_public(const Tensor &gate, const Tensor &up, const std::optional<Tensor> &routing,
                     double limit)
{
    validate_fusion_inference(gate, gate);
    validate_fusion_inference(up, gate);
    require(gate.dim() >= 1 && gate.sizes() == up.sizes() && gate.size(-1) > 0 &&
                gate.size(-1) <= INT_MAX && std::isfinite(limit) && limit >= 0 && limit <= FLT_MAX,
            "Invalid SwiGLU shapes or clipping limit");
    c10::cuda::CUDAGuard device(gate.device());
    Tensor gc = gate.contiguous(), uc = up.contiguous(), rc;
    if (routing) {
        validate_fusion_inference(*routing, *routing);
        require(routing->device() == gate.device() && routing->scalar_type() == at::kFloat,
                "SwiGLU routing values must be FP32 on the input device");
        auto shape = gate.sizes().vec();
        shape.pop_back();
        bool matching = routing->sizes().vec() == shape;
        shape.push_back(1);
        require(matching || routing->sizes().vec() == shape,
                "SwiGLU routing requires one value per input row");
        rc = routing->contiguous();
    }
    Tensor result = at::empty(gate.sizes(), gate.options());
    auto *context = get_context(gate.get_device(), default_algorithm);
    check(camblas_cuda_swiglu(context, gate.scalar_type() == at::kFloat ? 0 : 2, gate.numel(),
                              int(gate.size(-1)), float(limit), gc.const_data_ptr(),
                              uc.const_data_ptr(), routing ? rc.const_data_ptr<float>() : nullptr,
                              result.mutable_data_ptr()),
          context);
    return result;
}

Tensor quantized_matmul_public(const Tensor &x, const Tensor &x_scale, const Tensor &weight,
                               const Tensor &weight_scale, int activation_block)
{
    check_process();
    require(x.is_cuda() && x.scalar_type() == at::ScalarType::Float8_e4m3fn,
            "Quantized input must be CUDA FP8 E4M3");
    bool packed = weight.scalar_type() == at::ScalarType::Float4_e2m1fn_x2;
    require(packed || weight.scalar_type() == at::ScalarType::Float8_e4m3fn,
            "Quantized weight must be FP8 E4M3 or packed FP4 E2M1");
    require(x_scale.scalar_type() == at::ScalarType::Float8_e8m0fnu &&
                weight_scale.scalar_type() == at::ScalarType::Float8_e8m0fnu,
            "Quantized scales must be E8M0");
    for (const Tensor *tensor : {&x, &x_scale, &weight, &weight_scale}) {
        require(tensor->device() == x.device() && tensor->is_contiguous(),
                "Quantized operands must be contiguous and on the same CUDA device");
        require(!c10::GradMode::is_enabled() || !tensor->requires_grad(),
                "Quantized multiplication supports inference only");
        for (int64_t size : tensor->sizes())
            require(size <= INT_MAX, "LP64 dimension overflow");
    }
    require(x.dim() >= 1 && weight.dim() == 2 && x_scale.dim() == x.dim() &&
                weight_scale.dim() == 2,
            "Quantized matrix and scale ranks do not match");
    int64_t inner = x.size(-1), outputs = weight.size(0), rows = 1;
    require(inner >= 32 && inner % 32 == 0 && outputs >= 8 && outputs % 8 == 0 &&
                weight.size(1) == inner / (packed ? 2 : 1) &&
                (activation_block == 32 || (packed && activation_block == 128)) &&
                inner % activation_block == 0,
            "Invalid quantized matrix dimensions or activation block");
    for (int64_t index = 0; index + 1 < x.dim(); ++index) {
        require(x_scale.size(index) == x.size(index), "Input scale shape mismatch");
        require(rows <= 65535 / std::max<int64_t>(1, x.size(index)),
                "Quantized multiplication supports at most 65535 rows");
        rows *= x.size(index);
    }
    require(x_scale.size(-1) == inner / activation_block &&
                weight_scale.size(0) == (packed ? outputs : (outputs + 31) / 32) &&
                weight_scale.size(1) == inner / 32,
            "Quantized scale shapes do not match their blocks");
    c10::cuda::CUDAGuard device(x.device());
    auto shape = x.sizes().vec();
    shape.back() = outputs;
    Tensor result = at::empty(shape, x.options().dtype(at::kBFloat16));
    auto *context = get_context(x.get_device(), default_algorithm);
    check(camblas_cuda_quantized_matmul(context, packed, int(rows), int(outputs), int(inner),
                                        activation_block, x.const_data_ptr(),
                                        x_scale.const_data_ptr(), weight.const_data_ptr(),
                                        weight_scale.const_data_ptr(), result.mutable_data_ptr()),
          context);
    return result;
}

bool fp8_decode_supported_public(int device_id)
{
    check_process();
    if (device_id < 0)
        device_id = c10::cuda::current_device();
    c10::cuda::CUDAGuard guard(c10::Device(c10::kCUDA, device_id));
    return camblas_cuda_fp8_decode_supported(get_context(device_id, default_algorithm)) != 0;
}

Tensor fp8_product_public(const Tensor &x, const c10::optional<Tensor> &x_scale,
                          const Tensor &weight, const Tensor &weight_scale)
{
    check_process();
    bool quantise = !x_scale.has_value();
    require(x.is_cuda() &&
                x.scalar_type() == (quantise ? at::kBFloat16 : at::ScalarType::Float8_e4m3fn) &&
                weight.scalar_type() == at::ScalarType::Float8_e4m3fn,
            "FP8 linear requires CUDA BF16/FP8 input and E4M3 weight");
    require(x.dim() >= 1 && weight.dim() == 2 && weight_scale.dim() == 2 && x.is_contiguous() &&
                weight.is_contiguous(),
            "FP8 linear requires contiguous matrices and rank-two weight scales");
    require(weight_scale.scalar_type() == at::kFloat, "FP8 weight scales must be FP32");
    for (const Tensor *tensor : {&x, &weight, &weight_scale}) {
        require(tensor->device() == x.device() &&
                    (!c10::GradMode::is_enabled() || !tensor->requires_grad()),
                "FP8 operands must share a CUDA device and support inference only");
        for (int64_t size : tensor->sizes())
            require(size <= INT_MAX, "LP64 dimension overflow");
    }
    int64_t inner = x.size(-1), outputs = weight.size(0);
    require(inner >= 32 && inner <= 8192 && inner % 32 == 0 && outputs >= 16 && outputs % 16 == 0 &&
                x.numel() == inner && weight.size(1) == inner,
            "Invalid FP8 linear matrix dimensions");
    require(weight_scale.size(0) == (outputs + 31) / 32 && weight_scale.size(1) == inner / 32,
            "FP8 weight scale shape does not match its blocks");
    int64_t sx = 1, sn = weight_scale.stride(0), sk = weight_scale.stride(1);
    if (!quantise) {
        require(x_scale->device() == x.device() && x_scale->scalar_type() == at::kFloat &&
                    x_scale->dim() == x.dim() &&
                    (!c10::GradMode::is_enabled() || !x_scale->requires_grad()),
                "FP8 input scales require matching CUDA device/rank and FP32 inference storage");
        for (int64_t index = 0; index + 1 < x.dim(); ++index)
            require(x_scale->size(index) == x.size(index), "FP8 input scale shape mismatch");
        require(x_scale->size(-1) == inner / 32, "FP8 input scale shape mismatch");
        sx = x_scale->stride(-1);
    }
    require(sx > 0 && sx <= INT_MAX && sn > 0 && sn <= INT_MAX && sk > 0 && sk <= INT_MAX,
            "FP8 scale strides must be positive LP64 values");
    c10::cuda::CUDAGuard guard(x.device());
    auto *context = get_context(x.get_device(), default_algorithm);
    auto shape = x.sizes().vec();
    shape.back() = outputs;
    Tensor result = at::empty(shape, x.options().dtype(at::kBFloat16));
    size_t workspace_bytes =
        camblas_cuda_fp8_workspace_size(int(quantise), int(outputs), int(inner));
    Tensor workspace;
    if (workspace_bytes)
        workspace =
            at::empty({int64_t(workspace_bytes / sizeof(float))}, x.options().dtype(at::kFloat));
    void *scratch = workspace_bytes ? workspace.mutable_data_ptr() : nullptr;
    int status =
        quantise
            ? camblas_cuda_fp8_linear(context, int(outputs), int(inner), x.const_data_ptr(),
                                      weight.const_data_ptr(), weight_scale.const_data_ptr<float>(),
                                      int(sn), int(sk), scratch, workspace_bytes,
                                      result.mutable_data_ptr())
            : camblas_cuda_fp8_decode(context, int(outputs), int(inner), x.const_data_ptr(),
                                      x_scale->const_data_ptr<float>(), weight.const_data_ptr(),
                                      weight_scale.const_data_ptr<float>(), int(sx), int(sn),
                                      int(sk), scratch, workspace_bytes, result.mutable_data_ptr());
    check(status, context);
    return result;
}

Tensor fp8_decode_public(const Tensor &x, const Tensor &x_scale, const Tensor &weight,
                         const Tensor &weight_scale)
{
    return fp8_product_public(x, x_scale, weight, weight_scale);
}

Tensor fp8_linear_public(const Tensor &x, const Tensor &weight, const Tensor &weight_scale)
{
    return fp8_product_public(x, c10::nullopt, weight, weight_scale);
}

struct QuantizedGroups {
    std::vector<Tensor> weights, scales;
    std::vector<const void *> pointers;
    Tensor metadata;
    int outputs, inner;
    bool packed, trainable = false;

    QuantizedGroups(const std::vector<Tensor> &w, const std::vector<Tensor> &s)
    {
        check_process();
        require(!w.empty() && w.size() <= 65535 && w.size() == s.size(),
                "Supply matching non-empty weight and scale groups");
        require(w[0].is_cuda() && w[0].dim() == 2, "Grouped weights must be CUDA matrices");
        packed = w[0].scalar_type() == at::ScalarType::Float4_e2m1fn_x2;
        require(packed || w[0].scalar_type() == at::ScalarType::Float8_e4m3fn,
                "Grouped weights require FP8 or packed FP4 storage");
        require(w[0].size(0) >= 8 && w[0].size(0) <= INT_MAX && w[0].size(0) % 8 == 0 &&
                    w[0].size(1) >= (packed ? 16 : 32) &&
                    w[0].size(1) <= INT_MAX / (packed ? 2 : 1),
                "Invalid grouped weight dimensions");
        outputs = int(w[0].size(0));
        inner = int(w[0].size(1)) * (packed ? 2 : 1);
        require(inner % 32 == 0, "Grouped inner dimension must be a multiple of 32");
        Tensor host = at::empty({int64_t(w.size()), 2},
                                at::TensorOptions().device(at::kCPU).dtype(at::kLong));
        int64_t *addresses = host.mutable_data_ptr<int64_t>();
        for (size_t i = 0; i < w.size(); ++i) {
            require(w[i].device() == w[0].device() && w[i].scalar_type() == w[0].scalar_type() &&
                        w[i].sizes() == w[0].sizes() && w[i].is_contiguous() &&
                        !(uintptr_t(w[i].const_data_ptr()) & (packed ? 1 : 3)) &&
                        s[i].device() == w[0].device() &&
                        s[i].scalar_type() == at::ScalarType::Float8_e8m0fnu && s[i].dim() == 2 &&
                        s[i].size(0) == (packed ? outputs : (outputs + 31LL) / 32) &&
                        s[i].size(1) == inner / 32 && s[i].is_contiguous(),
                    "Grouped weight shapes, dtypes, devices or scales differ");
            trainable |= w[i].requires_grad() || s[i].requires_grad();
            weights.push_back(w[i].detach());
            scales.push_back(s[i].detach());
            pointers.push_back(w[i].const_data_ptr());
            pointers.push_back(s[i].const_data_ptr());
            addresses[i * 2] = int64_t(uintptr_t(pointers[i * 2]));
            addresses[i * 2 + 1] = int64_t(uintptr_t(pointers[i * 2 + 1]));
        }
        c10::cuda::CUDAGuard guard(w[0].device());
        metadata = host.to(w[0].device());
    }

    Tensor matmul(const Tensor &x, const Tensor &sx, const Tensor &active, const Tensor &counts,
                  int activation_block)
    {
        check_process();
        require(!c10::GradMode::is_enabled() || (!trainable && !x.requires_grad()),
                "Grouped multiplication supports inference only");
        require(x.device() == metadata.device() &&
                    x.scalar_type() == at::ScalarType::Float8_e4m3fn && x.dim() == 3 &&
                    x.size(0) <= 65535 && x.size(1) <= 65535 && x.size(2) == inner &&
                    (activation_block == 32 || (packed && activation_block == 128)) &&
                    inner % activation_block == 0,
                "Invalid grouped input dimensions or activation block");
        require(sx.device() == x.device() && sx.scalar_type() == at::ScalarType::Float8_e8m0fnu &&
                    sx.dim() == 3 && sx.size(0) == x.size(0) && sx.size(1) == x.size(1) &&
                    sx.size(2) == inner / activation_block,
                "Grouped input scale shape or dtype mismatch");
        for (const Tensor *index : {&active, &counts})
            require(index->device() == x.device() && index->scalar_type() == at::kInt &&
                        index->dim() == 1 && index->numel() == x.size(0),
                    "Grouped indices/counts require one CUDA int32 value per group");
        for (const Tensor *value : {&x, &sx, &active, &counts})
            require(value->is_contiguous(), "Grouped operands must be contiguous");
        for (size_t i = 0; i < weights.size(); ++i)
            require(weights[i].const_data_ptr() == pointers[i * 2] &&
                        scales[i].const_data_ptr() == pointers[i * 2 + 1],
                    "Captured weight storage changed; recreate the group");
        c10::cuda::CUDAGuard guard(x.device());
        Tensor result =
            at::empty({x.size(0), x.size(1), outputs}, x.options().dtype(at::kBFloat16));
        auto *context = get_context(x.get_device(), default_algorithm);
        check(camblas_cuda_grouped_quantized_matmul(
                  context, packed, int(x.size(0)), int(weights.size()), int(x.size(1)), outputs,
                  inner, activation_block, x.const_data_ptr(), sx.const_data_ptr(),
                  metadata.const_data_ptr(), active.const_data_ptr(), counts.const_data_ptr(),
                  result.mutable_data_ptr()),
              context);
        return result;
    }
};

std::tuple<Tensor, Tensor, Tensor> route_groups_public(const Tensor &experts, const Tensor &active,
                                                       int first_expert, int rows)
{
    check_process();
    require(experts.is_cuda() && experts.scalar_type() == at::kLong && experts.dim() == 2 &&
                experts.is_contiguous() && experts.size(0) <= INT_MAX && experts.size(1) >= 1 &&
                experts.size(1) <= 64 && experts.numel() <= INT_MAX &&
                active.device() == experts.device() && active.scalar_type() == at::kInt &&
                active.dim() == 1 && active.is_contiguous() && active.numel() >= 1 &&
                active.numel() <= 65535 && rows >= 1 && rows <= 65535 &&
                active.numel() * rows <= INT_MAX && first_expert >= 0,
            "Invalid expert routing dimensions, indices or devices");
    c10::cuda::CUDAGuard guard(experts.device());
    auto options = active.options();
    Tensor counts = at::empty({active.numel()}, options);
    Tensor slots = at::empty({active.numel(), rows}, options);
    Tensor reverse = at::empty(experts.sizes(), options);
    auto *context = get_context(experts.get_device(), default_algorithm);
    check(camblas_cuda_route_groups(
              context, int(experts.size(0)), int(experts.size(1)), first_expert,
              int(active.numel()), rows, experts.const_data_ptr(), active.const_data_ptr(),
              counts.mutable_data_ptr(), slots.mutable_data_ptr(), reverse.mutable_data_ptr()),
          context);
    return std::make_tuple(slots, reverse, counts);
}

Tensor reduce_groups_public(const Tensor &input, const Tensor &experts, const Tensor &reverse)
{
    check_process();
    require(input.is_cuda() && input.scalar_type() == at::kBFloat16 && input.dim() == 3 &&
                input.is_contiguous() && input.size(0) * input.size(1) <= INT_MAX &&
                input.size(2) >= 1 && input.size(2) <= INT_MAX &&
                experts.device() == input.device() && experts.scalar_type() == at::kLong &&
                experts.dim() == 2 && experts.is_contiguous() && experts.size(0) <= INT_MAX &&
                experts.size(1) >= 1 && experts.size(1) <= 64 && experts.numel() <= INT_MAX &&
                reverse.device() == input.device() && reverse.scalar_type() == at::kInt &&
                reverse.sizes() == experts.sizes() && reverse.is_contiguous() &&
                (!c10::GradMode::is_enabled() || !input.requires_grad()),
            "Invalid grouped reduction shapes, types, devices or gradients");
    c10::cuda::CUDAGuard guard(input.device());
    Tensor result = at::empty({experts.size(0), input.size(2)}, input.options().dtype(at::kFloat));
    auto *context = get_context(input.get_device(), default_algorithm);
    check(camblas_cuda_reduce_groups(context, int(experts.size(0)), int(experts.size(1)),
                                     int(input.size(2)), int(input.size(0) * input.size(1)),
                                     experts.const_data_ptr(), reverse.const_data_ptr(),
                                     input.const_data_ptr(), result.mutable_data_ptr()),
          context);
    return result;
}

Tensor rms_norm_public(const Tensor &x, const Tensor &weight, double epsilon,
                       bool round_before_weight = true)
{
    validate_fusion_inference(x, x);
    if (round_before_weight)
        validate_fusion_inference(weight, x);
    else {
        validate_fusion_inference(weight, weight);
        require(weight.device() == x.device() &&
                    (weight.scalar_type() == x.scalar_type() ||
                     (x.scalar_type() == at::kBFloat16 && weight.scalar_type() == at::kFloat)),
                "Final-round RMS weight must match input dtype or use FP32 with BF16 input");
    }
    require(x.dim() >= 1 && weight.dim() == 1 && x.size(-1) == weight.size(0),
            "RMS norm weight must match the input's last dimension");
    require(std::isfinite(epsilon) && epsilon >= 0 && epsilon <= FLT_MAX,
            "RMS norm epsilon must be finite and nonnegative");
    int64_t rows = 1;
    for (int64_t index = 0; index + 1 < x.dim(); ++index) {
        int64_t size = x.size(index);
        require(rows <= INT_MAX / std::max<int64_t>(1, size), "LP64 flattened row overflow");
        rows *= size;
    }
    c10::cuda::CUDAGuard device(x.device());
    Tensor xc = x.contiguous(), wc = weight.contiguous();
    auto *context = get_context(x.get_device(), default_algorithm);
    constexpr bool known_reduction =
        TORCH_VERSION_MAJOR == 2 && (TORCH_VERSION_MINOR == 8 || TORCH_VERSION_MINOR == 10);
    constexpr bool descending = TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR == 10;
    if (known_reduction && rows == 1 && weight.size(0) >= 2048 && weight.size(0) <= 65536 &&
        weight.size(0) % 4 == 0) {
        Tensor result = at::empty(x.sizes(), x.options());
        check(camblas_cuda_rms_norm_decode(context, x.scalar_type() == at::kFloat ? 0 : 2,
                                           weight.scalar_type() == at::kFloat ? 0 : 2,
                                           int(weight.size(0)), float(epsilon), round_before_weight,
                                           descending, xc.const_data_ptr(), wc.const_data_ptr(),
                                           result.mutable_data_ptr()),
              context);
        return result;
    }
    if (!known_reduction && rows == 1 && x.scalar_type() == at::kFloat && round_before_weight) {
        // ATen may change its reduction tree between releases. Preserve the
        // reference order until a native decode reduction is verified for it.
        Tensor scale = xc.pow(2).mean(at::IntArrayRef{-1}, true).add(epsilon).rsqrt();
        return wc.mul(xc.mul(scale));
    }
    if (!round_before_weight) {
        Tensor values = xc.to(at::kFloat);
        Tensor scale = values.pow(2).mean(at::IntArrayRef{-1}, true).add(epsilon).rsqrt();
        return values.mul(scale).mul(wc.to(at::kFloat)).to(x.scalar_type());
    }
    Tensor result = at::empty(x.sizes(), x.options());
    if (x.scalar_type() == at::kBFloat16 && x.numel()) {
        // Preserve the reference's reduction order: changing it can move BF16
        // rounding boundaries and accumulate logit drift across 80 layers.
        Tensor squares = at::empty(x.sizes(), x.options().dtype(at::kFloat));
        check(camblas_cuda_bfloat16_square(context, x.numel(), xc.const_data_ptr(),
                                           squares.mutable_data_ptr<float>()),
              context);
        Tensor means = squares.mean(at::IntArrayRef{-1}, true);
        check(camblas_cuda_rms_scale(context, 2, x.numel(), int(weight.size(0)), float(epsilon),
                                     xc.const_data_ptr(), wc.const_data_ptr(),
                                     means.const_data_ptr<float>(), result.mutable_data_ptr()),
              context);
        return result;
    }
    check(camblas_cuda_rms_norm(context, x.scalar_type() == at::kFloat ? 0 : 2, int(rows),
                                int(weight.size(0)), float(epsilon), xc.const_data_ptr(),
                                wc.const_data_ptr(), result.mutable_data_ptr()),
          context);
    return result;
}

std::tuple<Tensor, Tensor> add_rms_norm_public(const Tensor &x, const Tensor &residual,
                                               const Tensor &weight, double epsilon)
{
    validate_fusion_inference(x, x);
    validate_fusion_inference(residual, x);
    validate_fusion_inference(weight, x);
    require(x.dim() >= 1 && x.sizes() == residual.sizes() && weight.dim() == 1 &&
                x.size(-1) == weight.size(0),
            "Residual RMS dimensions must match");
    require(std::isfinite(epsilon) && epsilon >= 0 && epsilon <= FLT_MAX, "Invalid RMS epsilon");
    int64_t width = weight.size(0);
    if (x.numel() != width || width < 2048 || width > 65536 || width % 4 ||
        TORCH_VERSION_MAJOR != 2 || TORCH_VERSION_MINOR != 8) {
        Tensor added = x.add(residual);
        return std::make_tuple(added, rms_norm_public(added, weight, epsilon));
    }
    c10::cuda::CUDAGuard device(x.device());
    Tensor xc = x.contiguous(), rc = residual.contiguous(), wc = weight.contiguous();
    Tensor added = at::empty(x.sizes(), x.options()), output = at::empty(x.sizes(), x.options());
    auto *context = get_context(x.get_device(), default_algorithm);
    check(camblas_cuda_add_rms_norm(context, x.scalar_type() == at::kFloat ? 0 : 2, int(width),
                                    float(epsilon), xc.const_data_ptr(), rc.const_data_ptr(),
                                    wc.const_data_ptr(), added.mutable_data_ptr(),
                                    output.mutable_data_ptr()),
          context);
    return std::make_tuple(added, output);
}

Tensor gated_mlp_public(const Tensor &x, const Tensor &gate_weight, const Tensor &up_weight,
                        const Tensor &down_weight, int algorithm)
{
    validate_fusion_inference(x, x);
    validate_fusion_inference(gate_weight, x);
    validate_fusion_inference(up_weight, x);
    validate_fusion_inference(down_weight, x);
    require(x.dim() >= 1 && gate_weight.dim() == 2 && up_weight.dim() == 2 &&
                down_weight.dim() == 2 && gate_weight.sizes() == up_weight.sizes() &&
                x.size(-1) == gate_weight.size(1) && down_weight.size(1) == gate_weight.size(0),
            "Invalid gated MLP dimensions");
    Tensor gate = linear_public(x, gate_weight, std::nullopt, algorithm);
    Tensor up = linear_public(x, up_weight, std::nullopt, algorithm);
    c10::cuda::CUDAGuard device(x.device());
    auto *context = get_context(x.get_device(), algorithm < 0 ? default_algorithm : algorithm);
    check(camblas_cuda_silu_multiply(context, x.scalar_type() == at::kFloat ? 0 : 2, gate.numel(),
                                     gate.const_data_ptr(), up.const_data_ptr(),
                                     gate.mutable_data_ptr()),
          context);
    return linear_public(gate, down_weight, std::nullopt, algorithm);
}

std::tuple<Tensor, Tensor, Tensor> qkv_linear_public(const Tensor &x, const Tensor &q_weight,
                                                     const Tensor &k_weight, const Tensor &v_weight,
                                                     int algorithm)
{
    validate_fusion_inference(x, x);
    for (const Tensor &weight : {q_weight, k_weight, v_weight})
        validate_fusion_inference(weight, x);
    Tensor q = linear_public(x, q_weight, std::nullopt, algorithm);
    Tensor k = linear_public(x, k_weight, std::nullopt, algorithm);
    Tensor v = linear_public(x, v_weight, std::nullopt, algorithm);
    return std::make_tuple(q, k, v);
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

Tensor upload_transfer(const Tensor &value, c10::Device device)
{
    if (!value.is_non_overlapping_and_dense())
        return value.to(device);
    Tensor output = at::empty_strided(value.sizes(), value.strides(),
                                      value.options().device(device).pinned_memory(false));
    if (value.numel()) {
        cudaError_t error = cudaMemcpyAsync(output.mutable_data_ptr(), value.const_data_ptr(),
                                            value.nbytes(), cudaMemcpyHostToDevice,
                                            c10::cuda::getCurrentCUDAStream(device.index()));
        if (error != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(error));
    }
    return output;
}

Tensor transfer_result(const Tensor &value, const std::string &memory, HostCompletion &completion)
{
    require(memory == "pageable" || memory == "prefault" || memory == "pinned",
            "Unknown CPU output allocation policy");
    Tensor output = at::empty(value.sizes(),
                              value.options().device(at::kCPU).pinned_memory(memory == "pinned"));
    if (memory == "prefault")
        output.zero_();
    if (value.numel()) {
        // Every transfer operation produces a fresh contiguous CUDA result.
        cudaError_t error =
            cudaMemcpyAsync(output.mutable_data_ptr(), value.const_data_ptr(), value.nbytes(),
                            cudaMemcpyDeviceToHost, completion.stream);
        if (error != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(error));
    }
    completion.finish();
    return output;
}

void validate_transfer(const Tensor &value, const Tensor &reference)
{
    validate_host(value, reference);
    require(!c10::GradMode::is_enabled() || !value.requires_grad(),
            "Explicit transfer operations support inference only");
}

void validate_transfer_stream()
{
    cudaStreamCaptureStatus capture;
    cudaError_t error = cudaStreamIsCapturing(c10::cuda::getCurrentCUDAStream(), &capture);
    require(error == cudaSuccess && capture == cudaStreamCaptureStatusNone,
            "Synchronous CPU transfers cannot run during CUDA graph capture");
}

Tensor matmul_transfer(const Tensor &a, const Tensor &b, int device, const std::string &memory,
                       int algorithm)
{
    validate_transfer(a, a);
    validate_transfer(b, a);
    require(a.dim() == 2 && b.dim() == 2 && a.size(1) == b.size(0),
            "matmul requires compatible rank-two operands");
    c10::cuda::CUDAGuard guard(device);
    validate_transfer_stream();
    HostCompletion completion(device);
    Tensor ac = upload_transfer(a, c10::Device(c10::kCUDA, device));
    Tensor bc = upload_transfer(b, c10::Device(c10::kCUDA, device));
    return transfer_result(matmul_public(ac, bc, std::nullopt, 1., 0., algorithm), memory,
                           completion);
}

Tensor gram_transfer(const Tensor &a, int device, const std::string &memory, int algorithm)
{
    validate_transfer(a, a);
    require(a.dim() == 2, "Gram requires a rank-two operand");
    c10::cuda::CUDAGuard guard(device);
    validate_transfer_stream();
    HostCompletion completion(device);
    Tensor ac = upload_transfer(a, c10::Device(c10::kCUDA, device));
    return transfer_result(matmul_public(ac.t(), ac, std::nullopt, 1., 0., algorithm), memory,
                           completion);
}

Tensor mlp_transfer(const Tensor &x, const Tensor &w1, const Tensor &b1, const Tensor &w2,
                    const Tensor &b2, int device, const std::string &memory, int algorithm)
{
    for (const Tensor &value : {x, w1, b1, w2, b2})
        validate_transfer(value, x);
    c10::cuda::CUDAGuard guard(device);
    validate_transfer_stream();
    HostCompletion completion(device);
    c10::Device target(c10::kCUDA, device);
    std::array<Tensor, 5> uploaded;
    int index = 0;
    for (const Tensor &value : {x, w1, b1, w2, b2})
        uploaded[index++] = upload_transfer(value, target);
    return transfer_result(
        mlp_public(uploaded[0], uploaded[1], uploaded[2], uploaded[3], uploaded[4], algorithm),
        memory, completion);
}

Tensor attention_transfer(const Tensor &q, const Tensor &k, const Tensor &v,
                          std::optional<double> scale, int device, const std::string &memory,
                          int algorithm)
{
    for (const Tensor &value : {q, k, v})
        validate_transfer(value, q);
    c10::cuda::CUDAGuard guard(device);
    validate_transfer_stream();
    HostCompletion completion(device);
    c10::Device target(c10::kCUDA, device);
    Tensor qc = upload_transfer(q, target), kc = upload_transfer(k, target),
           vc = upload_transfer(v, target);
    return transfer_result(attention_public(qc, kc, vc, scale, algorithm), memory, completion);
}

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
    module.attr("quantized_tile_rows") = 16;
    py::class_<QuantizedGroups>(
        module, "QuantizedGroups",
        "Capture CUDA quantized weight storages and retain their lifetimes. In-place value "
        "updates are visible; recreate the group after storage relocation. Keep the object "
        "alive while its captured graphs may replay.")
        .def(py::init<const std::vector<Tensor> &, const std::vector<Tensor> &>(),
             py::arg("weights"), py::arg("scales"))
        .def("matmul", &QuantizedGroups::matmul, py::arg("input"), py::arg("input_scale"),
             py::arg("active"), py::arg("counts"), py::kw_only(), py::arg("activation_block") = 32,
             "Multiply [groups,rows,inner] FP8 input with selected captured weights. Invalid "
             "active indices and padding return zero; counts clip to [0,rows]. Inference only.");
    module.def("route_groups", &route_groups_public, py::arg("experts"), py::arg("active"),
               py::kw_only(), py::arg("first_expert") = 0, py::arg("rows"),
               "Return slots, reverse mapping and counts for expert/choice IDs. Excess rows "
               "are discarded and unassigned slots are -1; execution uses the active stream.");
    module.def("reduce_groups", &reduce_groups_public, py::arg("input"), py::arg("experts"),
               py::arg("reverse"),
               "Sum BF16 contributions into FP32 in expert/choice order. "
               "Invalid IDs/rows are ignored; requires inference and contiguous CUDA tensors.");
    module.def("matmul", &matmul);
    module.def("matmul_public", &matmul_public, py::arg("a"), py::arg("b"), py::kw_only(),
               py::arg("out") = py::none(), py::arg("alpha") = 1., py::arg("beta") = 0.,
               py::arg("algorithm") = -1,
               "Multiply two CUDA matrices with transpose views and autograd. Strassen changes "
               "rounding; use algorithm('classical') for classical cuBLAS multiplication.");
    module.def("set_default_algorithm", [](int algorithm) { default_algorithm = algorithm; });
    module.def("linear_public", &linear_public, py::arg("input"), py::arg("weight"),
               py::arg("bias") = py::none(), py::kw_only(), py::arg("algorithm") = -1,
               "Apply a CUDA linear transform with [outputs, inputs] weights and autograd.");
    module.def("affine", &affine);
    module.def("qkv_linear", &qkv_linear_public, py::arg("input"), py::arg("q_weight"),
               py::arg("k_weight"), py::arg("v_weight"), py::kw_only(), py::arg("algorithm") = -1,
               "Apply three bias-free linear projections in one inference dispatch.");
    module.def("matmul_transfer", &matmul_transfer, py::arg("a"), py::arg("b"), py::kw_only(),
               py::arg("device") = 0, py::arg("output_memory") = "pinned",
               py::arg("algorithm") = -1, "Upload both CPU operands and return fresh CPU output.");
    module.def("gram_transfer", &gram_transfer, py::arg("a"), py::kw_only(), py::arg("device") = 0,
               py::arg("output_memory") = "pinned", py::arg("algorithm") = -1,
               "Upload a CPU matrix once and return its Gram matrix.");
    module.def("mlp_transfer", &mlp_transfer, py::arg("x"), py::arg("w1"), py::arg("b1"),
               py::arg("w2"), py::arg("b2"), py::kw_only(), py::arg("device") = 0,
               py::arg("output_memory") = "pinned", py::arg("algorithm") = -1,
               "Upload every MLP operand and return fresh CPU output synchronously.");
    module.def("attention_transfer", &attention_transfer, py::arg("q"), py::arg("k"), py::arg("v"),
               py::kw_only(), py::arg("scale") = py::none(), py::arg("device") = 0,
               py::arg("output_memory") = "pinned", py::arg("algorithm") = -1,
               "Upload every attention operand and return fresh CPU output synchronously.");
    module.def("silu_multiply", &silu_multiply_public, py::arg("gate"), py::arg("up"),
               "Compute SiLU(gate)*up with storage rounding, for inference.");
    module.def("swiglu", &swiglu_public, py::arg("gate"), py::arg("up"), py::kw_only(),
               py::arg("routing") = py::none(), py::arg("limit") = 0.,
               "Compute SiLU(gate)*up in FP32 with final storage rounding. Positive limit "
               "clips gate above and up on both sides; optional FP32 routing scales each row.");
    module.def("quantized_matmul", &quantized_matmul_public, py::arg("input"),
               py::arg("input_scale"), py::arg("weight"), py::arg("weight_scale"), py::kw_only(),
               py::arg("activation_block") = 32,
               "Multiply block-scaled FP8 input by FP8 or packed FP4 weights, returning BF16. "
               "Requires contiguous CUDA operands, E8M0 scales and an SM90+ binary.");
    module.def("fp8_decode_supported", &fp8_decode_supported_public, py::arg("device") = -1,
               "Return whether this CUDA build and device support native FP8 decode.");
    module.def("fp8_decode", &fp8_decode_public, py::arg("input"), py::arg("input_scale"),
               py::arg("weight"), py::arg("weight_scale"),
               "Multiply one block32 E4M3 token by E4M3 weights, using FP32 scales and "
               "ordered FP32 accumulation. Return fresh BF16 output on the current stream. "
               "Requires an SM90a binary on SM90 hardware; positive scale strides are allowed.");
    module.def("fp8_linear", &fp8_linear_public, py::arg("input"), py::arg("weight"),
               py::arg("weight_scale"),
               "Quantise one finite BF16 token to block32 E4M3 with power-of-two FP32 scales "
               "and multiply E4M3 weights with ordered FP32 accumulation. Return fresh BF16 "
               "output on the current stream. Requires an SM90a build on SM90 hardware.");
    module.def("add_rms_norm", &add_rms_norm_public, py::arg("x"), py::arg("residual"),
               py::arg("weight"), py::arg("epsilon") = 1e-6,
               "Return fresh residual sum and RMS-normalised output.");
    module.def("rms_norm", &rms_norm_public, py::arg("input"), py::arg("weight"),
               py::arg("epsilon") = 1e-6, py::kw_only(), py::arg("round_before_weight") = true,
               "Apply RMS normalisation in FP32. Set round_before_weight=False to multiply "
               "by BF16/FP32 weight before the final storage conversion.");
    module.def("gated_mlp", &gated_mlp_public, py::arg("input"), py::arg("gate_weight"),
               py::arg("up_weight"), py::arg("down_weight"), py::kw_only(),
               py::arg("algorithm") = -1, "Apply a bias-free SiLU gated MLP for inference.");
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
