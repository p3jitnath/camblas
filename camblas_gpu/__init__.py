"""Experimental CAMBLAS CUDA operations with PyTorch tensor interoperability.

Build with ``python scripts/build_cuda.py --cuda-root /path/to/cuda``. Operations
use the active PyTorch CUDA stream and preserve FP32 or FP64 storage and compute
types; matrix multiplication and linear also support BF16 with FP32 accumulation.
Inputs on other streams require the same stream ordering and lifetime
management as ordinary PyTorch CUDA operations.
"""

from functools import partial

import torch
from torch.autograd.function import once_differentiable

from ._native import (
    _algorithm_name,
    affine_raw,
    algorithm,
    attention_raw,
    matmul_raw,
    mlp_backward_raw,
    mlp_raw,
    set_algorithm,
    stats,
    tensor_module,
)
from ._transfer import copy_to_cpu

__all__ = [
    "add_rms_norm",
    "affine",
    "algorithm",
    "attention",
    "attention_host",
    "attention_transfer",
    "copy_to_cpu",
    "gated_mlp",
    "gram_transfer",
    "linear",
    "matmul",
    "matmul_host",
    "matmul_transfer",
    "mlp",
    "mlp_host",
    "mlp_transfer",
    "qkv_linear",
    "quantized_matmul",
    "QuantizedGroups",
    "route_groups",
    "reduce_groups",
    "rms_norm",
    "set_algorithm",
    "stats",
    "silu_multiply",
    "swiglu",
]


class _Matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.algorithm = _algorithm_name()
        ctx.save_for_backward(a, b)
        return matmul_raw(a, b)

    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.saved_tensors
        # Use public operations so the gradient remains differentiable.
        with algorithm(ctx.algorithm):
            da = matmul(grad, b.T) if ctx.needs_input_grad[0] else None
            db = matmul(a.T, grad) if ctx.needs_input_grad[1] else None
        return da, db


def matmul(a, b, *, out=None, alpha=1.0, beta=0.0):
    """Multiply two CUDA matrices, supporting transpose views and autograd.

    Optional ``out``, ``alpha`` and ``beta`` provide GEMM semantics outside
    autograd. Strassen changes floating-point rounding; use
    ``with algorithm("classical")`` to request the classical cuBLAS path.
    """
    if torch.is_grad_enabled() and (a.requires_grad or b.requires_grad):
        if out is not None or beta != 0 or alpha != 1:
            raise ValueError("Autograd matmul requires alpha=1, beta=0 and no out")
        return _Matmul.apply(a, b)
    return matmul_raw(a, b, out, alpha, beta)


class _Affine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, relu):
        ctx.algorithm = _algorithm_name()
        output = affine_raw(x, weight, bias, relu)
        ctx.save_for_backward(x, weight, output)
        ctx.relu = relu
        return output

    @staticmethod
    def backward(ctx, grad):
        x, weight, output = ctx.saved_tensors
        if ctx.relu:
            grad = torch.where(output <= 0, 0, grad)
        with algorithm(ctx.algorithm):
            dx = matmul(grad, weight.T) if ctx.needs_input_grad[0] else None
            dw = matmul(x.T, grad) if ctx.needs_input_grad[1] else None
            db = grad.sum(0) if ctx.needs_input_grad[2] else None
        return dx, dw, db, None


def affine(x, weight, bias, *, relu=False):
    """Compute ``x @ weight + bias``, optionally followed by ReLU."""
    if torch.is_grad_enabled() and any(t.requires_grad for t in (x, weight, bias)):
        return _Affine.apply(x, weight, bias, relu)
    return affine_raw(x, weight, bias, relu)


class _Mlp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, b1, w2, b2):
        ctx.algorithm = _algorithm_name()
        output, hidden = mlp_raw(x, w1, b1, w2, b2)
        ctx.save_for_backward(x, w1, w2, hidden)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        x, w1, w2, hidden = ctx.saved_tensors
        native = mlp_backward_raw(
            x, w1, w2, hidden, grad, ctx.needs_input_grad, ctx.algorithm
        )
        if native is not None:
            return native
        with algorithm(ctx.algorithm):
            dw2 = matmul(hidden.T, grad) if ctx.needs_input_grad[3] else None
            db2 = grad.sum(0) if ctx.needs_input_grad[4] else None
            dx = dw1 = db1 = None
            if any(ctx.needs_input_grad[:3]):
                dh = torch.where(hidden <= 0, 0, matmul(grad, w2.T))
                dx = matmul(dh, w1.T) if ctx.needs_input_grad[0] else None
                dw1 = matmul(x.T, dh) if ctx.needs_input_grad[1] else None
                db1 = dh.sum(0) if ctx.needs_input_grad[2] else None
        return dx, dw1, db1, dw2, db2


def mlp(x, w1, b1, w2, b2):
    """Compute ``relu(x @ w1 + b1) @ w2 + b2`` with parameter gradients.

    MLP first derivatives are supported. Use two calls to :func:`affine` when
    higher derivatives through the hidden activation are required.
    """
    if torch.is_grad_enabled() and any(t.requires_grad for t in (x, w1, b1, w2, b2)):
        return _Mlp.apply(x, w1, b1, w2, b2)
    return mlp_raw(x, w1, b1, w2, b2)[0]


def attention(q, k, v, *, scale=None):
    """Compute dense single-head ``softmax(scale * q @ k.T) @ v``.

    The default scale is the inverse square root of the query depth. There is
    no mask or dropout. Gradient-enabled calls use differentiable CAMBLAS
    matrix products and PyTorch softmax; inference uses the fused native path.
    """
    if scale is None:
        scale = q.shape[1] ** -0.5
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        return matmul(torch.softmax(matmul(q, k.T) * scale, dim=-1), v)
    return attention_raw(q, k, v, scale)


def _host_binding():
    module = tensor_module()
    if module is None:
        raise RuntimeError("Coherent host operations require a build with --torch")
    return module


def matmul_host(a, b, *, out=None, alpha=1.0, beta=0.0, device=0):
    """Multiply CPU tensors synchronously on a GPU with coherent host page tables.

    This explicit inference path reads ordinary CPU allocations and returns
    completed CPU output. It issues no bulk tensor copies. Hardware memory
    traffic is still required, and large matrices can be slower than copying.
    """
    return _host_binding().matmul_host(
        a, b, out=out, alpha=alpha, beta=beta, device=device
    )


def mlp_host(x, w1, b1, w2, b2, *, device=0):
    """Run an MLP from CPU inputs to CPU output using coherent GPU memory access.

    Requires GPU support for pageable memory through host page tables. The
    hidden tensor stays on the GPU. The call completes before returning and
    supports inference only.
    """
    return _host_binding().mlp_host(x, w1, b1, w2, b2, device=device)


def attention_host(q, k, v, *, scale=None, device=0):
    """Run dense attention from CPU inputs to CPU output using coherent GPU access.

    The default scale is the inverse square root of the query depth. This
    synchronous inference path requires coherent host page tables, uses GPU
    score storage and has no mask or dropout.
    """
    return _host_binding().attention_host(q, k, v, scale=scale, device=device)


def _require_native(name, *args, **kwargs):
    """Report unavailable optional operations when using an older CUDA build."""
    module = tensor_module()
    if module is None or not hasattr(module, name):
        raise RuntimeError(f"{name} requires a current CUDA build with --torch")
    return getattr(module, name)(*args, **kwargs)


_tensor_binding = tensor_module()
if _tensor_binding is not None:
    matmul = _tensor_binding.matmul_public
    affine = _tensor_binding.affine_public
    mlp = _tensor_binding.mlp_public
    attention = _tensor_binding.attention_public
    matmul_host = _tensor_binding.matmul_host
    mlp_host = _tensor_binding.mlp_host
    attention_host = _tensor_binding.attention_host

# Share native implementations rather than duplicate Python wrappers. Older
# controls and ctypes-only builds report a clear error for unavailable features.
for _public, _native_name in {
    "add_rms_norm": "add_rms_norm",
    "linear": "linear_public",
    "rms_norm": "rms_norm",
    "silu_multiply": "silu_multiply",
    "swiglu": "swiglu",
    "gated_mlp": "gated_mlp",
    "qkv_linear": "qkv_linear",
    "quantized_matmul": "quantized_matmul",
    "QuantizedGroups": "QuantizedGroups",
    "route_groups": "route_groups",
    "reduce_groups": "reduce_groups",
    "matmul_transfer": "matmul_transfer",
    "gram_transfer": "gram_transfer",
    "mlp_transfer": "mlp_transfer",
    "attention_transfer": "attention_transfer",
}.items():
    globals()[_public] = getattr(_tensor_binding, _native_name, None) or partial(
        _require_native, _native_name
    )
