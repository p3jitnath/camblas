"""Private native kernels and diagnostics for backend integration and tests."""

from functools import partial

from . import _native as _native
from ._native import algorithm as algorithm
from ._native import set_algorithm as set_algorithm
from ._native import stats as stats
from ._native import tensor_module as tensor_module

_tensor_binding = tensor_module()


def _require_native(name, *args, **kwargs):
    """Report a missing internal kernel in a build without the tensor binding."""
    raise RuntimeError(f"{name} requires a current CUDA build with --torch")


for _name, _native_name in {
    "matmul": "matmul_public",
    "affine": "affine_public",
    "mlp": "mlp_public",
    "attention": "attention_public",
    "matmul_host": "matmul_host",
    "mlp_host": "mlp_host",
    "attention_host": "attention_host",
    "add_rms_norm": "add_rms_norm",
    "linear": "linear_public",
    "rms_norm": "rms_norm",
    "silu_multiply": "silu_multiply",
    "swiglu": "swiglu",
    "gated_mlp": "gated_mlp",
    "qkv_linear": "qkv_linear",
    "quantized_matmul": "quantized_matmul",
    "fp8_decode": "fp8_decode",
    "fp8_linear": "fp8_linear",
    "fp8_decode_supported": "fp8_decode_supported",
    "QuantizedGroups": "QuantizedGroups",
    "route_groups": "route_groups",
    "reduce_groups": "reduce_groups",
}.items():
    globals()[_name] = getattr(_tensor_binding, _native_name, None) or partial(
        _require_native, _native_name
    )
