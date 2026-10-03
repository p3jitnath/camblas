"""Call the standalone CAMBLAS CUDA library with PyTorch device tensors."""

import atexit
import ctypes as ct
import importlib.util
import os
import threading
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import torch

_LOCAL = threading.local()
_CONTEXTS = []
_LOCK = threading.Lock()
_ALGORITHMS = {
    "auto": 0,
    "classical": 1,
    "strassen": 2,
    "lt": 3,
    "symmetric": 4,
    "strassen2": 5,
}
_COUNTERS = (
    "classical",
    "strassen",
    "gram",
    "affine",
    "lt",
    "guard_fallback",
    "attention",
    "backward",
)


@lru_cache(maxsize=1)
def tensor_module():
    """Load the optional native tensor binding to minimise host call overhead."""
    directory = Path(
        os.environ.get(
            "CAMBLAS_CUDA_LIBRARY",
            Path(__file__).resolve().parents[1] / "build/cuda/libcamblas_cuda.so",
        )
    ).parent
    candidates = list(directory.glob("_camblas_cuda_torch*.so"))
    if not candidates:
        return None
    spec = importlib.util.spec_from_file_location("_camblas_cuda_torch", candidates[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _algorithm_id():
    return _ALGORITHMS[getattr(_LOCAL, "algorithm", "auto")]


def _algorithm_name():
    return getattr(_LOCAL, "algorithm", "auto")


@lru_cache(maxsize=1)
def library():
    """Load and declare the CUDA C ABI once per process."""
    path = Path(
        os.environ.get(
            "CAMBLAS_CUDA_LIBRARY",
            Path(__file__).resolve().parents[1] / "build/cuda/libcamblas_cuda.so",
        )
    )
    if not path.is_file():
        raise RuntimeError(
            f"Build the CUDA backend with scripts/build_cuda.py first: {path}"
        )
    lib = ct.CDLL(str(path))
    ptr = ct.c_void_p
    integer = ct.c_int
    declarations = {
        "create": ([integer, ptr, ct.POINTER(ptr)], integer),
        "destroy": ([ptr], integer),
        "error": ([ptr], ct.c_char_p),
        "set_algorithm": ([ptr, integer], integer),
        "stats": ([ptr, ct.POINTER(ct.c_uint64), integer], integer),
        "reset_stats": ([ptr], integer),
        "affine": (
            [ptr, integer, integer, integer, integer, ptr, ptr, ptr, integer, ptr],
            integer,
        ),
        "mlp": (
            [
                ptr,
                integer,
                integer,
                integer,
                integer,
                integer,
                ptr,
                ptr,
                ptr,
                ptr,
                ptr,
                ptr,
                ptr,
            ],
            integer,
        ),
    }
    for name, scalar in (("sgemm", ct.c_float), ("dgemm", ct.c_double)):
        declarations[name] = (
            [
                ptr,
                ct.c_char,
                ct.c_char,
                integer,
                integer,
                integer,
                scalar,
                ptr,
                integer,
                ptr,
                integer,
                scalar,
                ptr,
                integer,
            ],
            integer,
        )
    declarations["attention"] = (
        [
            ptr,
            integer,
            integer,
            integer,
            integer,
            integer,
            ct.c_double,
            ptr,
            ptr,
            ptr,
            ptr,
        ],
        integer,
    )
    for name, (arguments, result) in declarations.items():
        function = getattr(lib, "camblas_cuda_" + name)
        function.argtypes = arguments
        function.restype = result
    return lib


def _check(status, handle):
    if status:
        error = library().camblas_cuda_error(handle).decode()
        raise RuntimeError(error)


class Context:
    """Own a native handle for one device, stream and Python host thread."""

    def __init__(self, device, stream):
        self.pid = os.getpid()
        self.device = device
        self.handle = ct.c_void_p()
        _check(
            library().camblas_cuda_create(device, stream, ct.byref(self.handle)), None
        )
        self.algorithm = "auto"
        with _LOCK:
            _CONTEXTS.append(self)

    def set_algorithm(self, name):
        """Select a numerical algorithm without rebuilding the library."""
        if name != self.algorithm:
            _check(
                library().camblas_cuda_set_algorithm(self.handle, _ALGORITHMS[name]),
                self.handle,
            )
            self.algorithm = name


def context(device):
    """Return the current thread's context for the active stream on a device."""
    if not hasattr(_LOCAL, "contexts"):
        _LOCAL.contexts = {}
    stream = torch.cuda.current_stream(device)
    key = (device, stream.cuda_stream)
    native = _LOCAL.contexts.get(key)
    if native is None or not native.handle:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Warm up CAMBLAS on this stream before CUDA graph capture"
            )
        native = Context(device, stream.cuda_stream)
        _LOCAL.contexts[key] = native
    if native.pid != os.getpid():
        raise RuntimeError("CUDA contexts cannot be reused after fork; use spawn")
    native.set_algorithm(getattr(_LOCAL, "algorithm", "auto"))
    return native


def close():
    """Release all native contexts after their streams complete."""
    if tensor_module.cache_info().currsize:
        module = tensor_module()
        if module is not None:
            module.close()
    if not library.cache_info().currsize:
        return
    with _LOCK:
        for native in _CONTEXTS:
            if native.handle and native.pid == os.getpid():
                library().camblas_cuda_destroy(native.handle)
                native.handle = ct.c_void_p()
        _CONTEXTS.clear()
    if hasattr(_LOCAL, "contexts"):
        _LOCAL.contexts.clear()


atexit.register(close)


@contextmanager
def algorithm(name):
    """Choose auto, classical, strassen or lt for this host thread.

    Strassen changes summation and can increase error for cancellation. Auto
    only considers guarded Strassen for large even square products. Neither
    algorithm enables TF32 or reduces the input dtype.
    """
    if name not in _ALGORITHMS:
        raise ValueError(f"Unknown algorithm {name!r}; choose {tuple(_ALGORITHMS)}")
    previous = getattr(_LOCAL, "algorithm", "auto")
    module = tensor_module()
    if module is not None:
        module.set_default_algorithm(_ALGORITHMS[name])
    _LOCAL.algorithm = name
    try:
        yield
    finally:
        _LOCAL.algorithm = previous
        if module is not None:
            module.set_default_algorithm(_ALGORITHMS[previous])


def set_algorithm(name):
    """Select the default algorithm for this Python host thread."""
    if name not in _ALGORITHMS:
        raise ValueError(f"Unknown algorithm {name!r}; choose {tuple(_ALGORITHMS)}")
    module = tensor_module()
    if module is not None:
        module.set_default_algorithm(_ALGORITHMS[name])
    _LOCAL.algorithm = name


def stats(device=None, reset=False, all_threads=False):
    """Read launch counts for the calling thread's current device and stream.

    Use ``all_threads=True`` to include autograd workers and the separate
    coherent-host contexts on the requested device.
    """
    if device is None:
        device = torch.cuda.current_device()
    elif not isinstance(device, int):
        device = torch.device(device).index
        if device is None:
            device = torch.cuda.current_device()
    module = tensor_module()
    if module is not None:
        if all_threads:
            return module.stats_all(device, reset)
        return module.stats(device, reset, _algorithm_id())
    if all_threads:
        totals = dict.fromkeys(_COUNTERS, 0)
        with _LOCK:
            for native in _CONTEXTS:
                if native.device != device or native.pid != os.getpid():
                    continue
                counts = (ct.c_uint64 * 8)()
                _check(
                    library().camblas_cuda_stats(native.handle, counts, 8),
                    native.handle,
                )
                for name, count in zip(_COUNTERS, counts):
                    totals[name] += count
                if reset:
                    _check(
                        library().camblas_cuda_reset_stats(native.handle), native.handle
                    )
        return totals
    native = context(device)
    counts = (ct.c_uint64 * 8)()
    _check(library().camblas_cuda_stats(native.handle, counts, 8), native.handle)
    if reset:
        _check(library().camblas_cuda_reset_stats(native.handle), native.handle)
    return dict(zip(_COUNTERS, counts))


def _validate(*tensors):
    if not tensors:
        return
    first = tensors[0]
    if first.device.type != "cuda" or first.dtype not in (torch.float32, torch.float64):
        raise ValueError("CAMBLAS CUDA requires CUDA FP32 or FP64 tensors")
    if any(t.device != first.device or t.dtype != first.dtype for t in tensors):
        raise ValueError("All operands must have the same device and dtype")
    if any(any(s > 2**31 - 1 for s in t.shape + t.stride()) for t in tensors):
        raise ValueError("The CUDA backend uses LP64 dimensions and leading dimensions")


def _operand(tensor):
    # Interpret the logical transpose as a column-major operand. Padded row
    # storage and ordinary transpose views require no materialisation.
    rows, columns = tensor.shape
    sr, sc = tensor.stride()
    if sc == 1 and sr >= max(1, columns):
        return tensor, b"N", sr
    if sr == 1 and sc >= max(1, rows):
        return tensor, b"T", sc
    contiguous = tensor.contiguous()
    return contiguous, b"N", max(1, columns)


def matmul_raw(a, b, out=None, alpha=1.0, beta=0.0):
    """Enqueue a rank-two product, optionally into caller-owned contiguous output."""
    module = tensor_module()
    if module is not None:
        return module.matmul(a, b, out, alpha, beta, _algorithm_id())
    supplied_output = out is not None
    _validate(a, b)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("matmul requires compatible rank-two operands")
    m, k = a.shape
    _, n = b.shape
    if out is None:
        if beta != 0:
            raise ValueError("beta requires an existing output tensor")
        out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    else:
        _validate(a, out)
        if torch.is_grad_enabled() and out.requires_grad:
            raise ValueError(
                "Autograd matmul does not support an out tensor requiring gradients"
            )
        if out.shape != (m, n) or not out.is_contiguous():
            raise ValueError(
                "Output must have the product shape and contiguous row storage"
            )
        if out.numel() and out.untyped_storage().data_ptr() in (
            a.untyped_storage().data_ptr(),
            b.untyped_storage().data_ptr(),
        ):
            raise ValueError("GEMM output cannot alias an input")
    a, ta, lda = _operand(a)
    b, tb, ldb = _operand(b)
    native = context(a.device.index)
    call = (
        library().camblas_cuda_sgemm
        if a.dtype == torch.float32
        else library().camblas_cuda_dgemm
    )
    _check(
        call(
            native.handle,
            tb,
            ta,
            n,
            m,
            k,
            alpha,
            b.data_ptr(),
            ldb,
            a.data_ptr(),
            lda,
            beta,
            out.data_ptr(),
            max(1, n),
        ),
        native.handle,
    )
    if supplied_output:
        torch.autograd.graph.increment_version(out)
    return out


def affine_raw(x, weight, bias, relu=False, out=None):
    """Enqueue a contiguous row-major affine transform with optional ReLU."""
    module = tensor_module()
    if module is not None:
        return module.affine(x, weight, bias, relu, out, _algorithm_id())
    supplied_output = out is not None
    _validate(x, weight, bias)
    if (
        x.ndim != 2
        or weight.ndim != 2
        or bias.ndim != 1
        or x.shape[1] != weight.shape[0]
        or bias.shape[0] != weight.shape[1]
    ):
        raise ValueError("Invalid affine operand shapes")
    x, weight, bias = x.contiguous(), weight.contiguous(), bias.contiguous()
    rows, inner = x.shape
    columns = weight.shape[1]
    if out is None:
        out = torch.empty((rows, columns), device=x.device, dtype=x.dtype)
    else:
        _validate(x, out)
        if torch.is_grad_enabled() and out.requires_grad:
            raise ValueError(
                "Autograd affine does not support an out tensor requiring gradients"
            )
        if out.shape != (rows, columns) or not out.is_contiguous():
            raise ValueError("Invalid affine output shape or strides")
        if any(
            out.untyped_storage().data_ptr() == operand.untyped_storage().data_ptr()
            for operand in (x, weight, bias)
        ):
            raise ValueError("Affine output cannot alias an input")
    native = context(x.device.index)
    _check(
        library().camblas_cuda_affine(
            native.handle,
            int(x.dtype == torch.float64),
            rows,
            inner,
            columns,
            x.data_ptr(),
            weight.data_ptr(),
            bias.data_ptr(),
            int(relu),
            out.data_ptr(),
        ),
        native.handle,
    )
    if supplied_output:
        torch.autograd.graph.increment_version(out)
    return out


def mlp_raw(x, w1, b1, w2, b2):
    """Enqueue a two-layer MLP and return its output and ReLU hidden tensor."""
    module = tensor_module()
    if module is not None:
        return module.mlp(x, w1, b1, w2, b2, _algorithm_id())
    _validate(x, w1, b1, w2, b2)
    if (
        x.ndim != 2
        or w1.ndim != 2
        or w2.ndim != 2
        or b1.ndim != 1
        or b2.ndim != 1
        or x.shape[1] != w1.shape[0]
        or w1.shape[1] != w2.shape[0]
        or b1.shape[0] != w1.shape[1]
        or b2.shape[0] != w2.shape[1]
    ):
        raise ValueError("Invalid MLP operand shapes")
    x, w1, b1, w2, b2 = (t.contiguous() for t in (x, w1, b1, w2, b2))
    rows, inputs = x.shape
    hidden, outputs = w1.shape[1], w2.shape[1]
    h = torch.empty((rows, hidden), device=x.device, dtype=x.dtype)
    out = torch.empty((rows, outputs), device=x.device, dtype=x.dtype)
    native = context(x.device.index)
    _check(
        library().camblas_cuda_mlp(
            native.handle,
            int(x.dtype == torch.float64),
            rows,
            inputs,
            hidden,
            outputs,
            x.data_ptr(),
            w1.data_ptr(),
            b1.data_ptr(),
            w2.data_ptr(),
            b2.data_ptr(),
            h.data_ptr(),
            out.data_ptr(),
        ),
        native.handle,
    )
    return out, h


def attention_raw(q, k, v, scale):
    """Enqueue dense scaled attention, preserving the input compute dtype."""
    module = tensor_module()
    if module is not None:
        return module.attention(q, k, v, scale, _algorithm_id())
    _validate(q, k, v)
    if (
        q.ndim != 2
        or k.ndim != 2
        or v.ndim != 2
        or q.shape[1] != k.shape[1]
        or k.shape[0] != v.shape[0]
    ):
        raise ValueError("Invalid attention operand shapes")
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    queries, depth = q.shape
    keys, values = k.shape[0], v.shape[1]
    out = torch.empty((queries, values), device=q.device, dtype=q.dtype)
    native = context(q.device.index)
    _check(
        library().camblas_cuda_attention(
            native.handle,
            int(q.dtype == torch.float64),
            queries,
            keys,
            depth,
            values,
            scale,
            q.data_ptr(),
            k.data_ptr(),
            v.data_ptr(),
            out.data_ptr(),
        ),
        native.handle,
    )
    return out


def mlp_backward_raw(x, w1, w2, hidden, grad, needed, policy):
    """Enqueue fused MLP first derivatives when the tensor binding is available."""
    module = tensor_module()
    if module is None:
        return None
    return module.mlp_backward(x, w1, w2, hidden, grad, needed, _ALGORITHMS[policy])
