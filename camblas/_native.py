"""Manage private CUDA library loading, stream contexts and diagnostics."""

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
    "strassen3": 6,
    "strassen4": 7,
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
    """Load the optional native tensor binding once per process.

    Returns
    -------
    module or None
        Loaded tensor binding, or None when no binding exists beside the CUDA library.
    """
    override = os.environ.get("CAMBLAS_CUDA_LIBRARY")
    core = Path(
        override
        or Path(__file__).resolve().parents[1] / "build/cuda/libcamblas_cuda.so"
    )
    directory = core.parent
    candidates = list(directory.glob("_camblas_cuda_torch*.so"))
    if not candidates:
        return None
    if override:
        # Honour the requested core even when a copied binding retains the
        # build directory's RUNPATH. Fresh control processes must load the
        # control's SONAME before resolving the tensor binding's dependency.
        ct.CDLL(str(core), mode=ct.RTLD_GLOBAL)
    spec = importlib.util.spec_from_file_location("_camblas_cuda_torch", candidates[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def library():
    """Load and declare the CUDA C ABI once per process.

    Returns
    -------
    ctypes.CDLL
        CUDA library with argument and return types declared.

    Raises
    ------
    RuntimeError
        If the selected CUDA library has not been built.
    OSError
        If the dynamic loader cannot load the selected library.
    """
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
    for name, scalar in (
        ("sgemm", ct.c_float),
        ("dgemm", ct.c_double),
        ("bgemm", ct.c_float),
    ):
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
        if name == "bgemm" and not hasattr(lib, "camblas_cuda_bgemm"):
            continue
        function = getattr(lib, "camblas_cuda_" + name)
        function.argtypes = arguments
        function.restype = result
    return lib


def _check(status, handle):
    """Raise an exception when a native CUDA call reports failure.

    Parameters
    ----------
    status : int
        Native status code; zero means success.
    handle : ctypes.c_void_p or None
        Context handle, or None when context creation failed.

    Raises
    ------
    RuntimeError
        If the native call failed; includes its diagnostic message.
    """
    if status:
        error = library().camblas_cuda_error(handle).decode()
        raise RuntimeError(error)


class Context:
    """Own a native handle for one device, stream and host thread.

    Parameters
    ----------
    device : int
        CUDA device index.
    stream : int
        Raw CUDA stream handle; zero selects the default stream.

    Raises
    ------
    RuntimeError
        If the native context cannot be created.
    """

    def __init__(self, device, stream):
        """Initialise the native handle and register its lifetime."""
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
        """Select the multiplication algorithm for this native context.

        Parameters
        ----------
        name : str
            Name in the private algorithm table, such as auto, classical or lt.

        Raises
        ------
        KeyError
            If the name is unknown.
        RuntimeError
            If the native context rejects the selection.
        """
        if name != self.algorithm:
            _check(
                library().camblas_cuda_set_algorithm(self.handle, _ALGORITHMS[name]),
                self.handle,
            )
            self.algorithm = name


def context(device):
    """Return the current thread's context for the active CUDA stream.

    Parameters
    ----------
    device : int
        Required CUDA device index.

    Returns
    -------
    Context
        Reused native context with the requested device and active stream.
    """
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
    """Select a multiplication policy for the calling host thread.

    The policies are auto, classical, lt, symmetric, strassen, strassen2,
    strassen3 and strassen4. Strassen changes the summation order and can increase
    error when products cancel; auto considers it only for guarded large even
    square products. The policies keep TF32 disabled and preserve the declared
    storage and compute types.

    Parameters
    ----------
    name : str
        Multiplication policy to use within the context manager.

    Yields
    ------
    None
        The previous policy is restored when the context exits.

    Raises
    ------
    ValueError
        If the requested policy is unknown.
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
    """Select the multiplication policy for the calling host thread.

    Parameters
    ----------
    name : str
        Name in the private algorithm table, such as auto, classical or lt.

    Raises
    ------
    ValueError
        If the policy name is unknown.
    """
    if name not in _ALGORITHMS:
        raise ValueError(f"Unknown algorithm {name!r}; choose {tuple(_ALGORITHMS)}")
    module = tensor_module()
    if module is not None:
        module.set_default_algorithm(_ALGORITHMS[name])
    _LOCAL.algorithm = name


def stats(device=None, reset=False, all_threads=False):
    """Read launch counts for the current device and stream.

    Parameters
    ----------
    device : int, str or torch.device, optional
        CUDA device; None selects the current device.
    reset : bool, optional
        Clear the counters after reading them.
    all_threads : bool, optional
        Include autograd workers and coherent-host contexts on the device.

    Returns
    -------
    dict[str, int]
        Counts before any requested reset; CUDA graph replays do not increment
        host launch counters.
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
        return module.stats(
            device, reset, _ALGORITHMS[getattr(_LOCAL, "algorithm", "auto")]
        )
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
