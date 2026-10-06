"""Load CAMBLAS through PyTorch backend discovery without adding tensor APIs."""

import os


def _autoload():
    """Register native CUDA kernels when the process enables CAMBLAS.

    Notes
    -----
    PyTorch calls this entry point after its ordinary API is initialised.
    Disabled processes do not load a CAMBLAS library or initialise CUDA.
    """
    if os.environ.get("CAMBLAS_ENABLE", "0") != "1":
        return

    from camblas._native import _ALGORITHMS, tensor_module

    module = tensor_module()
    if module is None or not hasattr(module, "install_torch_backend"):
        raise RuntimeError(
            "Build the current PyTorch backend with scripts/build_cuda.py --torch"
        )
    name = os.environ.get(
        "CAMBLAS_CUDA_ALGORITHM", os.environ.get("CAMBLAS_SGLANG_ALGORITHM", "lt")
    )
    if name not in _ALGORITHMS:
        raise ValueError(f"Unknown CAMBLAS CUDA algorithm: {name}")
    module.install_torch_backend(_ALGORITHMS[name])
