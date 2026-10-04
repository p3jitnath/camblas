"""Synchronous CUDA downloads with explicit CPU allocation policies."""

import torch


def copy_to_cpu(value, *, memory="pinned"):
    """Allocate a fresh CPU tensor and synchronously copy every CUDA value.

    Parameters
    ----------
    value : torch.Tensor
        CUDA tensor to download. The returned tensor is detached from autograd.
    memory : {'pinned', 'prefault', 'pageable'}, optional
        Pinned allocation uses PyTorch's host allocator. Prefault zero-fills a
        fresh pageable allocation before copying. Pageable uses Tensor.cpu().
        Every policy allocates a new tensor and copies the complete output.

    Returns
    -------
    torch.Tensor
        Completed CPU output with the input's shape and dtype.

    Raises
    ------
    ValueError
        If the input is not a CUDA tensor or the allocation policy is unknown.

    Notes
    -----
    Include this entire call in transfer timings. The pinned allocator can
    retain freed host storage, so first-use and warmed allocation costs differ.
    Prefault includes a complete CPU write before the CUDA download.
    """
    if not isinstance(value, torch.Tensor) or not value.is_cuda:
        raise ValueError("copy_to_cpu requires a CUDA tensor")
    if memory not in ("pinned", "prefault", "pageable"):
        raise ValueError("Unknown CPU output allocation policy")
    value = value.detach()
    if memory == "pageable":
        return value.cpu()
    result = torch.empty_like(value, device="cpu", pin_memory=memory == "pinned")
    if memory == "prefault":
        result.zero_()
    result.copy_(value)
    return result
