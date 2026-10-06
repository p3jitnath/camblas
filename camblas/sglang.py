"""Opt-in CAMBLAS operations for SGLang's general plugin interface."""

import functools
import json
import os
from pathlib import Path

_calls = {}


def dispatch_counts():
    """Return host dispatch counts, excluding replayed CUDA graph operations.

    Returns
    -------
    dict[str, int]
        A copy of per-operation and per-shape counts for this process.
    """
    return dict(_calls)


def install_fp8_tiles(path):
    """Apply common SGLang tile settings for single-token GH200 FP8 products.

    Parameters
    ----------
    path : str or pathlib.Path
        JSON tile settings with the original 32 by 32 block scales.

    Raises
    ------
    ValueError
        If the settings change the block size or FP32 reduction order.
    """
    import torch
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    settings = json.loads(Path(path).read_text())
    if settings["block_shape"] != [32, 32]:
        raise ValueError("Only the original 32 by 32 FP8 scaling is supported")
    tiles = settings["tiles"]
    for tile in tiles.values():
        if tile["BLOCK_SIZE_K"] != 32 or tile.get("SPLIT_K", 1) != 1:
            raise ValueError("FP8 tiles must preserve the original reduction order")

    @functools.lru_cache
    def lookup(original, n, k, block_n, block_k, device):
        existing = original(n, k, block_n, block_k)
        tile = tiles.get(f"{n},{k}")
        if (
            existing is not None
            or tile is None
            or [block_n, block_k] != settings["block_shape"]
            or device != settings["device"]
        ):
            return existing
        baseline = dict(
            BLOCK_SIZE_M=64,
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_K=block_k,
            GROUP_SIZE_M=32,
            num_warps=4,
            num_stages=3,
        )
        # The nearest-key lookup selects the upstream default for M >= 2.
        return {1: tile, 2: baseline}

    def configured(original, N, K, block_n, block_k):
        if torch._dynamo.is_compiling():
            return original(N, K, block_n, block_k)
        return lookup(original, N, K, block_n, block_k, torch.cuda.get_device_name())

    HookRegistry.register(
        "sglang.kernels.ops.quantization.fp8_kernel.get_w8a8_block_fp8_configs",
        configured,
        HookType.AROUND,
    )


def canonical_moe_tokens(token_ids, expert_ids, count, block_size, choices):
    """Order token choices within pre-sorted expert blocks, retaining padding.

    Parameters
    ----------
    token_ids : torch.Tensor
        One-dimensional token-choice indices, including padded capacity.
    expert_ids : torch.Tensor
        Expert label for each existing block.
    count : torch.Tensor
        Scalar number of entries within live expert blocks.
    block_size : int
        Entries per expert block.
    choices : int
        Number of token choices; also the output padding sentinel.

    Returns
    -------
    torch.Tensor
        New indices with the original dtype and expert membership; unused
        capacity contains the padding sentinel.
    """
    import torch

    span = choices + 1
    labels = expert_ids.to(torch.int64).repeat_interleave(block_size)
    keys = (labels[: token_ids.numel()] + 1) * span + token_ids.to(torch.int64)
    live = torch.arange(token_ids.numel(), device=token_ids.device) < count
    sentinel = torch.iinfo(torch.int64).max
    keys = torch.where(live, keys, sentinel).sort().values
    return torch.where(keys == sentinel, choices, keys % span).to(token_ids.dtype)


def install_stable_moe():
    """Give Marlin prefill a repeatable token order without changing decode."""
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    def aligned(original, topk_ids, block_size, num_experts, *args, **kwargs):
        ids, experts, count = original(
            topk_ids, block_size, num_experts, *args, **kwargs
        )
        if topk_ids.shape[0] > 1:
            ids = canonical_moe_tokens(
                ids, experts, count, block_size, topk_ids.numel()
            )
        return ids, experts, count

    HookRegistry.register(
        "sglang.srt.layers.moe.moe_runner.triton_utils."
        "moe_align_block_size.moe_align_block_size",
        aligned,
        HookType.AROUND,
    )


def install():
    """Register selected inference operations through SGLang's hook registry.

    ``CAMBLAS_SGLANG_OPS=linear,fp8`` enables unquantised CUDA linear
    and batched products and selected single-token block32 and block128 FP8 products.
    CAMBLAS_SGLANG_FP8_TILES supplies common single-token tile settings,
    independently of the selected CAMBLAS operations. Unsupported storage,
    data types and gradient-enabled inputs use the original linear method.
    """
    if os.environ.get("CAMBLAS_SGLANG_STABLE_MOE") == "1":
        install_stable_moe()
    tiles = os.environ.get("CAMBLAS_SGLANG_FP8_TILES")
    if tiles:
        install_fp8_tiles(tiles)
    selected = set(filter(None, os.environ.get("CAMBLAS_SGLANG_OPS", "").split(",")))
    if not selected:
        return
    if selected - {"linear", "fp8"}:
        raise ValueError(f"Unknown CAMBLAS SGLang operations: {sorted(selected)}")

    import torch
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    import camblas._kernels as cb

    @functools.lru_cache
    def hopper_supported(device):
        import triton

        return torch.cuda.get_device_capability(device) == (9, 0) and tuple(
            map(int, triton.__version__.split(".")[:2])
        ) == (3, 7)

    cb.set_algorithm(os.environ.get("CAMBLAS_SGLANG_ALGORITHM", "lt"))
    if "linear" in selected:
        cb.tensor_module().install_torch_backend(
            cb._native._ALGORITHMS[os.environ.get("CAMBLAS_SGLANG_ALGORITHM", "lt")]
        )

    def linear(original, method, layer, x, bias=None):
        weight = layer.weight
        operands = (x, weight) if bias is None else (x, weight, bias)
        if (
            x.is_cuda
            and x.ndim >= 2
            and weight.ndim == 2
            and x.dtype in (torch.float32, torch.bfloat16)
            and all(t.is_contiguous() and t.device == x.device for t in operands)
            and all(t.dtype == x.dtype for t in operands)
            and not (torch.is_grad_enabled() and any(t.requires_grad for t in operands))
        ):
            _calls["linear"] = _calls.get("linear", 0) + 1
            output = torch.nn.functional.linear(x.view(-1, x.shape[-1]), weight, bias)
            return output.view(*x.shape[:-1], weight.shape[0])
        return original(method, layer, x, bias)

    if "linear" in selected:
        HookRegistry.register(
            "sglang.srt.layers.quantization.unquant.UnquantizedLinearMethod.apply",
            linear,
            HookType.AROUND,
        )

        def batched(original, layer, input):
            weight = layer.weight
            if (
                input.is_cuda
                and input.shape == (2, 1, 128)
                and weight.shape == (2, 2048, 128)
                and input.dtype == weight.dtype == torch.bfloat16
                and input.is_contiguous()
                and weight.is_contiguous()
                and weight.device == input.device
                and not torch.is_grad_enabled()
                and not torch._dynamo.is_compiling()
                and not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
                and hopper_supported(input.device)
            ):
                from camblas._hopper import bmm

                output = bmm(input, weight)
                _calls["bmm"] = _calls.get("bmm", 0) + 1
                return output
            return original(layer, input)

        HookRegistry.register(
            "sglang.srt.layers.linear.ColumnParallelBatchedLinear.forward",
            batched,
            HookType.AROUND,
        )

    if "fp8" in selected:

        def fp8_moe_fused(
            original,
            hidden_states,
            w1,
            w2,
            topk_output,
            moe_runner_config,
            *args,
            **kwargs,
        ):
            cfg = moe_runner_config
            ids, routing = topk_output.topk_ids, topk_output.topk_weights
            scales = (kwargs.get("w1_scale"), kwargs.get("w2_scale"))
            if (
                not args
                and not torch._dynamo.is_compiling()
                and hidden_states.is_cuda
                and hidden_states.dtype == torch.bfloat16
                and hidden_states.shape == (1, 4096)
                and w1.shape == (289, 1024, 4096)
                and w2.shape == (289, 4096, 512)
                and w1.dtype == w2.dtype == torch.float8_e4m3fn
                and cfg.activation == "silu"
                and cfg.is_gated
                and not cfg.no_combine
                and not cfg.apply_router_weight_on_input
                and cfg.swiglu_limit == 10
                and cfg.gemm1_alpha is None
                and cfg.gemm1_clamp_limit is None
                and cfg.num_experts == cfg.num_local_experts == 289
                and kwargs.get("use_fp8_w8a8")
                and kwargs.get("block_shape") == [128, 128]
                and not any(
                    kwargs.get(k, False)
                    for k in (
                        "use_int8_w8a8",
                        "use_int8_w8a16",
                        "use_int4_w4a16",
                        "per_channel_quant",
                        "fuse_swiglu_interleaved",
                    )
                )
                and all(
                    kwargs.get(k) is None
                    for k in (
                        "b1",
                        "b2",
                        "w1_zp",
                        "w2_zp",
                        "a1_scale",
                        "a2_scale",
                        "a1_q",
                    )
                )
                and ids.dtype == torch.int32
                and ids.shape == (1, 9)
                and routing.dtype == torch.float32
                and routing.shape == (1, 9)
                and all(t is not None and t.dtype == torch.float32 for t in scales)
                and scales[0].shape == (289, 8, 32)
                and scales[1].shape == (289, 32, 4)
                and all(
                    t.is_contiguous() and t.device == hidden_states.device
                    for t in (hidden_states, w1, w2, ids, routing, *scales)
                )
                and not torch.is_grad_enabled()
                and hopper_supported(hidden_states.device)
            ):
                from camblas._hopper import fused

                output = (
                    hidden_states if cfg.inplace else torch.empty_like(hidden_states)
                )
                factor = (
                    1.0
                    if cfg.routed_scaling_factor is None
                    else cfg.routed_scaling_factor
                )
                fused(hidden_states, w1, w2, *scales, ids, routing, output, factor)
                _calls["fp8"] = _calls.get("fp8", 0) + 1
                _calls["fp8_moe"] = _calls.get("fp8_moe", 0) + 1
                return output
            return original(
                hidden_states, w1, w2, topk_output, moe_runner_config, *args, **kwargs
            )

        HookRegistry.register(
            "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe.fused_experts",
            fp8_moe_fused,
            HookType.AROUND,
        )

        dense_shapes = {
            (2048, 4096),
            (4096, 1536),
            (4096, 4096),
            (6144, 4096),
            (4096, 3072),
        }

        def dense(
            original,
            input,
            weight,
            block_size,
            weight_scale,
            input_scale=None,
            bias=None,
        ):
            if (
                input.is_cuda
                and input.ndim >= 2
                and tuple(weight.shape) in dense_shapes
                and input.dtype == torch.bfloat16
                and input.numel() == weight.shape[1]
                and weight.dtype == torch.float8_e4m3fn
                and list(block_size) == [128, 128]
                and weight_scale.dtype == torch.float32
                and weight_scale.shape
                == (weight.shape[0] // 128, weight.shape[1] // 128)
                and input_scale is None
                and bias is None
                and all(
                    t.is_contiguous() and t.device == input.device
                    for t in (input, weight, weight_scale)
                )
                and not torch.is_grad_enabled()
                and not torch._dynamo.is_compiling()
                and hopper_supported(input.device)
            ):
                from camblas._hopper import product

                output = torch.empty(
                    (1, weight.shape[0]), device=input.device, dtype=torch.bfloat16
                )
                product(
                    input.view(1, -1),
                    weight.unsqueeze(0),
                    None,
                    weight_scale.unsqueeze(0),
                    None,
                    None,
                    output,
                    False,
                )
                _calls["fp8"] = _calls.get("fp8", 0) + 1
                _calls["fp8_dense_block128"] = _calls.get("fp8_dense_block128", 0) + 1
                return output.view(*input.shape[:-1], weight.shape[0])
            return original(input, weight, block_size, weight_scale, input_scale, bias)

        HookRegistry.register(
            "sglang.srt.layers.quantization.fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback",
            dense,
            HookType.AROUND,
        )

        fused_shapes = {
            (1152, 5120),
            (1792, 5120),
            (4096, 1280),
            (5120, 2048),
            (5120, 576),
            (8192, 1280),
        }
        split_shapes = fused_shapes - {(5120, 576)}
        fused_split_shapes = fused_shapes
        shapes = fused_shapes | {(25600, 6144)}

        @functools.lru_cache
        def supported(device):
            return cb.fp8_decode_supported(device)

        def fp8_product(original, A, B, As, Bs, block_size, output_dtype=torch.float16):
            operands = (A, B, As, Bs)
            if (
                not torch._dynamo.is_compiling()
                and A.is_cuda
                and A.ndim >= 2
                and B.ndim == 2
                and As.ndim == A.ndim
                and Bs.ndim == 2
                and A.numel() == A.shape[-1]
                and tuple(B.shape) in shapes
                and list(block_size) == [32, 32]
                and output_dtype == torch.bfloat16
                and A.dtype == B.dtype == torch.float8_e4m3fn
                and As.dtype == Bs.dtype == torch.float32
                and A.is_contiguous()
                and B.is_contiguous()
                and A.data_ptr() % 4 == B.data_ptr() % 4 == 0
                and all(t.device == A.device for t in operands)
                and all(
                    0 < stride <= 2147483647 for stride in (As.stride(-1), *Bs.stride())
                )
                and not (
                    torch.is_grad_enabled() and any(t.requires_grad for t in operands)
                )
                and supported(A.get_device())
            ):
                _calls["fp8"] = _calls.get("fp8", 0) + 1
                name = f"fp8_decode:{B.shape[0]},{B.shape[1]}"
                _calls[name] = _calls.get(name, 0) + 1
                if tuple(B.shape) in split_shapes:
                    _calls["fp8_split"] = _calls.get("fp8_split", 0) + 1
                return cb.fp8_decode(A, As, B, Bs)
            return original(A, B, As, Bs, block_size, output_dtype)

        HookRegistry.register(
            "sglang.kernels.ops.quantization.fp8_kernel.w8a8_block_fp8_matmul_triton",
            fp8_product,
            HookType.AROUND,
        )

        def fp8_linear(
            original,
            input,
            weight,
            block_size,
            weight_scale,
            input_scale=None,
            bias=None,
            act_scale_ue8m0=False,
            weight_bf16=None,
        ):
            operands = (input, weight, weight_scale)
            if (
                not torch._dynamo.is_compiling()
                and input.is_cuda
                and input.ndim >= 2
                and weight.ndim == 2
                and weight_scale.ndim == 2
                and input.numel() == input.shape[-1]
                and tuple(weight.shape) in shapes
                and (
                    tuple(weight.shape) in fused_shapes
                    or hopper_supported(input.device)
                )
                and list(block_size) == [32, 32]
                and act_scale_ue8m0
                and input_scale is None
                and bias is None
                and input.dtype == torch.bfloat16
                and weight.dtype == torch.float8_e4m3fn
                and weight_scale.dtype == torch.float32
                and input.is_contiguous()
                and weight.is_contiguous()
                and input.data_ptr() % 4 == weight.data_ptr() % 4 == 0
                and all(t.device == input.device for t in operands)
                and all(0 < stride <= 2147483647 for stride in weight_scale.stride())
                and not (
                    torch.is_grad_enabled() and any(t.requires_grad for t in operands)
                )
                and supported(input.get_device())
            ):
                _calls["fp8"] = _calls.get("fp8", 0) + 1
                _calls["fp8_linear"] = _calls.get("fp8_linear", 0) + 1
                name = f"fp8_linear:{weight.shape[0]},{weight.shape[1]}"
                _calls[name] = _calls.get(name, 0) + 1
                if tuple(weight.shape) in fused_split_shapes:
                    _calls["fp8_split"] = _calls.get("fp8_split", 0) + 1
                if (
                    hopper_supported(input.device)
                    and weight.data_ptr() % 16 == 0
                    and weight_scale.shape[0] >= weight.shape[0] // 32
                    and weight_scale.shape[1] >= weight.shape[1] // 32
                ):
                    from camblas._hopper import block32

                    output = block32(input.view(1, -1), weight, weight_scale)
                    _calls["fp8_tma"] = _calls.get("fp8_tma", 0) + 1
                    return output.view(*input.shape[:-1], weight.shape[0])
                return cb.fp8_linear(input, weight, weight_scale)
            return original(
                input,
                weight,
                block_size,
                weight_scale,
                input_scale,
                bias,
                act_scale_ue8m0,
                weight_bf16,
            )

        HookRegistry.register(
            "sglang.srt.layers.quantization.fp8_utils.triton_w8a8_block_fp8_linear",
            fp8_linear,
            HookType.AROUND,
        )
