"""Private block128 FP8 MoE products for single-token GH200 inference."""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia import hopper


@gluon.jit
def _products(
    X,
    W,
    IDS,
    OUT,
    N: gl.constexpr,
    K: gl.constexpr,
    CHOICES: gl.constexpr,
    DOWN: gl.constexpr,
    R: gl.constexpr,
):
    """Store each unscaled block128 Tensor Core contribution."""
    choice = gl.program_id(2)
    group = gl.program_id(1)
    expert = gl.load(IDS + choice).to(gl.int64)
    input_layout: gl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [1, 4], [0, 1])
    pk = gl.arange(0, 128, layout=gl.SliceLayout(1, input_layout))
    weight_layout: gl.constexpr = gl.BlockedLayout([1, 8], [2, 16], [4, 1], [1, 0])
    wr = gl.program_id(0) * R + gl.arange(0, R, layout=gl.SliceLayout(1, weight_layout))
    wk = gl.arange(0, 128, layout=gl.SliceLayout(0, weight_layout))
    mma_layout: gl.constexpr = gl.NVMMADistributedLayout([3, 0], [4, 1], [16, 8, 32])
    zero = gl.full((R, 8), 0, gl.float32, layout=mma_layout)
    cr = gl.program_id(0) * R + gl.arange(0, R, layout=gl.SliceLayout(1, mma_layout))
    cc = gl.arange(0, 8, layout=gl.SliceLayout(0, mma_layout))
    xrow = choice if DOWN else 0
    x = gl.load(X + xrow * K + group * 128 + pk)
    duplicated = gl.where(
        gl.full((128, 8), True, gl.int1, layout=input_layout),
        x[:, None],
        gl.full((128, 8), 0.0, gl.float8e4nv, layout=input_layout),
    )
    b = gl.allocate_shared_memory(
        gl.float8e4nv,
        (128, 8),
        gl.NVMMASharedLayout(32, 8, rank=2, transposed=True),
        duplicated,
    )
    w = gl.load(
        W + expert * (N * K) + wr[:, None] * K + group * 128 + wk[None, :],
        mask=wr[:, None] < N,
        other=0.0,
    )
    a = gl.convert_layout(w, gl.DotOperandLayout(0, mma_layout, 4))
    gl.barrier()
    hopper.fence_async_shared()
    dot = hopper.warpgroup_mma(
        a, b, zero, use_acc=False, max_num_imprecise_acc=128, is_async=True
    )
    dot = hopper.warpgroup_mma_wait(0, deps=[dot])
    gl.store(
        OUT + (group * CHOICES + choice) * N + cr[:, None] + cc[None, :] * 0,
        dot,
        mask=(cr[:, None] < N) & (cc[None, :] == 0),
    )


@triton.jit
def _reduce(
    DOT,
    AS,
    WS,
    IDS,
    ROUTING,
    OUT,
    N: tl.constexpr,
    G: tl.constexpr,
    CHOICES: tl.constexpr,
    DOWN: tl.constexpr,
    C: tl.constexpr,
):
    """Apply the original scale products and ordered FP32 fused additions."""
    choice = tl.program_id(1)
    cols = tl.program_id(0) * C + tl.arange(0, C)
    expert = tl.load(IDS + choice).to(tl.int64)
    xrow = choice if DOWN else 0
    acc = tl.full((C,), 0, tl.float32)
    for group in tl.range(0, G, loop_unroll_factor=2):
        dot = tl.load(
            DOT + (group * CHOICES + choice) * N + cols, mask=cols < N, other=0.0
        )
        a = tl.load(AS + xrow * G + group)
        w = tl.load(
            WS + expert * (tl.cdiv(N, 128) * G) + (cols // 128) * G + group,
            mask=cols < N,
            other=0.0,
        )
        acc = tl.fma(dot, a * w, acc)
    if DOWN:
        acc *= tl.load(ROUTING + choice)
    tl.store(OUT + choice * N + cols, acc.to(tl.bfloat16), mask=cols < N)


def product(
    input, weight, input_scale, weight_scale, expert_ids, routing, output, down
):
    """Write an inference-only block128 FP8 expert product.

    Parameters
    ----------
    input : torch.Tensor
        Contiguous E4M3 input, with one row for up or nine rows for down.
    weight : torch.Tensor
        Contiguous E4M3 expert weights in [experts, outputs, inputs] order.
    input_scale, weight_scale : torch.Tensor
        Contiguous FP32 dequantisation scales for the original 128-element blocks.
    expert_ids, routing : torch.Tensor
        Nine local expert indices and FP32 router weights on the input device.
    output : torch.Tensor
        Contiguous BF16 destination with nine output rows.
    down : bool
        Apply router weights after the ordered FP32 accumulation when true.
    """
    _, n, k = weight.shape
    choices = expert_ids.numel()
    rows = 64
    columns = 256 if down else 32
    reduce_warps = 4 if down else 1
    scratch = torch.empty(
        (k // 128, choices, n), device=input.device, dtype=torch.float32
    )
    with torch.cuda.device(input.device):
        _products[(triton.cdiv(n, rows), k // 128, choices)](
            input, weight, expert_ids, scratch, n, k, choices, down, rows, num_warps=4
        )
        _reduce[(triton.cdiv(n, columns), choices)](
            scratch,
            input_scale,
            weight_scale,
            expert_ids,
            routing,
            output,
            n,
            k // 128,
            choices,
            down,
            columns,
            num_warps=reduce_warps,
            num_stages=1,
        )


@triton.jit
def _activate_quant(
    IN,
    ACT,
    Q,
    SF,
    H: tl.constexpr,
    G: tl.constexpr,
    ROWS: tl.constexpr,
    GPB: tl.constexpr,
):
    """Preserve the clamp, BF16 rounding and block128 quantisation."""
    group = tl.program_id(0) * GPB + tl.arange(0, GPB)
    row = group // G
    block = group % G
    k = block[:, None] * 128 + tl.arange(0, 128)[None, :]
    gate = tl.load(
        IN + row[:, None] * (2 * H) + k, mask=row[:, None] < ROWS, other=0.0
    ).to(tl.float32)
    up = tl.load(
        IN + row[:, None] * (2 * H) + H + k, mask=row[:, None] < ROWS, other=0.0
    ).to(tl.float32)
    gate = tl.minimum(gate, 10.0)
    up = tl.minimum(tl.maximum(up, -10.0), 10.0)
    value = (
        tl.inline_asm_elementwise(
            "{ .reg .f32 e,t; mul.ftz.f32 e,$1,0fBFB8AA3B; ex2.approx.ftz.f32 e,e; add.ftz.f32 t,e,0f3F800000; div.approx.ftz.f32 $0,$1,t; }",
            constraints="=f,f",
            args=[gate],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        * up
    )
    rounded = value.to(tl.bfloat16)
    value = rounded.to(tl.float32)
    maximum = tl.maximum(tl.max(tl.abs(value), 1), 1e-10)
    scale = maximum * (1.0 / 448.0)
    inv = tl.inline_asm_elementwise(
        "div.approx.ftz.f32 $0,0f43E00000,$1;",
        constraints="=f,f",
        args=[maximum],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    quant = tl.minimum(tl.maximum(value * inv[:, None], -448.0), 448.0).to(
        tl.float8e4nv
    )
    tl.store(ACT + row[:, None] * H + k, rounded, mask=row[:, None] < ROWS)
    tl.store(Q + row[:, None] * H + k, quant, mask=row[:, None] < ROWS)
    tl.store(SF + group, scale, mask=row < ROWS)


@triton.jit
def _combine(
    DOT,
    AS,
    WS,
    IDS,
    ROUTING,
    OUT,
    N: tl.constexpr,
    G: tl.constexpr,
    CHOICES: tl.constexpr,
    FACTOR: tl.constexpr,
    C: tl.constexpr,
):
    """Apply ordered block scales, BF16 rounding and expert addition."""
    choices = tl.arange(0, 16)
    cols = tl.program_id(0) * C + tl.arange(0, C)
    expert = tl.load(IDS + choices, mask=choices < CHOICES, other=0).to(tl.int64)
    acc = tl.full((16, C), 0, tl.float32)
    for group in tl.static_range(G):
        dot = tl.load(
            DOT + (group * CHOICES + choices[:, None]) * N + cols[None, :],
            mask=(choices[:, None] < CHOICES) & (cols[None, :] < N),
            other=0.0,
        )
        a = tl.load(AS + choices * G + group, mask=choices < CHOICES, other=0.0)
        w = tl.load(
            WS + expert[:, None] * (N // 128 * G) + (cols[None, :] // 128) * G + group,
            mask=(choices[:, None] < CHOICES) & (cols[None, :] < N),
            other=0.0,
        )
        acc = tl.fma(dot, a[:, None] * w, acc)
    routing = tl.load(ROUTING + choices, mask=choices < CHOICES, other=0.0)
    rounded = (acc * routing[:, None]).to(tl.bfloat16).to(tl.float32)
    total = tl.full((C,), 0, tl.float32)
    for choice in tl.static_range(CHOICES):
        value = tl.sum(tl.where(choices[:, None] == choice, rounded, 0.0), axis=0)
        total = total + value
    tl.store(OUT + cols, (total * FACTOR).to(tl.bfloat16), mask=cols < N)


def fused(
    input,
    weight_up,
    weight_down,
    scale_up,
    scale_down,
    expert_ids,
    routing,
    output,
    factor,
):
    """Write the original-precision MoE result for one token.

    Parameters
    ----------
    input : torch.Tensor
        Contiguous BF16 input with one row.
    weight_up, weight_down : torch.Tensor
        Contiguous E4M3 expert weights.
    scale_up, scale_down : torch.Tensor
        Contiguous FP32 scales for the original 128-element blocks.
    expert_ids, routing : torch.Tensor
        Nine local expert indices and FP32 router weights.
    output : torch.Tensor
        BF16 destination; it can alias the input.
    factor : float
        Original scaling factor applied after the ordered expert sum.
    """
    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    choices = expert_ids.numel()
    _, n, k = weight_up.shape
    h = n // 2
    with torch.cuda.device(input.device):
        quantised, scales = sglang_per_token_group_quant_fp8(input, 128)
        up = torch.empty((choices, n), device=input.device, dtype=torch.bfloat16)
        product(quantised, weight_up, scales, scale_up, expert_ids, routing, up, False)
        activated = torch.empty((choices, h), device=input.device, dtype=torch.bfloat16)
        quantised = torch.empty_like(activated, dtype=torch.float8_e4m3fn)
        scales = torch.empty(
            (choices, h // 128), device=input.device, dtype=torch.float32
        )
        _activate_quant[(triton.cdiv(choices * (h // 128), 4),)](
            up,
            activated,
            quantised,
            scales,
            h,
            h // 128,
            choices,
            4,
            num_warps=4,
            enable_fp_fusion=False,
        )
        _, n, k = weight_down.shape
        scratch = torch.empty(
            (k // 128, choices, n), device=input.device, dtype=torch.float32
        )
        _products[(triton.cdiv(n, 64), k // 128, choices)](
            quantised,
            weight_down,
            expert_ids,
            scratch,
            n,
            k,
            choices,
            True,
            64,
            num_warps=4,
        )
        _combine[(triton.cdiv(n, 64),)](
            scratch,
            scales,
            scale_down,
            expert_ids,
            routing,
            output,
            n,
            k // 128,
            choices,
            factor,
            64,
            num_warps=1,
            num_stages=1,
        )
