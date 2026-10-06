"""Private original-precision Hopper kernels for SGLang inference."""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia import hopper
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor


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
    SF=None,
    QUANT: gl.constexpr = False,
    ACTIVATE: gl.constexpr = False,
):
    """Store each unscaled block128 Tensor Core contribution.

    Parameters
    ----------
    X, W : pointer
        Contiguous inputs and E4M3 weights in [expert, N, K] order.
    IDS : pointer or None
        Selected local expert indices; unused when CHOICES is one.
    OUT : pointer
        FP32 destination in [K / 128, CHOICES, N] order.
    N, K, CHOICES, R : constexpr int
        Output width, input width, expert choices and output rows per CTA.
    DOWN : constexpr bool
        Select a separate input row for each expert when true.
    SF : pointer or None
        FP32 [input rows, K / 128] scales written when QUANT is true.
    QUANT, ACTIVATE : constexpr bool
        Quantise BF16 input; optionally apply the original clipped SiLU and
        BF16 rounding before quantisation.

    Notes
    -----
    K must be divisible by 128. Store unscaled FP32 partial products; apply
    scales and ordered accumulation in the reduction kernel.
    """
    choice = gl.program_id(2)
    group = gl.program_id(1)
    expert = 0 if CHOICES == 1 else gl.load(IDS + choice).to(gl.int64)
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
    if ACTIVATE:
        gate = gl.load(X + xrow * (2 * K) + group * 128 + pk).to(gl.float32)
        up = gl.load(X + xrow * (2 * K) + K + group * 128 + pk).to(gl.float32)
        gate = gl.minimum(gate, 10.0)
        up = gl.minimum(gl.maximum(up, -10.0), 10.0)
        value = gl.inline_asm_elementwise(
            "{ .reg .f32 e,t; mul.ftz.f32 e,$1,0fBFB8AA3B; ex2.approx.ftz.f32 e,e; add.ftz.f32 t,e,0f3F800000; div.approx.ftz.f32 $0,$1,t; }",
            constraints="=f,f",
            args=[gate],
            dtype=gl.float32,
            is_pure=True,
            pack=1,
        )
        x = (value * up).to(gl.bfloat16)
    else:
        x = gl.load(X + xrow * K + group * 128 + pk)
    if QUANT:
        value = x.to(gl.float32)
        maximum = gl.maximum(gl.max(gl.abs(value), 0), 1e-10)
        scale = maximum * (1.0 / 448.0)
        inv = gl.inline_asm_elementwise(
            "div.approx.ftz.f32 $0,0f43E00000,$1;",
            constraints="=f,f",
            args=[maximum],
            dtype=gl.float32,
            is_pure=True,
            pack=1,
        )
        x = gl.minimum(gl.maximum(value * inv, -448.0), 448.0).to(gl.float8e4nv)
        if ACTIVATE:
            gl.store(SF + xrow * (K // 128) + group, scale, mask=gl.program_id(0) == 0)
        else:
            gl.store(SF + group, scale, mask=(gl.program_id(0) == 0) & (choice == 0))
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
    """Apply the original scale products and ordered FP32 fused additions.

    Parameters
    ----------
    DOT : pointer
        FP32 partial products in [G, CHOICES, N] order.
    AS, WS : pointer
        FP32 scales in [input rows, G] and [expert, ceil(N / 128), G] order.
    IDS, ROUTING : pointer or None
        Local expert indices and router weights; dense products omit both.
    OUT : pointer
        BF16 destination in [CHOICES, N] order.
    N, G, CHOICES, C : constexpr int
        Output width, input groups, expert choices and outputs per CTA.
    DOWN : constexpr bool
        Use per-expert input scales and apply routing after accumulation.

    Notes
    -----
    Accumulate input groups in their original order, then round once to BF16.
    """
    choice = tl.program_id(1)
    cols = tl.program_id(0) * C + tl.arange(0, C)
    expert = 0 if CHOICES == 1 else tl.load(IDS + choice).to(tl.int64)
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
    """Write an inference-only block128 FP8 dense or expert product.

    Parameters
    ----------
    input : torch.Tensor
        Contiguous BF16 or E4M3 input, with one row for up or one per expert for down.
    weight : torch.Tensor
        Contiguous E4M3 weights in [experts, outputs, inputs] order.
    input_scale : torch.Tensor or None
        Contiguous FP32 input block scales; BF16 input permits None and uses the original dynamic quantisation.
    weight_scale : torch.Tensor
        Contiguous FP32 checkpoint scales with shape [experts, ceil(outputs / 128), inputs / 128].
    expert_ids, routing : torch.Tensor or None
        Local expert indices and FP32 router weights; dense products omit both.
    output : torch.Tensor
        Contiguous BF16 destination with one row per selected expert, or one for a dense product.
    down : bool
        Apply router weights after the ordered FP32 accumulation when true.
    """
    _, n, k = weight.shape
    choices = 1 if expert_ids is None else expert_ids.numel()
    rows = 64
    columns = 256 if down else 32
    quant_input = not down and input.dtype == torch.bfloat16
    if quant_input:
        input_scale = torch.empty(
            (1, k // 128), device=input.device, dtype=torch.float32
        )
    reduce_warps = 4 if down else 1
    scratch = torch.empty(
        (k // 128, choices, n), device=input.device, dtype=torch.float32
    )
    with torch.cuda.device(input.device):
        _products[(triton.cdiv(n, rows), k // 128, choices)](
            input,
            weight,
            expert_ids,
            scratch,
            n,
            k,
            choices,
            down,
            rows,
            input_scale,
            quant_input,
            num_warps=4,
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
    """Apply ordered block scales, BF16 rounding and expert addition.

    Parameters
    ----------
    DOT : pointer
        FP32 partial products in [G, CHOICES, N] order.
    AS, WS : pointer
        FP32 scales in [CHOICES, G] and [expert, N / 128, G] order.
    IDS, ROUTING : pointer
        Local expert indices and original FP32 router weights.
    OUT : pointer
        BF16 destination with N elements.
    N, G, CHOICES, C : constexpr int
        Output width, input groups, at most 16 choices and outputs per CTA.
    FACTOR : constexpr float
        Original scaling factor applied after the expert sum.

    Notes
    -----
    N must be divisible by 128. Round each routed expert to BF16 before its
    ordered FP32 addition; round the scaled total to BF16.
    """
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
    choices = 1 if expert_ids is None else expert_ids.numel()
    _, n, k = weight_up.shape
    h = n // 2
    with torch.cuda.device(input.device):
        quantised, scales = input, None
        up = torch.empty((choices, n), device=input.device, dtype=torch.bfloat16)
        product(quantised, weight_up, scales, scale_up, expert_ids, routing, up, False)
        scales = torch.empty(
            (choices, h // 128), device=input.device, dtype=torch.float32
        )
        _, n, k = weight_down.shape
        scratch = torch.empty(
            (k // 128, choices, n), device=input.device, dtype=torch.float32
        )
        _products[(triton.cdiv(n, 64), k // 128, choices)](
            up,
            weight_down,
            expert_ids,
            scratch,
            n,
            k,
            choices,
            True,
            64,
            scales,
            True,
            True,
            num_warps=4,
            enable_fp_fusion=False,
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


@gluon.jit
def _bmm(X, W, OUT):
    """Write the original BF16 batched product with FP32 accumulation.

    Parameters
    ----------
    X, W : pointer
        Contiguous BF16 inputs in [batch, 128] and [batch, 128, 2048] order.
    OUT : pointer
        Contiguous BF16 destination in [batch, 2048] order.

    Notes
    -----
    Each CTA writes 64 outputs for one batch, using the original 128-term
    FP32 Tensor Core product before BF16 rounding.
    """
    N: gl.constexpr = 2048
    R: gl.constexpr = 64
    batch = gl.program_id(1)
    input_layout: gl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [1, 4], [0, 1])
    pk = gl.arange(0, 128, layout=gl.SliceLayout(1, input_layout))
    weight_layout: gl.constexpr = gl.BlockedLayout([1, 16], [4, 8], [4, 1], [1, 0])
    wr = gl.program_id(0) * R + gl.arange(0, R, layout=gl.SliceLayout(1, weight_layout))
    wk = gl.arange(0, 128, layout=gl.SliceLayout(0, weight_layout))
    mma_layout: gl.constexpr = gl.NVMMADistributedLayout([3, 0], [4, 1], [16, 8, 16])
    zero = gl.full((R, 8), 0, gl.float32, layout=mma_layout)
    cr = gl.program_id(0) * R + gl.arange(0, R, layout=gl.SliceLayout(1, mma_layout))
    cc = gl.arange(0, 8, layout=gl.SliceLayout(0, mma_layout))
    x = gl.load(X + batch * 128 + pk)
    duplicated = gl.where(
        gl.full((128, 8), True, gl.int1, layout=input_layout),
        x[:, None],
        gl.full((128, 8), 0.0, gl.bfloat16, layout=input_layout),
    )
    b = gl.allocate_shared_memory(
        gl.bfloat16,
        (128, 8),
        gl.NVMMASharedLayout(32, 16, rank=2, transposed=True),
        duplicated,
    )
    w = gl.load(
        W + batch * (N * 128) + wr[:, None] * 128 + wk[None, :],
        mask=wr[:, None] < N,
        other=0.0,
    )
    a = gl.convert_layout(w, gl.DotOperandLayout(0, mma_layout, 2))
    gl.barrier()
    hopper.fence_async_shared()
    dot = hopper.warpgroup_mma(a, b, zero, use_acc=False, is_async=True)
    dot = hopper.warpgroup_mma_wait(0, deps=[dot])
    gl.store(
        OUT + batch * N + cr[:, None] + cc[None, :] * 0,
        dot.to(gl.bfloat16),
        mask=(cr[:, None] < N) & (cc[None, :] == 0),
    )


def bmm(input, weight):
    """Return the original BF16 product for the selected batched projection.

    Parameters
    ----------
    input : torch.Tensor
        Contiguous BF16 input in [2, 1, 128] order.
    weight : torch.Tensor
        Contiguous BF16 weights in [2, 2048, 128] order.

    Returns
    -------
        torch.Tensor
            New BF16 output in [2, 1, 2048] order.
    """
    output = torch.empty((2, 1, 2048), device=input.device, dtype=torch.bfloat16)
    with torch.cuda.device(input.device):
        _bmm[(32, 2)](input, weight, output, num_warps=4)
    return output


@triton.jit
def _reduce32(
    PART,
    SCALE,
    Y,
    N: tl.constexpr,
    G: tl.constexpr,
    AS: tl.constexpr,
    B: tl.constexpr,
    U: tl.constexpr,
):
    """Apply the original ordered FP32 fused additions.

    Parameters
    ----------
    PART : pointer
        FP32 weight-scaled partial products in [G, N] order.
    SCALE : pointer
        FP32 power-of-two activation scales with stride AS.
    Y : pointer
        BF16 destination with N elements.
    N, G, AS, B, U : constexpr int
        Output width, group count, scale stride, outputs per CTA and unroll size.

    Notes
    -----
    Mask the final partial unroll. Apply one FP32 fused addition per group
    in increasing group order before rounding the final value to BF16.
    """
    rows = tl.program_id(0) * B + tl.arange(0, B)
    acc = tl.full((B,), 0, tl.float32)
    for start in range(0, G, U):
        for j in tl.static_range(U):
            group = start + j
            value = tl.load(
                PART + group * N + rows, mask=(rows < N) & (group < G), other=0.0
            )
            scale = tl.load(SCALE + group * AS, mask=group < G, other=0.0)
            acc = tl.fma(value, scale, acc)
    tl.store(Y + rows, acc, mask=rows < N)


@gluon.jit
def _input32(X, SCALE, group, K: gl.constexpr):
    """Preserve the original power-of-two activation scales.

    Parameters
    ----------
    X, SCALE : pointer
        Contiguous BF16 input and FP32 scale destination with K / 32 elements.
    group : int
        Index of the original 32-element quantisation group.
    K : constexpr int
        Input width, divisible by 32.

    Returns
    -------
    shared_memory_descriptor
        E4M3 [32, 8] input tile, with the same vector in all eight columns.

    Notes
    -----
    Round the scale upwards to a power of two. Only output CTA zero writes
    the scale for each valid group; mask padding beyond the last group.
    """
    input_layout: gl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [1, 4], [0, 1])
    pk = gl.arange(0, 32, layout=gl.SliceLayout(1, input_layout))
    xf = gl.load(X + group * 32 + pk, mask=group < K // 32, other=0.0).to(gl.float32)
    maximum = gl.maximum(gl.max(gl.abs(xf), 0), 1.0e-10)
    magnitude = (maximum * (1.0 / 448.0)).to(gl.int32, bitcast=True)
    exponent = (
        ((magnitude >> 23) & 255) - 127 + gl.where((magnitude & 0x7FFFFF) != 0, 1, 0)
    )
    scale = ((exponent + 127) << 23).to(gl.float32, bitcast=True)
    reciprocal = ((127 - exponent) << 23).to(gl.float32, bitcast=True)
    quant = gl.minimum(xf * reciprocal, 448.0).to(gl.float8e4nv)
    duplicated = gl.where(
        gl.full((32, 8), True, gl.int1, layout=input_layout),
        quant[:, None],
        gl.full((32, 8), 0.0, gl.float8e4nv, layout=input_layout),
    )
    b = gl.allocate_shared_memory(
        gl.float8e4nv,
        (32, 8),
        gl.NVMMASharedLayout(32, 8, rank=2, transposed=True),
        duplicated,
    )
    if gl.program_id(0) == 0:
        gl.store(SCALE + group, scale, mask=group < K // 32)
    return b


@gluon.jit
def _products32(
    X,
    WD,
    WS,
    OUT,
    SCALE,
    N: gl.constexpr,
    K: gl.constexpr,
    SK: gl.constexpr,
    SN: gl.constexpr,
    R: gl.constexpr,
    P: gl.constexpr,
):
    """Load four weight groups together and preserve each FP32 scale product.

    Parameters
    ----------
    X : pointer
        Contiguous BF16 input with K elements.
    WD : TensorDescriptor
        Contiguous E4M3 [N, K] weights, with an [R, P * 32] TMA tile.
    WS : pointer
        FP32 scales for the original 32 by 32 weight blocks.
    OUT, SCALE : pointer
        FP32 destinations for [K / 32, N] partials and K / 32 input scales.
    N, K, SK, SN, R, P : constexpr int
        Output width, input width, weight-scale strides, rows per CTA and
        input groups per CTA.

    Notes
    -----
    K must be divisible by 32. TMA loads may include masked padding.
    Round each Tensor Core product times its weight scale to FP32 before
    the separate ordered activation-scale reduction.
    """
    a = gl.allocate_shared_memory(WD.dtype, WD.block_type.shape, WD.layout)
    bar = mbarrier.allocate_mbarrier()
    mbarrier.init(bar, count=1)
    mbarrier.expect(bar, WD.block_type.nbytes)
    tma.async_copy_global_to_shared(
        WD, [gl.program_id(0) * R, gl.program_id(1) * P * 32], bar, a
    )
    mma: gl.constexpr = gl.NVMMADistributedLayout([3, 0], [4, 1], [16, 8, 32])
    zero = gl.full((R, 8), 0, gl.float32, layout=mma)
    cr = gl.program_id(0) * R + gl.arange(0, R, layout=gl.SliceLayout(1, mma))
    cc = gl.arange(0, 8, layout=gl.SliceLayout(0, mma))
    inputs = ()
    for p in gl.static_range(P):
        b = _input32(X, SCALE, gl.program_id(1) * P + p, K)
        inputs += (b,)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)
    gl.barrier()
    hopper.fence_async_shared()
    dots = ()
    for p in gl.static_range(P):
        dot = hopper.warpgroup_mma(
            a.slice(p * 32, 32, dim=1),
            inputs[p],
            zero,
            use_acc=False,
            max_num_imprecise_acc=32,
            is_async=True,
        )
        dots += (dot,)
    for p in gl.static_range(P):
        group = gl.program_id(1) * P + p
        dot = hopper.warpgroup_mma_wait(0, deps=[dots[p]])
        ws = gl.load(
            WS + (cr // 32) * SN + group * SK,
            mask=(cr < N) & (group < K // 32),
            other=0.0,
        )
        gl.store(
            OUT + group * N + cr[:, None] + cc[None, :] * 0,
            dot * ws[:, None],
            mask=(cr[:, None] < N) & (cc[None, :] == 0) & (group < K // 32),
        )


def block32(input, weight, weight_scale):
    """Return the original UE8M0 block32 FP8 product.

    Parameters
    ----------
    input : torch.Tensor
        Contiguous BF16 input with one row.
    weight : torch.Tensor
        Contiguous E4M3 weights in [outputs, inputs] order.
    weight_scale : torch.Tensor
        FP32 weight scales; positive padded or column-major strides are supported.

    Returns
    -------
        torch.Tensor
            New BF16 output with one row and one column per output channel.

    Notes
    -----
        The caller validates Hopper hardware, Triton 3.7 and the original UE8M0
        input scales before dispatch.
    """
    n, k = weight.shape
    rows = 64 if (n, k) in {(4096, 1280), (5120, 576)} else 128
    groups = 4
    desc = TensorDescriptor.from_tensor(
        weight, [rows, groups * 32], gl.NVMMASharedLayout(32, 8, rank=2)
    )
    scratch = torch.empty(
        (k // 32 * n + k // 32,), device=input.device, dtype=torch.float32
    )
    scales = scratch[k // 32 * n :]
    output = torch.empty((1, n), device=input.device, dtype=torch.bfloat16)
    with torch.cuda.device(input.device):
        _products32[(triton.cdiv(n, rows), triton.cdiv(k // 32, groups))](
            input,
            desc,
            weight_scale,
            scratch,
            scales,
            n,
            k,
            weight_scale.stride(1),
            weight_scale.stride(0),
            rows,
            groups,
            num_warps=4,
        )
        _reduce32[(triton.cdiv(n, 128),)](
            scratch, scales, output, n, k // 32, 1, 128, 16, num_warps=4, num_stages=1
        )
    return output
