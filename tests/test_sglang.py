"""Check the common SGLang numerical settings independently."""

import importlib.metadata
import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import camblas._kernels as cb
from camblas.sglang import canonical_moe_tokens


class SglangAlignmentTests(unittest.TestCase):
    """Preserve expert membership and padding under variable routing order."""

    def test_canonical_prefill_routing(self):
        """Different valid alignments must produce the same expert token order."""
        routed = torch.tensor([2, 0, 1, 2, 1, 0, 2, 2])
        ids = torch.tensor([5, 8, 1, 8, 4, 8, 8, 2, 7, 0, 6, 3, 123, -8, 43, 0])
        experts = torch.tensor([0, 1, 2, -1])
        original = ids.clone()
        count = torch.tensor([12])
        expected = torch.tensor([1, 5, 8, 8, 2, 4, 8, 8, 0, 3, 6, 7, 8, 8, 8, 8])
        actual = canonical_moe_tokens(ids, experts, count, 4, routed.numel())
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(ids, original))
        valid = actual < routed.numel()
        self.assertTrue(
            torch.equal(routed[actual[valid]], experts.repeat_interleave(4)[valid])
        )
        alternate = ids[
            torch.tensor([2, 0, 3, 1, 7, 4, 5, 6, 11, 10, 8, 9, 15, 14, 13, 12])
        ]
        self.assertTrue(
            torch.equal(canonical_moe_tokens(alternate, experts, count, 4, 8), expected)
        )


@unittest.skipUnless(
    importlib.util.find_spec("sglang") is not None and torch.cuda.is_available(),
    "Requires the pinned SGLang runtime and CUDA",
)
class SglangFp8Tests(unittest.TestCase):
    """Preserve block scaling and the complete BF16 product for each tuned shape."""

    @classmethod
    def setUpClass(cls):
        """Load the tile settings and disable reduced accumulation."""
        cls.settings = json.loads(
            (Path(__file__).parents[1] / "configs/sglang-gh200-fp8.json").read_text()
        )
        if torch.cuda.get_device_name() != cls.settings["device"]:
            raise unittest.SkipTest("These tile settings are specific to GH200")
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        from sglang.srt.plugins.hook_registry import HookRegistry

        plugin = next(
            entry
            for entry in importlib.metadata.entry_points(group="sglang.srt.plugins")
            if entry.name == "camblas"
        ).load()
        with (
            patch.dict("os.environ", CAMBLAS_SGLANG_OPS="linear,fp8"),
            patch.object(HookRegistry, "register") as register,
        ):
            plugin()
            assert register.call_count >= 2

    def test_hopper_dense_products(self):
        """Preserve dense FP8 quantisation and the BF16 batched projection."""
        from sglang.srt.layers.quantization.fp8_utils import (
            deepgemm_w8a8_block_fp8_linear_with_fallback,
        )

        from camblas._hopper import bmm, product

        torch.manual_seed(717)
        with torch.no_grad():
            weight = (torch.randn(2048, 4096, device="cuda") * 0.6).to(
                torch.float8_e4m3fn
            )
            scales = torch.rand((16, 32), device="cuda") * 0.02 + 0.001
            batched_weight = torch.randn(
                (2, 2048, 128), device="cuda", dtype=torch.bfloat16
            )
            for magnitude in (0.75, 80.0):
                input = (
                    torch.randn((1, 4096), device="cuda", dtype=torch.bfloat16)
                    * magnitude
                )
                scales.mul_(1.25)
                expected = deepgemm_w8a8_block_fp8_linear_with_fallback(
                    input, weight, [128, 128], scales
                )
                actual = torch.empty_like(expected)
                product(
                    input,
                    weight.unsqueeze(0),
                    None,
                    scales.unsqueeze(0),
                    None,
                    None,
                    actual,
                    False,
                )
                self.assertTrue(
                    torch.equal(actual.view(torch.int16), expected.view(torch.int16))
                )
                batched_input = (
                    torch.randn((2, 1, 128), device="cuda", dtype=torch.bfloat16)
                    * magnitude
                )
                expected = torch.bmm(batched_input, batched_weight.transpose(1, 2))
                actual = bmm(batched_input, batched_weight)
                self.assertTrue(
                    torch.equal(actual.view(torch.int16), expected.view(torch.int16))
                )

    def test_block128_moe_reduction_order(self):
        """Preserve up and routed down products when inputs and scales change."""
        import triton.language as tl
        from sglang.kernels.ops.attention.dsv4 import silu_and_mul_clamp
        from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
            invoke_fused_moe_kernel,
            moe_sum_reduce_triton,
        )
        from sglang.kernels.ops.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8,
        )
        from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
            moe_align_block_size,
        )

        from camblas._hopper import fused, product

        ids = torch.arange(9, device="cuda", dtype=torch.int32).view(1, 9)
        routing = torch.linspace(0.05, 1.0, 9, device="cuda").view(1, 9)
        sorted_ids, experts, padded = moe_align_block_size(ids, 64, 9)
        tile = dict(
            BLOCK_SIZE_M=64,
            BLOCK_SIZE_N=128,
            BLOCK_SIZE_K=128,
            GROUP_SIZE_M=32,
            num_warps=4,
            num_stages=3,
        )
        parameters = {}
        with torch.no_grad():
            for down, n, k in ((False, 1024, 4096), (True, 4096, 512)):
                torch.manual_seed(842)
                weight = (torch.randn(9, n, k, device="cuda") * 0.6).to(
                    torch.float8_e4m3fn
                )
                scales = (
                    torch.rand(9, n // 128, k // 128, device="cuda") * 0.001 + 0.001
                )
                parameters[down] = (weight, scales)
                for magnitude in (0.75, 80.0):
                    with self.subTest(down=down, magnitude=magnitude):
                        input = (
                            torch.randn(
                                9 if down else 1, k, device="cuda", dtype=torch.bfloat16
                            )
                            * magnitude
                        )
                        quant, input_scale = sglang_per_token_group_quant_fp8(
                            input, 128
                        )
                        scales.mul_(1.25)
                        expected = torch.empty(
                            9, n, device="cuda", dtype=torch.bfloat16
                        )
                        actual = torch.empty_like(expected)
                        invoke_fused_moe_kernel(
                            quant,
                            weight,
                            None,
                            expected,
                            input_scale,
                            scales,
                            None,
                            routing,
                            ids,
                            sorted_ids,
                            experts,
                            padded,
                            down,
                            1 if down else 9,
                            tile,
                            compute_type=tl.bfloat16,
                            use_fp8_w8a8=True,
                            use_int8_w8a8=False,
                            use_int8_w8a16=False,
                            use_int4_w4a16=False,
                            per_channel_quant=False,
                            block_shape=[128, 128],
                            filter_expert=False,
                        )
                        product(
                            quant,
                            weight,
                            input_scale,
                            scales,
                            ids,
                            routing,
                            actual,
                            down,
                        )
                        self.assertTrue(
                            torch.equal(
                                expected.view(torch.int16), actual.view(torch.int16)
                            )
                        )

            up_weight, up_scale = parameters[False]
            down_weight, down_scale = parameters[True]
            for magnitude in (0.75, 80.0):
                with self.subTest(fused=True, magnitude=magnitude):
                    input = (
                        torch.randn(1, 4096, device="cuda", dtype=torch.bfloat16)
                        * magnitude
                    )
                    q, sf = sglang_per_token_group_quant_fp8(input, 128)
                    up = torch.empty(9, 1024, device="cuda", dtype=torch.bfloat16)
                    down = torch.empty(9, 4096, device="cuda", dtype=torch.bfloat16)
                    common = dict(
                        compute_type=tl.bfloat16,
                        use_fp8_w8a8=True,
                        use_int8_w8a8=False,
                        use_int8_w8a16=False,
                        use_int4_w4a16=False,
                        per_channel_quant=False,
                        block_shape=[128, 128],
                        filter_expert=False,
                    )
                    invoke_fused_moe_kernel(
                        q,
                        up_weight,
                        None,
                        up,
                        sf,
                        up_scale,
                        None,
                        routing,
                        ids,
                        sorted_ids,
                        experts,
                        padded,
                        False,
                        9,
                        tile,
                        **common,
                    )
                    act = torch.empty(9, 512, device="cuda", dtype=torch.bfloat16)
                    silu_and_mul_clamp(up, act, 10.0)
                    q, sf = sglang_per_token_group_quant_fp8(act, 128)
                    invoke_fused_moe_kernel(
                        q,
                        down_weight,
                        None,
                        down,
                        sf,
                        down_scale,
                        None,
                        routing,
                        ids,
                        sorted_ids,
                        experts,
                        padded,
                        True,
                        1,
                        tile,
                        **common,
                    )
                    expected = torch.empty_like(input)
                    moe_sum_reduce_triton(down.view(1, 9, 4096), expected, 2.5)
                    actual = input.clone()
                    fused(
                        input,
                        up_weight,
                        down_weight,
                        up_scale,
                        down_scale,
                        ids,
                        routing,
                        actual,
                        2.5,
                    )
                    self.assertTrue(
                        torch.equal(
                            expected.view(torch.int16), actual.view(torch.int16)
                        )
                    )
                    fused(
                        actual.copy_(input),
                        up_weight,
                        down_weight,
                        up_scale,
                        down_scale,
                        ids,
                        routing,
                        actual,
                        2.5,
                    )
                    self.assertTrue(
                        torch.equal(
                            expected.view(torch.int16), actual.view(torch.int16)
                        )
                    )

    def test_products_and_changed_scales(self):
        """Compare each tile with upstream and selected independent FP64 outputs."""
        from sglang.kernels.ops.quantization import fp8_kernel as fp8

        baseline = dict(
            BLOCK_SIZE_M=64,
            BLOCK_SIZE_N=32,
            BLOCK_SIZE_K=32,
            GROUP_SIZE_M=32,
            num_warps=4,
            num_stages=3,
        )

        def product(a, b, a_scale, b_scale, tile):
            with patch.object(
                fp8, "get_w8a8_block_fp8_configs", return_value={1: tile}
            ):
                return fp8.w8a8_block_fp8_matmul_triton(
                    a, b, a_scale, b_scale, [32, 32], torch.bfloat16
                )

        for shape, tile in self.settings["tiles"].items():
            n, k = map(int, shape.split(","))
            for sample in range(2):
                with self.subTest(shape=shape, sample=sample):
                    torch.manual_seed(413 + sample)
                    a = (torch.randn(1, k, device="cuda") * 0.75).to(
                        torch.float8_e4m3fn
                    )
                    b = (torch.randn(n, k, device="cuda") * 0.6).to(torch.float8_e4m3fn)
                    a_scale = (
                        2.0 ** torch.randint(-3, 2, (1, k // 32), device="cuda")
                    ).float()
                    b_scale = (
                        2.0 ** torch.randint(-3, 2, (n // 32, k // 32), device="cuda")
                    ).float()
                    expected = product(a, b, a_scale, b_scale, baseline)
                    actual = product(a, b, a_scale, b_scale, tile)
                    self.assertTrue(
                        torch.equal(
                            expected.view(torch.int16), actual.view(torch.int16)
                        )
                    )

                    if cb.fp8_decode_supported():
                        if sample == 1:
                            # Serving also supplies padded/column-major FP32 scales.
                            storage = torch.empty((k // 32, 4), device="cuda")
                            a_scale = storage[:, 0].view(1, -1)
                            a_scale.uniform_(0.001, 0.08)
                            b_scale = b_scale.T.contiguous().T
                            actual = product(a, b, a_scale, b_scale, tile)
                        native = cb.fp8_decode(a, a_scale, b, b_scale)
                        self.assertTrue(
                            torch.equal(
                                native.view(torch.int16), actual.view(torch.int16)
                            )
                        )
                        actual = native

                    if cb.fp8_decode_supported():
                        bf16 = (
                            torch.randn(1, k, device="cuda") * (80 if sample else 0.75)
                        ).to(torch.bfloat16)
                        quantised, scales = fp8.sglang_per_token_group_quant_fp8(
                            bf16, 32, scale_ue8m0=True
                        )
                        fused_expected = product(quantised, b, scales, b_scale, tile)
                        fused = cb.fp8_linear(bf16, b, b_scale)
                        self.assertTrue(
                            torch.equal(
                                fused.view(torch.int16),
                                fused_expected.view(torch.int16),
                            )
                        )
                        from camblas._hopper import block32

                        fused = block32(bf16, b, b_scale)
                        self.assertTrue(
                            torch.equal(
                                fused.view(torch.int16),
                                fused_expected.view(torch.int16),
                            )
                        )
                        # Independently quantise on the CPU, including exact power-of-two scales.
                        values = bf16.float().cpu().view(1, k // 32, 32)
                        raw = values.abs().amax(-1).clamp_min(1.0e-10) * (1.0 / 448.0)
                        bits = raw.view(torch.int32).long()
                        exponent = ((bits >> 23) & 255) - 127 + ((bits & 0x7FFFFF) != 0)
                        cpu_scales = torch.ldexp(torch.ones_like(raw), exponent.int())
                        cpu_q = (
                            (values / cpu_scales.unsqueeze(-1))
                            .reshape(1, k)
                            .to(torch.float8_e4m3fn)
                        )
                        self.assertTrue(
                            torch.equal(
                                cpu_q.view(torch.uint8),
                                quantised.cpu().view(torch.uint8),
                            )
                        )
                        self.assertTrue(
                            torch.equal(
                                cpu_scales.view(torch.int32),
                                scales.cpu().view(torch.int32),
                            )
                        )
                        # The independent product below checks the fused result as well.
                        a, a_scale, actual = (
                            cpu_q.to(b.device),
                            cpu_scales.to(b.device),
                            fused,
                        )

                    # Reconstruct selected outputs independently in FP64.
                    indices = torch.arange(0, n, max(1, n // 16), device="cuda")[:16]
                    decoded_a = (
                        a.float().cpu().double()
                        * a_scale.cpu().double().repeat_interleave(32, dim=1)
                    )
                    decoded_b = b.index_select(0, indices).float().cpu().double()
                    decoded_b *= (
                        b_scale.index_select(0, indices // 32)
                        .cpu()
                        .double()
                        .repeat_interleave(32, dim=1)
                    )
                    reference = decoded_a @ decoded_b.T
                    received = actual.index_select(1, indices).float().cpu()
                    rounding_unit = reference.abs().clamp_min(1.0) / 128
                    self.assertTrue(
                        torch.all((reference - received).abs() <= rounding_unit)
                    )


if __name__ == "__main__":
    unittest.main()
