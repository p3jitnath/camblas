"""Check block-scaled multiplication against an independent CPU FP64 decoder."""

import ctypes
import unittest

import torch

import camblas._kernels as cb
from camblas import _native


def dequantize(x, scales, block, packed=False, weight=False):
    """Decode storage bytes and scale blocks independently of the CUDA kernel."""
    raw = x.cpu().view(torch.uint8)
    if packed:
        table = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
            dtype=torch.float64,
        )
        indices = torch.stack((raw & 15, raw >> 4), -1).flatten(-2).long()
        values = table[indices]
    else:
        values = raw.view(torch.float8_e4m3fn).double()
    exponents = scales.cpu().view(torch.uint8).long()
    powers = torch.ldexp(
        torch.ones_like(exponents, dtype=torch.float64), exponents - 127
    )
    powers[exponents == 255] = float("nan")
    if weight and not packed:
        powers = powers.repeat_interleave(32, 0)[: values.size(0)]
    return values * powers.repeat_interleave(block, -1)


class QuantizedTests(unittest.TestCase):
    """Cover scales, packed layout, mutations, invalid metadata and graph lifetime."""

    @classmethod
    def setUpClass(cls):
        """Require a current tensor binding and FP8 tensor-core hardware."""
        if not torch.cuda.is_available() or not hasattr(
            _native.tensor_module(), "quantized_matmul"
        ):
            raise unittest.SkipTest("Requires the current native CUDA tensor binding")
        if torch.cuda.get_device_capability()[0] < 9:
            raise unittest.SkipTest("Requires SM90 FP8 tensor cores")
        torch.set_num_threads(4)

    def operands(
        self, rows=1, outputs=8, inner=32, packed=True, block=32, device="cuda"
    ):
        """Generate finite values and non-uniform scale blocks with a fixed seed."""
        x = (torch.randn(rows, inner, device=device) * 8).to(torch.float8_e4m3fn)
        if packed:
            weight = torch.randint(
                0, 256, (outputs, inner // 2), dtype=torch.uint8, device=device
            )
            weight = weight.view(torch.float4_e2m1fn_x2)
        else:
            weight = (torch.randn(outputs, inner, device=device) * 4).to(
                torch.float8_e4m3fn
            )
        sx = torch.randint(
            122, 129, (rows, inner // block), dtype=torch.uint8, device=device
        )
        sw = torch.randint(
            121,
            128,
            (outputs if packed else (outputs + 31) // 32, inner // 32),
            dtype=torch.uint8,
            device=device,
        )
        return x, sx.view(torch.float8_e8m0fnu), weight, sw.view(torch.float8_e8m0fnu)

    def test_all_outputs_and_individual_mutations(self):
        """Each operand changes separately, preventing cancelling mutations hiding stale data."""
        torch.manual_seed(20261004)
        with torch.inference_mode():
            for packed in (False, True):
                for rows, outputs, inner, block in (
                    (1, 8, 32, 32),
                    (3, 40, 256, 128 if packed else 32),
                    (6, 40, 256, 128 if packed else 32),
                    (7, 40, 256, 128 if packed else 32),
                    (8, 40, 256, 128 if packed else 32),
                    (17, 40, 256, 128 if packed else 32),
                    (33, 40, 256, 128 if packed else 32),
                    (128, 40, 256, 128 if packed else 32),
                    (1, 2304, 5120, 32),
                    (1, 5120, 2304, 32),
                ):
                    x, sx, weight, sw = self.operands(
                        rows, outputs, inner, packed, block
                    )
                    if rows == 6:
                        x = x.reshape(2, 3, inner)
                        sx = sx.reshape(2, 3, inner // block)
                    for changed in (
                        "original",
                        "input",
                        "weight",
                        "input_scale",
                        "weight_scale",
                    ):
                        if changed == "input":
                            x.view(torch.uint8).bitwise_xor_(128)
                        elif changed == "weight":
                            weight.view(torch.uint8).bitwise_xor_(17 if packed else 128)
                        elif changed == "input_scale":
                            sx.view(torch.uint8).sub_(1)
                        elif changed == "weight_scale":
                            sw.view(torch.uint8).add_(1)
                        actual = cb.quantized_matmul(
                            x, sx, weight, sw, activation_block=block
                        )
                        expected = (
                            dequantize(x, sx, block)
                            @ dequantize(weight, sw, 32, packed=packed, weight=True).T
                        )
                        torch.testing.assert_close(
                            actual.cpu().double(), expected, rtol=0.008, atol=0.02
                        )
                        self.assertEqual(actual.dtype, torch.bfloat16)

    def test_fp8_decode_tail_graph_and_guards(self):
        """Keep tail outputs fresh on another device/stream and reject invalid batching."""
        device = torch.cuda.device_count() - 1
        if not hasattr(
            _native.tensor_module(), "fp8_decode_supported"
        ) or not cb.fp8_decode_supported(device):
            self.skipTest("Requires an SM90a build on SM90 hardware")
        for fused, n, k in (
            (False, 48, 96),
            (True, 48, 96),
            (False, 1152, 5120),
            (True, 1152, 5120),
            (True, 5120, 576),
        ):
            with self.subTest(fused=fused, outputs=n):
                dtype = torch.bfloat16 if fused else torch.float8_e4m3fn
                x = torch.zeros(1, k, device=device).to(dtype)
                weight = torch.ones(n, k, device=device).to(torch.float8_e4m3fn)
                sx = torch.ones(1, k // 32, device=device)
                sw = torch.ones((n + 31) // 32, k // 32, device=device)
                sw[-1].mul_(2)

                def product():
                    return (
                        cb.fp8_linear(x, weight, sw)
                        if fused
                        else cb.fp8_decode(x, sx, weight, sw)
                    )

                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.inference_mode(), torch.cuda.stream(stream):
                    before = product()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        output = product()
                    if fused:
                        x.fill_(0.5)
                    else:
                        x.view(torch.uint8).fill_(0x38)
                    weight.view(torch.uint8).fill_(0x40)
                    sx.mul_(0.5)
                    sw.mul_(2)
                    graph.replay()
                stream.synchronize()
                expected = torch.full((1, n), float(k * 2), dtype=torch.bfloat16)
                expected[:, ((n + 31) // 32 - 1) * 32 :] *= 2
                self.assertTrue(
                    torch.equal(
                        output.cpu().view(torch.int16), expected.view(torch.int16)
                    )
                )
                self.assertTrue((before == 0).all())
                graph.reset()
                weight.view(torch.uint8).fill_(0x7F)
                with torch.inference_mode():
                    self.assertTrue(product().isnan().all())
                if fused:
                    for operands in (
                        (x.expand(2, -1).contiguous(), weight, sw),
                        (x, weight, sw[:, :2]),
                        (x.requires_grad_(), weight, sw),
                    ):
                        with self.assertRaises(ValueError):
                            cb.fp8_linear(*operands)
                else:
                    for operands in (
                        (
                            x.expand(2, -1).contiguous(),
                            sx.expand(2, -1).contiguous(),
                            weight,
                            sw,
                        ),
                        (x, sx[:, :2], weight, sw),
                        (x, sx[:, :1].expand_as(sx), weight, sw),
                    ):
                        with self.assertRaises(ValueError):
                            cb.fp8_decode(*operands)

    def test_scale_extremes_and_nan(self):
        """Check scale extremes and NaNs across vector, tile and partial-tile paths."""
        with torch.inference_mode():
            for packed in (False, True):
                for rows in (1, 8, 17):
                    x, sx, weight, sw = self.operands(rows, packed=packed)
                    x.view(torch.uint8).fill_(0x38)
                    weight.view(torch.uint8).fill_(0x22 if packed else 0x38)
                    sw.view(torch.uint8).fill_(127)
                    for exponent in (0, 127, 254, 255):
                        sx.view(torch.uint8).fill_(exponent)
                        actual = cb.quantized_matmul(x, sx, weight, sw)
                        expected = (
                            dequantize(x, sx, 32)
                            @ dequantize(weight, sw, 32, packed=packed, weight=True).T
                        )
                        torch.testing.assert_close(
                            actual.cpu(),
                            expected.to(torch.bfloat16),
                            rtol=0,
                            atol=0,
                            equal_nan=True,
                        )
                    sx.view(torch.uint8).fill_(127)
                    x.view(torch.uint8).fill_(0x7F)
                    self.assertTrue(
                        cb.quantized_matmul(x, sx, weight, sw).isnan().all()
                    )

    def test_invalid_shapes_dtypes_views_and_devices(self):
        """Reject unsupported metadata before launching a kernel."""
        x, sx, weight, sw = self.operands()
        invalid = (
            (x.float(), sx, weight, sw),
            (x, sx.float(), weight, sw),
            (x, sx, weight.view(torch.uint8), sw),
            (x.cpu(), sx, weight, sw),
            (x, sx, weight[:7], sw[:7]),
            (x, sx[:, :0], weight, sw),
            (x, sx, weight, sw[:7]),
            (x, sx, weight, sw.float()),
        )
        for operands in invalid:
            with self.assertRaises(ValueError):
                cb.quantized_matmul(*operands)
        with self.assertRaises(ValueError):
            cb.quantized_matmul(x, sx, weight, sw, activation_block=64)
        x2, sx2, weight2, sw2 = self.operands(2, 8, 64)
        with self.assertRaises(ValueError):
            cb.quantized_matmul(x2.T, sx2, weight2, sw2)
        if torch.cuda.device_count() > 1:
            with self.assertRaises(ValueError):
                cb.quantized_matmul(x, sx.to("cuda:1"), weight, sw)
        with self.assertRaises(ValueError):
            cb.quantized_matmul(x.requires_grad_(), sx, weight, sw)

    def test_stream_graph_all_devices_and_fresh_output(self):
        """Replay captured operations after both operands change, on every GPU."""
        with torch.inference_mode():
            for device in range(torch.cuda.device_count()):
                x, sx, weight, sw = self.operands(device=device)
                x.view(torch.uint8).zero_()
                sx.view(torch.uint8).fill_(127)
                sw.view(torch.uint8).fill_(127)
                weight.view(torch.uint8).fill_(0x22)
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    before = cb.quantized_matmul(x, sx, weight, sw)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        output = cb.quantized_matmul(x, sx, weight, sw)
                    x.view(torch.uint8).fill_(0x38)
                    weight.view(torch.uint8).fill_(0x33)
                    graph.replay()
                stream.synchronize()
                torch.testing.assert_close(
                    output.cpu(),
                    torch.full((1, 8), 48, dtype=torch.bfloat16),
                    rtol=0,
                    atol=0,
                )
                self.assertTrue((before == 0).all())
                graph.reset()

    def test_grouped_products_masks_and_individual_mutations(self):
        """Decode every grouped output independently, including invalid and padded rows."""
        with torch.inference_mode():
            for packed in (False, True):
                block = 128 if packed else 32
                operands = [self.operands(17, 40, 128, packed, block) for _ in range(3)]
                weights, scales = [v[2] for v in operands], [v[3] for v in operands]
                pool = cb.QuantizedGroups(weights, scales)
                x = torch.stack([operands[i % 3][0] for i in range(4)])
                sx = torch.stack([operands[i % 3][1] for i in range(4)])
                active = torch.tensor([2, 0, -1, 5], device="cuda", dtype=torch.int32)
                counts = torch.tensor([17, 8, 12, 17], device="cuda", dtype=torch.int32)
                for changed in ("original", "input", "weight", "scale", "indices"):
                    if changed == "input":
                        x.view(torch.uint8).bitwise_xor_(128)
                    elif changed == "weight":
                        weights[0].view(torch.uint8).bitwise_xor_(17 if packed else 128)
                    elif changed == "scale":
                        sx.view(torch.uint8).sub_(1)
                        scales[2].view(torch.uint8).add_(1)
                    elif changed == "indices":
                        active.copy_(
                            torch.tensor([1, 2, 0, 1], device="cuda", dtype=torch.int32)
                        )
                        counts.copy_(
                            torch.tensor(
                                [30, -1, 17, 8], device="cuda", dtype=torch.int32
                            )
                        )
                    result = pool.matmul(x, sx, active, counts, activation_block=block)
                    for batch, (index, count) in enumerate(
                        zip(active.cpu().tolist(), counts.cpu().tolist())
                    ):
                        rows = max(0, min(count, 17)) if 0 <= index < 3 else 0
                        if rows:
                            expected = (
                                dequantize(x[batch, :rows], sx[batch, :rows], block)
                                @ dequantize(
                                    weights[index],
                                    scales[index],
                                    32,
                                    packed=packed,
                                    weight=True,
                                ).T
                            )
                            torch.testing.assert_close(
                                result[batch, :rows].cpu().double(),
                                expected,
                                rtol=0.008,
                                atol=0.02,
                            )
                            torch.testing.assert_close(
                                result[batch, :rows],
                                cb.quantized_matmul(
                                    x[batch, :rows],
                                    sx[batch, :rows],
                                    weights[index],
                                    scales[index],
                                    activation_block=block,
                                ),
                                rtol=0,
                                atol=0,
                            )
                        self.assertEqual(torch.count_nonzero(result[batch, rows:]), 0)
                with self.assertRaises(ValueError):
                    pool.matmul(x, sx, active.long(), counts, activation_block=block)
                with self.assertRaises(ValueError):
                    cb.QuantizedGroups(weights, scales[:2])
                with self.assertRaises(ValueError):
                    cb.QuantizedGroups(
                        [
                            torch.empty(
                                weights[0].shape, device="cuda", dtype=torch.float32
                            )
                        ],
                        [scales[0]],
                    )

    def test_routing_order_duplicates_truncation_and_changed_inputs(self):
        """Check slot ownership and sequential CPU sums with duplicates and invalid IDs."""
        indices = torch.tensor(
            [[5, 5, 4, 8, -1, 2**40], [4, 6, 5, 4, 8, 5]], device="cuda"
        )
        active = torch.tensor([1, 0, 1], device="cuda", dtype=torch.int32)
        with torch.inference_mode():
            for rows in (1, 8):
                for changed in (False, True):
                    if changed:
                        indices = indices.flip(1).contiguous()
                    slots, reverse, counts = cb.route_groups(
                        indices, active, first_expert=4, rows=rows
                    )
                    host_indices, host_slots, host_reverse = (
                        indices.cpu(),
                        slots.cpu(),
                        reverse.cpu(),
                    )
                    expected_counts = [0, 0, 0]
                    for slot, expert in enumerate(host_indices.flatten().tolist()):
                        group = 0 if expert == 5 else 1 if expert == 4 else -1
                        row = int(host_reverse.flatten()[slot])
                        if group >= 0:
                            expected_counts[group] += 1
                            if row >= 0:
                                self.assertEqual(row // rows, group)
                                self.assertEqual(int(host_slots.flatten()[row]), slot)
                        else:
                            self.assertEqual(row, -1)
                    self.assertEqual(counts.cpu().tolist(), expected_counts)
                    values = torch.randn(
                        3, rows, 40, device="cuda", dtype=torch.bfloat16
                    )
                    for _ in range(2):
                        result = cb.reduce_groups(values, indices, reverse)
                        expected = torch.zeros(2, 40, dtype=torch.float32)
                        host = values.cpu().flatten(0, 1).float()
                        for token in range(2):
                            order = sorted(
                                range(6),
                                key=lambda choice: (
                                    int(host_indices[token, choice]),
                                    choice,
                                ),
                            )
                            for choice in order:
                                row = int(host_reverse[token, choice])
                                if 0 <= int(
                                    host_indices[token, choice]
                                ) < 2**31 - 1 and 0 <= row < len(host):
                                    expected[token] += host[row]
                        torch.testing.assert_close(
                            result.cpu(), expected, rtol=0, atol=0
                        )
                        values.neg_()
                    invalid = torch.full_like(reverse, 2**31 - 1)
                    self.assertEqual(
                        torch.count_nonzero(cb.reduce_groups(values, indices, invalid)),
                        0,
                    )
            empty = torch.empty((0, 6), device="cuda", dtype=torch.int64)
            slots, reverse, counts = cb.route_groups(empty, active, rows=2)
            self.assertTrue((slots == -1).all())
            self.assertEqual(torch.count_nonzero(counts), 0)
            self.assertEqual(cb.reduce_groups(values, empty, reverse).shape, (0, 40))
            with self.assertRaises(ValueError):
                cb.route_groups(indices, active, rows=0)
            with self.assertRaises(ValueError):
                cb.reduce_groups(values.float(), indices, reverse)

    def test_grouped_graphs_weight_lifetimes_and_changes_all_devices(self):
        """Retain captured weight storages and replay changed selections on every GPU."""
        with torch.inference_mode():
            for device in range(torch.cuda.device_count()):
                x, sx, w0, s0 = self.operands(2, device=device)
                _, _, w1, s1 = self.operands(2, device=device)
                x = x.unsqueeze(0).expand(2, -1, -1).contiguous()
                sx = sx.unsqueeze(0).expand(2, -1, -1).contiguous()
                x.view(torch.uint8).fill_(0x38)
                sx.view(torch.uint8).fill_(127)
                for weight, scale in ((w0, s0), (w1, s1)):
                    weight.view(torch.uint8).fill_(0x22)
                    scale.view(torch.uint8).fill_(127)
                pool = cb.QuantizedGroups([w0, w1], [s0, s1])
                active = torch.tensor([1, 0], device=device, dtype=torch.int32)
                counts = torch.tensor([2, 1], device=device, dtype=torch.int32)
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    pool.matmul(x, sx, active, counts)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        output = pool.matmul(x, sx, active, counts)
                    x.view(torch.uint8).fill_(0x30)
                    w0.view(torch.uint8).fill_(0x33)
                    counts.copy_(torch.tensor([1, 2], device=device, dtype=torch.int32))
                    del w0, w1, s0, s1
                    graph.replay()
                stream.synchronize()
                expected = (
                    torch.tensor([[16, 0], [24, 24]], dtype=torch.bfloat16)
                    .unsqueeze(-1)
                    .expand(2, 2, 8)
                )
                torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
                graph.reset()

    def test_c_abi_dimensions_alignment_aliases_and_empty_rows(self):
        """Exercise the caller-owned C buffers and reject errors before launching."""
        context = _native.Context(0, torch.cuda.current_stream(0).cuda_stream)
        operation = _native.library().camblas_cuda_quantized_matmul
        operation.argtypes = (
            [ctypes.c_void_p] + [ctypes.c_int] * 5 + [ctypes.c_void_p] * 5
        )
        operation.restype = ctypes.c_int
        x, sx, weight, sw = self.operands(device=0)
        output = torch.empty((1, 8), device=0, dtype=torch.bfloat16)
        dimensions = [1, 1, 8, 32, 32]
        pointers = [
            x.data_ptr(),
            sx.data_ptr(),
            weight.data_ptr(),
            sw.data_ptr(),
            output.data_ptr(),
        ]
        self.assertEqual(operation(context.handle, 1, 0, 8, 32, 32, *([0] * 5)), 0)
        for index, bad in ((0, 2), (1, -1), (1, 65536), (2, 7), (3, 31), (4, 64)):
            invalid = dimensions.copy()
            invalid[index] = bad
            self.assertNotEqual(operation(context.handle, *invalid, *pointers), 0)
        for index in range(5):
            invalid = pointers.copy()
            invalid[index] = 0
            self.assertNotEqual(operation(context.handle, *dimensions, *invalid), 0)
        for index in (0, 2, 4):
            invalid = pointers.copy()
            invalid[index] += 1
            self.assertNotEqual(operation(context.handle, *dimensions, *invalid), 0)
        for pointer in pointers[:4]:
            self.assertNotEqual(
                operation(context.handle, *dimensions, *pointers[:4], pointer), 0
            )


if __name__ == "__main__":
    unittest.main()
