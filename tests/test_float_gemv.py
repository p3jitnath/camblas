"""Activate the custom FP32 decode path and check independent full outputs."""

import unittest

import torch

import camblas._kernels as cb
from camblas import _native


class GemvTests(unittest.TestCase):
    """Verify general single-vector FP32 products against CPU double arithmetic."""

    @classmethod
    def setUpClass(cls):
        """Require the optional native binding and initialise deterministic references."""
        torch.set_num_threads(64)
        torch.manual_seed(20261004)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    def check(self, actual, x, weight, alpha=1, beta=0, old=None):
        """Compare every output element with independent CPU double arithmetic."""
        expected = alpha * (x.cpu().double() @ weight.cpu().double().T)
        if beta:
            expected += beta * old.cpu().double()
        torch.testing.assert_close(
            actual.cpu(), expected.float(), rtol=2e-4, atol=5e-5, equal_nan=True
        )

    def test_long_inner_dimensions_scalars_padding_and_changes(self):
        """Check long reductions, alpha, beta, padded outputs and mutated operands."""
        for inner, columns in [(4096, 1024), (8192, 8192), (28672, 1024)]:
            x = torch.randn((1, inner), device="cuda", dtype=torch.float32) / 8
            backing = (
                torch.randn((columns, inner + 8), device="cuda", dtype=torch.float32)
                / 8
            )
            weight = backing[:, :inner]
            with cb.algorithm("auto"):
                output = torch.full(
                    (1, columns), float("nan"), device="cuda", dtype=x.dtype
                )
                cb.matmul(x, weight.T, out=output)
                self.check(output, x, weight)
                old = torch.randn_like(output) / 8
                output.copy_(old)
                x.add_(0.015625)
                weight.mul_(-0.5)
                cb.matmul(x, weight.T, out=output, alpha=-0.75, beta=0.25)
                self.check(output, x, weight, -0.75, 0.25, old)

    def test_exact_sparse_and_cancellation(self):
        """Preserve exact sparse products and cancellation cases."""
        inner, columns = 8192, 1024
        weight = torch.zeros((columns, inner), device="cuda", dtype=torch.float32)
        rows = torch.arange(columns, device="cuda")
        weight[rows, rows * 7] = 1
        x = torch.arange(inner, device="cuda").reshape(1, inner).float()
        with cb.algorithm("auto"):
            torch.testing.assert_close(
                cb.matmul(x, weight.T), x[:, rows * 7], rtol=0, atol=0
            )
            weight.zero_()
            weight[:, 0] = 256
            weight[:, 8] = 1
            weight[:, 16] = -256
            x.fill_(1)
            torch.testing.assert_close(
                cb.matmul(x, weight.T),
                torch.ones((1, columns), device="cuda", dtype=x.dtype),
                rtol=0,
                atol=0,
            )

    def test_column_major_output_padding(self):
        """Respect column-major output strides and untouched padding."""
        columns, inner = 1024, 4096
        weight = (
            torch.randn((columns, inner + 8), device="cuda", dtype=torch.float32) / 8
        )
        x = torch.randn((1, inner), device="cuda", dtype=torch.float32) / 8
        output = torch.full(
            (1, columns + 13), float("nan"), device="cuda", dtype=x.dtype
        )
        lib = _native.library()
        context = _native.context(0)
        with cb.algorithm("auto"):
            status = lib.camblas_cuda_sgemm(
                context.handle,
                b"T",
                b"N",
                columns,
                1,
                inner,
                1.0,
                weight.data_ptr(),
                inner + 8,
                x.data_ptr(),
                inner,
                0.0,
                output.data_ptr(),
                columns + 13,
            )
        self.assertEqual(status, 0)
        torch.cuda.synchronize()
        self.check(output[:, :columns], x, weight[:, :inner])
        self.assertTrue(torch.isnan(output[:, columns:]).all().item())

    def test_exceptional_inputs_and_zero_scalars(self):
        """Propagate exceptional values and avoid reading operands for zero scalars."""
        x = torch.ones((1, 4096), device="cuda", dtype=torch.float32)
        weight = torch.ones((1024, 4096), device="cuda", dtype=x.dtype)
        weight[0, 0] = float("inf")
        weight[1, 0] = float("nan")
        with cb.algorithm("auto"):
            output = cb.matmul(x, weight.T)
            self.check(output, x, weight)
            output.fill_(float("nan"))
            cb.matmul(x, weight.T, out=output, alpha=0, beta=0)
            torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)

    def test_neighbouring_dimensions_views_and_gradients(self):
        """Verify dispatch boundaries, unaligned views and derivatives."""
        for inner, columns in [(4095, 1024), (4096, 1023), (4104, 1025)]:
            x = (torch.randn((1, inner + 1), device="cuda", dtype=torch.float32) / 8)[
                :, 1:
            ]
            weight = torch.randn((columns, inner), device="cuda", dtype=x.dtype) / 8
            self.check(cb.matmul(x, weight.T), x, weight)
        x = (
            torch.randn((1, 4096), device="cuda", dtype=torch.float32) / 8
        ).requires_grad_()
        weight = (
            torch.randn((1024, 4096), device="cuda", dtype=x.dtype) / 8
        ).requires_grad_()
        cb.matmul(x, weight.T).sum().backward()
        expected_x = weight.detach().cpu().double().sum(0, keepdim=True).float()
        expected_w = x.detach().cpu().expand(1024, -1)
        torch.testing.assert_close(x.grad.cpu(), expected_x, rtol=2e-4, atol=5e-5)
        torch.testing.assert_close(weight.grad.cpu(), expected_w, rtol=0, atol=0)

    def test_graph_stream_ordering_and_changed_weights_all_devices(self):
        """Replay captured GEMV with changed operands on every device."""
        for device in range(torch.cuda.device_count()):
            x = torch.randn((1, 4096), device=device, dtype=torch.float32) / 8
            weight = torch.randn((1024, 4096), device=device, dtype=torch.float32) / 8
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream), cb.algorithm("auto"):
                for _ in range(3):
                    cb.matmul(x, weight.T)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    output = cb.matmul(x, weight.T)
                x.add_(0.015625)
                weight.mul_(-0.5)
                graph.replay()
            stream.synchronize()
            self.check(output, x, weight)
            graph.reset()

    def test_wide_prefill_workspace_growth_and_graph(self):
        """Keep classical calls and captured projections valid after workspace growth."""
        with torch.no_grad(), cb.algorithm("auto"):
            x = torch.randn((256, 16384), device="cuda") / 128
            weight = torch.randn((1024, 16384), device="cuda") / 128
            small = torch.randn((17, 13), device="cuda")
            right = torch.randn((13, 19), device="cuda")
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                early_x = (
                    torch.randn((128, 8192), device="cuda", dtype=torch.bfloat16) / 128
                )
                early_weight = (
                    torch.randn((2048, 8192), device="cuda", dtype=torch.bfloat16) / 128
                )
                with cb.algorithm("lt"):
                    for _ in range(3):
                        cb.matmul(early_x, early_weight.T)
                    early_graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(early_graph, stream=stream):
                        early_output = cb.matmul(early_x, early_weight.T)
                for _ in range(3):
                    cb.matmul(x, weight.T)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    output = cb.matmul(x, weight.T)
                x.add_(0.001953125)
                weight.mul_(-0.5)
                graph.replay()
                early_x.add_(0.00390625)
                early_weight.mul_(-0.5)
                early_graph.replay()
                with cb.algorithm("classical"):
                    small_output = cb.matmul(small, right)
            stream.synchronize()
            torch.testing.assert_close(
                early_output.cpu().double(),
                early_x.cpu().double() @ early_weight.cpu().double().T,
                rtol=0.02,
                atol=0.002,
            )
            early_graph.reset()
            expected = x.cpu().double() @ weight.cpu().double().T
            torch.testing.assert_close(
                output.cpu().double(), expected, rtol=2e-4, atol=2e-4
            )
            torch.testing.assert_close(
                small_output.cpu().double(),
                small.cpu().double() @ right.cpu().double(),
                rtol=2e-5,
                atol=2e-5,
            )
            graph.reset()


if __name__ == "__main__":
    unittest.main()
