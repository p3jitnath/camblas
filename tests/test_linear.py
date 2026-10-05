"""Protect the ordinary PyTorch interface when CAMBLAS supplies CUDA kernels."""

import os
import unittest

import torch
import torch.nn.functional as functional

from _camblas import _native


@unittest.skipUnless(
    torch.cuda.is_available() and os.environ.get("CAMBLAS_ENABLE") == "1",
    "Requires CAMBLAS_ENABLE=1 and a built CUDA PyTorch backend",
)
class LinearTests(unittest.TestCase):
    """Compare standard tensor calls with independent CPU arithmetic."""

    @classmethod
    def setUpClass(cls):
        """Keep reference products deterministic and disable TF32."""
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    def setUp(self):
        """Use a fixed input seed without caching output values."""
        torch.manual_seed(705537)

    def assert_product(self, actual, reference):
        """Compare the full output after its declared storage rounding.

        Parameters
        ----------
        actual : torch.Tensor
            Result from the active CUDA backend.
        reference : torch.Tensor
            Independent CPU result computed with FP64 accumulation.
        """
        tolerance = {
            torch.float64: (2e-13, 3e-14),
            torch.float32: (1e-5, 3e-6),
            torch.bfloat16: (0.015625, 0.002),
            torch.float16: (0.002, 0.0005),
            torch.complex64: (1e-5, 3e-6),
        }[actual.dtype]
        torch.testing.assert_close(
            actual.cpu(),
            reference.to(actual.dtype),
            rtol=tolerance[0],
            atol=tolerance[1],
            equal_nan=True,
        )

    def test_shapes_views_bias_and_empty(self):
        """Check linear arguments, broadcast bias and all leading batch axes."""
        for dtype in (torch.float32, torch.float64, torch.bfloat16):
            for shape in ((13,), (5, 13), (2, 3, 5, 13)):
                x = torch.randn(shape, device="cuda", dtype=dtype) / 4
                weight = torch.randn((7, 13), device="cuda", dtype=dtype) / 4
                bias = torch.randn(7, device="cuda", dtype=dtype) / 4
                for value in (None, bias):
                    _native.stats(reset=True)
                    actual = functional.linear(x, weight, value)
                    self.assertGreater(sum(_native.stats().values()), 0)
                    expected = functional.linear(
                        x.cpu().double(),
                        weight.cpu().double(),
                        None if value is None else value.cpu().double(),
                    )
                    self.assert_product(actual, expected)
            x = torch.randn((2, 5, 26), device="cuda", dtype=dtype)[..., ::2]
            weight = torch.randn((13, 7), device="cuda", dtype=dtype).T
            self.assert_product(
                functional.linear(x, weight),
                functional.linear(x.cpu().double(), weight.cpu().double()),
            )
            for shape, width in (((2, 3, 0), 7), ((0, 5, 13), 7), ((2, 3, 13), 0)):
                x = torch.empty(shape, device="cuda", dtype=dtype)
                weight = torch.empty((width, shape[-1]), device="cuda", dtype=dtype)
                self.assert_product(
                    functional.linear(x, weight),
                    torch.zeros((*shape[:-1], width), dtype=torch.float64),
                )
        with self.assertRaises(RuntimeError):
            functional.linear(x, torch.ones((7, 12), device=x.device))

    def test_out_scalars_and_fallbacks(self):
        """Preserve out ownership, one version increment and unsupported dtypes."""
        for dtype in (
            torch.float32,
            torch.float64,
            torch.bfloat16,
            torch.float16,
            torch.complex64,
        ):
            a = torch.randn((5, 7), device="cuda", dtype=dtype) / 4
            b = torch.randn((7, 3), device="cuda", dtype=dtype) / 4
            host_dtype = torch.complex128 if dtype.is_complex else torch.float64
            product = a.cpu().to(host_dtype) @ b.cpu().to(host_dtype)
            for out in (
                torch.empty((5, 3), device="cuda", dtype=dtype),
                torch.empty((3, 5), device="cuda", dtype=dtype).T,
            ):
                version = out._version
                self.assertIs(torch.mm(a, b, out=out), out)
                self.assertEqual(out._version, version + 1)
                self.assert_product(out, product)
            bias = torch.randn(3, device="cuda", dtype=dtype) / 4
            actual = torch.addmm(bias, a, b, alpha=0.7, beta=-0.3)
            self.assert_product(actual, 0.7 * product - 0.3 * bias.cpu().to(host_dtype))
            self.assert_product(torch.matmul(a, b), product)
            bias.fill_(float("nan"))
            self.assert_product(torch.addmm(bias, a, b, beta=0), product)
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            a = torch.eye(4, device="cuda")
            _native.stats(reset=True)
            torch.testing.assert_close(torch.mm(a, a), a)
            self.assertEqual(sum(_native.stats().values()), 0)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = False

    def test_first_and_second_derivatives(self):
        """Compare parameter gradients and an input Hessian contraction in FP64."""
        host = [
            torch.randn(shape, dtype=torch.float64).requires_grad_()
            for shape in ((2, 3, 13), (7, 13), (7,))
        ]
        device = [value.detach().cuda().requires_grad_() for value in host]
        reference = functional.linear(*host)
        actual = functional.linear(*device)
        self.assert_product(actual, reference)
        expected_grad = torch.autograd.grad(
            reference.square().sum(), host, create_graph=True
        )
        actual_grad = torch.autograd.grad(
            actual.square().sum(), device, create_graph=True
        )
        for result, expected in zip(actual_grad, expected_grad):
            torch.testing.assert_close(result.cpu(), expected, rtol=2e-12, atol=3e-13)
        expected_hessian = torch.autograd.grad(expected_grad[0].sum(), host[0])[0]
        actual_hessian = torch.autograd.grad(actual_grad[0].sum(), device[0])[0]
        torch.testing.assert_close(
            actual_hessian.cpu(), expected_hessian, rtol=2e-12, atol=3e-13
        )

    def test_graph_streams_and_changed_weights(self):
        """Replay current values on nondefault streams on every allocated GPU."""
        for device in range(torch.cuda.device_count()):
            for dtype in (torch.float32, torch.float64, torch.bfloat16):
                x = torch.randn((5, 32), device=device, dtype=dtype) / 4
                weight = torch.randn((7, 32), device=device, dtype=dtype) / 4
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        functional.linear(x, weight)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        output = functional.linear(x, weight)
                    x.add_(0.125)
                    weight.mul_(-0.5)
                    graph.replay()
                stream.synchronize()
                self.assert_product(
                    output, functional.linear(x.cpu().double(), weight.cpu().double())
                )
                graph.reset()


if __name__ == "__main__":
    unittest.main()
