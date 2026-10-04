"""Check the native linear adapter against independent CPU calculations."""

import unittest

import torch
import torch.nn.functional as functional

from camblas_gpu import _native


class LinearTests(unittest.TestCase):
    """Exercise leading batch axes, views, bias and higher derivatives."""

    @classmethod
    def setUpClass(cls):
        """Keep CPU double references small and deterministic."""
        if not hasattr(_native.tensor_module(), "linear_public"):
            raise unittest.SkipTest("Requires the native CUDA tensor binding")
        torch.set_num_threads(1)
        cls.linear = staticmethod(_native.tensor_module().linear_public)

    def test_shapes_views_empty_and_validation(self):
        """Compare every output value for all supported precisions and layouts."""
        torch.manual_seed(705537)
        for dtype in [torch.float32, torch.float64, torch.bfloat16]:
            for shape in [(13,), (5, 13), (2, 5, 13), (2, 3, 5, 13)]:
                x = torch.randn(shape, device="cuda", dtype=dtype) / 4
                weight = torch.randn((7, 13), device="cuda", dtype=dtype) / 4
                bias = torch.randn(7, device="cuda", dtype=dtype) / 4
                for value in [None, bias]:
                    actual = self.linear(x, weight, value)
                    reference = functional.linear(
                        x.cpu().double(),
                        weight.cpu().double(),
                        None if value is None else value.cpu().double(),
                    ).to(dtype)
                    tolerance = (
                        (0.015625, 0.002)
                        if dtype == torch.bfloat16
                        else (
                            (1e-5, 3e-6) if dtype == torch.float32 else (2e-13, 3e-14)
                        )
                    )
                    torch.testing.assert_close(
                        actual.cpu(), reference, rtol=tolerance[0], atol=tolerance[1]
                    )
                x = torch.randn((2, 5, 26), device="cuda", dtype=dtype)[..., ::2]
                weight = torch.randn((13, 7), device="cuda", dtype=dtype).T
                torch.testing.assert_close(
                    self.linear(x, weight).cpu(),
                    functional.linear(x.cpu().double(), weight.cpu().double()).to(
                        dtype
                    ),
                    rtol=tolerance[0],
                    atol=tolerance[1],
                )
            for shape, width in [((2, 3, 0), 7), ((0, 5, 13), 7), ((2, 3, 13), 0)]:
                x = torch.empty(shape, device="cuda", dtype=dtype)
                weight = torch.empty((width, shape[-1]), device="cuda", dtype=dtype)
                actual = self.linear(x, weight)
                torch.testing.assert_close(
                    actual,
                    torch.zeros((*shape[:-1], width), device="cuda", dtype=dtype),
                    rtol=0,
                    atol=0,
                )
        x = torch.empty((1048576, 1048576, 0), device="cuda")
        with self.assertRaisesRegex(ValueError, "flattened row overflow"):
            self.linear(x, torch.empty((7, 0), device="cuda"))
        with self.assertRaises(ValueError):
            self.linear(
                torch.ones((), device="cuda"), torch.ones((1, 1), device="cuda")
            )
        with self.assertRaises(ValueError):
            self.linear(
                torch.ones((2, 13), device="cuda"),
                torch.ones((7, 13), device="cuda"),
                torch.ones(8, device="cuda"),
            )

    def test_first_and_second_derivatives(self):
        """Compare all first derivatives and an input Hessian contraction in FP64."""
        torch.manual_seed(705537)
        host = [
            torch.randn(shape, dtype=torch.float64).requires_grad_()
            for shape in [(2, 3, 13), (7, 13), (7,)]
        ]
        device = [value.detach().cuda().requires_grad_() for value in host]
        reference = functional.linear(*host)
        actual = self.linear(*device)
        torch.testing.assert_close(actual.cpu(), reference, rtol=2e-13, atol=3e-14)
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


if __name__ == "__main__":
    unittest.main()
