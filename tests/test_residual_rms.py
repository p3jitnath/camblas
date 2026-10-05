"""Check residual fusion against independent CPU arithmetic and exact storage order."""

import ctypes
import unittest

import torch

from _camblas import _native


class ResidualRMSTests(unittest.TestCase):
    """Preserve independently checked residual and normalisation arithmetic."""

    @classmethod
    def setUpClass(cls):
        """Require the optional native residual fusion binding."""
        if not hasattr(_native.tensor_module(), "add_rms_norm"):
            raise unittest.SkipTest("Requires the native residual RMS binding")
        torch.set_num_threads(4)
        cls.binding = _native.tensor_module()

    def test_all_outputs_changed_inputs_and_boundaries(self):
        """Check all values against CPU double and exact typed storage order."""
        torch.manual_seed(20261004)
        with torch.inference_mode():
            for dtype in [torch.float32, torch.bfloat16]:
                for shape in [
                    (1, 1, 2048),
                    (1, 1, 2052),
                    (1, 1, 8192),
                    (1, 1, 65536),
                    (2, 3, 127),
                    (1, 1, 4095),
                ]:
                    x = torch.randn(shape, device="cuda", dtype=dtype) / 4
                    residual = torch.randn(shape, device="cuda", dtype=dtype) / 4
                    weight = torch.randn(shape[-1], device="cuda", dtype=dtype) / 4
                    for changed in [False, True]:
                        if changed:
                            x.mul_(-0.5).add_(0.125)
                            residual.add_(0.25)
                            weight.mul_(-0.5)
                        added, output = self.binding.add_rms_norm(
                            x, residual, weight, 1e-6
                        )
                        expected_added = (
                            x.cpu().double() + residual.cpu().double()
                        ).to(dtype)
                        torch.testing.assert_close(
                            added.cpu(), expected_added, rtol=0, atol=0
                        )
                        reference = expected_added.double()
                        expected = (
                            weight.cpu().double()
                            * (
                                reference
                                * torch.rsqrt(
                                    reference.square().mean(-1, keepdim=True) + 1e-6
                                )
                            )
                            .to(dtype)
                            .double()
                        ).to(dtype)
                        torch.testing.assert_close(
                            output.cpu(),
                            expected,
                            rtol=0.02 if dtype == torch.bfloat16 else 2e-5,
                            atol=0.002 if dtype == torch.bfloat16 else 2e-6,
                        )
                        value = added.float()
                        exact = weight * (
                            value
                            * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
                        ).to(dtype)
                        if dtype == torch.bfloat16 or (
                            shape[:-1] == (1, 1)
                            and shape[-1] % 4 == 0
                            and shape[-1] >= 2048
                        ):
                            torch.testing.assert_close(output, exact, rtol=0, atol=0)
                        self.assertNotEqual(added.data_ptr(), x.data_ptr())
                        self.assertNotEqual(output.data_ptr(), added.data_ptr())
                    with self.assertRaises(ValueError):
                        self.binding.add_rms_norm(x, residual, weight, -1)
                    with self.assertRaises(ValueError):
                        self.binding.add_rms_norm(x, residual[..., :3], weight, 1e-6)
        with self.assertRaises(ValueError):
            self.binding.add_rms_norm(
                x.float().requires_grad_(), residual.float(), weight.float(), 1e-6
            )

    def test_graph_all_devices_and_changed_operands(self):
        """Replay changed inputs on non-default streams on every device."""
        with torch.no_grad():
            for device in range(torch.cuda.device_count()):
                x = torch.randn((1, 1, 8192), device=device, dtype=torch.bfloat16)
                residual = torch.randn_like(x)
                weight = torch.randn(8192, device=device, dtype=x.dtype)
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        self.binding.add_rms_norm(x, residual, weight, 1e-6)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        added, output = self.binding.add_rms_norm(
                            x, residual, weight, 1e-6
                        )
                    x.add_(0.125)
                    residual.mul_(-0.5)
                    weight.mul_(-0.5)
                    graph.replay()
                stream.synchronize()
                expected_added = x + residual
                torch.testing.assert_close(added, expected_added, rtol=0, atol=0)
                value = expected_added.float()
                expected = weight * (
                    value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
                ).to(x.dtype)
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                graph.reset()

    def test_c_interface_selects_device_and_rejects_aliases(self):
        """Select the context device independently and preserve operands on errors."""
        function = _native.library().camblas_cuda_add_rms_norm
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        ] + [ctypes.c_void_p] * 5
        function.restype = ctypes.c_int
        device = torch.cuda.device_count() - 1
        original_device = torch.cuda.current_device()
        stream = torch.cuda.Stream(device=device)
        context = ctypes.c_void_p()
        self.assertEqual(
            _native.library().camblas_cuda_create(
                device, stream.cuda_stream, ctypes.byref(context)
            ),
            0,
        )
        try:
            with torch.no_grad():
                x = torch.randn(2048, device=device)
                residual = torch.randn_like(x)
                weight = torch.randn_like(x)
                added, output = torch.empty_like(x), torch.empty_like(x)
                stream.wait_stream(torch.cuda.current_stream(device))
                arguments = [
                    context,
                    0,
                    2048,
                    1e-6,
                    x.data_ptr(),
                    residual.data_ptr(),
                    weight.data_ptr(),
                    added.data_ptr(),
                    output.data_ptr(),
                ]
                for changed in [False, True]:
                    if changed:
                        x.mul_(-0.5)
                        residual.add_(0.125)
                        weight.mul_(-0.5)
                        stream.wait_stream(torch.cuda.current_stream(device))
                    self.assertEqual(function(*arguments), 0)
                    stream.synchronize()
                    expected_sum = (x.cpu().double() + residual.cpu().double()).float()
                    torch.testing.assert_close(
                        added.cpu(), expected_sum, rtol=0, atol=0
                    )
                    value = expected_sum.double()
                    expected = (
                        weight.cpu().double()
                        * value
                        * (value.square().mean() + 1e-6).rsqrt()
                    )
                    torch.testing.assert_close(
                        output.cpu().double(), expected, rtol=2e-5, atol=2e-5
                    )
                before = x.cpu().clone()
                invalid = list(arguments)
                invalid[-1] = x.data_ptr()
                self.assertNotEqual(function(*invalid), 0)
                invalid = list(arguments)
                invalid[2] = 2049
                self.assertNotEqual(function(*invalid), 0)
                torch.testing.assert_close(x.cpu(), before, rtol=0, atol=0)
                self.assertEqual(torch.cuda.current_device(), original_device)
        finally:
            self.assertEqual(_native.library().camblas_cuda_destroy(context), 0)


if __name__ == "__main__":
    unittest.main()
