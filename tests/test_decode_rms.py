"""Check the single-row RMS fusion's exact PyTorch arithmetic order."""

import unittest

import torch

from camblas_gpu import _native


class DecodeRMSTests(unittest.TestCase):
    """Preserve storage rounding and reduction order across widths and streams."""

    @classmethod
    def setUpClass(cls):
        """Require the optional native binding and initialise deterministic references."""
        if not hasattr(_native.tensor_module(), "rms_norm"):
            raise unittest.SkipTest("Requires the native CUDA tensor binding")

    def test_exact_reference_all_widths_dtypes_and_changed_values(self):
        """Check exact RMS results before and after mutations for all supported widths."""
        torch.manual_seed(20261004)
        binding = _native.tensor_module()
        with torch.inference_mode():
            for dtype in [torch.float32, torch.bfloat16]:
                for width in [2048, 2052, 4096, 8192, 16384, 32768, 65536]:
                    x = torch.randn((1, 1, width), device="cuda", dtype=dtype)
                    weight = torch.randn(width, device="cuda", dtype=dtype)
                    for changed in [False, True]:
                        if changed:
                            x.mul_(-0.5).add_(0.125)
                            weight.mul_(-0.5).add_(0.125)
                        for epsilon in [0, 1e-6, 0.01]:
                            value = x.float()
                            variance = value.pow(2).mean(-1, keepdim=True)
                            expected = weight * (
                                value * torch.rsqrt(variance + epsilon)
                            ).to(dtype)
                            torch.testing.assert_close(
                                binding.rms_norm(x, weight, epsilon),
                                expected,
                                rtol=0,
                                atol=0,
                            )

    def test_stream_graph_all_devices(self):
        """Replay graphs on non-default streams after input and weight mutations."""
        binding = _native.tensor_module()
        with torch.no_grad():
            for device in range(torch.cuda.device_count()):
                x = torch.randn((1, 1, 8192), device=device, dtype=torch.bfloat16)
                weight = torch.randn(8192, device=device, dtype=x.dtype)
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        binding.rms_norm(x, weight, 1e-6)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        output = binding.rms_norm(x, weight, 1e-6)
                    x.mul_(-0.5).add_(0.125)
                    weight.mul_(-0.5)
                    graph.replay()
                stream.synchronize()
                expected = weight * (
                    x.float()
                    * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
                ).to(x.dtype)
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                graph.reset()

    def test_final_round_with_bfloat16_and_float32_weights(self):
        """Compare final-only rounding with PyTorch and an independent CPU FP64 mean."""
        binding = _native.tensor_module()
        torch.manual_seed(20261004)
        with torch.inference_mode():
            for width in (128, 2048, 5120, 8192, 20480, 65536):
                for rows in (1, 3):
                    for weight_type in (torch.bfloat16, torch.float32):
                        x = (
                            torch.randn(
                                (rows, width), device="cuda", dtype=torch.bfloat16
                            )
                            / 8
                        )
                        weight = (
                            torch.randn(width, device="cuda", dtype=weight_type) / 8
                        )
                        for changed in ("original", "input", "weight"):
                            if changed == "input":
                                x.mul_(-0.5).add_(0.125)
                            elif changed == "weight":
                                weight.mul_(-0.5)
                            for epsilon in (0, 1e-20, 1e-6):
                                values = x.float()
                                expected = (
                                    weight.float()
                                    * (
                                        values
                                        * (
                                            values.square().mean(-1, keepdim=True)
                                            + epsilon
                                        ).rsqrt()
                                    )
                                ).to(x.dtype)
                                actual = binding.rms_norm(
                                    x, weight, epsilon, round_before_weight=False
                                )
                                torch.testing.assert_close(
                                    actual, expected, rtol=0, atol=0
                                )
                                host = x.cpu().double()
                                independent = (
                                    weight.cpu().double()
                                    * host
                                    * (
                                        host.square().mean(-1, keepdim=True) + epsilon
                                    ).rsqrt()
                                )
                                torch.testing.assert_close(
                                    actual.cpu().double(),
                                    independent,
                                    rtol=0.008,
                                    atol=0.002,
                                )


if __name__ == "__main__":
    unittest.main()
