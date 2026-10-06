"""Check grouped projections without assuming a particular CUDA GEMM."""

import unittest

import torch

from camblas import _native


class QKVTests(unittest.TestCase):
    """Verify each independently allocated grouped projection and stream ordering."""

    @classmethod
    def setUpClass(cls):
        """Require the optional native binding and initialise deterministic references."""
        if not hasattr(_native.tensor_module(), "qkv_linear"):
            raise unittest.SkipTest("Requires the native CUDA tensor binding")

    def test_independent_outputs_changed_inputs_and_validation(self):
        """Compare all QKV elements and reject invalid shapes, precision and gradients."""
        torch.set_num_threads(4)
        torch.manual_seed(20261004)
        binding = _native.tensor_module()
        with torch.inference_mode():
            for dtype in [torch.float32, torch.bfloat16]:
                x = torch.randn((2, 7, 128), device="cuda", dtype=dtype) / 8
                weights = [
                    torch.randn((width, 128), device="cuda", dtype=dtype) / 8
                    for width in [256, 32, 32]
                ]
                for changed in [False, True]:
                    if changed:
                        x.add_(0.125)
                        for weight in weights:
                            weight.mul_(-0.5)
                    outputs = binding.qkv_linear(x, *weights)
                    self.assertEqual(len({tensor.data_ptr() for tensor in outputs}), 3)
                    for output, weight in zip(outputs, weights):
                        expected = (x.cpu().double() @ weight.cpu().double().T).to(
                            dtype
                        )
                        torch.testing.assert_close(
                            output.cpu(),
                            expected,
                            rtol=0.015625 if dtype == torch.bfloat16 else 2e-5,
                            atol=0.002 if dtype == torch.bfloat16 else 2e-6,
                        )
                with self.assertRaises(ValueError):
                    binding.qkv_linear(x, weights[0][:, :64], *weights[1:])
                with self.assertRaises(ValueError):
                    binding.qkv_linear(
                        x.double(), *(weight.double() for weight in weights)
                    )
        x = torch.randn((1, 128), device="cuda", requires_grad=True)
        weights = [torch.randn((32, 128), device="cuda") for _ in range(3)]
        with self.assertRaises(ValueError):
            binding.qkv_linear(x, *weights)

    def test_graph_stream_ordering_all_devices(self):
        """Verify changed inputs and weights on captured streams across devices."""
        binding = _native.tensor_module()
        with torch.inference_mode():
            for device in range(torch.cuda.device_count()):
                x = torch.randn((1, 1, 128), device=device)
                weights = [torch.randn((32, 128), device=device) / 8 for _ in range(3)]
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        binding.qkv_linear(x, *weights)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        outputs = binding.qkv_linear(x, *weights)
                    x.add_(0.125)
                    for weight in weights:
                        weight.mul_(-0.5)
                    graph.replay()
                stream.synchronize()
                for output, weight in zip(outputs, weights):
                    expected = (x.cpu().double() @ weight.cpu().double().T).float()
                    torch.testing.assert_close(
                        output.cpu(), expected, rtol=2e-5, atol=2e-6
                    )
                graph.reset()


if __name__ == "__main__":
    unittest.main()
