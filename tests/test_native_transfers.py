"""Check CPU-to-GPU-to-CPU entry points against independent CPU arithmetic."""

import math
import unittest

import torch

from camblas_gpu import _native


class NativeTransferTests(unittest.TestCase):
    """Verify fresh CPU output ownership, copies, arithmetic and validation."""

    @classmethod
    def setUpClass(cls):
        """Require the optional native binding and initialise deterministic references."""
        if not hasattr(_native.tensor_module(), "matmul_transfer"):
            raise unittest.SkipTest("Requires the native CUDA tensor binding")
        torch.set_num_threads(4)
        torch.manual_seed(20261004)
        cls.binding = _native.tensor_module()

    def check(self, output, expected, pinned):
        """Compare every output element with independent CPU double arithmetic."""
        self.assertEqual(output.device.type, "cpu")
        self.assertEqual(output.is_pinned(), pinned)
        tolerance = 2e-5 if output.dtype == torch.float32 else 2e-11
        torch.testing.assert_close(
            output.double(), expected, rtol=tolerance, atol=tolerance
        )

    def test_all_operations_policies_views_and_changed_inputs(self):
        """Compare all transfer operations under each CPU allocation policy."""
        for dtype in [torch.float32, torch.float64]:
            for memory in ["pageable", "prefault", "pinned"]:
                x = torch.randn((13, 34), dtype=dtype)[:, ::2] / 4
                w = torch.randn((17, 23), dtype=dtype) / 4
                w2 = torch.randn((23, 11), dtype=dtype) / 4
                b = torch.randn(23, dtype=dtype) / 4
                b2 = torch.randn(11, dtype=dtype) / 4
                q = torch.randn((13, 17), dtype=dtype) / 4
                k = torch.randn((19, 17), dtype=dtype) / 4
                v = torch.randn((19, 11), dtype=dtype) / 4
                for changed in [False, True]:
                    if changed:
                        for value in [x, w, w2, b, b2, q, k, v]:
                            value.mul_(-0.5).add_(0.125)
                    kwargs = {"output_memory": memory}
                    expected = x.double() @ w.double()
                    output = self.binding.matmul_transfer(x, w, **kwargs)
                    self.check(output, expected, memory == "pinned")
                    other = self.binding.matmul_transfer(x, w, **kwargs)
                    self.assertNotEqual(output.data_ptr(), other.data_ptr())
                    other.zero_()
                    self.check(output, expected, memory == "pinned")
                    self.check(
                        self.binding.gram_transfer(x, **kwargs),
                        x.double().T @ x.double(),
                        memory == "pinned",
                    )
                    expected = (x.double() @ w.double() + b.double()).clamp_min(
                        0
                    ) @ w2.double() + b2.double()
                    self.check(
                        self.binding.mlp_transfer(x, w, b, w2, b2, **kwargs),
                        expected,
                        memory == "pinned",
                    )
                    for scale in [None, 0.7]:
                        expected = (
                            (q.double() @ k.double().T)
                            * (scale if scale is not None else 1 / math.sqrt(17))
                        ).softmax(-1) @ v.double()
                        self.check(
                            self.binding.attention_transfer(
                                q, k, v, scale=scale, **kwargs
                            ),
                            expected,
                            memory == "pinned",
                        )

    def test_nondefault_streams_and_devices_are_synchronous(self):
        """Check synchronous CPU results and restoration of the current CUDA device."""
        x = torch.randn((13, 17), dtype=torch.float64)
        w = torch.randn((17, 11), dtype=torch.float64)
        for device in range(torch.cuda.device_count()):
            original_device = torch.cuda.current_device()
            stream = torch.cuda.Stream(device=device)
            with torch.cuda.stream(stream):
                output = self.binding.matmul_transfer(x, w, device=device)
                self.check(output, x @ w, True)
            self.assertEqual(torch.cuda.current_device(), original_device)

    def test_dense_transposes_and_padded_cpu_views(self):
        """Preserve dense transpose storage and materialise irregular input strides."""
        storage = torch.randn((13, 34), dtype=torch.float64) / 4
        for x in (storage.T, storage[:, ::2], storage[::2, 1:]):
            weight = torch.randn((x.size(1), 11), dtype=x.dtype) / 4
            for memory in ("pageable", "prefault", "pinned"):
                for changed in (False, True):
                    if changed:
                        x.mul_(-0.5).add_(0.125)
                        weight.add_(0.25)
                    self.check(
                        self.binding.matmul_transfer(x, weight, output_memory=memory),
                        x @ weight,
                        memory == "pinned",
                    )

    def test_empty_and_invalid_arguments(self):
        """Check zero inner dimensions and reject incompatible arguments."""
        x = torch.empty((13, 0), dtype=torch.float64)
        w = torch.empty((0, 11), dtype=torch.float64)
        self.check(
            self.binding.matmul_transfer(x, w),
            torch.zeros((13, 11), dtype=x.dtype),
            True,
        )
        x = torch.ones((3, 5), dtype=torch.float64)
        w = torch.ones((5, 7), dtype=torch.float64)
        for bad_x, bad_w in [
            (x.float(), w),
            (x, w.float()),
            (x.bfloat16(), w.bfloat16()),
            (x, w.T),
            (x[0], w),
        ]:
            with self.assertRaises((ValueError, RuntimeError)):
                self.binding.matmul_transfer(bad_x, bad_w)
        with self.assertRaises((ValueError, RuntimeError)):
            self.binding.matmul_transfer(x, w, output_memory="invalid")
        with self.assertRaises((ValueError, RuntimeError)):
            self.binding.matmul_transfer(x.requires_grad_(), w)
        with self.assertRaises((ValueError, RuntimeError)):
            self.binding.matmul_transfer(x.detach().cuda(), w)

    def test_cpu_input_reuse_after_empty_results_and_errors(self):
        """Complete queued CPU reads even when the output is empty or validation fails late."""
        for pinned in (False, True):
            x = torch.ones((1, 512), dtype=torch.float64, pin_memory=pinned)
            weight = torch.ones((512, 2048), dtype=torch.float64, pin_memory=pinned)
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                output = self.binding.matmul_transfer(x, weight)
                self.assertTrue(stream.query())
                weight.fill_(2)
                self.check(
                    output, torch.full((1, 2048), 512, dtype=torch.float64), True
                )
                empty = self.binding.matmul_transfer(x[:0], weight)
                self.assertEqual(empty.shape, (0, 2048))
                self.assertTrue(stream.query())
                with self.assertRaises(ValueError):
                    self.binding.matmul_transfer(x, weight, output_memory="invalid")
                self.assertTrue(stream.query())
                weight.fill_(-3)


if __name__ == "__main__":
    unittest.main()
