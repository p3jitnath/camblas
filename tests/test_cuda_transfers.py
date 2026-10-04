"""Check complete synchronous CUDA downloads with fresh CPU outputs."""

import unittest

import torch

import camblas_gpu as cb


@unittest.skipUnless(torch.cuda.is_available(), "A CUDA-enabled PyTorch is required")
class CudaTransferTests(unittest.TestCase):
    """Verify values, ownership, views and active-stream ordering."""

    def test_full_outputs_changed_inputs_and_views(self):
        """Retain both outputs and require fresh storage after changing every value."""
        for dtype in [torch.float32, torch.float64, torch.bfloat16, torch.int64]:
            host = torch.arange(17 * 31).reshape(17, 31).to(dtype)
            source = host.cuda()
            for memory in ["pageable", "prefault", "pinned"]:
                for view in [source, source.T, source[:, ::2]]:
                    first = cb.copy_to_cpu(view, memory=memory)
                    reference = (
                        host
                        if view is source
                        else host.T
                        if view.shape == source.T.shape
                        else host[:, ::2]
                    )
                    torch.testing.assert_close(first, reference, rtol=0, atol=0)
                    second = cb.copy_to_cpu(view + 4, memory=memory)
                    torch.testing.assert_close(second, (reference + 4), rtol=0, atol=0)
                    torch.testing.assert_close(first, reference, rtol=0, atol=0)
                    self.assertNotEqual(first.data_ptr(), second.data_ptr())
                    self.assertEqual(first.is_pinned(), memory == "pinned")
                    self.assertEqual(first.device.type, "cpu")

    def test_streams_all_devices_and_immediate_cpu_access(self):
        """Read completed CPU values immediately after work on nondefault streams."""
        for device in range(torch.cuda.device_count()):
            with torch.cuda.device(device):
                stream = torch.cuda.Stream()
                for memory in ["pageable", "prefault", "pinned"]:
                    with torch.cuda.stream(stream):
                        source = torch.empty(
                            (257, 19), device=device, dtype=torch.float64
                        )
                        source.fill_(7.5)
                        output = cb.copy_to_cpu(source, memory=memory)
                    self.assertTrue(torch.equal(output, torch.full_like(output, 7.5)))

    def test_empty_outputs_detachment_and_validation(self):
        """Keep shapes and dtypes while returning inference outputs without gradients."""
        for memory in ["pageable", "prefault", "pinned"]:
            value = torch.ones((5, 7), device="cuda", requires_grad=True)
            output = cb.copy_to_cpu(value, memory=memory)
            self.assertFalse(output.requires_grad)
            self.assertIsNone(output.grad_fn)
            value = torch.empty((2, 0, 3), device="cuda", dtype=torch.float64)
            output = cb.copy_to_cpu(value, memory=memory)
            self.assertEqual(output.shape, value.shape)
            self.assertEqual(output.dtype, value.dtype)
        with self.assertRaises(ValueError):
            cb.copy_to_cpu(torch.ones(7))
        with self.assertRaises(ValueError):
            cb.copy_to_cpu(torch.ones(7, device="cuda"), memory="unknown")
