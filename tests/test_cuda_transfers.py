"""Protect the synchronous, fresh-output contract used by transfer benchmarks."""

import unittest

import torch

import camblas_gpu as cb


@unittest.skipUnless(torch.cuda.is_available(), "A CUDA-enabled PyTorch is required")
class CudaTransferTests(unittest.TestCase):
    """Check completed downloads, retained output ownership and allocation policies."""

    def test_download_completion_and_output_ownership(self):
        """Return completed independent storage from a strided view on every GPU."""
        host = torch.arange(257 * 19, dtype=torch.float64).reshape(257, 19).T
        for device in range(torch.cuda.device_count()):
            with torch.cuda.device(device):
                stream = torch.cuda.Stream()
                for memory in ("pageable", "prefault", "pinned"):
                    with torch.cuda.stream(stream):
                        source = host.T.contiguous().to(device).T
                        first = cb.copy_to_cpu(source, memory=memory)
                        source.add_(4)
                        second = cb.copy_to_cpu(source, memory=memory)
                    torch.testing.assert_close(first, host, rtol=0, atol=0)
                    torch.testing.assert_close(second, host + 4, rtol=0, atol=0)
                    self.assertNotEqual(first.data_ptr(), second.data_ptr())
                    self.assertEqual(first.is_pinned(), memory == "pinned")
        with self.assertRaises(ValueError):
            cb.copy_to_cpu(host)
        with self.assertRaises(ValueError):
            cb.copy_to_cpu(source, memory="unknown")
