"""Independent BF16 GEMM checks for the isolated Llama candidate."""

import unittest

import torch

import _camblas as cb
from _camblas import _native


class Bfloat16Tests(unittest.TestCase):
    """Check full outputs against CPU FP64 products and analytic cases."""

    @classmethod
    def setUpClass(cls):
        """Keep CPU reference products independent of CUDA algorithms."""
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False

    def setUp(self):
        """Use repeatable values while preserving changed-input checks."""
        torch.manual_seed(705537)

    def assert_product(self, actual, a, b, alpha=1.0, beta=0.0, old=None):
        """Compare every element with FP64 arithmetic followed by BF16 rounding."""
        expected = alpha * (a.cpu().double() @ b.cpu().double())
        if beta != 0:
            expected += beta * old.cpu().double()
        torch.testing.assert_close(
            actual.cpu(),
            expected.bfloat16(),
            rtol=0.015625,
            atol=0.0005,
            equal_nan=True,
        )

    def test_layouts_scalars_and_changed_weights(self):
        """Cover all transpose pairs, padding, irregular views and nonzero beta."""
        for policy in ["auto", "classical", "lt", "strassen4"]:
            for ta in [False, True]:
                for tb in [False, True]:
                    a = (
                        torch.randn(
                            (13, 17) if ta else (17, 13),
                            dtype=torch.bfloat16,
                            device="cuda",
                        )
                        / 4
                    )
                    b = (
                        torch.randn(
                            (23, 13) if tb else (13, 23),
                            dtype=torch.bfloat16,
                            device="cuda",
                        )
                        / 4
                    )
                    a, b = (a.T if ta else a), (b.T if tb else b)
                    with cb.algorithm(policy):
                        self.assert_product(cb.matmul(a, b), a, b)
                        c = torch.full(
                            (17, 23), float("nan"), dtype=torch.bfloat16, device="cuda"
                        )
                        cb.matmul(a, b, out=c)
                        self.assert_product(c, a, b)
                        a.add_(0.125)
                        b.mul_(-0.5)
                        old = torch.randn_like(c)
                        c.copy_(old)
                        cb.matmul(a, b, out=c, alpha=-0.7, beta=0.3)
                        self.assert_product(c, a, b, -0.7, 0.3, old)
            a = torch.randn((17, 26), dtype=torch.bfloat16, device="cuda")[:, ::2]
            b = torch.randn((13, 31), dtype=torch.bfloat16, device="cuda")[:, :23]
            with cb.algorithm(policy):
                self.assert_product(cb.matmul(a, b), a, b)

    def test_analytic_identity_and_fp32_accumulation(self):
        """Require exact index placement and preservation of a small residual."""
        for policy in ["auto", "classical", "lt"]:
            a = torch.eye(65, dtype=torch.bfloat16, device="cuda")
            b = torch.arange(65 * 33, device="cuda").reshape(65, 33).bfloat16()
            with cb.algorithm(policy):
                torch.testing.assert_close(cb.matmul(a, b), b, rtol=0, atol=0)
                b.copy_(b.flip(0))
                b.add_(4)
                torch.testing.assert_close(cb.matmul(a, b), b, rtol=0, atol=0)
                a = torch.tensor(
                    [[256.0, 1.0, -256.0]], dtype=torch.bfloat16, device="cuda"
                )
                b = torch.ones((3, 1), dtype=torch.bfloat16, device="cuda")
                torch.testing.assert_close(
                    cb.matmul(a, b), torch.ones_like(a[:, :1]), rtol=0, atol=0
                )

    def test_empty_zero_alpha_exceptional_and_validation(self):
        """Skip unused inputs and reject unsupported fused-operation dtypes."""
        a = torch.full((5, 7), float("nan"), dtype=torch.bfloat16, device="cuda")
        b = torch.full((7, 3), float("inf"), dtype=torch.bfloat16, device="cuda")
        c = torch.full((5, 3), float("nan"), dtype=torch.bfloat16, device="cuda")
        cb.matmul(a, b, out=c, alpha=0.0, beta=0.0)
        torch.testing.assert_close(c, torch.zeros_like(c), rtol=0, atol=0)
        c.fill_(3)
        cb.matmul(a, b, out=c, alpha=0.0, beta=-0.5)
        torch.testing.assert_close(c, torch.full_like(c, -1.5), rtol=0, atol=0)
        for m, k, n in [(0, 7, 3), (5, 7, 0), (5, 0, 3)]:
            a = torch.empty((m, k), dtype=torch.bfloat16, device="cuda")
            b = torch.empty((k, n), dtype=torch.bfloat16, device="cuda")
            output = cb.matmul(a, b)
            torch.testing.assert_close(
                output,
                torch.zeros((m, n), dtype=a.dtype, device=a.device),
                rtol=0,
                atol=0,
            )
        a = torch.eye(8, dtype=torch.bfloat16, device="cuda")
        b = torch.ones_like(a)
        b[2, 3] = float("nan")
        for policy in ["auto", "lt"]:
            with cb.algorithm(policy):
                torch.testing.assert_close(
                    cb.matmul(a, b), a @ b, equal_nan=True, rtol=0, atol=0
                )
        with self.assertRaises(ValueError):
            cb.matmul(a.half(), b.half())
        with self.assertRaises(ValueError):
            cb.matmul(a, b.float())
        with self.assertRaises(ValueError):
            cb.matmul(a, b, out=a)
        with self.assertRaises(ValueError):
            cb.affine(a, b, torch.ones(8, dtype=a.dtype, device=a.device))

    def test_c_interface_padding_and_invalid_arguments(self):
        """Exercise column-major leading dimensions and untouched output padding."""
        lib = _native.library()
        native = _native.context(0)
        for ta in [b"N", b"T"]:
            for tb in [b"N", b"T"]:
                m, k, n = 5, 7, 3
                ar, ac = (m, k) if ta == b"N" else (k, m)
                br, bc = (k, n) if tb == b"N" else (n, k)
                lda, ldb, ldc = ar + 2, br + 3, m + 4
                a = torch.randn((ac, lda), dtype=torch.bfloat16, device="cuda")
                b = torch.randn((bc, ldb), dtype=torch.bfloat16, device="cuda")
                c = torch.full(
                    (n, ldc), float("nan"), dtype=torch.bfloat16, device="cuda"
                )
                left, right = a[:, :ar].T, b[:, :br].T
                left = left if ta == b"N" else left.T
                right = right if tb == b"N" else right.T
                status = lib.camblas_cuda_bgemm(
                    native.handle,
                    ta,
                    tb,
                    m,
                    n,
                    k,
                    1.0,
                    a.data_ptr(),
                    lda,
                    b.data_ptr(),
                    ldb,
                    0.0,
                    c.data_ptr(),
                    ldc,
                )
                self.assertEqual(status, 0)
                torch.cuda.synchronize()
                self.assert_product(c[:, :m].T, left, right)
                self.assertTrue(torch.isnan(c[:, m:]).all().item())
                self.assertNotEqual(
                    lib.camblas_cuda_bgemm(
                        native.handle,
                        b"X",
                        tb,
                        m,
                        n,
                        k,
                        1.0,
                        a.data_ptr(),
                        lda,
                        b.data_ptr(),
                        ldb,
                        0.0,
                        c.data_ptr(),
                        ldc,
                    ),
                    0,
                )

    def test_llama_inner_dimensions(self):
        """Check long Llama dot products, including changed values and cancellation."""
        for inner in [8192, 28672]:
            a = torch.randn((4, inner), device="cuda", dtype=torch.bfloat16) / 8
            b = torch.randn((inner, 17), device="cuda", dtype=torch.bfloat16) / 8
            for policy in ["auto", "classical", "lt"]:
                with cb.algorithm(policy):
                    self.assert_product(cb.matmul(a, b), a, b)
                    a.add_(0.015625)
                    b.mul_(-0.5)
                    self.assert_product(cb.matmul(a, b), a, b)


if __name__ == "__main__":
    unittest.main()
