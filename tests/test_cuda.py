"""Verify CUDA GEMM semantics, numerical fallbacks, stream use and derivatives."""

import ctypes as ct
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

CUDA_AVAILABLE = torch is not None and torch.cuda.is_available()
if CUDA_AVAILABLE:
    import camblas._kernels as cb
    from camblas import _native


@unittest.skipUnless(CUDA_AVAILABLE, "A CUDA-enabled PyTorch is required")
class CudaTests(unittest.TestCase):
    """Exercise the standalone library through its tensor and C interfaces."""

    @classmethod
    def setUpClass(cls):
        """Disable TF32 so the reference retains full FP32 multiplication."""
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.set_num_threads(1)

    def setUp(self):
        """Make input distributions reproducible across test runs."""
        torch.manual_seed(73013)

    def assert_close(self, actual, expected, *, strassen=False):
        """Compare every output element with a dtype-specific error tolerance.

        Parameters
        ----------
        actual, expected : torch.Tensor
            Candidate and reference outputs with matching shape and precision.
        strassen : bool, optional
            Apply the absolute tolerance used for Strassen rounding.

        Raises
        ------
        AssertionError
            If any output exceeds the selected tolerance.
        """
        rtol, atol = (1e-5, 3e-6) if actual.dtype == torch.float32 else (2e-13, 3e-14)
        if strassen:
            atol *= 5
        torch.testing.assert_close(
            actual, expected, rtol=rtol, atol=atol, equal_nan=True
        )

    def test_layouts_and_alpha_beta(self):
        """Check transposes, padded leading dimensions and beta-zero NaNs."""
        for dtype in (torch.float32, torch.float64):
            for policy in ("classical", "lt", "auto"):
                for ta in (False, True):
                    for tb in (False, True):
                        a = torch.randn(
                            (17, 13) if not ta else (13, 17), device="cuda", dtype=dtype
                        )
                        b = torch.randn(
                            (13, 23) if not tb else (23, 13), device="cuda", dtype=dtype
                        )
                        a, b = (a.T if ta else a), (b.T if tb else b)
                        with cb.algorithm(policy):
                            self.assert_close(cb.matmul(a, b), a @ b)
                            c = torch.randn((17, 23), device="cuda", dtype=dtype)
                            reference = 0.7 * (a @ b) - 0.3 * c
                            cb.matmul(a, b, out=c, alpha=0.7, beta=-0.3)
                            self.assert_close(c, reference)
                            c.fill_(float("nan"))
                            cb.matmul(a, b, out=c)
                            self.assert_close(c, a @ b)
                a = torch.randn((17, 20), device="cuda", dtype=dtype)[:, :13]
                b = torch.randn((13, 31), device="cuda", dtype=dtype)[:, :23]
                with cb.algorithm(policy):
                    self.assert_close(cb.matmul(a, b), a @ b)
                a = torch.randn((17, 26), device="cuda", dtype=dtype)[:, ::2]
                with cb.algorithm(policy):
                    self.assert_close(cb.matmul(a, b), a @ b)

    def test_strassen_changed_inputs_and_gram(self):
        """Require recomputation, full output accuracy and symmetric Gram routing."""
        for dtype in (torch.float32, torch.float64):
            a = torch.randn((512, 512), device="cuda", dtype=dtype) / 512**0.5
            b = torch.randn_like(a) / 512**0.5
            with cb.algorithm("strassen"):
                self.assert_close(cb.matmul(a, b), a @ b, strassen=True)
                a.add_(0.02)
                b.mul_(0.5)
                self.assert_close(cb.matmul(a, b), a @ b, strassen=True)
                c = torch.randn_like(a) / 512**0.5
                reference = 0.7 * (a @ b) - 0.3 * c
                cb.matmul(a, b, out=c, alpha=0.7, beta=-0.3)
                self.assert_close(c, reference, strassen=True)
            cb.stats(reset=True)
            with cb.algorithm("symmetric"):
                c = cb.matmul(a.T, a)
                self.assert_close(c, a.T @ a)
                self.assert_close(c, c.T)
            self.assertEqual(cb.stats()["gram"], 1)

    def test_guarded_fallback(self):
        """Fall back on NaNs, infinities and values unsafe for Strassen sums."""
        for dtype in (torch.float32, torch.float64):
            a = torch.eye(256, device="cuda", dtype=dtype)
            b = torch.eye(256, device="cuda", dtype=dtype)
            for value in (float("nan"), float("inf"), torch.finfo(dtype).max / 2):
                a[0, 0] = value
                cb.stats(reset=True)
                with cb.algorithm("strassen"):
                    c = cb.matmul(a, b)
                self.assert_close(c, a @ b)
                self.assertEqual(cb.stats()["guard_fallback"], 1)

    def test_scratch_allocation_fallback(self):
        """Inject scratch allocation failure and require a complete classical result."""
        compiler = shutil.which("cc")
        if not compiler or sys.platform != "linux":
            self.skipTest("Allocation injection requires a Linux C compiler")
        root = Path(__file__).resolve().parents[1]
        code = """
import ctypes
import torch
import camblas._kernels as cb
from camblas import _native
ctypes.CDLL(_native.library()._name, mode=ctypes.RTLD_GLOBAL)
injector = ctypes.CDLL(None)
injector.camblas_test_fail_next_allocation.argtypes = []
injector.camblas_test_fail_next_allocation.restype = None
injector.camblas_test_allocation_pending.argtypes = []
injector.camblas_test_allocation_pending.restype = ctypes.c_int
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
for dtype in (torch.float32, torch.float64):
    a = torch.randn((512, 512), device='cuda', dtype=dtype) / 512**0.5
    b = torch.randn_like(a) / 512**0.5
    output = torch.empty_like(a)
    with cb.algorithm('classical'):
        cb.matmul(a, b, out=output)
    reference = (a @ b).clone()
    torch.cuda.synchronize()
    cb.stats(reset=True)
    injector.camblas_test_fail_next_allocation()
    with cb.algorithm('strassen'):
        cb.matmul(a, b, out=output)
    torch.cuda.synchronize()
    assert injector.camblas_test_allocation_pending() == 0
    counts = cb.stats()
    assert counts['guard_fallback'] == 1 and counts['classical'] == 1, counts
    assert counts['strassen'] == 0, counts
    torch.testing.assert_close(output, reference)
    # The context must remain usable and the CUDA error state must be clean.
    with cb.algorithm('strassen'):
        cb.matmul(a, b, out=output)
    torch.testing.assert_close(output, reference, rtol=1e-5, atol=2e-6)
    cb.stats(reset=True)
    injector.camblas_test_fail_next_allocation()
    with cb.algorithm('strassen3'):
        cb.matmul(a, b, out=output)
    torch.cuda.synchronize()
    assert injector.camblas_test_allocation_pending() == 0
    counts = cb.stats()
    assert counts['guard_fallback'] == 1 and counts['strassen'] == 1, counts
    assert counts['classical'] == 0, counts
    torch.testing.assert_close(output, reference, rtol=1e-5, atol=2e-6)
    with cb.algorithm('strassen3'):
        cb.matmul(a, b, out=output)
    torch.testing.assert_close(output, reference, rtol=1e-5, atol=2e-6)
    _native.close()
    # Four levels need less storage than the earlier arenas. Start a new
    # context so the injected failure reaches its first scratch allocation.
    with cb.algorithm('classical'):
        cb.matmul(a, b, out=output)
    cb.stats(reset=True)
    injector.camblas_test_fail_next_allocation()
    with cb.algorithm('strassen4'):
        cb.matmul(a, b, out=output)
    torch.cuda.synchronize()
    assert injector.camblas_test_allocation_pending() == 0
    counts = cb.stats()
    assert counts['guard_fallback'] == 1 and counts['classical'] == 1, counts
    assert counts['strassen'] == 0, counts
    torch.testing.assert_close(output, reference)
    with cb.algorithm('strassen4'):
        cb.matmul(a, b, out=output)
    torch.testing.assert_close(output, reference, rtol=1e-5, atol=2e-6)
    _native.close()
"""
        with tempfile.TemporaryDirectory(prefix="camblas-cuda-oom-") as directory:
            shim = Path(directory) / "fail_alloc.so"
            subprocess.run(
                [
                    compiler,
                    "-shared",
                    "-fPIC",
                    str(root / "tests/cuda_fail_alloc.c"),
                    "-ldl",
                    "-o",
                    str(shim),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            environment = dict(os.environ)
            environment["LD_PRELOAD"] = str(shim) + (
                ":" + environment["LD_PRELOAD"] if environment.get("LD_PRELOAD") else ""
            )
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=90,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_empty_and_invalid_arguments(self):
        """Cover zero inner dimensions, no-op shapes and rejected aliases."""
        for m, n, k in ((0, 7, 3), (5, 0, 3), (5, 7, 0)):
            a = torch.empty((m, k), device="cuda")
            b = torch.empty((k, n), device="cuda")
            self.assert_close(cb.matmul(a, b), a @ b)
        a = torch.eye(4, device="cuda")
        with self.assertRaises(ValueError):
            cb.matmul(a, a, out=a)
        with self.assertRaises(ValueError):
            cb.matmul(a, a.double())
        with self.assertRaises(ValueError):
            cb.matmul(a, a[:3])
        with self.assertRaises(ValueError):
            with cb.algorithm("invalid"):
                pass

    def test_full_fp32_mantissa(self):
        """Distinguish full FP32 multiplication from reduced TF32 mantissas."""
        size = 256
        a = torch.full((size, size), 1.0003, device="cuda")
        b = torch.full_like(a, 1.0003)
        expected = a.double().cpu() @ b.double().cpu()
        magnitude = expected.abs()
        epsilon = torch.finfo(torch.float32).eps
        # A length-k dot product has a gamma_k forward-error bound. Rounded
        # TF32 inputs would instead produce exactly k and exceed it by >10x.
        bound = size * epsilon / (1 - size * epsilon)
        tf32_error = (expected - size).abs() / magnitude
        self.assertGreater(tf32_error.max().item(), 10 * bound)
        for policy in (
            "classical",
            "lt",
            "auto",
            "strassen",
            "strassen2",
            "strassen3",
            "strassen4",
        ):
            with cb.algorithm(policy):
                actual = cb.matmul(a, b).double().cpu()
            error = (actual - expected).abs() / magnitude
            self.assertLess(error.max().item(), bound)

    def test_automatic_strassen_boundaries(self):
        """Check even, odd and threshold routing against every reference element."""
        for dtype, threshold in ((torch.float32, 4096), (torch.float64, 8192)):
            for size in (threshold - 2, threshold, threshold + 1):
                a = torch.randn((size, size), device="cuda", dtype=dtype) / size**0.5
                b = torch.randn_like(a) / size**0.5
                cb.stats(reset=True)
                with cb.algorithm("auto"):
                    actual = cb.matmul(a, b)
                self.assert_close(actual, a @ b, strassen=True)
                self.assertEqual(cb.stats()["strassen"], int(size == threshold))
                del a, b, actual

    def test_c_api_padding_and_null_inputs(self):
        """Preserve C padding and avoid reading operands when alpha is zero."""
        native = _native.context(0)
        lib = _native.library()
        for dtype, call in (
            (torch.float32, lib.camblas_cuda_sgemm),
            (torch.float64, lib.camblas_cuda_dgemm),
        ):
            # Five columns, seven rows of allocation, four logical rows.
            c = torch.full((5, 7), float("nan"), device="cuda", dtype=dtype)
            with cb.algorithm("classical"):
                status = call(
                    native.handle,
                    b"N",
                    b"N",
                    4,
                    5,
                    9,
                    0,
                    None,
                    4,
                    None,
                    9,
                    0,
                    c.data_ptr(),
                    7,
                )
            self.assertEqual(status, 0)
            self.assertEqual(torch.count_nonzero(c[:, :4]).item(), 0)
            self.assertTrue(torch.isnan(c[:, 4:]).all().item())
            before = c.clone()
            self.assertEqual(
                call(
                    native.handle,
                    b"N",
                    b"N",
                    4,
                    5,
                    9,
                    0,
                    None,
                    4,
                    None,
                    9,
                    1,
                    c.data_ptr(),
                    7,
                ),
                0,
            )
            self.assert_close(c, before)
            self.assertLess(
                call(
                    native.handle,
                    b"?",
                    b"N",
                    4,
                    5,
                    9,
                    1,
                    None,
                    4,
                    None,
                    9,
                    0,
                    c.data_ptr(),
                    7,
                ),
                0,
            )

    def test_affine_mlp_and_parameter_gradients(self):
        """Compare all forward values and all input and parameter gradients."""
        for dtype in (torch.float32, torch.float64):
            inputs = [
                torch.randn(s, device="cuda", dtype=dtype).requires_grad_()
                for s in ((11, 17), (17, 23), (23,), (23, 13), (13,))
            ]
            reference_inputs = [x.detach().clone().requires_grad_() for x in inputs]
            x, w1, b1, w2, b2 = reference_inputs
            reference = torch.relu(x @ w1 + b1) @ w2 + b2
            actual = cb.mlp(*inputs)
            self.assert_close(actual, reference)
            weight = torch.randn_like(actual)
            (actual * weight).sum().backward()
            (reference * weight).sum().backward()
            for tensor, oracle in zip(inputs, reference_inputs):
                self.assert_close(tensor.grad, oracle.grad)
            for relu in (False, True):
                x, w1, b1 = (x.detach().clone().requires_grad_() for x in inputs[:3])
                expected = x @ w1 + b1
                if relu:
                    expected = torch.relu(expected)
                self.assert_close(cb.affine(x, w1, b1, relu=relu), expected)

    def test_affine_epilogue_inputs_and_exceptional_values(self):
        """Check bias/ReLU plans against CPU doubles after inputs and policies change."""
        for dtype in (torch.float32, torch.float64):
            for rows, inner, columns in ((1, 17, 23), (13, 17, 1), (13, 17, 23)):
                x = torch.randn((rows, inner), dtype=dtype) * 0.1
                weight = torch.randn((inner, columns), dtype=dtype) * 0.1
                bias = torch.randn((columns,), dtype=dtype) * 0.1
                for policy in ("auto", "lt", "classical"):
                    with cb.algorithm(policy):
                        for relu in (False, True, False):
                            for changed in (False, True):
                                actual_x = x + (0.03 if changed else 0)
                                actual_weight = weight * (0.7 if changed else 1)
                                actual_bias = bias + (0.04 if changed else 0)
                                expected = (
                                    actual_x.double() @ actual_weight.double()
                                    + actual_bias.double()
                                )
                                if relu:
                                    expected = torch.relu(expected)
                                self.assert_close(
                                    cb.affine(
                                        actual_x.cuda(),
                                        actual_weight.cuda(),
                                        actual_bias.cuda(),
                                        relu=relu,
                                    ).cpu(),
                                    expected.to(dtype),
                                )
                        for value in (float("nan"), float("inf"), -float("inf")):
                            for operand in ("x", "weight", "bias"):
                                actual_x, actual_weight, actual_bias = (
                                    tensor.clone() for tensor in (x, weight, bias)
                                )
                                target = {
                                    "x": actual_x,
                                    "weight": actual_weight,
                                    "bias": actual_bias,
                                }[operand]
                                target.reshape(-1)[0] = value
                                expected = (
                                    actual_x.double() @ actual_weight.double()
                                    + actual_bias.double()
                                )
                                for relu in (False, True):
                                    self.assert_close(
                                        cb.affine(
                                            actual_x.cuda(),
                                            actual_weight.cuda(),
                                            actual_bias.cuda(),
                                            relu=relu,
                                        ).cpu(),
                                        (torch.relu(expected) if relu else expected).to(
                                            dtype
                                        ),
                                    )

    def test_gradcheck_and_gradgradcheck(self):
        """Check analytical matmul derivatives against finite differences."""
        a = torch.randn((3, 4), device="cuda", dtype=torch.float64, requires_grad=True)
        b = torch.randn((4, 2), device="cuda", dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(cb.matmul, (a, b)))
        self.assertTrue(torch.autograd.gradgradcheck(cb.matmul, (a, b)))
        w1 = torch.randn((4, 5), device="cuda", dtype=torch.float64, requires_grad=True)
        b1 = torch.randn((5,), device="cuda", dtype=torch.float64, requires_grad=True)
        w2 = torch.randn((5, 2), device="cuda", dtype=torch.float64, requires_grad=True)
        b2 = torch.randn((2,), device="cuda", dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(cb.mlp, (a, w1, b1, w2, b2)))

    def test_independent_streams(self):
        """Use different contexts and scratch for overlapping CUDA streams."""
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        values = []
        for i, stream in enumerate(streams):
            with torch.cuda.stream(stream), cb.algorithm("strassen"):
                a = torch.randn((512, 512), device="cuda") / 512**0.5
                b = torch.randn_like(a) / 512**0.5
                values.append((cb.matmul(a, b), a @ b, _native.context(0).handle.value))
        for stream in streams:
            stream.synchronize()
        self.assertNotEqual(values[0][2], values[1][2])
        for actual, reference, _ in values:
            self.assert_close(actual, reference, strassen=True)

    def test_other_device_and_current_device_restoration(self):
        """Operate on another GPU without changing the caller's active device."""
        if torch.cuda.device_count() < 2:
            self.skipTest("Two allocated CUDA devices are required")
        original = torch.cuda.current_device()
        other = (original + 1) % torch.cuda.device_count()
        for dtype in (torch.float32, torch.float64):
            a = torch.randn((17, 23), device=f"cuda:{other}", dtype=dtype)
            b = torch.randn((23, 13), device=f"cuda:{other}", dtype=dtype)
            self.assert_close(cb.matmul(a, b), a @ b)
            self.assertEqual(torch.cuda.current_device(), original)
            self.assertEqual(cb.stats(device=other)["strassen"], 0)
            self.assertEqual(torch.cuda.current_device(), original)
            with self.assertRaises(ValueError):
                cb.matmul(a, b.to(f"cuda:{original}"))
        host = [torch.randn((64, 256), dtype=torch.float64) / 16 for _ in range(3)]
        expected = torch.softmax((host[0] @ host[1].T) / 16, -1) @ host[2]
        inputs = [value.to(f"cuda:{other}") for value in host]
        self.assert_close(cb.attention(*inputs).cpu(), expected)
        self.assertEqual(torch.cuda.current_device(), original)

    def test_affine_higher_derivatives(self):
        """Preserve differentiable affine gradients away from the ReLU boundary."""
        x = torch.randn((3, 4), device="cuda", dtype=torch.float64) * 0.1
        w = torch.randn((4, 2), device="cuda", dtype=torch.float64) * 0.1
        b = torch.full((2,), 0.5, device="cuda", dtype=torch.float64)
        inputs = tuple(value.requires_grad_(True) for value in (x, w, b))
        for relu in (False, True):

            def call(x, w, b):
                """Evaluate the affine operation used by the graph-lifetime test.

                Parameters
                ----------
                x, w, b : torch.Tensor
                    Prepared input, weight and bias operands.

                Returns
                -------
                torch.Tensor
                    Affine output with the test's selected ReLU setting.
                """
                return cb.affine(x, w, b, relu=relu)

            self.assertTrue(torch.autograd.gradcheck(call, inputs))
            self.assertTrue(torch.autograd.gradgradcheck(call, inputs))

    def test_attention_graph_scratch_growth(self):
        """Replay captured attention after its context grows private score storage."""
        for dtype in (torch.float32, torch.float64):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                q = torch.randn((17, 16), device="cuda", dtype=dtype)
                k = torch.randn((513, 16), device="cuda", dtype=dtype)
                v = torch.randn((513, 13), device="cuda", dtype=dtype)
                for _ in range(3):
                    cb.attention(q, k, v)
            stream.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = cb.attention(q, k, v)
            with torch.cuda.stream(stream):
                larger_q = torch.randn((129, 16), device="cuda", dtype=dtype)
                larger_k = torch.randn((4097, 16), device="cuda", dtype=dtype)
                larger_v = torch.randn((4097, 13), device="cuda", dtype=dtype)
                cb.attention(larger_q, larger_k, larger_v)
                for _ in range(3):
                    q.add_(0.1)
                    v.mul_(0.9)
                    graph.replay()
                    reference = torch.softmax((q @ k.T) / 4, dim=-1) @ v
                    self.assert_close(captured, reference)
            stream.synchronize()

    def test_mlp_training_graph_scratch_growth(self):
        """Replay captured gradients after input changes and bias-scratch growth."""
        for dtype in (torch.float32, torch.float64):
            shapes = ((129, 17), (17, 23), (23,), (23, 13), (13,))
            inputs = [
                torch.randn(s, device="cuda", dtype=dtype, requires_grad=True)
                for s in shapes
            ]
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(stream):
                for _ in range(3):
                    for value in inputs:
                        value.grad = None
                    cb.mlp(*inputs).square().mean().backward()
                for value in inputs:
                    value.grad = None
                with torch.cuda.graph(graph, stream=stream):
                    output = cb.mlp(*inputs)
                    output.square().mean().backward()
                larger = [
                    torch.randn(s, device="cuda", dtype=dtype, requires_grad=True)
                    for s in ((513, 17), (17, 23), (23,), (23, 13), (13,))
                ]
                cb.mlp(*larger).square().mean().backward()
                for _ in range(3):
                    with torch.no_grad():
                        inputs[0].add_(0.03)
                        inputs[1].mul_(0.9)
                        inputs[2].sub_(0.01)
                    graph.replay()
                    reference = [
                        value.detach().clone().requires_grad_(True) for value in inputs
                    ]
                    x, w1, b1, w2, b2 = reference
                    expected = torch.relu(x @ w1 + b1) @ w2 + b2
                    expected.square().mean().backward()
                    self.assert_close(output, expected)
                    for actual, desired in zip(inputs, reference):
                        self.assert_close(actual.grad, desired.grad)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph.reset()

    def test_attention_preserves_live_score_storage(self):
        """Keep score operands intact when the second product is a Strassen-size square."""
        for dtype in (torch.float32, torch.float64):
            q = torch.randn((512, 16), device="cuda", dtype=dtype) * 0.1
            k = torch.randn_like(q) * 0.1
            v = torch.randn((512, 512), device="cuda", dtype=dtype) * 0.1
            reference = torch.softmax((q @ k.T) / 4, dim=-1) @ v
            with cb.algorithm("strassen2"):
                cb.stats(reset=True)
                self.assert_close(cb.attention(q, k, v), reference)
                self.assertEqual(cb.stats()["strassen"], 0)
                q.add_(0.2)
                v.mul_(0.8)
                reference = torch.softmax((q @ k.T) / 4, dim=-1) @ v
                self.assert_close(cb.attention(q, k, v), reference)

    def test_attention_scratch_size_overflow(self):
        """Reject an overflowing FP64 score allocation before touching operands."""
        library = _native.library()
        context = _native.context(torch.cuda.current_device())
        buffer = torch.full((2,), 7.0, device="cuda", dtype=torch.float64)
        status = library.camblas_cuda_attention(
            context.handle,
            1,
            2**31 - 1,
            2**30 + 1,
            0,
            1,
            1.0,
            None,
            None,
            buffer.data_ptr(),
            buffer.data_ptr() + buffer.element_size(),
        )
        self.assertLess(status, 0)
        self.assertIn(
            "scratch size overflow", library.camblas_cuda_error(context.handle).decode()
        )
        torch.cuda.synchronize()
        self.assert_close(buffer, torch.full_like(buffer, 7.0))

    def test_coherent_host_operations(self):
        """Read CPU tensors on supported GPUs and return fully completed CPU results."""
        module = _native.tensor_module()
        if module is None:
            self.skipTest("Coherent host operations require the native tensor binding")
        try:
            module.matmul_host(torch.eye(2), torch.eye(2))
        except ValueError as error:
            if "host page tables" in str(error):
                self.skipTest(str(error))
            raise
        for dtype in (torch.float32, torch.float64):
            a = torch.randn((11, 17), dtype=dtype)
            b = torch.randn((17, 13), dtype=dtype)
            for left in (a, a.T.contiguous().T, a[:, ::2]):
                right = b if left.shape[1] == 17 else b[::2]
                expected = (left.cuda() @ right.cuda()).cpu()
                self.assert_close(module.matmul_host(left, right), expected)
                c = torch.randn_like(expected)
                reference = 0.7 * expected - 0.3 * c
                self.assert_close(
                    module.matmul_host(left, right, out=c, alpha=0.7, beta=-0.3),
                    reference,
                )
            a.add_(0.25)
            b.mul_(0.5)
            self.assert_close(module.matmul_host(a, b), (a.cuda() @ b.cuda()).cpu())
            x, w1, b1, w2, b2 = (
                torch.randn(s, dtype=dtype)
                for s in ((11, 17), (17, 23), (23,), (23, 13), (13,))
            )
            expected = cb.mlp(*(t.cuda() for t in (x, w1, b1, w2, b2))).cpu()
            self.assert_close(module.mlp_host(x, w1, b1, w2, b2), expected)
            q, k, v = (
                torch.randn(s, dtype=dtype) for s in ((11, 17), (19, 17), (19, 13))
            )
            expected = cb.attention(q.cuda(), k.cuda(), v.cuda(), scale=0.25).cpu()
            self.assert_close(module.attention_host(q, k, v, scale=0.25), expected)
            q.add_(0.1)
            expected = cb.attention(q.cuda(), k.cuda(), v.cuda(), scale=0.25).cpu()
            self.assert_close(module.attention_host(q, k, v, scale=0.25), expected)
        q, k, v = [torch.randn((64, 256), dtype=torch.float64) / 16 for _ in range(3)]
        expected = torch.softmax((q @ k.T) / 16, -1) @ v
        self.assert_close(module.attention_host(q, k, v), expected)
        q.add_(0.03)
        v.mul_(0.7)
        expected = torch.softmax((q @ k.T) / 16, -1) @ v
        self.assert_close(module.attention_host(q, k, v), expected)
        with self.assertRaises(ValueError):
            module.matmul_host(a.requires_grad_(True), b)

    def test_attention_and_derivatives(self):
        """Check unequal sequence lengths, empty keys, exceptional values and gradients."""
        for dtype in (torch.float32, torch.float64):
            q = torch.randn((13, 17), device="cuda", dtype=dtype)
            k = torch.randn((19, 17), device="cuda", dtype=dtype)
            v = torch.randn((19, 23), device="cuda", dtype=dtype)
            for scale in (0.25, -0.5, 0):
                reference = torch.softmax((q @ k.T) * scale, dim=-1) @ v
                self.assert_close(cb.attention(q, k, v, scale=scale), reference)
            k0 = torch.empty((0, 17), device="cuda", dtype=dtype)
            v0 = torch.empty((0, 23), device="cuda", dtype=dtype)
            self.assert_close(
                cb.attention(q, k0, v0),
                torch.zeros((13, 23), device="cuda", dtype=dtype),
            )
            q[0, 0] = float("nan")
            reference = torch.softmax((q @ k.T) * 0, dim=-1) @ v
            self.assert_close(cb.attention(q, k, v, scale=0), reference)
        q = torch.randn((3, 4), device="cuda", dtype=torch.float64, requires_grad=True)
        k = torch.randn((5, 4), device="cuda", dtype=torch.float64, requires_grad=True)
        v = torch.randn((5, 2), device="cuda", dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(cb.attention, (q, k, v)))

    def test_attention_cached_row_boundaries(self):
        """Compare complete attention outputs at register and loop-path boundaries."""
        widths = (
            1,
            17,
            255,
            256,
            257,
            511,
            512,
            513,
            1023,
            1024,
            1025,
            2047,
            2048,
            2049,
            4095,
            4096,
            4097,
        )
        for dtype in (torch.float32, torch.float64):
            for keys in widths:
                q = torch.randn((7, 17), dtype=dtype) * 0.1
                k = torch.randn((keys, 17), dtype=dtype) * 0.1
                v = torch.randn((keys, 23), dtype=dtype) * 0.1
                for scale in (0.25, -0.5, 0, 256):
                    expected = (
                        torch.softmax((q.double() @ k.double().T) * scale, dim=-1)
                        @ v.double()
                    )
                    self.assert_close(
                        cb.attention(q.cuda(), k.cuda(), v.cuda(), scale=scale).cpu(),
                        expected.to(dtype),
                    )
                q.add_(0.03)
                v.mul_(0.7)
                expected = (
                    torch.softmax((q.double() @ k.double().T) * 0.25, dim=-1)
                    @ v.double()
                )
                self.assert_close(
                    cb.attention(q.cuda(), k.cuda(), v.cuda(), scale=0.25).cpu(),
                    expected.to(dtype),
                )
                if keys in (257, 1025, 4096, 4097):
                    q[0, 0] = float("nan")
                    expected = (
                        torch.softmax((q.double() @ k.double().T) * 0, dim=-1)
                        @ v.double()
                    )
                    self.assert_close(
                        cb.attention(q.cuda(), k.cuda(), v.cuda(), scale=0).cpu(),
                        expected.to(dtype),
                    )

    def test_attention_short_fp64_oracle(self):
        """Check complete CPU-double outputs, boundaries and exceptional short inputs."""
        for queries, keys in (
            (8, 32),
            (24, 32),
            (64, 64),
            (1024, 64),
            (9, 32),
            (64, 31),
            (64, 65),
            (64, 128),
        ):
            host = [
                torch.randn((rows, 256), dtype=torch.float64) / 16
                for rows in (queries, keys, keys)
            ]
            inputs = [value.cuda() for value in host]
            for scale in (0, 1 / 16, -0.25, 256):
                expected = torch.softmax((host[0] @ host[1].T) * scale, -1) @ host[2]
                self.assert_close(cb.attention(*inputs, scale=scale).cpu(), expected)
            inputs[0].add_(0.03)
            inputs[1].mul_(0.7)
            inputs[2].sub_(0.01)
            host = [value.cpu() for value in inputs]
            expected = torch.softmax((host[0] @ host[1].T) / 16, -1) @ host[2]
            self.assert_close(cb.attention(*inputs).cpu(), expected)
            if (queries, keys) == (64, 64):
                for index, value, scale in (
                    (0, float("nan"), 0),
                    (1, float("inf"), 1 / 16),
                    (2, float("nan"), 1 / 16),
                    (2, float("inf"), 1 / 16),
                ):
                    exceptional = [item.clone() for item in host]
                    exceptional[index][0, 0] = value
                    expected = (
                        torch.softmax((exceptional[0] @ exceptional[1].T) * scale, -1)
                        @ exceptional[2]
                    )
                    self.assert_close(
                        cb.attention(
                            *(item.cuda() for item in exceptional), scale=scale
                        ).cpu(),
                        expected,
                    )

    def test_attention_short_fp64_range_and_mantissa(self):
        """Keep large finite averages finite and retain FP64 input mantissas."""
        for keys in (32, 64):
            q = torch.zeros((64, 256), dtype=torch.float64)
            k = torch.zeros((keys, 256), dtype=torch.float64)
            v = torch.full_like(k, torch.finfo(torch.float64).max / 8)
            expected = torch.full_like(q, torch.finfo(torch.float64).max / 8)
            actual = cb.attention(q.cuda(), k.cuda(), v.cuda()).cpu()
            self.assert_close(actual, expected)
            q[:, 0] = 2.0**40
            k[:, 0] = 1 + torch.arange(keys, dtype=torch.float64) * 2.0**-40
            v = (
                torch.arange(keys, dtype=torch.float64)[:, None]
                .expand(-1, 256)
                .contiguous()
                / keys
            )
            scores = q @ k.T
            expected = torch.softmax(scores, -1) @ v
            actual = cb.attention(q.cuda(), k.cuda(), v.cuda(), scale=1).cpu()
            self.assert_close(actual, expected)
            rounded = torch.softmax(q.float().double() @ k.float().double().T, -1) @ v
            self.assertGreater((expected - rounded).abs().max().item(), 0.1)

    def test_attention_short_graph_and_alias(self):
        """Recompute graph inputs and preserve supported C-API output overlaps."""
        host = [torch.randn((64, 256), dtype=torch.float64) / 16 for _ in range(3)]
        q, k, v = [value.cuda() for value in host]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                cb.attention(q, k, v)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = cb.attention(q, k, v)
        with torch.cuda.stream(stream):
            q.add_(0.03)
            k.mul_(0.7)
            v.sub_(0.01)
            graph.replay()
        stream.synchronize()
        host = [value.cpu() for value in (q, k, v)]
        expected = torch.softmax((host[0] @ host[1].T) / 16, -1) @ host[2]
        self.assert_close(captured.cpu(), expected)
        native = _native.context(0)
        call = _native.library().camblas_cuda_attention
        for partial in (False, True):
            storage = torch.empty((64 * 256 + 8,), device="cuda", dtype=torch.float64)
            source = storage[: 64 * 256].view(64, 256)
            source.copy_(k)
            output = storage[8:].view(64, 256) if partial else source
            _native._check(
                call(
                    native.handle,
                    1,
                    64,
                    64,
                    256,
                    256,
                    1 / 16,
                    q.data_ptr(),
                    source.data_ptr(),
                    v.data_ptr(),
                    output.data_ptr(),
                ),
                native.handle,
            )
            self.assert_close(output.cpu(), expected)

    def test_native_control_override_routes_copied_binding(self):
        """Load a saved core even when its copied binding points at the build directory."""
        module = _native.tensor_module()
        if module is None:
            self.skipTest("The optional native tensor binding is required")
        root = Path(__file__).resolve().parents[1]
        code = """
import os
from pathlib import Path
import torch
import camblas._kernels as cb
directory = Path(os.environ['CAMBLAS_TEST_CONTROL_DIRECTORY'])
a = torch.ones((2, 2), device='cuda', dtype=torch.float64)
torch.testing.assert_close(cb.matmul(a, a), torch.full_like(a, 2))
paths = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
         if 'libcamblas_cuda.so' in line or '_camblas_cuda_torch' in line}
assert paths and all(Path(path).parent == directory for path in paths), paths
assert len(paths) == 2, paths
"""
        with tempfile.TemporaryDirectory(prefix="camblas-native-control-") as temporary:
            directory = Path(temporary).resolve()
            core = directory / "libcamblas_cuda.so"
            shutil.copy2(_native.library()._name, core)
            binding = directory / Path(module.__file__).name
            # Exercise older copied bindings with an absolute build RUNPATH.
            # Replace the optional sibling search component without resizing ELF data.
            binding.write_bytes(
                Path(module.__file__).read_bytes().replace(b"$ORIGIN", b"/absent")
            )
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=root,
                env=dict(
                    os.environ,
                    CAMBLAS_CUDA_LIBRARY=str(core),
                    CAMBLAS_TEST_CONTROL_DIRECTORY=str(directory),
                ),
                capture_output=True,
                text=True,
                timeout=120,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_cancellation_against_cpu_oracle(self):
        """Bound Strassen error by product magnitudes for cancelling dot products."""
        for dtype in (torch.float32, torch.float64):
            a = torch.randn((256, 256), dtype=dtype) * 1e4
            b = torch.randn((256, 256), dtype=dtype) * 1e4
            a[:, 128:] = a[:, :128]
            b[128:, :] = -b[:128, :]
            b[0, 0] += 0.1
            expected = a.double() @ b.double()
            magnitudes = a.double().abs() @ b.double().abs()
            for policy in ("strassen", "strassen2", "strassen3", "strassen4"):
                with cb.algorithm(policy):
                    actual = cb.matmul(a.cuda(), b.cuda()).cpu().double()
                scaled = (actual - expected).abs() / magnitudes.clamp_min(1)
                self.assertLess(scaled.max().item(), 50 * torch.finfo(dtype).eps)

    def test_two_level_strassen(self):
        """Check both recombination levels and their classical range fallback."""
        for dtype in (torch.float32, torch.float64):
            a = torch.randn((512, 512), device="cuda", dtype=dtype) / 512**0.5
            b = torch.randn_like(a) / 512**0.5
            with cb.algorithm("strassen2"):
                self.assert_close(cb.matmul(a, b), a @ b, strassen=True)
                c = torch.randn_like(a)
                reference = 0.3 * (a @ b) + 0.2 * c
                cb.matmul(a, b, out=c, alpha=0.3, beta=0.2)
                self.assert_close(c, reference, strassen=True)
                a[0, 0] = float("nan")
                cb.stats(reset=True)
                self.assert_close(cb.matmul(a, b), a @ b)
                self.assertEqual(cb.stats()["guard_fallback"], 1)

    def test_three_level_strassen(self):
        """Check CPU-double accuracy, level boundaries, alpha/beta and changed inputs."""
        for dtype in (torch.float32, torch.float64):
            for n in (504, 512, 514, 516):
                host_a = torch.randn((n, n + 3), dtype=dtype) / n**0.5
                host_b = torch.randn((n, n + 5), dtype=dtype) / n**0.5
                a, b = host_a.cuda()[:, :n], host_b.cuda()[:, :n]
                expected = host_a[:, :n].double() @ host_b[:, :n].double()
                with cb.algorithm("strassen3"):
                    self.assert_close(
                        cb.matmul(a, b).cpu(), expected.to(dtype), strassen=True
                    )
                    c = torch.randn((n, n), device="cuda", dtype=dtype)
                    reference = 0.3 * expected + 0.2 * c.double().cpu()
                    cb.matmul(a, b, out=c, alpha=0.3, beta=0.2)
                    self.assert_close(c.cpu(), reference.to(dtype), strassen=True)
                    a.add_(0.02)
                    b.mul_(0.8)
                    expected = a.double().cpu() @ b.double().cpu()
                    self.assert_close(
                        cb.matmul(a, b).cpu(), expected.to(dtype), strassen=True
                    )
                if n == 512:
                    for value in (
                        float("nan"),
                        float("inf"),
                        torch.finfo(dtype).max / 2,
                    ):
                        exceptional = a.clone()
                        exceptional[0, 0] = value
                        with cb.algorithm("classical"):
                            reference = cb.matmul(exceptional, b)
                        cb.stats(reset=True)
                        with cb.algorithm("strassen3"):
                            self.assert_close(cb.matmul(exceptional, b), reference)
                        counts = cb.stats()
                        self.assertEqual(counts["guard_fallback"], 1)
                        self.assertEqual(counts["classical"], 1)
                        self.assertEqual(counts["strassen"], 0)

    def test_four_level_strassen(self):
        """Check CPU-double accuracy, level boundaries, alpha/beta and changed inputs."""
        for dtype in (torch.float32, torch.float64):
            for n in (504, 512, 514, 516, 528):
                host_a = torch.randn((n, n + 3), dtype=dtype) / n**0.5
                host_b = torch.randn((n, n + 5), dtype=dtype) / n**0.5
                a, b = host_a.cuda()[:, :n], host_b.cuda()[:, :n]
                expected = host_a[:, :n].double() @ host_b[:, :n].double()
                with cb.algorithm("strassen4"):
                    self.assert_close(
                        cb.matmul(a, b).cpu(), expected.to(dtype), strassen=True
                    )
                    c = torch.randn((n, n), device="cuda", dtype=dtype)
                    reference = 0.3 * expected + 0.2 * c.double().cpu()
                    cb.matmul(a, b, out=c, alpha=0.3, beta=0.2)
                    self.assert_close(c.cpu(), reference.to(dtype), strassen=True)
                    a.add_(0.02)
                    b.mul_(0.8)
                    expected = a.double().cpu() @ b.double().cpu()
                    self.assert_close(
                        cb.matmul(a, b).cpu(), expected.to(dtype), strassen=True
                    )
                if n == 512:
                    for value in (
                        float("nan"),
                        float("inf"),
                        torch.finfo(dtype).max / 2,
                    ):
                        exceptional = a.clone()
                        exceptional[0, 0] = value
                        with cb.algorithm("classical"):
                            reference = cb.matmul(exceptional, b)
                        cb.stats(reset=True)
                        with cb.algorithm("strassen4"):
                            self.assert_close(cb.matmul(exceptional, b), reference)
                        counts = cb.stats()
                        self.assertEqual(counts["guard_fallback"], 1)
                        self.assertEqual(counts["classical"], 1)
                        self.assertEqual(counts["strassen"], 0)

    def test_four_level_large_existing_output(self):
        """Apply beta to C only after cancelling unscaled Strassen contributions."""
        n = 512
        for dtype, exponent in ((torch.float32, 52), (torch.float64, 498)):
            magnitude = 2.0**exponent
            a_host = torch.full((n, n), magnitude, dtype=dtype)
            b_host = torch.full((n, n), magnitude, dtype=dtype)
            b_host[n // 2 :] = -magnitude
            c_host = torch.full((n, n), torch.finfo(dtype).max, dtype=dtype)
            product = a_host.double() @ b_host.double()
            self.assertEqual(product.count_nonzero().item(), 0)
            a, b = a_host.cuda(), b_host.cuda()
            for alpha, beta in ((1, 1), (-0.5, 1), (0.03125, -1), (1, 0)):
                c = (c_host if beta else torch.full_like(c_host, float("nan"))).cuda()
                cb.stats(reset=True)
                with cb.algorithm("strassen4"):
                    cb.matmul(a, b, out=c, alpha=alpha, beta=beta)
                expected = alpha * product + (beta * c_host.double() if beta else 0)
                torch.testing.assert_close(c.cpu(), expected.to(dtype), rtol=0, atol=0)
                self.assertEqual(cb.stats()["strassen"], 1)
                self.assertEqual(cb.stats()["guard_fallback"], 0)

    def test_recursive_strassen_padding(self):
        """Cover recursive leaf boundaries, C padding, alpha/beta and changed operands."""
        for dtype, scalar in ((torch.float32, "sgemm"), (torch.float64, "dgemm")):
            for policy in ("strassen2", "strassen3", "strassen4"):
                for n in (504, 508, 512, 514, 516, 528):
                    a = torch.randn((n, n + 3), device="cuda", dtype=dtype) / n**0.5
                    b = torch.randn((n, n + 5), device="cuda", dtype=dtype) / n**0.5
                    c = torch.full((n, n + 7), 1234.0, device="cuda", dtype=dtype)
                    expected = 0.3 * (a[:, :n].T @ b[:, :n].T) + 0.2 * c[:, :n].T
                    with cb.algorithm(policy):
                        native = _native.context(0)
                        call = getattr(_native.library(), "camblas_cuda_" + scalar)
                        _native._check(
                            call(
                                native.handle,
                                b"N",
                                b"N",
                                n,
                                n,
                                n,
                                0.3,
                                a.data_ptr(),
                                n + 3,
                                b.data_ptr(),
                                n + 5,
                                0.2,
                                c.data_ptr(),
                                n + 7,
                            ),
                            native.handle,
                        )
                        self.assert_close(c[:, :n].T, expected, strassen=True)
                        self.assertTrue((c[:, n:] == 1234).all().item())
                        a.add_(0.02)
                        b.mul_(0.8)
                        self.assert_close(
                            cb.matmul(a[:, :n], b[:, :n]),
                            a[:, :n] @ b[:, :n],
                            strassen=True,
                        )

    def test_out_tensor_versions(self):
        """Reject differentiable out tensors and detect mutation of saved operands."""
        a = torch.eye(4, device="cuda")
        output = torch.ones_like(a, requires_grad=True)
        with self.assertRaises(ValueError):
            cb.matmul(a, a, out=output)
        with torch.no_grad():
            version = output._version
            cb.matmul(a, a, out=output)
            self.assertEqual(output._version, version + 1)
        output = torch.ones_like(a)
        parameter = torch.ones_like(a, requires_grad=True)
        loss = (parameter * output).sum()
        cb.matmul(a, a, out=output)
        with self.assertRaisesRegex(RuntimeError, "modified by an inplace operation"):
            loss.backward()
        with torch.inference_mode():
            output = torch.empty_like(a)
            cb.matmul(a, a, out=output)
            self.assert_close(output, a)

    def test_backward_policy_and_counters(self):
        """Preserve explicit algorithms across PyTorch's autograd host workers."""
        inputs = [
            torch.randn(s, device="cuda", requires_grad=True)
            for s in ((5, 7), (7, 11), (11,), (11, 3), (3,))
        ]
        cb.stats(reset=True, all_threads=True)
        with cb.algorithm("classical"):
            output = cb.mlp(*inputs)
        output.sum().backward()
        counts = cb.stats(all_threads=True)
        self.assertEqual(counts["backward"], int(_native.tensor_module() is not None))
        self.assertEqual(counts["lt"], 0)
        self.assertEqual(counts["classical"], 6)

    def test_cuda_graph_recomputes_changed_inputs(self):
        """Fall back from guarded Strassen to graph-safe classical multiplication."""
        a = torch.randn((256, 256), device="cuda") / 16
        b = torch.randn_like(a) / 16
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream), cb.algorithm("strassen"):
            for _ in range(3):
                cb.matmul(a, b)
            with torch.cuda.graph(graph, stream=stream):
                output = cb.matmul(a, b)
        torch.cuda.current_stream().wait_stream(stream)
        graph.replay()
        self.assert_close(output, a @ b)
        a.add_(0.1)
        b.mul_(0.5)
        graph.replay()
        self.assert_close(output, a @ b)

    def test_matmul_gradient_policy_and_subsets(self):
        """Keep the forward policy when workers compute either requested gradient."""
        for requested in ((True, False), (False, True), (True, True)):
            a = torch.randn((7, 11), device="cuda", dtype=torch.float64)
            b = torch.randn((11, 5), device="cuda", dtype=torch.float64)
            a.requires_grad_(requested[0])
            b.requires_grad_(requested[1])
            reference_a = a.detach().clone().requires_grad_(requested[0])
            reference_b = b.detach().clone().requires_grad_(requested[1])
            cb.stats(reset=True, all_threads=True)
            with cb.algorithm("classical"):
                output = cb.matmul(a, b)
            output.sum().backward()
            (reference_a @ reference_b).sum().backward()
            counts = cb.stats(all_threads=True)
            self.assertEqual(counts["classical"], 1 + sum(requested))
            self.assertEqual(counts["lt"], 0)
            if requested[0]:
                self.assert_close(a.grad, reference_a.grad)
            if requested[1]:
                self.assert_close(b.grad, reference_b.grad)
            with self.assertRaises(ValueError):
                cb.matmul(a, b, alpha=0.5)
            with torch.no_grad():
                self.assert_close(cb.matmul(a, b, alpha=0.5), 0.5 * (a @ b))

    def test_process_exit_and_context_reuse(self):
        """Clean up autograd worker contexts before the CUDA runtime shuts down."""
        code = """
import torch
import camblas._kernels as cb
from camblas._native import close, tensor_module
torch.set_num_threads(4)
for repeat in range(2):
    for policy in ('classical', 'auto'):
        values = [torch.randn(s, device='cuda', requires_grad=True)
                  for s in ((11, 17), (17, 23), (23,), (23, 13), (13,))]
        with cb.algorithm(policy):
            output = cb.mlp(*values)
        output.square().mean().backward()
        assert all(torch.isfinite(x.grad).all() for x in values)
    assert cb.stats(all_threads=True)['backward'] == (2 if tensor_module() else 0)
    close()
    assert not any(cb.stats(all_threads=True).values())
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_unaligned_contiguous_views(self):
        """Select algorithms for actual data alignment after slicing tensor storage."""
        for dtype in (torch.float32, torch.float64):
            aligned_a = torch.randn((128, 256), device="cuda", dtype=dtype)
            aligned_b = torch.randn((256, 64), device="cuda", dtype=dtype)
            a = torch.randn((128 * 256 + 1,), device="cuda", dtype=dtype)[1:].view(
                128, 256
            )
            b = torch.randn((256 * 64 + 1,), device="cuda", dtype=dtype)[1:].view(
                256, 64
            )
            bias = torch.randn((65,), device="cuda", dtype=dtype)[1:]
            expected = a.double().cpu() @ b.double().cpu()
            magnitudes = a.double().cpu().abs() @ b.double().cpu().abs()
            bound = 8 * torch.finfo(dtype).eps * magnitudes.clamp_min(1)
            for policy in ("auto", "lt"):
                with cb.algorithm(policy):
                    cb.matmul(aligned_a, aligned_b)
                    actual = cb.matmul(a, b).double().cpu()
                    self.assertTrue(((actual - expected).abs() <= bound).all().item())
                    actual = cb.affine(a, b, bias).double().cpu()
                    affine_expected = expected + bias.double().cpu()
                    self.assertTrue(
                        ((actual - affine_expected).abs() <= bound).all().item()
                    )

    def test_fork_rejects_inherited_contexts(self):
        """Reject inherited CUDA state before using it and preserve the parent's handle."""
        if _native.tensor_module() is None:
            self.skipTest("The tensor binding provides the early fork check")
        code = """
import os
import torch
import camblas._kernels as cb
from camblas._native import close
torch.set_num_threads(1)
a = torch.eye(4, device='cuda')
cb.matmul(a, a)
torch.cuda.synchronize()
child = os.fork()
if child == 0:
    try:
        cb.matmul(a, a)
    except ValueError as error:
        if 'fork' not in str(error):
            os._exit(2)
    else:
        os._exit(3)
    close()
    os._exit(0)
pid, status = os.waitpid(child, 0)
assert os.waitstatus_to_exitcode(status) == 0, status
torch.testing.assert_close(cb.matmul(a, a), a)
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_mlp_requested_gradient_subsets(self):
        """Return correct gradients when only one input or parameter needs one."""
        for index in range(5):
            inputs = [
                torch.randn(s, device="cuda", dtype=torch.float64).requires_grad_(
                    i == index
                )
                for i, s in enumerate(((5, 7), (7, 11), (11,), (11, 3), (3,)))
            ]
            reference_inputs = [
                x.detach().clone().requires_grad_(i == index)
                for i, x in enumerate(inputs)
            ]
            x, w1, b1, w2, b2 = reference_inputs
            expected = torch.relu(x @ w1 + b1) @ w2 + b2
            actual = cb.mlp(*inputs)
            actual.sum().backward()
            expected.sum().backward()
            self.assert_close(inputs[index].grad, reference_inputs[index].grad)


@unittest.skipUnless(CUDA_AVAILABLE, "A CUDA-enabled PyTorch is required")
class CudaLargeTests(unittest.TestCase):
    """Optional complete-output verification at the four-level automatic threshold."""

    @unittest.skipUnless(
        os.environ.get("CAMBLAS_TEST_LARGE_CUDA") == "1",
        "Set CAMBLAS_TEST_LARGE_CUDA=1 for complete large-output checks",
    )
    def test_four_level_first_automatic_sizes(self):
        """Check exact identity products at the first FP32/FP64 automatic sizes."""
        if torch.cuda.get_device_properties(0).total_memory < 40 * 2**30:
            self.skipTest("The complete FP64 boundary check requires 40 GiB")
        torch.set_num_threads(1)
        for dtype, n in ((torch.float32, 12288), (torch.float64, 24576)):
            _native.close()
            torch.cuda.empty_cache()
            indices = torch.arange(n, device="cuda")
            blocks = indices.div(n // 16, rounding_mode="floor").to(dtype)
            a = torch.eye(n, device="cuda", dtype=dtype)
            b = 16 * blocks[:, None] + blocks[None, :] + 1
            c = torch.full_like(a, float("nan"))
            cb.stats(reset=True)
            torch.cuda.synchronize()
            before = torch.cuda.mem_get_info()[0]
            with cb.algorithm("auto"):
                cb.matmul(a, b, out=c)
            torch.cuda.synchronize()
            arena = 3 * 343 * (n // 16) ** 2 * a.element_size()
            used = before - torch.cuda.mem_get_info()[0]
            self.assertGreaterEqual(used, arena)
            self.assertLessEqual(used, arena + 512 * 2**20)
            self.assertTrue(torch.equal(c, b))
            self.assertEqual(cb.stats()["strassen"], 1)
            self.assertEqual(cb.stats()["guard_fallback"], 0)
            a.mul_(2)
            b.add_(0.5)
            with cb.algorithm("auto"):
                cb.matmul(a, b, out=c, alpha=0.5)
            self.assertTrue(torch.equal(c, b))
            self.assertEqual(cb.stats()["strassen"], 2)
            del a, b, c, indices, blocks
            _native.close()
            torch.cuda.empty_cache()

    @unittest.skipUnless(
        os.environ.get("CAMBLAS_TEST_LARGE_CUDA") == "1",
        "Set CAMBLAS_TEST_LARGE_CUDA=1 for complete large-output checks",
    )
    def test_four_level_distinct_quadrants(self):
        """Check every output with distinct quadrants, changed inputs and alpha/beta."""
        if torch.cuda.get_device_properties(0).total_memory < 40 * 2**30:
            self.skipTest("The complete FP64 alpha/beta check requires 40 GiB")
        torch.set_num_threads(1)
        n = 16384
        for dtype in (torch.float32, torch.float64):
            _native.close()
            torch.cuda.empty_cache()
            indices = torch.arange(n, device="cuda")
            rows = (1 + indices.div(n // 4, rounding_mode="floor")).to(dtype)
            columns = (1 + indices.div(n // 16, rounding_mode="floor")).to(dtype)
            keys = indices.div(n // 16, rounding_mode="floor").to(dtype)
            a = (rows[:, None] / n).expand(n, n).contiguous()
            b = keys[:, None] + columns[None, :]
            c = torch.full_like(a, float("nan"))
            expected = rows[:, None] * (columns[None, :] + 7.5)
            cb.stats(reset=True)
            with cb.algorithm("strassen4"):
                cb.matmul(a, b, out=c)
            self.assertTrue(torch.equal(c, expected))
            a.mul_(2)
            b.add_(0.5)
            with cb.algorithm("strassen4"):
                cb.matmul(a, b, out=c, alpha=-0.5, beta=0.125)
            changed = -rows[:, None] * (columns[None, :] + 8) + 0.125 * expected
            self.assertTrue(torch.equal(c, changed))
            self.assertEqual(cb.stats()["strassen"], 2)
            self.assertEqual(cb.stats()["guard_fallback"], 0)
            del a, b, c, expected, changed, indices, rows, columns, keys
            _native.close()
            torch.cuda.empty_cache()

    @unittest.skipUnless(
        os.environ.get("CAMBLAS_TEST_LARGE_CUDA") == "1",
        "Set CAMBLAS_TEST_LARGE_CUDA=1 for the 32768-square full-output check",
    )
    def test_four_level_automatic_full_output(self):
        """Check all billion entries against an analytic oracle and observe arena size."""
        if torch.cuda.get_device_properties(0).total_memory < 85 * 2**30:
            self.skipTest("The complete FP64 check requires an allocated 96-GiB GPU")
        torch.set_num_threads(1)
        n = 32768
        for dtype in (torch.float32, torch.float64):
            _native.close()
            torch.cuda.empty_cache()
            a = torch.full((n, n), 1 / n, device="cuda", dtype=dtype)
            pattern = torch.arange(n, device="cuda", dtype=dtype).remainder(32) / 32
            b = pattern.expand(n, n).contiguous()
            c = torch.empty_like(a)
            torch.cuda.synchronize()
            before = torch.cuda.mem_get_info()[0]
            cb.stats(reset=True)
            with cb.algorithm("auto"):
                cb.matmul(a, b, out=c)
            torch.cuda.synchronize()
            arena = 3 * 343 * (n // 16) ** 2 * a.element_size()
            used = before - torch.cuda.mem_get_info()[0]
            self.assertGreaterEqual(used, arena)
            self.assertLessEqual(used, arena + 512 * 2**20)
            self.assertTrue(torch.equal(c, pattern.expand(n, n)))
            self.assertEqual(cb.stats()["strassen"], 1)
            self.assertEqual(cb.stats()["guard_fallback"], 0)
            a.mul_(2)
            b.add_(1 / 32)
            with cb.algorithm("auto"):
                cb.matmul(a, b, out=c)
            expected = 2 * (pattern + 1 / 32)
            self.assertTrue(torch.equal(c, expected.expand(n, n)))
            self.assertEqual(cb.stats()["strassen"], 2)
            del a, b, c, pattern, expected
            _native.close()
            torch.cuda.empty_cache()


@unittest.skipUnless(CUDA_AVAILABLE, "A CUDA-enabled PyTorch is required")
class CudaLimitsTests(unittest.TestCase):
    """Exercise LP64 limits separately from ordinary kernel instrumentation."""

    def test_zero_row_bias_at_lp64_limit(self):
        """Fill the entire bias gradient at INT_MAX width without grid overflow."""
        if torch.cuda.mem_get_info()[0] < 24 * 2**30:
            self.skipTest("The LP64 boundary check requires 24 GiB of free GPU memory")
        library = _native.library()
        context = _native.context(torch.cuda.current_device())
        call = library.camblas_cuda_mlp_backward
        call.argtypes = [ct.c_void_p] + [ct.c_int] * 5 + [ct.c_void_p] * 11
        call.restype = ct.c_int
        for dtype, code in ((torch.float32, 0), (torch.float64, 1)):
            with self.subTest(dtype=dtype):
                output = torch.full(
                    (2**31 - 1,), float("nan"), device="cuda", dtype=dtype
                )
                status = call(
                    context.handle,
                    code,
                    0,
                    0,
                    0,
                    2**31 - 1,
                    *([None] * 10),
                    output.data_ptr(),
                )
                self.assertEqual(status, 0)
                self.assertEqual(torch.count_nonzero(output).item(), 0)
                del output


if __name__ == "__main__":
    unittest.main()
