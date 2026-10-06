"""Independent full-output checks for RMSNorm and gated-MLP inference fusion."""

import unittest

import torch
import torch.nn.functional as functional

from camblas import _native


class InferenceFusionTests(unittest.TestCase):
    """Cover storage rounding, exceptional values, views and graph replay."""

    @classmethod
    def setUpClass(cls):
        """Use CPU double arithmetic independently of CUDA kernels."""
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        cls.module = _native.tensor_module()
        if cls.module is None:
            raise unittest.SkipTest(
                "Inference fusion requires the native tensor binding"
            )

    def setUp(self):
        """Start every test from repeatable, independently changed values."""
        torch.manual_seed(705537)

    def test_rms_full_outputs_and_changed_values(self):
        """Compare every value with CPU double norms and explicit storage rounding."""
        for dtype in [torch.float32, torch.bfloat16]:
            for shape in [(13,), (3, 31), (2, 3, 8192), (0, 8192), (2, 0)]:
                x = torch.randn(shape, device="cuda", dtype=dtype)
                weight = torch.randn(shape[-1], device="cuda", dtype=dtype)
                for changed in [False, True]:
                    if changed:
                        x.mul_(-0.5)
                        weight.add_(0.25)
                    actual = self.module.rms_norm(x, weight, 1e-6)
                    host = x.cpu().double()
                    normalised = (
                        host * (host.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
                    ).to(dtype)
                    expected = (normalised.double() * weight.cpu().double()).to(dtype)
                    torch.testing.assert_close(
                        actual.cpu(),
                        expected,
                        rtol=0.015625 if dtype == torch.bfloat16 else 4e-6,
                        atol=0.0078125 if dtype == torch.bfloat16 else 2e-6,
                    )
                if x.numel():
                    storage = torch.empty(
                        (*shape[:-1], shape[-1] * 2), device="cuda", dtype=dtype
                    )
                    storage[..., ::2] = x
                    torch.testing.assert_close(
                        self.module.rms_norm(storage[..., ::2], weight).cpu(),
                        actual.cpu(),
                        rtol=0,
                        atol=0,
                    )

    def test_silu_storage_rounding_and_exceptional_values(self):
        """Preserve BF16 SiLU rounding before multiplication, including NaNs."""
        for dtype in [torch.float32, torch.bfloat16]:
            gate = torch.linspace(-20, 20, 4097, device="cuda", dtype=dtype)
            up = torch.randn_like(gate)
            for changed in [False, True]:
                if changed:
                    gate.add_(0.5)
                    up.mul_(-2)
                host = gate.cpu().double()
                activated = (host / (1 + (-host).exp())).to(dtype)
                expected = (activated.double() * up.cpu().double()).to(dtype)
                torch.testing.assert_close(
                    self.module.silu_multiply(gate, up).cpu(),
                    expected,
                    rtol=0.0078125 if dtype == torch.bfloat16 else 4e-6,
                    atol=0.0078125 if dtype == torch.bfloat16 else 2e-6,
                )
            gate = torch.tensor(
                [float("nan"), float("inf"), -float("inf"), 0.0, -1000.0],
                device="cuda",
                dtype=dtype,
            )
            up = torch.tensor([1, 1, 1, float("inf"), 1], device="cuda", dtype=dtype)
            torch.testing.assert_close(
                self.module.silu_multiply(gate, up),
                functional.silu(gate) * up,
                rtol=0,
                atol=0,
                equal_nan=True,
            )
            torch.testing.assert_close(
                self.module.silu_multiply(gate.flip(0), up.flip(0)),
                (functional.silu(gate) * up).flip(0),
                rtol=0,
                atol=0,
                equal_nan=True,
            )

    def test_rms_analytic_exceptional_and_validation(self):
        """Check exact constant rows, propagated exceptional inputs and bad metadata."""
        for dtype in [torch.float32, torch.bfloat16]:
            x = torch.full((3, 8192), 4.0, device="cuda", dtype=dtype)
            weight = torch.arange(8192, device="cuda").remainder(16).to(dtype)
            torch.testing.assert_close(
                self.module.rms_norm(x, weight, 0), weight.expand_as(x), rtol=0, atol=0
            )
            x[0, 4] = float("nan")
            x[1, 0] = float("inf")
            x[2].zero_()
            expected = weight * (
                x.float() * (x.float().square().mean(-1, keepdim=True)).rsqrt()
            ).to(dtype)
            torch.testing.assert_close(
                self.module.rms_norm(x, weight, 0),
                expected,
                rtol=0,
                atol=0,
                equal_nan=True,
            )
            with self.assertRaises(ValueError):
                self.module.rms_norm(x, weight, -1)
            with self.assertRaises(ValueError):
                self.module.rms_norm(x, weight, float("nan"))
            with self.assertRaises(ValueError):
                self.module.rms_norm(x, weight[:4])
            with self.assertRaises(ValueError):
                self.module.rms_norm(x.requires_grad_(), weight)
        with self.assertRaises(ValueError):
            self.module.rms_norm(
                torch.ones((2, 13), device="cuda", dtype=torch.float64),
                torch.ones(13, device="cuda", dtype=torch.float64),
            )
        with self.assertRaises(ValueError):
            self.module.silu_multiply(
                torch.ones(13, device="cuda"), torch.ones(7, device="cuda")
            )

    def test_gated_mlp_full_outputs_and_changed_weights(self):
        """Use CPU products and activation rounding for each stage independently."""
        for dtype in [torch.float32, torch.bfloat16]:
            x = torch.randn((2, 3, 17), device="cuda", dtype=dtype) / 4
            weights = [
                torch.randn(shape, device="cuda", dtype=dtype) / 4
                for shape in [(31, 17), (31, 17), (13, 31)]
            ]
            for changed in [False, True]:
                if changed:
                    x.add_(0.125)
                    for w in weights:
                        w.mul_(-0.5)
                gate = (x.cpu().double() @ weights[0].cpu().double().T).to(dtype)
                up = (x.cpu().double() @ weights[1].cpu().double().T).to(dtype)
                g = gate.double()
                activated = (g / (1 + (-g).exp())).to(dtype)
                hidden = (activated.double() * up.double()).to(dtype)
                expected = (hidden.double() @ weights[2].cpu().double().T).to(dtype)
                actual = self.module.gated_mlp(x, *weights)
                torch.testing.assert_close(
                    actual.cpu(),
                    expected,
                    rtol=0.02 if dtype == torch.bfloat16 else 1e-5,
                    atol=0.002 if dtype == torch.bfloat16 else 2e-6,
                )
            with self.assertRaises(ValueError):
                self.module.gated_mlp(x, weights[0], weights[1][:5], weights[2])

    def test_swiglu_final_round_clipping_routing_and_changed_operands(self):
        """Check every output against PyTorch and independent CPU double arithmetic."""
        torch.manual_seed(20261004)
        for dtype in (torch.bfloat16, torch.float32):
            gate = torch.randn((3, 17, 2304), device="cuda", dtype=dtype) * 8
            up = torch.randn_like(gate) * 8
            routing = torch.randn((3, 17, 1), device="cuda", dtype=torch.float32)
            for changed in ("original", "gate", "up", "routing"):
                if changed == "gate":
                    gate.mul_(-0.5).add_(0.25)
                elif changed == "up":
                    up.mul_(3)
                elif changed == "routing":
                    routing.add_(0.25)
                for limit in (0, 10):
                    for weights in (None, routing):
                        g, u = gate.float(), up.float()
                        host_g, host_u = gate.cpu().double(), up.cpu().double()
                        if limit:
                            g, u = g.clamp(max=limit), u.clamp(-limit, limit)
                            host_g = host_g.clamp(max=limit)
                            host_u = host_u.clamp(-limit, limit)
                        expected = functional.silu(g) * u
                        independent = host_g / (1 + (-host_g).exp()) * host_u
                        if weights is not None:
                            expected = weights * expected
                            independent = weights.cpu().double() * independent
                        actual = self.module.swiglu(
                            gate, up, routing=weights, limit=limit
                        )
                        torch.testing.assert_close(
                            actual, expected.to(dtype), rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            actual.cpu().double(),
                            independent,
                            rtol=0.008 if dtype == torch.bfloat16 else 4e-6,
                            atol=0.0001 if dtype == torch.bfloat16 else 2e-6,
                        )
            with self.assertRaises(ValueError):
                self.module.swiglu(gate, up, routing=routing.to(dtype=torch.float64))
            with self.assertRaises(ValueError):
                self.module.swiglu(gate, up, routing=routing[..., 0, 0])
            for limit in (-1, float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    self.module.swiglu(gate, up, limit=limit)

    def test_swiglu_exceptional_values_empty_views_and_graphs(self):
        """Preserve NaNs, views, empty shapes and changed graph inputs on every device."""
        for device in range(torch.cuda.device_count()):
            gate = torch.tensor(
                [[float("nan"), float("inf"), -float("inf"), 0, -1000, 1000]],
                device=device,
                dtype=torch.bfloat16,
            )
            up = torch.tensor(
                [[1, 1, 1, float("inf"), 1, 1]], device=device, dtype=gate.dtype
            )
            routing = torch.ones(1, device=device, dtype=torch.float32)
            for limit in (0, 10):
                g, u = gate.float(), up.float()
                if limit:
                    g, u = g.clamp(max=limit), u.clamp(-limit, limit)
                actual = self.module.swiglu(gate, up, routing=routing, limit=limit)
                torch.testing.assert_close(
                    actual,
                    (functional.silu(g) * u).to(gate.dtype),
                    rtol=0,
                    atol=0,
                    equal_nan=True,
                )
                self.assertEqual(
                    self.module.swiglu(gate[:0], up[:0], routing=routing[:0]).shape,
                    (0, 6),
                )
            gate = torch.randn((3, 34), device=device, dtype=torch.bfloat16)[:, ::2]
            up = torch.randn_like(gate)
            routing = torch.randn(3, device=device, dtype=torch.float32)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                self.module.swiglu(gate, up, routing=routing, limit=10)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    output = self.module.swiglu(gate, up, routing=routing, limit=10)
                gate.add_(1)
                up.mul_(-0.5)
                routing.add_(0.25)
                graph.replay()
            stream.synchronize()
            expected = (
                routing[:, None]
                * (
                    functional.silu(gate.float().clamp(max=10))
                    * up.float().clamp(-10, 10)
                )
            ).to(gate.dtype)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            graph.reset()

    def test_graphs_streams_changed_inputs_all_devices(self):
        """Replay changed normalisation weights and MLP operands on each GPU."""
        for device in range(torch.cuda.device_count()):
            with torch.cuda.device(device):
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream), torch.no_grad():
                    x = torch.randn((2, 3, 17), device=device, dtype=torch.bfloat16)
                    weight = torch.randn(17, device=device, dtype=x.dtype)
                    projections = [
                        torch.randn(shape, device=device, dtype=x.dtype) / 4
                        for shape in [(31, 17), (31, 17), (13, 31)]
                    ]
                    self.module.rms_norm(x, weight)
                    self.module.gated_mlp(x, *projections)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        norm = self.module.rms_norm(x, weight)
                        result = self.module.gated_mlp(norm, *projections)
                    x.add_(0.5)
                    weight.mul_(-0.5)
                    projections[0].add_(0.125)
                    graph.replay()
                    reference_norm = weight * (
                        x.float()
                        * (x.float().square().mean(-1, keepdim=True) + 1e-6).rsqrt()
                    ).to(x.dtype)
                    reference = functional.linear(
                        functional.silu(
                            functional.linear(reference_norm, projections[0])
                        )
                        * functional.linear(reference_norm, projections[1]),
                        projections[2],
                    )
                stream.synchronize()
                torch.testing.assert_close(
                    norm, reference_norm, rtol=0.015625, atol=0.0078125
                )
                torch.testing.assert_close(result, reference, rtol=0.02, atol=0.01)
