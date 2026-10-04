"""Benchmark local Llama 70B with paired PyTorch and CAMBLAS CUDA inference."""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as functional
from transformers import LlamaConfig, LlamaForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    """Return the SHA256 identity of a source or native binary."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    """Measure paired complete inference calls and verify logits and generated tokens."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-directory", type=Path)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--backends", nargs="+", default=["pytorch", "camblas"])
    parser.add_argument("--algorithm", default="auto")
    parser.add_argument(
        "--camblas-library",
        type=Path,
        help="Exact candidate or saved control core with matching tensor binding",
    )
    parser.add_argument(
        "--linear-entry",
        choices=["matmul", "native"],
        default="matmul",
        help="Choose the original adapter or native F.linear-compatible entry",
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--generated-tokens", type=int, default=16)
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--toy-cpu", action="store_true")
    parser.add_argument(
        "--copy-weights-every-request",
        action="store_true",
        help="Transfer mode also uploads every parameter from CPU once per request",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Export one untimed full-request trace per backend",
    )
    parser.add_argument(
        "--bf16-reduction",
        choices=["full", "reduced"],
        default="reduced",
        help="PyTorch BF16 reduction policy; CAMBLAS uses full FP32 accumulation",
    )
    parser.add_argument("--fuse-rms", action="store_true")
    parser.add_argument("--fuse-mlp", action="store_true")
    parser.add_argument("--fuse-qkv", action="store_true")
    parser.add_argument("--fuse-residual", action="store_true")
    parser.add_argument(
        "--weights-manifest",
        type=Path,
        help="Previously SHA256-verified weight identity; sizes and mtimes must still match",
    )
    args = parser.parse_args()
    if args.batch < 1 or args.prompt_length < 1 or args.generated_tokens < 2:
        parser.error("Use a positive batch/prompt and at least two generated tokens")
    if args.repetitions < 1 or args.warmups < 1:
        parser.error("Use at least one warm-up and one timed call")
    assert set(args.backends) <= {"pytorch", "camblas"}
    torch.set_num_threads(1 if args.toy_cpu else 64)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
        args.bf16_reduction == "reduced"
    )
    torch.manual_seed(20261003)
    dtype = getattr(torch, args.dtype)
    devices = [] if args.toy_cpu else list(range(torch.cuda.device_count()))
    assert args.toy_cpu or devices
    cb = None
    native_linear = None
    if not args.toy_cpu and "camblas" in args.backends:
        if args.camblas_library:
            assert args.camblas_library.is_file()
            os.environ["CAMBLAS_CUDA_LIBRARY"] = str(args.camblas_library.resolve())
        import camblas_gpu as cb

        if args.linear_entry == "native":
            native_linear = cb._native.tensor_module().linear_public

    def sync():
        for device in devices:
            torch.cuda.synchronize(device)

    started = datetime.now(timezone.utc).isoformat()
    loading = time.perf_counter()
    if args.toy_cpu:
        config = LlamaConfig(
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=512,
            max_position_embeddings=8192,
        )
        config._attn_implementation = "sdpa"
        model = LlamaForCausalLM(config).to(dtype=dtype).eval()
    else:
        assert args.model_directory and args.model_directory.is_dir()
        assert (args.model_directory / "model.safetensors.index.json").is_file()
        model = LlamaForCausalLM.from_pretrained(
            str(args.model_directory),
            local_files_only=True,
            dtype=dtype,
            device_map="balanced",
            max_memory={device: "88GiB" for device in devices},
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        ).eval()
        assert model.config.num_hidden_layers == 80
        assert model.config.hidden_size == 8192
        assert sum(p.numel() for p in model.parameters()) == 70553706496
        assert all(p.is_cuda and p.dtype == dtype for p in model.parameters())
        assert not any(str(v) in {"cpu", "disk"} for v in model.hf_device_map.values())
    if (
        args.prompt_length + args.generated_tokens - 1
        > model.config.max_position_embeddings
    ):
        parser.error("Prompt and decode steps exceed this checkpoint's context length")
    sync()
    loading_seconds = time.perf_counter() - loading
    device = model.get_input_embeddings().weight.device
    generator = torch.Generator().manual_seed(20261003)
    host_inputs = torch.randint(
        3,
        model.config.vocab_size,
        (args.batch, args.prompt_length),
        generator=generator,
    )
    resident_inputs = host_inputs.to(device)
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    host_weights = []
    snapshot_started = time.perf_counter()
    if args.copy_weights_every_request:
        if not args.toy_cpu:
            available = next(
                int(line.split()[1]) * 1024
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")
            )
            assert available > parameter_bytes + 16 * 2**30, (
                "Insufficient available host memory for a full parameter copy"
            )
        if not args.toy_cpu:
            relative = Path("/proc/self/cgroup").read_text().strip().split(":", 2)[2]
            group = Path("/sys/fs/cgroup") / relative.lstrip("/")
            while group != Path("/sys/fs/cgroup"):
                limit = (group / "memory.max").read_text().strip()
                if limit != "max":
                    current = int((group / "memory.current").read_text())
                    # Cached model files are reclaimable; anonymous snapshots are not.
                    stats = dict(
                        line.split()
                        for line in (group / "memory.stat").read_text().splitlines()
                    )
                    available = int(limit) - current + int(stats.get("file", 0))
                    assert available > parameter_bytes + 16 * 2**30, (
                        "Allocation memory limit is too small for a full weight snapshot"
                    )
                group = group.parent
        host_weights = [
            (parameter, parameter.detach().cpu().clone())
            for parameter in model.parameters()
        ]
    parameter_snapshot_seconds = time.perf_counter() - snapshot_started
    weight_identity = None
    if args.weights_manifest:
        weight_identity = json.loads(args.weights_manifest.read_text())
        assert weight_identity["state"] == "passed"
        assert (
            Path(weight_identity["directory"]).resolve()
            == args.model_directory.resolve()
        )
        for name, entry in weight_identity["weights"].items():
            path = args.model_directory / name
            assert (
                path.stat().st_size == entry["bytes"]
                and path.stat().st_mtime_ns == entry["mtime_ns"]
            ), name
    shapes = Counter()

    def attention_replacement(module, native_module):
        # Follow Transformers 4.57.1 LlamaAttention's SDPA/RoPE/cache pipeline;
        # group only the three bias-free GEMMs into one native Python dispatch.
        from transformers.models.llama.modeling_llama import (
            ALL_ATTENTION_FUNCTIONS,
            apply_rotary_pos_emb,
            eager_attention_forward,
        )

        assert all(
            projection.bias is None
            for projection in [module.q_proj, module.k_proj, module.v_proj]
        )
        weights = (module.q_proj.weight, module.k_proj.weight, module.v_proj.weight)
        operation = native_module.qkv_linear

        def forward(
            self,
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values=None,
            cache_position=None,
            **kwargs,
        ):
            input_shape = hidden_states.shape[:-1]
            shape = (*input_shape, -1, self.head_dim)
            q, k, v = operation(hidden_states, *weights)
            q = q.view(shape).transpose(1, 2)
            k = k.view(shape).transpose(1, 2)
            v = v.view(shape).transpose(1, 2)
            cosine, sine = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cosine, sine)
            if past_key_values is not None:
                k, v = past_key_values.update(
                    k,
                    v,
                    self.layer_idx,
                    {"sin": sine, "cos": cosine, "cache_position": cache_position},
                )
            attention = (
                eager_attention_forward
                if self.config._attn_implementation == "eager"
                else ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
            )
            result, attention_weights = attention(
                self,
                q,
                k,
                v,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )
            return self.o_proj(
                result.reshape(*input_shape, -1).contiguous()
            ), attention_weights

        return forward

    def decoder_replacement(module, native):
        norm = native.rms_norm
        add_norm = native.add_rms_norm
        mlp = native.gated_mlp
        norm_weight, norm_epsilon = (
            module.input_layernorm.weight,
            module.input_layernorm.variance_epsilon,
        )
        post_weight, post_epsilon = (
            module.post_attention_layernorm.weight,
            module.post_attention_layernorm.variance_epsilon,
        )
        mlp_weights = (
            module.mlp.gate_proj.weight,
            module.mlp.up_proj.weight,
            module.mlp.down_proj.weight,
        )

        def forward(
            self,
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            use_cache=False,
            cache_position=None,
            position_embeddings=None,
            **kwargs,
        ):
            residual = hidden_states
            hidden_states = norm(hidden_states, norm_weight, norm_epsilon)
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            residual, hidden_states = add_norm(
                hidden_states, residual, post_weight, post_epsilon
            )
            return residual + mlp(hidden_states, *mlp_weights)

        return forward

    @contextmanager
    def backend(label, collect_shapes=False):
        original = functional.linear
        replacements = []
        if label == "camblas":
            if not args.toy_cpu:
                native_module = cb._native.tensor_module()
                for module in model.modules():
                    name = type(module).__name__
                    replacement = None
                    if args.fuse_rms and name == "LlamaRMSNorm":

                        def replacement(
                            _module,
                            x,
                            weight=module.weight,
                            epsilon=module.variance_epsilon,
                            operation=native_module.rms_norm,
                        ):
                            return operation(x, weight, epsilon)
                    elif args.fuse_mlp and name == "LlamaMLP":
                        assert all(
                            projection.bias is None
                            for projection in [
                                module.gate_proj,
                                module.up_proj,
                                module.down_proj,
                            ]
                        )

                        def replacement(
                            _module,
                            x,
                            gate=module.gate_proj.weight,
                            up=module.up_proj.weight,
                            down=module.down_proj.weight,
                            operation=native_module.gated_mlp,
                        ):
                            return operation(x, gate, up, down)

                    if args.fuse_residual and name == "LlamaDecoderLayer":
                        replacement = decoder_replacement(module, native_module)
                    if args.fuse_qkv and name == "LlamaAttention":
                        replacement = attention_replacement(module, native_module)
                    if replacement is not None:
                        attr = (
                            "_old_forward"
                            if hasattr(module, "_hf_hook")
                            and hasattr(module, "_old_forward")
                            else "forward"
                        )
                        replacements.append((module, attr, getattr(module, attr)))
                        setattr(module, attr, MethodType(replacement, module))

            def linear(x, weight, bias=None):
                if collect_shapes:
                    shapes[(str(x.dtype), tuple(x.shape), tuple(weight.shape))] += 1
                if args.linear_entry == "native":
                    return (
                        original(x, weight, bias)
                        if args.toy_cpu
                        else native_linear(x, weight, bias)
                    )
                matrix = x.reshape(-1, x.shape[-1])
                value = (
                    matrix @ weight.T if args.toy_cpu else cb.matmul(matrix, weight.T)
                )
                if bias is not None:
                    value = value + bias
                return value.reshape(*x.shape[:-1], weight.shape[0])

            functional.linear = (
                native_linear
                if native_linear is not None and not collect_shapes
                else linear
            )
        try:
            if cb is not None and label == "camblas":
                with cb.algorithm(args.algorithm):
                    yield
            else:
                yield
        finally:
            functional.linear = original
            for module, attr, old in reversed(replacements):
                setattr(module, attr, old)

    def prefill(inputs):
        return model(input_ids=inputs, use_cache=True, logits_to_keep=1)

    def decode(output):
        past = output.past_key_values
        token = output.logits[:, -1].argmax(-1)
        generated = [token]
        for _ in range(args.generated_tokens - 1):
            output = model(
                input_ids=token[:, None],
                past_key_values=past,
                use_cache=True,
                logits_to_keep=1,
            )
            past = output.past_key_values
            token = output.logits[:, -1].argmax(-1)
            generated.append(token)
        return {"logits": output.logits, "tokens": torch.stack(generated, 1)}

    def call(phase, mode):
        prepared = prefill(resident_inputs) if phase == "decode" else None
        sync()
        before = time.perf_counter()
        if args.copy_weights_every_request and mode == "transfer" and phase != "decode":
            for parameter, host_value in host_weights:
                parameter.copy_(host_value)
        inputs = (
            host_inputs.to(device)
            if mode == "transfer" and phase != "decode"
            else resident_inputs
        )
        if phase == "prefill":
            output = {"logits": prefill(inputs).logits}
        elif phase == "decode":
            output = decode(prepared)
        else:
            output = decode(prefill(inputs))
        if mode == "transfer":
            output = {key: value.cpu() for key, value in output.items()}
        sync()
        elapsed = time.perf_counter() - before
        # Numerical snapshots and cache destruction are outside the timed interval.
        snapshot = {key: value.detach().cpu().clone() for key, value in output.items()}
        del output, prepared
        return elapsed, snapshot

    sources = [Path(__file__).resolve(), *sorted((ROOT / "camblas_gpu").glob("*.py"))]
    if not args.toy_cpu:
        sources += sorted((ROOT / "src/cuda").glob("*"))
        sources += [ROOT / "include/camblas_cuda.h", ROOT / "scripts/build_cuda.py"]
    identities = {str(p): digest(p) for p in sources if p.is_file()}
    record = {
        "started": started,
        "state": "running",
        "command": sys.argv,
        "toy_cpu": args.toy_cpu,
        "allocation": os.environ.get("SLURM_JOB_ID"),
        "dtype": args.dtype,
        "round": args.round,
        "affinity": sorted(os.sched_getaffinity(0)),
        "camblas_linear_entry": args.linear_entry,
        "camblas_fusion": {
            "rms_norm": args.fuse_rms,
            "gated_mlp": args.fuse_mlp,
            "qkv": args.fuse_qkv,
            "residual_rms_norm": args.fuse_residual,
        },
        "weight_identity": weight_identity,
        "parameters": sum(p.numel() for p in model.parameters()),
        "parameter_bytes": parameter_bytes,
        "parameter_upload_bytes_per_transfer_request": parameter_bytes
        if args.copy_weights_every_request
        else 0,
        "host_parameter_snapshot_seconds": parameter_snapshot_seconds,
        "model_directory": str(args.model_directory),
        "model_load_seconds": loading_seconds,
        "model_configuration": model.config.to_dict(),
        "device_map": getattr(model, "hf_device_map", {"model": "cpu"}),
        "batch": args.batch,
        "prompt_length": args.prompt_length,
        "generated_tokens": args.generated_tokens,
        "decode_steps": args.generated_tokens - 1,
        "input_sha256": hashlib.sha256(host_inputs.numpy().tobytes()).hexdigest(),
        "source_sha256": identities,
        "torch_version": torch.__version__,
        "matmul_precision": {
            "float32": torch.get_float32_matmul_precision(),
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        },
        "transfer_contract": "Weights and KV cache stay resident; copy CPU prompt to GPU and final logits/tokens to CPU. Decode-only starts from a prepared resident KV cache.",
        "camblas_contract": "CAMBLAS replaces all functional linear GEMMs; both paths use identical PyTorch SDPA, RMSNorm, RoPE and KV cache operations.",
        "measurements": {},
        "numerical_checks": {},
    }
    if args.copy_weights_every_request:
        record["transfer_contract"] = (
            "Transfer prefill/request uploads all parameter weights and CPU token inputs on every call, then returns final logits/tokens to CPU. Weights and KV cache remain resident throughout each request; decode-only starts from a prepared resident KV cache."
        )
    if not args.toy_cpu:
        record["gpu"] = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv",
            ],
            text=True,
        )
        if args.fuse_rms or args.fuse_mlp or args.fuse_qkv or args.fuse_residual:
            record["camblas_contract"] = (
                "CAMBLAS replaces functional linear GEMMs and selected RMSNorm, gated-MLP and grouped QKV operations. Both paths retain identical PyTorch SDPA, RoPE and KV cache operations; native fusion preserves intermediate storage rounding."
            )
        core = Path(
            os.environ.get(
                "CAMBLAS_CUDA_LIBRARY", ROOT / "build/cuda/libcamblas_cuda.so"
            )
        ).resolve()
        build = json.loads((core.parent / "build.json").read_text())
        source_root = core.parent / "source"
        if not source_root.is_dir():
            source_root = next(
                Path(item).parents[2]
                for item in build["command"]
                if item.endswith("/src/cuda/backend.cu")
            )
        for name, expected in build["source_sha256"].items():
            path = source_root / name
            assert digest(path) == expected, str(path)
            identities[str(path)] = expected
        record["build"] = build
        record["camblas_source_root"] = str(source_root)
        assert digest(core) == build["library_sha256"]
        bindings = sorted(core.parent.glob("_camblas_cuda_torch*.so"))
        if cb is not None:
            assert len(bindings) == 1
            assert digest(bindings[0]) == build["tensor_binding_sha256"]
            loaded = {
                str(Path(line.split()[-1]).resolve())
                for line in Path("/proc/self/maps").read_text().splitlines()
                if "libcamblas_cuda.so" in line and line.split()[-1].startswith("/")
            }
            assert str(core) in loaded, (str(core), loaded)
            record["loaded_camblas_core"] = str(core)
        for path in [core, *bindings]:
            record.setdefault("library_sha256", {})[str(path)] = digest(path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assert not args.output.exists()

    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(args.output)

    save()
    oracles = {}
    labels = (
        args.backends[args.round % len(args.backends) :]
        + args.backends[: args.round % len(args.backends)]
    )
    try:
        with torch.inference_mode():
            for label in labels:
                record["measurements"][label] = {}
                with backend(label):
                    for phase in ["prefill", "decode", "request"]:
                        record["measurements"][label][phase] = {}
                        for mode in (
                            ["resident", "transfer"]
                            if args.round % 2 == 0
                            else ["transfer", "resident"]
                        ):
                            for _ in range(args.warmups):
                                call(phase, mode)
                            samples = []
                            for _ in range(args.repetitions):
                                elapsed, snapshot = call(phase, mode)
                                samples.append(elapsed)
                            for key, value in snapshot.items():
                                oracle_key = (phase, key)
                                if oracle_key not in oracles:
                                    oracles[oracle_key] = value
                                elif key == "tokens":
                                    assert torch.equal(value, oracles[oracle_key]), (
                                        label,
                                        phase,
                                        "generated token mismatch",
                                    )
                                else:
                                    reference = oracles[oracle_key]
                                    difference = (
                                        value.float() - reference.float()
                                    ).abs()
                                    record["numerical_checks"][
                                        f"{label}_{phase}_{mode}"
                                    ] = {
                                        "max_abs_error": difference.max().item(),
                                        "rms_error": difference.square()
                                        .mean()
                                        .sqrt()
                                        .item(),
                                    }
                                    torch.testing.assert_close(
                                        value,
                                        reference,
                                        rtol=2e-4 if dtype == torch.float32 else 2e-2,
                                        atol=2e-4 if dtype == torch.float32 else 2e-2,
                                    )
                            record["measurements"][label][phase][mode] = {
                                "seconds": samples,
                                "median_ms": 1000 * statistics.median(samples),
                            }
                            save()
                print(label, record["measurements"][label], flush=True)
            # Check recomputation after changing every input token.
            if "camblas" in args.backends:
                with backend("camblas", collect_shapes=True):
                    decode(prefill(resident_inputs))
            changed = ((host_inputs + 7) % (model.config.vocab_size - 3) + 3).to(device)
            outputs = {}
            for label in args.backends:
                with backend(label):
                    outputs[label] = prefill(changed).logits.cpu()
            if len(outputs) == 2:
                torch.testing.assert_close(
                    outputs["camblas"],
                    outputs["pytorch"],
                    rtol=2e-4 if dtype == torch.float32 else 2e-2,
                    atol=2e-4 if dtype == torch.float32 else 2e-2,
                )
            assert shapes or "camblas" not in args.backends
            record["changed_token_inputs_checked"] = len(outputs) == 2
            if args.profile:
                activities = [torch.profiler.ProfilerActivity.CPU]
                if not args.toy_cpu:
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                for label in args.backends:
                    with backend(label):
                        with torch.profiler.profile(
                            activities=activities, record_shapes=True
                        ) as profile:
                            with torch.profiler.record_function(
                                label + "_full_request"
                            ):
                                decode(prefill(resident_inputs))
                            sync()
                    path = args.output.with_name(
                        args.output.stem + "_" + label + "_trace.json"
                    )
                    profile.export_chrome_trace(str(path))
                    record.setdefault("profiles", {})[label] = str(path)
        for path, expected in identities.items():
            assert digest(Path(path)) == expected, path
        if weight_identity is not None:
            for name, entry in weight_identity["weights"].items():
                path = args.model_directory / name
                assert path.stat().st_size == entry["bytes"], name
                assert path.stat().st_mtime_ns == entry["mtime_ns"], name
        record["linear_shapes"] = [
            {"dtype": key[0], "input": key[1], "weight": key[2], "calls": count}
            for key, count in shapes.items()
        ]
        if cb is not None:
            record["camblas_counters"] = {
                str(d): cb.stats(device=d, all_threads=True) for d in devices
            }
            assert (
                sum(
                    c["classical"] + c["lt"] + c["strassen"]
                    for c in record["camblas_counters"].values()
                )
                > 0
            )
        record["memory"] = {
            str(d): {
                "allocated": torch.cuda.memory_allocated(d),
                "reserved": torch.cuda.memory_reserved(d),
                "peak_allocated": torch.cuda.max_memory_allocated(d),
                "free_total": torch.cuda.mem_get_info(d),
            }
            for d in devices
        }
        record["state"] = "passed"
    except BaseException as error:
        record["state"] = "failed"
        record["error"] = repr(error)
        raise
    finally:
        record["finished"] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    main()
