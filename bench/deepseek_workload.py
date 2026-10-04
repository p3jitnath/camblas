"""Measure the complete pinned DeepSeek checkpoint on four CUDA ranks."""

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    """Return the SHA256 identity of a source or library file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def host_affinity(device):
    """Choose sixteen allowed CPU cores local to a CUDA device, without rank overlap.

    Parameters
    ----------
    device : int
        Logical CUDA device ordinal under the allocation's visible-device mapping.

    Returns
    -------
    tuple
        CPU indices and the closest host NUMA node, or -1 when unavailable.
    """
    runtime = ctypes.CDLL("libcudart.so.12")
    runtime.cudaDeviceGetAttribute.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
    ]
    nodes = []
    for index in range(torch.cuda.device_count()):
        node = ctypes.c_int(-1)
        # cudaDevAttrHostNumaId from CUDA 12.9's driver_types.h.
        status = runtime.cudaDeviceGetAttribute(ctypes.byref(node), 134, index)
        nodes.append(node.value if status == 0 else -1)
    allowed = os.sched_getaffinity(0)
    node = nodes[device]
    local = set()
    if node >= 0:
        for item in (
            Path(f"/sys/devices/system/node/node{node}/cpulist")
            .read_text()
            .strip()
            .split(",")
        ):
            bounds = [int(value) for value in item.split("-")]
            local.update(range(bounds[0], bounds[-1] + 1))
    cpus = sorted(allowed & local) if local else sorted(allowed)
    position = sum(other == node for other in nodes[:device]) if local else device
    selected = cpus[position * 16 : (position + 1) * 16]
    if len(selected) != 16:
        raise ValueError("Allocate sixteen distinct host cores per CUDA rank")
    return selected, node


def prepare_reference(source, destination, manifest):
    """Verify pinned sources and add the shared activation-quantiser barrier fix.

    Parameters
    ----------
    source, destination : pathlib.Path
        Downloaded inference directory and ignored benchmark working directory.
    manifest : dict
        Pinned source identities; neither downloaded sources nor weights change.

    Returns
    -------
    dict
        Original and corrected quantiser source identities.
    """
    destination.mkdir(parents=True, exist_ok=True)
    for name, expected in manifest["reference_sources"].items():
        path = source / name
        if digest(path) != expected:
            raise ValueError(f"Pinned reference source mismatch: {name}")
        shutil.copyfile(path, destination / name)
    path = destination / "kernel.py"
    original = path.read_text()
    target = "                T.copy(y_local, y_shared)"
    if original.count(target) != 2:
        raise ValueError("Activation quantiser barrier sites changed")
    path.write_text(
        original.replace(
            target,
            "                T.sync_threads()  # Finish reads before shared-buffer reuse.\n"
            + target,
        )
    )
    return dict(original=digest(source / "kernel.py"), corrected=digest(path))


def install_offload(reference):
    """Keep complete Engram tables on CPU, timing every lookup and row transfer."""
    original = reference.ParallelEngramEmbedding.__init__

    def initialise(self, num_embeddings, dim):
        """Allocate the full table on CPU under the same policy for both backends."""
        with torch.device("cpu"):
            original(self, num_embeddings, dim)

    def forward(self, indices):
        """Gather encoded rows on CPU and dequantise them on the requesting GPU."""
        target = indices.device
        host = indices.cpu()
        mask = (host < self.vocab_start_idx) | (host >= self.vocab_end_idx)
        local = (host - self.vocab_start_idx).masked_fill(mask, 0)
        values = self.weight.view(torch.uint8)[local].view(self.weight.dtype).to(target)
        scales = self.scale.view(torch.uint8)[local].view(self.scale.dtype).to(target)
        values = (
            (
                values.float().unflatten(-1, (-1, self.block_size))
                * scales.float().unsqueeze(-1)
            )
            .flatten(-2)
            .to(torch.bfloat16)
        )
        values = values.masked_fill(mask.to(target).unsqueeze(-1), 0)
        if reference.world_size > 1:
            dist.all_reduce(values)
        return values

    reference.ParallelEngramEmbedding.__init__ = initialise
    reference.ParallelEngramEmbedding.forward = forward


def load_rank(model, checkpoint):
    """Load every tensor, retaining CPU file mappings and streaming GPU weights.

    Parameters
    ----------
    model : torch.nn.Module
        Complete reference model with identical CPU table placement on each path.
    checkpoint : pathlib.Path
        Converted safetensors file for this tensor-parallel rank.

    Returns
    -------
    dict
        Loaded tensor count and CPU/GPU storage bytes.
    """
    parameters = dict(model.named_parameters())
    loaded = set()
    libc = ctypes.CDLL(None, use_errno=True)
    libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    page = os.sysconf("SC_PAGE_SIZE")
    with safe_open(str(checkpoint), framework="pt", device="cpu") as weights:
        for name in weights.keys():
            if name not in parameters:
                raise ValueError(f"Unexpected checkpoint tensor: {name}")
            target, source = parameters[name], weights.get_tensor(name)
            # The pinned model explicitly promotes compressed projections,
            # vocabulary heads and confidence projections to FP32 at runtime.
            promote = source.dtype == torch.bfloat16 and target.dtype == torch.float32
            if source.shape != target.shape or (
                source.dtype != target.dtype and not promote
            ):
                raise ValueError(f"Checkpoint tensor metadata mismatch: {name}")
            if target.device.type == "cpu":
                module = model
                fields = name.split(".")
                for field in fields[:-1]:
                    module = getattr(module, field)
                parameter = torch.nn.Parameter(source, requires_grad=False)
                setattr(module, fields[-1], parameter)
                parameters[name] = parameter
            else:
                with torch.no_grad():
                    target.copy_(source)
                address = source.data_ptr()
                begin = (address + page - 1) // page * page
                end = (address + source.numel() * source.element_size()) // page * page
                if end > begin and libc.madvise(begin, end - begin, 4):
                    raise OSError(ctypes.get_errno(), "Cannot release loaded GPU pages")
            loaded.add(name)
    if loaded != set(parameters):
        raise ValueError("Checkpoint does not contain every model parameter")
    for value in parameters.values():
        if value.device.type == "cpu":
            value.view(torch.uint8).reshape(-1)[::page].sum().item()
    return dict(
        tensors=len(loaded),
        cpu_bytes=sum(
            p.numel() * p.element_size()
            for p in parameters.values()
            if p.device.type == "cpu"
        ),
        gpu_bytes=sum(
            p.numel() * p.element_size() for p in parameters.values() if p.is_cuda
        ),
    )


def install_camblas(reference):
    """Dispatch generic native FP4 products and final-round RMS with exact fallbacks."""
    import camblas_gpu as cb

    module = cb._native.tensor_module()
    fp4_reference, rms_reference = reference.fp4_gemm, reference.RMSNorm.forward
    expert_reference = reference.Expert.forward
    moe_reference = reference.MoE.forward
    quantized = hasattr(module, "quantized_matmul")
    counts = dict(
        fp4_native=0,
        fp4_reference=0,
        rms_native=0,
        rms_reference=0,
        swiglu_native=0,
        expert_reference=0,
        quantization_reused=0,
        grouped_experts=0,
    )

    def fp4(x, sx, weight, sw, scale_dtype=torch.float32, act_block_size=128):
        """Use the native small-row product only for its validated storage layouts."""
        rows = x.numel() // x.size(-1)
        eligible = (
            quantized
            and rows <= (512 if hasattr(module, "quantized_tile_rows") else 32)
            and weight.size(0) % 8 == 0
            and scale_dtype == torch.float8_e8m0fnu
            and all(t.is_contiguous() for t in (x, sx, weight, sw))
        )
        counts["fp4_native" if eligible else "fp4_reference"] += 1
        return (
            cb.quantized_matmul(x, sx, weight, sw, activation_block=act_block_size)
            if eligible
            else fp4_reference(x, sx, weight, sw, scale_dtype, act_block_size)
        )

    def rms(self, x):
        """Preserve the reference's final-only BF16 rounding and versioned reduction."""
        width = x.size(-1)
        eligible = (
            quantized
            and x.dtype == torch.bfloat16
            and x.numel() == width
            and 2048 <= width <= 65536
            and width % 4 == 0
            and x.is_contiguous()
        )
        counts["rms_native" if eligible else "rms_reference"] += 1
        return (
            cb.rms_norm(x, self.weight, self.eps, round_before_weight=False)
            if eligible
            else rms_reference(self, x)
        )

    def expert(self, x, weights=None):
        """Reuse gate/up activation quantisation and fuse final-round SwiGLU."""
        if not hasattr(module, "swiglu") or x.dtype != torch.bfloat16:
            counts["expert_reference"] += 1
            return expert_reference(self, x, weights)
        left, right = self.w1.weight, self.w3.weight
        if left.dtype == right.dtype and left.dtype in (
            torch.float4_e2m1fn_x2,
            torch.float8_e4m3fn,
        ):
            q, scales = reference.act_quant(
                x, reference.fp8_block_size, reference.scale_fmt, reference.scale_dtype
            )
            if left.dtype == torch.float4_e2m1fn_x2:
                gate = fp4(
                    q,
                    scales,
                    left,
                    left.scale,
                    reference.scale_dtype,
                    reference.fp8_block_size,
                )
                up = fp4(
                    q,
                    scales,
                    right,
                    right.scale,
                    reference.scale_dtype,
                    reference.fp8_block_size,
                )
            else:
                gate = reference.fp8_gemm(
                    q,
                    scales,
                    left,
                    left.scale,
                    reference.scale_dtype,
                    block_size=reference.fp8_block_size,
                )
                up = reference.fp8_gemm(
                    q,
                    scales,
                    right,
                    right.scale,
                    reference.scale_dtype,
                    block_size=reference.fp8_block_size,
                )
            counts["quantization_reused"] += 1
        else:
            gate, up = self.w1(x), self.w3(x)
        counts["swiglu_native"] += 1
        return self.w2(
            cb.swiglu(gate, up, routing=weights, limit=max(0, self.swiglu_limit))
        )

    def moe(self, x, image_mask=None):
        """Group sufficiently large token batches, retaining the existing decode route."""
        if not hasattr(module, "QuantizedGroups") or x.numel() // self.dim < 8:
            return moe_reference(self, x, image_mask)
        shape = x.shape
        x = x.view(-1, self.dim)
        weights, indices = self.gate(
            x, None if image_mask is None else image_mask.flatten()
        )
        sizes = torch.bincount(
            indices.flatten(), minlength=self.n_routed_experts
        ).tolist()
        chosen = [
            index
            for index in range(self.experts_start_idx, self.experts_end_idx)
            if sizes[index]
        ]
        rows = max((sizes[index] for index in chosen), default=0)
        if len(chosen) < 2 or rows > 65535:
            y = torch.zeros_like(x, dtype=torch.float32)
            for index in chosen:
                row, choice = torch.where(indices == index)
                y[row] += self.experts[index](x[row], weights[row, choice, None])
        else:
            if not hasattr(self, "_camblas_groups"):
                local = self.experts[self.experts_start_idx : self.experts_end_idx]
                self._camblas_groups = tuple(
                    cb.QuantizedGroups(
                        [getattr(expert, projection).weight for expert in local],
                        [getattr(expert, projection).weight.scale for expert in local],
                    )
                    for projection in ("w1", "w3", "w2")
                )
            active = torch.tensor(
                [index - self.experts_start_idx for index in chosen],
                dtype=torch.int32,
                device="cpu",
            ).to(x.device)
            slots, reverse, group_counts = cb.route_groups(
                indices, active, first_expert=self.experts_start_idx, rows=rows
            )
            valid = slots.clamp_min(0).long()
            row_ids = (valid // indices.size(1)).flatten()
            q, scales = reference.act_quant(
                x, reference.fp8_block_size, reference.scale_fmt, reference.scale_dtype
            )
            q = (
                q.view(torch.uint8)[row_ids]
                .view(q.dtype)
                .view(len(chosen), rows, self.dim)
            )
            scales = (
                scales.view(torch.uint8)[row_ids]
                .view(scales.dtype)
                .view(len(chosen), rows, self.dim // reference.fp8_block_size)
            )
            gate, up = (
                group.matmul(
                    q,
                    scales,
                    active,
                    group_counts,
                    activation_block=reference.fp8_block_size,
                )
                for group in self._camblas_groups[:2]
            )
            routing = weights.flatten()[valid].masked_fill(slots < 0, 0).unsqueeze(-1)
            activation = cb.swiglu(
                gate,
                up,
                routing=routing,
                limit=max(0, self.experts[chosen[0]].swiglu_limit),
            )
            q, scales = reference.act_quant(
                activation,
                reference.fp8_block_size,
                reference.scale_fmt,
                reference.scale_dtype,
            )
            output = self._camblas_groups[2].matmul(
                q,
                scales,
                active,
                group_counts,
                activation_block=reference.fp8_block_size,
            )
            y = cb.reduce_groups(output, indices, reverse)
            counts["grouped_experts"] += len(chosen)
        if reference.world_size > 1:
            dist.all_reduce(y)
        y += self.shared_experts(x)
        return y.to(x.dtype).view(shape)

    reference.fp4_gemm, reference.RMSNorm.forward = fp4, rms
    reference.Expert.forward = expert
    reference.MoE.forward = moe
    return counts


def main():
    """Run one fresh backend process group, checking full logits and repeated inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-directory", type=Path, required=True)
    parser.add_argument("--weights-manifest", type=Path, required=True)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--backend", choices=["pytorch", "camblas"], required=True)
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--generated-tokens", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.prompt_length < 1
        or args.generated_tokens < 2
        or min(args.warmups, args.repetitions) < 1
    ):
        parser.error(
            "Use positive prompt, warm-ups/repetitions and at least two generated tokens"
        )
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 4:
        parser.error(
            "The pinned converted checkpoint requires four tensor-parallel ranks"
        )
    worker_sha = digest(Path(__file__))
    cpus, host_numa = host_affinity(local_rank)
    os.sched_setaffinity(0, set(cpus))
    torch.set_num_threads(16)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    dist.barrier(device_ids=[local_rank])
    manifest = json.loads(args.weights_manifest.read_text())
    if manifest.get("state") != "passed":
        raise ValueError("Provide a complete, passed weight verification record")
    for name, expected_sha in manifest["tokenizer_sources"].items():
        if digest(args.checkpoint / name) != expected_sha:
            raise ValueError(f"Pinned tokenizer mismatch: {name}")
    shard = args.checkpoint / f"model{rank}-mp4.safetensors"
    expected = manifest["converted_weights"][shard.name]
    if len(expected.get("sha256", "")) != 64:
        raise ValueError("Converted checkpoint SHA256 verification is missing")
    stat = shard.stat()
    if (stat.st_size, stat.st_mtime_ns) != (expected["bytes"], expected["mtime_ns"]):
        raise ValueError(
            "Converted weight verification is stale; rehash the changed checkpoint"
        )
    working = args.output.parent / "reference"
    if rank == 0:
        prepare_reference(args.reference_directory, working, manifest)
    dist.barrier()
    sys.path[:0] = [str(working.resolve()), str(ROOT)]
    import model as reference

    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(20261004)
    install_offload(reference)
    config = json.loads((working / "config.json").read_text())
    config.update(
        max_batch_size=1,
        max_seq_len=args.prompt_length + args.generated_tokens + 128,
        temperature=0,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    start = time.perf_counter()
    with torch.device("cuda"):
        model = reference.Transformer(reference.ModelArgs(**config), tokenizer)
    storage = load_rank(model, shard)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    loading_seconds = time.perf_counter() - start
    counts = {}
    identities = {}
    if args.backend == "camblas":
        if not args.library:
            parser.error(
                "CAMBLAS requires an explicit library and matching Torch binding"
            )
        os.environ["CAMBLAS_CUDA_LIBRARY"] = str(args.library.resolve())
        counts = install_camblas(reference)
        identities = {
            p.name: digest(p)
            for p in [
                args.library,
                *args.library.parent.glob("_camblas_cuda_torch*.so"),
            ]
        }
    prompt = torch.randint(
        3,
        128000,
        (1, args.prompt_length),
        generator=torch.Generator().manual_seed(20261004),
    )
    resident = prompt.cuda()
    input_sha = hashlib.sha256(prompt.numpy().tobytes()).hexdigest()
    torch.set_default_device("cuda")

    @torch.inference_mode()
    def request(mode, phase, prepared):
        """Include all table offload and labelled prompt/result copies in the request."""
        if phase == "decode":
            token, generated = prepared, []
        else:
            token, logits, _ = model(
                prompt.cuda() if mode == "transfer" else resident, 0
            )
            generated = [token]
            if phase == "prefill":
                return (
                    (logits.cpu(), token.cpu().view(1, 1))
                    if mode == "transfer"
                    else (logits, token.view(1, 1))
                )
        for step in range(1, args.generated_tokens):
            token, logits, _ = model(token.view(1, 1), args.prompt_length + step - 1)
            generated.append(token)
        tokens = torch.stack(generated, -1)
        return (logits.cpu(), tokens.cpu()) if mode == "transfer" else (logits, tokens)

    @torch.inference_mode()
    def timed(mode, phase):
        """Time the slowest rank, excluding prepared prefill and numerical checks."""
        prepared = model(resident, 0)[0] if phase == "decode" else None
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()
        begin = time.perf_counter()
        result = request(mode, phase, prepared)
        torch.cuda.synchronize()
        elapsed = torch.tensor(
            [time.perf_counter() - begin], dtype=torch.float64, device="cuda"
        )
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        return elapsed.item() * 1000, tuple(v.detach().cpu().clone() for v in result)

    record = dict(
        state="running",
        started=datetime.now(timezone.utc).isoformat(),
        rank=rank,
        backend=args.backend,
        round=args.round,
        allocation=os.environ.get("SLURM_JOB_ID"),
        node=os.uname().nodename,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        allocator=os.environ.get("PYTORCH_ALLOC_CONF"),
        model=manifest["model"],
        revision=manifest["revision"],
        input_sha256=input_sha,
        prompt_length=args.prompt_length,
        generated_tokens=args.generated_tokens,
        storage=storage,
        loading_seconds=loading_seconds,
        library_sha256=identities,
        reference_fix=dict(
            original=manifest["reference_sources"]["kernel.py"],
            corrected=digest(working / "kernel.py"),
        ),
        worker_sha256=worker_sha,
        host_affinity=cpus,
        host_numa=host_numa,
        measurements={},
    )
    try:
        for phase in ("prefill", "decode", "request"):
            record["measurements"][phase] = {}
            for mode in (
                ["resident", "transfer"]
                if args.round % 2 == 0
                else ["transfer", "resident"]
            ):
                print(rank, phase, mode, flush=True)
                for _ in range(args.warmups):
                    timed(mode, phase)
                samples, previous = [], None
                for _ in range(args.repetitions):
                    elapsed, result = timed(mode, phase)
                    if not torch.isfinite(result[0]).all() or (
                        previous is not None
                        and any(not torch.equal(a, b) for a, b in zip(previous, result))
                    ):
                        raise ValueError(
                            "Full-model logits/tokens are non-finite or not repeatable"
                        )
                    samples.append(elapsed)
                    previous = result
                logits, tokens = result
                numerator = (
                    args.prompt_length
                    if phase == "prefill"
                    else args.generated_tokens - (phase == "decode")
                )
                record["measurements"][phase][mode] = dict(
                    median_ms=statistics.median(samples),
                    samples_ms=samples,
                    tokens_per_second=numerator * 1000 / statistics.median(samples),
                    logits_sha256=hashlib.sha256(
                        logits.contiguous().numpy().tobytes()
                    ).hexdigest(),
                    logits=logits.float().tolist(),
                    tokens=tokens.tolist(),
                )
        resident = ((resident + 7) % 128000) + 3
        _, (logits, tokens) = timed("resident", "request")
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite changed-input logits")
        record.update(
            state="passed",
            changed_tokens=dict(
                logits=logits.float().tolist(),
                tokens=tokens.tolist(),
                logits_sha256=hashlib.sha256(
                    logits.contiguous().numpy().tobytes()
                ).hexdigest(),
            ),
            dispatch_counts=counts,
            finished=datetime.now(timezone.utc).isoformat(),
        )
    except BaseException as error:
        record.update(state="failed", error=repr(error))
        raise
    finally:
        output = args.output.with_name(
            args.output.stem + f".rank{rank}" + args.output.suffix
        )
        output.write_text(json.dumps(record) + "\n")
        if record["state"] == "passed":
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
