"""Measure one full-checkpoint SGLang worker, including request transfers."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path


def install_verifier():
    """Capture complete raw vocabulary logits only on untimed accuracy calls."""
    directory = os.environ.get("CAMBLAS_LLM_VERIFY_DIRECTORY")
    if not directory:
        return
    import torch
    import torch.distributed as dist
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    from camblas.sglang import dispatch_counts

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    step = 0

    def capture(
        original, sampler, logits_output, sampling_info, return_logprob, *a, **k
    ):
        """Record untimed raw vocabulary logits, then call the original sampler."""
        nonlocal step
        if return_logprob:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Accuracy capture must run outside CUDA graph capture"
                )
            rank = dist.get_rank()
            if step == 0:
                libraries = {}
                runtime_libraries = {}
                for line in Path("/proc/self/maps").read_text().splitlines():
                    fields = line.split()
                    if len(fields) < 6:
                        continue
                    path = Path(fields[-1])
                    camblas = path.name == "libcamblas_cuda.so" or path.name.startswith(
                        "_camblas_cuda_torch"
                    )
                    runtime = any(
                        part in path.parts
                        for part in (
                            "sgl_kernel",
                            "flashinfer",
                            "deep_gemm",
                            "triton",
                            "tilelang",
                            "tvm_ffi",
                        )
                    ) or path.name.startswith("sgl_kernel_jit_")
                    runtime |= "/.cache/flashinfer/" in str(path)
                    if path.is_file() and (camblas or runtime):
                        target_libraries = libraries if camblas else runtime_libraries
                        if str(path) in target_libraries:
                            continue
                        with path.open("rb") as library:
                            target_libraries[str(path)] = hashlib.file_digest(
                                library, "sha256"
                            ).hexdigest()
                (Path(directory) / f"rank{rank}-libraries.json").write_text(
                    json.dumps(libraries, indent=2) + "\n"
                )
                (Path(directory) / f"rank{rank}-runtime.json").write_text(
                    json.dumps(runtime_libraries, indent=2) + "\n"
                )
            target = Path(directory) / f"rank{rank}-step{step:03d}.pt"
            positions = k.get("positions", a[-1] if a else None)
            if positions is None:
                raise ValueError("Accuracy capture requires prediction positions")
            torch.save(
                dict(
                    logits=logits_output.next_token_logits.detach().float().cpu(),
                    rank=rank,
                    step=step,
                    prediction_position=int(positions.detach().cpu().flatten()[-1]),
                    allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                    allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                    host_dispatch_counts=dispatch_counts(),
                ),
                target,
            )
            step += 1
        return original(sampler, logits_output, sampling_info, return_logprob, *a, **k)

    HookRegistry.register(
        "sglang.srt.layers.sampler.Sampler.forward", capture, HookType.AROUND
    )


# SGLang spawns scheduler processes by re-importing this module. Register the
# shared accuracy hook in every process before Engine applies plugin hooks.
install_verifier()


def request(engine, token_ids, generated_tokens, *, accuracy=False):
    """Time CPU token input to completed CPU token output for one request."""
    start = time.perf_counter()
    first = None
    final = None
    for response in engine.generate(
        input_ids=token_ids,
        sampling_params=dict(
            temperature=0, max_new_tokens=generated_tokens, ignore_eos=True
        ),
        stream=True,
        return_logprob=accuracy,
    ):
        received = time.perf_counter()
        if response["meta_info"]["completion_tokens"] > 0 and first is None:
            first = received
        final = response
    finished = time.perf_counter()
    if final is None or first is None:
        raise ValueError("SGLang returned no generated tokens")
    meta = final["meta_info"]
    output = list(final["output_ids"])
    if meta["completion_tokens"] != generated_tokens or len(output) != generated_tokens:
        raise ValueError("Incomplete generation cannot be reported as throughput")
    if meta.get("cached_tokens", 0):
        raise ValueError("Prefix cache reuse would invalidate the prefill measurement")
    return dict(
        request_ms=(finished - start) * 1000,
        ttft_ms=(first - start) * 1000,
        decode_ms=(finished - first) * 1000,
        output_ids=output,
        meta_info=meta,
    )


def main():
    """Load the full model once, measure serial requests, then verify changed input."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--node-rank", type=int, default=0)
    args = parser.parse_args()
    case = json.loads(args.case.read_text())
    cutoff = datetime.fromisoformat(case["gpu_cutoff"])
    if datetime.now(timezone.utc) >= cutoff:
        raise RuntimeError("Allocation cutoff passed")
    engine_args = dict(case["engine_args"])
    engine_args["node_rank"] = args.node_rank
    if args.node_rank:
        logits_dir = args.output / "logits"
        while not logits_dir.is_dir():
            if datetime.now(timezone.utc) >= cutoff:
                raise RuntimeError(
                    "Allocation cutoff passed while waiting for rank zero"
                )
            time.sleep(0.1)
        os.environ["CAMBLAS_LLM_VERIFY_DIRECTORY"] = str(logits_dir.absolute())
        install_verifier()
        import sglang

        engine = sglang.Engine(**engine_args)
        engine.shutdown()
        return
    args.output.mkdir(parents=True, exist_ok=False)
    if "weights_verification" in case:
        from bench.verify_llm_weights import check_verified_files

        check_verified_files(
            Path(case["model_directory"]), case["weights_verification"]
        )
    logits_dir = args.output / "logits"
    logits_dir.mkdir()
    # The parent imported before this path existed; explicitly register now.
    # Spawned children inherit the variable and register during module import.
    os.environ["CAMBLAS_LLM_VERIFY_DIRECTORY"] = str(logits_dir.absolute())
    install_verifier()

    import sglang
    import torch
    from tokenizers import Tokenizer

    directory = Path(case["model_directory"])
    config = json.loads((directory / "config.json").read_text())
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    prefix = config.get("bos_token_id", 0)
    if isinstance(prefix, list):
        prefix = prefix[0] if prefix else None
    prefix_ids = [] if prefix is None else [prefix]
    prompt = case["prompt_length"]
    text = "Explain how matrix multiplication is used in language model inference. "
    changed_text = "Describe the differences between prefill and token generation. "
    token_ids = (
        prefix_ids
        + tokenizer.encode(text * prompt, add_special_tokens=False).ids[
            : prompt - len(prefix_ids)
        ]
    )
    changed_ids = (
        prefix_ids
        + tokenizer.encode(changed_text * prompt, add_special_tokens=False).ids[
            : prompt - len(prefix_ids)
        ]
    )
    if (
        len(token_ids) != prompt
        or len(changed_ids) != prompt
        or token_ids == changed_ids
    ):
        raise ValueError("Both distinct prompts must have the requested token length")

    started = datetime.now(timezone.utc).isoformat()
    setup = time.perf_counter()
    engine = sglang.Engine(**engine_args)
    setup_seconds = time.perf_counter() - setup
    try:
        samples = []
        expected = None
        for index in range(case["warmups"] + case["repetitions"]):
            if datetime.now(timezone.utc) >= cutoff:
                raise RuntimeError("Allocation cutoff passed")
            result = request(engine, token_ids, case["generated_tokens"])
            if expected is not None and result["output_ids"] != expected:
                raise ValueError("Repeated greedy generations differ")
            expected = result["output_ids"]
            if index >= case["warmups"]:
                samples.append(result)
                print(
                    json.dumps(
                        dict(
                            repetition=index - case["warmups"],
                            request_ms=result["request_ms"],
                            ttft_ms=result["ttft_ms"],
                            decode_tokens_per_second=(case["generated_tokens"] - 1)
                            * 1000
                            / result["decode_ms"],
                        )
                    ),
                    flush=True,
                )

        accuracy = []
        for ids in (token_ids, token_ids, changed_ids):
            accuracy.append(
                request(engine, ids, case["accuracy_tokens"], accuracy=True)
            )
        if accuracy[0]["output_ids"] != accuracy[1]["output_ids"]:
            raise ValueError("Repeated accuracy generations differ")
        vocabulary = config.get("text_config", config)["vocab_size"]
        captures = []
        dispatches = {}
        for path in sorted(logits_dir.glob("rank*-step*.pt")):
            record = torch.load(path, map_location="cpu", weights_only=True)
            if record["allow_tf32"] or record["allow_bf16_reduced_precision_reduction"]:
                raise ValueError("A worker changed the FP32 accumulation policy")
            # The overlap scheduler may sample one further token after the
            # requested length. Verify every returned token, identifying it by
            # its prediction position instead of counting look-ahead calls.
            if (
                not prompt - 1
                <= record["prediction_position"]
                < (prompt - 1 + case["accuracy_tokens"])
            ):
                continue
            captures.append(str(path.relative_to(args.output)))
            logits = record["logits"]
            if logits.shape != (1, vocabulary) or not torch.isfinite(logits).all():
                raise ValueError("Invalid complete vocabulary logits")
            dispatches[record["rank"]] = record["host_dispatch_counts"]
        if len(captures) != case["tensor_parallel"] * case["accuracy_tokens"] * 3:
            raise ValueError("Missing full-vocabulary accuracy captures")
        if set(dispatches) != set(range(case["tensor_parallel"])):
            raise ValueError("Missing accuracy or dispatch evidence for a model rank")
        if case["backend"] == "camblas" and any(
            not all(
                counts.get(name, 0)
                for name in case.get("camblas_operations", ["linear"])
            )
            for counts in dispatches.values()
        ):
            raise ValueError(
                "A CAMBLAS rank did not dispatch every requested native operation"
            )
        from bench.compare_llm import load_verified_logits

        load_verified_logits(args.output, dict(case=case, accuracy_files=captures))
        if case.get("profile"):
            engine.start_profile(
                output_dir=str((args.output / "profile").absolute()),
                activities=["CPU", "GPU"],
                record_shapes=True,
                profile_id=case["model"].split("/")[-1],
            )
            request(engine, changed_ids, 16)
            engine.stop_profile()
        measurements = {
            phase: dict(
                median_ms=statistics.median(r[key] for r in samples),
                seconds=[r[key] / 1000 for r in samples],
            )
            for phase, key in (
                ("prefill", "ttft_ms"),
                ("decode", "decode_ms"),
                ("request", "request_ms"),
            )
        }
        output = dict(
            state="passed",
            case=case,
            started_at=started,
            finished_at=datetime.now(timezone.utc).isoformat(),
            setup_seconds=setup_seconds,
            torch=torch.__version__,
            torch_cuda=torch.version.cuda,
            sglang_version=sglang.__version__,
            sglang_source=str(Path(sglang.__file__).absolute()),
            packages={
                name: importlib.metadata.version(name)
                for name in case.get(
                    "runtime_packages",
                    ("torch", "sglang", "sglang-kernel", "flashinfer-python"),
                )
            },
            input_ids=token_ids,
            changed_input_ids=changed_ids,
            input_sha256=hashlib.sha256(json.dumps(token_ids).encode()).hexdigest(),
            measurements=measurements,
            samples=samples,
            accuracy=accuracy,
            accuracy_files=captures,
            host_dispatch_counts=dispatches,
            server_info=engine.get_server_info(),
        )
        if "weights_verification" in case:
            check_verified_files(directory, case["weights_verification"])
        (args.output / "result.json").write_text(json.dumps(output, indent=2) + "\n")
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
