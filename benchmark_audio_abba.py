"""Control process/order effects by alternating exact baseline and compact code."""

import argparse
import ast
import functools
import importlib
import json
import pathlib
import statistics
import subprocess
import time
import types

import numpy as np
import soundfile as sf
from eval_audio_models import install_capture, serialize_outputs

from vllm import LLM, SamplingParams


def select_variant(model, variant, source):
    module = importlib.import_module(type(model).__module__)
    is_qwen = type(model).__name__.startswith("Qwen2Audio")
    if not hasattr(model, "_mask_abba_variants"):
        tree = ast.parse(source)
        if is_qwen:
            cls = next(
                x
                for x in tree.body
                if isinstance(x, ast.ClassDef) and x.name == type(model).__name__
            )
            node = next(
                x
                for x in cls.body
                if isinstance(x, ast.FunctionDef) and x.name == "_process_audio_input"
            )
            compact = model._process_audio_input
        else:
            node = next(
                x
                for x in tree.body
                if isinstance(x, ast.FunctionDef)
                and x.name == "_build_audio_encoder_attention_mask"
            )
            compact = module._build_audio_encoder_attention_mask
        namespace = dict(vars(module))
        exec(
            compile(
                ast.Module(body=[node], type_ignores=[]), "exact-git-baseline", "exec"
            ),
            namespace,
        )
        baseline = namespace[node.name]
        if is_qwen:
            baseline = types.MethodType(baseline, model)
        model._mask_abba_variants = {"baseline": baseline, "compact": compact}
    if is_qwen:
        model._process_audio_input = model._mask_abba_variants[variant]
    else:
        module._build_audio_encoder_attention_mask = model._mask_abba_variants[variant]
    model._audio_mask_audit = []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["qwen2_audio", "audioflamingo3"], required=True
    )
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[2]
    is_qwen = args.model == "qwen2_audio"
    model_path = (
        root
        / ".models"
        / ("Qwen2-Audio-7B-Instruct" if is_qwen else "audio-flamingo-3-hf")
    )
    placeholder = (
        "Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\n" if is_qwen else "<sound>"
    )
    long_duration = 29 if is_qwen else 35
    audio, sr = sf.read(pathlib.Path(__file__).with_name("mary_had_lamb.ogg"))
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    long_clip = np.tile(audio, int(np.ceil(long_duration * sr / len(audio))))[
        : long_duration * sr
    ]
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{placeholder}Transcribe the speech.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    requests = [
        {"prompt": prompt, "multi_modal_data": {"audio": (clip, sr)}}
        for clip in [audio[: 2 * sr], audio, long_clip]
    ]
    source = subprocess.check_output(
        [
            "git",
            "show",
            "283d48f4ebb404b00e76264ad8f759d7e9a05dfd:"
            f"vllm/model_executor/models/{args.model}.py",
        ],
        text=True,
    )
    llm = LLM(
        model=str(model_path),
        dtype="bfloat16",
        max_model_len=2048,
        max_num_seqs=4,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=0.60,
        enforce_eager=True,
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
        limit_mm_per_prompt={"audio": 1},
        seed=62350,
        disable_log_stats=True,
    )
    runtime = llm.apply_model(install_capture)
    sampling = SamplingParams(temperature=0, max_tokens=32, logprobs=5)
    rounds = []
    reference_outputs = {}
    for variant in ["baseline", "compact", "compact", "baseline"]:
        llm.apply_model(
            functools.partial(select_variant, variant=variant, source=source)
        )
        cases = {}
        for name, inputs in [("single", requests[:1]), ("mixed", requests)]:
            for _ in range(5):
                llm.generate(inputs, sampling, use_tqdm=False)
            times, outputs = [], []
            for _ in range(20):
                started = time.perf_counter()
                actual = llm.generate(inputs, sampling, use_tqdm=False)
                times.append((time.perf_counter() - started) * 1000)
                serialized = serialize_outputs(actual)
                if name not in reference_outputs:
                    reference_outputs[name] = serialized
                assert serialized == reference_outputs[name]
                outputs.append(serialized)
            cases[name] = dict(
                batch_ms=times,
                median_ms=statistics.median(times),
                population_stddev_ms=statistics.pstdev(times),
                outputs=outputs,
            )
        masks = llm.apply_model(lambda model: model._audio_mask_audit)
        expected_queries = 1500 if variant == "baseline" else 1
        for rank in masks:
            assert rank
            assert all(mask["shape"][2] == expected_queries for mask in rank)
        row = dict(variant=variant, cases=cases, captured_masks=masks)
        rounds.append(row)
        print(
            variant,
            {
                name: {
                    k: v for k, v in case.items() if k not in ["batch_ms", "outputs"]
                }
                for name, case in cases.items()
            },
            flush=True,
        )
    result = dict(
        model=args.model,
        runtime=runtime,
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        order=[x["variant"] for x in rounds],
        warmups_per_case_per_round=5,
        repeats_per_case_per_round=20,
        all_text_token_ids_and_top5_logprobs_exactly_equal=True,
        rounds=rounds,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
