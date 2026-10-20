"""Evaluate paired mask variants on a deterministic LibriSpeech test-clean slice."""

import argparse
import ast
import functools
import hashlib
import json
import pathlib
import random
import statistics
import subprocess
import time
import types

import numpy as np
import soundfile as sf
import torch
import transformers
from eval_audio_models import install_capture
from jiwer import process_words
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

from vllm import LLM, SamplingParams

SOURCE_COMMIT = "283d48f4ebb404b00e76264ad8f759d7e9a05dfd"
BATCH_SIZES = [1, 2, 4, 8, 16]
NORMALIZER_PATH = pathlib.Path(__file__).with_name("whisper-normalizer.json")
NORMALIZER_DATA = json.loads(NORMALIZER_PATH.read_text())
NORMALIZER = EnglishTextNormalizer(NORMALIZER_DATA)


def source_for(model_name):
    path = f"vllm/model_executor/models/{model_name}.py"
    return subprocess.check_output(
        ["git", "show", f"{SOURCE_COMMIT}:{path}"], text=True
    )


def set_variant(model, variant, source):
    module = __import__(type(model).__module__, fromlist=["*"])
    is_qwen = type(model).__name__.startswith("Qwen2Audio")
    if not hasattr(model, "_librispeech_mask_variants"):
        tree = ast.parse(source)
        if is_qwen:
            cls = next(
                node
                for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == type(model).__name__
            )
            function = next(
                node
                for node in cls.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "_process_audio_input"
            )
            compact = model._process_audio_input
        else:
            function = next(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "_build_audio_encoder_attention_mask"
            )
            compact = module._build_audio_encoder_attention_mask
        namespace = dict(vars(module))
        exec(
            compile(
                ast.Module(body=[function], type_ignores=[]),
                "pinned-unmodified-source",
                "exec",
            ),
            namespace,
        )
        baseline = namespace[function.name]
        if is_qwen:
            baseline = types.MethodType(baseline, model)
        model._librispeech_mask_variants = {
            "baseline": baseline,
            "compact": compact,
        }
    if is_qwen:
        model._process_audio_input = model._librispeech_mask_variants[variant]
    else:
        module._build_audio_encoder_attention_mask = model._librispeech_mask_variants[
            variant
        ]
    model._audio_mask_audit = []


def clear_mask_audit(model):
    model._audio_mask_audit.clear()
    torch.cuda.reset_peak_memory_stats()
    model._eval_start_memory = torch.cuda.memory_allocated()


def read_mask_audit(model):
    return {
        "masks": model._audio_mask_audit,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "incremental_peak_bytes": (
            torch.cuda.max_memory_allocated() - model._eval_start_memory
        ),
    }


def load_samples(root, count):
    dataset = root / ".models/librispeech-test-clean/LibriSpeech/test-clean"
    records = []
    for transcript_path in sorted(dataset.rglob("*.trans.txt")):
        for line in transcript_path.read_text().splitlines():
            sample_id, reference = line.split(" ", 1)
            audio_path = transcript_path.parent / f"{sample_id}.flac"
            duration = sf.info(audio_path).duration
            records.append(
                {
                    "id": sample_id,
                    "path": audio_path,
                    "reference": reference,
                    "duration": duration,
                }
            )
    records.sort(key=lambda row: row["duration"])
    selected = [
        records[min(len(records) - 1, int((i + 0.5) * len(records) / count))]
        for i in range(count)
    ]
    random.Random(62350).shuffle(selected)
    for row in selected:
        audio, sample_rate = sf.read(row["path"], dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        row["audio"] = np.asarray(audio, dtype=np.float32)
        row["sample_rate"] = sample_rate
        del row["path"]
    return records, selected


def levenshtein(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, ref_word in enumerate(reference, 1):
        current = [i]
        for j, hyp_word in enumerate(hypothesis, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (ref_word != hyp_word),
                )
            )
        previous = current
    return previous[-1]


def normalized_words(text):
    return NORMALIZER(text).split()


def build_prompt(model, audio_count, style):
    instruction = "Transcribe the speech."
    if style == "transcript_only":
        instruction = (
            "Transcribe the speech in English. Output only the transcription, "
            "without any introduction or explanation."
        )
    if model == "qwen2_audio":
        audio = "".join(
            f"Audio {index + 1}: <|audio_bos|><|AUDIO|><|audio_eos|>\n"
            for index in range(audio_count)
        )
        return (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{audio}{instruction}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{'<sound>' * audio_count}{instruction}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def summarize_latency(values):
    return {
        "calls": len(values),
        "median_ms": statistics.median(values),
        "p95_ms": sorted(values)[max(0, int(len(values) * 0.95) - 1)],
        "mean_ms": statistics.mean(values),
        "samples_ms": values,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["qwen2_audio", "audioflamingo3"], required=True
    )
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=BATCH_SIZES)
    parser.add_argument(
        "--prompt-style",
        choices=["original", "transcript_only"],
        default="transcript_only",
    )
    parser.add_argument("--logprobs", type=int, default=5)
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[2]
    _, samples = load_samples(root, args.samples)
    model_path = (
        root
        / ".models"
        / (
            "Qwen2-Audio-7B-Instruct"
            if args.model == "qwen2_audio"
            else "audio-flamingo-3-hf"
        )
    )
    llm = LLM(
        model=str(model_path),
        dtype="bfloat16",
        max_model_len=4096,
        max_num_seqs=max(args.batch_sizes),
        max_num_batched_tokens=32768,
        gpu_memory_utilization=0.75,
        enforce_eager=True,
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
        limit_mm_per_prompt={"audio": 1},
        seed=62350,
        disable_log_stats=True,
    )
    runtime = llm.apply_model(install_capture)
    source = source_for(args.model)
    sampling = SamplingParams(temperature=0, max_tokens=256, logprobs=args.logprobs)
    prompt = build_prompt(args.model, 1, args.prompt_style)
    warmup = samples[0]
    warmup_request = {
        "prompt": prompt,
        "multi_modal_data": {"audio": (warmup["audio"], warmup["sample_rate"])},
    }
    for variant in ["baseline", "compact"]:
        llm.apply_model(functools.partial(set_variant, variant=variant, source=source))
        llm.generate([warmup_request], sampling, use_tqdm=False)

    result = {
        "model": args.model,
        "dataset": "LibriSpeech test-clean",
        "dataset_archive_md5": "32fa31d27d2e1cad72775fee3f4849a9",
        "split_samples": 2620,
        "selected_samples": len(samples),
        "sampling": (
            "duration-stratified deterministic midpoint selection; "
            "shuffled with seed 62350"
        ),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "baseline_source_commit": SOURCE_COMMIT,
        "model_revision": (model_path / "source-revision.txt").read_text().strip(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": torch.cuda.get_device_name(),
        "runtime": runtime,
        "batch_sizes": args.batch_sizes,
        "prompt_style": args.prompt_style,
        "prompt": prompt,
        "requested_logprobs": args.logprobs,
        "warmup_calls_per_variant": 1,
        "generation_max_tokens": 256,
        "cache_settings": {"prefix": False, "multimodal_processor_gb": 0},
        "normalization": (
            "vLLM ASR test's Transformers Whisper EnglishTextNormalizer and "
            "pinned Whisper large-v3 normalizer.json"
        ),
        "normalizer_sha256": hashlib.sha256(NORMALIZER_PATH.read_bytes()).hexdigest(),
        "samples": [
            {
                "id": row["id"],
                "reference": row["reference"],
                "duration_seconds": row["duration"],
            }
            for row in samples
        ],
        "batch_results": {},
    }
    for batch_size in args.batch_sizes:
        variant_rows = {"baseline": [], "compact": []}
        latencies = {"baseline": [], "compact": []}
        mask_summaries = {
            variant: {"calls": 0, "audio_items": 0, "shapes": {}, "storage_bytes": {}}
            for variant in variant_rows
        }
        memory_peaks = {variant: [] for variant in variant_rows}
        mismatches = []
        batches = [
            samples[index : index + batch_size]
            for index in range(0, len(samples), batch_size)
        ]
        for batch_index, batch in enumerate(batches):
            inputs = [
                {
                    "prompt": prompt,
                    "multi_modal_data": {"audio": (row["audio"], row["sample_rate"])},
                }
                for row in batch
            ]
            variants = (
                ["baseline", "compact"]
                if batch_index % 2 == 0
                else ["compact", "baseline"]
            )
            paired_outputs = {}
            for variant in variants:
                llm.apply_model(
                    functools.partial(set_variant, variant=variant, source=source)
                )
                llm.apply_model(clear_mask_audit)
                started = time.perf_counter()
                outputs = llm.generate(inputs, sampling, use_tqdm=False)
                elapsed = time.perf_counter() - started
                latencies[variant].append(elapsed * 1000)
                paired_outputs[variant] = [
                    {
                        "text": output.outputs[0].text,
                        "token_ids": list(output.outputs[0].token_ids),
                        "output_tokens": len(output.outputs[0].token_ids),
                        "finish_reason": output.outputs[0].finish_reason,
                        "logprobs": [
                            {str(token): lp.logprob for token, lp in item.items()}
                            for item in (output.outputs[0].logprobs or [])
                        ],
                    }
                    for output in outputs
                ]
                variant_rows[variant].extend(paired_outputs[variant])
                audit_by_rank = llm.apply_model(read_mask_audit)
                masks = [mask for rank in audit_by_rank for mask in rank["masks"]]
                memory_peaks[variant].append(
                    [
                        {k: v for k, v in rank.items() if k != "masks"}
                        for rank in audit_by_rank
                    ]
                )
                assert masks
                summary = mask_summaries[variant]
                summary["calls"] += len(masks)
                for mask in masks:
                    batch_count = mask["shape"][0]
                    query_width = 1500 if variant == "baseline" else 1
                    assert mask["shape"][1:] == [1, query_width, 1500]
                    assert mask["storage_bytes"] == batch_count * query_width * 1500 * 2
                    summary["audio_items"] += batch_count
                    shape = str(mask["shape"])
                    summary["shapes"][shape] = summary["shapes"].get(shape, 0) + 1
                    key = str(mask["storage_bytes"])
                    summary["storage_bytes"][key] = (
                        summary["storage_bytes"].get(key, 0) + 1
                    )
            differing = [
                index
                for index, (a, b) in enumerate(
                    zip(
                        paired_outputs["baseline"],
                        paired_outputs["compact"],
                        strict=True,
                    )
                )
                if a["token_ids"] != b["token_ids"] or a["text"] != b["text"]
            ]
            if differing:
                controls = {}
                for variant in ["baseline", "compact"]:
                    llm.apply_model(
                        functools.partial(set_variant, variant=variant, source=source)
                    )
                    controls[variant] = []
                    for _ in range(2):
                        repeated = llm.generate(inputs, sampling, use_tqdm=False)
                        controls[variant].append(
                            [
                                {
                                    "text": out.outputs[0].text,
                                    "token_ids": list(out.outputs[0].token_ids),
                                }
                                for out in repeated
                            ]
                        )
                mismatches.append(
                    {
                        "batch_index": batch_index,
                        "sample_ids": [row["id"] for row in batch],
                        "differing_positions": differing,
                        "paired_outputs": paired_outputs,
                        "same_variant_repeats": controls,
                    }
                )
                print(
                    "Mismatch:",
                    batch_size,
                    batch_index,
                    [batch[i]["id"] for i in differing],
                    flush=True,
                )
        per_variant = {}
        references = [normalized_words(row["reference"]) for row in samples]
        word_count = sum(len(words) for words in references)
        for variant in variant_rows:
            hypotheses = [
                normalized_words(row["text"]) for row in variant_rows[variant]
            ]
            measured = process_words(
                [" ".join(ref) for ref in references],
                [" ".join(hyp) for hyp in hypotheses],
            )
            errors = measured.substitutions + measured.deletions + measured.insertions
            assert errors == sum(
                levenshtein(ref, hyp)
                for ref, hyp in zip(references, hypotheses, strict=True)
            )
            total_audio_seconds = sum(row["duration"] for row in samples)
            total_time = sum(latencies[variant]) / 1000
            per_variant[variant] = {
                "wer": errors / word_count,
                "word_errors": errors,
                "substitutions": measured.substitutions,
                "deletions": measured.deletions,
                "insertions": measured.insertions,
                "reference_words": word_count,
                "exact_text_matches": sum(
                    ref == hyp for ref, hyp in zip(references, hypotheses, strict=True)
                ),
                "mean_batch_ms": statistics.mean(latencies[variant]),
                "median_batch_ms": statistics.median(latencies[variant]),
                "p95_batch_ms": sorted(latencies[variant])[
                    max(0, int(len(latencies[variant]) * 0.95) - 1)
                ],
                "samples_per_second": len(samples) / total_time,
                "audio_seconds_per_second": total_audio_seconds / total_time,
                "max_output_tokens": max(
                    row["output_tokens"] for row in variant_rows[variant]
                ),
                "max_token_cap_reached": any(
                    row["output_tokens"] >= sampling.max_tokens
                    for row in variant_rows[variant]
                ),
                "batch_latency_ms": latencies[variant],
                "predictions": variant_rows[variant],
                "mask_capture": mask_summaries[variant],
                "batch_memory_peaks": memory_peaks[variant],
            }
            assert mask_summaries[variant]["audio_items"] == len(samples)
        result["batch_results"][str(batch_size)] = {
            "batch_count": len(batches),
            "alternating_pair_order": True,
            "all_text_token_ids_equal": not mismatches,
            "mismatches": mismatches,
            "baseline_compact_wer_delta": (
                per_variant["compact"]["wer"] - per_variant["baseline"]["wer"]
            ),
            "variants": per_variant,
        }
        print(
            json.dumps(
                {
                    "model": args.model,
                    "batch_size": batch_size,
                    "wer": {key: value["wer"] for key, value in per_variant.items()},
                    "mean_batch_ms": {
                        key: value["mean_batch_ms"]
                        for key, value in per_variant.items()
                    },
                    "equal": not mismatches,
                }
            ),
            flush=True,
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
