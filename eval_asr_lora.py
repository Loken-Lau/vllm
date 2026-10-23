"""Measure actual native encoder LoRA routing against independent matrix products."""

import ast
import functools
import hashlib
import json
import pathlib
import subprocess
import textwrap
import time
import types

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import save_file
from transformers import AutoProcessor

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.utils.gpu_sync_debug import gpu_sync_allowed

BASELINE = "f3871c3cbf8a608f7ee833ec4c2c5277f77bee41"
MODEL_FILE = "vllm/model_executor/models/qwen3_omni_moe_thinker.py"


def install_audit(model, source, matrices):
    import vllm.model_executor.models.qwen3_omni_moe_thinker as module

    namespace = dict(vars(module))
    exec(compile(source, "baseline_audio_encoder_forward", "exec"), namespace)
    model.audio_lora_audit = {
        "baseline_forward": namespace["forward"],
        "fixed_forward": type(model.audio_tower).forward,
        "records": [], "variant": "fixed", "adapters": {},
        "matrices": {int(key): (a.to(model.audio_tower.device), b.to(model.audio_tower.device))
                     for key, (a, b) in matrices.items()},
    }

    def before_tower(encoder, args, kwargs):
        with gpu_sync_allowed():
            model.audio_lora_audit["lengths"] = kwargs["feature_lens"].tolist()
            model.audio_lora_audit["valid_lengths"] = kwargs["aftercnn_lens"].tolist()

    @gpu_sync_allowed()
    def after_projection(layer, args, output):
        audit = model.audio_lora_audit
        x = args[0]
        actual = output.reshape(-1, output.shape[-1])
        projected_rows = actual.shape[0]
        if x.ndim == 3:
            chunks = []
            for length in audit["lengths"]:
                full, tail = divmod(length, 100)
                chunks.extend([100] * full)
                if tail:
                    chunks.append(tail)
            cnn_lengths = torch.tensor([(length + 7) // 8 for length in chunks], device=x.device)
            mask = torch.arange(x.shape[1], device=x.device)[None, :] < cnn_lengths[:, None]
            with gpu_sync_allowed():
                x = x[mask]
                actual = actual[mask.reshape(-1)]
        base = layer.base_layer(x)
        if isinstance(base, tuple):
            base = base[0]
        expected = base.float()
        expected_delta = torch.zeros_like(expected)
        start = 0
        for length, rows in zip(audit["lengths"], audit["valid_lengths"], strict=True):
            adapter_id = audit["adapters"].get(str(length), 0)
            if adapter_id:
                a, b = audit["matrices"][adapter_id]
                expected_delta[start:start + rows] = (x[start:start + rows].float() @ a.float().T) @ b.float().T
            start += rows
        expected = (expected + expected_delta).to(actual.dtype)
        difference = (actual - expected).abs()
        with gpu_sync_allowed():
            audit["records"].append({
                "variant": audit["variant"], "lengths": audit["lengths"],
                "valid_lengths": audit["valid_lengths"],
                "projected_rows": projected_rows, "valid_rows": start,
                "max_abs_projection_error": difference.max().item(),
                "mean_abs_projection_error": difference.float().mean().item(),
                "max_abs_expected_lora_delta": expected_delta.abs().max().item(),
                "rows_with_error_above_0p02": int((difference.max(dim=1).values > 0.02).sum()),
            })

    model.audio_tower.register_forward_pre_hook(before_tower, with_kwargs=True)
    model.audio_tower.conv_out.register_forward_hook(after_projection)
    return {"model": type(model).__name__, "projection": type(model.audio_tower.conv_out).__name__}


def configure(model, variant, adapters):
    audit = model.audio_lora_audit
    model.audio_tower.forward = types.MethodType(audit[f"{variant}_forward"], model.audio_tower)
    audit["variant"] = variant
    audit["adapters"] = adapters
    audit["records"] = []


def records(model):
    return model.audio_lora_audit["records"]


def serialize(outputs):
    return [{"text": row.outputs[0].text, "token_ids": list(row.outputs[0].token_ids),
             "finish_reason": row.outputs[0].finish_reason,
             "logprobs": [{str(token): value.logprob for token, value in group.items()}
                          for group in row.outputs[0].logprobs]}
            for row in outputs]


def main():
    root = pathlib.Path(__file__).resolve().parents[2]
    artifacts = pathlib.Path(__file__).resolve().parent
    model_path = root / ".models/Qwen3-ASR-0.6B"
    raw_source = subprocess.check_output(["git", "show", f"{BASELINE}:{MODEL_FILE}"], text=True)
    tree = ast.parse(raw_source)
    encoder = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Qwen3OmniMoeAudioEncoder")
    forward = next(node for node in encoder.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    source = textwrap.dedent(ast.get_source_segment(raw_source, forward))
    config = json.loads((model_path / "config.json").read_text())["thinker_config"]["audio_config"]
    width = config["downsample_hidden_size"] * ((config["num_mel_bins"] + 7) // 8)
    generator = torch.Generator().manual_seed(61608)
    a = torch.zeros(8, width, dtype=torch.bfloat16)
    a[0, 0] = 1
    b = torch.zeros(config["d_model"], 8, dtype=torch.bfloat16)
    b[:, 0] = 1
    matrices = {1: (a, b), 2: (a, -b)}
    for adapter_id, (left, right) in matrices.items():
        destination = artifacts / f"adapter-{adapter_id}"
        destination.mkdir(exist_ok=True)
        save_file({"base_model.model.thinker.audio_tower.conv_out.lora_A.weight": left,
                   "base_model.model.thinker.audio_tower.conv_out.lora_B.weight": right},
                  str(destination / "adapter_model.safetensors"))
        (destination / "adapter_config.json").write_text(json.dumps({
            "peft_type": "LORA", "r": 8, "lora_alpha": 8,
            "target_modules": ["conv_out"], "bias": "none",
            "base_model_name_or_path": str(model_path), "task_type": "CAUSAL_LM",
        }, indent=2) + "\n")

    audio, sr = sf.read(root / ".contributions/infra-screen/mary_had_lamb.ogg", dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    clips = [np.asarray(audio[:samples], dtype=np.float32) for samples in [16160, 32160, 48160]]
    prompt = ("<|im_start|>system\n<|im_end|>\n<|im_start|>user\n"
              "<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
              "<|im_start|>assistant\n")
    processor = AutoProcessor.from_pretrained(model_path)
    feature_lengths = [int(processor(text=prompt, audio=clip, return_tensors="pt")["input_features_mask"].sum())
                       for clip in clips]
    inputs = [{"prompt": prompt, "multi_modal_data": {"audio": (clip, sr)}} for clip in clips]
    loras = [LoRARequest(f"adapter-{i}", i, str(artifacts / f"adapter-{i}")) for i in [1, 2]]
    llm = LLM(
        model=str(model_path), dtype="bfloat16", max_model_len=1024,
        max_num_seqs=8, max_num_batched_tokens=2048,
        gpu_memory_utilization=0.4, enforce_eager=True,
        enable_prefix_caching=False, mm_processor_cache_gb=0,
        limit_mm_per_prompt={"audio": 1}, seed=61608,
        enable_lora=True, enable_tower_connector_lora=True,
        max_lora_rank=8, max_loras=2, disable_log_stats=True,
    )
    result = {"baseline_commit": BASELINE,
              "baseline_forward_sha256": hashlib.sha256(source.encode()).hexdigest(),
              "candidate_source_sha256": hashlib.sha256((root / "vllm" / MODEL_FILE).read_bytes()).hexdigest(),
              "checkpoint_revision": (model_path / "source-revision.txt").read_text().strip(),
              "torch": torch.__version__, "feature_lengths": feature_lengths,
              "runtime": llm.apply_model(functools.partial(install_audit, source=source, matrices=matrices)),
              "cases": {}}
    sampling = SamplingParams(temperature=0, max_tokens=64, logprobs=5)
    cases = [("same_adapter", [0, 1, 2], [1, 1, 1]),
             ("single_1", [0], [1]), ("single_2", [1], [2]),
             ("mixed", [0, 1], [1, 2]), ("swapped", [0, 1], [2, 1]),
             ("mixed_base", [0, 1, 2], [0, 0, 0])]
    for case, indices, adapter_ids in cases:
        result["cases"][case] = {}
        for variant in ["baseline", "fixed"]:
            adapters = {str(feature_lengths[index]): adapter_id
                        for index, adapter_id in zip(indices, adapter_ids, strict=True)}
            llm.apply_model(functools.partial(configure, variant=variant, adapters=adapters))
            started = time.perf_counter()
            outputs = llm.generate([inputs[index] for index in indices], sampling,
                                   lora_request=[loras[i - 1] if i else None for i in adapter_ids],
                                   use_tqdm=False)
            result["cases"][case][variant] = {
                "seconds_including_audit_hooks": time.perf_counter() - started,
                "outputs": serialize(outputs), "projection_audit": llm.apply_model(records),
            }
            (artifacts / "native-lora-routing.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps({"case": case, "variant": variant,
                              "projection_audit": result["cases"][case][variant]["projection_audit"]}), flush=True)
    print("All native cases completed; audit timings are not latency benchmarks.", flush=True)


if __name__ == "__main__":
    main()
