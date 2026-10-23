"""Paired pretrained ASR evaluation with pinned baseline encoder source."""

import ast
import functools
import hashlib
import json
import pathlib
import random
import subprocess
import textwrap
import types

import soundfile as sf
import torch
import transformers
from jiwer import process_words
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

from vllm import LLM, SamplingParams
from vllm.utils.gpu_sync_debug import gpu_sync_allowed

BASELINE = "f3871c3cbf8a608f7ee833ec4c2c5277f77bee41"
MODEL_FILE = "vllm/model_executor/models/qwen3_omni_moe_thinker.py"
PROMPT = (
    "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n"
    "<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
    "<|im_start|>assistant\n"
)


def install(model, baseline_source):
    import vllm.model_executor.models.qwen3_omni_moe_thinker as module

    namespace = dict(vars(module))
    exec(compile(baseline_source, "pinned_baseline_forward", "exec"), namespace)
    model.dataset_audit = {
        "baseline": namespace["forward"],
        "fixed": type(model.audio_tower).forward,
        "comparisons": [],
    }

    @gpu_sync_allowed()
    def compare(encoder, args, kwargs, output):
        if encoder.forward.__func__ is model.dataset_audit["baseline"]:
            return
        expected = model.dataset_audit["baseline"](encoder, *args, **kwargs)
        model.dataset_audit["comparisons"].append({
            "shape": list(output.shape),
            "equal": torch.equal(output, expected),
            "max_abs_diff": (output - expected).abs().max().item(),
        })

    model.audio_tower.register_forward_hook(compare, with_kwargs=True)
    return {"encoder": type(model.audio_tower).__name__,
            "attention_backend": str(model.audio_tower.attn_backend)}


def set_variant(model, variant):
    model.audio_tower.forward = types.MethodType(model.dataset_audit[variant], model.audio_tower)
    model.dataset_audit["comparisons"] = []


def get_comparisons(model):
    return model.dataset_audit["comparisons"]


def main():
    artifacts = pathlib.Path(__file__).resolve().parent
    root = artifacts.parents[1]
    repo = root / "vllm"
    model_path = root / ".models/Qwen3-ASR-0.6B"
    dataset = root / ".models/librispeech-test-clean/LibriSpeech/test-clean"
    all_samples = []
    for path in sorted(dataset.rglob("*.trans.txt")):
        for line in path.read_text().splitlines():
            sample_id, reference = line.split(" ", 1)
            audio_path = path.with_name(sample_id + ".flac")
            all_samples.append({"id": sample_id, "reference": reference,
                                "path": audio_path, "seconds": sf.info(audio_path).duration})
    all_samples.sort(key=lambda row: (row["seconds"], row["id"]))
    samples = [all_samples[int((i + 0.5) * len(all_samples) / 64)] for i in range(64)]
    random.Random(61608).shuffle(samples)
    inputs = []
    for sample in samples:
        audio, sr = sf.read(sample["path"], dtype="float32")
        sample["audio_sha256"] = hashlib.sha256(sample["path"].read_bytes()).hexdigest()
        del sample["path"]
        inputs.append({"prompt": PROMPT, "multi_modal_data": {"audio": (audio, sr)}})
    raw_source = subprocess.check_output(["git", "show", f"{BASELINE}:{MODEL_FILE}"], cwd=repo, text=True)
    encoder = next(n for n in ast.parse(raw_source).body if isinstance(n, ast.ClassDef) and n.name == "Qwen3OmniMoeAudioEncoder")
    forward = next(n for n in encoder.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    baseline_source = textwrap.dedent(ast.get_source_segment(raw_source, forward))
    normalizer_path = root / ".contributions/infra-screen/whisper-normalizer.json"
    normalizer = EnglishTextNormalizer(json.loads(normalizer_path.read_text()))
    references = [normalizer(s["reference"]) for s in samples]
    result = {"baseline": BASELINE, "baseline_forward_sha256": hashlib.sha256(baseline_source.encode()).hexdigest(),
              "candidate_sha256": hashlib.sha256((repo / MODEL_FILE).read_bytes()).hexdigest(),
              "checkpoint_revision": (model_path / "source-revision.txt").read_text().strip(),
              "normalizer_sha256": hashlib.sha256(normalizer_path.read_bytes()).hexdigest(),
              "torch": torch.__version__, "transformers": transformers.__version__,
              "device": torch.cuda.get_device_name(), "samples": samples, "cases": {}}
    llm = LLM(model=str(model_path), dtype="bfloat16", max_model_len=2048,
              max_num_seqs=8, max_num_batched_tokens=4096, gpu_memory_utilization=0.4,
              enforce_eager=True, enable_prefix_caching=False, mm_processor_cache_gb=0,
              limit_mm_per_prompt={"audio": 1}, seed=61608, disable_log_stats=True)
    result["runtime"] = llm.apply_model(functools.partial(install, baseline_source=baseline_source))
    sampling = SamplingParams(temperature=0, max_tokens=256, logprobs=5)
    for batch_size in [1, 8]:
        case = result["cases"][str(batch_size)] = {}
        for variant in ["baseline", "fixed"]:
            llm.apply_model(functools.partial(set_variant, variant=variant))
            rows = []
            for start in range(0, len(inputs), batch_size):
                outputs = llm.generate(inputs[start:start + batch_size], sampling, use_tqdm=False)
                for output in outputs:
                    generated = output.outputs[0]
                    transcription = generated.text.split("<asr_text>", 1)[-1]
                    rows.append({"text": generated.text, "normalized": normalizer(transcription),
                                 "token_ids": list(generated.token_ids), "finish_reason": generated.finish_reason,
                                 "logprobs": [{str(token): lp.logprob for token, lp in group.items()} for group in generated.logprobs]})
            words = process_words(references, [r["normalized"] for r in rows])
            case[variant] = {"outputs": rows, "wer": words.wer, "hits": words.hits,
                             "substitutions": words.substitutions, "deletions": words.deletions,
                             "insertions": words.insertions, "encoder_comparisons": llm.apply_model(get_comparisons)}
            if variant == "fixed":
                case["equal_text_token_pairs"] = sum(
                    a["text"] == b["text"] and a["token_ids"] == b["token_ids"]
                    for a, b in zip(case["baseline"]["outputs"], rows, strict=True))
            (artifacts / "dataset-evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps({"batch_size": batch_size, "variant": variant,
                              "wer": words.wer, "pairs": case.get("equal_text_token_pairs"),
                              "encoder": case[variant]["encoder_comparisons"]}), flush=True)
    llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
