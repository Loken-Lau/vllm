"""Independent-engine A/B inference with mask capture on real audio towers."""

import argparse
import json
import pathlib
import subprocess
import time

import numpy as np
import soundfile as sf
import torch

from vllm import LLM, SamplingParams


def install_capture(model):
    model._audio_mask_audit = []

    def capture(module, args, kwargs):
        mask = kwargs["attention_mask"]
        model._audio_mask_audit.append(
            {
                "shape": list(mask.shape),
                "dtype": str(mask.dtype),
                "device": str(mask.device),
                "storage_bytes": mask.untyped_storage().nbytes(),
            }
        )

    model.audio_tower.register_forward_pre_hook(capture, with_kwargs=True)
    return {
        "model_class": type(model).__name__,
        "encoder_class": type(model.audio_tower).__name__,
        "attention_backend": model.audio_tower.config._attn_implementation,
    }


def serialize_outputs(outputs):
    return [
        {
            "text": output.outputs[0].text,
            "token_ids": list(output.outputs[0].token_ids),
            "prompt_tokens": len(output.prompt_token_ids),
            "logprobs": [
                {str(token): lp.logprob for token, lp in group.items()}
                for group in output.outputs[0].logprobs
            ],
        }
        for output in outputs
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["qwen2_audio", "audioflamingo3"], required=True
    )
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[2]
    if args.model == "qwen2_audio":
        model_path = root / ".models/Qwen2-Audio-7B-Instruct"
        placeholder = "Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\n"
        long_duration = 29
    else:
        model_path = root / ".models/audio-flamingo-3-hf"
        placeholder = "<sound>"
        long_duration = 35
    audio, sr = sf.read(pathlib.Path(__file__).with_name("mary_had_lamb.ogg"))
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    long_clip = np.tile(audio, int(np.ceil(long_duration * sr / len(audio))))[
        : long_duration * sr
    ]
    clips = [audio[: 2 * sr], audio, long_clip]
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{placeholder}Transcribe the speech.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    requests = [
        {"prompt": prompt, "multi_modal_data": {"audio": (clip, sr)}} for clip in clips
    ]
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
    cases = {}
    for case, inputs in [("single", requests[:1]), ("mixed", requests)]:
        llm.generate(inputs, sampling, use_tqdm=False)
        timings = []
        repeat_outputs = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            outputs = llm.generate(inputs, sampling, use_tqdm=False)
            timings.append(time.perf_counter() - started)
            repeat_outputs.append(serialize_outputs(outputs))
        cases[case] = {"batch_seconds": timings, "repeated_outputs": repeat_outputs}
    masks = llm.apply_model(lambda model: model._audio_mask_audit)
    result = dict(
        model=args.model,
        model_revision=(model_path / "source-revision.txt").read_text().strip(),
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        torch=torch.__version__,
        device=torch.cuda.get_device_name(),
        runtime=runtime,
        repeats=args.repeats,
        warmup_batches_per_case=1,
        audio_seconds=[len(clip) / sr for clip in clips],
        long_audio_kind="repeated real speech",
        cases=cases,
        captured_masks=masks,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ["cases", "captured_masks"]}
        ),
        flush=True,
    )
    print(
        "Captured masks:",
        sorted({tuple(x["shape"]) for r in masks for x in r}),
        flush=True,
    )
    llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
