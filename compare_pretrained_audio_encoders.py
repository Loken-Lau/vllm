"""Isolate pretrained encoder mask effects from autoregressive scheduling noise."""

import argparse
import json
import pathlib

import torch
from eval_librispeech_batch_sizes import load_samples
from safetensors import safe_open
from transformers import AutoConfig, WhisperFeatureExtractor
from transformers.models.qwen2_audio import Qwen2AudioEncoder

from vllm.model_executor.models.audioflamingo3 import AudioFlamingo3Encoder


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, choices=["qwen2_audio", "audioflamingo3"]
    )
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[2]
    _, samples = load_samples(root, 128)
    path = (
        root
        / ".models"
        / (
            "Qwen2-Audio-7B-Instruct"
            if args.model == "qwen2_audio"
            else "audio-flamingo-3-hf"
        )
    )
    config = AutoConfig.from_pretrained(path).audio_config
    config._attn_implementation = "sdpa"
    cls = Qwen2AudioEncoder if args.model == "qwen2_audio" else AudioFlamingo3Encoder
    with torch.device("meta"):
        encoder = cls(config)
    index = json.loads((path / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    files = sorted(
        {shard for key, shard in index.items() if key.startswith("audio_tower.")}
    )
    state = {}
    for file in files:
        with safe_open(path / file, framework="pt", device="cpu") as source:
            for key in source.keys():  # noqa: SIM118
                if key.startswith("audio_tower."):
                    state[key.removeprefix("audio_tower.")] = source.get_tensor(key).to(
                        device="cuda", dtype=torch.bfloat16
                    )
    encoder.load_state_dict(state, strict=True, assign=True)
    encoder.eval()
    del state
    extractor = WhisperFeatureExtractor(
        feature_size=config.num_mel_bins,
        sampling_rate=16000,
        chunk_length=30,
        return_attention_mask=True,
    )
    rows = []
    for batch_size in [1, 2, 4, 8, 16]:
        max_diff = 0.0
        max_repeat_diff = 0.0
        comparisons = 0
        all_equal = True
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            features = extractor(
                [row["audio"] for row in batch],
                sampling_rate=16000,
                return_tensors="pt",
                return_attention_mask=True,
            )
            inputs = features.input_features.to(device="cuda", dtype=torch.bfloat16)
            lengths = features.attention_mask.sum(-1).to(device="cuda")
            width = (inputs.shape[-1] - 1) // 2 + 1
            valid_lengths = (lengths - 1) // 2 + 1
            pad = torch.arange(width, device="cuda")[None, :] >= valid_lengths[:, None]
            compact = torch.zeros(
                len(batch), 1, 1, width, device="cuda", dtype=torch.bfloat16
            ).masked_fill_(pad[:, None, None, :], -float("inf"))
            dense = compact.expand(len(batch), 1, width, width).clone()

            def encode(mask, inputs=inputs):
                output = encoder(inputs, attention_mask=mask)
                return (
                    output.last_hidden_state
                    if hasattr(output, "last_hidden_state")
                    else output
                )

            before = encode(dense)
            after = encode(compact)
            repeat = encode(dense)
            torch.testing.assert_close(after, before)
            diff = (after - before).abs().max().item()
            repeat_diff = (repeat - before).abs().max().item()
            max_diff = max(max_diff, diff)
            max_repeat_diff = max(max_repeat_diff, repeat_diff)
            all_equal = all_equal and torch.equal(after, before)
            comparisons += len(batch)
        row = dict(
            batch_size=batch_size,
            samples=comparisons,
            max_abs_diff=max_diff,
            exact_equal=all_equal,
            dense_repeat_max_abs_diff=max_repeat_diff,
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = dict(
        model=args.model,
        model_revision=(path / "source-revision.txt").read_text().strip(),
        backend="sdpa",
        dtype="bfloat16",
        samples=128,
        rows=rows,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
