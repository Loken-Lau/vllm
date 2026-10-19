"""Exercise real model call sites across dtypes, padding cases and devices."""

import argparse
import json
import pathlib
import subprocess
from types import SimpleNamespace

import torch
import transformers
from transformers.models.qwen2_audio import Qwen2AudioEncoder
from transformers.models.qwen2_audio.configuration_qwen2_audio import (
    Qwen2AudioEncoderConfig,
)

from vllm.model_executor.models.audioflamingo3 import (
    AudioFlamingo3Encoder,
    _build_audio_encoder_attention_mask,
)
from vllm.model_executor.models.audioflamingo3 import (
    AudioFlamingo3ForConditionalGeneration as FlamingoModel,
)
from vllm.model_executor.models.qwen2_audio import (
    Qwen2AudioForConditionalGeneration as QwenModel,
)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(__file__).with_name("audit-audio-mask-matrix.json"),
    )
    args = parser.parse_args()
    assert torch.cuda.is_available()
    torch.manual_seed(62350)
    rows = []
    for device in ["cpu", "cuda"]:
        dtypes = (
            [torch.float32]
            if device == "cpu"
            else [torch.float32, torch.float16, torch.bfloat16]
        )
        for dtype in dtypes:
            for backend in [None, "sdpa", "eager"]:
                for name in ["qwen2_audio", "audioflamingo3"]:
                    config = Qwen2AudioEncoderConfig(
                        num_mel_bins=4,
                        d_model=16,
                        encoder_layers=2,
                        encoder_attention_heads=2,
                        encoder_ffn_dim=32,
                        max_source_positions=8,
                    )
                    config._attn_implementation = backend
                    cls = (
                        Qwen2AudioEncoder
                        if name == "qwen2_audio"
                        else AudioFlamingo3Encoder
                    )
                    encoder = cls(config).to(device=device, dtype=dtype).eval()
                    model = SimpleNamespace(
                        audio_tower=encoder, multi_modal_projector=torch.nn.Identity()
                    )
                    for lengths in [[16], [4], [5], [4, 5, 8, 9, 15, 16]]:
                        inputs = torch.randn(
                            len(lengths), 4, 16, device=device, dtype=dtype
                        )
                        feature_mask = (
                            torch.arange(16, device=device)[None, :]
                            < torch.tensor(lengths, device=device)[:, None]
                        )
                        captured = []
                        handle = encoder.register_forward_pre_hook(
                            lambda module,
                            args,
                            kwargs,
                            captured=captured: captured.append(
                                kwargs["attention_mask"]
                            ),
                            with_kwargs=True,
                        )
                        try:
                            if name == "qwen2_audio":
                                run = QwenModel._process_audio_input
                                actual = run(
                                    model,
                                    dict(
                                        type="audio_features",
                                        input_features=inputs,
                                        feature_attention_mask=feature_mask,
                                    ),
                                )
                            else:
                                run = FlamingoModel._encode_audio_features
                                actual = run(model, inputs, feature_mask)
                        finally:
                            handle.remove()
                        expected_lengths = (
                            torch.tensor(lengths, device=device) - 1
                        ) // 2 + 1
                        valid = (
                            torch.arange(8, device=device)[None, :]
                            < expected_lengths[:, None]
                        )
                        dense = torch.zeros(
                            len(lengths), 1, 8, 8, device=device, dtype=dtype
                        ).masked_fill_(~valid[:, None, None, :], float("-inf"))
                        torch.testing.assert_close(
                            captured[0].expand_as(dense), dense, rtol=0, atol=0
                        )
                        expected = encoder(inputs, attention_mask=dense)
                        if name == "qwen2_audio":
                            out_lengths = ((expected_lengths - 2) // 2 + 1).tolist()
                            expected = tuple(
                                row[:length]
                                for row, length in zip(
                                    expected.last_hidden_state, out_lengths
                                )
                            )
                            diffs = [
                                (x - y).abs().max().item()
                                for x, y in zip(actual, expected)
                            ]
                        else:
                            diffs = [(actual - expected).abs().max().item()]
                        torch.testing.assert_close(actual, expected)
                        rows.append(
                            dict(
                                device=device,
                                dtype=str(dtype),
                                backend=backend,
                                model=name,
                                lengths=lengths,
                                mask_shape=list(captured[0].shape),
                                storage_bytes=captured[0].untyped_storage().nbytes(),
                                max_abs_diff=max(diffs),
                            )
                        )
                    del encoder
    boundary_rows = []
    for width in [1, 2, 3, 14, 15, 16]:
        lengths = torch.tensor([0, 1, width])
        features = torch.arange(width)[None, :] < lengths[:, None]
        mask = _build_audio_encoder_attention_mask(
            features, dtype=torch.float32, device=torch.device("cpu")
        )
        expected_width = (width - 1) // 2 + 1
        keep = torch.arange(expected_width)[None, :] < ((lengths - 1) // 2 + 1)[:, None]
        expected = torch.zeros(3, 1, 1, expected_width).masked_fill_(
            ~keep[:, None, None, :], float("-inf")
        )
        torch.testing.assert_close(mask, expected, rtol=0, atol=0)
        boundary_rows.append(
            dict(mel_width=width, lengths=lengths.tolist(), shape=list(mask.shape))
        )
    # A control that demonstrates padding is semantically active: changing masked
    # values must not change the output, while removing the mask must change it.
    q = torch.randn(2, 2, 8, 8, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    mask = torch.zeros(2, 1, 1, 8, device="cuda")
    mask[:, :, :, 4:] = -float("inf")
    padded_v = v.clone()
    padded_v[:, :, 4:, :] += 100
    reference = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask
    )
    actual = torch.nn.functional.scaled_dot_product_attention(
        q, k, padded_v, attn_mask=mask
    )
    torch.testing.assert_close(reference, actual, rtol=0, atol=0)
    no_mask = torch.nn.functional.scaled_dot_product_attention(q, k, padded_v)
    control_diff = (actual - no_mask).abs().max().item()
    assert control_diff > 1
    result = dict(
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        torch=torch.__version__,
        transformers=transformers.__version__,
        device=torch.cuda.get_device_name(),
        encoder_cases=rows,
        boundary_cases=boundary_rows,
        padding_control=dict(masked_diff=0.0, unmasked_diff=control_diff),
    )
    target = args.output
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"PASS: {len(rows)} encoder comparisons, "
        f"{len(boundary_rows)} boundary cases and padding control.",
        flush=True,
    )
    print(
        "Maximum encoder difference:", max(x["max_abs_diff"] for x in rows), flush=True
    )


if __name__ == "__main__":
    main()
