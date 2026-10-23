"""Validate the full pretrained Omni audio tower; no language-model claims."""

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import textwrap

import numpy as np
import soundfile as sf
from safetensors import safe_open
import torch
from transformers import WhisperFeatureExtractor
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoderConfig,
)

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.lora import LoRAConfig
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.lora.layers import LoRAMapping, ReplicatedLinearWithLoRA
from vllm.lora.punica_wrapper import get_punica_wrapper
import vllm.model_executor.models.qwen3_omni_moe_thinker as module

ARTIFACTS = Path(__file__).resolve().parent
ROOT = ARTIFACTS.parents[1]
REVISION = "26291f793822fb6be9555850f06dfe95f2d7e695"
BASELINE = "f3871c3cbf8a608f7ee833ec4c2c5277f77bee41"
MODEL_FILE = "vllm/model_executor/models/qwen3_omni_moe_thinker.py"


@torch.inference_mode()
def main():
    model_path = ROOT / ".models/Qwen3-Omni-audio-26291f7"
    shard = model_path / "model-00001-of-00015.safetensors"
    metadata = json.loads((ARTIFACTS / "omni-model-metadata.json").read_text())
    assert metadata["sha"] == REVISION
    expected_shard = next(s for s in metadata["siblings"] if s["rfilename"] == shard.name)
    with shard.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    assert shard.stat().st_size == expected_shard["size"]
    assert digest == expected_shard["lfs"]["sha256"]
    index = json.loads((ARTIFACTS / "omni-weight-index.json").read_text())
    keys = sorted(k for k in index["weight_map"] if k.startswith("thinker.audio_tower."))
    assert len(keys) == 525
    assert all(index["weight_map"][k] == shard.name for k in keys)
    source = subprocess.check_output(["git", "show", f"{BASELINE}:{MODEL_FILE}"], text=True)
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "Qwen3OmniMoeAudioEncoder")
    forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    namespace = dict(vars(module))
    exec(compile(textwrap.dedent(ast.get_source_segment(source, forward)), "pinned_original_forward", "exec"), namespace)
    baseline = namespace["forward"]
    config = Qwen3OmniMoeAudioEncoderConfig(**json.loads((ARTIFACTS / "omni-config.json").read_text())["thinker_config"]["audio_config"])
    extractor = WhisperFeatureExtractor.from_pretrained(model_path)
    audio, sr = sf.read(ROOT / ".contributions/infra-screen/mary_had_lamb.ogg", dtype="float32")
    assert sr == extractor.sampling_rate
    # Repeat real speech to cover the 801-frame attention-boundary case.
    audio = np.tile(audio, 2)
    feature = extractor(audio, sampling_rate=sr, padding=False, return_tensors="pt")["input_features"][0]
    assert feature.shape[1] >= 801
    result = {"model": "Qwen/Qwen3-Omni-30B-A3B-Instruct", "revision": REVISION,
              "scope": "full 32-layer pretrained audio tower, not full LLM",
              "shard_sha256": digest, "baseline": BASELINE,
              "source_sha256": hashlib.sha256((ROOT / "vllm" / MODEL_FILE).read_bytes()).hexdigest(),
              "torch": torch.__version__, "device": torch.cuda.get_device_name(),
              "unadapted": [], "lora": []}
    fd, dist_path = tempfile.mkstemp()
    os.close(fd)
    try:
        with set_current_vllm_config(VllmConfig()):
            init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="nccl", distributed_init_method=f"file://{dist_path}")
            initialize_model_parallel(1, 1)
            original_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.bfloat16)
            encoder = module.Qwen3OmniMoeAudioEncoder(config).to("cuda").eval()
            torch.set_default_dtype(original_dtype)
            with safe_open(shard, framework="pt", device="cpu") as checkpoint:
                loaded = encoder.load_weights((k.removeprefix("thinker.audio_tower."), checkpoint.get_tensor(k)) for k in keys)
            assert set(dict(encoder.named_parameters())) <= loaded
            result["loaded_parameters"] = len(loaded)
            result["checkpoint_tensors"] = len(keys)
            result["attention_backend"] = str(encoder.attn_backend)
            for lengths in [[1], [99], [100], [101], [200], [201], [1, 100], [101, 201], [799, 801]]:
                features = torch.cat([feature[:, :n] for n in lengths], dim=1).to(device="cuda", dtype=torch.bfloat16)
                lens = torch.tensor(lengths, device="cuda")
                output_lens = torch.tensor([(n // 100) * 13 + ((n % 100) + 7) // 8 for n in lengths], device="cuda")
                expected = baseline(encoder, features, lens, output_lens)
                actual = encoder(features, lens, output_lens)
                assert actual.shape == expected.shape == (int(output_lens.sum()), config.output_dim)
                cosine = torch.nn.functional.cosine_similarity(actual.float().reshape(1, -1), expected.float().reshape(1, -1)).item()
                row = {"lengths": lengths, "shape": list(actual.shape), "equal": torch.equal(actual, expected),
                       "max_abs_diff": (actual - expected).abs().max().item(), "cosine_similarity": cosine}
                result["unadapted"].append(row)
                print(json.dumps(row), flush=True)

            layer = ReplicatedLinearWithLoRA(encoder.conv_out)
            lora_config = LoRAConfig(max_loras=2, max_lora_rank=8, lora_dtype=torch.bfloat16)
            punica = get_punica_wrapper(4096, 16, "cuda", lora_config=lora_config)
            layer.create_lora_weights(2, lora_config)
            layer.set_mapping(punica)
            a = torch.zeros(8, layer.input_size, device="cuda", dtype=torch.bfloat16)
            b = torch.zeros(layer.output_size, 8, device="cuda", dtype=torch.bfloat16)
            a[0, 0] = 1
            b[:, 0] = 1
            layer.set_lora(0, a, b)
            layer.set_lora(1, a, -b)
            encoder.conv_out = layer
            audit = {}

            def capture(current_layer, args, output):
                x = args[0]
                flattened = x.reshape(-1, x.shape[-1])
                valid = flattened[audit["padded_valid_indices"]] if x.ndim == 3 else flattened
                observed = output.reshape(-1, output.shape[-1])
                if x.ndim == 3:
                    observed = observed[audit["padded_valid_indices"]]
                base = current_layer.base_layer(valid)
                delta = valid[:, :1].float() * audit["signs"]
                expected = (base.float() + delta).to(base.dtype)
                audit["projected_rows"] = flattened.shape[0]
                audit["exact"] = torch.equal(observed, expected)
                audit["max_abs_error"] = (observed - expected).abs().max().item()
                audit["adapter_has_effect"] = bool(torch.any(expected != base))

            handle = layer.register_forward_hook(capture)
            for lengths, adapters in [([101, 201], [1, 1]), ([101, 201], [1, 2]), ([201, 101], [2, 1]), ([1, 100], [1, 2])]:
                counts = [(n // 100) * 13 + ((n % 100) + 7) // 8 for n in lengths]
                features = torch.cat([feature[:, :n] for n in lengths], dim=1).to(device="cuda", dtype=torch.bfloat16)
                # Keep chunks in audio-item order, including each tail.
                chunks = []
                for n in lengths:
                    chunks.extend([100] * (n // 100))
                    if n % 100:
                        chunks.append(n % 100)
                width = (max(chunks) + 7) // 8
                valid_indices = torch.tensor([i * width + t for i, n in enumerate(chunks) for t in range((n + 7) // 8)], device="cuda")
                mapping = [adapter for adapter, count in zip(adapters, counts, strict=True) for _ in range(count)]
                for variant, call in [("baseline", baseline), ("candidate", type(encoder).forward)]:
                    punica.update_metadata(LoRAMapping(tuple(mapping), tuple(adapters), is_prefill=True), [1, 2, None], 3, 1)
                    audit.clear()
                    audit["padded_valid_indices"] = valid_indices
                    audit["signs"] = torch.tensor([1 if adapter == 1 else -1 for adapter in mapping], device="cuda").float().unsqueeze(1)
                    output = call(encoder, features, torch.tensor(lengths, device="cuda"), torch.tensor(counts, device="cuda"))
                    row = {k: v for k, v in audit.items() if k not in {"padded_valid_indices", "signs"}}
                    row.update(variant=variant, lengths=lengths, adapters=adapters, valid_rows=sum(counts), output_shape=list(output.shape))
                    result["lora"].append(row)
                    print(json.dumps(row), flush=True)
                    if variant == "candidate":
                        assert row["projected_rows"] == row["valid_rows"]
                        assert row["exact"] and row["adapter_has_effect"]
            handle.remove()
            cleanup_dist_env_and_memory()
    finally:
        Path(dist_path).unlink(missing_ok=True)
    (ARTIFACTS / "omni-audio-validation.json").write_text(json.dumps(result, indent=2) + chr(10))
    print("Omni pretrained audio tower validation completed", flush=True)


if __name__ == "__main__":
    main()
