"""Compare the checked-out mask builder with its pinned unmodified baseline."""

import ast
import gc
import json
import pathlib
import subprocess

import torch
import torch.nn.functional as F
from torch.utils.benchmark import Timer

from vllm.model_executor.models.audioflamingo3 import (
    _build_audio_encoder_attention_mask as compact,
)

ROOT = pathlib.Path(__file__).resolve().parents[2] / "vllm"
BASELINE_REF = "fb61ac07bd1d5b8b9722c80085b99ef770efb265"
source = subprocess.check_output(
    ["git", "show", f"{BASELINE_REF}:vllm/model_executor/models/audioflamingo3.py"],
    cwd=ROOT,
    text=True,
)
node = next(
    node
    for node in ast.parse(source).body
    if isinstance(node, ast.FunctionDef)
    and node.name == "_build_audio_encoder_attention_mask"
)
namespace = {"torch": torch}
exec(
    compile(ast.Module(body=[node], type_ignores=[]), "git-HEAD-baseline", "exec"),
    namespace,
)
baseline = namespace[node.name]


@torch.inference_mode()
def main():
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(62350)
    results = {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", BASELINE_REF], cwd=ROOT, text=True
        ).strip(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "dtype": str(dtype),
        "rows": [],
    }
    for batch_size in [1, 4, 16]:
        lengths = torch.linspace(1000, 3000, batch_size, device=device).long()
        features = torch.arange(3000, device=device)[None, :] < lengths[:, None]
        row = {"batch": batch_size, "mel_length": 3000, "encoder_length": 1500}
        outputs = []
        for name, fn in [("baseline", baseline), ("compact", compact)]:
            for _ in range(10):
                fn(features, dtype=dtype, device=device)
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            output = fn(features, dtype=dtype, device=device)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() - before
            measurements = [
                Timer(
                    stmt="fn(features, dtype=dtype, device=device)",
                    globals=dict(fn=fn, features=features, dtype=dtype, device=device),
                ).blocked_autorange(min_run_time=0.3)
                for _ in range(5)
            ]
            row[name] = {
                "shape": list(output.shape),
                "storage_bytes": output.untyped_storage().nbytes(),
                "peak_allocation_bytes": peak,
                "median_us_per_repeat": [m.median * 1e6 for m in measurements],
                "iqr_us_per_repeat": [m.iqr * 1e6 for m in measurements],
            }
            outputs.append(output)
        q, k, v = [
            torch.randn(batch_size, 2, 1500, 64, device=device, dtype=dtype)
            for _ in range(3)
        ]
        reference = F.scaled_dot_product_attention(q, k, v, attn_mask=outputs[0])
        actual = F.scaled_dot_product_attention(q, k, v, attn_mask=outputs[1])
        torch.testing.assert_close(actual, reference)
        row["sdpa_max_abs_diff"] = (actual - reference).abs().max().item()
        row["storage_reduction_factor"] = (
            row["baseline"]["storage_bytes"] / row["compact"]["storage_bytes"]
        )
        results["rows"].append(row)
        print(json.dumps(row), flush=True)
        del outputs, output, q, k, v, reference, actual
    target = pathlib.Path(__file__).with_name("benchmark-audio-masks.json")
    target.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
