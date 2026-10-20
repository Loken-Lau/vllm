# Audio encoder mask validation artifacts

These are standalone validation artifacts for vLLM issue #62350, separate from
the three-file source PR. Raw JSON and logs are retained, including unfavorable
timings. No end-to-end speedup or WER improvement is claimed.

## Environment and source provenance

- Source baseline: `283d48f4ebb404b00e76264ad8f759d7e9a05dfd`.
- Initial microbenchmark / regression baseline: `fb61ac07bd1d5b8b9722c80085b99ef770efb265`.
  The affected files are identical at these two revisions.
- Source patch: `audio-encoder-compact-mask.patch`; only two model implementations
  and the existing audio processing test file change.
- Python 3.12.15, torch 2.14.0+cu130, Transformers 5.19.0. Supplementary compatibility
  runs use Transformers 5.16.1 and huggingface-hub 1.33.0 in an isolated target.
- Editable local Python source, official nightly native extensions from
  `852989f580820849113d67d13fd8053e279d0444`. The installed version string still
  contains the earlier editable-install revision; JSON `source_commit` records
  the actual checkout revision.
- NVIDIA A800 80GB PCIe (SM80), driver 550.54.15, CUDA 13 compatibility library.
  A separate GPU on the host runs existing services and was not used.
- Pretrained Qwen/Qwen2-Audio-7B-Instruct revision
  `0a095220c30b7b31434169c3086508ef3ea5bf0a` and nvidia/audio-flamingo-3-hf revision
  `7d4bae64ee29878af6504ae6f6bb3e40492838ad`. All LFS weight shards were SHA-256
  checked before inference; download logs are included.

## Artifact map

| Claim / check | Script | Raw evidence |
| --- | --- | --- |
| Mask storage and builder timing | `benchmark_audio_masks.py` | `benchmark-audio-masks.json`, `.log` |
| Both real encoder call sites, dtype/backend/valid-length matrix | `audit_audio_mask_matrix.py` | `audit-audio-mask-matrix{,-tf516}.json`, `.log` |
| Pretrained output equivalence and actual runtime masks | `eval_audio_models.py` | `{qwen2-audio,audioflamingo3}-{baseline,compact}-v2.json`, `.log` |
| Check every measured A/B output | `compare_audio_e2e.py` | `e2e-audio-models-comparison.json` |
| Control process/order effects on latency | `benchmark_audio_abba.py`, `summarize_abba.py` | `{qwen2_audio,audioflamingo3}-abba.json`, `abba-comparison.json`, `.log` |
| Regression test catches original square-mask allocation | Added pytest cases in patch | `pytest-audio-masks-baseline.log` |
| Current and minimum Transformers regression runs | Existing pytest file | `pytest-audio-masks-latest.log`, `pytest-audio-masks-tf516.log` |
| Style, types and CI tethering | pre-commit | `pre-commit-audio-masks-latest.log` |

## Reproduction

The standalone scripts use the original workspace layout:

```text
workspace/
  vllm/                         # source repo + .venv
  .tools/bin/uv                 # or replace with your uv executable
  .models/Qwen2-Audio-7B-Instruct/
  .models/audio-flamingo-3-hf/
  .contributions/infra-screen/   # copy artifacts here
```

Check out the source baseline in `vllm/`, install its CUDA development/test
environment according to vLLM's contributing guide, then apply the source patch
without committing it. The baseline runner relies on this unstaged source diff
and restores it with an EXIT trap. Do not run it alongside another process that
imports these two model modules from the same checkout.

Download the two pinned public checkpoints into the paths above. The included
download helpers use hf-mirror.com because direct Hugging Face access was not
available on this host. You can use the original Hugging Face hub instead; retain
the pinned revisions. Create each model directory's `source-revision.txt` with
its revision followed by a newline. The long audio case repeats the same real
speech asset; it is explicitly not a second independent recording.

Download the audio asset without redistributing it here:

```bash
curl -fL https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/mary_had_lamb.ogg \
  -o ../.contributions/infra-screen/mary_had_lamb.ogg
```

From `vllm/`, run:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:/usr/local/cuda-13.0/lib64
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_GPU_SYNC_CHECK=error
# Needed only for our trusted apply_model hook in this local offline process.
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

../.tools/bin/uv run --no-project --python .venv/bin/python python -m pytest \
  tests/models/multimodal/processing/test_audioflamingo3.py -q

canhazgpu run --nonblock --gpu-ids 1 --timeout 5m -- \
  ../.tools/bin/uv run --no-project --python .venv/bin/python python \
  ../.contributions/infra-screen/benchmark_audio_masks.py

canhazgpu run --nonblock --gpu-ids 1 --timeout 5m -- \
  ../.tools/bin/uv run --no-project --python .venv/bin/python python \
  ../.contributions/infra-screen/audit_audio_mask_matrix.py

canhazgpu run --nonblock --gpu-ids 1 --timeout 10m -- \
  ../.tools/bin/uv run --no-project --python .venv/bin/python python \
  ../.contributions/infra-screen/eval_audio_models.py --model qwen2_audio \
  --output ../.contributions/infra-screen/qwen2-audio-compact-v2.json

canhazgpu run --nonblock --gpu-ids 1 --timeout 10m -- \
  ../.tools/bin/uv run --no-project --python .venv/bin/python python \
  ../.contributions/infra-screen/eval_audio_models.py --model audioflamingo3 \
  --output ../.contributions/infra-screen/audioflamingo3-compact-v2.json

canhazgpu run --nonblock --gpu-ids 1 --timeout 20m -- \
  bash ../.contributions/infra-screen/run_baseline_e2e.sh

../.tools/bin/uv run --no-project --python .venv/bin/python python \
  ../.contributions/infra-screen/compare_audio_e2e.py

for model in qwen2_audio audioflamingo3; do
  canhazgpu run --nonblock --gpu-ids 1 --timeout 10m -- \
    ../.tools/bin/uv run --no-project --python .venv/bin/python python \
    ../.contributions/infra-screen/benchmark_audio_abba.py --model "$model" \
    --output "../.contributions/infra-screen/${model}-abba.json"
done

../.tools/bin/uv run --no-project --python .venv/bin/python python \
  ../.contributions/infra-screen/summarize_abba.py
```

`canhazgpu` is this host's GPU reservation utility. On another machine select an
otherwise idle GPU with `CUDA_VISIBLE_DEVICES` and run the inner command directly.
The CUDA compatibility library is specific to this host's driver/toolkit setup.

For minimum-version compatibility without modifying the main environment:

```bash
../.tools/bin/uv pip install --python .venv/bin/python --no-deps \
  --target /tmp/vllm-audio-tf516 transformers==5.16.1 huggingface-hub==1.33.0

PYTHONPATH=/tmp/vllm-audio-tf516 .venv/bin/python -m pytest \
  tests/models/multimodal/processing/test_audioflamingo3.py -q

canhazgpu run --nonblock --gpu-ids 1 --timeout 5m -- \
  env PYTHONPATH=/tmp/vllm-audio-tf516 \
  ../.tools/bin/uv run --no-project --python .venv/bin/python python \
  ../.contributions/infra-screen/audit_audio_mask_matrix.py \
  --output ../.contributions/infra-screen/audit-audio-mask-matrix-tf516.json

GOPROXY=https://goproxy.cn,direct .venv/bin/pre-commit run --files \
  vllm/model_executor/models/qwen2_audio.py \
  vllm/model_executor/models/audioflamingo3.py \
  tests/models/multimodal/processing/test_audioflamingo3.py
```

## Limits

The E2E runs are BF16, TP=1, eager execution on one Ampere GPU, with greedy
generation capped at 32 tokens, one warmup and ten measured batches per case.
Prefix caching and multimodal processor caching are disabled. Encoder hooks
verify the runtime backend and mask allocation. Output parity is a smoke
evaluation using one speech recording at several lengths, including an AF3
35-second two-chunk case. It is not a dataset-level WER/accuracy evaluation.

The supplemental encoder matrix uses real small randomly initialized encoders;
it is not a pretrained-model evaluation. It covers CPU FP32 and CUDA FP32/FP16/
BF16 with default, SDPA and eager selection, and single/mixed valid lengths.
Boundary mask-builder checks include zero valid keys; they do not claim that a
full model accepts empty audio. A negative control confirms that unmasking padded
keys changes SDPA outputs.

The initial separate-engine latency samples include first-measured-iteration
spikes and mixed directions of change. These raw samples have not been removed.
The additional same-engine ABBA experiment replaces only the mask construction
with exact AST-extracted baseline code, reusing the same weights and worker. It
runs baseline/compact/compact/baseline, five warmups then twenty measurements per
case per round (forty measurements per variant/case). Hooks verify each variant's
mask shape. All measured outputs are compared exactly, including top-5 logprobs.
ABBA medians change from -0.87% to +0.42%, with overlapping sample ranges; this
supports no end-to-end speedup claim. The baseline reference was HEAD during
execution; the script now pins that same revision to remain reproducible after
the source patch is committed.

No other GPU architectures, tensor-parallel configurations, quantization modes
or new FlashAttention encoder dispatch were evaluated. Upstream CI status is
separate from these local results.
