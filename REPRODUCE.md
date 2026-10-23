# Reproducing the Qwen3 audio LoRA experiments

Use a vLLM editable checkout at baseline `f3871c3cbf8a608f7ee833ec4c2c5277f77bee41`
with the change under review applied. The validation scripts expect the checkout at
`<workspace>/vllm` and this artifact directory at
`<workspace>/.contributions/vllm-new-20261023`. The Python environment used here is
recorded in `requirements-frozen.txt`; this includes the actual precompiled binary
dependencies used for the tests. Do not replace those binary dependencies with
current nightlies and assume identical ABI or results.

The host driver was 550.54.15. CUDA 13.0 compatibility libraries were already
installed on this host; commands used:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:/usr/local/cuda/lib64
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_GPU_SYNC_CHECK=error
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
```

The serialization setting is for the local `LLM.apply_model` audit callbacks.
Timing values recorded with these callbacks are not serving benchmarks.

## Regression test

```bash
../.tools/bin/uv run --no-project --python .venv/bin/python python -m pytest \
  tests/lora/test_qwen3_asr.py -v -s
```

To reproduce the before case, reverse only `production-final.patch`, preserving
the test, and run the same command. Restore the patch afterwards. The baseline
log must fail on `26 projected rows vs 14 valid rows`; missing dependencies and
environment errors are not valid reproductions.

## Pinned checkpoints and samples

- Qwen3-ASR-0.6B revision: `5eb144179a02acc5e5ba31e748d22b0cf3e303b0`.
  Place its standard model files in `<workspace>/.models/Qwen3-ASR-0.6B` and
  record that revision in `source-revision.txt`.
- Qwen3-Omni-30B-A3B-Instruct revision:
  `26291f793822fb6be9555850f06dfe95f2d7e695`. Only its first safetensors shard
  and `preprocessor_config.json` are needed by `validate_omni_audio.py`. Put
  these files in `<workspace>/.models/Qwen3-Omni-audio-26291f7`.
  The script checks the shard SHA-256 against `omni-model-metadata.json`
  before loading all 525 audio-tower tensors through vLLM's weight loader.
- The short audio is the public vLLM `AudioAsset("mary_had_lamb")` fixture.
  Copy it to `<workspace>/.contributions/infra-screen/mary_had_lamb.ogg`.
- Dataset: original LibriSpeech test-clean, extracted under
  `<workspace>/.models/librispeech-test-clean/LibriSpeech/test-clean`.
  `eval_dataset.py` deterministically selects 64 utterances by duration,
  shuffles with seed 61608, and records their IDs, durations, references,
  and audio-file SHA-256 in `dataset-evaluation.json`.
- Copy the accompanying `whisper-normalizer.json` to
  `<workspace>/.contributions/infra-screen/whisper-normalizer.json`.

The checkpoint metadata is included so the revision, LFS shard sizes, and hashes
can be independently verified. Model weights and the speech dataset are not
included in the artifact branch.

## Audits

Run each script through the same `uv run --no-project --python .venv/bin/python`
prefix used above, from the vLLM checkout:

```bash
python ../.contributions/vllm-new-20261023/eval_asr_lora.py
python ../.contributions/vllm-new-20261023/eval_dataset.py
python ../.contributions/vllm-new-20261023/audit_results.py
python ../.contributions/vllm-new-20261023/validate_omni_audio.py
```

These shorthand `python` commands denote the uv-managed interpreter, not system
Python. `check_encoder_boundaries.py` uses the same prefix and ran on GPU 0.
The dataset audit recomputes word-edit distance independently of `jiwer` and
verifies the ASR weight SHA-256. The Omni script exercises the complete audio
tower, including actual Punica kernels with same and mixed sparse adapters;
it does not load or evaluate the 30B language model, vision tower, or talker.

## Checks

```bash
.venv/bin/python -m pre_commit run --files \
  vllm/model_executor/models/qwen3_omni_moe_thinker.py \
  tests/lora/test_qwen3_asr.py docs/features/lora.md .buildkite/test_areas/lora.yaml
git diff --check
```

`experiment-report.md` states the measured results and limitations. Failed setup
attempts are not counted as model-validation evidence.
