# Qwen3 audio tower LoRA row alignment

## Environment and issue

- Host: Intel Xeon Gold 5416S, 64 logical CPUs, 755 GiB RAM, NVIDIA A100 80GB and A800 80GB.
- Runtime: Ubuntu 22.04 kernel 5.15, NVIDIA driver 550.54.15, CUDA 13.0 compatibility libraries, PyTorch 2.14.0+cu130, Transformers 5.19.0, Python 3.12.15.
- Upstream baseline: `f3871c3cbf8a608f7ee833ec4c2c5277f77bee41`.
- Issue: [#61608](https://github.com/vllm-project/vllm/issues/61608), specifically its Qwen3-ASR and Qwen3-Omni audio section. This work does not claim to fix the other model cases.
- Duplicate check: open PR #50793 fixes Qwen3-ASR QKV adapter name mapping. Open PR #62494 quantizes Qwen3 audio linears and removes a host synchronization, but its current diff retains padded `conv_out` projection before the post-CNN mask. Neither implements this row-alignment fix. Open PRs for LLaVA and Ultravox address other models.

## Change

The shared Qwen3 audio encoder padded partial mel chunks to 100 frames, then ran the LoRA-wrapped `conv_out` on every padded post-CNN row before filtering invalid rows. Tower mappings contain only valid rows. The encoder now gathers valid chunk/position pairs before `conv_out`, applies positional embeddings using the retained within-chunk positions, and leaves convolution and attention chunk boundaries unchanged.

## Reproducer

The repository regression test constructs the real `Qwen3-ASR-0.6B` LLM, loads a synthetic rank-8 `conv_out` adapter, and batches 1.01, 2.01, and 3.01 second clips. Its sparse adapter has one nonzero A element and one nonzero B column, so the expected update is exactly the first input feature copied to each output feature. It asserts the wrapped projection receives exactly the valid post-CNN row count and that the adapter update matches an independent base-layer calculation exactly.

On the upstream baseline, the test failed inside the real LLM call with `26 projected rows vs 14 valid rows`. With the change restored, the same test passed. This was repeated against the final formatted patch.

Command, run with `CUDA_VISIBLE_DEVICES=1`, `VLLM_USE_V2_MODEL_RUNNER=1`, `VLLM_GPU_SYNC_CHECK=error`, `VLLM_WORKER_MULTIPROC_METHOD=spawn`, and the CUDA 13 compatibility library path:

```bash
../.tools/bin/uv run --no-project --python .venv/bin/python python -m pytest tests/lora/test_qwen3_asr.py -v -s
```

Results: upstream baseline `1 failed` with the row-count assertion during the real LLM request; candidate `1 passed` and completed generation. Detailed logs: `pytest-e2e-baseline-final.log` and `pytest-e2e-final-restored.log`.

## Encoder correctness

Compared the candidate encoder against the pinned upstream forward with the same inputs, random weights and seed on the A100. This boundary matrix used a small one-layer config (`d_model=32`, eight mel bins), not the pretrained model. All 13 shape cases passed exactly (`torch.equal`) in both fp32 and bf16, 26/26 total. Cases cover feature lengths 1, 7, 8, 9, 99, 100, 101, 200, 201, mixed batches `[1,100]`, `[101,201]`, `[100,200]`, and `[799,801]`. Output rows matched valid post-CNN lengths in every case. See `encoder-boundaries.json` and `encoder-boundaries.log`.

## Pretrained evaluation

Used `Qwen/Qwen3-ASR-0.6B` pinned at revision `5eb144179a02acc5e5ba31e748d22b0cf3e303b0`; its weight shard SHA-256 matched Hub metadata (`79d6cbd4c98c7bbffe9db2edac07f56cd6637d0d5944b27f6c2b8353840323ea`). Evaluated a deterministic, duration-stratified 64-utterance subset of LibriSpeech test-clean with greedy decoding, bf16, eager/V2 runner, and batch sizes 1 and 8. Both variants used the same weights, requests, and seed.

| Requests per batch | Variant | Word errors / reference words | WER | Exact text and token pairs | Normalized text pairs |
|---:|---|---:|---:|---:|---:|
| 1 | Baseline | 32 / 1,284 | 2.4922% | 64 / 64 | 64 / 64 |
| 1 | Candidate | 32 / 1,284 | 2.4922% | 64 / 64 | 64 / 64 |
| 8 | Baseline | 32 / 1,284 | 2.4922% | 64 / 64 | 64 / 64 |
| 8 | Candidate | 32 / 1,284 | 2.4922% | 63 / 64 | 64 / 64 |

An independent dynamic-programming audit recomputed the edit counts and hypotheses; it matched evaluator WER for all four runs. The sole non-identical batch-8 pair changed punctuation/capitalization (`dignity.` vs `dignity."`); normalized words and total errors were unchanged. Encoder outputs were exactly identical on 49/64 batch-1 calls and 18/21 batch-8 calls; the other calls had maximum absolute differences of 0.05884 and 0.01734 respectively. This evidence is retained; the results are not described as bitwise model parity. The observed subset showed no WER change. This does not establish parity on other data, models, GPU types, or configurations. See `dataset-evaluation.json` and `independent-wer-audit.json`.

The native LoRA audit used a deterministic sparse rank-8 adapter on real ASR runs. For feature lengths 101, 201, and 301, the encoder runner made two calls. The baseline projected 26 rows for 14 valid rows in the first call and 91 rows for 67 valid rows in the second; the candidate projected 14 and 67 rows respectively. Baseline adapter output differed from an independent reference by up to 0.16016 in the second call; candidate projections matched exactly. Same-adapter, mixed-adapter, swapped-adapter, and no-adapter cases are recorded in `native-lora-routing.json`.

## Additional pretrained Omni audio validation

Loaded all 525 audio-tower checkpoint tensors from Qwen3-Omni-30B-A3B-Instruct
revision `26291f793822fb6be9555850f06dfe95f2d7e695` through the native weight
loader (397 native parameters after QKV packing). Verified the entire first
checkpoint shard against the published SHA-256
`f95d142b727fbd0698f0ed9478e222d4f29467df855e21862e46bd1401049065`.
This uses the complete pretrained 32-layer, width-1280 audio tower and real
speech features, including repeated speech for the 799/801-frame boundary case.

- Nine unadapted input cases: six exactly equal, maximum absolute output
  difference 0.001983642578125, minimum flattened cosine similarity 0.99990791.
  Shape and valid-row counts agree in every case. These numbers describe
  measured differences, not a guarantee of bitwise parity or output quality.
- Four actual Punica/ReplicatedLinearWithLoRA cases: same-adapter, mixed-adapter,
  reversed order, and short/full-chunk mixed input. Original projection errors
  versus the independent sparse-LoRA reference were 0.5 in every case; all four
  candidate projections matched exactly, with a verified nonzero adapter effect.
- For `[101,201]` the baseline projected 65 rows for 41 valid rows; the candidate
  projected 41. For `[1,100]` the baseline projected 26 for 14, versus 14 after
  the fix. Full encoder outputs have the expected valid-row counts.

Evidence: `validate_omni_audio.py`, `omni-audio-validation.json`, and
`omni-audio-validation.log`. The first two attempts exposed incorrect input
length and global dtype settings in the standalone audit and are not counted
as successful model validation. The final run restores the runtime default dtype
after constructing the BF16 tower, matching the model loader's dtype scoping.
No 30B language-model, vision, or talker evaluation was performed.

## Checks

- `pre-commit run --files vllm/model_executor/models/qwen3_omni_moe_thinker.py tests/lora/test_qwen3_asr.py docs/features/lora.md .buildkite/test_areas/lora.yaml`: passed all applicable hooks, including Ruff, markdownlint, test tethering and mypy.
- `git diff --check`: passed.
- Repository `check_logprobs_close` comparisons on paired pretrained outputs for batches 1 and 8: passed.
- `VLLM_GPU_SYNC_CHECK=error`: enabled for both real LLM regression runs.
- No throughput improvement is claimed or benchmarked. Measured encoder floating-point differences and the raw punctuation difference are disclosed above.

## Scope and review

The PR changes the shared Qwen3 audio encoder used by Qwen3-ASR and Qwen3-Omni audio, the LoRA test Buildkite tether, the LoRA feature page and feature-index note, and one real-LLM regression test. Independent documentation discovery identified an existing feature-index statement that LoRA applied only to language backbones; the PR corrects that contradiction and distinguishes adding model support from enabling existing support. It does not modify Qwen2.5-Omni, QKV mappings, audio chunk construction, convolution weights, or attention segmentation.

AI assistance was used for implementation and local validation. The submitting user reviewed the change and reported no problems on 2026-10-23. Test commands were executed on the user's machine with Codex assistance; this report does not claim the user personally re-ran them.
