# PR Checklist Review

## 1. Design fit

- Scope: one shared model-specific audio encoder, its LoRA regression, the existing LoRA Buildkite test step, and the existing LoRA feature page and feature-index note. No scheduler, runner, or global LoRA mapping changes.
- Reuse: the existing `padded_mask_after_cnn` supplies both valid chunk and position indices. Convolution, existing mask creation, transformer input order, and attention chunk boundaries remain unchanged.
- Compatibility: Qwen3-ASR and Qwen3-Omni audio share `Qwen3OmniMoeAudioEncoder`. A real Qwen3-ASR LLM/LoRA test and A100 encoder comparisons passed. Additional full pretrained Qwen3-Omni audio-tower validation loaded all 525 checkpoint tensors and passed four actual Punica LoRA projection comparisons exactly. Nine unadapted cases had maximum absolute encoder difference 0.001984. Complete Omni language-model inference was not performed; this limitation is disclosed.
- Tradeoff: valid-row gather now precedes a pointwise linear and the old positional embedding slice is replaced by explicit per-row positions. Boundary tests show exact equality for 13 shapes in fp32 and bf16. No latency or throughput improvement is claimed. The previous masked gather and new mask-derived indexing are both inside the existing `gpu_sync_allowed()` region; the serving reproducer also ran with `VLLM_GPU_SYNC_CHECK=error`.
- Duplication: current open #62494 edits the same encoder for quantization and host-sync work, but its current diff still projects padded rows. #50793 changes QKV adapter mapping. This change addresses neither PR's behavior and the PR body explains the relationship.

## 2. Testing and validation

- Test contract: a real `Qwen3-ASR-0.6B` LLM has a LoRA-wrapped `conv_out`; projection receives one row per valid post-CNN position, and a deterministic sparse adapter must match the independent base-plus-LoRA result exactly.
- Baseline evidence: final formatted patch reversed, same test run, failure `26 projected rows vs 14 valid rows`; candidate restored, same command, `1 passed`. Logs are `pytest-e2e-baseline-final.log` and `pytest-e2e-final-restored.log`.
- Encoder reference: 13 length/batch cases × fp32/bf16, 26 exact A100 comparisons against the upstream forward.
- Model evaluation: pinned Qwen3-ASR-0.6B on 64 duration-stratified LibriSpeech test-clean samples at batch 1 and 8. Independent WER recomputation confirms 32/1,284 errors (2.4922%) in each run. 127/128 raw output pairs are exact; normalized text is equal for 128/128. Encoder floating-point differences and one raw punctuation difference are reported. No throughput gain is claimed.
- Determinism: fixed seeds and pinned checkpoint, greedy decoding, sparse rank-8 adapter; exact comparison only for deliberately exact sparse arithmetic and encoder shape tests.
- CI integration: `tests/lora/test_qwen3_asr.py` is selected by the existing LoRA test command. NVIDIA and AMD LoRA steps now declare the modified model file as a source dependency. The `check-test-tethering` hook passed.
- Human review: the user reviewed the current change and stated no problems on 2026-10-23. Local test commands were executed on the submitter's machine with Codex assistance; the report does not claim they were personally typed by the user.

## 3. Code quality

- Ruff check, Ruff format, markdownlint, SPDX, mypy, test tethering and all other applicable pre-commit hooks passed on the changed files.
- `git diff --check` passed.
- The new test adds no mocks; it constructs the production LLM and adapter through existing interfaces. Its helper functions are used by the real test hook and keep the assertions focused.
- The LoRA feature page documents the existing opt-in flags and the corrected partial-chunk behavior. A fresh documentation-discovery agent identified a contradictory backbone-only statement in the feature index and ambiguous enabling-vs-implementing wording; both have been corrected, and the documentation commit's pre-commit checks passed. No new CLI or configuration surface is introduced. Web indexing is unavailable for the unmerged branch; GitHub/local discovery are checked separately.

## 4. PR contents

- Description uses the repository template sections and links the Qwen3 cases in #61608 without closing the umbrella issue, which also tracks other unfixed models.
- Claims include reproduction, baseline and candidate results, model-evaluation outcomes, raw-text/numerical differences, and the absence of a throughput claim.
- Root cause and implementation details explain why valid rows must be gathered before the pointwise LoRA-wrapped projection and how positional indices are preserved.
- AI assistance is disclosed in the description and commit attribution. User review is recorded above; the implementation commit includes the required DCO sign-off under the authorized contribution workflow.
