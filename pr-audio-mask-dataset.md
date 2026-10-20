<!-- markdownlint-disable -->

## Overview

Keep Qwen2-Audio and AudioFlamingo3 additive padding masks at `[B, 1, 1, S]` instead of materializing identical query rows into `[B, 1, S, S]`. Related to #62350; this patch addresses mask allocation only.

## Claims

- Mask storage changes from `B*S*S` to `B*S` elements, preserving additive values, dtype/device and length calculations.
- At BF16 / `S=1500`, actual mask storage per audio chunk falls from **4,500,000 to 3,000 bytes**. This is mask storage, not whole-model VRAM or throughput.
- No WER regression was observed in the paired dataset evaluation below. No accuracy or end-to-end speedup improvement is claimed.

## Validation

[Dataset evaluation: reproduction, predictions, scripts and diagnostics](https://github.com/Loken-Lau/vllm/blob/validation/audio-mask-librispeech-20261020/LIBRISPEECH-EVALUATION.md). [Original regression/microbenchmark/smoke-evaluation artifacts](https://github.com/Loken-Lau/vllm/blob/validation/audio-mask-20261019/REPRODUCE.md).

A800 80GB / SM80, Python 3.12.15, torch 2.14.0+cu130, Transformers 5.19.0. PR commit `f59ef21490198e1d35c6c49c4f19653edeb21f2b`, parent `283d48f4ebb404b00e76264ad8f759d7e9a05dfd`; both affected baseline files are also unchanged at PR base `a608b6e6e05f4b309f77981f010e93c2caa1c677`. Editable Python source uses official nightly native extensions from `852989f580820849113d67d13fd8053e279d0444`.

**Dataset E2E:** fixed 128-utterance LibriSpeech `test-clean` subset, selected by duration strata, shuffled with seed 62350; 39 speakers, 1.81–29.14s, 2,637 normalized reference words. Both pretrained models at request batches **1/2/4/8/16**, alternating baseline/compact order within each batch in the same loaded model. Baseline code is extracted verbatim from the parent; compact is the actual PR code. BF16, TP=1, V2 runner, eager, greedy, max 256 output tokens; prefix/processor caches disabled; `VLLM_GPU_SYNC_CHECK=error`. Prompt requests transcription only. WER uses vLLM ASR tests' Whisper English normalizer and jiwer. No output reached the token cap.

WER (%; baseline/compact use identical clips and decode settings):

| Request batch | Qwen baseline WER | Qwen compact WER | AF3 baseline WER | AF3 compact WER |
| --- | --- | --- | --- | --- |
| 1 | 2.692% | 2.692% | 2.351% | 2.351% |
| 2 | 2.730% | 2.730% | 2.351% | 2.351% |
| 4 | 2.655% | 2.655% | 2.351% | 2.351% |
| 8 | 2.655% | 2.617% | 2.351% | 2.313% |
| 16 | 2.730% | 2.730% | 2.313% | 2.313% |

- **1,276/1,280** paired text/token sequences match (2,560 primary transcriptions). Request batches 1/2/4/16 match exactly. At batch 8, Qwen differs on 3/128 clips (two punctuation changes and one name); AF3 differs on 1/128 (one leading word). Same-variant repeat controls reproduce changes for every affected clip. Each model's WER difference is one fewer word error for compact; this is not claimed as an accuracy improvement. Some large-batch logprobs differ even when tokens match. Raw outputs, top-5 logprobs and repeats are retained.
- Native scheduling split submitted groups into actual encoder batches of **1–11**; observed shapes/storage are recorded separately. Additional **pretrained encoder** comparisons on the same 128 clips at actual encoder batches **1/2/4/8/16** are bitwise identical for both models (maximum difference 0; dense-repeat controls also 0).
- Per-batch timings, allocation peaks, preliminary original-prompt runs and an initial exact-equality failure are retained in the dataset artifacts. These are descriptive dataset timings, not repeated speedup trials. This is a test-clean subset, not a full-benchmark or general accuracy guarantee.

**Regression and allocation checks:**

```bash
uv run --no-project --python .venv/bin/python python -m pytest \
  tests/models/multimodal/processing/test_audioflamingo3.py -q

.venv/bin/pre-commit run --files \
  vllm/model_executor/models/qwen2_audio.py \
  vllm/model_executor/models/audioflamingo3.py \
  tests/models/multimodal/processing/test_audioflamingo3.py
```

- **7 tests passed** on each of Transformers 5.16.1 and 5.19.0; the four added cases fail on the original square-mask source. All applicable pre-commit hooks passed, including mypy and CI tethering.
- Supplementary small-encoder matrix: **192 comparisons passed**, CPU FP32 and CUDA FP32/FP16/BF16, default/SDPA/eager, single/mixed odd/even valid lengths; maximum output difference 0. Boundary and active-padding negative controls passed.
- BF16 / `S=1500`, batch 1/4/16: mask storage **4.5/18/72 MB → 3/12/48 KB**; SDPA difference 0. Five independent builder timing medians/IQRs retained; batch 1 did not show a construction speedup.

## Details

Every query row has the same key-padding bias. `.expand(..., S, S).to(dtype)` allocates a square floating-point tensor; retaining a singleton query dimension avoids that allocation. Two call sites change without new branches/configuration/dependencies. Extra encoder backend dispatch in #62350 is outside scope.

Before opening this PR, authenticated issue/comment reads and open-PR searches for `62350 in:body`, `audio mask in:title`, and `AudioFlamingo3 in:title` found no matching implementation.

AI assistance: OpenAI Codex assisted with investigation, implementation, review and validation. The human submitter reviewed and understood every source change, ran the relevant local tests and authorized publication. Dataset evaluation scripts were executed with AI assistance. The source commit includes DCO sign-off and AI attribution. This review follow-up changes validation/description only.

---

<details>
<summary> Pull Request Checklist </summary>

- [x] I used vLLM's `/pr-checklist` skill, including re-review of maintainer feedback.
- [x] AI assistance was used during the creation of this PR.
- [x] **Design Fit:** Minimizes core impact and preserves the existing mask interface.
- [x] **Testing and Validation:** Dataset E2E across request batch sizes, pretrained encoder comparison, focused regressions and recorded CI integration.
- [x] **Code Quality and Style:** Minimal source diff; applicable local pre-commit checks passed.
- [x] **Pull Request Contents:** Claims, configurations, reproduction artifacts and limitations are included.
</details>
