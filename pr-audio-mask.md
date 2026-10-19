<!-- markdownlint-disable -->

## Overview

Qwen2-Audio and AudioFlamingo3 materialize identical padding-mask query rows when converting an expanded boolean view to floating point. Keep the additive masks at `[B, 1, 1, S]` and let attention broadcast them. Related to #62350; this patch addresses mask allocation only.

## Claims

- Mask storage changes from `B*S*S` to `B*S` elements, preserving the additive values, dtype, device and length calculations.
- At BF16 / `S=1500`, actual mask storage per audio chunk falls from **4,500,000 to 3,000 bytes**. This is a mask-storage reduction, not a whole-model VRAM reduction or a throughput claim.
- Both pretrained models retain identical generated text, token IDs and top-5 logprobs in the evaluated workloads.

## Validation

[Reproduction commands, scripts and raw results](https://github.com/Loken-Lau/vllm/blob/validation/audio-mask-20261019/REPRODUCE.md) are provided separately from the source PR.

Local environment: A800 80GB / SM80, Python 3.12.15, torch 2.14.0+cu130, Transformers 5.19.0. Source baseline `283d48f4ebb404b00e76264ad8f759d7e9a05dfd`; editable Python source uses official nightly native extensions from `852989f580820849113d67d13fd8053e279d0444`.

```bash
uv run --no-project --python .venv/bin/python python -m pytest \
  tests/models/multimodal/processing/test_audioflamingo3.py -q

.venv/bin/pre-commit run --files \
  vllm/model_executor/models/qwen2_audio.py \
  vllm/model_executor/models/audioflamingo3.py \
  tests/models/multimodal/processing/test_audioflamingo3.py
```

- **7 pytest cases passed**, also **7 passed on minimum supported Transformers 5.16.1**. The four new parameterized cases fail against the original square-mask source, then pass with this change. They exercise the real model methods and small encoders under SDPA/eager, checking storage and output parity against an independently built dense mask.
- All applicable pre-commit hooks passed, including mypy and CI test tethering. The added cases run in the existing multimodal processing CI suite.
- Supplementary real-encoder matrix: **192 comparisons passed**, covering both Transformers versions, CPU FP32 and CUDA FP32/FP16/BF16, default/SDPA/eager, single and mixed odd/even valid lengths. Maximum encoder output difference was 0. Each version also passed six mask-builder boundary cases and an active-padding negative control.
- A800 BF16 AudioFlamingo3 mask-builder benchmark, `S=1500`, batches 1/4/16: measured storage **4.5/18/72 MB → 3/12/48 KB**; SDPA maximum output difference 0. Five independent `blocked_autorange` medians and IQRs per variant are retained. Construction medians were **92.81/104.13/375.57 μs → 95.06/86.76/87.05 μs**. Batch 1 does not show a construction speedup.

Pretrained native-vLLM / V2-runner A/B: BF16, TP=1, eager execution, greedy generation capped at 32 tokens; prefix and processor caches disabled; `VLLM_GPU_SYNC_CHECK=error` enabled. Each variant ran in a separate engine, with one warmup and ten measured batches per single/mixed case. Hooks confirmed SDPA and the actual square/compact mask shapes and storage.

| Model | Audio durations | Compared request pairs | Text / token IDs | Max top-5 logprob difference |
| --- | --- | --- | --- | --- |
| Qwen2-Audio-7B-Instruct | 2 / 15.948 / 29 seconds | 40 | Identical | 0 |
| AudioFlamingo3 | 2 / 15.948 / 35 seconds | 40 | Identical | 0 |

Pinned model revisions and verified weight hashes are recorded in the artifacts. The 29/35-second inputs repeat the real speech asset; the AF3 case exercises two audio chunks. This is an output-parity smoke evaluation, not dataset-level WER evaluation.

Additional latency control uses the **same loaded model process**, switching only the exact baseline/compact mask code in ABBA order. Each round has five warmups and twenty measurements per case, giving forty samples per variant/case. All measured text, token IDs and top-5 logprobs remain identical. Median batch latency ± population standard deviation (ms):

| Model / case | Baseline | Compact | Median change |
| --- | --- | --- | --- |
| Qwen2-Audio / Single | 216.41 ± 2.88 | 214.53 ± 3.40 | -0.87% |
| Qwen2-Audio / Mixed | 562.61 ± 6.08 | 564.11 ± 3.04 | +0.27% |
| AudioFlamingo3 / Single | 224.46 ± 3.23 | 222.55 ± 2.78 | -0.85% |
| AudioFlamingo3 / Mixed | 599.71 ± 20.52 | 602.21 ± 7.30 | +0.42% |

The ranges overlap and changes are below 1%; no end-to-end speedup is claimed. Earlier separate-engine measurements, including outliers and slower compact results, are retained in the artifacts. Other GPU architectures, TP configurations, quantization and encoder compilation modes were not evaluated.


## Details

Every query row has the same key-padding bias. The current `.expand(..., S, S).to(dtype)` allocates a square floating-point tensor; retaining a singleton query dimension avoids that allocation. The implementation removes the expansion at two call sites without adding branches, dependencies or configuration. Extra encoder backend dispatch proposed in #62350 is outside this patch.

Duplicate-work checks on 2026-10-19: the issue and all comments were read, and open-PR searches for `62350 in:body`, `audio mask in:title`, and `AudioFlamingo3 in:title` returned no matching implementation. Authenticated REST reads were used because this host's gh 2.4.0 `issue view --comments` hits a retired Projects classic GraphQL field.

AI assistance: OpenAI Codex assisted with investigation, implementation, review and validation. The human submitter reviewed and understood every changed line, ran the relevant local test and authorized publication. AI attribution and DCO sign-off are included in the commit.

---

<details>
<summary> Pull Request Checklist </summary>

- [x] I used vLLM's `/pr-checklist` skill.
- [x] AI assistance was used during the creation of this PR.
- [x] **Design Fit:** Minimizes impact on core components, reuses existing functionality, and justifies added complexity.
- [x] **Testing and Validation:** Validates the change and ensures any added tests are meaningful and reliable, with CI coverage or documented CI resource constraints and validation performed outside CI.
- [x] **Code Quality and Style:** Keeps code and comments clear and concise, and updates relevant documentation and examples.
- [x] **Pull Request Contents:** Includes a brief summary and relevant links, supports claims with evidence, explains root causes and implementation trade-offs, and follows the contributing guide.
</details>
