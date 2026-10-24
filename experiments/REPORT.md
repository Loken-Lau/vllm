# Harmony Responses logprobs validation — 2026-10-24

This report covers a local, uncommitted patch on vLLM `e2b672312328d3f95980f5b82869f7937194fd62`.
No commit, push, or PR is performed by these experiments.

## Scope

1. Run the same regression through the real HTTP entry point before and after the change.
2. Compare sampled probabilities, top-k candidates, text, and token bytes against the same real engine generation.
3. Exercise the complete parser → context → streaming/full generator path across Unicode, channels, tool arguments, chunk sizes, and length truncation.
4. Run actual function-call round trips and the complete nearby Harmony integration module.
5. Check non-Harmony Responses behavior with Qwen3-1.7B.
6. Validate H200 automatic backends with compilation/CUDA graphs enabled.
7. Measure repeated HTTP serving performance with the feature disabled and enabled.

## Completed evidence

- `harmony-full-final.log`: complete integration module after the fix, **47 passed, 2 existing skipped, 1 existing xpassed** in 330.60s. The skips are the already-disabled web-search and streaming code-interpreter tests. The XPASS is the existing non-strict pirate-personality check; no skip/xfail marker was added or removed.
- `h200-api-final.json` / `h200-api-final.log`: **18 real HTTP requests pass** on H200, with automatic FlashAttention/FlashInfer backends, torch compilation, and CUDA graphs enabled. Coverage: stream/non-stream Unicode, reasoning truncation, function call and tool-result round trip, 8 mixed concurrent requests, and 2 no-logprobs controls. Tools return a fixed 17°C result to validate the protocol, not external weather data. `h200-default-validated-server.log` is the successful server startup/request log.
- `regression-final.log`: **260 passed, 1 existing xfailed**, 26 warnings after the Unicode-tail fix, across the three complete unit/parser files.
- `utf8-base-before.log` / `utf8-flush-after.log`: the same **12** parser cases fail on the unchanged base, and pass after the fix (3 channels × 2 chunk sizes × implicit/explicit message end).
- `engine-pipeline-final.json`: **30** complete-generator cases pass: 6 replays of 3 actual A800 generations; 12 multi-message/chunk/channel cases; 12 incomplete-UTF8 cases with either a length limit or explicit return. Every sampled score, top-k entry, token string, and raw token byte is checked against the same engine output. The three real prompts contain **21 visible generated tokens** in total; this is a focused mapping test, not a broad model-quality evaluation.
- `qwen-basic-pytest.log`: the complete non-Harmony `test_basic.py` module passes, **6 passed** (real Qwen3-1.7B on A100), including existing logprobs and streaming tests. Snapshot: `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
- `no-logprobs-base.json` / `no-logprobs-fixed-final.json`: **9** frozen real-output conversion cases (3 generations × 3 chunk sizes) are exactly identical after removing random response IDs. Both files have SHA-256 `a7324eb70db8ff84b9bd9bae58b5ff68cd54e13cab2b8417ae361ec02ea27551`.
- `before-shared-fixture.log`: the final HTTP regression fails with the original **400** on the unchanged base; `same-test-proof.json` confirms the regression function and shared server fixture have identical ASTs on base and patched trees.
- `pre-commit-final.log`: all applicable hooks pass for all nine edited files.
- `harmony-full-pytest.log`: initial complete integration run, **47 passed, 2 existing skipped, 1 existing xpassed**. A final run after the Unicode-tail fix is recorded separately.
- `unit-regression.log`: **248 passed, 1 existing xfailed**, 26 warnings. Full Responses serving/unit conversion and Harmony parser files.
- `before-pytest.log`: the new HTTP regression fails on the unchanged base with the original **400: logprobs are not supported with gpt-oss models**. An initial launcher setup error was corrected before recording this result; it is not counted as bug reproduction.
- `engine-pipeline.json`: 3 real A800 generations, replayed through full and streaming response generators, plus 12 deterministic boundary cases (3 chunk sizes × 2 end conditions × 2 response modes). Raw engine token IDs and probabilities are in `engine-pipeline.raw-*.json`. This checks exact copying/mapping of engine probabilities, not independent recomputation of model logits.
- `after-pytest.log` is an intermediate run: **8 passed, 2 failed**. The two assertions incorrectly required nonempty scores/done events for an empty text part when the request had stopped during reasoning. The test now explicitly validates empty text/no scores/no text events. This intermediate result is retained, not counted as a final pass.

## Performance results

All planned experiments are complete. `benchmark-summary.json` and
`benchmark-summary.md` contain the final timing data, calculated by the retained
`summarize_benchmarks.py` script. Exact request records and server commands are
in `bench-final-*.json` and `bench-final-*.command.json`.

Configuration: A100 80GB, GPT-OSS 20B, eager, Marlin/Triton attention, fixed
2048 context / 4 sequences / 45% memory budget, prefix caching disabled.
Each request generates **128 engine output tokens**, including reasoning;
throughput is not computed from visible text alone. Each concurrency is warmed
before measurement, then measured 3 times with 8 requests per repeat.
The table covers **144 timed requests** plus separate warmups. Local integration
and compilation had finished before timing began. The enabled mode requests
`top_logprobs=3`.

| Mode | Concurrency | Median tok/s (repeat range) | Median latency | P95 latency | Median TTFT |
|---|---:|---:|---:|---:|---:|
| Base, feature off | 1 | 36.85 (36.81–37.36) | 3.471s | 3.489s | 0.141s |
| Base, feature off | 4 | 148.65 (148.52–148.93) | 3.440s | 3.451s | 0.164s |
| Patched, feature off | 1 | 37.80 (37.78–37.85) | 3.384s | 3.408s | 0.137s |
| Patched, feature off | 4 | 149.54 (149.03–149.91) | 3.425s | 3.440s | 0.162s |
| Patched, feature on | 1 | 37.09 (36.99–37.14) | 3.450s | 3.479s | 0.140s |
| Patched, feature on | 4 | 144.75 (143.42–144.93) | 3.521s | 3.604s | 0.164s |

Interpretation: no feature-disabled regression was observed on this workload.
The small positive differences are not evidence of an optimization. Compared
with the patched feature-disabled mode, requesting probabilities reduces median
throughput by **1.87% / 3.21%** at concurrency 1 / 4 and increases median latency
by **1.94% / 2.81%**. This is a small, sequentially ordered experiment without
confidence intervals or an extensive workload/hardware sweep.

All 24 measured texts at each concurrency match the corresponding base texts
exactly in both patched modes (`benchmark-output-parity.json`). Median
serialized event volume increases from approximately 25.9KB to 165.9–169.3KB
when probabilities are included. Driver-reported GPU use at the post-measurement
snapshot is **36,943 MiB in all three modes** under the fixed KV reservation;
this is not a peak-memory measurement or evidence of equal host-memory cost.
Both local GPUs were released after benchmarking (`local-cleanup.txt`).

## Failure found and corrected

When generation stops inside the multi-token encoding of `🌍`, Harmony's completed message contains `Hello �`, but the underlying parser does not emit `�` through `last_content_delta`. The previous streaming path therefore returned `Hello ` in delta events while the final response contained `Hello �`; the remaining token probability was also absent from delta events. This occurred at both implicit EOS and explicit message termination.

The parser now tracks the emitted character count for the active message and emits any remaining completed-message tail before the completion segment. The retained regression covers final, analysis, and commentary channels, without mocks or altered tolerances. For incomplete UTF-8, raw token bytes are intentionally preserved; decoding their concatenation with replacement reconstructs the final text.

## Setup attempts and measurement limits

- H200 automatic selection is `FLASH_ATTN` plus `FLASHINFER_CUTLASS_MXFP4_BF16`. Its first startup compiles FlashInfer kernels, compiles the model, captures CUDA graphs, and autotunes kernels; these setup costs are not request-latency measurements.
- A restart overlapped two of this task's H200 instances, and releasing the first instance's memory during profiling triggered vLLM's explicit memory-profiling assertion. Both task instances were cleaned up before a fresh single-instance launch. Existing unrelated GPU jobs were not stopped.
- The first benchmark was a preliminary run (`bench-base-off.json`), with a cold parallel batch causing high variability. Final benchmarks prime every measured concurrency before recording three repeats; the preliminary result is retained and excluded from final comparison.
- Qwen download initially failed through Xet authorization; downloading over the regular mirror transport succeeded. This is an environment/download issue, not a model test failure.
- Final artifact verification caught two missing list type annotations in the new parser regression (`pre-commit-before-annotations.log`). The annotations were added, the same base/patched parser regression was rerun, and all hooks were rerun. This last edit changes only test annotations; implementation hashes and all inference/benchmark code are unchanged.

## Hardware and environment

- Local: A100 80GB (GPU 0), A800 80GB (GPU 1), NVIDIA driver 550.54.15. CUDA 13 compatibility library explicitly enabled. Python 3.12.15; PyTorch 2.14.0+cu130. The source is editable and uses existing precompiled CUDA extensions; this is not a clean source build.
- Remote: H200 NVL, NVIDIA driver 590.48.01, CUDA 13.1; other jobs initially occupied approximately 31GB. Validation uses a separate directory and a 35% GPU-memory budget.
- GPT-OSS model snapshot: `openai/gpt-oss-20b`, `6cee5e81ee83917806bbde320786a8fb61efebee`.
- The runtime's copied `_version.py` shows an older version string; use `source-head.txt` and `working.patch`, not that string, to identify the tested Python implementation.

## Reproduction commands

All local commands run from `/path/to/workspace/pr3/vllm`, with:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export PATH="$PWD/.venv/bin:/path/to/workspace/opensource/.tools/bin:$PATH"
export PYTHONPATH="$PWD"
EVIDENCE=/path/to/workspace/pr3/evidence/validation-20261024
MODEL=/path/to/workspace/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee

CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q \
  tests/entrypoints/openai/responses/test_serving_responses.py \
  tests/entrypoints/openai/responses/test_harmony_utils.py tests/parser/test_harmony.py
CUDA_VISIBLE_DEVICES=1 .venv/bin/python "$EVIDENCE/validate_engine_pipeline.py" \
  --model "$MODEL" --output "$EVIDENCE/engine-pipeline.json"
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest -v -s \
  tests/entrypoints/openai/responses/test_harmony.py
.venv/bin/python "$EVIDENCE/api_workloads.py" \
  --url http://H200_HOST:18863/v1 --output "$EVIDENCE/h200-api-workloads.json"
# Use each of the three modes with the same hardware and arguments:
.venv/bin/python "$EVIDENCE/bench_responses.py" \
  --repo /path/to/workspace/pr3/vllm-base --model "$MODEL" --output "$EVIDENCE/bench-base-off.json"
.venv/bin/python "$EVIDENCE/bench_responses.py" \
  --repo /path/to/workspace/pr3/vllm --model "$MODEL" --output "$EVIDENCE/bench-fixed-off.json"
.venv/bin/python "$EVIDENCE/bench_responses.py" \
  --repo /path/to/workspace/pr3/vllm --model "$MODEL" --enabled --output "$EVIDENCE/bench-fixed-on.json"
# Formal measurements use these same modes, after all local integration ends:
bash "$EVIDENCE/run_formal_benchmarks.sh"
.venv/bin/python "$EVIDENCE/summarize_benchmarks.py"

# The unchanged base checkout has only the new tests copied into it.
cd /path/to/workspace/pr3/vllm-base
export PYTHONPATH="$PWD"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest -x -v -s \
  'tests/entrypoints/openai/responses/test_harmony.py::test_output_logprobs[0-False]'
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q \
  tests/parser/test_harmony.py -k flush_incomplete_utf8
```

Exact benchmark server commands are saved alongside results in `*.command.json`.
The final benchmark files have the `bench-final-` prefix; the earlier unprefixed
base file is a preliminary run.

The H200 server command (run from its separate checkout over SSH) is:

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 \
  .venv/bin/python -m vllm.entrypoints.cli.main serve \
  /path/to/workspace/pr3-harmony-validation/model \
  --served-model-name openai/gpt-oss-20b --host 0.0.0.0 --port 18863 \
  --max-model-len 2048 --max-num-seqs 4 --gpu-memory-utilization 0.35
```

The remote directory is `/path/to/workspace/pr3-harmony-validation/vllm`. Hashes of
all five changed implementation files match the local checkout exactly, as
recorded in `h200-hardware-and-source.txt` and `local-source-sha256.txt`.
The test server was stopped after validation; `h200-cleanup.txt` confirms remote
GPU use returned to the pre-existing 30,992 MiB.

## Limits

These checks do not establish model-accuracy gains, independently verify model logits, cover all hardware/backends, or constitute upstream CI. Full-card H200 results are not MIG-35GB results. Human code review, DCO sign-off, and upstream CI remain separate requirements.
