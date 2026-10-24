# Final PR review

Publication commit: `a7f47029d07f20ecd68ae6dd6b272eb10e2071fc`, parent
`08aa305fb5c56d66566f98f55d51a39b43af936f`.

Fully tested target-equivalent snapshot: `f407e07537269ea74468e0073675bcf0bc044bdb`,
based on vLLM `49d634bd6a3dc7433d8fb464821c95c76db85350`.
The previous experiments used `e2b67231`; the 32 intervening upstream commits
did not modify any file edited by this PR. All five implementation files are
identical to the earlier tested implementation.

The publication branch excludes an unrelated upstream workflow update, avoiding
an additional workflow authorization requirement. All nine edited files are
byte-identical between the publication commit and the fully tested snapshot.
Applying the feature patch cleanly to target `49d634bd` yields the exact fully
tested tree `04c0a2c1084f353bd82476e1fa5d23fd8deac8eb` (`merge-tree-proof.json`).

Additional validation on the final publication parent:

- `compatible-base-unit.log`: **339 passed, 1 existing xfailed**, 26 warnings.
- `compatible-base-http.log`: **10 passed, 40 deselected**, all new real-model
  HTTP cases, in 59.91s.

## Review findings addressed

- Read the project, test, and Harmony parser agent instructions and applied the
  complete `pr-checklist`: design fit, correctness/compatibility, default-path
  cost, test reliability/CI selection, docs, root cause, evidence, and PR contents.
- Reused the existing Harmony server fixture; no second model/server allocation.
- Consolidated duplicate stream assertions into the existing per-item helper.
  It checks text, sampled scores, top-k scores, event item/content IDs, done
  events, and completed responses within the same generation.
- Removed a bit-exact text assertion between two independent inference requests:
  cache/batch-dependent numeric variation can change greedy output. The control
  still requires a completed nonempty response with absent logprobs. Exact
  no-feature output parity is demonstrated by the earlier frozen-token
  comparison (9 cases) and benchmark output comparison (48 requests per mode).
- Decode concatenated bytes with replacement for the explicitly supported case
  of a length limit cutting a UTF-8 sequence. The checks retain every raw byte
  and every probability; no numeric tolerance was relaxed.
- Confirmed `LogprobsProcessor` creates an empty score container when output
  logprobs are requested and returns an empty slice for terminal empty chunks.
- Confirmed message objects remain retained by `HarmonyContext`; per-message
  identity keys cannot be recycled within the conversation.
- Confirmed the feature adds no tensor transfers or GPU synchronizations and
  that no scheduler, model runner, or sampling algorithm is changed.
- Confirmed the existing H200 Responses CI step selects the complete directory
  and its source dependencies include the changed implementation files.

## Current-base validation

- `unit.log`: **339 passed, 1 existing xfailed**, 26 warnings. All eight selected
  complete serving/unit/parser/stream/protocol/error/utils/sampling files.
- `before.log`: the same final HTTP regression fails on the unchanged current
  base with **400: logprobs are not supported with gpt-oss models**.
- `harmony-full.log`: **47 passed, 2 existing skipped, 1 existing non-strict
  xpassed** in 291.00s on A800, including all 10 added HTTP cases.
- `pre-commit.log`: all applicable hooks pass for the nine changed files,
  including Ruff, mypy for Python 3.11, and Buildkite test selection.

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q \
  tests/entrypoints/openai/responses/test_serving_responses.py \
  tests/entrypoints/openai/responses/test_harmony_utils.py \
  tests/parser/test_harmony.py \
  tests/entrypoints/openai/responses/test_streaming_events.py \
  tests/entrypoints/openai/responses/test_protocol.py \
  tests/entrypoints/openai/responses/test_errors.py \
  tests/entrypoints/openai/responses/test_responses_utils.py \
  tests/entrypoints/openai/responses/test_sampling_params.py
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest -v -s \
  tests/entrypoints/openai/responses/test_harmony.py
# On an unchanged 49d634bd checkout, with only the final test file copied in:
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest -x -v -s \
  'tests/entrypoints/openai/responses/test_harmony.py::test_output_logprobs[0-False]'
```

Use the Python 3.12 environment described in `../experiments/REPORT.md`.
`LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat` was needed on the local driver.
The three HTTP/parser tests and all five implementation files retain the earlier
correctness, H200, frozen-output, and performance evidence described there.
This final review changes only tests and synchronizes the upstream base.

## Duplicate checks

The issue remains open with no comments. Open-PR searches for `61629 in:body`,
`gpt-oss Responses logprobs`, and `Harmony Responses logprob` found no matching
implementation. Related #58018 concerns item status and zero-delta lifecycle;
#59387 concerns malformed tool headers. Neither implements output logprobs or
the incomplete-UTF8 tail correction.

## Scope and limits

Earlier hardware validation: 30 full-generator mapping/boundary cases, 18 H200
compiled/CUDA-graph HTTP requests, 6 real Qwen3 Responses tests, 144 timed A100
benchmark requests. The mapping test uses 3 actual generations and 21 visible
tokens, not a model-quality evaluation or independent model-logit recomputation.
The benchmark is a small sequential smoke test on the earlier base; no speedup
claim is made. Full-card H200 validation does not substitute for H200 MIG CI.
Two pre-existing integration skips and a pre-existing non-strict XPASS remain.
Upstream CI will be reported separately after PR creation.

AI assistance was used for implementation and validation. The human reviewer
reported reviewing the change and approved submission before this final audit.
