# Dataset E2E evaluation requested in PR #62469

Reviewer request: [DarkLight1337, 2026-10-20](https://github.com/vllm-project/vllm/pull/62469#issuecomment-6282477030).

## Design

Use the official [LibriSpeech ASR test-clean](https://www.openslr.org/12) archive:

```bash
mkdir -p ../.models/librispeech-test-clean
curl -fL --retry 3 \
  https://www.openslr.org/resources/12/test-clean.tar.gz \
  -o ../.models/librispeech-test-clean/test-clean.tar.gz
md5sum ../.models/librispeech-test-clean/test-clean.tar.gz
tar -xzf ../.models/librispeech-test-clean/test-clean.tar.gz \
  -C ../.models/librispeech-test-clean
```

The official checksum is `32fa31d27d2e1cad72775fee3f4849a9` (matched).
SHA-256 is `39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23`.
The corpus has 2,620 test-clean utterances. Sort by duration, choose the midpoint
of each of 128 equally sized strata, then shuffle with seed 62350. The resulting
128 distinct clips cover 39 speakers and durations 1.81–29.14 seconds. No clipping,
repetition, reference-dependent filtering or selection by model output is used.
This is a fixed evaluation subset, not the full test-clean benchmark.

Each model processes the same 128 clips at request batch sizes 1, 2, 4, 8 and 16.
Within each batch, run baseline/compact or compact/baseline alternately. Both
variants share loaded weights and identical decode settings. Only the two-line
mask construction differs: baseline is extracted verbatim from the unmodified
parent source, while compact is the actual PR code. Runtime hooks verify which
mask shape/storage reaches the real encoder. Actual encoder batches may be
smaller than submitted request batches due to continuous batching and scheduling;
both counts are recorded separately.

Baseline source is `283d48f4ebb404b00e76264ad8f759d7e9a05dfd`; GitHub API blob checks
confirm both affected model files are also identical at current PR base
`a608b6e6e05f4b309f77981f010e93c2caa1c677`. Compact source is PR commit
`f59ef21490198e1d35c6c49c4f19653edeb21f2b`. No source change is made for this review.

Configuration: NVIDIA A800 80GB, BF16, TP=1, V2 runner, eager execution, greedy,
max output 256 tokens, max model length 4096, max_num_seqs=16,
max_num_batched_tokens=32768, GPU utilization=0.75. Prefix and multimodal
processor caches are disabled; VLLM_GPU_SYNC_CHECK=error enabled. Existing Python,
PyTorch, Transformers and official native-extension revisions are unchanged and
recorded in the original validation README.

The final prompt explicitly asks for only the English transcription with no
introduction/explanation. WER is corpus-level word edit distance after the same
Whisper English normalization used by vLLM's ASR correctness tests. Spelling map
is `openai/whisper-large-v3` revision
`06f233fe06e710322aca913c1bc4249a0d71fce1`, `normalizer.json` SHA-256
`bf1c507dc8724ca9cf9903640dacfb69dae2f00edee4f21ceba106a7392f26dd`.
`jiwer==3.0.5` calculates WER, independently checked against a dynamic-programming
word edit-distance implementation. Raw text is not stripped of model preambles
or edited after generation. Generated token IDs and top-5 logprobs are retained.

## Reproduce

Copy scripts and artifacts into the original layout documented in `REPRODUCE.md`
and execute from `workspace/vllm/`:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:/usr/local/cuda-13.0/lib64
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_GPU_SYNC_CHECK=error
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
../.tools/bin/uv pip install --python .venv/bin/python jiwer==3.0.5

for model in audioflamingo3 qwen2_audio; do
  canhazgpu run --nonblock --gpu-ids 1 --timeout 30m -- \
    ../.tools/bin/uv run --no-project --python .venv/bin/python python \
    ../.contributions/infra-screen/eval_librispeech_batch_sizes.py \
    --model "$model" --samples 128 \
    --output "../.contributions/infra-screen/librispeech-${model}-128-final.json"
done

../.tools/bin/uv run --no-project --python .venv/bin/python python \
  ../.contributions/infra-screen/summarize_librispeech.py \
  ../.contributions/infra-screen/librispeech-audioflamingo3-128-final.json \
  ../.contributions/infra-screen/librispeech-qwen2_audio-128-final.json \
  --output ../.contributions/infra-screen/librispeech-final-summary.json
```

The same-variant repeat controls on a differing batch are excluded from measured
latencies and the primary WER; they are retained in `mismatches` for diagnosis.
Measured call latency covers native `LLM.generate()` including preprocessing and
scheduling, excluding file decoding (loaded before inference) and audit RPCs.
It is an offline native-vLLM E2E evaluation, not an HTTP serving benchmark.

## Preliminary runs and diagnostic findings

These records are retained even when superseded by the final ASR prompt:

- The 32-clip Qwen pilot validates the five-batch-size flow.
- `librispeech-qwen-128-original-prompt.json` uses the original short instruction,
  which often elicits a preamble. Its normalized WER is around 30%, despite mostly
  correct transcriptions; baseline/compact text and tokens match at all sizes.
- `librispeech-audioflamingo3-128-initial-failure.log` records the initial AF3 run:
  sizes 1/2/4 matched, then an exact-output assertion stopped at size 8. That run
  predates incremental result saving, so unrecorded individual outputs are not
  presented as evidence.
- `librispeech-af3-original-diagnostics.json` reruns sizes 8/16 with the original
  prompt, saving all predictions/logprobs. Size 8 matched; size 16 had three
  differing utterances. Two repeat baseline runs reproduced both sides of all
  three differences, while compact repeat runs were stable. Thus exact output
  differences can occur even without changing the mask; this alone does not
  prove the patch caused a regression or establish its numerical cause.
- Final transcript-only prompting is used to evaluate ASR quality without
  postprocessing away preambles. Initial results are not discarded.

## Dataset attribution

LibriSpeech (c) 2014 Vassil Panayotov, distributed under
[Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).
References and utterance IDs in these artifacts are a selected subset of the
original corpus; predictions, computed statistics and the selection metadata
are added by this evaluation. Audio is not redistributed in this branch.

## Dataset E2E results

Each WER uses the same 2,637 normalized reference words. Paired compact WER is equal or one word error lower; these small differences are not claimed as accuracy improvement.

| Request batch | Qwen baseline WER | Qwen compact WER | AF3 baseline WER | AF3 compact WER |
| --- | --- | --- | --- | --- |
| 1 | 2.692% | 2.692% | 2.351% | 2.351% |
| 2 | 2.730% | 2.730% | 2.351% | 2.351% |
| 4 | 2.655% | 2.655% | 2.351% | 2.351% |
| 8 | 2.655% | 2.617% | 2.351% | 2.313% |
| 16 | 2.730% | 2.730% | 2.313% | 2.313% |

For both models, all 128 paired text/token sequences match at request batches 1, 2, 4 and 16. At request batch 8, Qwen matches 125/128 (two punctuation-only differences and one proper name) and AF3 matches 127/128 (one leading word). Every differing utterance can change during a repeat with an unchanged variant; detailed baseline and compact repeat outputs are retained. Some large-batch top-5 logprobs differ even when generated tokens match. The experiment does not establish that all these decode/logprob differences originate in attention-mask selection.

In total, 1,276/1,280 paired text/token sequences match. These are 2,560 measured native-vLLM transcriptions (two models × two variants × five request batch sizes × 128 clips), excluding warmups and diagnostic repeats. No output reaches the 256-token cap (max Qwen 92, AF3 97).

The original-prompt AF3 diagnostic also shows same-variant variations, including a first exact-output assertion failure; those logs and raw predictions are linked alongside final results.

### Measured batch latency

The input duration mixture and generated sequence lengths vary by batch. Each row includes one dataset pass per variant: 128/64/32/16/8 calls for request batch 1/2/4/8/16. These descriptive timings are not repeated throughput trials or a formal speedup/noninferiority test. RPC diagnostics and extra repeat controls are outside each measured call. No end-to-end speedup is claimed.

| Model / request batch | Baseline mean ± population stddev (ms) | Compact mean ± population stddev (ms) |
| --- | --- | --- |
| audioflamingo3 / 1 | 360.08 ± 219.40 | 358.83 ± 218.21 |
| audioflamingo3 / 2 | 512.93 ± 245.12 | 510.83 ± 244.91 |
| audioflamingo3 / 4 | 730.97 ± 258.14 | 727.42 ± 257.27 |
| audioflamingo3 / 8 | 1083.90 ± 228.65 | 1077.98 ± 225.93 |
| audioflamingo3 / 16 | 1527.15 ± 221.42 | 1520.37 ± 224.51 |
| qwen2_audio / 1 | 375.66 ± 212.55 | 369.01 ± 210.07 |
| qwen2_audio / 2 | 516.86 ± 237.35 | 516.45 ± 238.08 |
| qwen2_audio / 4 | 721.74 ± 236.81 | 717.94 ± 237.93 |
| qwen2_audio / 8 | 1083.14 ± 256.59 | 1054.88 ± 210.78 |
| qwen2_audio / 16 | 1521.41 ± 179.64 | 1509.43 ± 181.41 |

Actual encoder batch sizes in the native runs were 1–11, because scheduling splits submitted request groups. Full observed shapes, tensor storage sizes, and per-call incremental PyTorch allocation peaks are in each raw JSON. Whole-process reserved VRAM is not used as a proxy for mask allocation.

### Direct pretrained encoder comparison

`compare_pretrained_audio_encoders.py` loads only the audio tower's exact checkpoint tensors (strict state loading), constructs independently equal dense and compact additive masks, and compares outputs for all 128 utterances at actual encoder batches 1, 2, 4, 8, 16. It also repeats the dense encoder call as a control. This complements, rather than replaces, the native model E2E dataset evaluation.

Reproduce on an idle GPU:

```bash
for model in audioflamingo3 qwen2_audio; do
  canhazgpu run --nonblock --gpu-ids 1 --timeout 10m -- \
    ../.tools/bin/uv run --no-project --python .venv/bin/python python \
    ../.contributions/infra-screen/compare_pretrained_audio_encoders.py \
    --model "$model" \
    --output "../.contributions/infra-screen/librispeech-${model}-pretrained-encoder.json"
done
```

The full test-clean corpus, other ASR domains, other GPUs, tensor-parallel configurations and compiled encoder execution remain outside these results.

**Direct encoder result:** both pretrained audio towers are bitwise identical for dense versus compact masks for all 128 utterances at all five actual encoder batch sizes (1,280 utterance-level comparisons across the two models). Maximum absolute difference is 0, and the dense-repeat control difference is also 0. Exact checkpoint tensors were loaded with strict state-dict validation.

## Downloaded artifact files

Large raw JSON files are losslessly compressed as `.json.gz` for publication. The larger compressed files are additionally split into ordered `.partNNN` files to accommodate transport issues. Reassemble and restore original names before running the summarizer:

```bash
for first in *.json.gz.part000; do
  [ -f "$first" ] || continue
  combined="${first%.part000}"
  cat "$combined".part* > "$combined"
done
for file in *.json.gz; do gzip -dk "$file"; done
```

`dataset-manifest.json` contains compressed artifact checksums and original uncompressed JSON checksums. No measurements, predictions, repeat controls or logprobs are removed by compression. The original workspace layout and model setup are also included in `REPRODUCE.md`.
