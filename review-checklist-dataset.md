# PR checklist re-review — dataset evaluation (2026-10-20)

Skill: vLLM `.agents/skills/pr-checklist/SKILL.md`. Read conversation comments,
review submissions and inline comments. The sole actionable maintainer request
is dataset E2E evaluation across more batch sizes; no source revision was requested.

## Design fit

The reviewed upstream three-file patch is unchanged. Source SHA-256 values match
those reviewed by the human submitter and the published PR commit. Additional
one-off evaluation scripts/data live in an independent validation branch.

## Testing and validation

Completed native-vLLM full-model generation on a deterministic 128-utterance
LibriSpeech test-clean subset, 39 speakers, mixed durations 1.81–29.14s, request
batches 1/2/4/8/16, both pretrained models and both mask variants. Total 2,560
primary transcriptions. Corpus WER calculated with vLLM ASR tests' Whisper
normalizer and jiwer, independently cross-checked by word edit-distance DP.
None of the final generations hit the 256-token cap. Both existing caches disabled,
GPU sync check enabled. Runtime masks/storage and actual encoder batch sizes
captured; request batch size is explicitly distinguished from encoder batch size.

A first AF3 exact-equality failure and original-prompt runs are retained. The
original prompt elicited preambles, so final quality runs explicitly request only
transcription without removing preambles from results afterwards. All primary
outputs and mismatch repeats are saved. At request batch 8, Qwen has 3 differing
text/token sequences and AF3 has 1; same-variant controls reproduce changes for
every affected utterance. WER differs by one error in each model, favoring compact;
no accuracy improvement or general noninferiority claim is made. Large-batch
logprobs can differ even when generated tokens match, and this is disclosed.

Standalone strict-checkpoint pretrained encoder comparisons on the same 128 clips
at actual encoder batches 1/2/4/8/16 are bitwise identical for both models; dense
repeat controls also have zero difference. This supports local mask correctness
without pretending full autoregressive runs are bitwise deterministic.

No new source edit or source test was added, so the previously passing 7-case
regression suite in each supported dependency environment and all applicable
pre-commit checks remain the source validation. Ruff checks for the new standalone
scripts passed. Dataset checksum, spelling-map revision/hash, model revisions,
source commits, raw outputs/logs and commands are recorded. GPU1 reservation was
released and existing services were left alone.

## Code quality/style

No new dependencies in the upstream PR. jiwer3.0.5 installed only in the local
existing venv through uv, alongside rapidfuzz; Python/model runtime versions remain
unchanged. Supplemental scripts are not added to upstream CI. No kernel change,
new backend support, compiled execution, TP or other hardware coverage is claimed.

## PR contents and feedback closure

Prepared a concise updated PR description retaining the upstream template,
checksum-backed public artifacts and a direct response to DarkLight1337 with
WER at every batch size and mismatch diagnostics. The 128-sample limit, observed
encoder batching, and absence of a throughput claim are explicit. AI assistance
is disclosed. User explicitly authorized updating the PR and replying after
experiments complete. No new DCO/source commit is necessary; artifact commits
include sign-off and AI attribution. Existing CI admission policy is unchanged;
no AI request for verified/ready labels is made.
