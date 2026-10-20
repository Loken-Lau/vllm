Thanks, I added a dataset E2E evaluation for both models at request batch sizes **1, 2, 4, 8, 16**. The source patch is unchanged.

I used a fixed **128-utterance LibriSpeech test-clean subset** spanning 39 speakers and 1.81–29.14s, selected by duration strata with a fixed shuffle seed. Both variants use the same clips, weights and decode settings, alternating order within each batch. BF16/TP1/V2 runner/eager, greedy, max 256 tokens, caches disabled, GPU sync check enabled. WER uses the Whisper English normalization from vLLM's ASR correctness tests. None of the outputs hit the token cap.

| Request batch | Qwen baseline WER | Qwen compact WER | AF3 baseline WER | AF3 compact WER |
| --- | --- | --- | --- | --- |
| 1 | 2.692% | 2.692% | 2.351% | 2.351% |
| 2 | 2.730% | 2.730% | 2.351% | 2.351% |
| 4 | 2.655% | 2.655% | 2.351% | 2.351% |
| 8 | 2.655% | 2.617% | 2.351% | 2.313% |
| 16 | 2.730% | 2.730% | 2.313% | 2.313% |

Across the 2,560 primary transcriptions, **1,276/1,280 paired text/token sequences match**. All pairs match at request batches 1/2/4/16. At batch 8, Qwen differs on 3 utterances (two punctuation changes and one name), and AF3 on 1 (one leading word). Repeating unchanged variants reproduces output variation for every affected utterance. Some large-batch logprobs also differ when generated tokens match. The one-error WER differences are not claimed as quality improvements.

The native scheduler's actual encoder batches ranged from 1 to 11, so I also compared **both pretrained audio encoders directly** on the same 128 utterances at actual encoder batches 1/2/4/8/16: outputs are bitwise identical, maximum absolute difference **0**, with zero difference in dense-mask repeat controls too.

[Reproduction commands, sample IDs/references, per-utterance predictions/token IDs/top-5 logprobs, timing/allocation data, and same-variant diagnostics](https://github.com/Loken-Lau/vllm/blob/validation/audio-mask-librispeech-20261020/LIBRISPEECH-EVALUATION.md). Preliminary runs and the first exact-output assertion failure are retained as well. The PR description now includes these results and the 128-sample scope. No end-to-end speedup is claimed.
