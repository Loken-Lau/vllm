"""Independently recompute WER and paired agreement from dataset predictions."""

import argparse
import hashlib
import json
import math
import pathlib
import statistics

from jiwer import process_words
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    mapping = pathlib.Path(__file__).with_name("whisper-normalizer.json")
    normalize = EnglishTextNormalizer(json.loads(mapping.read_text()))
    results = []
    for file in args.files:
        raw = json.loads(file.read_text())
        refs = [normalize(row["reference"]).strip() for row in raw["samples"]]
        assert (
            raw["normalizer_sha256"] == hashlib.sha256(mapping.read_bytes()).hexdigest()
        )
        summary = {
            "file": file.name,
            "model": raw["model"],
            "samples": len(refs),
            "prompt_style": raw["prompt_style"],
            "speakers": len({r["id"].split("-")[0] for r in raw["samples"]}),
            "duration_min_seconds": min(r["duration_seconds"] for r in raw["samples"]),
            "duration_max_seconds": max(r["duration_seconds"] for r in raw["samples"]),
            "batch_sizes": {},
        }
        for size, row in raw["batch_results"].items():
            variants = row["variants"]
            baseline = variants["baseline"]["predictions"]
            compact = variants["compact"]["predictions"]
            assert len(baseline) == len(compact) == len(refs)
            equal_text = sum(
                a["text"] == b["text"] for a, b in zip(baseline, compact, strict=True)
            )
            equal_tokens = sum(
                a["token_ids"] == b["token_ids"]
                for a, b in zip(baseline, compact, strict=True)
            )
            item = {
                "equal_text_count": equal_text,
                "equal_token_sequence_count": equal_tokens,
                "variants": {},
                "mismatches": [],
            }
            max_lp_diff = 0.0
            lp_compared = 0
            topk_set_differences = 0
            for index, (a, b) in enumerate(zip(baseline, compact, strict=True)):
                same_prefix = 0
                for tok_a, tok_b, lp_a, lp_b in zip(
                    a["token_ids"], b["token_ids"], a["logprobs"], b["logprobs"]
                ):
                    common = set(lp_a) & set(lp_b)
                    if set(lp_a) != set(lp_b):
                        topk_set_differences += 1
                    for token in common:
                        max_lp_diff = max(max_lp_diff, abs(lp_a[token] - lp_b[token]))
                        lp_compared += 1
                    if tok_a != tok_b:
                        break
                    same_prefix += 1
                if a["text"] != b["text"] or a["token_ids"] != b["token_ids"]:
                    item["mismatches"].append(
                        {
                            "id": raw["samples"][index]["id"],
                            "first_differing_token_index": same_prefix,
                            "baseline_text": a["text"],
                            "compact_text": b["text"],
                        }
                    )
            item["shared_prefix_max_top5_logprob_diff"] = max_lp_diff
            item["shared_prefix_compared_logprobs"] = lp_compared
            item["shared_prefix_topk_set_differences"] = topk_set_differences
            for variant, data in variants.items():
                predictions = data["predictions"]
                calculated = process_words(
                    refs, [normalize(r["text"]).strip() for r in predictions]
                )
                assert math.isclose(calculated.wer, data["wer"], abs_tol=1e-15)
                assert (
                    calculated.substitutions
                    + calculated.deletions
                    + calculated.insertions
                ) == data["word_errors"]
                assert data["mask_capture"]["audio_items"] == len(refs)
                query_width = 1500 if variant == "baseline" else 1
                for shape in data["mask_capture"]["shapes"]:
                    assert json.loads(shape)[1:] == [1, query_width, 1500]
                values = data["batch_latency_ms"]
                item["variants"][variant] = {
                    "wer_percent": calculated.wer * 100,
                    "word_errors": data["word_errors"],
                    "reference_words": data["reference_words"],
                    "batch_calls": len(values),
                    "batch_mean_ms": statistics.mean(values),
                    "batch_median_ms": statistics.median(values),
                    "batch_population_stddev_ms": statistics.pstdev(values),
                    "samples_per_second": data["samples_per_second"],
                    "cap_reached": data["max_token_cap_reached"],
                    "max_output_tokens": data["max_output_tokens"],
                    "max_incremental_peak_bytes": max(
                        rank["incremental_peak_bytes"]
                        for batch in data["batch_memory_peaks"]
                        for rank in batch
                    ),
                    "observed_encoder_batch_sizes": sorted(
                        {
                            json.loads(shape)[0]
                            for shape in data["mask_capture"]["shapes"]
                        }
                    ),
                }
            item["same_variant_repeat_diagnostics"] = row["mismatches"]
            summary["batch_sizes"][size] = item
        results.append(summary)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(
        json.dumps(
            [
                {
                    **row,
                    "batch_sizes": {
                        key: {
                            k: v
                            for k, v in data.items()
                            if k != "same_variant_repeat_diagnostics"
                        }
                        for key, data in row["batch_sizes"].items()
                    },
                }
                for row in results
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
