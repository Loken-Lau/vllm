"""Check every measured output and summarize independent-engine A/B runs."""

import json
import pathlib
import statistics


def main():
    root = pathlib.Path(__file__).resolve().parent
    results = {}
    for model in ["qwen2-audio", "audioflamingo3"]:
        baseline = json.loads((root / f"{model}-baseline-v2.json").read_text())
        compact = json.loads((root / f"{model}-compact-v2.json").read_text())
        assert baseline["model_revision"] == compact["model_revision"]
        assert baseline["source_commit"] == compact["source_commit"]
        assert baseline["runtime"] == compact["runtime"]
        assert baseline["audio_seconds"] == compact["audio_seconds"]
        assert baseline["repeats"] == compact["repeats"] == 10
        cases = {}
        for name in baseline["cases"]:
            before = baseline["cases"][name]
            after = compact["cases"][name]
            max_diff = 0.0
            count = 0
            for a, b in zip(
                before["repeated_outputs"], after["repeated_outputs"], strict=True
            ):
                for x, y in zip(a, b, strict=True):
                    assert x["text"] == y["text"]
                    assert x["token_ids"] == y["token_ids"]
                    assert x["prompt_tokens"] == y["prompt_tokens"]
                    for p, q in zip(x["logprobs"], y["logprobs"], strict=True):
                        assert p.keys() == q.keys()
                        max_diff = max(max_diff, *(abs(p[k] - q[k]) for k in p))
                    count += 1
            assert max_diff == 0.0
            timings = {}
            for variant, data in [("baseline", before), ("compact", after)]:
                values = [x * 1000 for x in data["batch_seconds"]]
                assert len(values) == 10
                timings[variant] = {
                    "median_ms": statistics.median(values),
                    "population_stddev_ms": statistics.pstdev(values),
                    "min_ms": min(values),
                    "max_ms": max(values),
                    "samples_ms": values,
                }
            cases[name] = {
                "compared_request_pairs": count,
                "text_and_token_ids_identical": True,
                "max_top5_logprob_abs_diff": max_diff,
                "prompt_tokens": [
                    x["prompt_tokens"] for x in after["repeated_outputs"][0]
                ],
                "output_tokens": [
                    len(x["token_ids"]) for x in after["repeated_outputs"][0]
                ],
                "timings": timings,
            }
        mask_rows = {}
        for variant, data, query_width in [
            ("baseline", baseline, 1500),
            ("compact", compact, 1),
        ]:
            masks = [mask for rank in data["captured_masks"] for mask in rank]
            assert masks
            for mask in masks:
                batch, heads, queries, keys = mask["shape"]
                assert (heads, queries, keys) == (1, query_width, 1500)
                assert mask["dtype"] == "torch.bfloat16"
                assert mask["storage_bytes"] == batch * heads * queries * keys * 2
            encoded_chunks = sum(mask["shape"][0] for mask in masks)
            expected_chunks = 11 * (1 + (4 if model == "audioflamingo3" else 3))
            assert encoded_chunks == expected_chunks
            mask_rows[variant] = {
                "forward_calls": len(masks),
                "encoded_audio_chunks_including_warmups": encoded_chunks,
                "unique_shapes": sorted({tuple(mask["shape"]) for mask in masks}),
                "bytes_per_audio_chunk": query_width * 1500 * 2,
            }
        results[model] = {
            "cases": cases,
            "actual_encoder_runtime": compact["runtime"],
            "captured_masks": mask_rows,
            "audio_seconds": compact["audio_seconds"],
            "limitations": (
                "One speech recording plus repeated long speech; eager BF16 TP1 "
                "on A800, not a WER evaluation or an end-to-end speedup claim."
            ),
        }
    target = root / "e2e-audio-models-comparison.json"
    target.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
