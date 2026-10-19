"""Summarize all ABBA samples without dropping warm-run outliers."""

import json
import pathlib
import statistics


def main():
    root = pathlib.Path(__file__).resolve().parent
    results = {}
    for model in ["qwen2_audio", "audioflamingo3"]:
        raw = json.loads((root / f"{model}-abba.json").read_text())
        assert raw["order"] == ["baseline", "compact", "compact", "baseline"]
        assert raw["all_text_token_ids_and_top5_logprobs_exactly_equal"]
        cases = {}
        for name in ["single", "mixed"]:
            case = {}
            reference = raw["rounds"][0]["cases"][name]["outputs"][0]
            for variant in ["baseline", "compact"]:
                rounds = [r for r in raw["rounds"] if r["variant"] == variant]
                values = [t for r in rounds for t in r["cases"][name]["batch_ms"]]
                assert len(values) == 40
                for r in rounds:
                    assert all(x == reference for x in r["cases"][name]["outputs"])
                case[variant] = dict(
                    samples=len(values),
                    median_ms=statistics.median(values),
                    population_stddev_ms=statistics.pstdev(values),
                    min_ms=min(values),
                    max_ms=max(values),
                    round_medians_ms=[r["cases"][name]["median_ms"] for r in rounds],
                )
            case["median_change_percent"] = (
                case["compact"]["median_ms"] / case["baseline"]["median_ms"] - 1
            ) * 100
            cases[name] = case
        results[model] = cases
    (root / "abba-comparison.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
