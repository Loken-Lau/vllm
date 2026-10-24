import json
import statistics
from pathlib import Path

root = Path(__file__).parent
summary = []
for name in ("base-off", "fixed-off", "fixed-on"):
    path = root / f"bench-final-{name}.json"
    data = json.loads(path.read_text())
    for concurrency in (1, 4):
        runs = [r for r in data["runs"] if r["concurrency"] == concurrency]
        assert len(runs) == 3
        rows = [row for run in runs for row in run["rows"]]
        assert len(rows) == 24
        assert all(row["output_tokens"] == 128 for row in rows)
        tps = [r["tokens_per_second"] for r in runs]
        latency = sorted(row["latency_s"] for row in rows)
        row = {
            "mode": name, "concurrency": concurrency, "repeats": 3, "requests": 24,
            "tps_median": statistics.median(tps), "tps_min": min(tps), "tps_max": max(tps),
            "latency_median_s": statistics.median(latency),
            "latency_p95_s": latency[round((len(latency)-1)*.95)],
            "ttft_median_s": statistics.median(r["ttft_s"] for r in rows),
            "event_bytes_median": statistics.median(r["serialized_event_bytes"] for r in rows),
            "visible_logprob_tokens_median": statistics.median(r["logprob_tokens"] for r in rows),
        }
        if name != "base-off":
            base = next(r for r in summary if r["mode"] == "base-off" and r["concurrency"] == concurrency)
            row["tps_change_vs_base_pct"] = (row["tps_median"] / base["tps_median"] - 1)*100
        summary.append(row)
        print(json.dumps(row))
(root / "benchmark-summary.json").write_text(json.dumps(summary, indent=2))
lines = ["| Mode | Concurrency | Median tok/s (range) | Median latency | P95 latency | Median TTFT |",
         "|---|---:|---:|---:|---:|---:|"]
for row in summary:
    lines.append(f"| {row['mode']} | {row['concurrency']} | {row['tps_median']:.2f} "
                 f"({row['tps_min']:.2f}–{row['tps_max']:.2f}) | "
                 f"{row['latency_median_s']:.3f}s | {row['latency_p95_s']:.3f}s | {row['ttft_median_s']:.3f}s |")
(root / "benchmark-summary.md").write_text("\n".join(lines)+"\n")
