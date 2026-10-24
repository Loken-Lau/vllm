"""Small repeated HTTP serving benchmark; not an upstream test or speed claim."""
import argparse
import asyncio
import json
import os
import signal
import statistics
import subprocess
import time
from pathlib import Path

import httpx
from openai import AsyncOpenAI


def percentile(values, q):
    values = sorted(values)
    return values[round((len(values)-1)*q)]


async def measure(url, enabled, repeats=3, count=8):
    report = []
    async with AsyncOpenAI(base_url=url, api_key="test", max_retries=0, timeout=180) as client:
        async def request(i):
            start = time.perf_counter()
            first = None
            stream = await client.responses.create(
                model="openai/gpt-oss-20b",
                input="Write a detailed numbered list of 100 useful facts about astronomy.",
                reasoning={"effort": "low"}, temperature=0, max_output_tokens=128,
                stream=True, **({"include": ["message.output_text.logprobs"],
                                 "top_logprobs": 3} if enabled else {}),
            )
            response_bytes = 0
            async for event in stream:
                response_bytes += len(event.model_dump_json().encode())
                if first is None and event.type in (
                    "response.output_text.delta", "response.reasoning_text.delta"):
                    first = time.perf_counter()
                if event.type in ("response.completed", "response.incomplete"):
                    response = event.response
            end = time.perf_counter()
            assert first is not None
            assert response.usage.output_tokens == 128, response.usage
            text = response.output_text
            visible = [lp for item in response.output if item.type == "message"
                       for c in item.content for lp in (c.logprobs or [])]
            if enabled:
                assert visible and text
            return {"latency_s": end-start, "ttft_s": first-start,
                    "output_tokens": response.usage.output_tokens,
                    "text": text, "logprob_tokens": len(visible),
                    "serialized_event_bytes": response_bytes}

        await request(-1)
        await request(-2)
        for concurrency in (1, 4):
            # Prime every measured batch shape, not only single-request kernels.
            await asyncio.gather(*(request(-10-i) for i in range(concurrency)))
            for repeat in range(repeats):
                semaphore = asyncio.Semaphore(concurrency)
                async def limited(i):
                    async with semaphore:
                        return await request(i)
                start = time.perf_counter()
                rows = await asyncio.gather(*(limited(i) for i in range(count)))
                elapsed = time.perf_counter()-start
                result = {"concurrency": concurrency, "repeat": repeat,
                          "requests": count, "elapsed_s": elapsed,
                          "tokens_per_second": sum(r["output_tokens"] for r in rows)/elapsed,
                          "latency_median_s": statistics.median(r["latency_s"] for r in rows),
                          "latency_p95_s": percentile([r["latency_s"] for r in rows], .95),
                          "ttft_median_s": statistics.median(r["ttft_s"] for r in rows),
                          "rows": rows}
                report.append(result)
                print(json.dumps({k:v for k,v in result.items() if k != "rows"}), flush=True)
    return report


async def main(args):
    process = None
    output = Path(args.output)
    if args.repo:
        command = [str(Path(args.repo)/".venv/bin/python"), "-m", "vllm.entrypoints.cli.main",
                   "serve", args.model, "--served-model-name", "openai/gpt-oss-20b",
                   "--host", "127.0.0.1", "--port", str(args.port), "--enforce-eager",
                   "--max-model-len", "2048", "--max-num-seqs", "4",
                   "--gpu-memory-utilization", "0.45", "--moe-backend", "marlin",
                   "--attention-backend", "TRITON_ATTN", "--no-enable-prefix-caching"]
        output.with_suffix(".command.json").write_text(json.dumps(command, indent=2))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTHONPATH=args.repo)
        process = subprocess.Popen(command, cwd=args.repo, env=env, start_new_session=True,
                                   stdout=output.with_suffix(".server.log").open("w"),
                                   stderr=subprocess.STDOUT)
    try:
        async with httpx.AsyncClient(timeout=2) as health:
            deadline = time.monotonic()+600
            while time.monotonic() < deadline:
                if process and process.poll() is not None:
                    raise RuntimeError("Server exited before readiness")
                try:
                    if (await health.get(f"http://{args.host}:{args.port}/health")).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(.5)
            else:
                raise TimeoutError("Server not ready after 600s")
        report = await measure(f"http://{args.host}:{args.port}/v1", args.enabled)
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu",
                                       "--format=csv"], text=True)
        output.write_text(json.dumps({"repo": args.repo, "logprobs": args.enabled,
                                      "gpu_snapshot": gpu, "runs": report}, ensure_ascii=False, indent=2))
    finally:
        if process:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=40)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo")
    p.add_argument("--model")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18864)
    p.add_argument("--gpu", default="0")
    p.add_argument("--enabled", action="store_true")
    p.add_argument("--output", required=True)
    asyncio.run(main(p.parse_args()))
