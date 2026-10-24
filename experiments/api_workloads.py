"""Validate the same new integration assertions against a configured server."""
import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
from openai import AsyncOpenAI

from tests.entrypoints.openai.responses.test_harmony import (
    GET_WEATHER_SCHEMA,
    _assert_text_logprobs,
    _collect_logprob_response,
)


async def main(args):
    report = []
    async with httpx.AsyncClient(timeout=2) as health:
        deadline = time.monotonic()+1800
        while time.monotonic() < deadline:
            try:
                if (await health.get(args.url.replace("/v1", "/health"))).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            await asyncio.sleep(.5)
        else:
            raise TimeoutError("H200 server not ready")
    async with AsyncOpenAI(base_url=args.url, api_key="test", timeout=180, max_retries=0) as client:
        async def save(label, stream, **kwargs):
            response = await _collect_logprob_response(client, stream=stream, **kwargs)
            _assert_text_logprobs(response)
            row = {"case": label, "stream": stream, "status": response.status,
                   "text": response.output_text, "output": response.model_dump(mode="json")}
            report.append(row)
            print(json.dumps({k:v for k,v in row.items() if k != "output"}, ensure_ascii=False), flush=True)
            Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2))
            return response

        for stream in (False, True):
            response = await save("unicode", stream, input="Repeat exactly: 你好，世界！🌍🫠", max_output_tokens=256)
            assert "你好" in response.output_text and "🌍" in response.output_text
            response = await save("reasoning_truncated", stream,
                                  input="Write a detailed 1000 word story about a cat.", max_output_tokens=16)
            assert response.status == "incomplete"
            history = [{"role": "user", "content": "Get the weather at 48.85, 2.35."}]
            call = await save("tool_call", stream, input=history, tools=[GET_WEATHER_SCHEMA],
                              tool_choice={"type": "function", "name": "get_weather"}, max_output_tokens=256)
            calls = [item for item in call.output if item.type == "function_call"]
            assert len(calls) == 1
            assert set(json.loads(calls[0].arguments)) == {"latitude", "longitude"}
            history.extend(item.model_dump() for item in call.output)
            history.append({"type": "function_call_output", "call_id": calls[0].call_id,
                            "output": '{"temperature_celsius":17}'})
            response = await save("tool_answer", stream, input=history, tools=[GET_WEATHER_SCHEMA],
                                  tool_choice="none", max_output_tokens=256)
            assert "17" in response.output_text
        await asyncio.gather(*(save(f"concurrent-{i}", bool(i % 2),
            input=f"Give the product of {i+7} and 9 in a short sentence.", max_output_tokens=256)
            for i in range(8)))
        for stream in (False, True):
            result = await client.responses.create(
                model="openai/gpt-oss-20b", input="Say hello in Chinese.",
                reasoning={"effort": "low"}, temperature=0, max_output_tokens=256, stream=stream)
            if stream:
                events = [event async for event in result]
                result = events[-1].response
                assert all(not event.logprobs for event in events
                           if event.type in ("response.output_text.delta", "response.output_text.done"))
            assert result.status == "completed" and result.output_text
            assert all(c.logprobs is None for item in result.output if item.type == "message"
                       for c in item.content)
            report.append({"case": "no_logprobs", "stream": stream,
                           "status": result.status, "text": result.output_text})
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"PASS: {len(report)} requests", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--output", required=True)
    asyncio.run(main(p.parse_args()))
