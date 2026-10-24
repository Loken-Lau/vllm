"""Reproducible, real-engine logprob and complete streaming-path validation."""

import argparse
import asyncio
import copy
import json
from functools import partial
from pathlib import Path

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.generate.base.protocol import RequestResponseMetadata
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.openai.parser.harmony_utils import get_encoding
from vllm.entrypoints.openai.responses.context import HarmonyContext
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
from vllm.logprobs import Logprob
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.parser.harmony import HarmonyParser
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM


def create_serving(engine, model):
    online = OnlineRenderer(
        engine.model_config, engine.renderer, request_logger=None,
        chat_template=None, chat_template_content_format="auto",
    )
    return OpenAIServingResponses(
        engine, OpenAIServingModels(engine, [BaseModelPath("test", model)]),
        online, request_logger=None, chat_template=None,
        chat_template_content_format="auto",
    )


def visible_groups(ids, scores):
    """Independent header/body scan; does not use HarmonyParser or context."""
    enc = get_encoding()
    header, body, visible = "assistant", False, False
    groups = []
    for tid, score in zip(ids, scores):
        text = enc.decode([tid])
        if enc.is_special_token(tid):
            if text == "<|start|>":
                header, body, visible = "", False, False
            elif text == "<|channel|>":
                header += text
            elif text == "<|message|>":
                body = True
                channel = header.split("<|channel|>")[-1]
                visible = channel in ("final", "commentary") and "to=" not in header
                if visible:
                    groups.append([])
            elif text in ("<|end|>", "<|return|>", "<|call|>"):
                body, visible = False, False
            continue
        if not body:
            header += text
        elif visible:
            groups[-1].append((tid, score))
    return groups


def assert_scores(content, expected):
    enc = get_encoding()
    assert len(content.logprobs) == len(expected)
    assert bytes(b for lp in content.logprobs for b in lp.bytes).decode(errors="replace") == content.text
    for lp, (tid, raw) in zip(content.logprobs, expected):
        assert lp.logprob == raw[tid].logprob
        assert lp.token == (raw[tid].decoded_token if raw[tid].decoded_token is not None else enc.decode([tid]))
        assert lp.bytes == list(enc.decode([tid], errors="surrogateescape").encode(
            "utf-8", errors="surrogateescape"))
        assert len(lp.top_logprobs) == min(3, len(raw))
        for top, (candidate, value) in zip(lp.top_logprobs, raw.items()):
            assert top.logprob == value.logprob
            assert top.token == (value.decoded_token if value.decoded_token is not None else enc.decode([candidate]))
            assert top.bytes == list(enc.decode([candidate], errors="surrogateescape").encode(
                "utf-8", errors="surrogateescape"))


def values(lps):
    return [(lp.token, lp.logprob, [(t.token, t.logprob) for t in lp.top_logprobs])
            for lp in lps]


async def replay(serving, chunks, expected, *, stream):
    tokenizer = serving.renderer.tokenizer
    context = HarmonyContext(
        [], [], frozenset({"weather"}), HarmonyParser(tokenizer),
        logprobs_processor=partial(serving._create_response_logprobs,
                                  tokenizer=tokenizer, top_logprobs=3),
    )
    async def results():
        for output in chunks:
            context.append_output(copy.deepcopy(output))
            yield context

    request = ResponsesRequest(input="test", store=False, stream=stream,
                               include=["message.output_text.logprobs"], top_logprobs=3)
    kwargs = dict(request=request, sampling_params=SamplingParams(max_tokens=512),
                  result_generator=results(), context=context, model_name="test",
                  tokenizer=tokenizer, request_metadata=RequestResponseMetadata(request_id="test"))
    events = []
    if stream:
        events = [event async for event in serving.responses_stream_generator(**kwargs)]
        assert events[-1].type in ("response.completed", "response.incomplete")
        response = events[-1].response
    else:
        response = await serving.responses_full_generator(**kwargs)
    messages = [item for item in response.output if item.type == "message"]
    assert len(messages) == len(expected), (response, expected)
    for msg, group in zip(messages, expected):
        assert len(msg.content) == 1
        content = msg.content[0]
        assert_scores(content, group)
        if stream:
            deltas = [event for event in events if event.type == "response.output_text.delta"
                      and event.item_id == msg.id]
            done = [event for event in events if event.type == "response.output_text.done"
                    and event.item_id == msg.id]
            assert len(done) == 1
            assert "".join(e.delta for e in deltas) == content.text
            assert [v for e in deltas for v in values(e.logprobs)] == values(content.logprobs)
            assert values(done[0].logprobs) == values(content.logprobs)
            item_done = [e for e in events if e.type == "response.output_item.done"
                         and e.item.id == msg.id]
            assert len(item_done) == 1
            assert item_done[0].item.content[0].logprobs == content.logprobs
    if stream:
        assert [e.sequence_number for e in events] == list(range(len(events)))
    return {"status": response.status, "text": "".join(
                c.text for msg in messages for c in msg.content),
            "visible_tokens": sum(len(g) for g in expected), "events": len(events),
            "api_output": [msg.model_dump(mode="json") for msg in messages]}


def chunk_output(ids, scores, size, finish):
    outputs = []
    for start in range(0, len(ids), size):
        last = start + size >= len(ids)
        outputs.append(RequestOutput(
            request_id="test", prompt=None, prompt_token_ids=[10, 11],
            prompt_logprobs=None,
            outputs=[CompletionOutput(index=0, text="", token_ids=ids[start:start+size],
                                      cumulative_logprob=None, logprobs=scores[start:start+size],
                                      finish_reason=finish if last else None)],
            finished=last,
        ))
    return outputs


async def main(args):
    engine_args = AsyncEngineArgs(
        model=args.model, max_model_len=2048, max_num_seqs=4,
        gpu_memory_utilization=0.45, enforce_eager=True,
        attention_backend="TRITON_ATTN", moe_backend="marlin",
        enable_prefix_caching=False, disable_log_stats=True,
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    report = []
    try:
        serving = create_serving(engine, args.model)
        enc = get_encoding()
        for prompt in ("Say hello in Chinese in one short sentence.",
                       "Copy exactly: 你好，世界！🌍🫠", "What is 7 times 9?"):
            prefix = ("<|start|>system<|message|>You are a helpful assistant.\n"
                      "Reasoning: low<|end|><|start|>user<|message|>" + prompt +
                      "<|end|><|start|>assistant")
            sampling = SamplingParams(temperature=0, max_tokens=384, logprobs=3,
                                      output_kind=RequestOutputKind.DELTA,
                                      stop_token_ids=[enc.encode("<|return|>", allowed_special="all")[0]])
            raw = [copy.deepcopy(out) async for out in engine.generate(
                {"prompt_token_ids": enc.encode(prefix, allowed_special="all")},
                sampling, "raw-" + str(len(report)))]
            ids = [t for out in raw for t in out.outputs[0].token_ids]
            scores = [lp for out in raw for lp in out.outputs[0].logprobs]
            expected = visible_groups(ids, scores)
            assert expected and sum(map(len, expected)) > 0
            Path(args.output).with_suffix(f".raw-{len(report)}.json").write_text(json.dumps(
                {"prompt": prompt, "ids": ids,
                 "scores": [{str(t): v.logprob for t,v in row.items()} for row in scores]},
                ensure_ascii=False, indent=2))
            for stream in (False, True):
                result = await replay(serving, raw, expected, stream=stream)
                report.append({"kind": "real_engine", "prompt": prompt, "stream": stream,
                               "all_sampled_topk_bytes_exact": True, **result})
                print(json.dumps({k:v for k,v in report[-1].items() if k != "api_output"},
                                 ensure_ascii=False), flush=True)

        # The same real serving objects drive deterministic multi-message chunks.
        for ending in ("<|return|>", ""):
            text = ("<|channel|>analysis<|message|>Private thought<|end|>"
                    "<|start|>assistant<|channel|>commentary<|message|>Checking 🌍."
                    "<|end|><|start|>assistant to=functions.weather<|channel|>commentary"
                    '<|message|>{"city":"Paris"}<|call|>'
                    "<|start|>assistant<|channel|>final<|message|>你好 𝄞🫠 Sunny." + ending)
            ids = enc.encode(text, allowed_special="all")
            scores = [{t: Logprob(-float(i+1)), (t+17) % 190000: Logprob(-float(i+1001))}
                      for i,t in enumerate(ids)]
            expected = visible_groups(ids, scores)
            for size in (1, 7, 1000):
                for stream in (False, True):
                    result = await replay(serving, chunk_output(
                        ids, scores, size, "stop" if ending else "length"), expected, stream=stream)
                    assert result["status"] == ("completed" if ending else "incomplete")
                    report.append({"kind": "synthetic_boundaries", "chunk_size": size,
                                   "ending": ending, "stream": stream, **result})
                    print(json.dumps({k:v for k,v in report[-1].items() if k != "api_output"},
                                     ensure_ascii=False), flush=True)
        for ending in ("", "<|return|>"):
            ids = enc.encode("<|channel|>final<|message|>Hello ", allowed_special="all")
            ids += enc.encode("🌍")[:-1]
            ids += enc.encode(ending, allowed_special="all")
            scores = [{t: Logprob(-float(i+1))} for i,t in enumerate(ids)]
            expected = visible_groups(ids, scores)
            for size in (1, 7, 1000):
                for stream in (False, True):
                    result = await replay(serving, chunk_output(
                        ids, scores, size, "stop" if ending else "length"), expected, stream=stream)
                    assert result["text"] == "Hello �"
                    report.append({"kind": "incomplete_utf8", "chunk_size": size,
                                   "ending": ending, "stream": stream, **result})
                    print(json.dumps({k:v for k,v in report[-1].items() if k != "api_output"},
                                     ensure_ascii=False), flush=True)
    finally:
        engine.shutdown()
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    asyncio.run(main(parser.parse_args()))
