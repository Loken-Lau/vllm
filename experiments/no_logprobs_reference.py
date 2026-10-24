"""Compare unchanged output conversion on identical saved real-engine tokens."""
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from vllm.entrypoints.openai.responses.context import HarmonyContext
from vllm.entrypoints.openai.responses.harmony import harmony_to_response_output
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.parser.harmony import HarmonyParser


def normalize(value):
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items() if k not in ("id", "call_id")}
    if isinstance(value, list):
        return [normalize(v) for v in value]
    return value


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    report = []
    for source in sorted(Path(args.evidence).glob("engine-pipeline.raw-*.json")):
        raw = json.loads(source.read_text())
        ids = raw["ids"]
        for size in (1, 7, 1000):
            context = HarmonyContext([], [], frozenset(), HarmonyParser(tokenizer))
            for start in range(0, len(ids), size):
                finished = start + size >= len(ids)
                context.append_output(RequestOutput(
                    request_id="test", prompt=None, prompt_token_ids=[7, 8], prompt_logprobs=None,
                    outputs=[CompletionOutput(index=0, text="", token_ids=ids[start:start+size],
                        cumulative_logprob=None, logprobs=None,
                        finish_reason="stop" if finished else None)], finished=finished))
            items = [normalize(item.model_dump(mode="json")) for msg in context.messages
                     for item in harmony_to_response_output(msg, frozenset())]
            report.append({"source": source.name, "chunk_size": size, "output": items})
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved {len(report)} output-conversion cases from {HarmonyContext.__module__}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--evidence", required=True)
    p.add_argument("--output", required=True)
    main(p.parse_args())
