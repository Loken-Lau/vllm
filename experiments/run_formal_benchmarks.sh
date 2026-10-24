#!/usr/bin/env bash
set -euo pipefail
cd /path/to/workspace/pr3/vllm
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export PATH="$PWD/.venv/bin:/path/to/workspace/opensource/.tools/bin:$PATH"
evidence=/path/to/workspace/pr3/evidence/validation-20261024
model=/path/to/workspace/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee

# Avoid local integration/compilation load during timing measurements.
while pgrep -f '^.venv/bin/python -m pytest -v -s tests/entrypoints/openai/responses/test_harmony.py$' >/dev/null; do
    sleep 2
done
grep -E '^=+ .* passed' "$evidence/harmony-full-final.log"
for mode in base-off fixed-off fixed-on; do
    repo=/path/to/workspace/pr3/vllm
    extra=()
    if [[ "$mode" == base-off ]]; then repo=/path/to/workspace/pr3/vllm-base; fi
    if [[ "$mode" == fixed-on ]]; then extra+=(--enabled); fi
    echo "Starting $mode at $(date -u +%FT%TZ)"
    .venv/bin/python "$evidence/bench_responses.py" --repo "$repo" --model "$model" \
        "${extra[@]}" --output "$evidence/bench-final-$mode.json"
done
