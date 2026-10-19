#!/usr/bin/env bash
# Reverse only the source patch, run independent baseline engines, then restore.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "$0")/../../vllm" && pwd)"
cd "$repo_dir"
patch_file="$(mktemp /tmp/audio-mask-baseline.XXXXXX.patch)"
git diff --binary -- \
  vllm/model_executor/models/qwen2_audio.py \
  vllm/model_executor/models/audioflamingo3.py > "$patch_file"
test -s "$patch_file"
git apply --reverse --check "$patch_file"
git apply --reverse "$patch_file"
restore_compact() {
  git apply "$patch_file"
  rm "$patch_file"
}
trap restore_compact EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
git diff --exit-code -- \
  vllm/model_executor/models/qwen2_audio.py \
  vllm/model_executor/models/audioflamingo3.py
for model in qwen2_audio audioflamingo3; do
  file_model="${model//_/-}"
  if [[ "$model" == "audioflamingo3" ]]; then file_model="$model"; fi
  ../.tools/bin/uv run --no-project --python .venv/bin/python python \
    ../.contributions/infra-screen/eval_audio_models.py --model "$model" \
    --output "../.contributions/infra-screen/${file_model}-baseline-v2.json" \
    > "../.contributions/infra-screen/${file_model}-baseline-v2.log" 2>&1
done
