#!/usr/bin/env bash
# Downloads the default models into ./models (about 3.5 GB).
# Usage: scripts/download-models.sh [--large]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p models

files=(
  "Qwen/Qwen3-Embedding-0.6B-GGUF Qwen3-Embedding-0.6B-Q8_0.gguf"
  "Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"
  "Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF qwen2.5-coder-0.5b-instruct-fp16.gguf"
)
if [[ "${1:-}" == "--large" ]]; then
  files+=("Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
fi

for entry in "${files[@]}"; do
  read -r repo file <<<"$entry"
  if [[ -f "models/$file" ]]; then echo "present  $file"; continue; fi
  echo "download $file"
  curl -L --fail --retry 5 -C - -o "models/$file.part" "https://huggingface.co/$repo/resolve/main/$file"
  mv "models/$file.part" "models/$file"
done
echo "models are in $(pwd)/models"
