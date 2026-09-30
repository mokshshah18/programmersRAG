# Downloads the default models into .\models (about 3.5 GB).
# Usage: powershell -ExecutionPolicy Bypass -File scripts\download-models.ps1 [-Large]
param([switch]$Large)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$dir = Join-Path $root "models"
New-Item -ItemType Directory -Force $dir | Out-Null

$files = @(
  @{ repo = "Qwen/Qwen3-Embedding-0.6B-GGUF";        file = "Qwen3-Embedding-0.6B-Q8_0.gguf" },
  @{ repo = "Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF"; file = "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf" },
  @{ repo = "Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF"; file = "qwen2.5-coder-0.5b-instruct-fp16.gguf" }
)
if ($Large) {
  $files += @{ repo = "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF"; file = "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf" }
}

foreach ($f in $files) {
  $out = Join-Path $dir $f.file
  if (Test-Path $out) { Write-Host "present  $($f.file)"; continue }
  $url = "https://huggingface.co/$($f.repo)/resolve/main/$($f.file)"
  Write-Host "download $($f.file)"
  & curl.exe -L --fail --retry 5 --ssl-revoke-best-effort -C - -o "$out.part" $url
  if ($LASTEXITCODE -ne 0) { throw "download failed: $url" }
  Move-Item -Force "$out.part" $out
}
Write-Host "models are in $dir"
