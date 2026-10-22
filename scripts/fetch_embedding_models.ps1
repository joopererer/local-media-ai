<#
.SYNOPSIS
  Downloads the EmbeddingGemma 2 model files and verifies their sha256.

.DESCRIPTION
  No model weights are stored in this repository. This script fetches them from
  the pinned upstream revision and verifies every file against the sha256 listed
  in models/embeddinggemma2_onnx_manifest.json (ONNX export) or
  models/embeddinggemma2_gguf_manifest.json (GGUF quantisation).

  The GGUF files are only needed for the llama.cpp backend; the ONNX export is
  what embed_core uses.

.PARAMETER OutDir
  Target directory. Default: ./runtime/models

.PARAMETER Backend
  onnx (default) or gguf.

.EXAMPLE
  ./scripts/fetch_embedding_models.ps1 -OutDir ./runtime/models
  ./scripts/fetch_embedding_models.ps1 -Backend gguf -OutDir ./runtime/models
#>
param(
  [string]$OutDir = '',
  [ValidateSet('onnx', 'gguf')]
  [string]$Backend = 'onnx',
  [string]$Mirror = 'https://huggingface.co'
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$root = Split-Path -Parent $PSScriptRoot
if (-not $OutDir) { $OutDir = Join-Path $root 'runtime/models' }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

if ($Backend -eq 'gguf') {
  $manifestPath = Join-Path $root 'models/embeddinggemma2_gguf_manifest.json'
  $repo = 'ggml-org/embeddinggemma-2-GGUF'
} else {
  $manifestPath = Join-Path $root 'models/embeddinggemma2_onnx_manifest.json'
  $repo = 'onnx-community/embeddinggemma-2-ONNX'
}

$manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
$commit = $manifest.commit
$base = "$Mirror/$repo/resolve/$commit"

Write-Output "Fetching $repo @ $commit"
Write-Output "  -> $OutDir"

foreach ($f in $manifest.files) {
  $target = Join-Path $OutDir $f.path
  $parent = Split-Path -Parent $target
  New-Item -ItemType Directory -Force -Path $parent | Out-Null

  if ((Test-Path $target) -and ((Get-Item $target).Length -eq $f.size)) {
    $hash = (Get-FileHash -Path $target -Algorithm SHA256).Hash.ToLower()
    if ($hash -eq $f.sha256.ToLower()) {
      Write-Output "  = $($f.path) (already present, sha256 ok)"
      continue
    }
  }

  $url = "$base/$($f.path)"
  Write-Output "  + $($f.path) ($([math]::Round($f.size / 1MB, 1)) MB)"
  $tmp = "$target.part"
  Invoke-WebRequest -Uri $url -OutFile $tmp

  $hash = (Get-FileHash -Path $tmp -Algorithm SHA256).Hash.ToLower()
  if ($hash -ne $f.sha256.ToLower()) {
    Remove-Item $tmp -Force -ErrorAction SilentlyContinue
    throw "sha256 mismatch for $($f.path): expected $($f.sha256), got $hash"
  }
  Move-Item $tmp $target -Force
}

Write-Output "Done. Model directory: $OutDir"
Write-Output "embed_core expects this layout: tokenizer.json + onnx/*_quantized.onnx(_data)"
