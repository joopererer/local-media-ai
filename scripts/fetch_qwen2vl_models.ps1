<#
.SYNOPSIS
  Downloads Qwen2-VL 2B GGUF weights and vision multimodal projector for llama.cpp.

.DESCRIPTION
  Fetches the pinned Qwen2-VL 2B GGUF model and mmproj files and verifies
  their sha256 checksums from models/qwen2vl_gguf_manifest.json.

.PARAMETER OutDir
  Target directory. Default: ./runtime/models/qwen2-vl

.PARAMETER Mirror
  Download mirror. Default: https://hf-mirror.com (supports https://huggingface.co)

.EXAMPLE
  ./scripts/fetch_qwen2vl_models.ps1
  ./scripts/fetch_qwen2vl_models.ps1 -Mirror https://huggingface.co
#>
param(
  [string]$OutDir = '',
  [string]$Mirror = 'https://hf-mirror.com'
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$root = Split-Path -Parent $PSScriptRoot
if (-not $OutDir) { $OutDir = Join-Path $root 'runtime/models/qwen2-vl' }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$manifestPath = Join-Path $root 'models/qwen2vl_gguf_manifest.json'
if (-not (Test-Path $manifestPath)) {
  # Check parent directory fallback
  $manifestPath = Join-Path (Split-Path -Parent $root) 'local-media-ai/models/qwen2vl_gguf_manifest.json'
}

$manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
$repo = $manifest.repo
$commit = $manifest.commit
$base = "$Mirror/$repo/resolve/$commit"

Write-Output "Fetching $repo @ $commit"
Write-Output "  -> Target Directory: $OutDir"

foreach ($f in $manifest.files) {
  $target = Join-Path $OutDir $f.path
  $parent = Split-Path -Parent $target
  New-Item -ItemType Directory -Force -Path $parent | Out-Null

  if ((Test-Path $target) -and ((Get-Item $target).Length -eq $f.size)) {
    Write-Output "  Validating existing file: $($f.path)..."
    $hash = (Get-FileHash -Path $target -Algorithm SHA256).Hash.ToLower()
    if ($hash -eq $f.sha256.ToLower()) {
      Write-Output "  = $($f.path) (already present, sha256 verified)"
      continue
    }
  }

  $url = "$base/$($f.path)"
  $sizeMb = [math]::Round($f.size / 1MB, 1)
  Write-Output "  + Downloading $($f.path) ($sizeMb MB)..."
  $tmp = "$target.part"
  
  # Use .NET WebClient or BITS / Invoke-WebRequest
  try {
    Invoke-WebRequest -Uri $url -OutFile $tmp
  } catch {
    # Try fallback to curl.exe if available for large files with progress
    Write-Output "  Invoke-WebRequest failed, retrying with curl.exe..."
    & curl.exe -L --progress-bar -o $tmp $url
  }

  Write-Output "  Validating checksum for $($f.path)..."
  $hash = (Get-FileHash -Path $tmp -Algorithm SHA256).Hash.ToLower()
  if ($hash -ne $f.sha256.ToLower()) {
    Remove-Item $tmp -Force -ErrorAction SilentlyContinue
    throw "sha256 mismatch for $($f.path): expected $($f.sha256), got $hash"
  }
  Move-Item $tmp $target -Force
  Write-Output "  ✓ $($f.path) downloaded and verified."
}

Write-Output "All Qwen2-VL models ready in $OutDir"
