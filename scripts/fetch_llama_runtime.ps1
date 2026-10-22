<#
.SYNOPSIS
  Downloads the llama.cpp runtime (and generates the import libraries) for the
  llama.cpp embedding backend.

.DESCRIPTION
  The llama.cpp worker links against llama / mtmd / ggml. The prebuilt Windows
  package does not ship import libraries, so they are generated from the DLL
  exports (dumpbin + lib, i.e. the MSVC toolchain).

  The pinned checksum lives in scripts/checksums/llama-<version>-<platform>.sha256.

  Only needed for the llama.cpp backend; the ONNX Runtime backend needs none of
  this.

.PARAMETER OutDir
  Where to place the runtime DLLs and import libraries. Default: ./runtime/llama

.PARAMETER ModelDir
  Optional directory holding the GGUF files to copy next to the runtime.

.EXAMPLE
  ./scripts/fetch_llama_runtime.ps1 -OutDir ./runtime/llama
#>
param(
  [string]$OutDir = '',
  [string]$ModelDir = '',
  [string]$Version = 'b11669',
  [string]$Platform = 'windows-x64-vulkan',
  [switch]$SkipImportLib
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$root = Split-Path -Parent $PSScriptRoot
if (-not $OutDir) { $OutDir = Join-Path $root 'runtime/llama' }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$asset = "llama-$Version-bin-win-vulkan-x64.zip"
$url = "https://github.com/ggml-org/llama.cpp/releases/download/$Version/$asset"

$sumFile = Join-Path $PSScriptRoot "checksums/llama-$Version-$Platform.sha256"
$expected = $null
if (Test-Path $sumFile) {
  foreach ($line in Get-Content $sumFile) {
    if ($line -match '^\s*([^\s#]+)\s+([0-9a-fA-F]{64})\s*$') {
      $expected = $matches[2].ToLower()
    }
  }
}
if (-not $expected) {
  throw "No pinned checksum for $asset (expected $sumFile). Pin it before publishing."
}

$cache = Join-Path $root "runtime/cache"
New-Item -ItemType Directory -Force -Path $cache | Out-Null
$zip = Join-Path $cache $asset

if (-not (Test-Path $zip)) {
  Write-Output "Downloading $url"
  Invoke-WebRequest -Uri $url -OutFile $zip
}

$actual = (Get-FileHash -Path $zip -Algorithm SHA256).Hash.ToLower()
if ($actual -ne $expected) {
  throw "sha256 mismatch for $asset: expected $expected, got $actual"
}

$extract = Join-Path $cache "$Version-extract"
if (Test-Path $extract) { Remove-Item -Recurse -Force $extract }
Expand-Archive -Path $zip -DestinationPath $extract -Force

if (-not $SkipImportLib) {
  $defDir = Join-Path $cache "$Version-def"
  New-Item -ItemType Directory -Force -Path $defDir | Out-Null
  foreach ($dll in @('llama.dll', 'mtmd.dll', 'ggml.dll', 'ggml-base.dll', 'ggml-vulkan.dll')) {
    $path = Get-ChildItem $extract -Recurse -Filter $dll | Select-Object -First 1
    if (-not $path) { continue }
    $name = [System.IO.Path]::GetFileNameWithoutExtension($dll)
    $def = Join-Path $defDir "$name.def"
    'EXPORTS' | Set-Content $def
    & dumpbin /exports $path.FullName |
      Select-Object -Skip 19 |
      ForEach-Object { ($_ -split '\s+')[3] } |
      Where-Object { $_ -and $_ -ne '' -and $_ -notmatch '^\[' } |
      Sort-Object -Unique | Add-Content $def
    & lib /def:$def /out:(Join-Path $cache "$name.lib") /machine:x64 | Out-Null
    Write-Output "import library: $name.lib"
  }
}

Get-ChildItem $extract -Recurse -Filter *.dll | ForEach-Object { Copy-Item $_.FullName -Destination $OutDir -Force }
Get-ChildItem $cache -Filter *.lib -ErrorAction SilentlyContinue | Copy-Item -Destination $OutDir -Force

$required = @('llama.dll', 'mtmd.dll', 'ggml-base.dll')
$missing = $required | Where-Object { -not (Test-Path (Join-Path $OutDir $_)) }
if ($missing) {
  throw "runtime incomplete, missing: $($missing -join ', ') (in $OutDir)"
}

if ($ModelDir) {
  foreach ($m in @('embeddinggemma-2-Q8_0.gguf', 'mmproj-embeddinggemma-2-Q8_0.gguf')) {
    $src = Join-Path $ModelDir $m
    if (Test-Path $src) { Copy-Item $src $OutDir -Force }
  }
}

Write-Output "llama.cpp runtime staged in $OutDir"
Write-Output "Point FO_LLAMA_LIB_DIR at that directory when building the worker."
