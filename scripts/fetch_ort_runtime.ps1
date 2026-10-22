<#
.SYNOPSIS
  Download and verify the ONNX Runtime used by embed_core / ocr_core.

.DESCRIPTION
  The ONNX Runtime is loaded dynamically (load-dynamic), so the shared library
  is not part of this repository. This script downloads the pinned release,
  verifies its sha256 against scripts/sha256sums.txt and stages the library.

.EXAMPLE
  scripts/fetch_ort_runtime.ps1 -OutDir ./runtime
  scripts/fetch_ort_runtime.ps1 -Platform linux-x64 -OutDir ./runtime
#>
param(
  [string]$Platform = 'win-x64',
  [string]$OutDir = '',
  [string]$Version = '1.30.0'
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$root = Split-Path -Parent $PSScriptRoot
if (-not $OutDir) { $OutDir = Join-Path $root 'runtime/ort' }

# Pinned release assets: file name -> sha256 (see scripts/sha256sums.txt).
$assets = @{
  'win-x64' = @{ zip = "onnxruntime-win-x64-$Version.zip"; lib = 'onnxruntime.dll' }
  'win-x86' = @{ zip = "onnxruntime-win-x86-$Version.zip"; lib = 'onnxruntime.dll' }
  'macos-arm64' = @{ zip = "onnxruntime-osx-arm64-$Version.zip"; lib = 'libonnxruntime.dylib' }
  'macos-x64' = @{ zip = "onnxruntime-osx-x86_64-$Version.zip"; lib = 'libonnxruntime.dylib' }
  'linux-x64' = @{ zip = "onnxruntime-linux-x64-$Version.tgz"; lib = 'libonnxruntime.so' }
}
if (-not $assets.ContainsKey($Platform)) {
  throw "Unsupported platform '$Platform'. Known: $($assets.Keys -join ', ')"
}
$asset = $assets[$Platform]
$url = "https://github.com/microsoft/onnxruntime/releases/download/v$Version/$($asset.zip)"

$sums = Join-Path $root 'scripts/sha256sums.txt'
if (-not (Test-Path $sums)) { throw "Missing $sums" }
$expected = $null
foreach ($line in Get-Content $sums) {
  if ($line -match '^\s*([^\s#]+)\s+([0-9a-fA-F]{64})\s*$' -and $matches[1] -eq $asset.zip) {
    $expected = $matches[2].ToLower()
  }
}
if (-not $expected) { throw "No sha256 recorded for $($asset.zip) in $sums" }

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$work = Join-Path ([System.IO.Path]::GetTempPath()) "ort-$Version-$Platform"
New-Item -ItemType Directory -Force -Path $work | Out-Null
$archive = Join-Path $work $asset.zip

if (-not (Test-Path $archive)) {
  Write-Output "Downloading $url"
  Invoke-WebRequest -Uri $url -OutFile $archive
}

$actual = (Get-FileHash -Path $archive -Algorithm SHA256).Hash.ToLower()
if ($actual -ne $expected) {
  throw "SHA-256 mismatch for $($asset.zip).`n  expected: $expected`n  actual:   $actual"
}
Write-Output "sha256 ok: $($asset.zip)"

if ($asset.zip.EndsWith('.tgz')) {
  tar -xzf $archive -C $work
} else {
  Expand-Archive -Path $archive -DestinationPath $work -Force
}

$lib = Get-ChildItem $work -Recurse -Filter $asset.lib | Select-Object -First 1
if (-not $lib) { throw "Runtime library $($asset.lib) not found in the archive" }
Copy-Item $lib.FullName -Destination (Join-Path $OutDir $asset.lib) -Force
foreach ($extra in @('LICENSE', 'ThirdPartyNotices.txt')) {
  $f = Get-ChildItem $work -Recurse -Filter $extra | Select-Object -First 1
  if ($f) { Copy-Item $f.FullName -Destination (Join-Path $OutDir $extra) -Force }
}

Write-Output "ONNX Runtime $Version staged in $OutDir"
