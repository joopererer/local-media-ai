<#
.SYNOPSIS
  Downloads the ONNX Runtime shared library and verifies its sha256.

.DESCRIPTION
  The crates load the ONNX Runtime dynamically from an absolute path at runtime
  (load-dynamic); nothing is linked in and no binary is stored in this repository.

  The pinned checksums live in scripts/checksums/onnxruntime-<platform>.sha256.
  A platform without a pinned checksum is refused unless -AllowUnpinned is given:
  the script then prints the computed hash so it can be reviewed and pinned.

.PARAMETER Platform
  One of: windows-x64, linux-x64, osx-arm64, osx-x64.

.PARAMETER OutDir
  Where to place the library. Default: ./runtime/ort

.PARAMETER Version
  ONNX Runtime version to fetch. Default: 1.30.0

.EXAMPLE
  ./scripts/fetch_onnxruntime.ps1 -Platform windows-x64 -OutDir ./runtime/ort
#>
param(
  [ValidateSet('windows-x64', 'linux-x64', 'osx-arm64', 'osx-x64')]
  [string]$Platform = 'windows-x64',
  [string]$OutDir = '',
  [string]$Version = '1.30.0',
  [switch]$AllowUnpinned
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$root = Split-Path -Parent $PSScriptRoot
if (-not $OutDir) { $OutDir = Join-Path $root 'runtime/ort' }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$asset = switch ($Platform) {
  'windows-x64' { "onnxruntime-win-x64-$Version.zip" }
  'linux-x64'   { "onnxruntime-linux-x64-$Version.tgz" }
  'osx-arm64'   { "onnxruntime-osx-arm64-$Version.tgz" }
  'osx-x64'     { "onnxruntime-osx-x64-$Version.tgz" }
}
$url = "https://github.com/microsoft/onnxruntime/releases/download/v$Version/$asset"

$sumFile = Join-Path $PSScriptRoot "checksums/onnxruntime-$Platform.sha256"
$expected = $null
if (Test-Path $sumFile) {
  foreach ($line in Get-Content $sumFile) {
    if ($line -match '^\s*([^\s#]+)\s+([0-9a-fA-F]{64})\s*$') {
      $expected = $matches[2].ToLower()
    }
  }
}
if (-not $expected -and -not $AllowUnpinned) {
  Write-Error "No pinned checksum for $Platform $Version (expected $sumFile).`n" +
    "Run once with -AllowUnpinned, review the printed hash, then pin it in that file."
}

$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("lma-ort-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
try {
  $archive = Join-Path $tmp $asset
  Write-Output "Downloading $url"
  Invoke-WebRequest -Uri $url -OutFile $archive

  $actual = (Get-FileHash -Path $archive -Algorithm SHA256).Hash.ToLower()
  if ($expected -and $actual -ne $expected) {
    throw "sha256 mismatch for ${asset}: expected $expected, got $actual"
  }
  if (-not $expected) {
    Write-Warning "Unpinned download. Computed sha256: $actual"
    Write-Warning "Pin it in scripts/checksums/onnxruntime-$Platform.sha256"
  }

  if ($asset.EndsWith('.zip')) {
    Expand-Archive -Path $archive -DestinationPath $tmp -Force
  } else {
    & tar -xzf $archive -C $tmp
  }

  # Place the library (and its licence files) next to each other.
  $found = $null
  foreach ($name in @('onnxruntime.dll', 'libonnxruntime.so', 'libonnxruntime.dylib')) {
    $hit = Get-ChildItem $tmp -Recurse -File -Filter $name -ErrorAction SilentlyContinue |
      Where-Object { $_.DirectoryName -notmatch 'runtimes' } | Select-Object -First 1
    if ($hit) { $found = $hit; break }
  }
  if (-not $found) {
    $found = Get-ChildItem $tmp -Recurse -File -Filter 'onnxruntime*' -ErrorAction SilentlyContinue |
      Where-Object { $_.Extension -in '.dll', '.so', '.dylib' } | Select-Object -First 1
  }
  if (-not $found) { throw "no ONNX Runtime library found inside $asset" }

  Copy-Item $found.FullName -Destination (Join-Path $OutDir $found.Name) -Force
  foreach ($doc in @('LICENSE', 'ThirdPartyNotices.txt')) {
    $hit = Get-ChildItem $tmp -Recurse -File -Filter $doc -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($hit) { Copy-Item $hit.FullName -Destination (Join-Path $OutDir $doc) -Force }
  }

  Write-Output "ONNX Runtime $Version ($Platform) staged in $OutDir"
  Write-Output "  $(Split-Path -Leaf $found.FullName)"
} finally {
  Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}
