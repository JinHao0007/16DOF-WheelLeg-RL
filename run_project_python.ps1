param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PythonArguments
)
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$cache = Join-Path $root ".cache"
$env:UV_CACHE_DIR = "F:\RC2027\.uv-cache"
$env:PYTHONPATH = Join-Path $root "src"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONDONTWRITEBYTECODE = "1"
foreach ($name in @("TMP", "TEMP", "TMPDIR")) {
    Set-Item "Env:$name" (Join-Path $cache "tmp")
}
foreach ($item in @{
    WARP_CACHE_PATH="warp"; CUDA_CACHE_PATH="cuda"; MPLCONFIGDIR="mpl";
    TORCHINDUCTOR_CACHE_DIR="torchinductor"; TRITON_CACHE_DIR="triton"
}.GetEnumerator()) {
    Set-Item "Env:$($item.Key)" (Join-Path $cache $item.Value)
}
New-Item -ItemType Directory -Force -Path $env:TMP | Out-Null
Set-Location $root
# Use the existing environment directly; do not resolve or download packages.
& (Join-Path $root ".venv\Scripts\python.exe") @PythonArguments
if ($LASTEXITCODE -ne 0) { throw "Python exited with code $LASTEXITCODE" }
