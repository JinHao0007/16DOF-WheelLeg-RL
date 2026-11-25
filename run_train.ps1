param(
    [string]$Task = "Robot-16DOF-Stand-v0",
    [int]$Envs = 512,
    [int]$Iters = 600,
    [string]$ExperimentName = "",
    [switch]$CaptureNan
)

# Usage:
#   .\run_train.ps1                              # 16DOF zero4590 standing, 512 envs
#   .\run_train.ps1 Robot-Flat-v0 1024 1000      # flat baseline
#   .\run_train.ps1 Robot-StepStone-v0 512 300   # quick test

# Keep Continue: uv writes progress to stderr, and "Stop" would turn that into a
# terminating error at the end of an otherwise successful training run.
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("Path", "User")
}

# Keep every cache/temp dir inside the project, otherwise Warp/CUDA write to C:
$cache = Join-Path $root ".cache"
$env:UV_CACHE_DIR     = "F:\RC2027\.uv-cache"
$env:WARP_CACHE_PATH  = Join-Path $cache "warp"
$env:CUDA_CACHE_PATH  = Join-Path $cache "cuda"
$env:MPLCONFIGDIR     = Join-Path $cache "mpl"
$env:TMP              = Join-Path $cache "tmp"
$env:TEMP             = $env:TMP
$env:TMPDIR           = $env:TMP
$env:PYTHONIOENCODING  = "utf-8"
$env:PYTHONPATH       = Join-Path $root "src"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:TORCHINDUCTOR_CACHE_DIR = Join-Path $cache "torchinductor"
$env:TRITON_CACHE_DIR  = Join-Path $cache "triton"
New-Item -ItemType Directory -Force -Path $env:TMP | Out-Null

Set-Location $root
Write-Host "task=$Task  envs=$Envs  iters=$Iters  cwd=$root"
$arguments = @($Task, "--env.scene.num-envs", "$Envs", "--agent.max-iterations", "$Iters")
if ($ExperimentName) {
    $arguments += @("--agent.experiment-name", $ExperimentName)
}
if ($CaptureNan) {
    $nanDir = Join-Path $root "logs\rsl_rl\nan_dumps\$Task"
    New-Item -ItemType Directory -Force -Path $nanDir | Out-Null
    $arguments += @(
        "--enable-nan-guard", "True",
        "--env.sim.nan-guard.output-dir", $nanDir
    )
    Write-Host "NaN state capture: $nanDir"
}
uv run --no-sync train @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Training failed with exit code $LASTEXITCODE"
}
