param([switch]$CalibrationPose)
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$cache = Join-Path $root ".cache"
$env:TMP = Join-Path $cache "tmp"
$env:TEMP = $env:TMP
$env:TMPDIR = $env:TMP
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:MPLCONFIGDIR = Join-Path $cache "mpl"
New-Item -ItemType Directory -Force -Path $env:TMP | Out-Null
$python = Join-Path $root ".venv\Scripts\pythonw.exe"
$script = Join-Path $root "tools\open_model_native_16dof.py"
$nativeArgs = @("-B", $script)
if ($CalibrationPose) { $nativeArgs += "--calibration-pose" }
$nativeProcess = Start-Process -FilePath $python -ArgumentList $nativeArgs `
    -WorkingDirectory $root -WindowStyle Normal -PassThru `
    -RedirectStandardOutput (Join-Path $cache "model_native.out") `
    -RedirectStandardError (Join-Path $cache "model_native.err")
Write-Output "MuJoCo model inspector process: $($nativeProcess.Id)"
