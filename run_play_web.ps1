# 启动 viser 回放网站（网页版，带速度滑块）
#
# 用法：
#   .\run_play_web.ps1                                   # 16DOF 站立，最新一次训练
#   .\run_play_web.ps1 Robot-16DOF-Flat-v0               # 指定任务
#   .\run_play_web.ps1 Robot-16DOF-Stand-v0 <ckpt路径>    # 指定权重
#
# 打开浏览器访问： http://localhost:8080
# 停止服务： Get-NetTCPConnection -LocalPort 8080 | % { Stop-Process -Id $_.OwningProcess -Force }

param(
    [string]$Task = "Robot-16DOF-Stand-v0",
    [string]$Checkpoint = ""
)

$ErrorActionPreference = "Continue"
$root  = $PSScriptRoot
$cache = "$root\.cache"

# ---- 环境变量：必须在当前进程设置，子进程才能继承 ----
$env:UV_CACHE_DIR            = "F:\RC2027\.uv-cache"
$env:WARP_CACHE_PATH         = "$cache\warp"
$env:CUDA_CACHE_PATH         = "$cache\cuda"
$env:MPLCONFIGDIR            = "$cache\mpl"
$env:TMP                     = "$cache\tmp"
$env:TEMP                    = "$cache\tmp"
$env:TMPDIR                  = "$cache\tmp"
$env:PYTHONIOENCODING        = "utf-8"
$env:PYTHONPATH              = "$root\src"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:TORCHINDUCTOR_CACHE_DIR = "$cache\torchinductor"
$env:TRITON_CACHE_DIR        = "$cache\triton"
New-Item -ItemType Directory -Force -Path $env:TMP | Out-Null

# ---- 自动找最新权重 ----
if ([string]::IsNullOrWhiteSpace($Checkpoint)) {
    $expDirs = if ($Task -match "Robot-16DOF-Stand") {
        # The latest 16DOF stance retraining has a robot-specific experiment name.
        @("robot_stand_16dof_zero4590_v1_calf121", "robot_stand_16dof_v2")
    } elseif ($Task -match "Robot-16DOF-Flat") { @("robot_flat_16dof_zero4590_v1_calf121") }
    elseif ($Task -match "Stand") { @("robot_stand") }
    elseif ($Task -match "Flat") { @("robot_flat") }
    else { @("robot_stand") }
    $runDir = $null
    foreach ($expDir in $expDirs) {
        $runDir = Get-ChildItem "$root\logs\rsl_rl\$expDir" -Directory -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($runDir) { break }
    }
    if (-not $runDir) { throw "找不到训练目录: $root\logs\rsl_rl\$expDir" }
    $ckpt = Get-ChildItem $runDir.FullName -File -Filter "model_*.pt" |
        Sort-Object { [int]($_.BaseName -replace "model_", "") } | Select-Object -Last 1
    if (-not $ckpt) { throw "找不到权重: $($runDir.FullName)" }
    $Checkpoint = $ckpt.FullName
}

if ($Task -eq "Robot-16DOF-Stand-v0" -and
    $Checkpoint -match 'robot_stand_16dof_v2[\\/]') {
    $Task = "Robot-16DOF-Stand-LegacyV2-v0"
    Write-Host "Using the original pose reference for this pre-zero4590 checkpoint."
}

Write-Host "任务 : $Task"
Write-Host "权重 : $Checkpoint"
Write-Host "网址 : http://localhost:8080"

# ---- 清端口 ----
$conn = Get-NetTCPConnection -LocalPort 8080 -State Listen -ErrorAction SilentlyContinue
if ($conn) {
    foreach ($p in ($conn.OwningProcess | Select-Object -Unique)) {
        Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 2
}

$log = "$cache\play_web.log"
$exe = "$root\.venv\Scripts\play.exe"

# ---- 启动（子进程继承上面的环境变量）----
Start-Process -FilePath $exe `
    -ArgumentList @($Task, "--checkpoint-file", $Checkpoint, "--num-envs", "1", "--viewer", "viser") `
    -WorkingDirectory $root `
    -RedirectStandardOutput "$cache\play_web.out" `
    -RedirectStandardError  "$cache\play_web.err" `
    -WindowStyle Hidden | Out-Null

# ---- 等待就绪 ----
Write-Host "启动中，等待 8080 就绪..."
for ($i = 0; $i -lt 40; $i++) {
    Start-Sleep -Seconds 3
    try {
        $r = Invoke-WebRequest "http://127.0.0.1:8080/" -UseBasicParsing -TimeoutSec 3
        if ($r.StatusCode -eq 200) {
            Write-Host ""
            Write-Host "就绪 -> 浏览器打开 http://localhost:8080" -ForegroundColor Green
            exit 0
        }
    } catch { }
}

Write-Host "启动超时。错误见: $cache\play_web.err" -ForegroundColor Red
Get-Content "$cache\play_web.err" -Tail 15 -ErrorAction SilentlyContinue
exit 1
