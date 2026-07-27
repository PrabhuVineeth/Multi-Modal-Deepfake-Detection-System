param(
    [int]$IntervalMinutes = 45
)

$ProjectDir = "C:\Users\Nitte\Desktop\NNM24AD071\Multi-Modal-Deepfake-Detection-System-main"
$LogDir = Join-Path $ProjectDir "output\training_monitor"
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
$MonitorLog = Join-Path $LogDir "monitor.log"

function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$timestamp] $Message"
    Write-Host $line
    Add-Content -Path $MonitorLog -Value $line
}

Write-Log "=== Training Monitor Started ==="
Write-Log "Monitoring interval: $IntervalMinutes minutes"
Write-Log "Project directory: $ProjectDir"

while ($true) {
    $now = Get-Date
    Write-Log "--- CHECK @ $now ---"

    # Check Python training process
    $processes = Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object {
        $cmd = (Get-WmiObject -Class Win32_Process -Filter "ProcessId=$($_.Id)" -ErrorAction SilentlyContinue).CommandLine
        $cmd -like "*train_and_eval_backbones*"
    }

    if ($processes) {
        foreach ($proc in $processes) {
            $cmd = (Get-WmiObject -Class Win32_Process -Filter "ProcessId=$($proc.Id)").CommandLine
            Write-Log "PROCESS RUNNING: PID=$($proc.Id), Started=$($proc.StartTime), CPU=$($proc.CPU)s, Memory=$([math]::Round($proc.WorkingSet / 1GB, 2))GB, Responding=$($proc.Responding)"
            Write-Log "  Command: $cmd"
        }
    } else {
        Write-Log "PROCESS NOT FOUND: train_and_eval_backbones.py is not running"
    }

    # Check checkpoints
    $ckptDir = Join-Path $ProjectDir "checkpoints"
    if (Test-Path $ckptDir) {
        $checkpoints = Get-ChildItem -Path $ckptDir -Filter "best_model_*face*.pth" | Sort-Object LastWriteTime -Descending
        Write-Log "CHECKPOINTS (face detector combos):"
        foreach ($ck in $checkpoints) {
            $age = (Get-Date) - $ck.LastWriteTime
            $ageStr = "$($age.Minutes)m $($age.Seconds)s ago"
            Write-Log "  $($ck.Name) | Modified: $($ck.LastWriteTime) ($ageStr) | Size: $([math]::Round($ck.Length / 1MB, 1))MB"
        }
    }

    # Check for any log files modified recently
    $recentLogs = Get-ChildItem -Path $ProjectDir -Recurse -Include "*.log" -ErrorAction SilentlyContinue |
        Where-Object { $_.LastWriteTime -gt (Get-Date).AddHours(-2) } |
        Sort-Object LastWriteTime -Descending

    if ($recentLogs) {
        Write-Log "RECENT LOG FILES (modified in last 2h):"
        foreach ($log in $recentLogs | Select-Object -First 5) {
            Write-Log "  $($log.FullName) | Modified: $($log.LastWriteTime) | Size: $([math]::Round($log.Length / 1KB, 1))KB"
        }
    } else {
        Write-Log "No log files modified in the last 2 hours"
    }

    # Check GPU usage if nvidia-smi is available
    $nvidia = Get-Command "nvidia-smi" -ErrorAction SilentlyContinue
    if ($nvidia) {
        $gpuInfo = & nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>$null
        if ($gpuInfo) {
            Write-Log "GPU: $gpuInfo"
        }
    }

    Write-Log "--- END CHECK ---"
    Write-Log ""

    Start-Sleep -Seconds ($IntervalMinutes * 60)
}
