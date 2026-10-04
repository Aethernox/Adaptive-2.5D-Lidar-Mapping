# PowerShell Interactive Launcher
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host " Adaptive Variable-Resolution 2.5D LiDAR Perception Launcher" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan

if (Test-Path ".venv\Scripts\Activate.ps1") {
    & .venv\Scripts\Activate.ps1
}

Write-Host ""
Write-Host "Select Execution Mode:" -ForegroundColor Yellow
Write-Host "  [1] Interactive Web Dashboard (3D LiDAR + 2.5D Polar Grid + Metrics)"
Write-Host "  [2] CLI Replay Simulation (Sequence 00)"
Write-Host "  [3] Benchmark (Adaptive vs. Uniform Grid Profiling)"
Write-Host "  [4] Train Perception Model (AdaptivePolarNet)"
Write-Host "  [5] Validate Dataset"
Write-Host "  [6] Run Unit & Integration Tests"
Write-Host ""

$choice = Read-Host "Enter choice (1-6) [default: 1]"
if ([string]::IsNullOrWhiteSpace($choice)) { $choice = "1" }

switch ($choice) {
    "1" {
        Write-Host "Launching Interactive Web Dashboard at http://localhost:8080 ..." -ForegroundColor Green
        Start-Process "http://localhost:8080"
        python -m tools.replay_kitti --sequence 00 --start-frame 0 --end-frame 500 --fps 10 --dashboard
    }
    "2" {
        Write-Host "Starting CLI Virtual LiDAR Replay ..." -ForegroundColor Green
        python -m tools.replay_kitti --sequence 00 --start-frame 0 --end-frame 200 --fps 10
    }
    "3" {
        Write-Host "Running Benchmark Profiling ..." -ForegroundColor Green
        python -m tools.benchmark --sequence 00 --frames 50
    }
    "4" {
        Write-Host "Starting Model Training ..." -ForegroundColor Green
        python -m training.train --config configs/training.yaml
    }
    "5" {
        Write-Host "Validating Dataset ..." -ForegroundColor Green
        python -m tools.validate_dataset
    }
    "6" {
        Write-Host "Running Test Suite ..." -ForegroundColor Green
        python -m pytest tests/ -v
    }
    Default {
        Write-Host "Invalid choice." -ForegroundColor Red
    }
}
