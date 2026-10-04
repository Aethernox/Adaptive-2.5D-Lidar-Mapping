# PowerShell Interactive Launcher
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host " Adaptive Variable-Resolution 2.5D LiDAR Perception Launcher" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan

if (Test-Path ".venv\Scripts\Activate.ps1") {
    & .venv\Scripts\Activate.ps1
}

Write-Host ""
Write-Host "Select Execution Mode:" -ForegroundColor Yellow
Write-Host "  [1] High-FPS Web Simulation (Terminal Pre-processing -> Localhost:8080)" -ForegroundColor Green
Write-Host "  [2] Batch Raw Data Preprocessor (Terminal Only)"
Write-Host "  [3] Fast CLI Virtual LiDAR Replay (Sequence 00)"
Write-Host "  [4] Benchmark Profiling (Adaptive vs. Uniform Grid)"
Write-Host "  [5] Train Perception Model (AdaptivePolarNet)"
Write-Host "  [6] Validate Dataset"
Write-Host "  [7] Run Unit & Integration Tests"
Write-Host ""

$choice = Read-Host "Enter choice (1-7) [default: 1]"
if ([string]::IsNullOrWhiteSpace($choice)) { $choice = "1" }

switch ($choice) {
    "1" {
        Write-Host "Starting High-FPS Simulation Pipeline..." -ForegroundColor Green
        Start-Process "http://localhost:8080"
        python -m tools.replay_kitti --sequence 00 --start-frame 0 --end-frame 200 --fps 30 --dashboard
    }
    "2" {
        Write-Host "Starting Terminal Batch Raw Data Preprocessor..." -ForegroundColor Green
        python -m tools.preprocess_sequence --sequence 00 --start-frame 0 --end-frame 200 --mode ground_truth
    }
    "3" {
        Write-Host "Starting CLI Virtual LiDAR Replay ..." -ForegroundColor Green
        python -m tools.replay_kitti --sequence 00 --start-frame 0 --end-frame 200 --fps 30
    }
    "4" {
        Write-Host "Running Benchmark Profiling ..." -ForegroundColor Green
        python -m tools.benchmark --sequence 00 --frames 50
    }
    "5" {
        Write-Host "Starting Model Training ..." -ForegroundColor Green
        python -m training.train --config configs/training.yaml
    }
    "6" {
        Write-Host "Validating Dataset ..." -ForegroundColor Green
        python -m tools.validate_dataset
    }
    "7" {
        Write-Host "Running Test Suite ..." -ForegroundColor Green
        python -m pytest tests/ -v
    }
    Default {
        Write-Host "Invalid choice." -ForegroundColor Red
    }
}
