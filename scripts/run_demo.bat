@echo off
setlocal enabledelayedexpansion

echo ==================================================================
echo  Adaptive Variable-Resolution 2.5D LiDAR Perception Launcher
echo ==================================================================

REM Activate Virtual Environment
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

echo.
echo Select Execution Mode:
echo   [1] Interactive Web Dashboard (3D LiDAR + 2.5D Polar Grid + Metrics)
echo   [2] CLI Replay Simulation (Sequence 00)
echo   [3] Benchmark (Adaptive vs. Uniform Grid Profiling)
echo   [4] Train Perception Model (AdaptivePolarNet)
echo   [5] Validate Dataset
echo   [6] Run Unit & Integration Tests
echo.

set /p CHOICE="Enter choice (1-6) [default: 1]: "
if "%CHOICE%"=="" set CHOICE=1

if "%CHOICE%"=="1" (
    echo Launching Interactive Web Dashboard at http://localhost:8080 ...
    start "" http://localhost:8080
    python -m tools.replay_kitti --sequence 00 --start-frame 0 --end-frame 500 --fps 10 --dashboard
) else if "%CHOICE%"=="2" (
    echo Starting CLI Virtual LiDAR Replay ...
    python -m tools.replay_kitti --sequence 00 --start-frame 0 --end-frame 200 --fps 10
) else if "%CHOICE%"=="3" (
    echo Running Benchmark Profiling ...
    python -m tools.benchmark --sequence 00 --frames 50
) else if "%CHOICE%"=="4" (
    echo Starting Model Training ...
    python -m training.train --config configs/training.yaml
) else if "%CHOICE%"=="5" (
    echo Validating Dataset ...
    python -m tools.validate_dataset
) else if "%CHOICE%"=="6" (
    echo Running Test Suite ...
    python -m pytest tests/ -v
) else (
    echo Invalid choice.
)

pause
