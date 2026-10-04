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
echo   [1] High-FPS Web Simulation (Terminal Preprocessing -^> Localhost:8080)
echo   [2] Batch Raw Data Preprocessor (Terminal Only)
echo   [3] Fast CLI Virtual LiDAR Replay (Sequence 00)
echo   [4] Benchmark (Adaptive vs. Uniform Grid Profiling)
echo   [5] Train Perception Model (AdaptivePolarNet)
echo   [6] Validate Dataset
echo   [7] Run Unit ^& Integration Tests
echo.

set /p CHOICE="Enter choice (1-7) [default: 1]: "
if "%CHOICE%"=="" set CHOICE=1

if "%CHOICE%"=="1" (
    echo Starting High-FPS Simulation Pipeline at http://localhost:8080 ...
    start "" http://localhost:8080
    python -m tools.replay_kitti --sequence 00 --start-frame 0 --end-frame 200 --fps 30 --dashboard
) else if "%CHOICE%"=="2" (
    echo Starting Terminal Batch Raw Data Preprocessor ...
    python -m tools.preprocess_sequence --sequence 00 --start-frame 0 --end-frame 200 --mode ground_truth
) else if "%CHOICE%"=="3" (
    echo Starting Fast CLI Virtual LiDAR Replay ...
    python -m tools.replay_kitti --sequence 00 --start-frame 0 --end-frame 200 --fps 30
) else if "%CHOICE%"=="4" (
    echo Running Benchmark Profiling ...
    python -m tools.benchmark --sequence 00 --frames 50
) else if "%CHOICE%"=="5" (
    echo Starting Model Training ...
    python -m training.train --config configs/training.yaml
) else if "%CHOICE%"=="6" (
    echo Validating Dataset ...
    python -m tools.validate_dataset
) else if "%CHOICE%"=="7" (
    echo Running Test Suite ...
    python -m pytest tests/ -v
) else (
    echo Invalid choice.
)

pause
