@echo off
echo ==================================================================
echo  Setting up Adaptive LiDAR Perception & Mapping Environment (Windows)
echo ==================================================================

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist ".venv" (
    echo Creating virtual environment in .venv...
    python -m venv .venv
)

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo Installing requirements...
pip install -r requirements.txt

REM Create required folders
if not exist "checkpoints" mkdir checkpoints
if not exist "logs\tensorboard" mkdir logs\tensorboard
if not exist "results" mkdir results

REM Validate dataset
echo Validating dataset...
python -m tools.validate_dataset

echo ==================================================================
echo  Setup completed successfully!
echo ==================================================================
pause
