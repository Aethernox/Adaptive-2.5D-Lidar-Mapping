# Setup script for Windows PowerShell
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host " Setting up Adaptive LiDAR Perception & Mapping (Windows PowerShell)" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan

# Check Python
try {
    $pyVer = python --version
    Write-Host "Found: $pyVer" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python is not installed or not in PATH." -ForegroundColor Red
    exit 1
}

# Create virtual environment if needed
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment in .venv..." -ForegroundColor Yellow
    python -m venv .venv
}

# Activate virtual environment
& .venv\Scripts\Activate.ps1

# Upgrade pip and install requirements
Write-Host "Installing dependencies..." -ForegroundColor Yellow
python -m pip install --upgrade pip
pip install -r requirements.txt

# Create directories
New-Item -ItemType Directory -Force -Path "checkpoints", "logs/tensorboard", "results" | Out-Null

# Validate dataset
Write-Host "Validating dataset structure..." -ForegroundColor Yellow
python -m tools.validate_dataset

Write-Host "==================================================================" -ForegroundColor Green
Write-Host " Setup completed successfully!" -ForegroundColor Green
Write-Host "==================================================================" -ForegroundColor Green
