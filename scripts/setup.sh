#!/usr/bin/env bash
set -e

echo "=== Setting up Adaptive LiDAR Perception & Mapping Environment ==="

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is required but not installed."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Detected Python: $PYTHON_VERSION"

# Create virtual environment if not already present
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment in .venv..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
mkdir -p checkpoints logs/tensorboard results

# Validate dataset
echo "Validating dataset structure..."
python -m tools.validate_dataset

echo "=== Setup completed successfully! ==="
