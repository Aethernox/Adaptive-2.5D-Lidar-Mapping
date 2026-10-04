import sys
import platform
import os
import psutil

print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"CPU Count: {os.cpu_count()}")
try:
    ram_gb = psutil.virtual_memory().total / (1024**3)
    print(f"System RAM: {ram_gb:.2f} GB")
except Exception as e:
    print(f"RAM check error: {e}")

try:
    import torch
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Device VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
except ImportError:
    print("PyTorch not installed")

for pkg in ["numpy", "scipy", "yaml", "open3d", "torchvision", "fastapi", "uvicorn", "websockets"]:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "installed")
        print(f"{pkg}: {ver}")
    except ImportError:
        print(f"{pkg}: NOT installed")
