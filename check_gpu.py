"""GPU / system check printed during setup."""
import sys
print(f"  [INFO] Python {sys.version.split()[0]}")
try:
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        print(f"  [GPU]  {name} ({mem} GB) — CUDA training ready!")
    else:
        print("  [WARN] No CUDA GPU detected — training will run on CPU (slower).")
except ImportError:
    print("  [WARN] PyTorch not importable — check that installation completed.")
