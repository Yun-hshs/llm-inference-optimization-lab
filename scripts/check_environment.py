from __future__ import annotations

import platform
import sys


def main() -> None:
    print("LLM Inference Optimization Lab environment")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")

    try:
        import numpy as np

        print(f"NumPy: {np.__version__}")
    except ImportError:
        print("NumPy: not installed")

    try:
        import torch

        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch: not installed")


if __name__ == "__main__":
    main()
