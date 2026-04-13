from __future__ import annotations

import torch

def detect_device(device_arg: str | None = None) -> str:
    """
    Automatically detects whether CUDA is available and returns the appropriate device string.

    Args:
        device_arg (str | None): 
            - "auto" → automatic selection
            - "cuda" → tries to use GPU, falls back to CPU if unavailable
            - "cpu"  → forces CPU
            - None   → equivalent to "auto"

    Returns:
        str: "cuda" or "cpu"
    """
    # automatic device detection
    if not device_arg or device_arg.lower() == "auto":
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(torch.cuda.current_device())
            print(f"⚙️  Auto-detected CUDA device: {device_name}")
            return "cuda"
        else:
            print("⚙️  No CUDA device available → using CPU.")
            return "cpu"

    # CUDA requested but not available
    if device_arg.lower() == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA requested but not available → fallback to CPU.")
        return "cpu"

    # use manually specified device
    print(f"⚙️  Using manually specified device: {device_arg}")
    return device_arg.lower()
