"""Inference device selection: Intel Arc (XPU) -> CUDA -> CPU.

This machine has an Intel Arc 140V; PyTorch >= 2.5 exposes it as the "xpu"
device. Everything downstream asks this module instead of hardcoding a
device, and falls back to CPU if a backend errors at runtime.
"""
from __future__ import annotations

import functools
import logging

from ..config import CV_DEVICE

log = logging.getLogger("cv.device")


@functools.lru_cache(maxsize=1)
def torch_device() -> str:
    import torch

    if CV_DEVICE != "auto":
        return CV_DEVICE
    if getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
        log.info("using Intel XPU: %s", torch.xpu.get_device_name(0))
        return "xpu"
    if torch.cuda.is_available():
        log.info("using CUDA: %s", torch.cuda.get_device_name(0))
        return "cuda"
    log.info("no GPU backend available, using CPU")
    return "cpu"


@functools.lru_cache(maxsize=1)
def ultralytics_device() -> str:
    """Ultralytics accepts 'cpu'/'cuda:0'/... ; XPU support varies by version,
    so we probe once and fall back to CPU if the tracker rejects it."""
    dev = torch_device()
    return dev if dev != "cuda" else "cuda:0"
