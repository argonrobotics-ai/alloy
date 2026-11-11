# SPDX-License-Identifier: Apache-2.0
"""PyTorch, or a stand-in when it is not installed.

The denoiser modules import ``torch``, ``nn`` and ``F`` from here. Without PyTorch they still import (so torch-free
installs and tools that import every module work), and the first real use raises an ImportError that says what to
install. Example: ``from alloy.camera.denoiser._torch import AVAILABLE`` tells whether the denoiser can run.
"""
from __future__ import annotations

import types
from typing import Any

MESSAGE = ("alloy.camera.denoiser needs PyTorch: pip install torch (the CPU build runs it; a CUDA build is "
           "much faster). Everything else in alloy works without it.")

try:
    import torch
    import torch.nn.functional as F
    from torch import nn

    AVAILABLE = True
except ImportError:
    AVAILABLE = False

    class _Missing:
        """Stands in for any torch class: subclassing works, constructing raises ImportError(MESSAGE)."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(MESSAGE)

    def _decorator(*args: Any, **kwargs: Any) -> Any:
        """``torch.inference_mode()`` / ``torch.no_grad()`` without torch: leaves the function unchanged."""
        return lambda fn: fn

    class _Stub(types.ModuleType):
        def __getattr__(self, name: str) -> Any:
            if name.startswith("__"):
                raise AttributeError(name)
            return _decorator if name in ("inference_mode", "no_grad") else _Missing

    torch = _Stub("torch")
    nn = _Stub("torch.nn")
    F = _Stub("torch.nn.functional")

__all__ = ["AVAILABLE", "MESSAGE", "F", "nn", "torch"]
