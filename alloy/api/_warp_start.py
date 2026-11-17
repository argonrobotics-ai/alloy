# SPDX-License-Identifier: Apache-2.0
"""Starting Warp quietly: ``quiet_warp()`` before Alloy's first Warp call (``alloy.build``, CPU workers, the CLI).

Warp prints a start-up banner and a line per kernel module it loads; ``wp.config.quiet`` turns those off. On a machine
with no NVIDIA driver its native start-up also writes "Warp CUDA warning: Could not find or load the NVIDIA CUDA driver.
GPU execution will not be available." to stderr, once per Python process, cold and warm (0.1.6 from PyPI on a CPU
machine: at the first ``alloy.build`` and ``alloy.envs.make``). No Warp setting skips that, and Python's warning
filters cannot reach a C ``fprintf``, so when the driver library cannot be loaded here (``libcuda.so`` /
``nvcuda.dll``) ``wp.init()`` runs with the process's stderr pointed at the null device: Alloy then runs on the CPU,
which ``alloy.build`` says where it matters (``_cpu_note``). With a driver, or with ``ALLOY_VERBOSE=1``, nothing is
muted.

Example: ``quiet_warp()``, then any Warp call.
"""

from __future__ import annotations

import ctypes
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager

_LIBRARIES = {"linux": ("libcuda.so", "libcuda.so.1"), "win32": ("nvcuda.dll",)}
_started = [False]


def verbose() -> bool:
    """``ALLOY_VERBOSE`` set to anything but "" or "0". Example: ``verbose()``."""
    return os.environ.get("ALLOY_VERBOSE", "") not in ("", "0")


def quiet_warp() -> None:
    """Silence Warp's banner and per-module load lines unless ``ALLOY_VERBOSE=1``, and start Warp, without its
    no-driver line on a machine with no NVIDIA driver (module docstring). Example: ``quiet_warp()``."""
    import warp as wp  # noqa: PLC0415

    if not verbose():
        wp.config.quiet = True
        if hasattr(wp, "LOG_WARNING"):
            wp.config.log_level = wp.LOG_WARNING
    if _started[0]:
        return
    if verbose() or driver_present():
        wp.init()
    else:
        with _stderr_muted():
            wp.init()
    _started[0] = True


def driver_present(platform: str | None = None) -> bool:
    """Whether the NVIDIA driver library Warp looks for loads here (True where Warp has no CUDA to look for, as on
    macOS). Example: ``driver_present()``."""
    names = _LIBRARIES.get(platform or sys.platform, ())
    if not names:
        return True
    for name in names:
        try:
            ctypes.CDLL(name)
        except OSError:
            continue
        return True
    return False


@contextmanager
def _stderr_muted() -> Iterator[None]:
    """File descriptor 2 on the null device for the block (C writes too), restored after; a no-op without one."""
    try:
        sys.stderr.flush()
        saved = os.dup(2)
    except (OSError, ValueError, AttributeError):
        yield
        return
    null = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(null)
