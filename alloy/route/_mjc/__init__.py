# SPDX-License-Identifier: Apache-2.0
"""Alloy's MuJoCo C extension for the CPU route (SCR-WP8-6; owner WP2, WP8 calls it from ``alloy/route/mjc_core``).

C ports of the scoped noslip (``alloy.contact.noslip``) and of the box-box line contacts
(``alloy.contact.mjw_boxbox_line``), plus a tick driver that runs MuJoCo C steps with them, the contact export, the
state pull and the health counters. ABI: ``alloy_mjc.h`` / :mod:`.abi` (agreed with WP8 in [internal note]
notes/wp8-results/cpu-item11/ABI.md).

``load()`` returns the ctypes library, or None when it cannot be built or loaded here; ``reason`` then says why (WP8
falls back to its numpy plug-ins with a one-line warning naming it). ``ALLOY_NO_EXT=1`` switches the extension off
the same way (release CI tests both paths, SCR-OSS1-8).
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys

from .abi import ABI, ERRORS, MjcConfig, MjcOut, declare
from .build import BuildError, library_path, mujoco_paths

log = logging.getLogger("alloy.route")

NO_EXT_ENV = "ALLOY_NO_EXT"            # "1": never build or load the extension (the numpy step, with its warning)
_LIB: ctypes.CDLL | None = None
_TRIED = False
reason = ""


def _share_mujoco() -> None:
    """macOS: the library takes MuJoCo's symbols from the libmujoco ``import mujoco`` loaded (``build.link_flags``);
    reopening it RTLD_GLOBAL (the same file, so the same image) makes them visible to that lookup."""
    lib = mujoco_paths()[2]
    try:
        ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
    except OSError as e:
        raise BuildError(f"cannot share MuJoCo's symbols from {lib}: {e}") from e


def _open() -> ctypes.CDLL:
    import mujoco

    path = library_path()
    if sys.platform == "darwin":
        _share_mujoco()
    try:
        lib = declare(ctypes.CDLL(str(path)))
    except (OSError, AttributeError) as e:
        raise BuildError(f"cannot load {path}: {e}") from e
    if lib.alloy_mjc_abi() != ABI:
        raise BuildError(f"{path} has ABI {lib.alloy_mjc_abi()}, this Alloy needs {ABI}")
    if lib.alloy_mjc_mujoco_version() != mujoco.mj_version():
        raise BuildError(f"{path} was built for MuJoCo {lib.alloy_mjc_mujoco_version()}, "
                         f"the installed one is {mujoco.mj_version()}")
    return lib


def switched_off() -> str:
    """``ALLOY_NO_EXT``'s value when it switches the extension off, else "" (unset, empty, 0, false, no, off)."""
    value = os.environ.get(NO_EXT_ENV, "").strip()
    return "" if value.lower() in ("", "0", "false", "no", "off") else value


def load() -> ctypes.CDLL | None:
    """The extension (built on first use), or None with ``reason`` set."""
    global _LIB, _TRIED, reason
    if not _TRIED:
        _TRIED = True
        off = switched_off()
        if off:
            reason = f"{NO_EXT_ENV}={off} switches the compiled CPU-route extension off"
            log.debug("liballoy_mjc not loaded: %s", reason)
            return None
        try:
            _LIB = _open()
        except BuildError as e:
            reason = str(e)
            log.debug("liballoy_mjc unavailable: %s", reason)
    return _LIB


def require() -> ctypes.CDLL:
    """The extension, or BuildError naming why it is unavailable (tests, scripts)."""
    lib = load()
    if lib is None:
        raise BuildError(reason)
    return lib


__all__ = ["ERRORS", "NO_EXT_ENV", "BuildError", "MjcConfig", "MjcOut", "load", "reason", "require",
           "switched_off"]
