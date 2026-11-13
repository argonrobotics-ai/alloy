# SPDX-License-Identifier: Apache-2.0
"""The desktop ("gl") backend's window mode: a real window needs a display, ``headless=True`` draws off screen (EGL).
pyglet fixes the mode for the whole process the first time its window code loads, so the checks here run before that
load and turn each way it can fail into a ``FeatureUnavailableError`` whose fix is one the user can still follow.

Standard library only at import time; pyglet loads inside ``prepare``.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from alloy.api.errors import FeatureUnavailableError

VISER_FIX = "use backend='viser' and open view.url in a browser"
LIBS_FIX = ("install the system OpenGL / EGL libraries (Debian or Ubuntu: `sudo apt install libegl1 libgl1 "
            "libglu1-mesa`; on an NVIDIA GPU machine also the driver's GL / EGL package), or " + VISER_FIX)
NEW_PROCESS = "start a new Python process and make this the first gl launch in it"


def has_display() -> bool:
    """True when a window can be asked for: a display is set, or the platform needs none (Windows, macOS)."""
    if not sys.platform.startswith("linux"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def loaded_mode() -> str | None:
    """The mode pyglet's window code is already loaded in, in this process: "headless", "window", or None."""
    display = sys.modules.get("pyglet.display")
    cls = getattr(display, "Display", None)
    if cls is None:
        return None
    return "headless" if "headless" in str(getattr(cls, "__module__", "")) else "window"


def prepare(headless: bool) -> None:
    """Check that a gl viewer can start in the asked mode, then set pyglet's mode. Raises before pyglet's window
    code loads, so a failed window launch does not stop a later ``headless=True`` launch in the same process."""
    mode = loaded_mode()
    asked = "headless" if headless else "window"
    if mode is not None and mode != asked:
        raise FeatureUnavailableError(
            f"the gl viewer backend was already loaded in {mode} mode in this process and cannot switch to "
            f"{asked} mode", f"{NEW_PROCESS}, or {VISER_FIX}", field="backend")
    if not headless and not has_display():
        raise FeatureUnavailableError(
            "the gl viewer backend needs a display and found none (DISPLAY is not set)",
            f"run it in a desktop session (DISPLAY set), pass headless=True to draw off screen, or {VISER_FIX}",
            field="backend")
    import pyglet  # noqa: PLC0415 - optional package; the option must be set before pyglet's window code loads

    pyglet.options["headless"] = bool(headless)


def explain(exc: BaseException, kwargs: dict[str, Any]) -> FeatureUnavailableError | None:
    """The API error for a failure of the gl viewer's constructor, or None for one that is not about the display or
    the OpenGL libraries (it is then raised as it is)."""
    headless = bool(kwargs.get("headless"))
    text = str(exc)
    cause = exc.__cause__
    if isinstance(exc, ImportError) or isinstance(cause, ImportError):
        return FeatureUnavailableError(
            f"the gl viewer backend could not load an OpenGL library ({cause or exc})",
            f"{LIBS_FIX}; if a Python package is broken instead, reinstall it: pip install pyglet imgui_bundle",
            field="backend")
    if headless:
        return FeatureUnavailableError(
            f"the gl viewer backend could not start off-screen OpenGL (EGL) on this machine ({text})", LIBS_FIX,
            field="backend")
    if type(exc).__name__ == "NoSuchDisplayException" or "display" in text.lower() or not has_display():
        return FeatureUnavailableError(
            f"the gl viewer backend could not open a window on this display ({text})",
            f"run it in a desktop session with a working display, or {VISER_FIX}; to draw off screen, "
            f"{NEW_PROCESS} with headless=True", field="backend")
    return None


__all__ = ["explain", "has_display", "loaded_mode", "prepare"]
