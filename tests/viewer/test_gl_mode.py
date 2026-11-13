# SPDX-License-Identifier: Apache-2.0
"""The desktop ("gl") backend's window mode: following the fix of a failed launch must work or name what is missing.

A window launch with no display used to load pyglet's window code in window mode; the fix it gave ("pass
headless=True") then failed in the same process with the same fix. On a server without the OpenGL libraries the
off-screen launch raised a bare Exception saying pyglet was not installed.
"""

from __future__ import annotations

import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

import alloy
from alloy.api.errors import AlloyAPIError
from alloy.viewer import _backends, _gl_mode
from alloy.viewer._backends import BackendOptions, BackendSpec

REPO = Path(__file__).resolve().parents[2]
GL = BackendSpec("gl", (), "pip install pyglet imgui_bundle", "desktop window")


@pytest.fixture
def no_display(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")


def _fake_pyglet(monkeypatch, display_module: str | None) -> types.ModuleType:
    """A stand-in pyglet in sys.modules; ``display_module`` is where its loaded Display class lives (None: the
    window code is not loaded yet)."""
    pyglet = types.ModuleType("pyglet")
    pyglet.options = {}
    monkeypatch.setitem(sys.modules, "pyglet", pyglet)
    monkeypatch.delitem(sys.modules, "pyglet.display", raising=False)
    if display_module is not None:
        display = types.ModuleType("pyglet.display")
        display.Display = type("Display", (), {"__module__": display_module})
        monkeypatch.setitem(sys.modules, "pyglet.display", display)
    return pyglet


def test_window_launch_without_a_display_fails_before_pyglet_loads(no_display, monkeypatch):
    pyglet = _fake_pyglet(monkeypatch, None)
    with pytest.raises(alloy.FeatureUnavailableError) as err:
        _backends._gl(None, BackendOptions())
    assert "headless=True" in err.value.fix and "viser" in err.value.fix
    assert pyglet.options == {} and _gl_mode.loaded_mode() is None        # nothing fixed for the process
    _backends._gl(types.SimpleNamespace(ViewerGL=object), BackendOptions(headless=True))   # the fix can be followed
    assert pyglet.options["headless"] is True


def test_headless_after_a_window_load_says_new_process_not_headless(no_display, monkeypatch):
    _fake_pyglet(monkeypatch, "pyglet.display.xlib")
    assert _gl_mode.loaded_mode() == "window"
    with pytest.raises(alloy.FeatureUnavailableError) as err:
        _backends._gl(None, BackendOptions(headless=True))
    assert "window mode" in err.value.message
    assert "new Python process" in err.value.fix and "pass headless=True" not in err.value.fix


def test_window_after_a_headless_load_is_refused(monkeypatch):
    monkeypatch.setenv("DISPLAY", ":0")
    _fake_pyglet(monkeypatch, "pyglet.display.headless")
    assert _gl_mode.loaded_mode() == "headless"
    with pytest.raises(alloy.FeatureUnavailableError) as err:
        _backends._gl(None, BackendOptions())
    assert "headless mode" in err.value.message and "new Python process" in err.value.fix


def _raise_missing_library(**kwargs):
    try:
        raise ImportError("Library \"EGL\" not found.")
    except ImportError as exc:
        raise Exception("OpenGLRenderer requires pyglet (version >= 2.0) to be installed.") from exc


def test_missing_opengl_library_is_an_api_error_naming_it(no_display):
    with pytest.raises(alloy.FeatureUnavailableError) as err:
        _backends._construct(GL, _raise_missing_library, {"headless": True})
    assert "EGL" in err.value.message and "OpenGL library" in err.value.message
    assert "libegl1" in err.value.fix and "viser" in err.value.fix


def test_off_screen_failure_never_advises_headless(no_display):
    def boom(**kwargs):
        raise RuntimeError("Cannot connect to \"None\"")

    with pytest.raises(alloy.FeatureUnavailableError) as err:
        _backends._construct(GL, boom, {"headless": True})
    assert "off-screen" in err.value.message and "headless=True" not in err.value.fix


def test_unrelated_constructor_error_is_not_relabelled(monkeypatch):
    monkeypatch.setenv("DISPLAY", ":0")

    def boom(**kwargs):
        raise ValueError("width must be positive")

    with pytest.raises(ValueError):
        _backends._construct(GL, boom, {})


RETRY = """
import os, sys
os.environ.pop("DISPLAY", None); os.environ.pop("WAYLAND_DISPLAY", None)
import alloy, alloy.viewer
from alloy.api.errors import AlloyAPIError, FeatureUnavailableError
s = alloy.Scene(worlds=1, dt=0.01); s.add_ground()
s.add_box("b", size=(0.05, 0.05, 0.05), pos=(0, 0, 0.1), mass=0.1)
sim = alloy.build(s, device="cpu")
try:
    alloy.viewer.launch(sim, backend="gl")
    print("STEP1 opened"); sys.exit(4)
except FeatureUnavailableError as e:
    assert "headless=True" in e.fix, e.fix
try:
    v = alloy.viewer.launch(sim, backend="gl", headless=True, frames=2)
    v.step(1); v.close(); print("STEP2 ok")
except AlloyAPIError as e:
    print("STEP2 api-error", "circular" if "pass headless=True" in e.fix else "clear", "|", e.message, "|", e.fix)
"""


def test_following_the_headless_fix_in_the_same_process():
    """The audit's sequence, in a clean process: window launch fails, then ``headless=True`` draws off screen, or
    raises an API error that names the missing OpenGL and no longer says to pass headless=True."""
    pytest.importorskip("pyglet")
    pytest.importorskip("imgui_bundle")
    if not sys.platform.startswith("linux"):
        pytest.skip("only Linux needs a display for a window")
    out = subprocess.run([sys.executable, "-c", RETRY], cwd=REPO, capture_output=True, text=True, timeout=170,
                         env={**os.environ, "PYTHONPATH": str(REPO)})
    assert out.returncode == 0, out.stdout[-1500:] + out.stderr[-1500:]
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("STEP2")][-1]
    assert line.startswith("STEP2 ok") or line.startswith("STEP2 api-error clear"), line


def test_explain_returns_api_errors_only():
    for exc in (ImportError("x"), RuntimeError("no display")):
        assert isinstance(_gl_mode.explain(exc, {"headless": False}), AlloyAPIError)
