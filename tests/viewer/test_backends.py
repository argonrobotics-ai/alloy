# SPDX-License-Identifier: Apache-2.0
"""alloy.viewer backends: the installed-or-not table and its pip lines, the light import, unknown names and options,
missing or too-old packages (clear errors naming the pip line), and the gl backend without a display. No scene is
built here; the null backend is made without a model."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import alloy
import alloy.viewer
from alloy.viewer import _backends
from alloy.viewer._backends import BackendOptions, create

REPO = Path(__file__).resolve().parents[2]
PIP = {"viser": 'pip install "viser>=1.1.1"', "gl": "pip install pyglet imgui_bundle", "rerun": "pip install rerun-sdk",
       "usd": "pip install usd-core"}


def test_backends_lists_each_with_its_pip_line():
    table = alloy.viewer.backends()
    assert set(table) == {"viser", "gl", "rerun", "usd", "null"}
    assert table["null"]["installed"] is True
    for name, line in PIP.items():
        assert table[name]["install"] == line
        assert isinstance(table[name]["installed"], bool)
        assert table[name]["about"]


def test_import_loads_no_engine_or_viewer_package():
    """``import alloy.viewer`` (and its __main__) stays light: no Warp, NumPy, Newton or optional viewer package."""
    heavy = ("warp", "numpy", "alloy._vendor.newton", "viser", "pyglet", "imgui_bundle", "rerun", "pxr", "torch")
    code = ("import sys, alloy.viewer, alloy.viewer.__main__; "
            f"print(','.join(m for m in {heavy!r} if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=60,
                         env={**os.environ, "PYTHONPATH": str(REPO)})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ""


def test_unknown_backend_lists_the_valid_names():
    with pytest.raises(alloy.SceneError) as err:
        alloy.viewer.launch(object(), backend="vtk")
    assert err.value.field == "backend"
    for name in ("viser", "gl", "rerun", "usd", "null"):
        assert name in err.value.fix


@pytest.mark.parametrize("name", sorted(PIP))
def test_missing_package_names_the_pip_line(monkeypatch, name):
    monkeypatch.setattr(_backends, "_module_present", lambda module: False)
    assert alloy.viewer.backends()[name]["installed"] is False
    with pytest.raises(alloy.FeatureUnavailableError) as err:
        create(name, BackendOptions(path="out.usda"))
    assert "not installed" in err.value.message
    assert PIP[name] in err.value.fix


def test_old_viser_is_a_clear_error(monkeypatch):
    monkeypatch.setattr(_backends, "_module_present", lambda module: True)
    monkeypatch.setattr(_backends, "_dist_version", lambda dist: "0.2.23")
    assert alloy.viewer.backends()["viser"]["installed"] is False
    with pytest.raises(alloy.FeatureUnavailableError) as err:
        create("viser", BackendOptions())
    assert "0.2.23 is too old" in err.value.message
    assert PIP["viser"] in err.value.fix


def test_import_failure_inside_the_backend_names_the_pip_line(monkeypatch):
    """A package that is found but fails to import (a broken install) still gives the pip line."""
    def broken(nv, options):
        def factory(**kwargs):
            raise ImportError("libGL.so.1: cannot open shared object file")
        return factory, {}

    monkeypatch.setattr(_backends, "_module_present", lambda module: True)
    monkeypatch.setitem(_backends._FACTORIES, "gl", broken)
    with pytest.raises(alloy.FeatureUnavailableError) as err:
        create("gl", BackendOptions())
    assert "libGL" in err.value.message and PIP["gl"] in err.value.fix


@pytest.mark.parametrize(("text", "parts"), [("1.1.1", (1, 1, 1)), ("1.1.1rc2", (1, 1, 1)), ("0.2.23", (0, 2, 23)),
                                             ("2", (2,)), ("dev", ())])
def test_version_parts(text, parts):
    assert _backends._version_tuple(text) == parts


def test_options_split_known_fields_from_backend_keywords():
    opts = BackendOptions.from_kwargs({"port": 8771, "path": "a.rrd", "serve_web_viewer": False}, paused=True)
    assert (opts.port, opts.path, opts.paused) == (8771, "a.rrd", True)
    assert opts.extra == {"serve_web_viewer": False}
    assert _backends._rerun(_Names(), opts)[1] == {"web_port": 8771, "record_to_rrd": "a.rrd", "serve_web_viewer": True}
    quiet = _backends._rerun(_Names(), BackendOptions())[1]
    assert quiet == {"record_to_rrd": "alloy_run.rrd", "serve_web_viewer": False}        # headless-safe default
    assert _backends._viser(_Names(), opts)[1] == {"port": 8771, "paused": True}


def test_usd_needs_a_path():
    with pytest.raises(alloy.SceneError) as err:
        _backends._usd(_Names(), BackendOptions())
    assert err.value.field == "path"


def test_null_backend_and_unknown_option():
    viewer = create("null", BackendOptions(frames=3))
    assert viewer.num_frames == 3 and viewer.is_running()
    viewer.close()
    with pytest.raises(alloy.SceneError) as err:
        create("null", BackendOptions(extra={"bogus": 1}))
    assert "bogus" in err.value.message


def test_rerun_default_records_a_file_and_opens_no_server(tmp_path):
    pytest.importorskip("rerun")
    out = tmp_path / "run.rrd"
    viewer = create("rerun", BackendOptions(path=str(out)))
    assert viewer._grpc_server_uri is None and viewer._viewer_process is None
    viewer.close()
    assert out.exists()


def test_gl_without_a_display_is_a_clear_error(monkeypatch):
    pytest.importorskip("pyglet")
    pytest.importorskip("imgui_bundle")
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        pytest.skip("this machine has a display")
    with pytest.raises(alloy.FeatureUnavailableError) as err:
        create("gl", BackendOptions())
    assert "display" in err.value.message and "viser" in err.value.fix


class _Names:
    """Stands in for the Newton viewer module: every attribute is its own name."""

    def __getattr__(self, name: str) -> str:
        return name
