# SPDX-License-Identifier: Apache-2.0
"""The camera package: lazy exports (subpackages import without the renderer) and the backend switch."""
from __future__ import annotations

import subprocess
import sys

import pytest

import alloy.camera as cam_pkg
from alloy.camera import Camera, CameraError, RenderSettings


def test_denoiser_subpackage_does_not_import_the_renderer():
    code = ("import sys, alloy.camera.denoiser; "
            "print(any(m in sys.modules for m in ('alloy.camera.camera', 'alloy.camera.tracer')))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "False"


def test_every_export_resolves_and_unknown_names_say_what_exists():
    for name in cam_pkg.__all__:
        assert getattr(cam_pkg, name) is not None
    with pytest.raises(AttributeError, match="public names"):
        cam_pkg.NoSuchThing  # noqa: B018


def test_tactile_sensor_is_found_next_to_the_camera():
    import alloy.camera.tactile as tactile
    assert cam_pkg.TactileSensor is tactile.TactileSensor and cam_pkg.TactileRenderer is tactile.TactileRenderer
    assert "TactileSensor" in cam_pkg.__all__


def test_backend_comes_from_the_environment_or_the_camera(monkeypatch):
    monkeypatch.setenv("ALLOY_CAMERA_BACKEND", "auto")
    assert RenderSettings().backend == "auto"
    monkeypatch.delenv("ALLOY_CAMERA_BACKEND")
    assert RenderSettings().backend == "warp"
    st = Camera("c", backend="optix")._with_backend(RenderSettings(seed=3))
    assert (st.backend, st.seed) == ("optix", 3)
    assert Camera("c")._with_backend(st) is st


def test_unknown_backend_is_refused_on_first_render():
    from tests.camera.helpers import plane_scene
    with pytest.raises(CameraError, match="unknown render backend"):
        Camera("c", width=4, height=4, spp=1, backend="vulkan").render(plane_scene())
