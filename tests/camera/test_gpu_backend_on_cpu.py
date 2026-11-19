# SPDX-License-Identifier: Apache-2.0
"""A GPU-only render backend asked for by name ("wavefront-warp", "optix") on the CPU raises a CameraError that says
it needs an NVIDIA GPU and how to render instead, before anything is allocated; it does not fall back to "warp"."""
from __future__ import annotations

import pytest

from alloy.camera.errors import CameraError
from alloy.camera.wavefront.tracers import resolve_backend


@pytest.fixture
def cpu_sim():
    """A fresh sim per test: a sim keeps the render settings of its first full-quality render."""
    import alloy
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_box("block", size=(0.1, 0.1, 0.1), pos=(0, 0, 0.05), mass=0.2)
    return alloy.build(scene, device="cpu")


def _says_what_to_do(err: CameraError, backend: str) -> None:
    assert f"render backend '{backend}' needs an NVIDIA GPU" in str(err), str(err)
    assert "ALLOY_CAMERA_BACKEND" in err.fix and "backend='warp'" in err.fix, err.fix


@pytest.mark.parametrize("backend", ["wavefront-warp", "optix"])
def test_gpu_backend_on_cpu_is_a_camera_error(backend):
    with pytest.raises(CameraError) as err:
        resolve_backend(backend, "cpu")
    _says_what_to_do(err.value, backend)
    assert resolve_backend("warp", "cpu") == "warp" and resolve_backend("auto", "cpu") == "warp"


@pytest.mark.parametrize("backend", ["wavefront-warp", "optix"])
def test_full_render_with_gpu_backend_variable_on_cpu(cpu_sim, monkeypatch, backend):
    """sim.render(quality="full") under ALLOY_CAMERA_BACKEND=<GPU backend>: before, a bare RuntimeError from a
    pinned host allocation ("Failed to allocate 16 bytes on device 'cpu'") on a machine without a CUDA driver. The
    render names the CPU: a snapshot renders on the GPU when there is one (a CPU sim's too), where they can run."""
    monkeypatch.setenv("ALLOY_CAMERA_BACKEND", backend)
    with pytest.raises(CameraError) as err:
        cpu_sim.render(width=32, height=24, spp=1, quality="full", device="cpu")
    _says_what_to_do(err.value, backend)


def test_fast_camera_with_gpu_backend_variable_on_cpu(cpu_sim, monkeypatch):
    from alloy.camera import Camera
    monkeypatch.setenv("ALLOY_CAMERA_BACKEND", "wavefront-warp")
    cam = Camera("c", look_at=((0.5, -0.5, 0.4), (0, 0, 0.05)), width=32, height=24, spp=1)
    with pytest.raises(CameraError) as err:
        cam.render(cpu_sim)
    _says_what_to_do(err.value, "wavefront-warp")
