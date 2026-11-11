# SPDX-License-Identifier: Apache-2.0
"""Tactile sensor on CUDA (OSS-21): the same images and press maps as on CPU, many worlds, torch output."""
from __future__ import annotations

import numpy as np
import pytest

from alloy.camera.tactile import Augment, TactileSensor
from alloy.camera.tactile.bench import bench_model, frame_poses

pytestmark = pytest.mark.gpu


def render_on(device: str, worlds: int = 8):
    import warp as wp
    model, base_q = bench_model(worlds, device)
    state = model.state()
    state.body_q.assign(frame_poses(base_q, 1))
    s = TactileSensor("t", attach_to="pad", pose=((0, 0, 0.002), (0, 0, 0, 1)), width=160,
                      augment=Augment.strong(seed=3))
    out = s.render(model, state, output="all", seed=7)
    wp.synchronize_device(device)
    return out.rgb.numpy().copy(), out.press.numpy().copy(), out.markers.numpy().copy()


def test_cpu_and_gpu_agree():
    cpu, gpu = render_on("cpu"), render_on("cuda:0")
    assert np.abs(cpu[1] - gpu[1]).max() < 1e-4  # press [mm]
    assert np.abs(cpu[2] - gpu[2]).max() < 1e-4  # marker centres [mm]
    diff = np.abs(cpu[0].astype(int) - gpu[0].astype(int))
    # Taxim's table is indexed by the bin a slope falls in: a slope on a bin edge (1e-6 apart) may pick the
    # neighbouring bin, a step of a few levels at a handful of pixels
    assert diff.max() <= 12 and (diff > 0).mean() < 1e-3


def test_many_worlds_and_torch():
    torch = pytest.importorskip("torch")
    model, base_q = bench_model(256, "cuda:0")
    state = model.state()
    state.body_q.assign(frame_poses(base_q, 0))
    s = TactileSensor("t", attach_to="pad", pose=((0, 0, 0.002), (0, 0, 0, 1)), width=160)
    img = s.render_torch(model, state)
    assert img.shape == (256, 120, 160, 3) and img.dtype == torch.uint8 and img.is_cuda
    assert (img.float().std(dim=(1, 2, 3)) > 0).all()
