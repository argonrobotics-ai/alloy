# SPDX-License-Identifier: Apache-2.0
"""The learned denoiser hands PyTorch's working memory back after each render, so Warp (the next sim.step, a bigger
batch) can use it."""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

MIB = 2 ** 20


def test_denoised_colour_is_a_warp_array_and_the_cache_is_released(monkeypatch):
    """CPU proxy of the GPU law below: chunked output lands in one Warp-owned array (no torch tensor kept), and the
    cache release runs once per call, after every chunk, on the denoised images' device (a Warp array that
    Warp frees has a deleter; one wrapping a torch tensor has none)."""
    torch = pytest.importorskip("torch")
    import alloy.camera.denoise as D

    released = []

    class Halve:
        def denoise(self, rgb, albedo, normal, spp, tta=1):
            return (rgb * 0.5).half()  # the model's fp16 output, as on a GPU

    dn = D.LearnedDenoiser.__new__(D.LearnedDenoiser)
    dn.model, dn.name, dn.tta = Halve(), "stub", 1
    monkeypatch.setattr(D, "chunk_pixels", lambda device: 2 * 4 * 4)
    monkeypatch.setattr(D, "release_torch_cache", lambda device: released.append(str(device)))
    rgb = wp.array(np.arange(5 * 4 * 4 * 3, dtype=np.float32).reshape(5, 4, 4, 3) / 64, dtype=wp.vec3, device="cpu")
    out = dn(rgb, rgb, rgb, 4)
    assert isinstance(out, wp.array) and out.deleter is not None and out.dtype == wp.vec3 and out.shape == rgb.shape
    assert np.allclose(out.numpy(), rgb.numpy() * 0.5, atol=1e-3) and released == ["cpu"]
    D.release_torch_cache(torch.device("cpu"))  # the real one: nothing to release off the GPU


@pytest.mark.gpu
def test_learned_render_leaves_no_torch_cache_behind():
    """After cam.render(denoise="learned"), PyTorch holds no unused GPU memory and no image: before, the image and
    the denoiser's working memory stayed in PyTorch's cache (10.2 GB of an L4's 22 GB after 1,024 images of
    311 x 175 px), where the next sim.step could not allocate it."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("PyTorch has no CUDA")
    import alloy
    from alloy.camera import Camera

    scene = alloy.Scene(worlds=64, dt=0.01)
    scene.add_ground()
    scene.add_box("block", size=(0.1, 0.1, 0.1), pos=(0, 0, 0.05), mass=0.2)
    sim = alloy.build(scene, device="cuda:0")
    cam = Camera("dn", look_at=((0.6, -0.6, 0.4), (0, 0, 0.05)), width=256, height=256, spp=4, denoise="learned")
    cam.render(sim)  # loads the weights
    torch.cuda.synchronize()
    weights = torch.cuda.memory_allocated()
    out = cam.render(sim, output="all")
    wp.synchronize()
    torch.cuda.synchronize()
    held = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    assert torch.cuda.memory_allocated() == weights, "the denoised colour should live in Warp, not in PyTorch"
    assert held < 32 * MIB, f"PyTorch still caches {held / MIB:.0f} MiB after the render"
    lin = wp.to_torch(out.linear)
    assert lin.shape == (64, 256, 256, 3) and torch.isfinite(lin).all() and lin.mean() > 0.01
    ref = cam._denoiser.model.denoise(*(wp.to_torch(a) for a in (out.noisy, out.albedo, out.normal)), cam.spp)
    assert torch.allclose(lin, ref.float(), atol=1e-3), "the image should be the model's own output, unchanged"
