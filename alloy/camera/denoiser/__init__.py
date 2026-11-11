# SPDX-License-Identifier: Apache-2.0
"""Learned denoiser for 1-4 spp path-traced frames (Cycles denoising-pass conventions).

    import numpy as np
    from alloy.camera.denoiser import load

    dn = load("default")                                    # downloads once; cuda fp16 if available, else cpu fp32
    clean = dn.denoise(rgb, albedo, normal, spp=1)          # (..., H, W, 3) float32 linear in, same shape out
    clean = dn.denoise(rgb, albedo, normal, spp=1, world_to_cam=R, tta=2)  # world-space normals, flip-averaged

``rgb`` is the noisy linear radiance, ``albedo`` the Cycles "Denoising Albedo", ``normal`` the Cycles
"Denoising Normal" in camera space (x right, y up, z forward). Torch tensors or numpy arrays both work.
Weights: ``list_weights()`` shows the manifest. They download once from the Alloy library on the Hugging Face Hub
(public; ``ALLOY_DENOISER_URL`` points at another folder, an empty value turns downloads off) into
``~/.cache/alloy/denoiser``; or pass a local ``.pt`` path to ``load``. Licence and training data: ``WEIGHTS.md``.
"""
from .infer import Denoiser, Ensemble, load
from .model import NetConfig, UNet
from .weights import list_weights, resolve_weights

__all__ = ["Denoiser", "Ensemble", "NetConfig", "UNet", "list_weights", "load", "resolve_weights"]
