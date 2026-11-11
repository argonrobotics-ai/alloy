# SPDX-License-Identifier: Apache-2.0
"""Full-material path tracer (Cycles 4.5 parity): Principled BSDF, shader nodes, Bump and Bevel, area lights, HDRI
world, thin-lens camera. Scenes come from scene packs (Blender's materials as node graphs, compiled to Warp).

Example::

    from alloy.camera.pbr import PbrScene, PbrRenderer
    r = PbrRenderer(PbrScene.from_pack("c3_pack", frame=420), 128, 128)
    rgb, albedo, normal = r.render(spp=16)
"""
from .renderer import PbrRenderer, PbrScene, settings_from_pack

__all__ = ["PbrRenderer", "PbrScene", "settings_from_pack"]
