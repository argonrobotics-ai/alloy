# SPDX-License-Identifier: Apache-2.0
"""``alloy.vary``: scene variants for visual and placement diversity (the engine side is ``alloy.scene.variants``)."""

from __future__ import annotations

from typing import Any


def vary(scene: Any, n: int, pos: float = 0.005, yaw_deg: float = 0.0, textures: str = "same-family", seed: int = 0,
         **options: Any) -> list[Any]:
    """``n`` edited copies of ``scene`` (each its own ``alloy.Scene``), the same for the same ``seed``. Not
    ``Scene.randomize``, which draws values per world inside one simulation.

    - Placement: every free object's start x / y moves within +-``pos`` m (and its yaw within +-``yaw_deg``);
      the ground, static, kinematic and held objects, robots, fixtures and liquids stay. Objects that touch at the
      start move together; a draw that overlaps something or takes an object off what it rests on is redrawn.
    - Looks: each object's look takes another texture of its family (wood -> wood, fabric -> fabric) and a small
      colour tint. ``textures``: "same-family", "textured" (only looks that already have a texture) or "keep".
    - Keywords: ``tint`` (relative colour jitter, default 0.05; 0: none), ``out`` (a folder for
      ``variant_000.json`` ... and ``variants.json``, what each variant moved and swapped), ``contact_sheet=True``
      (also ``contact_sheet.png``; ``sheet={"width", "height", "spp", "quality": "full" | "fast"}``),
      ``settle`` (s: build each variant on the CPU and run ``alloy.check``, redrawing until it passes) and
      ``max_tries`` (draws per variant before a ``SceneError``, default 200).

    Example::

        scenes = alloy.vary(scene, 8, pos=0.01, yaw_deg=10, seed=3, out="variants", contact_sheet=True)
        sims = [alloy.build(s) for s in scenes[:2]]"""
    from alloy.scene.variants import vary as engine_vary  # noqa: PLC0415 - NumPy and the card library on first use

    return engine_vary(scene, n, pos, yaw_deg, textures, seed, **options)
