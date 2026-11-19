# SPDX-License-Identifier: Apache-2.0
"""Fixture parts give the MPM voxel rule their fixture's inner width, up to what their wall holds (from 0.1.8).

The router sizes an MPM object's voxel as a quarter of the narrowest feature the material can meet
(``alloy.route.phase.feature_size``), and takes a container part's middle extent as that container's inner size. A
fixture (``add_open_box``, ``add_open_cylinder``, ``add_block_with_hole``) is built from boxes, and a box's middle
extent is not the cavity. Each of ``add_open_cylinder``'s 24 wall slabs is 4 mm across by default, so through 0.1.7 a
12 mm cube of sand in a vial of 10 mm inner radius got a 1 mm voxel and 13,824 particles, where its own size gives
3 mm and 512. A kinematic fixture's parts were no container at all, so sand tilted in one got its own size's voxel
whatever the wall.

``with_cavities`` wraps the build's router so that every fixture part, static or kinematic, reports its fixture's inner
width (the narrower one: the scene item's ``opening``, which the liquid-accuracy check also reads) and its own wall
thickness in ``SceneContext.cavities``. The router then takes ``part_size``: the part's own middle extent, or where that
is smaller, the fixture's width up to 4/3 of the wall, so the voxel stays within a third of the wall there
(``alloy.route.phase``; the granular regime and jamming check read the fixture's width itself). Thinner walls let sand
through, and health does not flag it (docs/known_gaps/mpm-thin-walls.md). Counting every grain that ever crossed a
part's far face, in the default vial with its 12 mm cube (CPU): with gravity turned sideways, 16 of 512 grains crossed
4 mm walls at a 3 mm voxel, 14 of 1,728 at 2 mm, none of 5,832 at 1.33 mm; turned 90 degrees outward about its lip, 2,
0 and 0 (0-1 at 1.33 mm when tipped to pour). A 20 mm cube turned outward in a vial of 15 mm radius at its own 5 mm
voxel had 75 of 512 cross (1-2 tipped to pour), none at 1.33 mm. The default vial now gets a 1.33 mm voxel and
5,832 particles, and a 9 mm wall gives 512. ``ALLOY_MPM_CAVITY=off`` keeps the 0.1.7 rule (each static part's own size,
kinematic parts not containers), for A/B runs and the before rows.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

log = logging.getLogger("alloy.integration")

__all__ = ["ENV", "WALL_VOXELS", "CavityRouter", "cavities", "enabled", "part_size", "with_cavities"]

ENV = "ALLOY_MPM_CAVITY"
WALL_VOXELS = 3.0           # a fixture wall holds MPM material when at least this many voxels thick (module docstring)


def enabled() -> bool:
    """Whether fixture parts report their fixture's inner width (default yes; ``ALLOY_MPM_CAVITY=off`` turns it off).
    Example: ``enabled()`` -> True."""
    return os.environ.get(ENV, "on").strip().lower() != "off"


def part_size(cavity: float, thickness: float, own: float) -> float:
    """The size a fixture part gives the MPM voxel rule (voxel = a quarter of the narrowest size): the larger of its
    own middle extent ``own`` (what a static part gave through 0.1.7) and its fixture's inner width ``cavity`` held
    to ``4 / WALL_VOXELS`` times its wall ``thickness``. Where the wall term wins, the voxel stays within a third of
    the wall; a static part never asks for a finer voxel than through 0.1.7. Example:
    ``part_size(0.02, 0.004, 0.004)`` -> 0.00533 (a 1.33 mm voxel where 0.1.7 took 1 mm)."""
    held = round(min(float(cavity), 4.0 * float(thickness) / WALL_VOXELS), 9)    # to the nm, so equal walls tie
    return max(float(own), held)


def cavities(items: Mapping[str, Mapping[str, Any]]) -> dict[str, tuple[float, float]]:
    """{part: (its fixture's narrower inner width, the part's wall thickness), m} for every fixture part among a
    Scene's ``items``. The thickness is the part's width across the line from its fixture's axis (the floor's centre),
    the floor's its height: a round cup's slab gives its wall, not its narrower width along the rim.
    Example: ``cavities(scene.items)["vial_wall0"]`` -> ``(0.02, 0.004)``."""
    out = {}
    for n, it in items.items():
        fixture, opening = it.get("fixture"), it.get("opening")
        floor = items.get(f"{fixture}_floor")
        if fixture and opening and floor is not None:
            out[str(n)] = (float(opening), round(_thickness(it, floor["pose"]), 9))     # to the nm: equal walls tie
    return out


def _thickness(item: Mapping[str, Any], floor_pose: Any) -> float:
    """A fixture part's (a box turned about z only) width along the horizontal line from its fixture's axis through
    its centre; on the axis (the floor), its height."""
    half = [float(v) for v in item["shape"]["primitive"]["size"]]
    x, y, _, _, _, qz, qw = (float(v) for v in item["pose"])
    dx, dy = x - float(floor_pose[0]), y - float(floor_pose[1])
    r = math.hypot(dx, dy)
    if r < 1e-9:
        return 2.0 * half[2]
    yaw = 2.0 * math.atan2(qz, qw)
    ux = (dx * math.cos(yaw) + dy * math.sin(yaw)) / r              # the line in the part's own frame
    uy = (-dx * math.sin(yaw) + dy * math.cos(yaw)) / r
    return 2.0 * (abs(ux) * half[0] + abs(uy) * half[1])


class CavityRouter:
    """A router that routes every object with ``cavities`` added to its ``SceneContext``; every other call goes to
    the wrapped router. Example: ``CavityRouter(router, {"vial_wall0": (0.02, 0.004)})``."""

    def __init__(self, router: Any, parts: Mapping[str, tuple[float, float]]):
        self.router = router
        self.cavities = dict(parts)                # not .parts: FixtureRouter, which it may wrap, has one

    def route(self, obj: Any, ctx: Any) -> Any:
        return self.router.route(obj, replace(ctx, cavities={**ctx.cavities, **self.cavities}))

    def __getattr__(self, name: str) -> Any:
        return getattr(self.router, name)


def with_cavities(router: Any, items: Mapping[str, Mapping[str, Any]]) -> Any:
    """``router`` itself, or a ``CavityRouter`` around it when the scene has fixture parts and ``enabled()``.
    Example: ``router = with_cavities(router, scene.items)``."""
    parts = cavities(items) if enabled() else {}
    if not parts:
        return router
    log.debug("mpm_cavity: %d fixture parts report their fixture's inner width", len(parts))
    return CavityRouter(router, parts)
