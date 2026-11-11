# SPDX-License-Identifier: Apache-2.0
"""MPM material with a body above 64 GPU worlds: one build-time warning, because that range is not validated
(physics rulings 2026-11-11).

Known gap docs/known_gaps/mpm-moving-container-many-worlds.md, fixed for 64 worlds and below. At many GPU worlds MPM
grains sank into bodies they touched (a lifted box's floor, a kinematic tool or press, a robot's finger, a free box
resting on sand), because warp.fem's packed-grid cell lookup worked in float32. ``nanogrid_lookup`` now does it in
integers, and on one L4 every 64-world row read as one world. Above 64 worlds, an MPM entry with a body ran out of
Newton's sparse-grid capacity ("sparse grid rebuild failed with status 8") or of memory, with the fix as without it,
and once ended in CUDA error 700, so the fix's law could not run there.

``warn`` keys on what the build knows:
- MPM material (an entry with particles);
- a body that is not welded to the world, anywhere in the scene: free, kinematic, or a link of an articulation, with or
  without a servo. Static shapes, the ground included, are not bodies and do not count;
- a CUDA device;
- more than ``WORLDS`` worlds.
It warns once per build (``RuntimeWarning``, also logged and returned as a build note) and never refuses.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import warp as wp

log = logging.getLogger("alloy.integration")

__all__ = ["PAGE", "WORLDS", "moving_bodies", "warn"]

WORLDS = 64                             # the fix's law ran up to here on one L4 (the known gap's rows)
PAGE = "docs/known_gaps/mpm-moving-container-many-worlds.md"


def moving_bodies(built: Any) -> list[str]:
    """Names of the bodies of ``built`` that are not welded to the world (each once, in body order): free,
    kinematic, or articulation links. Example: ``moving_bodies(built) == ["box"]`` for a box resting on sand."""
    from alloy.integration.mpm_robots import _actuated, _kind  # noqa: PLC0415

    m = built.model
    n = int(m.body_count or 0)
    if not n:
        return []
    actuated = _actuated(m)
    joints = {int(c): j for j, c in enumerate(m.joint_child.numpy()) if c >= 0}
    labels = list(getattr(m, "body_label", None) or [])
    out: list[str] = []
    for b in range(n):
        if _kind(m, b, joints, actuated) == "welded":
            continue
        name = labels[b].rsplit("/", 1)[-1] if b < len(labels) else f"body {b}"   # the model's label's last part
        if name not in out:
            out.append(name)
    return out


def _is_cuda(built: Any) -> bool:
    return bool(wp.get_device(built.model.device).is_cuda)


def warn(built: Any) -> list[str]:
    """The warning (module doc) when ``built`` is a GPU scene of more than ``WORLDS`` worlds with MPM material and a
    body that can move; its text as a one-item note list, else []. Example: ``notes += warn(built)``."""
    worlds = int(built.world_count)
    if worlds <= WORLDS or not any(e.capability == "mpm" and len(e.particles) for e in built.entries):
        return []
    if not _is_cuda(built):
        return []
    names = moving_bodies(built)
    if not names:
        return []
    shown = ", ".join(names[:3]) + (f" and {len(names) - 3} more" if len(names) > 3 else "")
    text = (f"{worlds} GPU worlds of sand, soil or paste (MPM) with a body ({shown}): above {WORLDS} worlds MPM with "
            "a body is not validated: Newton's sparse grid runs out of capacity (status 8) or memory. Build "
            f"{WORLDS} worlds or fewer per GPU, or check the lowest grain against the floor or tool yourself "
            f"(sim.check() does not flag a sunk grain) ({PAGE})")
    warnings.warn(text, RuntimeWarning, stacklevel=2)
    log.warning("%s", text)
    return [text]
