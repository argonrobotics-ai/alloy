# SPDX-License-Identifier: Apache-2.0
"""VBD self-contact reach from a motion bound (coordinator ruling 2026-11-08 08:40Z (b), owner WP11).

Newton's penetration-free truncation (DAT) caps every particle's motion between self-contact detections at
0.5·γ·(margin + gap) (γ = ``dat_conservative_bound_relaxation``, 0.85). A step detects twice: the inertial guess is
truncated against the pre-step detection, the iterations against the post-initialization one. So nothing in a
self-contact entry moves faster than

    v_cap = γ (margin + gap) / dt.

The route used gap = margin = the sheet thickness h. That capped a 2 mm sheet at 2.72 m/s and the 0.6 mm towel card
at 0.82 m/s at the 1.25 ms step: a dropped towel fell at a constant 0.82 m/s ([internal note] notes/wp11-sc/). The margin
stays physical (the thickness: where two layers start to push); the gap, a detection-only slack, comes from the motion
the entry must allow:

    gap = max(0, min(speed · dt / γ, REACH_CELLS · ℓ) − margin)

where ℓ is the median rest edge length of the entry's triangles. The cap keeps the reach under a cell: on a fine mesh a
reach of several cells lists tens of thousands of pairs per surface and overflows Newton's fixed per-vertex pair
buffers (32 / 64), and overflowing pairs are dropped silently. SOFT-3's skin and fat pads at a 7.35 mm reach:
1,890 vertices and 4,527 edges over capacity, and the skin dipped 1.02 mm into the fat.

Measured (notes/wp11-sc/, on the old membrane map; CPU laundry rows on one VM, drape on an L4 at 64 worlds):
- 0.6 mm towel, ~6 mm cells, 1.25 ms: at 3 m/s (reach 4.4 mm, 0.73 of a cell) it falls to 3.0 m/s, where the old
  reach held it at 0.82. 0 crossings in the half/thirds folds, the fast and C7 pinches; the half fold matches the old
  quality (hem 2.1 mm). It costs 1.5-2.8x per step. At 5 m/s (1.2 cells) the fast pinch ran 9x slower and the half
  fold failed (hem 20 mm).
- 4 mm drape sheet, 8 mm cells, 1/960 s: the reach falls to the margin (gap 0), 3.35x faster, the same drape
  strain, 0 crossings in 64 worlds.
- Two topological rings instead of one let the squeezed ridge cross once the reach grows (fast pinch: 9 crossings at
  5 m/s, 3 at 3 m/s), so the route keeps one ring.

With ``alloy_self_contact_speed`` in its kwargs, ``make_vbd`` mixes `ReachMixin` in front of the solver class. It sets
the gap from each step's dt before Newton detects, so a coupled entry stepping finer than its target dt also queries
less. A captured CUDA graph replays one dt, so its reach is fixed with it. An explicit ``particle_self_contact_gap``
wins: ``make_vbd`` then drops the rule.
"""

from __future__ import annotations

import logging
import math

import numpy as np

log = logging.getLogger("alloy.solvers.vbd")

SPEED_KEY = "alloy_self_contact_speed"
SELF_CONTACT_SPEED = 3.0       # [m/s] the route's motion bound (module docstring: 5 m/s cost 9x on the towel pinch)
REACH_CELLS = 0.75             # the reach's cap, in median rest edge lengths


def reach_gap(margin: float, dt: float, speed: float, relaxation: float, cap: float = math.inf) -> float:
    """Detection slack [m] that lets a particle move at ``speed`` for one step without truncation, with the reach
    (margin + gap) at most ``cap``."""
    return max(0.0, min(speed * dt / relaxation, cap) - margin)


def median_edge(model) -> float:
    """Median rest length [m] of the model's triangle edges (inf without triangles)."""
    if not model.tri_count:
        return math.inf
    q, t = model.particle_q.numpy(), model.tri_indices.numpy()
    e = np.concatenate([q[t[:, 1]] - q[t[:, 0]], q[t[:, 2]] - q[t[:, 1]], q[t[:, 0]] - q[t[:, 2]]])
    return float(np.median(np.linalg.norm(e, axis=1)))


def speed_cap(margin: float, gap: float, dt: float, relaxation: float) -> float:
    """The fastest a self-contact particle can move [m/s] at this margin, gap and step."""
    return relaxation * (margin + gap) / dt


class ReachMixin:
    """Sets Newton's self-contact gap from each step's dt (module docstring). Mixed in front of a SolverVBD class."""

    def __init__(self, model, *args, alloy_self_contact_speed: float, **kwargs) -> None:
        super().__init__(model, *args, **kwargs)
        self.alloy_self_contact_speed = float(alloy_self_contact_speed)
        self.reach_cap = REACH_CELLS * median_edge(model)
        self._reach_dt: float | None = None

    def step(self, state_in, state_out, control, contacts, dt):
        if self.particle_enable_self_contact and dt != self._reach_dt:
            self._set_reach(float(dt))
        return super().step(state_in, state_out, control, contacts, dt)

    def _set_reach(self, dt: float) -> None:
        margin, gamma = float(self.particle_self_contact_margin), float(self.dat_conservative_bound_relaxation)
        gap = reach_gap(margin, dt, self.alloy_self_contact_speed, gamma, self.reach_cap)
        self.particle_self_contact_gap = gap
        self._self_contact_query_radius = margin + gap
        pipeline = getattr(self, "collision_pipeline", None)       # a solver-owned pipeline detects with its own copy
        if pipeline is not None and hasattr(pipeline, "soft_self_contact_gap"):
            pipeline.soft_self_contact_gap = gap
        if self._reach_dt is None:
            log.info("VBD self contact: margin %.3g m, gap %.3g m at dt %.3g s (motion bound %.3g m/s, reach cap %.3g m:"
                     " moves up to %.3g m/s)", margin, gap, dt, self.alloy_self_contact_speed, self.reach_cap,
                     speed_cap(margin, gap, dt, gamma))
        self._reach_dt = dt


def with_reach(cls: type) -> type:
    """``cls`` with `ReachMixin` in front."""
    return type(f"{cls.__name__}Reach", (ReachMixin, cls), {})


__all__ = ["REACH_CELLS", "SELF_CONTACT_SPEED", "SPEED_KEY", "ReachMixin", "median_edge", "reach_gap", "speed_cap",
           "with_reach"]
