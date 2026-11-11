# SPDX-License-Identifier: Apache-2.0
"""The plug-in's own test scene and its bully B0 target (§3.8 item 7).

``drop_model``: per world a row of particles held above the plane (the contract checks build this). The bully target
``<name>_drop`` pushes particle 0 with a legal force (``particle_f`` actions) while the others fall and rest.
"""

from __future__ import annotations

import numpy as np

from alloy._vendor import newton

from alloy.bully.direct import DirectSim
from alloy.bully.targets import register_target
from alloy.bully.types import ActionSpace, BullyTarget

from .solver import PLUGIN_NAME, SolverTemplate

PARTICLES = 4
RADIUS = 0.01  # [m]
MASS = 0.05  # [kg]
F_MAX = 0.2  # [N] per axis on particle 0: at most 4 m/s^2, so 8 m/s after a 2 s horizon (below the 10 m/s limit)


def drop_model(worlds: int, device: str) -> newton.Model:
    """``PARTICLES`` particles per world at heights 0.05 .. 0.2 m, spaced 5 cm apart along x."""
    b = newton.ModelBuilder()
    SolverTemplate.register_custom_attributes(b)
    for i in range(PARTICLES):
        b.add_particle((0.05 * i, 0.0, 0.05 * (i + 1)), (0.0, 0.0, 0.0), MASS, radius=RADIUS)
    scene = newton.ModelBuilder()
    SolverTemplate.register_custom_attributes(scene)
    scene.replicate(b, copy_count=worlds)
    return scene.finalize(device=device)


def make_sim(worlds: int, device: str) -> DirectSim:
    model = drop_model(worlds, device)
    return DirectSim(model, SolverTemplate(model), control_dt=0.02, substeps=16, collide=False)


def bully_target() -> BullyTarget:
    rate = 50.0
    actions = ActionSpace(low=np.full(3, -F_MAX), high=np.full(3, F_MAX), rate_hz=rate,
                          max_delta_per_tick=np.full(3, F_MAX), initial=np.zeros(3))
    return BullyTarget(name=f"{PLUGIN_NAME}_drop", scene=None, make_sim=make_sim, actions=actions,
                       action_kind="particle_f", action_index=(0, 1, 2))


register_target(f"{PLUGIN_NAME}_drop", bully_target)
