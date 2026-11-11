# SPDX-License-Identifier: Apache-2.0
"""TOP-2 SimTwin on virtual-node copies (WP10 item 9 step 3; coordinator ruling 2, 2026-11-06).

``KnifeForceTwin``'s rig and readout with the cut model swapped:
- same block, material, knife (capsule edge + thin back box), descent and knife force;
- but no pre-split seam. Every tet carries copies (``tet_copies``) and splits by the ``work`` rule (``vnode_work``),
  with the seam twin's cut gate (``cut_min_strain``) and the blade's softness.

The force-vs-depth curve is compared with the seam twin's on the pre-cut segment (within 15 %) and on the cut timing
(coordinator ruling 2026-11-06T18:59Z; tests/topology/test_twin_copies.py). Past the first cut the seam twin's plateau
is a stock-contact artefact of the reference (its cut twins stay under the edge), a known gap in the results note.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.bench.types import Condition
from alloy.topology.meshes import grid_tet_mesh
from alloy.topology.tet_copies import TetCopies, add_tet_copies, concat_copies, recolor_model
from alloy.topology.twin import CUT_KEYS, KnifeForceTwin, _clamp, _World, lame
from alloy.topology.types import Blade
from alloy.topology.vnode_cut import VNodeCutter


class KnifeStepperCopies(newton.solvers.SolverBase):
    """One step = the cutter's passes (splits, then the copies' contact mute), VBD. Stock blade contacts otherwise,
    as the seam twin's ``KnifeStepper`` (no BladeContact), so the two twins differ only in the cut model."""

    def __init__(self, model: newton.Model, cutter: VNodeCutter, iterations: int = 10) -> None:
        super().__init__(model)
        self.cutter = cutter
        self.vbd = newton.solvers.SolverVBD(model, iterations=iterations)

    def step(self, state_in, state_out, control, contacts, dt, *, observables=None) -> None:
        self.cutter.pre_step(state_in)
        self.cutter.filter_contacts(state_in, contacts, faces=False)
        self.vbd.step(state_in, state_out, control, contacts, dt)


@dataclass
class KnifeForceTwinCopies(KnifeForceTwin):
    """``KnifeForceTwin`` on copies (see the module docstring)."""

    rule: str = "work"                                  # VNodeCutter rule ("lead": no knife force, a reference)
    _parts: list[TetCopies] = field(default_factory=list)
    _cut_strain: float = 0.0

    def build_world(self, b: newton.ModelBuilder, cond: Condition, sample: Mapping[str, float]) -> None:
        p = {**cond.params, **sample}
        mu, lam = lame(float(p["youngs"]), float(p["poisson"]))
        v, t = grid_tet_mesh(*self.cells, self.cell)
        p0, t0 = b.particle_count, len(b.tet_indices)
        b.add_soft_mesh(pos=wp.vec3(0.0, 0.0, 0.0), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0.0, 0.0, 0.0),
                        vertices=[wp.vec3(*q) for q in v], indices=t.ravel().tolist(), density=float(p["density"]),
                        k_mu=mu, k_lambda=lam, k_damp=0.0, particle_radius=1.0e-3)
        self._clamp += [p0 + i for i in range(b.particle_count - p0) if b.particle_q[p0 + i][2] <= 1e-9]
        self._parts.append(add_tet_copies(b, range(t0, len(b.tet_indices)), float(p["density"])))
        self._cut_strain = float(p.get("cut_min_strain", 0.05))
        knife = b.add_body(xform=wp.transform(wp.vec3(self.plane_x, 0.5 * self.cells[1] * self.cell, self.top + 1e-3),
                                              wp.quat_identity()), mass=0.0)
        cfg = newton.ModelBuilder.ShapeConfig(density=0.0)
        b.add_shape_capsule(knife, radius=1.0e-4, half_height=0.02, cfg=cfg,
                            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(1, 0, 0), 0.5 * np.pi)))
        b.add_shape_box(knife, hx=1.0e-4, hy=0.02, hz=0.01, cfg=cfg,
                        xform=wp.transform(wp.vec3(0.0, 0.0, 0.01), wp.quat_identity()))
        self._worlds.append(_World(knife, self.top + 1e-3, dict(p)))

    def make_solver(self, model: newton.Model, solver: str = "vbd") -> newton.solvers.SolverBase:
        self._model = model
        _clamp(model, np.asarray(self._clamp, dtype=np.int64))
        model.soft_contact_ke, model.soft_contact_kd, model.soft_contact_mu = self.contact_ke, 1e-2, 0.3
        copies = concat_copies(self._parts)
        recolor_model(model, copies)
        blades = [Blade(w.knife, -1, (0.0, -0.02, 0.0), (0.0, 0.02, 0.0),
                        **{k: float(w.sample[k]) for k in CUT_KEYS if k in w.sample}) for w in self._worlds]
        cutter = VNodeCutter(model, copies, blades, 1.6e-3, rule=self.rule, cut_strain=self._cut_strain)
        dev = model.device
        self._knives = wp.array([w.knife for w in self._worlds], dtype=wp.int32, device=dev)
        self._z0 = wp.array([w.z0 for w in self._worlds], dtype=wp.float32, device=dev)
        self._ke = float(model.soft_contact_ke)
        return KnifeStepperCopies(model, cutter)
