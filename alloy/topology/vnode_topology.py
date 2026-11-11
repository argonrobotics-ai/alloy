# SPDX-License-Identifier: Apache-2.0
"""Virtual-node copies as a Simulation topology (WP10 item 9; coordinator GO 2026-11-06T19:36Z).

``VNodeTopology`` is a ``TopologyLike`` and a ``ContactAdjuster`` (alloy.core.protocols) over a model whose soft
objects carry copies (``copies_model``; a card opts in with ``topology: {cut_model: copies}``). The Simulation's
exchange calls:
- ``pre_step(state, contacts, dt)``: the cutter's passes before collide. Under the ``work`` rule that is the hubs'
  knife work, then the splits and the hand-over of mass to the copies. ``step_events`` counts the step's splits per
  world, and health rebases its reference on those worlds.
- ``adjust_contacts(state, contacts)``: after collide. Copies a hub stands for are muted, and sided hubs and free
  far-side copies meet the blade on their own face (``VNodeCutter.filter_contacts``).
- ``apply(state)``: nothing. A split changes no state after the solver step.
- ``state_arrays`` / ``state_worlds``: every array the cutter changes, model arrays included, with item -> world
  maps (``vnode_state``). The stepper adds SolverVBD's own history (coupling adapters), so a rollback re-cuts bit
  for bit.
The blades are the scene's ``blade:`` cards, one per world (``VNodeCutter`` refuses two in one world). The rule is
``lead``: a tet splits as the edge comes within its contact ball, and the cut faces then ride the blade faces. It
is the rule validated with face contacts on arbitrary paths (bully strokes, oblique cuts; cut faces within 0.2 mm of
the swept path). The knife meets no cutting resistance; it only pushes material it touches face-first. ``work``
(TOP-2's knife-force rule) is not safe with face contacts yet: it puts a node the edge passes within its contact
ball (here 1.9 mm) on the cut plane, and the node's far-side copy, still on the near side, met the far face 3.4 mm
deep (a PEN_FAIL in the first cut).
Limit, as with seams: cut faces whose vertices coincided at rest never collide with each other afterwards (the
self-contact rest exclusion keeps them apart), so pressing the halves back together interpenetrates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.topology.copies_model import cut_strain_gate
from alloy.topology.cut import resolve_reach
from alloy.topology.tet_copies import TetCopies
from alloy.topology.types import Blade, TopologyLedger
from alloy.topology.vnode_cut import VNodeCutter, world_of


@wp.kernel
def _count_splits(world: wp.array(dtype=wp.int32), split_now: wp.array(dtype=wp.int32),
                  step: wp.array(dtype=wp.int32), period: wp.array(dtype=wp.int32)):
    c = wp.tid()
    if split_now[c] != 0:
        wp.atomic_add(step, world[c], 1)
        wp.atomic_add(period, world[c], 1)


def split_lead(model: newton.Model, copies: TetCopies, blades: Sequence[Blade], reach: float | None = None) -> float:
    """[m] how far ahead of the edge a tet is a split candidate: the largest edge radius + blade reach + hub particle
    radius (the work rule's contact ball; the seam rig's 1.6 mm)."""
    radius = float(model.particle_radius.numpy()[copies.node_hub].max()) if len(copies.node_hub) else 0.0
    edge = max((float(b.edge_radius) for b in blades), default=0.0)
    return edge + resolve_reach(reach) + radius


class VNodeTopology:
    """The Simulation's topology over virtual-node copies (module docstring)."""

    def __init__(self, model: newton.Model, copies: TetCopies, blades: Sequence[Blade], *, rule: str = "lead",
                 reach: float | None = None) -> None:
        self.model, self.blades_spec = model, tuple(blades)
        self.cutter = VNodeCutter(model, copies, blades, split_lead(model, copies, blades, reach), rule=rule,
                                  reach=reach, cut_strain=cut_strain_gate(model, copies))
        dev, w = model.device, max(int(model.world_count), 1)
        self.step_events = wp.zeros(w, dtype=wp.int32, device=dev)
        self._period = wp.zeros(w, dtype=wp.int32, device=dev)
        self.body_active = wp.ones(max(int(model.body_count), 1), dtype=wp.int8, device=dev)
        self._mass0 = self._world_mass()
        self._state: newton.State | None = None

    @property
    def copy_mask(self) -> wp.array:
        """[P] int8, 1 = a copy: leave out of particle-count and throughput denominators (ruling 3)."""
        return self.cutter.copy_mask

    def pre_step(self, state: newton.State, contacts: newton.Contacts | None, dt: float) -> None:
        self._state = state
        self.step_events.zero_()
        self.cutter.pre_step(state)
        wp.launch(_count_splits, dim=self.cutter.copies.count, device=self.model.device,
                  inputs=[self.cutter.tets.world, self.cutter.split_now, self.step_events, self._period])

    def adjust_contacts(self, state: newton.State, contacts: newton.Contacts) -> None:
        self.cutter.filter_contacts(state, contacts, faces=True)

    def apply(self, state: newton.State) -> None:
        """Nothing: the splits happened in ``pre_step``."""

    def state_arrays(self) -> Mapping[str, wp.array]:
        return self.cutter.state_arrays()

    def state_worlds(self) -> Mapping[str, wp.array]:
        return self.cutter.state_worlds()

    def _world_mass(self) -> np.ndarray:
        m = self.model
        w = max(int(m.world_count), 1)
        return np.bincount(world_of(m.particle_world.numpy()), weights=m.particle_mass.numpy().astype(np.float64),
                           minlength=w)[:w]

    def _momentum(self) -> np.ndarray:
        m, w = self.model, max(int(self.model.world_count), 1)
        out = np.zeros((w, 6))
        if self._state is None or not m.particle_count:
            return out
        world = world_of(m.particle_world.numpy())
        mass = m.particle_mass.numpy().astype(np.float64)
        x, v = self._state.particle_q.numpy().astype(np.float64), self._state.particle_qd.numpy().astype(np.float64)
        p = mass[:, None] * v
        for k in range(3):
            out[:, k] = np.bincount(world, weights=p[:, k], minlength=w)[:w]
            out[:, 3 + k] = np.bincount(world, weights=np.cross(x, p)[:, k], minlength=w)[:w]
        return out

    def ledger(self) -> TopologyLedger:
        """Host copy of the period since the last call (a consuming read): splits per world, particle mass before and
        after (copies of kinematic hubs add mass, ``VNodeCutter.added_mass``), momentum at the last step."""
        w = max(int(self.model.world_count), 1)
        mass = self._world_mass()
        mom = self._momentum()
        led = TopologyLedger(events=self._period.numpy().astype(np.float64), mass_before=self._mass0,
                             mass_after=mass, momentum_before=mom, momentum_after=mom, overflow=np.zeros(w),
                             reset_required=np.zeros(w, bool))
        self._period.zero_()
        self._mass0 = mass
        return led
