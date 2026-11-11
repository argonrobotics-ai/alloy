# SPDX-License-Identifier: Apache-2.0
"""The kinematic bodies' own entry in a coupled scene with no rigid entry (gap wave 2, the follow-up of gaps-now #1).

Newton's coupler shows an entry the bodies it owns and the bodies proxied into it; the shapes of every other body
lose their collision flags in that entry's view. With a rigid (MuJoCo) entry, ``stepper.adopt_kinematic`` gives the
unowned KINEMATIC bodies (a held jug, a driven tray) to it and its proxies carry them into the particle entries.
With cloth and liquid only (VBD + SPH) no entry owned them: the water fell out of a kinematic jug and the cloth fell
through a kinematic tray (health: PEN_FAIL on the cloth, the world frozen).

Here they get an entry of their own, ``CARRIER``:
- its solver, ``SolverKinematicCarrier``, passes ``body_q`` / ``body_qd`` (and the free joints' coordinates)
  through: a kinematic body moves only as told (``sim.drive`` and ``sim.set_pose`` write the full state);
- it reports effective mass and inertia 0, so every proxy is immovable in its destination and sets no clock bound
  and no strong-pair class (the convention of ``alloy.coupling.kinematic``);
- one lagged one-way pair runs from it into every other entry, inside the one Scheduler: the destination sees the
  begin pose and the velocity each exchange, and nothing is fed back (no residuals, no relaxation, no bond work).

The scene's own pairs (cloth -> liquid staggered) and their iteration settings are unchanged, and a scene with a
rigid entry, with one entry, or with no unowned kinematic body gets no carrier.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import warp as wp

from alloy._vendor import newton
from alloy._vendor.newton.solvers import SolverBase
from alloy.core.newton_compat import CouplingInterface, SolverCoupledProxy
from alloy.core.protocols import DetCapability
from alloy.coupling.types import AlloyEntry, PairPolicy
from alloy.scene.types import EntrySpec

__all__ = ["CARRIER", "SolverKinematicCarrier", "carried_into_mpm", "carrier_entry", "carrier_pairs", "carrier_spec",
           "without_carrier"]

CARRIER = "kinematic"          # the entry's name and its capability tag (not a registry capability: it routes nothing)


class SolverKinematicCarrier(SolverBase, CouplingInterface):
    """Owns kinematic bodies and integrates nothing: a step copies their state through (module docstring)."""

    determinism = DetCapability.NATIVE          # copies only: allocation-free, the same bits every run

    def step(self, state_in: Any, state_out: Any, control: Any, contacts: Any, dt: float) -> None:
        del control, contacts, dt
        if state_in is state_out:
            return
        for name in ("body_q", "body_qd", "joint_q", "joint_qd"):
            src, dst = getattr(state_in, name, None), getattr(state_out, name, None)
            if src is not None and dst is not None and src.shape[0]:
                wp.copy(dst, src)

    def coupling_eval_effective_mass(self, endpoint_kind: Any, endpoint_index: Any, endpoint_local_pos: Any,
                                     out: Any) -> None:
        del endpoint_kind, endpoint_index, endpoint_local_pos
        out.zero_()

    def coupling_eval_effective_mass_block(self, endpoint_kind: Any, endpoint_index: Any, endpoint_local_pos: Any,
                                           out_mass: Any, out_inertia: Any = None) -> None:
        del endpoint_kind, endpoint_index, endpoint_local_pos
        out_mass.zero_()
        if out_inertia is not None:
            out_inertia.zero_()


def without_carrier(dyn: Sequence[EntrySpec]) -> list[EntrySpec]:
    """``dyn`` without the carrier entry (the scene's own entries, as the assembler listed them)."""
    return [e for e in dyn if e.capability != CARRIER]


def carrier_spec(dyn: Sequence[EntrySpec], model: Any) -> EntrySpec | None:
    """The carrier's entry for ``dyn``: the KINEMATIC bodies no entry owns and their joints. None when there is
    nothing to carry: fewer than two entries (a lone solver steps the full model), a rigid entry
    (``adopt_kinematic`` gave them to it), or no unowned kinematic body."""
    n = int(model.body_count or 0)
    if len(dyn) < 2 or n == 0 or any(e.capability == "mujoco" for e in dyn):
        return None
    owned = {int(b) for e in dyn for b in e.bodies}
    flags = model.body_flags.numpy()
    bodies = [b for b in range(n) if int(flags[b]) & int(newton.BodyFlags.KINEMATIC) and b not in owned]
    if not bodies:
        return None
    child = model.joint_child.numpy() if int(model.joint_count or 0) else ()
    joints = [j for j, c in enumerate(child) if int(c) in set(bodies)]
    return EntrySpec(name=CARRIER, capability=CARRIER, bodies=tuple(bodies), particles=(), joints=tuple(joints),
                     shapes=(), substeps=1, target_dt=min(float(e.target_dt) for e in dyn))


def carrier_pairs(dyn: Sequence[EntrySpec]) -> tuple[PairPolicy, ...]:
    """One lagged one-way pair from the carrier into every other entry of ``dyn`` (none without a carrier)."""
    if not any(e.capability == CARRIER for e in dyn):
        return ()
    return tuple(PairPolicy(source=CARRIER, destination=e.name, mode="lagged", one_way=True)
                 for e in without_carrier(dyn))


def carried_into_mpm(dyn: Sequence[EntrySpec], solvers: Mapping[str, Any]) -> list[str]:
    """The MPM entries the carrier feeds carry their proxy bodies through each substep
    (``alloy.integration.mpm_bodies``, its mode from ``ALLOY_MPM_BODY_CARRY``; "off": none). SolverImplicitMPM never
    writes ``state_out.body_q``, and with two or more MPM substeps per exchange the coupler feeds substep k + 1 the
    output of substep k: a driven box would be read at a pose an older exchange left there (WP12's review).
    ``mpm_bodies.install_for_pairs`` covers the scene's own two-way pairs only. Returns the entries wrapped."""
    from alloy.integration import mpm_bodies  # noqa: PLC0415 (it imports the MPM solver)

    how = mpm_bodies.mode()
    if how == "off" or not any(e.capability == CARRIER for e in dyn):
        return []
    names = [e.name for e in dyn if e.capability == "mpm" and e.name in solvers]
    return [n for n in names if mpm_bodies.install(solvers[n], how)]


def carrier_entry(spec: EntrySpec) -> AlloyEntry:
    """The carrier as a coupler entry: one substep per exchange, no clock of its own."""
    entry = SolverCoupledProxy.Entry(name=spec.name, solver=SolverKinematicCarrier, bodies=list(spec.bodies),
                                     particles=[], joints=list(spec.joints), substeps=1)
    return AlloyEntry(entry, initial_rung=1, adaptive=False, capability=CARRIER)
