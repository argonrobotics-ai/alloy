# SPDX-License-Identifier: Apache-2.0
"""Cut-front predicate for the health monitor's PEN rule (SCR-SC5-3; ruling at the end of [internal note]).

The blade cuts a TET_CUT bond in phase 1. By design (SCR-SC5-1), that bond's twins start between the blade faces,
and the blade contact pushes them out to the faces over a few steps. WP5's health monitor takes exactly those soft
contacts out of the PEN rule and holds them to their own limit. The TopologyController owns two device arrays for
this:

- ``cut_front_blade``, int32 [particle_count]: the body id of the Blade cutting a phase-1 TET_CUT bond that this
  particle is a twin of. It is -1 otherwise, and the largest body id when the particle twins several such bonds.
  The cutting Blade is the bond's nearest blade, the one the cut kernel tracks. ``pre_step`` refreshes the array
  right after the cut kernel, with one thread per twin particle, so during the step it shows the phase the solve
  ran with. ``health.update`` runs before ``topology.apply``.
- ``blade_half_thickness``, float32 [body_count]: for a Blade body, the blade's half-thickness: the thin half-extent
  of the box that BladeContact takes as the blade face (``blade_contact.blade_box``), or ``Blade.edge_radius`` when
  the body has no box. 0 for other bodies. Blades that share a body take the smallest value.

A soft contact (particle i, shape s) is a cut-front contact when ``shape_body[s] == cut_front_blade[i]``. Its
limit is ``blade_half_thickness[cut_front_blade[i]] + particle_radius[i]``. A Blade is a body, so every shape on it
counts (edge, blade face, handle). The scene wiring's ``Blade.shape`` is the object's first shape: in C9 that is the
edge capsule, while the contacts land on the blade box. This module only says which contacts are cut-front
contacts (phase-1 twins against their cutting Blade). WP5's health monitor owns the depth limit and the PEN_FAIL
fallback.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.topology.blade_contact import blade_box
from alloy.topology.cut import PHASE_CUTTING
from alloy.topology.types import Blade, BondKind, BondSet


@wp.kernel
def mark_cut_front(twins: wp.array[wp.int32], start: wp.array[wp.int32], bond: wp.array[wp.int32],
                   phase: wp.array[wp.int8], track_blade: wp.array[wp.int32], blade_body: wp.array[wp.int32],
                   out: wp.array[wp.int32]):
    """One thread per twin particle: the largest Blade body cutting (phase 1) one of its TET_CUT bonds, else -1."""
    t = wp.tid()
    best = int(-1)
    for k in range(start[t], start[t + 1]):
        i = bond[k]
        if phase[i] == wp.int8(PHASE_CUTTING) and track_blade[i] >= 0:
            best = wp.max(best, blade_body[track_blade[i]])
    out[twins[t]] = best


def _twin_lists(bonds: BondSet) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """TET_CUT twin particles, and per twin (CSR start [T+1], bond list) the TET_CUT bonds it is an end of."""
    cut = np.flatnonzero(bonds.kind.numpy() == int(BondKind.TET_CUT)) if bonds.count else np.zeros(0, np.int64)
    owners = np.concatenate([bonds.a.numpy()[cut], bonds.b.numpy()[cut]]).astype(np.int64)
    ids = np.concatenate([cut, cut])
    order = np.lexsort((ids, owners))
    twins, counts = np.unique(owners[order], return_counts=True)
    start = np.concatenate([[0], np.cumsum(counts)])
    return twins.astype(np.int32), start.astype(np.int32), ids[order].astype(np.int32)


def blade_half_thickness(model: newton.Model, blades: Sequence[Blade]) -> np.ndarray:
    """[body_count] float32: each Blade body's blade half-thickness [m] (see the module docstring); 0 elsewhere."""
    out = np.zeros(int(model.body_count), np.float32)
    scale = model.shape_scale.numpy() if model.shape_count else np.zeros((0, 3), np.float32)
    for bl in blades:
        box = blade_box(model, bl)
        half = float(scale[box].min()) if box >= 0 else float(bl.edge_radius)
        out[bl.body] = half if out[bl.body] == 0.0 else min(out[bl.body], half)
    return out


@dataclass
class CutFront:
    """The two predicate arrays plus the twin lists that refresh ``blade``. Built once; ``refresh`` only launches."""

    blade: wp.array          # [particle_count] int32, the Blade body cutting this twin (phase 1), else -1
    half: wp.array           # [body_count] float32, blade half-thickness of a Blade body [m], else 0
    twins: wp.array          # [T] int32, particles that are an end of a TET_CUT bond
    start: wp.array          # [T + 1] int32
    bond: wp.array           # their TET_CUT bonds

    @classmethod
    def build(cls, model: newton.Model, bonds: BondSet, blades: Sequence[Blade]) -> CutFront:
        dev = model.device
        twins, start, bond = _twin_lists(bonds)
        i32 = lambda a: wp.array(a, dtype=wp.int32, device=dev)  # noqa: E731
        return cls(wp.full(int(model.particle_count), -1, dtype=wp.int32, device=dev),
                   wp.array(blade_half_thickness(model, blades), dtype=wp.float32, device=dev),
                   i32(twins), i32(start), i32(bond))

    def refresh(self, phase: wp.array, track_blade: wp.array, blade_body: wp.array) -> None:
        """Rewrite ``blade`` for every twin from the current bond phases (kernels only; capturable)."""
        if self.twins.shape[0]:
            wp.launch(mark_cut_front, dim=self.twins.shape[0], device=self.blade.device,
                      inputs=[self.twins, self.start, self.bond, phase, track_blade, blade_body, self.blade])
