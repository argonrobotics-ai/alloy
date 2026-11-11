# SPDX-License-Identifier: Apache-2.0
"""SCR-SC4-4b: the D20 ledger's dynamic items leave out bodies welded to the world.

C16's yam_bench floor (1.8 t, 8.5 m, a fixed joint to the world) made E_scale 154 kJ, so the 3-37 J the lagged
proxy pumped into a finger during the gather never warned or escalated. A welded body cannot move: its energy is
constant, so it must not set the datum, the mass, the extent or the energy.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.coupling import energy as en

DEV = "cpu"


def _model(floor: bool, chain: bool = False) -> newton.Model:
    """A 1 kg free box at z 0.5 m; optionally a 1,800 kg 8 x 8 m floor slab on a fixed joint to the world at
    z -0.8 m (and a 50 kg plate fixed to the floor)."""
    b = newton.ModelBuilder()
    if floor:
        slab = b.add_link(xform=wp.transform((0.0, 0.0, -0.8), wp.quat_identity()), mass=1800.0)
        b.add_shape_box(slab, hx=4.0, hy=4.0, hz=0.05)
        b.add_joint_fixed(-1, slab)
        if chain:
            plate = b.add_link(xform=wp.transform((0.0, 0.0, -0.7), wp.quat_identity()), mass=50.0)
            b.add_shape_box(plate, hx=0.5, hy=0.5, hz=0.01)
            b.add_joint_fixed(slab, plate)
        b.add_articulation([j for j in range(b.joint_count)])
    box = b.add_body(xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()), mass=1.0)
    b.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05)
    top = newton.ModelBuilder()
    top.add_world(b)                                   # the ledger skips items of the global world (-1)
    return top.finalize(device=DEV)


def _ledger(model: newton.Model) -> dict[str, float]:
    inp = en.energy_inputs(model)
    state = model.state()
    f64 = {"dtype": wp.float64, "device": DEV}
    z_ref, m_tot, l_char = wp.zeros(1, **f64), wp.zeros(1, **f64), wp.zeros(1, **f64)
    energy, e_ref, e_scale = wp.zeros(1, **f64), wp.zeros(1, **f64), wp.zeros(1, **f64)
    vmax, bad = wp.zeros(1, dtype=float, device=DEV), wp.zeros(1, dtype=wp.int32, device=DEV)
    wp.launch(en.reference_kernel, dim=1, inputs=[inp, state.body_q, state.particle_q, z_ref, m_tot, l_char],
              device=DEV)
    en.launch_energy(inp, state, z_ref, energy, vmax, bad, DEV)
    wp.launch(en.scale_kernel, dim=1, inputs=[model.gravity, energy, m_tot, l_char, e_ref, e_scale], device=DEV)
    return {k: float(v.numpy()[0]) for k, v in (("z_ref", z_ref), ("m_tot", m_tot), ("l_char", l_char),
                                                ("energy", energy), ("e_scale", e_scale))}


def test_world_welded_bodies_are_not_dynamic_items():
    free = _ledger(_model(floor=False))
    for chain in (False, True):
        welded = _ledger(_model(floor=True, chain=chain))
        assert welded == free, (chain, welded, free)
    box_kg = float(_model(floor=False).body_mass.numpy()[0])      # 1 kg + the shape's default density
    assert free["m_tot"] == pytest.approx(box_kg) and free["e_scale"] < 10.0   # the box alone, not the floor's 1.4e5 J


def test_energy_inputs_mark_the_welded_chain():
    inp = en.energy_inputs(_model(floor=True, chain=True))
    assert inp.body_dynamic.numpy().tolist() == [0, 0, 1]
    assert np.all(en.energy_inputs(_model(floor=False)).body_dynamic.numpy() == 1)
