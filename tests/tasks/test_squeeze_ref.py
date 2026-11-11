# SPDX-License-Identifier: Apache-2.0
"""BUL-0014 squeeze law reference (alloy.tasks.squeeze_ref): the minimal wedge reproduces the trip in MuJoCo C, the
face-on control does not, and both scenes carry T12's route contact parameters."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from alloy.tasks.squeeze_ref import DRIVES, run_reference, scene_path

T12_SOLREF = 0.0020833        # sheet x block_3 on Alloy's route (explicit pair), [s]
T12_SOLIMP = (0.95, 0.99, 0.001, 0.5, 2.0)


@pytest.mark.parametrize("case", sorted(DRIVES))
def test_scene_carries_t12_route_contacts(case: str) -> None:
    m = mujoco.MjModel.from_xml_path(str(scene_path(case)))
    assert m.npair == 3 and np.allclose(m.pair_solref[:, 0], T12_SOLREF) and np.allclose(m.pair_solimp, T12_SOLIMP)
    assert np.allclose(m.pair_friction[:, :2], 0.375) and np.all(m.pair_dim == 4)
    assert np.all(m.geom_margin == 0.0)                          # Newton zeroes authored margins on the route
    assert m.body("block").mass[0] == pytest.approx(0.01265625)
    assert np.allclose(m.actuator_forcerange, [-30.0, 30.0])     # the law's 30 N cap
    assert m.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_MULTICCD) and m.opt.noslip_iterations == 50


def test_wedge_reproduces_bul0014_in_mujoco_c() -> None:
    # The single trajectory is chaotic (WP3: 6.89 / 0.36 / 3.63 / 1.84 mm at dt, dt/2, dt/3, dt/4; a 200-start
    # ensemble has 82 % over 1 mm), so this pins only that the case breaks the law's 1 mm bound; 7.47 mm here
    # with the 0.3 kg tip (6.89 mm with the 1 kg stand-in it replaced, coordinator 2026-11-05).
    res = run_reference("wedge")
    assert res.peak_mm > 1.0            # the deep edge-edge contact the squeeze law bounds at 1 mm
    assert res.peak_force_n < 30.0      # ... while that contact pushes back with ~11 N
    assert res.end_mm < 1.0             # transient: the block is pushed back out
    assert res.max_servo_n == pytest.approx(30.0)


def test_face_pinch_control_holds_in_mujoco_c() -> None:
    res = run_reference("face")
    assert res.peak_mm < 1.0 and res.max_servo_n == pytest.approx(30.0)   # 0.13 mm at the full 30 N


@pytest.mark.parametrize("pack", ["task_squeeze_face", "task_squeeze_wedge"])
def test_the_squeeze_packs_bring_their_own_floor(pack: str) -> None:
    """The block rests on the pack's table, whose top is at z = -12.7 mm. A harness that adds a ground to a file
    without a plane puts it at z = 0, 12.7 mm into the block (GAP-E1's sweep, 2026-11-10: START_PEN). The packs carry a
    floor plane at the table's underside, which nothing touches."""
    import mujoco

    from alloy.cards.library import default_root

    m = mujoco.MjModel.from_xml_path(str(default_root() / "robots" / pack / "scene.xml"))
    planes = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE]
    table = m.geom("table")
    assert len(planes) == 1 and m.geom_pos[planes[0], 2] <= table.pos[2] - table.size[2] + 1e-9
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    assert not [c for c in d.contact[:d.ncon] if planes[0] in (c.geom1, c.geom2)]
    assert min((c.dist for c in d.contact[:d.ncon]), default=0.0) > -1e-3      # no start overlap past 1 mm

