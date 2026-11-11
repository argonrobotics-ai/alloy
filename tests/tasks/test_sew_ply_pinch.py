# SPDX-License-Identifier: Apache-2.0
"""SEW-1 ply pinch law (CPU; sewing proof gap, coordinator 2026-11-06): two 16 x 6.4 mm pads land on a stack of
0.61 mm muslin plies (10 mm cells) on a table, press, close at 10 N and lift 10 mm (``alloy.tasks.sew_ply_pinch``).

Pre-registered (docs/notes/sewing-ply-pinch.md): at every press up to 100 N the world stays valid (no PEN_FAIL /
PAIR_PEN_FAIL, no pass-through, finite) and the pinched particles rise at least 40 % of the pads' rise.

Re-cut 2026-11-10 (coordinator ruling; GAP-10A's finding; the note's "Re-cut" section). The first rig's pads landed on
the edge of a vertex row; at 100 N they slid off it and sank between the mesh's vertices to the table, and the lift
gathered cloth that lay above their bottoms: the "96 % carry" of this law, in a world with no health flag. Now the
pads land centred on a vertex row, and every case also requires ``PinchResult.pressed``: each ply has a vertex under
each pad through press and close, no pad reaches the table, MuJoCo Warp drops no row. Rows with that (spot n2 VM, main
ec7a47cb2 + the rig, 20 VBD iterations; carry over the pads' 9.94 mm rise):
- log barrier, 2 plies: 10 N 99 %, 30 N 89 %, 100 N -9 % (-0.93 mm: the plies stay on the table); 3 plies: 30 N
  81 %, 100 N -8 %. Every row valid and pressed. The same at a 16 mm opening, 100 N: -0.90 mm.
- linear penalty, 2 plies: 10 N 99 %, 30 N 93 %, 50 N 39 %, valid; 100 N PEN_FAIL (a particle's centre at a pad's
  face, 0.305 mm) with one pass-through. The control is a control again.
So the pinch holds and lifts two plies pressed at up to 30 N and does not at 100 N: that case is an xfail (known gap
sewing-pinch-heavy-press). Three plies' carry stays ungated (2026-11-09: a threshold at the old landing; not re-read
around 30 N at the new one)."""

from __future__ import annotations

import numpy as np
import pytest

from alloy._vendor import newton
from alloy.tasks import sew_ply_pinch as rig
from alloy.tasks.sew_ply_pinch import PinchSpec, PlyStack, ply_pinch_model, run_ply_pinch

CARRY_MIN = 0.40          # pinched particles' rise / pads' rise
HEAVY = ("known gap docs/known_gaps/sewing-pinch-heavy-press.md: pressed at 100 N the closing pads gather nothing and "
         "the plies stay on the table (carry -0.93 mm of the pads' 9.94 mm; valid and pressed)")


def test_make_vbd_puts_body_contacts_on_the_log_barrier():
    from alloy.solvers.vbd import make_vbd

    model = ply_pinch_model(PlyStack(size=(0.02, 0.02)))
    assert make_vbd(model, {}).rigid_soft_contact_use_log_barrier
    assert not make_vbd(model, {"rigid_soft_contact_use_log_barrier": False}).rigid_soft_contact_use_log_barrier
    assert newton.solvers.SolverVBD(model).rigid_soft_contact_use_log_barrier is False     # Newton's own default


def test_the_default_opening_seats_each_pad_on_a_vertex_row():
    """The rig's own geometry, without a run: at ``PinchSpec()`` each pad's footprint holds one vertex of every ply,
    at least 3 mm inside its edges; at the first rig's 20 mm opening the vertex row is on the pad's inner edge."""
    plies = PlyStack(count=2)
    model = ply_pinch_model(plies, PinchSpec())
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    pads, pq = state.body_q.numpy()[1:3, :3], state.particle_q.numpy()
    assert rig.SEATED_GAP == pytest.approx(0.0136) and PinchSpec().open_gap == rig.SEATED_GAP
    for pad in pads:
        dx, dy = np.abs(pq[:, 0] - pad[0]), np.abs(pq[:, 1] - pad[1])
        inside = (dx < rig.PAD_HALF[0]) & (dy < rig.PAD_HALF[1])
        assert inside.reshape(2, -1).any(axis=1).all() and dy[inside].max() < 1.0e-4, (pad, dy[inside])
    got = rig._measure(state, plies, model.particle_count // 2)
    assert got["under"] == 2 and got["bottom"] == pytest.approx(plies.top + rig.CLEARANCE, abs=1e-6)
    edge = ply_pinch_model(plies, PinchSpec(open_gap=0.020))
    state = edge.state()
    newton.eval_fk(edge, edge.joint_q, edge.joint_qd, state)
    assert rig._measure(state, plies, edge.particle_count // 2)["under"] == 0


def test_pressed_needs_every_ply_under_the_pads_a_pad_off_the_table_and_no_dropped_row():
    ok = rig.PinchResult(pen_fail=False, max_pen=0.0, flags=0, plies=2, plies_under=2, pad_bottom=3.1e-4)
    assert ok.pressed
    for change in ({"plies_under": 1}, {"plies_under": 0}, {"pad_bottom": -4.0e-6}, {"row_overflow": 3}, {"plies": 0}):
        assert not rig.PinchResult(**{**vars(ok), **change}).pressed, change


def test_the_watch_counts_dropped_rows_and_contacts():
    from types import SimpleNamespace as NS

    import warp as wp

    data = NS(nefc=wp.array([7, 12], dtype=int, device="cpu"), njmax=10, nacon=wp.array([3], dtype=int, device="cpu"),
              naconmax=8)
    seen = []
    watch = rig._Watch(NS(update=lambda *a, **k: seen.append(a), poll=lambda: "report"),
                       NS(_entries={"mjc": NS(solver=NS(mjw_data=data))}))
    watch.update(1)
    data.nefc.assign([7, 9])
    watch.update(2)
    data.nacon.assign([9])
    watch.update(3)
    assert watch.overflow == 2 and len(seen) == 3 and watch.poll() == "report"


def _assert_valid(r) -> None:
    assert r.finite and not r.pen_fail and r.pass_through == 0, (r.failure, r)
    assert r.pressed, (r.plies_under, r.pad_bottom, r.row_overflow)      # the pads bore on the plies, not the table


def _assert_holds(r) -> None:
    assert r.carry >= CARRY_MIN * r.pad_rise, r


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(7200)
def test_two_plies_pressed_at_30_n_hold_and_lift():
    """Row (CPU, spot n2 VM, 35 min): valid, pressed (pad bottoms 0.62 mm over the table), carry 8.81 of 9.94 mm
    (89 %). At 10 N: 9.88 mm (99 %)."""
    r = run_ply_pinch(PlyStack(count=2), PinchSpec(press_n=30.0))
    _assert_valid(r)
    _assert_holds(r)


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(7200)
def test_two_plies_pressed_at_100_n_hold_and_lift():
    """Row (CPU, spot n2 VM, 34 min): valid and pressed (pad bottoms 0.31 mm over the table), carry -0.93 of 9.94 mm:
    the plies stay on the table. Validity is asserted outright (``pytest.fail`` is not an AssertionError); the carry is
    the known gap. The first rig read 96 % here with its pads on the table."""
    r = run_ply_pinch(PlyStack(count=2), PinchSpec(press_n=100.0))
    try:
        _assert_valid(r)
    except AssertionError as err:
        pytest.fail(f"the 100 N pinch is not valid or not pressed: {err}")
    if r.carry < CARRY_MIN * r.pad_rise:
        pytest.xfail(HEAVY + f"; this run: {1e3 * r.carry:.2f} of {1e3 * r.pad_rise:.2f} mm")


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(7200)
def test_three_plies_pressed_at_30_n_stay_valid():
    """Validity and pressed only: three plies' carry was a threshold at the first rig's landing (2026-11-09: 55 / 54 /
    52 / 25 / -26 % at 29 / 29.5 / 30 / 30.5 / 31 N) and has one row at the new one (30 N: 8.02 of 9.94 mm, 81 %, pad
    bottoms 0.87 mm over the table; 100 N: -0.80 mm). 69 min (spot n2 VM)."""
    _assert_valid(run_ply_pinch(PlyStack(count=3), PinchSpec(press_n=30.0)))


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(7200)
def test_the_linear_penalty_lets_a_particle_reach_the_pad_at_100_n():
    """The control: Newton's linear body-particle penalty under the same pressed 100 N pinch latches PEN_FAIL (a
    particle's centre at a pad's face, 0.305 mm, at 0.32 s) with one pass-through; at 10 / 30 / 50 N it is valid (carry
    99 / 93 / 39 %). On 2026-11-09 this test pinned the opposite ("holds since the axis proxy mass"): that row had the
    pads on the table (pad bottoms 4 um under its top at the 20 mm opening). 39 min (spot n2 VM)."""
    r = run_ply_pinch(PlyStack(count=2), PinchSpec(press_n=100.0, log_barrier=False))
    assert r.finite and r.pressed and r.pen_fail, (r.failure, r)
