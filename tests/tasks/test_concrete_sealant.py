# SPDX-License-Identifier: Apache-2.0
"""tool-sealant: the start port and the verdict (Gym-Anything task.py:654-680, task_set_progress.py:174-197)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from alloy.tasks import concrete_sealant_metrics as SM
from alloy.tasks import concrete_sealant_start as SS


def test_start_is_their_joint_region() -> None:
    """Their region (layout.py:486-527): 60 x 1650 cells of 1 mm, a 13 mm joint over the backer 8.5 mm deep, a convex
    bead 1.5-2.7 mm proud with two gun stops, the 40 mm stretch sagged off the north wall at -4 mm, a quarter of the
    edge cells out of contact, 10 min old; the skin forms at 2 h."""
    j = SS.joint_region(0)
    assert j.joint.shape == (60, 1650) and int(j.joint[:, 0].sum()) == 13
    assert np.all(j.base[j.joint] == -SS.CJ_BACKER) and np.all(j.top[~j.joint] == 0.0)
    assert int(j.thin.any(axis=0).sum()) == 40 and np.all(j.y[j.thin] > SS.CJ_Y)      # 40 rows, the north half
    assert np.allclose(j.top[j.thin], -0.004, rtol=0, atol=0)
    bead = j.top[j.joint & ~j.thin]
    assert 1.0e-3 < float(np.median(bead)) < 3.0e-3
    for edge, contact in ((j.edge_l, j.contact_l), (j.edge_r, j.contact_r)):
        assert 0.70 < float(contact[edge].mean()) < 0.80
    assert not j.contact_r[j.thin].any()
    assert SS.skin_s() == pytest.approx(7200.0 * SS.masco_factor(21.1), rel=0, abs=0)


def _tooled(j: SS.Joint, skin: float) -> SM.SealantGrid:
    """Every joint cell 1.5 mm below the faces (7 mm thick), wetted on both walls, worked a minute in."""
    h = np.where(j.joint, -1.5e-3, 0.0)
    s = SM.start_grid(j, skin)
    g = dataclasses.replace(s.g, h=h, t=np.where(j.joint, h - j.base, 0.0), worked=j.joint.astype(float))
    return dataclasses.replace(s, g=g, contact_l=j.edge_l.astype(float), contact_r=j.edge_r.astype(float),
                               seal_age=np.where(j.joint, j.age + 60.0, 0.0))


def _passed(v) -> dict:
    return {c.name: c.passed for c in v.criteria}


def test_the_start_fails_as_theirs_and_a_tooled_joint_passes() -> None:
    j = SS.joint_region(0)
    skin = SS.skin_s()
    start = SM.start_grid(j, skin)
    v0 = SM.tool_sealant(start, start)
    assert _passed(v0) == {"concave_and_thick": False, "wall_contact": False, "no_void": True, "smear": True,
                           "tooled_before_skin": False}
    assert v0["wall_contact"].value < 1.0 - 40 / 1550 + 1e-9         # the sagged stretch never touches its wall
    v = SM.tool_sealant(_tooled(j, skin), start)
    assert v.success and v.score == 1.0, v


@pytest.mark.parametrize("fast", [False, True])
def test_the_layer_skins_at_the_start_ports_age(fast: bool) -> None:
    """The process layer's sealant (2 h, or the 20 min fast grade) skins when the start port's verdict says."""
    from alloy.tasks import concrete_sealant as CS

    for temp in (10.0, 21.1, 32.0):
        task = CS.ToolSealant(fast=fast, temperature=temp)
        assert CS.material(task) == ("sealant_fast" if fast else "sealant")
        assert CS.skin(task) == pytest.approx(SS.skin_s(temp, fast=fast), rel=1e-12)


@pytest.mark.parametrize("fault", ["deep", "dry_wall", "void", "smear", "late"])
def test_each_criterion_fails_alone(fault: str) -> None:
    j = SS.joint_region(0)
    skin = SS.skin_s()
    s = _tooled(j, skin)
    rows = slice(800, 840)                                          # 40 of the segment's 1550 rows
    if fault == "deep":
        s = dataclasses.replace(s, g=dataclasses.replace(s.g, h=np.where(j.joint, -3.0e-3, 0.0)))
    elif fault == "dry_wall":
        c = s.contact_l.copy()
        c[:, rows] = 0.0
        s = dataclasses.replace(s, contact_l=c)
    elif fault == "void":
        t = s.g.t.copy()
        t[30, 900] = 0.5e-3
        s = dataclasses.replace(s, g=dataclasses.replace(s.g, t=t))
    elif fault == "smear":
        t = s.g.t.copy()
        t[:5, 600:700] = 0.3e-3                                      # 5 cm2 on the face 25-30 mm off the joint
        s = dataclasses.replace(s, g=dataclasses.replace(s.g, t=t))
    else:
        a = s.seal_age.copy()
        a[30, 1000] = skin + 1.0
        s = dataclasses.replace(s, seal_age=a)
    name = {"deep": "concave_and_thick", "dry_wall": "wall_contact", "void": "no_void", "smear": "smear",
            "late": "tooled_before_skin"}[fault]
    p = _passed(SM.tool_sealant(s))
    assert not p[name] and all(v for k, v in p.items() if k != name), p


@pytest.mark.compile_heavy
@pytest.mark.timeout(240)
def test_short_strokes_over_the_sagged_stretch_on_cpu() -> None:
    """Planned strokes down a 30 mm window over the stretch that sagged off the north wall, each wiped: the nose rests
    on both arrises, cuts the bead to its arc and fills the sagged cells up to it from its carry (WP15's profile fill),
    so the centre line ends 0.5-2.5 mm deep with at least 6 mm of sealant and every row is wetted against both walls
    (the pad's pressure far above twice the sealant's yield); nothing lands on the faces; the rag keeps
    exp(-wipe / 5 cm) of the load per wipe, and no stroke loads the nose past 80 % of what it holds when lifted on the
    wall (nothing drops); the film's books close."""
    from alloy.tasks import concrete_sealant as S  # noqa: PLC0415 - the simulation module, imported on use

    task = S.ToolSealant(window=(1.155, 1.185))
    res = S.run(task, npz=None)
    assert res.ledger_rel < 1e-12, res
    g = res.verdicts[0]["criteria"]
    assert res.success == 1, g                      # concave and thick, wetted on both walls (the sagged stretch's
                                                    # north wall too), no void, no smear, tooled before the skin
    assert res.wiped_ml[0] > 0.0 and res.dropped_ml[0] == 0.0, res
    assert res.carried_ml[0] <= math.exp(-task.wipe / S.RAG_WIPE) * res.liftoff_ml[0][-1] * (1 + 1e-9), res   # the
                                                                       # last wipe keeps 13.5 % of the last stroke's
    assert res.stick_ml == pytest.approx(_ga_vertical_stick_ml(S), rel=1e-12, abs=0.0)   # the scene's gravity
    plan = _planned_cuts(S, task, S.region(task))[0] * 1e6             # ml per stroke
    lift = np.asarray(res.liftoff_ml[0])
    assert len(lift) == len(plan) and lift.max() < 0.8 * res.stick_ml, res.liftoff_ml
    cut = lift - math.exp(-task.wipe / S.RAG_WIPE) * np.concatenate([[0.0], lift[:-1]])   # less the carried share
    assert np.all(cut <= 1.05 * plan), (cut, plan)                      # the engine cuts what the plan says
    kept = res.rag_kept[0]                                             # the wipes that had a load to wipe
    assert kept and kept == pytest.approx([math.exp(-task.wipe / S.RAG_WIPE)] * len(kept), rel=1e-12, abs=0.0)


def _ga_vertical_stick_ml(S) -> float:
    """GA's lifted knife on a vertical wall: 0.5·A·tau_y / (rho·g·6) with A = 4·hx·hy, times 0.5 for a sole facing
    sideways (coupling.py:922-938) [ml]."""
    area = 4.0 * S.NOSE_HALF * S.NOSE_W / 2
    return 0.5 * 0.5 * area * S.SEALANT["yield_stress"] / (S.SEALANT["density"] * 9.81 * 6.0) * 1e6


def test_a_lifted_nose_keeps_gas_vertical_wall_volume() -> None:
    """The engine's stick rule for the nose under the scene's gravity is GA's vertical-wall 0.521 ml; laid face down
    (gravity -z) it would keep twice that."""
    from alloy.tasks import concrete_sealant as S  # noqa: PLC0415

    keep = S.stick_ml(S.nose_tool(), S.film(), S.GRAVITY)
    assert keep == pytest.approx(_ga_vertical_stick_ml(S), rel=1e-12, abs=0.0)
    assert keep == pytest.approx(0.521, abs=5e-4)
    assert S.stick_ml(S.nose_tool(), S.film(), (0.0, 0.0, -9.81)) == pytest.approx(2.0 * keep, rel=1e-12)


def test_the_rag_schedule_keeps_exp_wipe_over_its_length_per_wipe() -> None:
    """tool_release acts per exchange: over the periods a wipe overlaps (it starts and ends mid-period), the shares
    keep exp(-wipe / RAG_WIPE) of the load, for any exchange count."""
    from alloy.tasks import concrete_sealant as S  # noqa: PLC0415

    task = S.ToolSealant()
    wipes = [(0.4567, 0.4567 + task.wipe / task.wipe_speed), (1.2345, 1.2345 + task.wipe / task.wipe_speed)]
    for n_x in (1, 16, S.exchanges()):
        s = S.release_schedule(wipes, 100, task, n_x)
        for k in (slice(0, 50), slice(50, 101)):                       # each wipe on its own
            kept = math.prod((1.0 - x) ** n_x for x in s[k])
            assert kept == pytest.approx(math.exp(-task.wipe / S.RAG_WIPE), rel=1e-12, abs=0.0)
    assert S.exchanges() == 16                                         # the harness exchanges every 1.25 ms


def _planned_cuts(S, task, r):
    """Each planned stroke's cut [m³], swept in order over the start fields, and the swept field."""
    from alloy.tasks import concrete_sealant_plan as SP  # noqa: PLC0415

    j, edges = r.j, S.stroke_edges(task, r)
    f = SP.Field(surf=np.where(j.joint, j.top, -np.inf), x=j.y[:, :1] - SS.CJ_Y, z=j.z[0].copy(), cell=j.cell)
    path, nose = SP.Path(task.run_in, task.run_out, task.rise), SP.Nose(S.NOSE_R, S.NOSE_HALF, S.NOSE_REST)
    cuts = []
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        cut, surf, rows = SP.sweep(f, SP.centres(a, b, path, nose), nose)
        f.surf[:, rows] = surf
        cuts.append(cut)
    return np.asarray(cuts), f


@pytest.mark.parametrize("seed", [0, 1])
def test_the_plan_keeps_every_stroke_in_budget_and_beats_the_fast_skin(seed: int) -> None:
    """Over the whole segment the planned strokes, run-in and run-out included, each cut at most the budget (shorter
    over the gun stops; the first comes down past the bead's end), leave nothing above the nose's arc between the tape
    marks, and the program ends before the fast grade skins (20 min, the sealant 10 min old at the start)."""
    from alloy.tasks import concrete_sealant as S  # noqa: PLC0415

    task = S.ToolSealant(seed=seed)
    r = S.region(task)
    j, edges = r.j, S.stroke_edges(task, r)
    cuts, f = _planned_cuts(S, task, r)
    assert max(cuts) <= task.budget and edges[-1] == pytest.approx(SS.SEGMENT[0])
    lengths = -np.diff(edges)
    assert lengths.min() < 0.0055 and lengths.max() > 0.01            # short over the gun stops, long elsewhere
    arc = S.NOSE_REST - np.sqrt(np.maximum(S.NOSE_R**2 - f.x**2, 0.0))
    seg = (f.z >= SS.SEGMENT[0]) & (f.z < SS.SEGMENT[1])
    assert np.all(np.where(np.isfinite(f.surf), f.surf - arc, 0.0)[:, seg] < 1e-12)
    duration = S.program(task, r)[2]
    assert duration < S.skin(S.ToolSealant(fast=True)) - j.age - 120.0, duration
