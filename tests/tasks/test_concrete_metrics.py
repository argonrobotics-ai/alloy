# SPDX-License-Identifier: Apache-2.0
"""Concrete-finishing success criteria on synthetic grids with known answers (CPU, fast)."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.tasks import concrete_metrics as cm
from alloy.tasks.concrete_metrics import PadScalars, PasteGrid


def failed(v) -> set[str]:
    return {c.name for c in v.criteria if not c.passed}


# ---------------------------------------------------------------- straightedge
def flat_core(nx=61, ny=45, cell=0.01):
    mask = np.ones((nx, ny), bool)
    g = PasteGrid(h=np.zeros((nx, ny)), t=np.zeros((nx, ny)), cell=cell, paste_mask=mask)
    return g, cm.pad_core(g)


def test_flat_surface_has_zero_gap():
    g, core = flat_core()
    gap, gaps = cm.straightedge(g.h, core, g.cell)
    assert gap == 0.0 and len(gaps) > 10


def test_single_dip_is_two_mm():
    g, core = flat_core()
    g.h[30, 19] = -0.002
    assert cm.straightedge(g.h, core, g.cell)[0] == pytest.approx(0.002)


def test_core_excludes_30_mm_border():
    g, core = flat_core()
    assert not core[:3].any() and core[3, 20] and not core[2, 20]       # (k + 0.5) x 10 mm > 30 mm from k = 3
    g.h[1, 19] = -0.002
    assert cm.straightedge(g.h, core, g.cell)[0] == 0.0
    g.h[4, 19] = -0.002
    assert cm.straightedge(g.h, core, g.cell)[0] == pytest.approx(0.002)


def test_convex_profiles_and_ramps_do_not_count_but_a_narrow_ridge_does():
    g, core = flat_core()
    i, j = np.meshgrid(np.arange(61), np.arange(45), indexing="ij")
    dome = -1e-6 * ((i - 30.0) ** 2 + (j - 22.0) ** 2)                   # upper hull of a concave-down line is itself
    assert cm.straightedge(dome, core, g.cell)[0] < 1e-12
    assert cm.straightedge(1e-4 * i + 2e-4 * j, core, g.cell)[0] < 1e-12
    ridge = np.zeros((61, 45))
    ridge[30, 19] = 0.002                                                # the straightedge rests on it: gap beside it
    assert 0.001 < cm.straightedge(ridge, core, g.cell)[0] <= 0.002


def test_diagonals_are_included():
    g, core = flat_core()
    i0, j0 = np.argwhere(core).min(axis=0)
    every = int(round(0.025 / g.cell))
    i, j = i0 + 1, j0 + 1                                                # on no x or y line, on the k = 0 diagonal
    assert (i - i0) % every and (j - j0) % every
    g.h[i, j] = -0.002
    gap, _ = cm.straightedge(g.h, core, g.cell)
    assert gap == pytest.approx(0.002)
    axes = [cm.hull_gap(g.h[i0:, jj], g.cell) for jj in range(j0, core.shape[1], every)]
    axes += [cm.hull_gap(g.h[ii, j0:], g.cell) for ii in range(i0, core.shape[0], every)]
    assert max(axes) == 0.0


def test_window_is_one_point_two_metres():
    z = np.zeros(300)
    z[0], z[299] = 0.0, 0.0
    z[150] = -0.002
    assert cm.hull_gap(z, 0.01) == pytest.approx(0.002)
    tilt = np.linspace(0, 0.01, 300)                                     # a ramp: gap 0 whatever the window
    assert cm.hull_gap(tilt, 0.01) < 1e-12


# ---------------------------------------------------------------- pads
CELL_P = 0.002


def good_pad(nx=150, ny=120) -> PasteGrid:
    mask = np.ones((nx, ny), bool)
    d = cm.form_distance(mask, CELL_P)
    rad = cm.EDGER_RADIUS
    drop = np.where(d < rad, rad - np.sqrt(np.maximum(rad**2 - (rad - d) ** 2, 0.0)), 0.0)
    z = np.zeros((nx, ny))
    return PasteGrid(h=-drop, t=np.full((nx, ny), 0.003), cell=CELL_P, paste_mask=mask, D=z + 0.9,
                     a=z + 2e-5, E=z.copy(),
                     S=z.copy(), q=z.copy(), bl=z.copy(), bleed_left=z.copy(), late=z.copy(), wet_worked=z > 1,
                     dwc=z.copy(), worked=z + 1.0, tau_w=z + 5e3)


def mutate(g: PasteGrid, **fields) -> PasteGrid:
    out = PasteGrid(**{k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in g.__dict__.items()})
    for k, v in fields.items():
        setattr(out, k, v)
    return out


def with_dip(g: PasteGrid, depth: float, base: float) -> PasteGrid:
    out = mutate(g)
    core = cm.pad_core(out)
    out.h[core] = base
    out.h[54:67, 54:67] = base - depth
    return out


def with_ramp(g: PasteGrid, rise: float) -> PasteGrid:
    out = mutate(g)
    core = cm.pad_core(out)
    out.h[core] = np.broadcast_to(np.linspace(0, rise, g.h.shape[0])[:, None], g.h.shape)[core]
    return out


def sparse(shape, share, seed=0):
    return np.random.default_rng(seed).random(shape) < share


def darby_cases():
    g = good_pad()
    core = cm.pad_core(g)
    e = g.E.copy()
    e[core & sparse(g.h.shape, 0.05)] = 0.05
    s = g.S.copy()
    s[70, 70] = 0.3
    ww = g.wet_worked.copy()
    ww[70, 70] = True
    return {"flat_straightedge": (with_dip(g, 0.0034, 0.0015), PadScalars()),
            "flat_plane": (with_ramp(g, 0.004), PadScalars()),
            "stone_down": (mutate(g, E=e), PadScalars()), "not_sealed": (mutate(g, S=s), PadScalars()),
            "no_water_worked": (mutate(g, wet_worked=ww), PadScalars()),
            "forms_undisturbed": (g, PadScalars(forms_peak_n=151.0)), "tools_back": (g, PadScalars(tools_off=1)),
            "arms_clear": (g, PadScalars(arms_over=2))}


def test_pad_darby_passes():
    v = cm.pad_darby(good_pad(), PadScalars(forms_peak_n=150.0))
    assert v.success and v.score_legacy == 1.0 and not failed(v) and v.score is not None and v.score == 1.0
    assert v["flat_straightedge"].value == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("name", list(darby_cases()))
def test_pad_darby_one_failure_each(name):
    g, p = darby_cases()[name]
    v = cm.pad_darby(g, p)
    assert failed(v) == {name} and not v.success and v.score_legacy == pytest.approx(7 / 8)


def trowel_cases(contract="progress-v1"):
    g = good_pad()
    core, on = cm.pad_core(g), g.paste_mask
    d = g.D.copy()
    d[core] = 0.5
    bl = g.bl.copy()
    bl[70, 70] = 0.001
    q = g.q.copy()
    q[sparse(q.shape, 0.02)] = 2e-5
    late, dd = g.late.copy(), g.D.copy()
    late[sparse(late.shape, 0.06, 1)] = 1.0
    dd[late > 0] = 0.99
    ww = g.wet_worked.copy()
    ww[70, 70] = True
    dwc = g.dwc.copy()
    dwc[70, 70] = 0.06
    return {"smooth_hard": (mutate(g, D=d), PadScalars()), "no_blister": (mutate(g, bl=bl), PadScalars()),
            "delamination_risk": (mutate(g, q=q), PadScalars()),
            "no_trowel_burn": (mutate(g, late=late, D=dd), PadScalars()),
            "flat_straightedge": (with_dip(g, 0.0034, 0.0015), PadScalars()),
            "flat_plane": (with_ramp(g, 0.004), PadScalars()),
            "radius_profile": (mutate(g, h=np.zeros_like(g.h)), PadScalars()),
            "no_water_worked": (mutate(g, wet_worked=ww), PadScalars()),
            "surface_wc": (mutate(g, dwc=dwc), PadScalars()),
            "forms_undisturbed": (g, PadScalars(forms_peak_n=200.0)), "tools_back": (g, PadScalars(tools_off=1)),
            "arms_clear": (g, PadScalars(arms_over=1))}


def test_pad_trowel_passes_and_radius_is_full():
    g = good_pad()
    assert cm.radius_profile(g) > 0.98              # 8 outermost lines (centre 1 mm in): no arc cells
    assert cm.radius_profile(mutate(g, h=np.zeros_like(g.h))) == 0.0
    v = cm.pad_trowel(g)
    assert v.success and v.score_legacy == 1.0 and len(v.criteria) == 12


@pytest.mark.parametrize("name", list(trowel_cases()))
def test_pad_trowel_one_failure_each(name):
    g, p = trowel_cases()[name]
    extra = {"no_water_worked": {"no_water_worked"}, "no_blister": {"no_blister"}}.get(name, {name})
    assert failed(cm.pad_trowel(g, p)) == extra


def test_trowel_committed_water_counts_in_progress_v1_not_in_legacy():
    g = good_pad()
    left = g.bleed_left.copy()
    left[sparse(left.shape, 0.02)] = 2e-5
    g = mutate(g, S=g.S + 1.0, bleed_left=left)
    assert failed(cm.pad_trowel(g)) == {"delamination_risk"}
    assert not failed(cm.pad_trowel(g, contract="legacy"))


def test_blister_lift_threshold_and_cap():
    q = np.zeros((40, 40))
    q[20, 20] = 0.01
    lift = cm.blister_lift(q, np.zeros_like(q), 0.002)
    assert lift.max() <= 0.003 and lift[20, 20] > 3e-4 and lift[0, 0] == 0
    assert (cm.blister_lift(q * 0 + 1.5e-4, np.zeros_like(q), 0.002) == 0).all()      # below BLISTER_Q 0.2 mm


# ---------------------------------------------------------------- patch-flush
def good_patch() -> PasteGrid:
    nx, ny = 160, 120
    rec = np.zeros((nx, ny), bool)
    rec[40:120, 30:90] = True
    z = np.zeros((nx, ny))
    return PasteGrid(h=z.copy(), t=z.copy(), cell=0.001, paste_mask=rec, a=np.where(rec, 3e-4, 0.0),
                     worked=np.where(rec, 1.0, 0.0), tau_w=np.where(rec, 20e3, 0.0))


def test_patch_flush_perfect():
    v = cm.patch_flush(good_patch())
    assert v.success and len(v.criteria) == 7 and v.score_legacy == 1.0


def test_patch_flush_failures():
    g = good_patch()
    rec = g.paste_mask
    ring = cm._inner_ring(rec)
    assert ring.sum() == 2 * (80 + 60) - 4
    lip = mutate(g)
    lip.h[40, 50] = 0.004
    assert failed(cm.patch_flush(lip)) == {"no_lip"}                      # 4 mm > 3 mm but < 6.4 mm
    assert cm.patch_flush(lip)["no_hard_lip"].passed
    hard = mutate(g)
    hard.h[40, 50] = 0.007
    assert failed(cm.patch_flush(hard)) == {"no_lip", "no_hard_lip"}
    assert failed(cm.patch_flush(mutate(g, h=np.where(rec, 0.002, 0.0)))) == {"flush_mean"}
    short = mutate(g)
    short.h[ring] = -0.002
    assert failed(cm.patch_flush(short)) == {"against_the_cut"}
    for dist, bad in ((15, True), (5, False), (10, False)):                # distance > 10 mm from the cut
        sm = mutate(g)
        sm.t[40 - dist, 60] = 0.001
        assert (failed(cm.patch_flush(sm)) == {"no_feather_edge"}) is bad
    assert failed(cm.patch_flush(mutate(g, a=np.where(rec, 1e-4, 0.0)))) == {"float_texture"}
    assert failed(cm.patch_flush(mutate(g, a=np.where(rec, 7e-4, 0.0)))) == {"float_texture"}
    late = mutate(g)
    late.tau_w[60, 60] = 31e3
    v = cm.patch_flush(late)
    assert failed(v) == {"finished_in_time"} and v["finished_in_time"].value == pytest.approx(31.0)
    assert failed(cm.patch_flush(mutate(g, worked=np.zeros_like(g.h)))) == {"float_texture", "finished_in_time"}
    assert failed(cm.patch_flush(mutate(g, off_m3=3e-8))) == {"no_feather_edge"}


def test_patch_progress_v1():
    g = good_patch()
    assert cm.patch_flush(g, start=g).score == 1.0
    lip4, lip7 = mutate(g), mutate(g)
    lip4.h[40, 50], lip7.h[40, 50] = 0.004, 0.007
    v = cm.patch_flush(lip4, start=lip7)
    assert v.progress.value == pytest.approx(0.9 * 0.75)                  # (-4 + 7) / (-3 + 7) of the way, x 0.9
    assert v.score == pytest.approx(0.95 * 0.675)
    stuck = cm.patch_flush(lip4, start=lip4)
    assert stuck.score == 0.0 and stuck.progress.work == 0.0
    assert cm.patch_flush(g, start=lip7).score == 1.0                      # success always scores 1
    late = mutate(g)
    late.tau_w[40:48, 40:90] = 31e3                                        # 8 x 50 of 4,800 worked cells too stiff
    p = cm.patch_flush(late, start=g).progress
    assert p.gates["timely"] == pytest.approx(1 - 400 / 4800)


# ---------------------------------------------------------------- tool-joints
CELL_J = 0.0015


def good_joint() -> PasteGrid:
    on = np.zeros((200, 30), bool)
    on[:, 10:16] = True  # 6 cells = 9 mm; centre = 3 cells in = 2 rows
    z = np.zeros(on.shape)
    return PasteGrid(h=np.where(on, -0.002, 0.0), t=np.where(on, 0.003, 0.0), cell=CELL_J, paste_mask=on,
                     C=np.where(on, 0.7, 0.0), tau_w=np.where(on, 10e3, 0.0), off_m3=0.0, D=z)


def test_tool_joints_passes_and_concave_band():
    g = good_joint()
    assert cm.tool_joints(g).success and len(cm.tool_joints(g).criteria) == 5
    shallow = mutate(g, h=np.where(g.paste_mask, -0.001, 0.0))
    assert failed(cm.tool_joints(shallow)) == {"concave"}
    deep = mutate(g, h=np.where(g.paste_mask, -0.0029, 0.0))
    assert failed(cm.tool_joints(deep)) == {"concave"}


def test_tool_joints_other_failures():
    g = good_joint()
    on = g.paste_mask
    assert failed(cm.tool_joints(mutate(g, C=np.where(on, 0.4, 0.0)))) == {"compacted"}
    assert failed(cm.tool_joints(mutate(g, tau_w=np.where(on, 25e3, 0.0)))) == {"tooled_thumbprint_hard"}
    assert failed(cm.tool_joints(mutate(g, tau_w=np.where(on, 4e3, 0.0)))) == {"tooled_thumbprint_hard"}
    void = mutate(g)
    void.h[50, 10] = -0.005                                                # joint edge cell, 5 mm deep (> 4.3 mm)
    assert failed(cm.tool_joints(void)) == {"no_void"}
    assert cm.tool_joints(mutate(g, off_m3=3e-8))["smear"].passed is False


def test_joint_length_formula_and_smear():
    assert cm.joint_length(400, 0.0015) == pytest.approx(400 * 0.0015 / 4)
    assert cm.joint_length(0, 0.0015) == 1e-3
    g = good_joint()
    assert (cm.tool_joints(g)["no_void"].value, cm.tool_joints(g)["smear"].value) == (0.0, 0.0)
    for n, ok in ((10, True), (20, False)):                              # 400 centre cells -> 0.15 m of joint
        sm = mutate(g)
        sm.t[:n, 3] = 0.001
        v = cm.tool_joints(sm)["smear"]
        assert v.value == pytest.approx(n * CELL_J**2 * 1e4 / 0.15)
        assert v.passed is ok


def test_tool_joints_progress_v1():
    g = good_joint()
    assert cm.tool_joints(g, start=g).score == 1.0
    start = mutate(g, h=np.where(g.paste_mask, -0.0005, 0.0))
    end = mutate(g, h=np.where(g.paste_mask, -0.0012, 0.0))
    v = cm.tool_joints(end, start=start)
    assert failed(v) == {"concave"} and 0.0 < v.progress.value < 0.9
    assert v.score == pytest.approx(0.95 * v.progress.value)
    assert cm.tool_joints(end, start=end).score == 0.0


def test_trowel_and_darby_progress_v1():
    g = good_pad()
    assert cm.pad_darby(g, start=g).score == 1.0 and cm.pad_trowel(g, start=g).score == 1.0
    start = mutate(g, D=g.D * 0 + 0.1, a=g.a * 0 + 2.5e-4)
    mid = mutate(g, D=g.D * 0 + 0.5, a=g.a * 0 + 1e-4)
    v = cm.pad_trowel(mid, PadScalars(tools_off=1, tools_total=4), start=start)
    assert not v.success and 0.0 < v.score < 0.95 and v.progress.gates["tools"] == pytest.approx(0.975)
    assert cm.pad_trowel(g, start=start).score == 1.0
    case, p = darby_cases()["stone_down"]
    assert cm.pad_darby(case, p, start=g).progress.gates["open_surface"] == 1.0
    assert cm.pad_darby(case, p, start=case).score == 0.0


def test_the_pad_core_and_the_edger_arc_measure_from_their_forms_true_faces(monkeypatch, caplog):
    """Their 1.219 m pad is 609.5 of their 2 mm cells: the grid's d_edge (their pad_d, T:225) puts the east face where
    theirs is, so pad-trowel seed 0's start has their 247,660 core cells and their radius_profile 0.99719 (WP15's
    probe of their whole evaluator, reviews/pad-trowel/p2_ga_whole, p6_core_diff); the mask's bounding box, the
    fallback without d_edge, drops the east column at 30.0 mm (247,233 cells) and reads 0.9925, and warns once."""
    from alloy.tasks.concrete_pad_start import FORM_H, pad_fields

    f = pad_fields("pad-trowel", 0)
    g = PasteGrid(h=f["h"] - FORM_H, t=f["tb"] + f["ts"], cell=0.002, paste_mask=f["paste_mask"], d_edge=f["d_edge"])
    assert int(cm.pad_core(g).sum()) == 247_660
    assert cm.radius_profile(g) == pytest.approx(0.99719, abs=5e-6)
    box = PasteGrid(h=g.h, t=g.t, cell=g.cell, paste_mask=g.paste_mask)
    monkeypatch.setattr(cm, "_BOX_WARNED", False)
    with caplog.at_level("WARNING", logger="alloy.tasks.concrete_metrics"):
        assert int(cm.pad_core(box).sum()) == 247_233 and cm.radius_profile(box) == pytest.approx(0.9925, abs=5e-5)
    assert sum("without PasteGrid.d_edge" in r.message for r in caplog.records) == 1        # warned, once
