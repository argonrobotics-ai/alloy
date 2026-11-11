# SPDX-License-Identifier: Apache-2.0
"""pad-finish (SC9): its start against theirs and the hand-off between stages (alloy.tasks.pad_finish_handoff),
CPU and without a simulation."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.tasks import concrete_pad_start as PS
from alloy.tasks import pad_finish_handoff as PH


@pytest.fixture(scope="module")
def start0():
    return PS.pad_fields("pad-finish", 0)


@pytest.mark.parametrize("seed", [0, 1])
def test_the_start_is_their_raw_strike_off(seed):
    """Their pad-finish starts as pad-darby does: the same region and draws (layout.py:224-290), 5 min x MASCO old
    (layout.py:211), with no stage's work applied (layout.py:271-278)."""
    fin, dar = PS.pad_fields("pad-finish", seed), PS.pad_fields("pad-darby", seed)
    assert sorted(fin) == sorted(dar)
    for k, v in dar.items():
        assert np.array_equal(np.asarray(fin[k]), np.asarray(v)), k


def _end(f, rng, drop=0.0):
    """A stage's end on ``f``'s cells: the surface moved by up to 1 mm (paste moved about, its volume kept) and
    ``drop`` m less paste in one corner; states as the layer would read them."""
    paste = f["paste_mask"]
    s = f["h"] + np.where(paste, rng.normal(0.0, 3e-4, f["h"].shape), 0.0)
    s = np.where(paste, s - (s - f["h"])[paste].mean(), f["h"])                   # moved, not made or lost
    s[:20, :20] -= drop
    states = {k: rng.random(f["h"].shape).astype(np.float32) for k in PH.START_STATES}
    states["age"] = np.full(f["h"].shape, 900.0)
    return s, states


def test_the_start_keeps_the_smear_as_their_tb(start0):
    """Theirs (layout.py:256-269): the surface is b + tb + ts everywhere; on the form tops the smear is their tb."""
    f = start0
    assert np.allclose(f["h"], f["b"] + f["tb"] + f["ts"], rtol=0.0, atol=1e-15)
    assert np.array_equal(f["tb"][f["form_mask"]], f["smear"][f["form_mask"]]) and f["smear"].max() > 1e-3


def test_a_hand_off_keeps_the_surface_states_and_volume(start0):
    """The next stage's fields: the surface exactly (paste: b + tb + ts; forms: b + smear), the cream kept where the
    surface still holds it, the layer's states on the paste, one clock; the paste's volume kept exactly."""
    f = start0
    s, states = _end(f, np.random.default_rng(3))
    s[f["form_mask"]] += 1e-3 * f["form_mask"][f["form_mask"]]                   # 1 mm left on the form tops
    g = PH.after_stage(f, s, states, extra={"compact": np.ones_like(s)})
    paste = f["paste_mask"]
    assert np.array_equal(g["h"], s)                                           # the surface itself
    assert np.allclose(g["b"] + g["tb"] + g["ts"], s, rtol=0.0, atol=1e-15)    # theirs: h = b + tb + ts
    forms = f["form_mask"]
    assert np.array_equal(g["tb"][forms], g["smear"][forms]) and np.all(g["ts"][forms] == 0.0)   # smear = their tb
    assert np.all(g["ts"] <= f["ts"]) and np.all(g["ts"] >= 0.0) and np.all(g["tb"][paste] >= 0.0)
    assert np.allclose(g["smear"][forms], f["smear"][forms] + 1e-3, rtol=0.0, atol=1e-12)
    assert np.all(g["smear"][paste] == 0.0)
    for k in PH.START_STATES:
        assert np.array_equal(g[k][paste], states[k][paste]), k
    assert g["age"] == 900.0 and np.all(g["compact"] == 1.0)
    dv = PH.paste_volume(g) - PH.paste_volume(f)
    assert abs(dv - 1e-3 * f["form_mask"].sum() * f["cell"] ** 2) < 1e-12                # the form tops' 1 mm only
    assert PH.paste_volume(PH.after_stage(f, f["h"], {"age": np.full_like(s, f["age"])})) == pytest.approx(
        PH.paste_volume(f), rel=1e-13, abs=0.0)


def test_a_form_left_a_float32_step_under_its_top_stays_there(start0):
    """A stage's film holds its base as float32, and 0.089 m is not one: a bare form's film emptied leaves its cell a
    float32 step under ``b``. The hand-off keeps that surface (no smear, ``h`` the surface), so the next stage's split
    starts there and nothing is put back that the books counted gone."""
    f = start0
    forms = f["form_mask"] & (f["smear"] == 0.0)
    s = np.where(forms, np.asarray(f["b"], np.float64) - 5.9e-9, f["h"])      # float32(0.089) is 5.9 nm under
    g = PH.after_stage(f, s, {"age": np.full_like(s, f["age"])})
    assert np.array_equal(g["h"], s) and np.all(g["smear"][forms] == 0.0) and np.all(g["tb"][forms] == 0.0)
    split = PH.exact_split({"base": np.asarray(f["b"]), "film": g["smear"], "paste": f["paste_mask"]}, g["h"])
    assert np.array_equal(split["base"].astype(np.float64)[forms] + split["film"][forms], s[forms])


def test_one_clock_and_the_records_accumulate():
    """Age is the site's one clock (a split clock is refused); a late or wet working stays recorded (OR) and the
    distance worked adds up over the stages."""
    paste = np.ones((3, 3), bool)
    assert PH.age_of(np.full((3, 3), 7200.0), paste) == 7200.0
    with pytest.raises(ValueError, match="one clock"):
        PH.age_of(np.array([[7200.0, 7201.0, 7200.0]] * 3), paste)
    rec = PH.Records()
    one = {"late": np.array([[0, 1]]), "wet_worked": np.array([[1, 0]]), "worked": np.array([[0.1, 0.0]])}
    two = {"late": np.array([[0, 0]]), "wet_worked": np.array([[0, 0]]), "worked": np.array([[0.2, 0.3]])}
    rec.add(one)
    rec.add(two)
    assert rec.fields["late"].tolist() == [[0, 1]] and rec.fields["wet_worked"].tolist() == [[1, 0]]
    assert np.allclose(rec.fields["worked"], [[0.3, 0.3]], rtol=1e-12, atol=0.0)


def test_the_split_carries_the_surface_exactly(start0):
    """WP15: a float32 base at 0.1 m rounds by ~7e-9 m; picked first, the film is the surface less it in float64, so
    base + film is the surface to float64, where a film from the float64 base would miss by the rounding."""
    f = start0
    s = f["h"] + np.where(f["paste_mask"], 3.3e-4, 0.0)
    base = np.where(f["paste_mask"], s - 1.234567e-3, f["b"])
    naive = base.astype(np.float32).astype(np.float64) + (s - base)
    assert np.abs(naive - s).max() > 1e-9                                         # what it avoids
    out = PH.exact_split({"base": base, "film": s - base, "paste": f["paste_mask"]}, s)
    assert out["base"].dtype == np.float32 and out["film"].min() >= 0.0       # the form tops: base at the surface
    assert np.abs(out["base"].astype(np.float64) + out["film"] - s).max() <= 1e-15
    with pytest.raises(ValueError, match="above the surface"):
        PH.exact_split({"base": s + 1e-3, "film": np.zeros_like(s), "paste": f["paste_mask"]}, s)


def test_stages_share_one_chart(start0):
    """Fields map 1:1 only between stages on the same cells."""
    PH.same_chart(start0, dict(start0))
    other = dict(start0, x=start0["x"] + 1e-3)
    with pytest.raises(ValueError, match="charts differ"):
        PH.same_chart(start0, other)


# --- the self-chain (DC5's test 1): a hand-off at a rest point is a pause ------------------------------------------

BLADE = (0.03, 0.04, 0.001)                      # m: a small held blade, half extents (the hand-off is the tool's)
SPEED, LIFT, REST_S = 0.2, 0.02, 1.0


def _blade_sim(f, prog):
    from alloy.plugins.harness.scene import TestScene, build_simulation
    from alloy.plugins.surface_flow import PlaneChart, SurfaceFlow
    from alloy.plugins.surface_flow.applicators import Applicator
    from alloy.tasks import pad_run as PR
    from alloy.tasks.concrete_paste import ConcretePaste, SoleWork
    from alloy.tasks.concrete_tools import ToolCarriage

    tool = Applicator(body="blade", shape="rect", size=BLADE[:2], center=(0.0, 0.0, -BLADE[2]),
                      stiffness=PR.PAD_STIFFNESS, damping=0.0, implicit=True, roll_lift=0.001)
    flow = SurfaceFlow(charts=(PlaneChart("pad", cells=f["h"].shape),), material=PR.surface_film(f, 21.1),
                       inertia=False, max_depth=0.025, u_cap=PR.U_CAP, tools=(tool,), tool_speed=SPEED,
                       per_cell_rheology=True)
    carriage = ToolCarriage(programs=(prog,), omega=PR.CARRIAGE_OMEGA, omega_rot=PR.CARRIAGE_OMEGA)
    layer = ConcretePaste(material="concrete", film_yield="surface", flow=flow, carriage=carriage,
                          tools=(SoleWork("blade", 3.0e-5, 1.0, 1.0, half=BLADE[:2]),))
    body = PR.ToolBody("blade", BLADE, 0.2, prog.keys[0].pos)
    sim = build_simulation(TestScene(build=PR.build_scene(f, (body,)), ground=False, worlds=1, dt=PR.EXCHANGE_DT),
                           [carriage, flow, layer], "cpu")
    return sim, carriage, flow, layer


def _chart(f):
    """Film from 0.75 mm under the form-top plane up (a held blade's work), the forms' smear as film."""
    paste, top = f["paste_mask"], f["b"] + f["tb"] + f["ts"]
    base = np.where(paste, np.minimum(top, PS.FORM_H - 7.5e-4), f["b"])
    return {"base": base, "film": top - base, "paste": paste}


def _programs(f):
    """Two passes of the blade across the window with a rest between, on whole periods; the program up to the rest
    (B) and the rest of it shifted to start at the hand-off (C)."""
    import math

    from alloy.tasks import pad_finish_stage as ST
    from alloy.tasks import pad_run as PR
    from alloy.tasks.concrete_tools import HEIGHT, Stroke, ToolProgram

    x0, y0 = float(f["x"][0]), float(f["y"][0])
    x, z = x0 + 0.07, PS.FORM_H + 1.0e-4 + BLADE[2]
    rest = (x, y0 + 0.26, PS.FORM_H + 0.05)                            # the blade off the chart, clear of the paste
    keys, t, cut = [Stroke(0.0, rest, mode=HEIGHT)], 0.1, []
    for ya, yb in ((y0 + 0.11, y0 + 0.06), (y0 + 0.06, y0 + 0.11)):
        t += 0.4
        k, t = ST.run_keys((x, ya), (x, yb), 0.0, z, t, lift=LIFT, speed=SPEED)
        keys += k
        t += 0.4
        keys.append(Stroke(t, rest, mode=HEIGHT))                       # back off the chart
        t = math.ceil((t + REST_S) / PR.PERIOD) * PR.PERIOD
        keys.append(Stroke(t, rest, mode=HEIGHT))                       # and at rest there
        cut.append(t)
    h = cut[0]
    full = ToolProgram("blade", tuple(keys))
    first = ToolProgram("blade", tuple(k for k in keys if k.t <= h + 1e-12))
    rest_keys = tuple(Stroke(k.t - h, k.pos, yaw=k.yaw, mode=k.mode) for k in keys if k.t > h + 1e-12)
    second = ToolProgram("blade", (Stroke(0.0, rest, mode=HEIGHT), *rest_keys))
    return full, first, second, round(cut[1] / PR.PERIOD), round(h / PR.PERIOD)


def _snapshot_into(src, dst):
    """Every registered state of ``src`` into ``dst`` (test only): the plug-ins' (the clock with them), the body
    solver's (WP4's adapter: MuJoCo's warm start and the float32 coordinate carry's residual, WP3) and both of
    Newton's body buffers."""
    from alloy.coupling.adapters import mujoco_state_arrays

    for a, b in ((src.state_arrays(), dst.state_arrays()),
                 (mujoco_state_arrays(src.stepper.solver), mujoco_state_arrays(dst.stepper.solver))):
        for k, arr in a.items():
            b[k].assign(arr.numpy())
    for buf in ("state", "_next"):
        for k in ("body_q", "body_qd", "body_f", "joint_q", "joint_qd"):
            getattr(getattr(dst, buf), k).assign(getattr(getattr(src, buf), k).numpy())


@pytest.mark.compile_heavy
@pytest.mark.timeout(900)
def test_a_hand_off_at_rest_is_a_pause():
    """DC5's self-chain: two passes in one simulation against one pass, a hand-off at the rest point and the second
    pass in a fresh one (its program shifted to the hand-off). Carried through the host API (the film's depth and
    base, the layer's start states and age; records and books folded on the host), the film's velocity, flux and
    pending inflow and the tool's load are 0 at the hand-off. With every registered state copied (test only: the
    plug-ins' with their clock, the body solver's, both body buffers) the two runs are equal bit for bit: nothing
    escapes the state registries. The chain does not carry the tool body's solver state (it restarts at its program's
    rest key: MuJoCo's warm start and WP3's coordinate-carry residual, under one float32 spacing of the pose), the
    tool's low-passed advance (2e-4 m/s here, under the 1 mm/s gate) or the clock (the next program starts at its own
    zero). Measured on this window (CPU, Warp route): the surface 9.8e-8 m on 379 of 7,200 cells, the film's volume to
    1e-17 m, D 6.6e-7, E 9.9e-8, a 1.4e-9 m, worked 4.5e-8 m, the books equal to 3e-13 m³; without the solver state
    the snapshot gives the same 9.8e-8 m, so the body restart is what the chain drops. Bounds: those, rounded up
    (DC5: a regression fails, not drifts)."""
    from alloy.tasks import pad_finish_stage as ST
    from alloy.tasks import pad_run as PR

    f = PR.window("pad-trowel", 0, 21.1, "still", (0.14, 0.12))
    full, first, second, n_all, n_first = _programs(f)
    sims = {}
    for name, prog in (("A", full), ("B", first)):
        sim, car, flow, layer = _blade_sim(f, prog)
        ST.start_stage(flow, layer, [f], _chart)
        for _ in range(n_all if name == "A" else n_first):
            sim.step_period()
        sims[name] = (sim, car, flow, layer)
    A = ST.carry(sims["A"][2], sims["A"][3], sims["A"][1])
    B = ST.carry(sims["B"][2], sims["B"][3], sims["B"][1])
    assert B.rest["flow.u"] == 0.0 and B.rest["flow.q"] == 0.0 and B.rest["flow.src_acc"] == 0.0   # the film rests
    assert B.rest["carriage.load"] < 1e-3 and B.rest["layer.advance"] < 1e-3      # their wait allows 0.05 N; under
                                                                                   # the 1 mm/s gate: not advancing
    out = {}
    for variant in ("pause", "chain"):
        sim, car, flow, layer = _blade_sim(f, full if variant == "pause" else second)
        ST.resume(B, flow, layer)
        if variant == "pause":                       # test only: a snapshot, and the whole program
            _snapshot_into(sims["B"][0], sim)
        for _ in range(n_all - n_first):
            sim.step_period()
        if variant == "pause":                       # the snapshot holds the first pass's records and books
            out[variant] = ST.carry(flow, layer, car)
            continue
        rec = PH.Records()
        rec.add(B.records.fields)
        out[variant] = ST.carry(flow, layer, car, records=rec, books=dict(B.books))
    for variant, C in out.items():
        exact = variant == "pause"
        assert np.array_equal(A.layer["age"], C.layer["age"]) and np.array_equal(A.base, C.base)
        assert np.abs(A.h - C.h).max() <= (0.0 if exact else 1e-7), (variant, np.abs(A.h - C.h).max())
        assert abs(float((A.h - C.h).sum())) <= (0.0 if exact else 1e-17), variant      # the film's volume kept
        for k in A.layer:
            tol = 0.0 if exact or k in ("C", "w", "q", "dwc", "bl") else {"D": 1e-6, "a": 2e-9, "E": 1e-7}.get(k, 1e-10)
            assert np.abs(A.layer[k].astype(float) - C.layer[k].astype(float)).max() <= tol, (variant, k)
        dw = np.abs(A.records.fields["worked"] - C.records.fields["worked"]).max()
        assert dw <= (0.0 if exact else 1e-7), (variant, dw)
        for k in A.books:
            assert np.abs(A.books[k] - C.books[k]).max() <= 1e-12 * max(1.0, float(np.abs(A.books[k]).max())), k
