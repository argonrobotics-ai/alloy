# SPDX-License-Identifier: Apache-2.0
"""pad-finish (SC9): the chain's pieces (alloy.tasks.pad_finish, pad_finish_stages, pad_finish_stage) on the host,
without a simulation: the tools' heights from the darby's plane, the waits' readiness, the hand-off's placement of
stage-local states, the verdict's grid and the whole chain's paste books."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.tasks import concrete_materials as CM
from alloy.tasks import concrete_pad_start as PS
from alloy.tasks import pad_finish as PF
from alloy.tasks import pad_finish_stage as ST
from alloy.tasks import pad_finish_stages as SG
from alloy.tasks.concrete_tools import HEIGHT, Stroke, ToolProgram
from alloy.tasks.pad_finish_handoff import START_STATES, Records


@pytest.fixture(scope="module")
def start0():
    return PS.pad_fields("pad-finish", 0)


def _end(f, plane=-1.5e-3, worlds=2, bl=0.0):
    """A stage end on ``f``'s cells: the paste a plane ``plane`` under the form tops (the smear on the forms), states
    0.5, a blister lift ``bl`` on the paste, one late and one wet cell in world 0."""
    paste = np.asarray(f["paste_mask"], bool)
    s = np.where(paste, PS.FORM_H + plane, f["h"])
    states = {k: np.full((worlds, *s.shape), 0.5) for k in START_STATES}
    states["bl"] = np.where(paste, bl, 0.0)[None].repeat(worlds, 0)
    states["age"] = np.full((worlds, *s.shape), 7200.0)
    rec = {k: np.zeros((worlds, *s.shape)) for k in ("late", "wet_worked", "worked")}
    rec["late"][0, 300, 200] = rec["wet_worked"][0, 301, 200] = 1
    return SG.StageEnd(surface=np.repeat(s[None], worlds, 0), states=states, records=Records(rec), books={},
                       stepped_s=1.0, waited_s=0.0, tools_off=np.zeros(worlds, int), rest={})


def test_the_edger_and_trowel_heights_come_from_the_darbys_plane(start0):
    """WP6's darby leaves a plane 1.5 mm under the form tops; WP4's heights are from the form-top plane. The plane is
    the median core surface; the edger's keys off the cart move by it, its cart keys stay; the trowel's levels shift
    by it (the first; the lowest is the plane: no planing), and its blade ends come ``RIM`` from the forms, so it
    passes the 11.5-30 mm strip pad-finish reads."""
    end = _end(start0)
    end.surface[1, 300, 200] += 0.01                                  # one high cell does not move a median
    ref = SG.plane(end.surface, [start0, start0])
    assert ref == pytest.approx(-1.5e-3, abs=1e-12)
    cart = (0.0, 0.8, 0.3)
    prog = ToolProgram("edger", (Stroke(0.0, cart, mode=HEIGHT), Stroke(1.0, (0.1, 0.2, 0.0505), yaw=0.5,
                                                                         pitch=0.01, mode=HEIGHT),
                                 Stroke(2.0, cart, mode=HEIGHT)))
    low = SG.lowered(prog, ref, cart)
    assert low.keys[0] == prog.keys[0] and low.keys[2] == prog.keys[2]
    assert low.keys[1].pos[2] == pytest.approx(0.0505 + ref) and low.keys[1].pos[:2] == (0.1, 0.2)
    assert (low.keys[1].yaw, low.keys[1].pitch, low.keys[1].mode) == (0.5, 0.01, HEIGHT)
    task = SG.trowel_task(ref)
    lv = SG.PT.levels(task)
    assert lv[0] == pytest.approx(ref + SG.PT.LEVEL0) and lv[-1] == pytest.approx(ref + SG.FLOOR)
    assert SG.PT.levels(SG.trowel_task(ref, floor=SG.PT.LEVEL_MIN))[-1] == pytest.approx(ref + SG.PT.LEVEL_MIN)
    assert task.rim == SG.RIM < SG.PT.RIM and SG.trowel_task(ref, rim=SG.PT.RIM).rim == SG.PT.RIM
    strip = np.asarray(start0["paste_mask"], bool) & (start0["d_edge"] > 0.0115) & (start0["d_edge"] < 0.03)
    assert float((SG.PT.credit(task, start0)[strip] >= 1.0).mean()) > 0.97      # the trowel reaches 11.5-30 mm


def test_an_opening_that_steps_past_the_program_is_refused():
    """WP15's M1: a stage's opening watch and wait step the device clock; one that reaches the program's first key
    (``START``) is refused, and so is any ride check that would fall inside the opening (it would go unchecked)."""
    ST.check_opening(10 * SG.PERIOD, SG.START)                           # the 0.2 s watch: fine
    for stepped in (SG.START, SG.FILM_SETTLE_S):                         # to the first key; a watch at its cap
        with pytest.raises(ValueError, match="first key"):
            ST.check_opening(stepped, SG.START)
    hooks = {300: "a", 900: "b"}
    assert ST.shifted_hooks(hooks, 10) == {290: "a", 890: "b"}
    with pytest.raises(ValueError, match="inside the opening"):
        ST.shifted_hooks(hooks, 301)


def test_the_waits_end_on_the_sheen_and_the_bleed():
    """The edger waits for its band's sheen to go (WP4's rule), the trowel for the bleed to end and every sheen to
    go, in every world."""
    mask = np.array([True, True, False])
    sheen = SG.sheen_gone(mask)
    dry, wet_off = np.zeros((2, 3)), np.array([[0.0, 0.0, 1e-3], [0.0, 0.0, 0.0]])
    assert sheen({"w": dry}) and sheen({"w": wet_off})               # sheen off the mask does not count
    assert not sheen({"w": np.array([[0.0, 0.0, 0.0], [0.0, 2 * CM.SHEEN_W, 0.0]])})
    end = CM.Concrete().bleed_end_time(21.1)
    over = SG.bleed_over(mask, 21.1)
    assert over({"w": dry, "age": np.full((2, 3), end + 1.0)})
    assert not over({"w": dry, "age": np.array([[end + 1.0] * 3, [end - 1.0, end + 1.0, end + 1.0]])})


class _P:
    def __init__(self, *names):
        self.names = names

    def state_arrays(self):
        return dict.fromkeys(self.names)


def test_stage_local_states_are_placed_and_others_stop_the_hand_off():
    """A stage's stores, tub, supply and drips end with it (placed by prefix); an unknown film or layer state stops
    the hand-off until it is placed."""
    flow = _P("h", "u", "supplied", "to_tools", "tool_load", "tray_volume", "net_volume", "src_drip_v", "pool_avail",
              "nose_low", "nose_seen")
    layer = _P("age", *START_STATES, "late", "wet_worked", "worked", "tau_w", "skin_torn", "advance")
    assert ST.unlisted(flow, layer, _P("load", "release")) == []
    assert ST.unlisted(_P("h", "new_film_state"), _P("age", "tool_like"), _P("load")) == ["flow.new_film_state",
                                                                                        "layer.tool_like"]
    assert "to_tools" in ST.FLOW_BOOKS and "src_drip_v" in ST.MOVING["flow"]
    ST.require_placed(flow, layer, _P("load", "release"))
    with pytest.raises(ValueError, match="new_film_state"):
        ST.require_placed(_P("h", "new_film_state"), layer, _P("load"))


def test_the_verdict_grid_reads_the_blisters_the_chains_records_and_the_lip(start0):
    """Their surface is b + tb + ts + bl: the layer keeps bl off the film, so the grid adds it; the records are the
    chain's union; C is the lip's mark carried as compact."""
    end = _end(start0, bl=2e-4)
    f = dict(start0, compact=np.where(start0["d_edge"] < 0.002, 1.0, 0.0))
    g = PF.end_grid(f, end, 0)
    paste = np.asarray(start0["paste_mask"], bool)
    assert np.allclose(g.h[paste], -1.5e-3 + 2e-4, atol=1e-12, rtol=0)
    assert g.late[300, 200] == 1 and g.wet_worked[301, 200] == 1 and g.late.sum() == 1
    assert np.array_equal(g.C, f["compact"])
    nxt = PF.next_fields([start0], SG.StageEnd(**{**end.__dict__, "extra": {"compact": f["compact"][None]}}))
    assert np.array_equal(nxt[0]["compact"], f["compact"]) and nxt[0]["age"] == 7200.0


def test_the_chains_paste_books_close_and_catch_a_leak(start0):
    """Start + supplied - gone (the film's own outs, the darby's paste off the pad) = end, per world; a cell's paste
    gone unbooked shows as a residual."""
    end = _end(start0, worlds=1)
    cell2 = float(start0["cell"]) ** 2
    start = float(((start0["h"] - start0["b"]) * cell2).sum())
    final = float(((end.surface[0] - start0["b"]) * cell2).sum())
    end.books = {"supplied": np.array([2e-4]), "spilled": np.array([1e-5]), "to_tools": np.array([3e-5]),
                 "off_pad": np.array([start + 2e-4 - 1e-5 - 3e-5 - final]), "drained": np.array([5.0])}
    b = PF.paste_books([start0], end)
    assert abs(float(b["residual"][0])) <= 1e-12                      # drained is not an out (drained_total is)
    end.surface[0, 300, 200] -= 1e-3
    assert abs(float(PF.paste_books([start0], end)["residual"][0])) * start == pytest.approx(1e-3 * cell2, rel=1e-6)


def test_a_saved_stage_end_loads_back_whole(start0, tmp_path):
    """A stage's end saved for a rerun (``--out`` / ``--resume``) loads back field for field: surface, states,
    records, books, extra fields, the pre-wait snapshot, seconds, tools off, a rest refusal, and its rest summary and
    diagnostics (DC5 N5; not the film's ledger or the ride's log)."""
    end = _end(start0)
    end.books, end.extra = {"supplied": np.array([1e-4, 2e-4]), "off_pad": np.array([0.0, 3e-6])}, {
        "compact": np.ones((2, *start0["h"].shape))}
    end.pre_wait, end.rest_error, end.waited_s = {"surface": end.surface - 1e-6, "w": end.states["w"]}, "x", 60.0
    end.start_wait = {"before_age": end.states["age"] - 60.0, "after_age": end.states["age"]}
    end.rest = {"film.rule": "held", "film.max_bound": 1.05, "film.stepped_on_checks": 1}
    end.diag = {"faces_at_cap": 0, "start_waited_s": 60.0, "ride_ok": [True, False], "pressed_m3": np.zeros(2),
                "ledger": [1.0], "ride_log": ["x"]}
    PF._save(str(tmp_path), "edge", end)
    got = PF.load_stage(str(tmp_path), "edge")
    for k in ("surface", "tools_off"):
        assert np.array_equal(getattr(got, k), getattr(end, k)), k
    for name in ("states", "books", "extra", "pre_wait", "start_wait"):
        a, b = getattr(end, name), getattr(got, name)
        assert sorted(a) == sorted(b) and all(np.array_equal(a[k], b[k]) for k in a), name
    assert all(np.array_equal(end.records.fields[k], got.records.fields[k]) for k in end.records.fields)
    assert (got.stepped_s, got.waited_s, got.rest_error) == (1.0, 60.0, "x")
    assert got.rest == end.rest and got.diag["ride_ok"] == [True, False] and got.diag["pressed_m3"] == [0.0, 0.0]
    assert got.diag["start_waited_s"] == 60.0 and "ledger" not in got.diag and got.diag["tools_total"] == 1
    assert PF.load_stage(str(tmp_path), "edge").rest_error == "x"
    PF._save(str(tmp_path), "trowel", _end(start0))
    assert PF.load_stage(str(tmp_path), "trowel").rest_error is None


class _Arr:
    def __init__(self, a):
        self.a = np.asarray(a)

    def numpy(self):
        return self.a


class _Flow:
    """The film's arrays ``film_rest`` reads, on a 2 x 3 chart of 2 mm cells (two equal rows)."""

    def __init__(self, h, base, tau, u):
        from types import SimpleNamespace as NS
        self.h, self.base, self.u, self.q = _Arr([h + h]), _Arr([base + base]), _Arr([u]), _Arr([[0.0] * len(u)])
        self._ins = {"yield_stress": _Arr([tau + tau])}
        self.config = NS(charts=(NS(cells=(2, 3)),), material=NS(density=2400.0))
        self.tables = NS(area_np=np.full(6, 4e-6))

    def inputs(self):
        return self._ins

    def read(self):
        return {}


def test_the_film_rests_by_its_bound_or_by_its_own_step():
    """The two rules on the arrays alone: under the bound (``pad_run.rest_ratio``'s value) a film rests whatever its
    faces' last speed; over it, only if its own step held every face (``held``), else ``none``; and a hand-off asks
    for a rest stepped with the tools clear (``rest_after``'s), not one read off the arrays."""
    from alloy.tasks.pad_run import rest_ratio

    fin = dict(h=[0.024, 0.004, 0.004], base=[0.089, 0.077, 0.077], tau=[9000.0] * 3)
    held, moving = ST.film_rest(_Flow(**fin, u=[0.0, 0.0])), ST.film_rest(_Flow(**fin, u=[0.0, 1e-6]))
    assert held["bound"] == rest_ratio(_Flow(**fin, u=[0.0]), (2, 3), 0.002)
    assert held["bound"] > 1.0 and held["rule"] == "held" and held["rest"] and held["over"] == 2
    assert moving["rule"] == "none" and not moving["rest"]
    flat = ST.film_rest(_Flow(h=[0.001] * 3, base=[0.08] * 3, tau=[900.0] * 3, u=[0.0, 1e-6]))
    assert flat["bound"] < 1.0 and flat["rule"] == "bound" and flat["rest"]
    with pytest.raises(ValueError, match="stepped with the tools clear"):
        ST.check_rest(_Flow(**fin, u=[0.0]), None, held)
    assert ST.check_rest(_Flow(**fin, u=[0.0]), None, {**held, "periods": 10})["rule"] == "held"


# --- the rest rules on a film (WP15's conditions, 11-07): a fin on a form top beside the lip's groove -------------

NX, NY, CELL = 6, 3, 0.002


def _fin(taus):
    """One world per (form-top yield, pad yield): a 24 mm fin on the 2 mm form top (cells 0-1, form top 89 mm) beside
    a 12 mm groove with 4 mm of paste in it, rest ratio 1.10 at 8.2 kPa; the film's own law alone, no tools."""
    from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
    from alloy.plugins.surface_flow import FilmMaterial, PlaneChart, SurfaceFlow
    from alloy.tasks.concrete_tools import add_tool_body

    def build(b):
        b.add_shape_box(-1, xform=body_xform(0.0, 0.0, -0.025), hx=NX * CELL / 2, hy=NY * CELL / 2, hz=0.025,
                        label="pad")
        add_tool_body(b, "spare", (0.01, 0.01, 0.01), 0.1, (0.0, 0.5, 0.3), collide=False)
    W, i = len(taus), np.arange(NX)[:, None] * np.ones((1, NY), int)
    flow = SurfaceFlow(charts=(PlaneChart("pad", cells=(NX, NY)),), material=FilmMaterial(
        "paste", density=2400.0, viscosity=5.0, yield_stress=1e4, surface_tension=0.0), inertia=False,
        max_depth=0.03, u_cap=0.2, per_cell_rheology=True)
    sim = build_simulation(TestScene(build=build, ground=False, worlds=W, dt=2.5e-3), [flow], "cpu")
    flow.set_input("base", np.repeat(np.where(i < 2, 0.089, 0.077).reshape(1, -1), W, 0).astype(np.float32))
    flow.set_depth(np.repeat(np.where(i == 1, 0.024, np.where(i == 0, 0.001, 0.004)).reshape(1, -1), W, 0))
    flow.set_input("yield_stress", np.stack([np.where(i < 2, a, b).reshape(-1) for a, b in taus]).astype(np.float32))
    return sim, flow


@pytest.mark.compile_heavy
@pytest.mark.timeout(900)
def test_a_fin_held_by_the_films_own_law_rests_and_one_that_moves_does_not():
    """The rest ratio reads the donor's yield (8.2 kPa: 1.10); the film's face law also reads the stiffer pad beside
    it (30 kPa) and holds every face: ``held``, at rest. The same fin over a pad as soft as it (1 kPa) still moves
    after the 0.2 s watch: refused. Two worlds where one holds and one moves: refused. The watch steps a moving film
    on until it rests, at most ``SETTLE_S``: one creeping at its yield is still refused there."""
    sim, flow = _fin([(8.2e3, 3e4)])
    assert ST.film_rest(flow)["bound"] == pytest.approx(1.1025, abs=1e-3)
    held = ST.rest_after(sim, flow, None)
    assert held["rule"] == "held" and held["rest"] and held["over"] == 3 and held["face_max"] == 0.0
    sim, flow = _fin([(1e3, 1e3)])
    moving = ST.rest_after(sim, flow, None)
    assert moving["rule"] == "none" and not moving["rest"] and moving["face_max"] > 0.0
    sim, flow = _fin([(8.2e3, 3e4), (1e3, 1e3)])
    both = ST.rest_after(sim, flow, None)
    assert both["rule"] == "none" and not both["rest"]
    sim, flow = _fin([(1e3, 1e3)])                   # creeps at its yield (bound 1.000x): stepped on to the cap,
    watch = SG.Watch()                                # still moving, refused
    last = watch.after(sim, flow, None)
    assert not last["rest"] and last["periods"] == round(SG.FILM_SETTLE_S / SG.PERIOD) == watch.periods
    sm = watch.summary()
    assert sm["film.rule"] == "none" and sm["film.checks"] == 1 and sm["film.refused_checks"] == 1
    sim, flow = _fin([(1.5e3, 1.5e3)])                # slumps under the bound within the watch: rest by the bound
    assert SG.Watch().after(sim, flow, None)["rule"] == "bound"


class _Layer:
    """The paste layer's host API ``start_stage`` uses: set_start, its paste input, taus, the film's yield rule. As
    the real layer, it ages only the paste's cells: off them ``taus`` reads ``stale`` (the build age's tau_b)."""

    def __init__(self, C, tb, ts, paste=None, stale=None):
        from types import SimpleNamespace as NS
        self.paste = np.zeros((1, C), np.int32) if paste is None else np.asarray(paste, np.int32).reshape(1, C)
        self.tb, self.ts, self.stale, self.started = tb, ts, tb if stale is None else stale, {}
        self.config = NS(film_yield="surface")

    def set_start(self, paste, **kw):
        self.paste, self.started = np.asarray(paste), kw

    def inputs(self):
        return {"paste": _Arr(self.paste)}

    def taus(self):
        on = self.paste > 0
        return np.where(on, self.tb, self.stale), np.where(on, self.ts, self.stale)


class _FlowIn:
    """The film's host API ``start_stage`` uses, with a build-time yield of 900 Pa."""

    def __init__(self, C):
        self.ins = {"base": None, "yield_stress": _Arr(np.full((1, C), 900.0, np.float32)),
                    "bulk_yield": _Arr(np.full((1, C), 900.0, np.float32))}

    def set_input(self, name, v):
        self.ins[name] = _Arr(np.asarray(v))

    def set_depth(self, v):
        self.depth = np.asarray(v)

    def inputs(self):
        return self.ins


@pytest.mark.compile_heavy
@pytest.mark.timeout(900)
def test_the_watch_reads_the_smears_yield_at_its_age():
    """The layer writes the film's yields on the paste only, and ages only the paste, so a form-top smear keeps its
    build yield (the darby's 5 min). With the layer, every rest check first writes them as the paste has them now, the
    smear at the paste's age: the fin creeping at a stale 1 kPa rests by the bound at its aged 1.5 kPa; without it,
    or with the layer's own off-paste yield, it is refused at the cap (seed 4's darby hand-off, 7ef38a4cb)."""
    sim, flow = _fin([(1e3, 1e3)])
    pad = (np.arange(NX)[:, None] >= 2) & np.ones((1, NY), bool)          # the fin's form top (i < 2) is off it
    layer = _Layer(NX * NY, tb=1.5e3, ts=1.5e3, paste=pad, stale=1e3)
    r = SG.Watch().after(sim, flow, None, layer)
    assert r["rule"] == "bound" and r["rest"]
    assert np.all(flow.inputs()["yield_stress"].numpy() == np.float32(1.5e3))


def test_a_stage_starts_with_the_yields_its_paste_has_now(start0):
    """The film's yield inputs are written by the layer after each film step, so a fresh stage's first step would
    read the film material's build-time yield (900 Pa here). ``start_stage`` writes the carried paste's own: tau_s on
    the paste for a surface-layer film, tau_b off it (the forms' smear, their tb, at the paste's age: the layer's own
    reads its build age there) and as the bulk yield. A world with no paste keeps the layer's own."""
    f = dict(start0, age=7200.0)
    C = f["h"].size
    flow, layer = _FlowIn(C), _Layer(C, tb=4.0e4, ts=6.0e3, stale=1.6e3)                # off the paste: build age
    ST.start_stage(flow, layer, [f], lambda g: {"base": g["b"], "film": g["h"] - g["b"],
                                                  "paste": g["paste_mask"]})
    on = np.asarray(f["paste_mask"], bool).reshape(-1)
    ys, yb = flow.ins["yield_stress"].numpy()[0], flow.ins["bulk_yield"].numpy()[0]
    assert np.all(ys[on] == np.float32(6.0e3)) and np.all(yb[on] == np.float32(4.0e4))
    assert np.all(ys[~on] == np.float32(4.0e4)) and np.all(yb[~on] == np.float32(4.0e4))
    assert np.all(layer.started["age"] == 7200.0)
    tb = np.array([[1.0, 2.0, 5.0], [3.0, 3.0, 3.0]])
    assert ST.site_taus(tb, np.array([[0, 1, 0], [0, 0, 0]])).tolist() == [[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
