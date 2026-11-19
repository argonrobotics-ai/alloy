# SPDX-License-Identifier: Apache-2.0
"""The bag's crack length from its seams (alloy/tasks/gift_bag_tear.py) and the tearable bag's laws (physics ruling
2026-11-18: the card asks 40 VBD iterations; i stands, ii a pulled rim tears, iv nothing else moves; iii is in
test_gift_bag.py). The unit tests are numpy only; the laws build the bag."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.tasks import gift_bag_candle as T
from alloy.tasks.gift_bag_checks import Limits, standing
from alloy.tasks.gift_bag_tear import TEAR_TOL_MM, bag_not_torn, bag_seams, seam_lengths, torn_mm, torn_now
from alloy.topology.types import BondKind

CLOTH_EDGE = int(BondKind.CLOTH_EDGE)        # from the topology package, not the module under test

SIGMA, H = 35e6, 135e-6                   # Pa, m: a kraft sheet


def _bonds():
    lengths = np.array([0.008, 0.010, 0.004, 0.009, 0.0])              # m; the last is another bond kind
    kind = np.array([CLOTH_EDGE] * 4 + [1])
    force = np.where(kind == CLOTH_EDGE, lengths * SIGMA * H, 5.0)
    world = np.array([0, 0, 1, 1, 1])
    return force, kind, world, lengths


def test_seam_lengths_come_back_from_the_break_forces():
    force, kind, _, lengths = _bonds()
    got = seam_lengths(force, kind, SIGMA, H)
    assert np.allclose(got[:4], lengths[:4], rtol=1e-12, atol=0.0) and got[4] == 0.0
    assert seam_lengths(np.array([np.inf]), np.array([CLOTH_EDGE]), SIGMA, H)[0] == 0.0     # unbreakable


def test_torn_length_sums_the_broken_seams_per_world():
    force, kind, world, _ = _bonds()
    lengths = seam_lengths(force, kind, SIGMA, H)
    intact = np.array([1, 0, 0, 1, 0])                 # world 0: the 10 mm seam; world 1: the 4 mm seam (+ a non-seam)
    t = torn_mm(intact, world, lengths, 2)
    assert np.allclose(t, [10.0, 4.0], rtol=1e-9, atol=0.0)
    assert np.all(torn_mm(np.ones(5), world, lengths, 2) == 0.0)


def test_the_verdict_follows_the_references_five_millimetres():
    assert bag_not_torn(0.0)["bag_not_torn"] and bag_not_torn(TEAR_TOL_MM)["bag_not_torn"]
    assert not bag_not_torn(TEAR_TOL_MM + 0.1)["bag_not_torn"]
    assert bag_not_torn(None) == {"bag_not_torn": None}


# --- laws on the tearable bag (gift_bag_kraft_tear, film.vbd_iterations 40; physics ruling 2026-11-18) -----------

FRONT_Y = -0.0572                          # the bag's front face (mesh y min) [m]
RIM_MM = 1e3 * (0.2604 + T._heights(T.CandleSpec())["bag"])     # the rim's start height [mm] (the plain bag keeps it)


def _bag_scene(card: str, board: bool = True):
    """The bag (and board, unless ``board=False``) on the mat, the task's layout."""
    import alloy

    h = T._heights(T.CandleSpec())
    sc = alloy.Scene(dt=0.01 if board else PULL_DT)
    sc.add_ground()
    sc.add_box("mat", size=T.MAT, pos=(0.0, 0.0, 0.5 * T.MAT[2]), static=True, friction=0.9)
    sc.add_object("bag", card, pos=(0.0, 0.0, h["bag"]))
    if board:
        sc.add_object("board", "gift_bag_board", pos=(0.0, 0.0, h["board"]))
    return sc


def stand(card: str, seconds: float, device: str) -> dict:
    """The bag standing on the mat: the largest tilt, the rim at the end against its start [mm], seams torn, health."""
    import alloy

    sim = alloy.build(_bag_scene(card), device=device)
    L, seams = T._layout(T.CandleSpec()), None
    if card.endswith("_tear"):
        seams = bag_seams(sim, T.kraft_strength(), T.KRAFT_H)
    tilt = 0.0
    for _ in range(int(round(seconds / 0.05))):
        sim.run(0.05)
        p = sim.state(["bag"])["bag.points"][0]
        tilt = max(tilt, standing(L, p, Limits())["bag_tilt_deg"])
    rim = 1e3 * float(p[L.rim, 2].mean())
    torn = torn_now(sim, seams)
    return {"tilt_deg": tilt, "rim_drop_mm": RIM_MM - rim, "torn_mm": 0.0 if torn is None else float(torn[0]),
            "severity": sim.health()["worlds"][0]["severity"], "ok": bool(sim.check()["ok"])}


def _stands(r: dict) -> bool:
    return r["tilt_deg"] <= 1.0 and abs(r["rim_drop_mm"]) <= 5.0 and r["torn_mm"] == 0.0 and r["severity"] == "OK"


@pytest.mark.compile_heavy
def test_law_i_the_tear_bag_stands_on_the_cpu_short():
    """Law i, short (CPU, 0.3 s: the 3 s run is the GPU test below). At the film route's own 10 iterations the bag
    is falling by now (rim 88 mm down, tilt 10 deg at 0.3 s); at its card's 40 it stands."""
    r = stand("gift_bag_kraft_tear", 0.3, "cpu")
    assert _stands(r), r


@pytest.mark.gpu
def test_law_i_the_tear_bag_stands_three_seconds():
    """Law i (NVIDIA GPU): 3 s on the mat, tilt <= 1 deg, rim within 5 mm of where the plain bag stays, 0 torn."""
    r = stand("gift_bag_kraft_tear", 3.0, "cuda:0")
    plain = stand("gift_bag_kraft", 0.5, "cuda:0")
    assert abs(plain["rim_drop_mm"]) <= 1.0, plain
    assert _stands(r), r


PULL_DT = 0.0025                           # s: the pull's control period (the grip moves 0.25 mm a period)
GRIP_X = (0.0, 0.04)                       # m: the stretch of the front rim the grip holds


def _seam_particles(topo) -> np.ndarray:
    b = topo.bonds
    seam = b.kind.numpy() == CLOTH_EDGE
    return np.union1d(b.a.numpy()[seam], b.b.numpy()[seam])


def _hold(sim, ids: np.ndarray) -> None:
    """Hold particles kinematic from now on: mass 0 and inactive, as the film's own tear rigs pin a grip
    (alloy/solvers/film/rigs.py), here after the build."""
    m = sim.engine.model
    for arr in (m.particle_mass, m.particle_inv_mass, m.particle_flags):
        a = arr.numpy().copy()
        a[ids] = 0
        arr.assign(a)


def _place(sim, ids: np.ndarray, q: np.ndarray) -> None:
    for st in (sim.engine.state, sim.engine._next):
        x = st.particle_q.numpy().copy()
        x[ids] = q
        st.particle_q.assign(x)


def _grip_and_base(sim) -> tuple[np.ndarray, np.ndarray]:
    """The particles the pull moves (the front rim within GRIP_X) and holds (the bottom panel), seam particles left
    free in both."""
    L, rng = T._layout(T.CandleSpec()), sim.index.particle_range("bag")
    q = sim.engine.state.particle_q.numpy()
    rim, bottom = rng.start + np.asarray(L.rim), rng.start + np.asarray(L.bottom)
    front = rim[(q[rim, 1] < FRONT_Y + 0.003) & (q[rim, 0] >= GRIP_X[0]) & (q[rim, 0] <= GRIP_X[1])]
    seam = _seam_particles(sim.engine.topology)
    return np.setdiff1d(front, seam), np.setdiff1d(bottom, seam)


def pull_rim(device: str, lift: float = 0.1, seconds: float = 0.4) -> dict:
    """Law ii: the bag's bottom panel held, a 4 cm stretch of its front rim gripped and lifted at ``lift`` m/s for
    ``seconds``. Torn length [mm] (the readout) every 0.05 s, broken seams, health, and the length from the seams'
    rest geometry (each broken seam: half the distance to its two nearest seam neighbours)."""
    import alloy

    sim = alloy.build(_bag_scene("gift_bag_kraft_tear", board=False), device=device)
    seams = bag_seams(sim, T.kraft_strength(), T.KRAFT_H)
    topo = sim.engine.topology
    grip, base = _grip_and_base(sim)
    _hold(sim, np.concatenate([grip, base]))
    g0 = sim.engine.state.particle_q.numpy()[grip].copy()
    is_seam = topo.bonds.kind.numpy() == CLOTH_EDGE
    x0 = sim.engine.state.particle_q.numpy()[topo.bonds.a.numpy()[is_seam]].copy()
    steps, every, grown = int(round(seconds / PULL_DT)), int(round(0.05 / PULL_DT)), []
    for k in range(1, steps + 1):
        _place(sim, grip, g0 + np.array([0.0, 0.0, lift * k * PULL_DT]))
        sim.step(1)
        if k % every == 0:
            grown.append(round(float(torn_now(sim, seams)[0]), 1))
    broken = topo.bonds.intact.numpy()[is_seam] == 0
    d = np.linalg.norm(x0[:, None, :] - x0[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    d[d < 1e-6] = np.inf                                # seams sharing a vertex where tear lines cross
    geo = 0.5 * np.sort(d, axis=1)[:, :2].sum(axis=1)[broken].sum() * 1e3
    return {"torn_mm": grown[-1], "grown_mm": grown, "broken": int(broken.sum()), "geometry_mm": float(geo),
            "grip": len(grip), "base": len(base), "severity": sim.health()["worlds"][0]["severity"],
            "ok": bool(sim.check()["ok"])}


@pytest.mark.gpu
def test_law_ii_a_pulled_rim_tears_the_bag_and_the_readout_counts_it():
    """Law ii (NVIDIA GPU): a pulled rim tears the bag, the tear grows with the pull, and the readout from the intact
    flags matches the broken seams' length from their rest geometry."""
    r = pull_rim("cuda:0")
    assert r["ok"] and r["broken"] > 0 and r["torn_mm"] > 5.0, r
    assert r["grown_mm"][0] <= r["grown_mm"][-1] and abs(r["torn_mm"] - r["geometry_mm"]) <= 0.3 * r["geometry_mm"], r


@pytest.mark.compile_heavy
def test_law_iv_only_the_tear_card_asks_for_more_iterations():
    """Law iv, in the suite: the plain bag, the tear sheet and an add_cloth film keep the film route's 10 iterations;
    the tear bag runs its card's 40 (the full probe over every shipped film card is in the merge's record)."""
    import alloy

    def iters(add) -> int:
        sc = alloy.Scene(dt=0.01)
        sc.add_ground()
        add(sc)
        sim = alloy.build(sc, device="cpu")
        return next(x for x in sim.engine.solvers if hasattr(x, "film_topology")).iterations

    assert iters(lambda sc: sc.add_object("bag", "gift_bag_kraft", pos=(0, 0, 0.004))) == 10
    assert iters(lambda sc: sc.add_object("w", "paper_wrap_tear_sheet", pos=(0, 0, 0.004))) == 10
    assert iters(lambda sc: sc.add_cloth("s", size=(0.1, 0.1), pos=(0, 0, 0.01), resolution=6, thickness=1e-4,
                                         material="paper_kraft")) == 10
    assert iters(lambda sc: sc.add_object("bag", "gift_bag_kraft_tear", pos=(0, 0, 0.004))) == 40
