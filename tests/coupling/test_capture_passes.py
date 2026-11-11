# SPDX-License-Identifier: Apache-2.0
"""Lever #1 ([internal note] notes/mixed-speed-1107.md): a capture runs only the coupling passes that converge worlds.

CPU stands in for a capture by patching ``capture_passes.capturing`` (the loops then unroll their passes as under CUDA
capture). Two scenes, 4 worlds each: WP4's ADMM ball + cloth (``admm_model``, gamma 0, iterations 2..20 per world) and
the box on cloth with a staggered pair (``cloth_scheduler``, Aitken, the proxy loop). Each runs three ways from the
same start: the residual-gated host loop (reference), the full unroll (k_max passes, frozen worlds keep their result)
and the capped unroll (k_min passes, short worlds flagged). The full unroll must equal the reference bitwise. The
capped one must flag exactly the worlds the reference iterated past k_min, and each world must follow the reference
bitwise until its first short exchange (worlds are independent: test_admm's batch invariance).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.coupling import capture_passes as cpm
from alloy.coupling import residuals as rs
from alloy.coupling.capture_passes import CapturedPasses, short_worlds
from alloy.testing.coupling import (ClothSceneSpec, admm_model, admm_scheduler, cloth_scheduler,
                                    coupling_box_on_cloth)

DT = 1.0 / 480.0
WORLDS = 4
SPEC = ClothSceneSpec(box_mass=0.2, cells=8)
STEPS = {"admm": 30, "staggered": 48}       # the staggered boxes land on the cloth from exchange 31 on


def _scene(name: str):
    if name == "admm":
        model = admm_model([0.0, 0.01, 0.02, 0.03], device="cpu")
        return model, admm_scheduler(model, gamma=0.0)
    model = coupling_box_on_cloth(WORLDS, SPEC, drop_offsets=[0.0, 0.003, 0.006, 0.009], device="cpu")
    return model, cloth_scheduler(model, SPEC, mode="staggered", vbd_adaptive=False)


def _report(cp: CapturedPasses) -> SimpleNamespace:
    return SimpleNamespace(notes=cp.notes())


def test_captured_passes_bookkeeping() -> None:
    cp = CapturedPasses(3, "cpu", enabled=True, passes=2, raise_after=2, window=4, decay_after=3)
    assert cp.captured(2, 20) == 20 and cp.graph_key(2, 20) == ()        # no driver re-runs short worlds yet
    cp.driver = True
    assert cp.captured(2, 20) == 2 and cp.capped(2, 20) and cp.graph_key(2, 20) == ("passes", 2)
    assert cp.captured(2, 2) == 2 and not cp.capped(2, 2) and cp.graph_key(2, 2) == ()
    off = CapturedPasses(3, "cpu", enabled=False, passes=2)
    assert off.captured(2, 20) == 20 and off.graph_key(2, 20) == () and off.notes() == {}
    converged = wp.array([int(rs.ACTIVE), int(rs.FROZEN), int(rs.ACTIVE)], dtype=wp.int32, device="cpu")

    cp.begin_attempt(False, 2)
    cp.mark(converged)
    rep = _report(cp)
    assert short_worlds(rep).tolist() == [True, False, True]
    assert cp.wants_rerun(rep) and cp.accept(rep, 2, 20) and cp.passes == 2      # one short period: no raise yet
    cp.begin_attempt(True, 2)                                                      # the re-run: full loop, no flags
    assert cp.full and cp.captured(2, 20) == 20 and cp.graph_key(2, 20) == ("passes", 20)
    assert not short_worlds(_report(cp)).any() and not cp.wants_rerun(rep)

    cp.begin_attempt(False, 2)                                                     # next period: capped again
    assert not cp.full and cp.graph_key(2, 20) == ("passes", 2)
    cp.mark(converged)
    assert cp.accept(_report(cp), 2, 20) and cp.passes == 4 and cp.reruns == 2     # 2 short periods within 4
    cp.begin_attempt(True, 2)
    for _ in range(3):                                                             # 3 quiet periods halve it back
        cp.begin_attempt(False, 2)
    assert cp.passes == 2 and cp.captured(2, 20) == 2
    assert not short_worlds(SimpleNamespace(notes=None)).any()

    once = CapturedPasses(3, "cpu", enabled=True, driver=True, passes=2, raise_after=2, window=4)
    once.begin_attempt(False, 2)
    for _ in range(2):            # a short attempt, then an escalation re-run (capped again) short in the same period
        once.mark(converged)
        assert once.accept(_report(once), 2, 20)
        once.begin_attempt(True, 2)
        once.full = False
    assert once.passes == 2, "two short attempts in one period count as one short period (WP4's review)"


def test_a_whole_rollback_gives_every_world_its_attempt_start() -> None:
    """WP4's review: when Simulation rolls every world back (a state array without a world axis, or all worlds
    short), it restores the period start, which predates begin_period. Every world must get the watchdog arrays its
    attempt began with, not only the short ones: here world 0 ended the last period ESCALATED (accepted), and its
    replay must not start from that."""
    from alloy.coupling.types import WatchState

    model = admm_model([0.0, 0.01, 0.02, 0.03], device="cpu")
    sched = admm_scheduler(model, gamma=0.0)
    sched.allow_capped_captures()
    wd = sched._watchdog
    wd.watch_state.assign(np.array([int(WatchState.ESCALATED), 0, 0, 0], dtype=np.int32))
    period_start = {k: v.numpy().copy() for k, v in wd.state_arrays().items()}
    sched.choose_clock(0.02, None)                     # begin_period clears transient watch levels
    attempt_start = {k: v.numpy().copy() for k, v in wd.state_arrays().items()}
    assert attempt_start["watch.watch_state"][0] != int(WatchState.ESCALATED)
    for k, v in wd.state_arrays().items():             # Simulation's whole rollback: every world to the period start
        v.assign(period_start[k])
    short = np.array([False, True, False, False])
    report = SimpleNamespace(watch_state=np.zeros(4, np.int32), notes={"short": short})
    assert sched.rollback_requested(report)
    sched.on_rollback(None)
    for k, v in wd.state_arrays().items():
        assert np.array_equal(v.numpy(), attempt_start[k]), k
    dt_x, _ = sched.choose_clock(0.02, report)         # the replay: same clock, full loop
    assert dt_x == sched._dt_x and sched.iteration_loop.captured_passes.full


def _trajectory(name: str, mode: str, monkeypatch, start: int = 0) -> dict[str, np.ndarray]:
    """``mode``: "host" (residual-gated loop), "full" (k_max unrolled) or "capped" (k_min unrolled), from exchange
    ``start`` on (the host loop before it)."""
    step = [0]
    monkeypatch.setattr(cpm, "capturing", lambda device: mode != "host" and step[0] >= start)
    model, sched = _scene(name)
    loop, cp = sched.iteration_loop, sched.iteration_loop.captured_passes
    assert cp.enabled and not cp.active and cp.passes == loop.cfg.k_min < loop.cfg.k_max
    sched.allow_capped_captures()                  # as IntegratedSimulation does: it re-runs short worlds
    cp.full = mode == "full"
    s0, s1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    pipe = newton.CollisionPipeline(model)
    contacts = pipe.contacts()
    nb, npart = model.body_count // WORLDS, model.particle_count // WORLDS
    rec: dict[str, list] = {"body_q": [], "particle_q": [], "iters": [], "short": [], "k_min": loop.cfg.k_min}
    for _ in range(STEPS[name]):
        cp.short.zero_()
        s0.clear_forces()
        pipe.collide(s0, contacts)
        sched.step(s0, s1, control, contacts, DT)
        s0, s1 = s1, s0
        rec["body_q"].append(s0.body_q.numpy().reshape(WORLDS, nb, 7).copy())
        rec["particle_q"].append(s0.particle_q.numpy().reshape(WORLDS, npart, 3).copy())
        rec["iters"].append(loop.iters_used.numpy().copy())
        rec["short"].append(cp.short.numpy().astype(bool))
        step[0] += 1
    return {k: np.array(v) for k, v in rec.items()}


@pytest.fixture(scope="module", params=["admm", "staggered"])
def runs(request) -> dict[str, dict[str, np.ndarray]]:
    """The capped run switches to captures at the first exchange where the reference has worlds converging in k_min
    passes beside worlds that need more, so both kinds start that exchange from the reference state."""
    mp = pytest.MonkeyPatch()
    try:
        name = request.param
        out = {mode: _trajectory(name, mode, mp) for mode in ("host", "full")}
        past = out["host"]["iters"] > int(out["host"]["k_min"])
        mixed = np.flatnonzero(past.any(axis=1) & ~past.all(axis=1))
        assert mixed.size, "the reference never mixes worlds past k_min with worlds within it"
        out["capped"] = _trajectory(name, "capped", mp, start=int(mixed[0]))
        out["start"] = int(mixed[0])
        return out
    finally:
        mp.undo()


def test_full_unroll_equals_the_residual_gated_loop(runs) -> None:
    host, full = runs["host"], runs["full"]
    assert host["iters"].max() > host["k_min"], "the scene must have worlds that iterate past k_min"
    for key in ("body_q", "particle_q", "iters"):
        assert host[key].tobytes() == full[key].tobytes(), key
    assert not full["short"].any()


def test_capped_unroll_flags_exactly_the_worlds_past_k_min(runs) -> None:
    """Per world (worlds are independent): until its first short exchange it follows the reference bitwise with the
    reference converging in <= k_min passes, and at that exchange the reference needed more than k_min."""
    host, capped, start, k_min = runs["host"], runs["capped"], runs["start"], int(runs["host"]["k_min"])
    short = capped["short"]
    assert short[start].any() and not short[start].all(), "the first capped exchange must mix short and converged"
    assert not short[:start].any()
    steps = len(short)
    for w in range(WORLDS):
        first = int(np.argmax(short[:, w])) if short[:, w].any() else steps
        assert (host["iters"][start:first, w] <= k_min).all(), w
        for key in ("body_q", "particle_q"):
            assert capped[key][:first, w].tobytes() == host[key][:first, w].tobytes(), (w, key)
        if first < steps:
            assert host["iters"][first, w] > k_min, w
    print(f"capped from exchange {start}: first short exchange per world "
          f"{[int(np.argmax(short[:, w])) for w in range(WORLDS)]}, "
          f"short worlds per exchange {short.sum(axis=1).tolist()}")
