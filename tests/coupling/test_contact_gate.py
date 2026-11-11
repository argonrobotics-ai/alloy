# SPDX-License-Identifier: Apache-2.0
"""Lever #2 ([internal note] notes/mixed-speed-1107.md): the contact-aware exchange clock (``ClockPolicy.contact_gate``).

CPU, through IntegratedSimulation, on WP4's box on cloth (a 0.2 kg MuJoCo box over a VBD cloth, lagged pair), 2 worlds.
Its proxy bound (0.5 sqrt(0.2 / 1e5) = 0.71 ms) sets 16 exchanges per 10 ms; the 2 ms cap alone allows 8.
- Near from the start: the gated run equals today's clock bitwise (CPU floor 0).
- Dropped from 10-14 cm: the free fall runs 8 exchanges per period; each box is armed before its first contact force
  (no safety-net re-run); in contact the gated run stays far closer to today's than a run with the bound off (the
  clock lever that drifted a draped towel 17.5 mm). Its only offset is the free-fall integration at 8 exchanges.
- From one state with every proxy armed, the gated run equals today's clock bitwise: in contact the gate changes
  nothing.
- Safety net: a margin that never arms leaves arming to touches; every touch re-runs its period at today's clock, and
  no accepted period has a coupling force on an unarmed proxy.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.core.registry import isolated_registry
from alloy.coupling.clocks import ClockController
from alloy.coupling.types import ClockPolicy, SchedulerPolicy
from alloy.testing import integration as I
from alloy.testing.coupling import ClothSceneSpec, cloth_scheduler, coupling_box_on_cloth

pytestmark = [pytest.mark.integration, pytest.mark.timeout(1800)]

SPEC = ClothSceneSpec(box_mass=0.2, cells=8)
CONTROL_DT = 0.01
FAR = [0.10, 0.14]
PERIODS = 30                    # the boxes land at periods 15-17; about 13 periods in contact


def _run(offsets, periods: int, *, gate: bool, margin: float = 0.01, bound_off: bool = False,
         gate_off_at: int | None = None, graph: str = "off", stats: dict | None = None) -> list[dict]:
    """Per period: exchanges, armed proxies, whether the accepted attempt touched unarmed, digests, box heights."""
    from alloy.integration import build_simulation
    from alloy.scene.builtin_plugins import register_builtin_solvers

    with isolated_registry(empty=True):
        register_builtin_solvers()
        model = coupling_box_on_cloth(len(offsets), SPEC, drop_offsets=offsets, device="cpu")
        clock = ClockPolicy(contact_gate=gate, gate_margin=margin)
        sched = cloth_scheduler(model, SPEC, policy=SchedulerPolicy(clock=clock), vbd_adaptive=False)
        built = I.built_from_model(model, [I.entry_spec(model, n, 1 / 480) for n in ("mjc", "vbd")],
                                   control_dt=CONTROL_DT)
        sim = build_simulation(built, "standard", stepper=sched, graph=graph)
        if bound_off:
            sched._interface_bound = lambda: math.inf
        rows = []
        for k in range(periods):
            if gate_off_at is not None and k == gate_off_at:
                sched._gate = None
            sim.step_period()
            notes = sim._last_coupling.notes or {}
            rows.append({"exch": sim.clock.steps_per_period, "armed": notes.get("gate_armed"),
                         "rerun": bool(notes.get("gate_rerun")), "reruns": notes.get("gate_reruns", 0),
                         "digest": I.digests(sim).copy(), "z": sim.state.body_q.numpy()[:, 2].copy()})
        if stats is not None:
            stats.update(sim.graph_stats())
        return rows


@pytest.fixture(scope="module")
def far() -> dict[str, list[dict]]:
    return {"today": _run(FAR, PERIODS, gate=False), "gate": _run(FAR, PERIODS, gate=True),
            "bound_off": _run(FAR, PERIODS, gate=False, bound_off=True)}


def test_near_proxies_keep_todays_clock_bitwise() -> None:
    today, gated = _run([0.0, 0.002], 10, gate=False), _run([0.0, 0.002], 10, gate=True)
    assert all(r["armed"] == 2 for r in gated)
    for k, (a, b) in enumerate(zip(today, gated)):
        assert a["exch"] == b["exch"] and np.array_equal(a["digest"], b["digest"]), f"period {k}"


def test_far_proxies_release_the_clock_and_arm_before_contact(far) -> None:
    today, gated, off = far["today"], far["gate"], far["bound_off"]
    released = [k for k, r in enumerate(gated) if r["armed"] == 0]
    assert released and all(gated[k]["exch"] == 8 and today[k]["exch"] == 16 for k in released)
    assert not any(r["rerun"] for r in gated) and gated[-1]["reruns"] == 0, "a box touched before it was armed"
    armed_from = next(k for k in range(1, len(gated)) if gated[k]["armed"])
    dz = [1e3 * float(np.abs(a["z"] - b["z"]).max()) for a, b in zip(today, gated)]
    dz_off = [1e3 * float(np.abs(a["z"] - b["z"]).max()) for a, b in zip(today, off)]
    arrival, contact, contact_off = dz[armed_from - 1], max(dz[armed_from:]), max(dz_off[armed_from:])
    print(f"free fall at 8 exchanges: {arrival:.3f} mm at arming (period {armed_from}); in contact gate "
          f"{contact:.3f} mm, bound off {contact_off:.3f} mm")
    assert contact <= 3.0 * arrival + 0.05 and contact < 0.5 * contact_off


def test_armed_gate_equals_todays_clock_from_the_same_state(far) -> None:
    gated = far["gate"]
    all_armed = next(k for k in range(1, len(gated)) if gated[k]["armed"] == 2)
    later = [k for k in range(all_armed, len(gated)) if gated[k]["armed"] == 2]
    switched = _run(FAR, later[-1] + 1, gate=True, gate_off_at=all_armed)
    for k in later:
        assert np.array_equal(switched[k]["digest"], gated[k]["digest"]), f"period {k}"


def test_gate_after_captured_periods_equals_eager() -> None:
    """WP4's review: after a replayed period the gate reads the State object the period was captured from, which
    holds the period's result (IntegratedSimulation keeps its State objects and copies an odd chunk count back,
    SCR-WP8-3). CPU graphs (graph="on") across a release and a re-arm: the arming, the clock and every digest equal
    the eager run's."""
    stats: dict = {}
    eager, graphs = _run(FAR, 12, gate=True), _run(FAR, 12, gate=True, graph="on", stats=stats)
    assert stats["mode"] == "graph" and stats["launches"] > 0, stats
    armed = [r["armed"] for r in eager]
    assert 0 in armed[1:] and armed[-1] == 2, armed                       # released, then re-armed
    assert [r["armed"] for r in graphs] == armed
    for k, (a, b) in enumerate(zip(eager, graphs)):
        assert a["exch"] == b["exch"] and np.array_equal(a["digest"], b["digest"]), f"period {k}"


class _WholeRerun:
    """StepperLike: attempt 1 asks for a whole-period re-run (notes ``rerun_all``, the gate's safety net) while it also
    reports world 1 short of passes and world 2 escalated beyond onset; attempt 2 is clean."""

    def __init__(self, solver, dt_x: float) -> None:
        from alloy.testing import fakes as F

        self.inner = F.FakeStepper(solver, dt_x=dt_x)
        self.world_count, self.attempt, self.masks, self.reports = self.inner.world_count, 0, [], []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def choose_clock(self, control_dt, last):
        self.attempt += 1
        return self.inner.choose_clock(control_dt, last)

    def report(self):
        import dataclasses

        from alloy.coupling.types import CouplingReport, WatchState

        w = self.world_count
        short, ws = np.zeros(w, bool), np.zeros(w, np.int32)
        if self.attempt == 1:
            short[1], ws[2] = True, int(WatchState.ESCALATED)
        rep = dataclasses.replace(CouplingReport.quiet(self.inner.dt_x, {self.inner.name: 1}, w), watch_state=ws,
                                  notes={"short": short, "rerun_all": self.attempt == 1})
        self.reports.append(rep)
        return rep

    def rollback_requested(self, report) -> bool:
        return bool((report.notes or {}).get("rerun_all"))

    def on_rollback(self, world_mask=None) -> None:
        self.masks.append(None if world_mask is None else np.asarray(world_mask).astype(int).tolist())


def test_a_whole_period_rerun_in_the_simulation() -> None:
    """WP12's review, IntegratedSimulation's two branches: a ``rerun_all`` report re-runs every world even with short
    and escalated worlds in it (the mask is all ones, on_rollback gets None), counts as a rollback and not a short
    re-run, and leaves the graphs (the gate's re-run drops the attempt's escalations: no mode or rho switch)."""
    from types import SimpleNamespace

    from alloy.integration.simulation import IntegratedSimulation, _rerun_mask
    from alloy.testing import fakes as F
    from alloy.testing.scenes import box_on_plane

    sc = box_on_plane(4, gap=0.3)
    built = I.built_from_model(sc.model, [I.entry_spec(sc.model, "fake", 1.25e-3)], control_dt=0.01)
    stepper = _WholeRerun(F.FakeSolver(sc.model, F.FakeSolver.Config(floor_z=None)), 1.25e-3)
    sim = IntegratedSimulation(built, stepper=stepper, use_graph=False)
    invalidated: list[int] = []
    sim._graphs = SimpleNamespace(invalidate=lambda: invalidated.append(1))
    sim.eager_keys[(1.25e-3, (("fake", 1),))] = 0      # the fake cache never captures: every period runs eagerly
    stepper.attempt = 0                                 # construction asks for the clock once
    sim.step_period()
    assert _rerun_mask(stepper.reports[0]).tolist() == [1, 1, 1, 1]
    assert stepper.attempt == 2 and stepper.masks == [None]
    assert sim.rollbacks == 1 and sim.short_reruns == 0 and invalidated == []


def test_safety_net_rearms_a_proxy_that_touched_unarmed() -> None:
    rows = _run([0.0, 0.002], 8, gate=True, margin=-1.0)          # never near: only touches arm
    assert rows[-1]["reruns"] >= 1, "nothing touched unarmed: the safety net was not exercised"
    assert not any(r["rerun"] for r in rows), "an accepted period kept a coupling force on an unarmed proxy"
    rerun_periods = [k for k in range(1, len(rows)) if rows[k]["reruns"] > rows[k - 1]["reruns"]]
    assert all(rows[k]["exch"] >= 16 for k in rerun_periods)         # re-run at today's clock (or finer)


def test_all_armed_bound_reads_todays_pair_terms(monkeypatch) -> None:
    """With every proxy armed (the gate's start) its bound is today's: the same per-pair hooks (``_pair_k_c``,
    ``_pair_c_d``), so a test or subclass that changes one changes both. The damping here binds (10 us)."""
    from alloy.coupling.mixin import SchedulerMixin
    from alloy.coupling.scheduler import Scheduler

    monkeypatch.setattr(Scheduler, "_pair_c_d", lambda self, pair: 1.0e4)
    bounds = {}
    for damping in (False, True):
        model = coupling_box_on_cloth(2, SPEC, drop_offsets=FAR, device="cpu")
        clock = ClockPolicy(contact_gate=True, damping_bound=damping)
        sched = cloth_scheduler(model, SPEC, policy=SchedulerPolicy(clock=clock), vbd_adaptive=False)
        assert sched._gate.armed_host.all()
        bounds[damping] = sched._gate.bound()
        assert bounds[damping] == SchedulerMixin._interface_bound(sched)
    assert bounds[True] < 0.1 * bounds[False]


def _heavy_light_post(heavy_dz: float):
    """One world: a 2 kg box ``heavy_dz`` above its 12 mm drop gap over the cloth, a 0.05 kg box 0.6 m up, and a fixed
    post 5 mm under the cloth (infinite own bound). Today's bound is the light box's."""
    from alloy._vendor import newton
    import warp as wp

    from alloy.testing.coupling import BOX_HALF, CLOTH_Z, _world_builder

    spec = ClothSceneSpec(box_mass=2.0, cells=8)
    b = _world_builder(spec, heavy_dz)
    cfg = b.default_shape_cfg.copy()
    cfg.density = 0.05 / (2.0 * BOX_HALF) ** 3
    light = b.add_body(xform=wp.transform(p=wp.vec3(-0.15, -0.15, CLOTH_Z + 0.6), q=wp.quat_identity()), label="light")
    b.add_shape_box(light, hx=BOX_HALF, hy=BOX_HALF, hz=BOX_HALF, cfg=cfg)
    at = wp.transform(p=wp.vec3(0.15, 0.15, CLOTH_Z - BOX_HALF - 0.005), q=wp.quat_identity())
    post = b.add_link(xform=at, label="post")
    b.add_shape_box(post, hx=BOX_HALF, hy=BOX_HALF, hz=BOX_HALF, cfg=b.default_shape_cfg.copy())
    b.add_articulation([b.add_joint_fixed(-1, post, parent_xform=at)])
    top = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(top)
    top.add_world(b)
    top.color()
    model = top.finalize(device="cpu")
    model.soft_contact_ke, model.soft_contact_mu = spec.contact_ke, spec.contact_mu
    clock = ClockPolicy(contact_gate=True)
    return cloth_scheduler(model, spec, policy=SchedulerPolicy(clock=clock), vbd_adaptive=False), model


@pytest.mark.parametrize("heavy_near", [True, False])
def test_a_pair_arms_whole_and_fixed_bodies_never_arm_it(heavy_near) -> None:
    """A near movable proxy arms its whole pair, so the clock is today's (the light box's bound, not the heavy box's:
    per-proxy arming let a towel drape over heavy Panda links at their coarse step, 42 mm off on the L4). A fixed body
    at the cloth cannot hold the clock: it stays armed but never arms the pair (a towel on a table leaves it free)."""
    from alloy.coupling.mixin import SchedulerMixin

    sched, model = _heavy_light_post(0.0 if heavy_near else 0.5)
    gate = sched._gate
    assert np.isinf(gate.ends.bound_s).tolist() == [False, False, True]          # heavy, light, post
    state = model.state()
    for _ in range(gate.hold_periods + 1):
        gate.evaluate(state, CONTROL_DT)
    if heavy_near:
        assert gate.armed_host.all() and gate.bound() == SchedulerMixin._interface_bound(sched)
        assert gate.bound() < 0.5 * gate.ends.bound_s[0]                        # the light box's, not the heavy's
    else:
        assert gate.armed_host.tolist() == [False, False, True] and math.isinf(gate.bound())
        sched._band_bound = 1.0e-4                     # WP4's R2-E3 contact-band bound holds with nothing armed
        assert sched._interface_bound() == 1.0e-4


def test_the_gate_on_wp4s_p3_chain() -> None:
    """MuJoCo -> VBD -> SPH (WP4's P3 chain, test_p3_chain.py's scene): the cloth's particles mirrored into the liquid
    have no bound (contact-free), so they are always armed and never arm their pair; the block resting on the cloth
    arms its pair, so the bound is today's."""
    import dataclasses

    import alloy
    import alloy.coupling.compose as compose
    from alloy.coupling.contact_gate import gate_for
    from alloy.coupling.mixin import SchedulerMixin

    if not compose.proxy_dag_supported():
        pytest.skip("Newton without N-entry proxy coupling (pin < 11-07)")
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_open_box("tank", inner=(0.05, 0.05, 0.05), wall=0.006, pos=(0, 0, 0))
    scene.add_liquid("water", size=(0.05, 0.05, 0.016), pos=(0, 0, 0.008))
    scene.add_cloth("sheet", size=(0.04, 0.04), pos=(0, 0, 0.020), resolution=6, mass=0.003)
    scene.add_box("block", size=(0.03, 0.03, 0.03), pos=(0, 0, 0.037), mass=0.2)
    sim = alloy.build(scene, device="cpu")
    st = sim.engine.stepper
    assert st.clock_mode() is None                                         # today's clock: nothing to name
    st._alloy_policy = dataclasses.replace(st._alloy_policy,
                                           clock=dataclasses.replace(st._alloy_policy.clock, contact_gate=True))
    st._gate = gate = gate_for(st)
    assert st.clock_mode()["gated_pairs"] == ["mujoco -> vbd", "vbd -> sph"]
    assert gate.ends.dest_names == ["vbd", "sph"] and gate.ends.is_body.tolist() == [1] + [0] * (gate.count - 1)
    assert gate.always.tolist() == [False] + [True] * (gate.count - 1)
    for _ in range(gate.hold_periods + 1):
        gate.evaluate(sim.engine.state, CONTROL_DT)
    assert gate.armed_host.all() and gate.bound() == SchedulerMixin._interface_bound(st)


def test_a_gate_change_jumps_the_clock_and_the_rungs() -> None:
    """When the gate changes the armed pairs (``jump``), dt_x goes straight to what the new bound needs; when that is
    finer, each adaptive rung goes straight to the larger of its estimator's rung and the rung that keeps its own step
    (dt_x / rung). Without ``jump`` the hysteresis holds them (one grid or ladder step per 10 quiet periods): a released
    clock stays fine, and a re-armed pair keeps the released rung (on the L4 a re-armed towel ran 78 us with VBD's 8
    substeps). WP4's review: an entry whose rung rose on its own estimate keeps its own step through the jump."""
    from alloy.coupling.clocks import pick_rung

    policy = ClockPolicy(downshift_quiet_periods=10)
    stable = {"a": 2.0e-4}                                   # the entry's own stable step [s]

    def rung(dt: float, est: float = stable["a"]) -> int:
        return pick_rung(dt, est, policy.safety, policy.substep_ladder)

    fine, coarse = 0.01 / 16, 0.01 / 8                       # the bound's grid value; the 2 ms cap's
    assert rung(coarse) == 2 * rung(fine) > 1
    held = ClockController(policy, {"a": 1}, {"a": True})
    held.decide(0.01, fine, stable)
    assert held.decide(0.01, math.inf, stable).dt_x == pytest.approx(fine)                 # hysteresis: still fine
    for hold in (False, True):
        c = ClockController(policy, {"a": 1}, {"a": True})
        c.decide(0.01, fine, stable)
        d = c.decide(0.01, math.inf, stable, jump=True)                                     # released
        assert d.dt_x == pytest.approx(coarse) and d.substeps["a"] == rung(coarse)
        d = c.decide(0.01, fine, stable, jump=not hold)                                     # re-armed
        assert d.dt_x == pytest.approx(fine)
        assert d.substeps["a"] == (rung(coarse) if hold else rung(fine))
    whip, calm = coarse / 8 / policy.safety, 1.0                                            # rung 8, then rung 1
    c = ClockController(policy, {"a": 1}, {"a": True})
    assert c.decide(0.01, coarse, {"a": whip}).substeps["a"] == 8 and rung(fine, calm) == 1
    assert c.decide(0.01, coarse, {"a": calm}).substeps["a"] == 8                           # its own hysteresis
    d = c.decide(0.01, fine, {"a": calm}, jump=True)                                        # another pair arms
    assert d.substeps["a"] == 4 and fine / 4 == pytest.approx(coarse / 8)                   # its own step kept
