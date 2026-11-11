# SPDX-License-Identifier: Apache-2.0
"""Device twin log, world ranges, per-world measures and run_law mechanics (SPEC §3.10)."""

from __future__ import annotations

import dataclasses
import math

from alloy._vendor import newton
import numpy as np
import pytest
import warp as wp

from alloy.bench import runner as RN
from alloy.bench.measures import EnergyMeter, MomentumMeter, reset_heights
from alloy.bench.rigjudge import judge_rig
from alloy.bench.twin import DeviceTwinLog, WorldRanges, record_body_position
from alloy.bench.types import Condition, LawSpec, Tolerance
from alloy.testing.bench import fake_rig_dataset


def _spec(**kw) -> LawSpec:
    base = dict(id="T.run", suite="analytic", metrics={"err": Tolerance(abs=0.5, source="t")}, stat={"err": "mean"},
                dt=1e-2, substeps=2, duration_s=0.1, worlds_per_condition=2, perturb={"k": 0.1})
    base.update(kw)
    return LawSpec(**base)


def test_log_columns_and_buffers() -> None:
    log = DeviceTwinLog(3, 10, 0.01, "cpu", stride=4)
    assert log.n_samples == 3 and [log.column(s) for s in (0, 4, 5, 8, 11)] == [0, 1, -1, 2, -1]
    assert np.allclose(log.times(), [0.0, 0.04, 0.08])
    ch = log.channel("a")
    assert ch.shape == (3, 3) and np.isnan(log.series("a")).all()
    assert log.channel("v", wp.vec3d).shape == (3, 3) and log.accumulator("g").shape == (3,)
    with pytest.raises(KeyError):
        log.series("missing")


def _two_world_model() -> newton.Model:
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -10.0))
    for w in range(2):
        b.begin_world()
        body = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0 + w), wp.quat_identity()))
        b.add_shape_box(body, hx=0.1, hy=0.2, hz=0.3)
        b.end_world()
    return b.finalize(device="cpu")


def test_world_ranges_and_recording() -> None:
    m = _two_world_model()
    rng = WorldRanges.bodies(m)
    assert rng.start.numpy().tolist() == [0, 1] and rng.end.numpy().tolist() == [1, 2]
    with pytest.raises(ValueError):
        WorldRanges._from_item_world(np.array([0, 1, 0]), 2, "cpu")
    log = DeviceTwinLog(2, 1, 0.01, "cpu")
    record_body_position(log, "", m.state().body_q, rng.start, 0)
    assert np.allclose(log.series("z")[:, 0], [1.0, 2.0])


def test_energy_and_momentum_match_numpy() -> None:
    m = _two_world_model()
    s = m.state()
    qd = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 2.0], [0.0, 0.5, 0.0, 1.0, 0.0, 0.0]], np.float32)
    s.body_qd.assign(qd)
    log = DeviceTwinLog(2, 1, 0.01, "cpu")
    assert np.allclose(reset_heights(m, s), [1.0, 2.0])
    EnergyMeter(m, s, log).update(s, 0)
    mass = m.body_mass.numpy()
    inertia = m.body_inertia.numpy()
    # KE = 1/2 m v^2 + 1/2 w.I.w (identity rotation); PE datum = reset height -> 0
    ke = 0.5 * mass * np.sum(qd[:, :3] ** 2, 1) + 0.5 * np.einsum("bi,bij,bj->b", qd[:, 3:], inertia, qd[:, 3:])
    assert np.allclose(log.series("energy")[:, 0], ke, rtol=1e-6)
    MomentumMeter(m, log).update(s, 0)
    p = log.series("p")[:, 0]
    assert np.allclose(p, mass[:, None] * qd[:, :3], rtol=1e-6)
    x = np.array([[0, 0, 1.0], [0, 0, 2.0]])
    ang = np.cross(x, mass[:, None] * qd[:, :3]) + np.einsum("bij,bj->bi", inertia, qd[:, 3:])
    assert np.allclose(log.series("L")[:, 0], ang, rtol=1e-6)


def test_plan_worlds_quick_cap_and_samples() -> None:
    conds = [Condition(f"c{i}", {}) for i in range(3)]
    ctx = RN.RunContext(_spec(worlds_per_condition=16), "cpu", 0, "xpbd", True)
    worlds = RN.plan_worlds(_spec(worlds_per_condition=16), conds, ctx)
    assert len(worlds) == 15 and worlds[0].sample == {"k": 0.0}  # 16 // 3 = 5 per condition; sample 0 nominal
    again = RN.plan_worlds(_spec(worlds_per_condition=16), conds, ctx)
    assert [w.sample for w in again] == [w.sample for w in worlds]  # seeded
    expanded = RN.plan_worlds(_spec(), conds, ctx, lambda c, q, n: [{"i": 1.0}, {"i": 2.0}])
    assert len(expanded) == 6 and expanded[1].sample == {"i": 2.0}


class _FallTwin:
    """A tiny passive twin: one ball per world, reports |z_end - z_law| as `err`."""

    law_id = "T.run"

    def conditions(self, quick: bool) -> tuple[Condition, ...]:
        return (Condition("a", {}), Condition("b", {}))

    def build_world(self, b, cond, sample) -> None:
        body = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()))
        b.add_shape_sphere(body, radius=0.05)

    def make_solver(self, model, solver):
        self.model = model
        self.first = WorldRanges.bodies(model).start
        return newton.solvers.SolverSemiImplicit(model, angular_damping=0.0)

    def make_pipeline(self, model, solver):
        return None

    def drive(self, step, t, state, control) -> None:
        pass

    def record(self, step, state, contacts, obs, log) -> None:
        record_body_position(log, "", state.body_q, self.first, step)

    def scalars(self, log) -> dict[str, np.ndarray]:
        t = log.times()[-1]
        return {"err": np.abs(log.series("z")[:, -1] - (1.0 - 0.5 * 9.81 * t * t))}


def test_run_law_with_a_plain_twin() -> None:
    res = RN.run_law(_spec(), _FallTwin(), None, device="cpu", solver="semi_implicit")
    assert res.verdicts[0].verdict == "pass" and res.monitors["worlds"] == 4 and res.monitors["finite"]
    assert res.config["solver"] == "semi_implicit" and res.provenance["suite"] == "analytic"



def test_contact_hooks_run_after_every_collide() -> None:
    class Hooked(_FallTwin):
        made, seen = 0, 0

        def make_pipeline(self, model, solver):
            return newton.CollisionPipeline(model)

        def make_contacts(self, pipeline):
            Hooked.made += 1
            return newton.Contacts(pipeline.rigid_contact_max, 0, device=self.model.device,
                                   per_contact_shape_properties=True)

        def after_collide(self, state, contacts):
            assert contacts.rigid_contact_stiffness is not None  # the twin's own Contacts reached the loop
            Hooked.seen += 1

    spec = _spec()
    res = RN.run_law(spec, Hooked(), None, device="cpu", solver="semi_implicit")
    assert res.verdicts[0].verdict == "pass" and Hooked.made == 1
    assert Hooked.seen == round(spec.duration_s / spec.dt) * spec.substeps

def test_run_law_records_errors_instead_of_raising() -> None:
    class Broken(_FallTwin):
        def build_world(self, b, cond, sample) -> None:
            raise RuntimeError("bad scene")

    res = RN.run_law(_spec(), Broken(), None, device="cpu", solver="semi_implicit")
    assert res.verdict == "error" and "bad scene" in res.monitors["error"]


def test_rig_law_needs_data() -> None:
    from alloy.core.errors import LawError

    with pytest.raises(LawError):
        RN.run_law(_spec(suite="rig"), _FallTwin(), None, device="cpu", solver="xpbd")


def test_judge_rig_scalar_rate_and_calib() -> None:
    data = fake_rig_dataset({"f": 10.0, "jam": 1.0}, noise=0.0, n=10)
    spec = _spec(suite="rig", metrics={"f": Tolerance(rel=0.05, source="t"), "jam": Tolerance(abs=0.1, source="t")},
                 stat={"f": "mean", "jam": "rate"})
    values = {"f": np.array([10.2, 10.2]), "jam": np.array([1.0, 1.0])}
    verdicts, calib = judge_rig(spec, object(), values, data, ["c0", "c0"], None)
    by = {v.name: v for v in verdicts}
    assert by["f@c0"].verdict == "pass" and by["f@c0"].error == pytest.approx(0.2)
    assert by["jam@c0"].verdict == "pass" and calib["f@c0"] == pytest.approx(0.2)
    assert math.isinf(by["f@c0"].tur or math.inf)  # zero-noise rig: U_real = 0 -> TUR unbounded (None)



def test_judge_rig_calib_only_condition_reports_fit_error_without_a_verdict() -> None:
    # R10a: calib poses and valid poses are separate conditions; a calib pose is fitted, never judged
    data = fake_rig_dataset({"f": 10.0}, noise=0.0, n=10)
    calib_pose = Condition("c1", {})
    trials = tuple(dataclasses.replace(t, condition_id="c1", split="calib") for t in data.split("calib"))
    data = dataclasses.replace(data, conditions=data.conditions + (calib_pose, Condition("c2", {})),
                               trials=data.trials + trials)
    spec = _spec(suite="rig", metrics={"f": Tolerance(rel=0.05, source="t")})
    verdicts, calib = judge_rig(spec, object(), {"f": np.array([10.2, 10.3, 9.0])}, data, ["c0", "c1", "c2"], None)
    by = {v.name: v.verdict for v in verdicts}
    assert by == {"f@c0": "pass", "f@c2": "not_run"}  # c1 (calib only): no verdict; c2 (no trials): not_run
    assert calib["f@c1"] == pytest.approx(0.3)


def test_judge_rig_distribution_unit() -> None:
    # SCR-WP2-8: a twin declaring units[metric] = "1" is judged by W1 alone; undeclared keeps metres + PRF-02
    data = fake_rig_dataset({"nodes": 7.0}, noise=0.0, n=10)
    spec = _spec(suite="rig", metrics={"nodes": Tolerance(abs=1.0, source="t")}, stat={"nodes": "distribution"})
    values = {"nodes": np.array([8.0, 8.0])}

    class Counted:
        units = {"nodes": "1"}

    counted = judge_rig(spec, Counted(), values, data, ["c0", "c0"], None)[0][0]
    assert counted.verdict == "pass" and counted.unit == "1" and counted.error == pytest.approx(1.0)
    metres = judge_rig(spec, object(), values, data, ["c0", "c0"], None)[0][0]
    assert metres.verdict == "fail" and metres.unit == "m"  # PRF-02: median 1 m > 5 mm

def test_incomplete_twin_is_not_run() -> None:
    class Old:  # pre-SCR-WP6-1 twin: no conditions()
        law_id = "T.run"

        def build_world(self, b, cond, sample): ...
        def make_solver(self, model): ...
        def drive(self, step, t, state, control): ...
        def record(self, step, state, contacts, obs, log): ...
        def scalars(self, log): ...

    assert RN.missing_hooks(Old(), None) == ["conditions"]
    res = RN.run_law(_spec(), Old(), None, device="cpu", solver="xpbd")
    assert res.verdict == "not_run" and "conditions" in res.monitors["not_run_reason"]


def test_reference_hook_turns_values_into_errors() -> None:
    class Ref(_FallTwin):
        def scalars(self, log):
            return {"err": log.series("z")[:, -1].astype(np.float64)}  # raw z, not an error

        @staticmethod
        def reference(cond):
            return {"err": 1.0 - 0.5 * 9.81 * 0.1**2}

    res = RN.run_law(_spec(), Ref(), None, device="cpu", solver="semi_implicit")
    assert res.verdicts[0].verdict == "pass" and res.verdicts[0].error < 0.01
