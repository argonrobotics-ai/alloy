# SPDX-License-Identifier: Apache-2.0
"""``alloy.bully.audit`` (coordinator 2026-11-06, OSS ``alloy audit``): random-legal episodes on any scene card,
findings with class, cause, first tick, body and a replay handle; CPU, small scenes."""

from __future__ import annotations

import json

import mujoco
import numpy as np
import pytest

import alloy.bully
from alloy.bully.actions import PASSIVE, PassiveActions, bind_actions
from alloy.bully.replay import EpisodeRunner
from alloy.bully.scene_audit import ReplayHandle, audit, replay_finding
from alloy.bully.scene_target import AuditOptions, ctrl_box
from alloy.bully.types import ActionSpace, BullyLimits, BullyTarget
from alloy.core.errors import BullyError

SCENE = "demo_peg_d12"  # MuJoCo route with a precision pair; one free body (the peg) -> push channel
TOUCHY = BullyLimits(pen_s1=1e-7)  # every resting contact trips pen_s1_ratio: S1 findings to inspect


def test_package_exports_the_call_and_the_brief_module_does_not_shadow_it():
    import alloy.bully.brief  # noqa: F401  (the B2 brief, formerly alloy.bully.audit)

    assert alloy.bully.audit is audit
    assert callable(alloy.bully.audit) and alloy.bully.AuditOptions is AuditOptions
    with pytest.raises(AttributeError):
        alloy.bully.no_such_name  # noqa: B018


def test_clean_push_audit_reports_counts_channel_and_json():
    r = audit(SCENE, episodes=2, seed=0, device="cpu", horizon_s=0.1, batch=2)
    assert r.episodes == 2 and r.counts["clean"] + r.counts["S1"] + r.counts["S2"] + r.counts["S3"] == 2
    assert r.channel.startswith("push: force +-0.2 x weight per axis on 1 free bodies")
    assert r.ok == (r.counts["S1"] == 0) and r.s1_upper95 == pytest.approx(1.5 if r.ok else r.s1_upper95)
    data = json.loads(r.to_json())
    assert data["scene"] == SCENE and data["device"] == "cpu" and data["ok"] == r.ok


def test_findings_name_cause_tick_body_and_replay_alone():
    r = audit(SCENE, episodes=2, seed=3, device="cpu", horizon_s=0.1, batch=2, limits=TOUCHY, attribute=1)
    assert r.counts["S1"] == 2 and not r.ok and r.by_cause == {"S1:pen_s1_ratio": 2}
    f = r.findings[0]
    assert (f.severity, f.cause, f.handle.seed, f.handle.batch, f.handle.world) == ("S1", "pen_s1_ratio", 3, 2, 0)
    assert f.first_tick >= 0 and f.time_s == pytest.approx(0.02 * (f.first_tick + 1)) and f.value > f.threshold
    assert f.replays_alone is True and f.body.startswith("peg_d12") and f.body_detail.startswith("deepest contact")
    assert "against block_bore_d12_2" in f.body_detail or "against table_top_plywood" in f.body_detail
    assert r.findings[1].body is None  # attribute=1: one attributed finding per (class, cause)
    again = replay_finding(f.handle.to_dict(), "cpu", limits=TOUCHY)
    assert again is not None and (again.cause, again.first_tick, again.body) == (f.cause, f.first_tick, f.body)
    assert again.handle == f.handle


def test_episodes_after_a_reset_match_a_fresh_build():
    """Batch 1 of a run (after a reset) equals batch 0 of a fresh run at the same seed: no carried-over health."""
    a = audit(SCENE, episodes=4, seed=5, device="cpu", horizon_s=0.06, batch=2, limits=TOUCHY, attribute=0)
    b = audit(SCENE, episodes=2, seed=6, device="cpu", horizon_s=0.06, batch=2, limits=TOUCHY, attribute=0)
    key = [(f.handle.world, f.cause, f.first_tick, round(f.value, 6)) for f in a.findings if f.handle.seed == 6]
    assert key == [(f.handle.world, f.cause, f.first_tick, round(f.value, 6)) for f in b.findings]


def test_bad_arguments_raise():
    with pytest.raises(BullyError):
        AuditOptions(channel="wiggle")
    with pytest.raises(BullyError):
        audit(SCENE, episodes=0, device="cpu")
    with pytest.raises(BullyError):  # demo_peg_d12 has no actuators
        audit(SCENE, episodes=1, device="cpu", horizon_s=0.02, options=AuditOptions(channel="ctrl"))


def test_mjw_route_option_builds_the_ordered_route():
    import dataclasses

    from alloy.bully.scene_target import build_scene_sim
    from alloy.check.source import load_source

    src = load_source(SCENE)
    card = dataclasses.replace(src.card, control_dt=0.02)
    assert build_scene_sim(src.lib, card, 1, "cpu", AuditOptions(mjw_route=True)).ordered is True


def test_handle_round_trips_through_json():
    h = ReplayHandle("demo_peg_d12", 7, 4, 2, 0.5, {"channel": "push"})
    assert ReplayHandle(**json.loads(json.dumps(h.to_dict()))) == h


_MJCF = """<mujoco><compiler angle="radian"/><worldbody><body><joint name="h" type="hinge" range="-1 1" limited="true"/>
  <geom size="0.1" mass="1"/><body pos="0 0 0.3"><joint name="s" type="slide" range="0 0.04" limited="true"/>
  <geom size="0.05" mass="0.1"/><body pos="0 0 0.3"><joint name="free_spin" type="hinge" limited="false"/>
  <geom size="0.05" mass="0.1"/></body></body></body></worldbody>
  <actuator><position joint="h" kp="10" ctrlrange="-2 2" ctrllimited="true"/>
  <position joint="s" kp="10"/><motor joint="h" ctrlrange="-5 5" ctrllimited="true"/>
  <velocity joint="free_spin" kv="1"/></actuator></mujoco>"""


def test_ctrl_box_servo_slide_motor_and_unbounded():
    low, high, servo, slide, joint = ctrl_box(mujoco.MjModel.from_xml_string(_MJCF))
    assert joint.tolist() == [0, 1, 0, 2]
    np.testing.assert_allclose(low[:3], [-1.0, 0.0, -5.0])  # servo: ctrl range cut to the joint range
    np.testing.assert_allclose(high[:3], [1.0, 0.04, 5.0])
    assert servo.tolist() == [True, True, False, False] and slide.tolist() == [False, True, False, False]
    assert not np.isfinite(low[3]) and not np.isfinite(high[3])  # no box: not commanded


class _Sim:
    def __init__(self, worlds: int) -> None:
        from alloy._vendor import newton

        b = newton.ModelBuilder()
        b.add_body(mass=1.0)
        s = newton.ModelBuilder()
        s.replicate(b, worlds)
        self.model = s.finalize(device="cpu")
        self.state, self.control = self.model.state(), self.model.control()
        self.periods = 0

    def step_period(self) -> None:
        self.periods += 1

    def reset(self, world_mask=None) -> None:
        self.periods = 0

    def report(self):
        return {}


def test_passive_kind_runs_a_zero_width_trace():
    empty = np.zeros(0)
    target = BullyTarget(name="passive", scene=None, make_sim=lambda w, d: _Sim(w),
                         actions=ActionSpace(low=empty, high=empty, rate_hz=50.0, max_delta_per_tick=empty,
                                             initial=empty), action_kind=PASSIVE, robot_bodies=())
    sim = _Sim(2)
    bound = bind_actions(sim, sim.model, PASSIVE, (), np.zeros((2, 3, 0)))
    assert isinstance(bound, PassiveActions) and bound.ticks == 3
    with pytest.raises(BullyError):
        bound.apply(3)
    res = EpisodeRunner(target, 2, "cpu", BullyLimits()).run(np.zeros((3, 0)))
    assert res.severity.tolist() == [0, 0]


@pytest.mark.parametrize("book", [True, False])
def test_push_work_is_booked_with_health_energy(book):
    """A push is external work: WP5's energy check (on for this passive scene) must not read it as energy created.
    The GPU smoke's ENERGY_JUMP (INVALID) worlds came from it before ``SceneSim`` booked the work."""
    import dataclasses

    from alloy.bully.scene_target import build_scene_sim
    from alloy.check.source import load_source
    from alloy.precision.types import HealthFlags

    src = load_source(SCENE)
    sim = build_scene_sim(src.lib, dataclasses.replace(src.card, control_dt=0.02), 1, "cpu", AuditOptions())
    if not book:
        sim._ledger = None
    weight = 9.81 * float(sim.model.body_mass.numpy()[0])
    bound = bind_actions(sim, sim.model, "body_f", (0, 1, 2), np.tile([2 * weight, 0.0, 2 * weight], (1, 15, 1)))
    for tick in range(15):  # 0.3 s at twice the weight up and sideways (beyond the legal box, on purpose)
        bound.apply(tick)
        sim.step_period()
    jumped = bool(int(sim.report()["health"].flags[0]) & int(HealthFlags.ENERGY_JUMP))
    assert jumped is (not book)
