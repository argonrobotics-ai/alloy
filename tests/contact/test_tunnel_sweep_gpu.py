# SPDX-License-Identifier: Apache-2.0
"""CT-2 (reduced, SPEC §5.2 / §7 WP3): tunnelling counted by GapMonitor's swept test on 0.25-1 mm walls.

256 worlds, one steel coin (Ø10 x 1 mm) per world above a static plate 0.25 / 0.5 / 0.75 / 1.0 mm thick (world w
uses thickness w % 4; 120 mm wide so impacts stay on the face, not its edge). Eight launches per world, 15 ms apart:
speed U(0.2, 2) m/s within 45° of the plate normal, tilt ≤ 30°, offset ≤ 8 mm (2,048 impacts; the full CT-2 is
4,096 x 32). Then CON-67 (reduced): every coin is seated flat on its plate at rest (``_seat``) and pressed
quasi-statically with 10 N (``_press``). Routed run at the §3.5 L2 step for 2 m/s: 0 tunnels on both native paths.

Seating matters: 15 ms after the last launch some coins are still airborne (bounced up to 14 mm, sliding at ~1 m/s);
pressing those accelerated a 0.6 g coin at ~170 g onto the plate's edge at > 4 m/s with 400 rad/s spin, outside the
2 m/s card, and the monitor counted that as tunnels (seed 4, world 53, both native paths; docs/notes/wp3-results.md).

Detector check: the same impacts at 1.25 ms with ``ccd="off"`` (no L0/L1) must report tunnels > 0, i.e. the swept
test counts full pass-throughs that leave the sample outside the wall at both ends (2.5 mm per step).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.contact.pipeline import AlloyContactLayer
from alloy.contact.precision import ladder_rungs, size_l2
from alloy.contact.shadow import set_flat_solimp
from alloy.contact.types import PrecisionPairSpec

from peg_harness import CONTROL_DT, _health, record

pytestmark = pytest.mark.gpu
WORLDS = 256
DEVICE = "cuda:0"
THICK = (2.5e-4, 5e-4, 7.5e-4, 1e-3)
COIN_R, COIN_HH, DENSITY = 5e-3, 5e-4, 7800.0
LAUNCHES, PERIOD = 8, 0.015
V_MAX, PRESS_N = 2.0, 10.0
PLATE_HALF = 0.06          # coins (offset ≤ 8 mm, ≤ 1.4 m/s sideways for 15 ms) never reach the plate edge


def _template(thick: float, ccd: str):
    cfg = newton.ModelBuilder.ShapeConfig(density=DENSITY, mu=0.3, gap=5e-4, margin=0.0)
    t = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(t)
    plate = t.add_shape_box(-1, xform=wp.transform((0.0, 0.0, -0.5 * thick), wp.quat_identity()), hx=PLATE_HALF,
                            hy=PLATE_HALF, hz=0.5 * thick, cfg=cfg)
    body = t.add_body(xform=wp.transform((0.0, 0.0, 5e-3), wp.quat_identity()))
    coin = t.add_shape_cylinder(body, radius=COIN_R, half_height=COIN_HH, cfg=cfg)
    t.add_shape_collision_filter_pair(plate, coin)
    for shape in (plate, coin):
        set_flat_solimp(t, shape, 0.99)
    spec = PrecisionPairSpec(plate, coin, mode="sdf", ccd=ccd, wall_thickness=THICK[0])
    return t, spec


def _scene(ccd: str):
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    spec = None
    for w in range(WORLDS):
        t, spec = _template(THICK[w % len(THICK)], ccd)
        b.add_world(t)
    model = b.finalize(device=DEVICE)
    coins = list(range(WORLDS))          # one body per world, in world order
    return model, spec, coins


class _Run:
    def __init__(self, ccd: str, native: str, dt: float | None):
        self.model, self.spec, self.coins = _scene(ccd)
        self.mass = float(self.model.body_mass.numpy()[0])
        self.dt = dt or size_l2(self.spec, m_eff=self.mass, design_force=50.0, contact_speed=V_MAX,
                                control_dt=CONTROL_DT).dt
        self.layer = AlloyContactLayer(self.model, [self.spec], native=native, precision_contacts_per_world=32)
        l1 = ccd != "off"
        self.solver = newton.solvers.SolverMuJoCo(
            self.model, use_mujoco_contacts=(native == "mujoco"), use_speculative_contacts=(native == "newton" and l1),
            integrator="euler", cone="elliptic", impratio=self.spec.mj_impratio, nconmax=64, njmax=256,
            iterations=50, ls_iterations=25)
        if native == "mujoco":
            self.layer.bind_mujoco(self.solver).install()
        self.monitor = self.layer.monitor(sample_spacing=5e-4)
        self.health = _health(self.model.world_count, self.model.device)
        self.s0, self.s1, self.ctrl = self.model.state(), self.model.state(), self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.s0)
        self.rungs = ladder_rungs(self.spec, "euler", native)
        self.steps = 0

    def launch(self, rng: np.random.Generator) -> None:
        q, qd = self.s0.body_q.numpy(), self.s0.body_qd.numpy()
        for b in self.coins:
            tilt_ax = np.r_[rng.normal(size=2), 0.0]
            tilt_ax /= np.linalg.norm(tilt_ax)
            ang = math.radians(rng.uniform(0.0, 30.0))
            polar, azim = math.radians(rng.uniform(0.0, 45.0)), rng.uniform(0.0, 2 * math.pi)
            v = rng.uniform(0.2, V_MAX) * np.array([math.sin(polar) * math.cos(azim),
                                                    math.sin(polar) * math.sin(azim), -math.cos(polar)])
            q[b, :3] = (*rng.uniform(-8e-3, 8e-3, 2), 3e-3 + COIN_R)
            q[b, 3:7] = np.r_[tilt_ax * math.sin(ang / 2), math.cos(ang / 2)]
            qd[b] = np.r_[v, rng.normal(scale=2.0, size=3)]
        self.s0.body_q.assign(q)
        self.s0.body_qd.assign(qd)
        newton.eval_ik(self.model, self.s0, self.s0.joint_q, self.s0.joint_qd)

    def press(self, force: float) -> None:
        f = np.zeros((self.model.body_count, 6), np.float32)
        f[self.coins, 2] = -force
        for st in (self.s0, self.s1):
            st.body_f.assign(f)

    def advance(self, seconds: float) -> None:
        for _ in range(max(1, int(round(seconds / self.dt)))):
            self.layer.collide(self.s0, self.layer.contacts(), dt=self.dt)
            self.solver.step(self.s0, self.s1, self.ctrl, self.layer.contacts(), self.dt)
            self.monitor.update(self.s0, self.s1, self.health)
            bf = self.s0.body_f
            self.s0, self.s1 = self.s1, self.s0
            self.s0.body_f.assign(bf)
            self.steps += 1
        wp.synchronize_device(self.model.device)


def _seat(run: _Run) -> None:
    """Every coin flat on its plate's face at rest, at its last x/y (clamped 1 mm inside the face), 1 µm above it."""
    q, qd = run.s0.body_q.numpy(), run.s0.body_qd.numpy()
    lim = PLATE_HALF - COIN_R - 1e-3
    for b in run.coins:
        q[b, :2] = np.clip(q[b, :2], -lim, lim)
        q[b, 2:7] = (COIN_HH + 1e-6, 0.0, 0.0, 0.0, 1.0)
        qd[b] = 0.0
    run.s0.body_q.assign(q)
    run.s0.body_qd.assign(qd)
    newton.eval_ik(run.model, run.s0, run.s0.joint_q, run.s0.joint_qd)


def _press(run: _Run) -> None:
    """CON-67 (reduced): quasi-static seal press of seated coins. Seat, settle 20 ms, ramp 0 -> 10 N over 50 ms in
    10 steps and hold 50 ms (a load on a coin still in the air hurls it onto the plate edge, beyond the card)."""
    _seat(run)
    run.advance(0.02)
    for k in range(1, 11):
        run.press(PRESS_N * k / 10)
        run.advance(0.005)
    run.advance(0.05)


def _impacts(run: _Run, seed: int, press: bool) -> dict:
    rng = np.random.default_rng(seed)
    for _ in range(LAUNCHES):
        run.launch(rng)
        run.advance(PERIOD)
    if press:
        _press(run)
    rep = run.monitor.report()
    ev = rep.tunnel_events[0]
    by_thick = {f"{THICK[k] * 1e3:.2f}mm": int(ev[k::len(THICK)].sum()) for k in range(len(THICK))}
    return dict(dt=run.dt, steps=run.steps, tunnels=int(ev.sum()), tunnels_by_wall=by_thick,
                worlds_with_tunnel=int((ev > 0).sum()), p99_m=float(rep.pair_p99_m[0]),
                max_m=float(rep.pair_max_m[0]), samples=int(rep.pair_samples[0]), rungs=list(run.rungs),
                impacts=WORLDS * LAUNCHES, overflow=run.layer.overflow_report())


@pytest.mark.parametrize("native", ["mujoco", "newton"])
def test_ct2_routed_no_tunnels(native):
    run = _Run("speculative", native, None)
    row = dict(metric="CT-2", variant="routed_L2", native=native, worlds=WORLDS, **_impacts(run, 2, press=True))
    record("ct2", row)
    assert row["tunnels"] == 0, row
    assert sum(row["overflow"].values()) == 0, row


def test_ct2_detector_counts_pass_throughs():
    run = _Run("off", "mujoco", 1.25e-3)
    row = dict(metric="CT-2", variant="detector_ccd_off_1.25ms", native="mujoco", worlds=WORLDS,
               **_impacts(run, 2, press=False))
    record("ct2", row)
    assert row["tunnels"] > 0, row
