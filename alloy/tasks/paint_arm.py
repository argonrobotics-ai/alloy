# SPDX-License-Identifier: Apache-2.0
"""What the painting tasks' fixed arms share: the joint plan's bookkeeping, IK tracking of tool-point poses, smooth
joint moves, and servo targets led by the position servos' lag.

The arms are MuJoCo position servos ``kp (ctrl - q) - kv qd`` (the FR3's: kv/kp = 0.1 s). Written the plan's own
angles, a joint moving at a steady rate trails them by ``kv/kp`` of its motion (the roller's cover 26.7 mm at
0.25 m/s); :func:`led_targets` writes the angle at each period's middle plus that lag.

A plan books its peak joint speed, acceleration and jerk (differences of its angles a control period apart) against
the FR3's limits (Franka's specifications: 2.62-5.26 rad/s, 10 rad/s² and 5,000 rad/s³ every joint). ``Plan.ok``
asks for each joint under 0.9 of its speed limit; ``Plan.within_fr3`` for the acceleration and jerk.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

GAP = 0.005             # m: contacts start this close (Newton's default 0.1 floods MuJoCo's buffers)
FR3_ACCEL = 10.0        # rad/s²: the FR3's joint acceleration limit, every joint (Franka's specifications)
FR3_JERK = 5000.0        # rad/s³: its joint jerk limit
FR3_SPEED = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])  # rad/s: its joint speed limits, joints 1-7
SPEED_SHARE = 0.85      # the share of each joint's speed limit a plan's joint moves run at, at their peak


@dataclass
class Plan:
    """The robot's steps, and how the plan went."""

    steps: list = field(default_factory=list)
    misses: int = 0                 # steps the IK could not reach (tolerance)
    near_limits: int = 0            # steps with a joint within the IK's limit margin
    peak_speed: float = 0.0         # rad/s
    speed_share: float = 0.0        # the peak of each joint's speed over its FR3 limit, the largest
    clearance: float = math.inf     # m: the tool's least clearance where the plan checks it
    notes: list = field(default_factory=list)
    peak_accel: float = 0.0         # rad/s²
    peak_jerk: float = 0.0          # rad/s³
    over_accel: int = 0             # steps with a joint past FR3_ACCEL
    over_jerk: int = 0              # steps with a joint past FR3_JERK

    @property
    def ok(self) -> bool:
        """Reached, inside the joints' limits with a margin, at most 2 rad/s, clear of the statics. The acceleration
        and jerk are booked, not asked (the module's note)."""
        return self.misses == 0 and self.near_limits == 0 and self.speed_share <= 0.9 and self.clearance > 0.02

    @property
    def within_fr3(self) -> bool:
        """Every step inside the FR3's acceleration and jerk limits."""
        return self.over_accel == 0 and self.over_jerk == 0

    def report(self) -> dict:
        return {"misses": self.misses, "near_limits": self.near_limits, "peak_joint_speed": round(self.peak_speed, 3),
                "peak_speed_share": round(self.speed_share, 3),
                "peak_joint_accel": round(self.peak_accel, 1), "peak_joint_jerk": round(self.peak_jerk),
                "steps_over_fr3_accel": self.over_accel, "steps_over_fr3_jerk": self.over_jerk,
                "clearance_mm": round(float(self.clearance) * 1e3, 1), "notes": self.notes[:3]}

    def finish(self, dt: float) -> Plan:
        """Books the peak joint speed (and each joint's share of its FR3 limit), acceleration and jerk over the steps
        (``.q``, the FR3's 7 joint angles, ``dt`` s apart: first, second and third differences), and how many steps
        have a joint past the FR3's acceleration and jerk limits."""
        qs = np.array([st.q for st in self.steps], np.float64).reshape(len(self.steps), -1)
        if len(qs) and qs.shape[1] != FR3_SPEED.size:
            raise ValueError(f"a plan's steps are the FR3's {FR3_SPEED.size} joint angles, got {qs.shape[1]}")
        v = np.diff(qs, axis=0) / dt
        a = np.diff(v, axis=0) / dt
        j = np.diff(a, axis=0) / dt
        self.peak_speed = float(np.abs(v).max()) if v.size else 0.0
        self.speed_share = float((np.abs(v) / FR3_SPEED).max()) if v.size else 0.0
        self.peak_accel = float(np.abs(a).max()) if a.size else 0.0
        self.peak_jerk = float(np.abs(j).max()) if j.size else 0.0
        self.over_accel = int((np.abs(a) > FR3_ACCEL).any(axis=1).sum()) if a.size else 0
        self.over_jerk = int((np.abs(j) > FR3_JERK).any(axis=1).sum()) if j.size else 0
        return self


def ik_starts(rigged, target: np.ndarray, n: int = 64, keep: int = 8) -> list[np.ndarray]:  # noqa: ANN001
    """Up to ``keep`` distinct IK solutions ``[1, 1, J]`` of the tool-point pose ``target`` ``[1, 7]`` from ``n``
    random starts, widest joint-limit margin first."""
    ik = rigged.make_ik(n, device="cpu")
    lo, hi = ik.bind.lower[0], ik.bind.upper[0]
    q, err = ik.solve(np.broadcast_to(target, (n, 1, 7)), np.random.default_rng(0).uniform(lo, hi, (n, 1, lo.size)),
                      check=False)  # fmt: skip
    ok = (err[:, 0, 0] < ik.tol.pos_m) & (err[:, 0, 1] < ik.tol.ang_rad)
    margin = np.where(ok, np.minimum(q[:, 0] - lo, hi - q[:, 0]).min(axis=1), -np.inf)
    out: list[np.ndarray] = []
    for i in np.argsort(-margin):
        if not ok[i] or len(out) == keep:
            break
        if all(np.abs(q[i, 0] - o[0, 0]).max() > 0.1 for o in out):
            out.append(q[i][None])
    return out


def smooth(a: np.ndarray, b: np.ndarray, dt: float, rate: float = 0.8, least: float = 0.5,
           accel: float = 0.8 * FR3_ACCEL, peak: np.ndarray | None = None) -> list[np.ndarray]:
    """Vectors from ``a`` to ``b`` ``dt`` s apart on a minimum-jerk profile: the largest component at ``rate`` per s on
    average, each at most ``peak`` per s at the profile's peak (1.875·Δ/T) when given, at most ``accel`` per s² (its
    peak, 5.77·Δ/T²), and at least ``least`` s; the last is ``b``."""
    span = float(np.abs(b - a).max())
    n = max(int(round(least / dt)), int(math.ceil(span / (rate * dt))),
            int(math.ceil(math.sqrt(5.7735 * span / accel) / dt)))
    if peak is not None:
        n = max(n, int(math.ceil(float((1.875 * np.abs(b - a) / peak).max()) / dt)))
    s = (np.arange(1, n + 1) / n)[:, None]
    return list(a + (b - a) * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5))


def _ramp_share(t: np.ndarray) -> np.ndarray:
    """The distance, as a share of V·T, a minimum-jerk speed ramp from rest to V covers by τ = t/T: the integral of
    10τ³ - 15τ⁴ + 6τ⁵ (no jump in speed or acceleration at either end; 0.5 at τ = 1)."""
    return 2.5 * t ** 4 - 3.0 * t ** 5 + t ** 6


def ramped_path(path: np.ndarray, speed: float, dt: float, accel: float) -> list[np.ndarray]:
    """Places ``dt`` apart along the polyline ``path`` [n, 3] from its first point (its last not included): from rest
    up to ``speed`` and back to rest, each ramp minimum-jerk over ``T = 1.875·speed/accel`` (peak acceleration
    ``accel``), the cruise between at a speed trimmed so the path takes whole periods. A path shorter than
    ``1.875·speed²/accel`` is all ramp, ``T = length/speed`` at about ``speed``: its peak acceleration is then about
    ``1.875·speed²/length``, above ``accel``. The plans check the joints' limits (``Plan.within_fr3``), not this."""
    path = np.asarray(path, np.float64)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    length = float(arc[-1])
    ramp = min(1.875 * speed / accel, length / speed)       # the two ramps take at most the whole path
    n = max(1, int(math.ceil((length / speed + ramp) / dt - 1e-9)))
    v = length / (n * dt - ramp)                           # the cruise: length / v + ramp == n·dt
    t, end = np.arange(n) * dt, n * dt
    d = np.where(t < ramp, v * ramp * _ramp_share(t / ramp), 0.5 * v * ramp + v * (t - ramp))
    down = t > end - ramp
    tau = (t[down] - (end - ramp)) / ramp
    d[down] = length - 0.5 * v * ramp + v * ramp * (tau - _ramp_share(tau))
    return [np.array([np.interp(dk, arc, path[:, i]) for i in range(path.shape[1])]) for dk in d]


class Tracker:
    """IK tracking of tool-point poses, each solve seeded by the last; books misses and limits in ``plan``."""

    def __init__(self, rigged, plan: Plan) -> None:  # noqa: ANN001
        self.rigged = rigged
        self.ik = rigged.make_ik(1, device="cpu")
        self.lo, self.hi, self.margin = self.ik.bind.lower[0], self.ik.bind.upper[0], self.ik.tol.limit_margin
        self.plan = plan


    def start(self, targets: list[np.ndarray]) -> np.ndarray:
        """The seed for tracking ``targets``: of the distinct IK solutions at the first, the one whose track (every
        10th target) keeps the widest joint-limit margin (a 7-joint arm has a spare degree of freedom, and a seed good
        at the start can drift a joint into its limit further on)."""
        cands = ik_starts(self.rigged, targets[0][None])
        if not cands:
            raise RuntimeError(f"no IK solution at {np.round(targets[0][:3], 3)}")
        best, best_margin = cands[0], -np.inf
        for seed in cands:
            q, worst = seed, np.inf
            for t in targets[::10]:
                q, err = self.ik.solve(t[None, None], q, check=False)
                miss = err[0, 0, 0] > self.ik.tol.pos_m or err[0, 0, 1] > self.ik.tol.ang_rad
                worst = min(worst, -1.0 if miss else float(np.minimum(q[0, 0] - self.lo, self.hi - q[0, 0]).min()))
            if worst > best_margin:
                best, best_margin = seed, worst
        return best

    def track(self, targets: list[np.ndarray], seed: np.ndarray, until_miss: bool = False) -> list[np.ndarray]:
        """Joint vectors for ``targets``; with ``until_miss`` (a leg in the air, whose path is free) it stops before
        the first target the IK misses or that brings a joint near its limit, booking neither."""
        out = []
        for t in targets:
            sol, err = self.ik.solve(t[None, None], seed, check=False)
            missed = bool(err[0, 0, 0] > self.ik.tol.pos_m or err[0, 0, 1] > self.ik.tol.ang_rad)
            near = bool(((sol[0, 0] < self.lo + self.margin) | (sol[0, 0] > self.hi - self.margin)).any())
            if until_miss and (missed or near):
                break
            if missed and len(self.plan.notes) < 5:
                self.plan.notes.append(f"miss at {np.round(t[:3], 3)} err {np.round(err[0, 0], 4)}")
            self.plan.misses += int(missed)
            self.plan.near_limits += int(near)
            out.append(sol[0, 0].copy())
            seed = sol
        return out


def servo_lead(biasprm: np.ndarray) -> np.ndarray:
    """s, per actuator row of MuJoCo's ``biasprm`` ``(0, -kp, -kv)``: how far a position servo ``kp (ctrl - q) -
    kv qd`` trails a joint moving at a steady rate, ``kv / kp``."""
    biasprm = np.asarray(biasprm, np.float64)
    return biasprm[..., 2] / biasprm[..., 1]


def led_targets(qs: np.ndarray, lead: np.ndarray, dt: float) -> np.ndarray:
    """``[n, J]`` servo targets for a plan's joint angles ``qs`` ``[n, J]``, ``dt`` s apart: the angle at each
    period's middle plus the servo's lag at the plan's rate, ``q_k + qd_k (dt/2 + lead)``, ``qd_k = (q_k+1 - q_k)/dt``
    (0 after the last)."""
    qd = np.zeros_like(qs)
    qd[:-1] = np.diff(qs, axis=0) / dt
    return qs + qd * (0.5 * dt + np.asarray(lead))


def down_tcp(point, offset: float) -> np.ndarray:  # noqa: ANN001
    """The Franka tool point's pose (x, y, z, qx, qy, qz, qw) for a tool held straight down, its working point
    ``offset`` below the tool point at ``point``: the hand's approach down, its fingers closing along world x."""
    from scipy.spatial.transform import Rotation  # noqa: PLC0415

    from alloy.arms.specs import FRANKA_TCP  # noqa: PLC0415

    hand = Rotation.from_matrix(np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]))
    pos = np.asarray(point, np.float64) + (0.0, 0.0, offset)
    return np.concatenate([pos, (hand * Rotation.from_quat(FRANKA_TCP[3:])).as_quat()])


def mounted_model(xml: str, base: tuple, boxes: list):  # noqa: ANN201
    """MuJoCo C: the arm's MJCF ``xml`` on ``base`` (x, y, z, qx, qy, qz, qw) and static ``boxes`` (label, centre,
    half extents). Returns the model, the boxes' geoms and the arm's colliding geoms."""
    import xml.etree.ElementTree as ET  # noqa: PLC0415

    import mujoco  # noqa: PLC0415

    root = ET.fromstring(xml)
    world = root.find("worldbody")
    qx, qy, qz, qw = base[3:]
    mount = ET.Element("body", name="mount", pos=" ".join(map(str, base[:3])), quat=f"{qw} {qx} {qy} {qz}")
    for child in list(world):
        world.remove(child)
        mount.append(child)
    world.append(mount)
    for label, pos, half in boxes:
        ET.SubElement(world, "geom", name=label, type="box", pos=" ".join(map(str, pos)), size=" ".join(map(str, half)))
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    statics = [m.geom(label).id for label, _, _ in boxes]
    arm = [g for g in range(m.ngeom) if m.geom_bodyid[g] != 0 and (m.geom_contype[g] or m.geom_conaffinity[g])]
    return m, statics, arm


def static_clearance(xml: str, base: tuple, boxes: list, qs: list[np.ndarray]) -> float:
    """m: the arm's least distance to static ``boxes`` over the joint vectors ``qs`` (:func:`mounted_model`; only the
    arm's colliding geoms)."""
    import mujoco  # noqa: PLC0415

    m, statics, arm = mounted_model(xml, base, boxes)
    d, fromto, best = mujoco.MjData(m), np.zeros(6), math.inf
    for q in qs:
        d.qpos[: q.size] = q
        mujoco.mj_kinematics(m, d)
        best = min(best, *(mujoco.mj_geomDistance(m, d, a, s, 0.5, fromto) for a in arm for s in statics))
    return best


def dry_run(driver, scene, n: int, planned: Callable, after: Callable | None = None) -> np.ndarray:  # noqa: ANN001
    """``driver``'s first ``n`` steps in its arm ``scene`` without the film (no plug-ins): after each step, the tool
    body's distance [m] from ``planned(k + 1)``, its planned place at the next step's start (NaN where ``None``).
    ``after(sim, k)`` runs after step ``k``."""
    from alloy.plugins.harness.scene import build_simulation  # noqa: PLC0415

    sim = build_simulation(scene, [], "cpu")
    driver.bind(sim)
    out = np.full(n, np.nan)
    for k in range(n):
        driver.apply(sim, driver.steps[k])
        sim.step_period()
        p = planned(k + 1)
        if p is not None:
            out[k] = float(np.linalg.norm(sim.state.body_q.numpy()[driver.body][:3].astype(np.float64) - p))
        if after is not None:
            after(sim, k)
    return out


@dataclass(frozen=True)
class ArmScene:
    """A fixed arm's scene: its MJCF (the tool in it) at ``base`` (x, y, z, qx, qy, qz, qw), its rig, the static
    shapes (``statics(builder, cfg)`` adds them), the arm's start, the control period and MuJoCo's step."""

    xml: str
    base: tuple
    rigged: object
    statics: Callable
    q0: np.ndarray
    control_dt: float
    dt: float


def arm_scene(spec: ArmScene, init):  # noqa: ANN001, ANN201 - a harness TestScene
    """The harness scene of ``spec``: MuJoCo, gravity compensated (the tool too), the arm at ``q0``."""
    from alloy.plugins.harness.scene import TestScene  # noqa: PLC0415

    def make(plugins, device: str, worlds: int, graph: bool):  # noqa: ANN001, ANN202, ARG001
        import warp as wp  # noqa: PLC0415

        from alloy._vendor import newton  # noqa: PLC0415
        from alloy._vendor.newton.solvers import SolverMuJoCo  # noqa: PLC0415

        from alloy.plugins.host import simulation  # noqa: PLC0415
        from alloy.scene.builtin_plugins import register_builtin_solvers  # noqa: PLC0415
        from alloy.testing.integration import built_from_model, entry_spec  # noqa: PLC0415

        register_builtin_solvers()
        t = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(t)
        t.rigid_gap = GAP
        t.add_mjcf(spec.xml, xform=wp.transform(spec.base[:3], wp.quat(*spec.base[3:])), ctrl_direct=True,
                   parse_visuals=False)  # fmt: skip
        spec.statics(t, newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.5))
        root = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(root)
        root.replicate(t, worlds)
        model = root.finalize(device=device)
        bind = spec.rigged.bind(model)
        spec.rigged.gravcomp(model, bind)
        spec.rigged.write_ready(model, bind, np.broadcast_to(spec.q0, (worlds, 1, spec.q0.size)).copy(), 0.0)
        entry = entry_spec(model, "mujoco", spec.dt, nconmax=64, njmax=256)
        return simulation(built_from_model(model, [entry], control_dt=spec.control_dt), list(plugins))

    return TestScene(build=lambda b: None, init=init, worlds=1, control_dt=spec.control_dt, dt=spec.dt, make=make)


__all__ = ["FR3_ACCEL", "FR3_JERK", "FR3_SPEED", "GAP", "SPEED_SHARE", "ArmScene", "Plan", "Tracker", "arm_scene",
           "down_tcp", "dry_run", "ik_starts", "led_targets", "mounted_model", "ramped_path", "servo_lead", "smooth",
           "static_clearance"]
