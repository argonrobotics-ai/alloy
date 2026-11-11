# SPDX-License-Identifier: Apache-2.0
"""task_football_kick (Gym-Anything football "goal"): a kicking leg shoots a football from the penalty mark.

Scene (one world per kick):
- the pitch (``PressureBall``'s turf);
- an IFAB goal (posts and bar, 12 cm cylinders) with its goal line at x = 0;
- a net: a 120 mm cable net 2 m behind the line, pinned to the frame and the ground (``GoalNet``);
- a FIFA ball (0.8 bar) on the penalty mark, 11 m out;
- a mechanical kicking leg, like the legs that test footballs:
  - a 0.85 m link (8 kg, plus a 1 kg boot block) on a hip hinge 0.96 m up;
  - a hip spring (400 N·m/rad) pulls it to vertical;
  - it is released from the backswing angle ``power``, so the boot meets the ball at the bottom of the arc, at the
    leg's top speed.

The kicks differ per world:
- ``aim`` turns the swing plane;
- ``loft`` tilts the boot face up;
- ``curl`` yaws the face against the swing, so the boot rubs past the ball and spins it (sidespin curls the flight).

MuJoCo steps the leg and the ball. The ball's contacts (boot, posts, bar, pitch) are ``PressureBall``'s, the air is
``BallAero``'s and the net is ``GoalNet``'s.
- A cloth net in the coupled VBD route was tried first. Through the stock proxy coupler the ball gained energy.
- With the whole scene in VBD, its rigid bodies (AVBD) take their velocity from float32 pose differences, so a
  step's move rounds to ulp(|x|). A slowly accelerated body stalls when a·dt² < ½·ulp(x): 2 N on the 0.43 kg ball at
  z 2.1 m and a 0.16 ms step, or gravity alone at z 20 m (WP13 found the cause).

Success per world (IFAB Law 10): the whole ball crosses the goal line (its centre passes the line's back edge plus
a radius) inside the frame (|y| <= 3.66 m, z <= 2.44 m), within ``SECONDS``. The row also reports the launch speed,
spin and angle, where it crossed, and whether the net held it in the goal.

    python -m alloy.tasks.football_kick --worlds 8 --device cpu --no-net --out rows.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.harness.scene import body_xform
from alloy.plugins.sports import BallAero, GoalNet, PressureBall
from alloy.plugins.sports import model as M
from alloy.plugins.sports.scene import GOAL_HEIGHT, GOAL_WIDTH, PENALTY_MARK, POST_R, add_ball, add_goal

LEG_L, LEG_MASS, BOOT_MASS = 0.85, 8.0, 1.0         # hip to boot centre [m]; link and boot [kg]
LEG_K, LEG_KD = 400.0, 2.0                          # hip spring [N·m/rad] and damping [N·m·s/rad] (ESTIMATE)
BOOT_HALF = (0.10, 0.07, 0.08)                      # boot block half extents: along the swing, across, up [m]
BOOT_MU = 0.6                                       # ball-boot friction (ESTIMATE)
POST_MU = 0.3                                       # ball-post friction (ESTIMATE)
LINE_BACK = 2.0 * POST_R                            # the goal line's back edge (posts stand on the line)
NET_X, NET_WIDTH, NET_HEIGHT = LINE_BACK + 2.0, GOAL_WIDTH + 4.0 * POST_R, GOAL_HEIGHT + 2.0 * POST_R
BALL_X = -PENALTY_MARK
DT, CONTROL_DT, SECONDS = 5.0e-4, 0.02, 2.5
COLLIDERS = ("boot", "post_l", "post_r", "bar")


@dataclass(frozen=True)
class Kick:
    power: float            # backswing angle [rad] the leg is released from
    aim: float = 0.0        # swing-plane yaw [deg], + towards +y
    loft: float = 0.0       # boot face pitched up [deg]
    curl: float = 0.0       # boot face yawed against the swing [deg]


KICKS = (Kick(1.3, 0.0, 10.0), Kick(1.3, 12.0, 8.0), Kick(1.3, 24.0, 8.0), Kick(1.3, -14.0, 8.0, 12.0),
         Kick(1.5, 0.0, 25.0), Kick(0.8, 4.0, 0.0), Kick(1.1, -8.0, 12.0, -10.0), Kick(1.4, 6.0, 5.0))


def leg_inertia() -> tuple[float, float]:
    """(I about the hip [kg·m²], COM distance below the hip [m]): a uniform link plus the boot at its end."""
    i_hip = LEG_MASS * LEG_L ** 2 / 3.0 + BOOT_MASS * LEG_L ** 2
    return i_hip, (LEG_MASS * 0.5 * LEG_L + BOOT_MASS * LEG_L) / (LEG_MASS + BOOT_MASS)


def foot_speed(power: float) -> float:
    """Boot speed at the bottom of the arc from the hip spring and gravity (energy balance, no damping)."""
    i_hip, c = leg_inertia()
    energy = 0.5 * LEG_K * power ** 2 + (LEG_MASS + BOOT_MASS) * M.G * c * (1.0 - math.cos(power))
    return math.sqrt(2.0 * energy / i_hip) * LEG_L


def _leg(b: newton.ModelBuilder, k: Kick) -> None:
    yaw = math.radians(k.aim)
    d = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    hip = np.array([BALL_X, 0.0, M.R_BALL]) - d * (BOOT_HALF[0] + M.R_BALL) + np.array([0.0, 0.0, LEG_L])
    q_aim = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw)
    hip_xf = wp.transform(wp.vec3(*hip), q_aim)
    i_hip, c = leg_inertia()
    m = LEG_MASS + BOOT_MASS
    i_c = i_hip - m * c * c
    leg = b.add_link(xform=hip_xf, mass=m, com=wp.vec3(0.0, 0.0, -c),
                     inertia=wp.mat33(i_c, 0.0, 0.0, 0.0, i_c, 0.0, 0.0, 0.0, 0.01), label="leg")
    ghost = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False)
    b.add_shape_capsule(leg, xform=body_xform(0.0, 0.0, -0.45 * LEG_L), radius=0.05, half_height=0.3 * LEG_L,
                        cfg=ghost, label="shank")
    q_face = wp.mul(wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.radians(k.curl)),
                    wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -math.radians(k.loft)))
    b.add_shape_box(leg, xform=wp.transform(wp.vec3(0.0, 0.0, -LEG_L), q_face), hx=BOOT_HALF[0], hy=BOOT_HALF[1],
                    hz=BOOT_HALF[2], cfg=ghost, label="boot")
    j = b.add_joint_revolute(-1, leg, parent_xform=hip_xf, axis=(0.0, 1.0, 0.0), target_pos=0.0, target_ke=LEG_K,
                             target_kd=LEG_KD, label="hip")
    b.add_articulation([j], label="leg")


def world(k: Kick) -> newton.ModelBuilder:
    b = newton.ModelBuilder()
    add_goal(b, 0.0)
    _leg(b, k)
    add_ball(b, (BALL_X, 0.0, M.R_BALL))
    return b


def plugins(net: bool = True) -> list:
    plugs = [PressureBall(colliders=COLLIDERS, collider_mu=(BOOT_MU, POST_MU, POST_MU, POST_MU)), BallAero()]
    if net:
        plugs.append(GoalNet(origin=(NET_X, -0.5 * NET_WIDTH, 0.0), width=NET_WIDTH, height=NET_HEIGHT))
    return plugs


def build(kicks: tuple[Kick, ...], device: str = "cpu", net: bool = True, with_plugins: bool = True,
          graph: bool = False) -> Any:
    """The simulation: one world per kick, one MuJoCo entry, the sports plug-ins, legs at their backswing."""
    from alloy.plugins.host import simulation  # noqa: PLC0415
    from alloy.scene.builtin_plugins import register_builtin_solvers  # noqa: PLC0415
    from alloy.testing.integration import built_from_model, entry_spec  # noqa: PLC0415

    register_builtin_solvers()
    top = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(top)
    for k in kicks:
        top.add_world(world(k))
    model = top.finalize(device=device)
    built = built_from_model(model, [entry_spec(model, "mujoco", DT)], control_dt=CONTROL_DT)
    plugs = plugins(net) if with_plugins else []
    if graph:
        from alloy.integration.simulation import IntegratedSimulation  # noqa: PLC0415

        sim = simulation(built, plugs, cls=IntegratedSimulation, use_graph=True, cpu_graph=True)
    else:
        sim = simulation(built, plugs)
    _backswing(sim, kicks)
    return sim


def _backswing(sim: Any, kicks: tuple[Kick, ...]) -> None:
    model = sim.model
    q = sim.state.joint_q.numpy()
    starts, labels = model.joint_q_start.numpy(), [str(x).rsplit("/", 1)[-1] for x in model.joint_label]
    hips = [j for j, x in enumerate(labels) if x == "hip"]
    for j, k in zip(hips, kicks):
        q[starts[j]] = k.power
    sim.state.joint_q.assign(q)
    newton.eval_fk(model, sim.state.joint_q, sim.state.joint_qd, sim.state)


def run(kicks: tuple[Kick, ...], device: str = "cpu", net: bool = True, graph: bool = False,
        seconds: float = SECONDS) -> dict:
    """Run every kick; the per-world outcome (``score``) and the wall ms per exchange step."""
    sim = build(kicks, device, net, True, graph)
    labels = np.array([str(x).rsplit("/", 1)[-1] for x in sim.model.body_label])
    ball = np.flatnonzero(labels == "ball")
    xs, vs = [], []
    t_wall, steps = 0.0, 0
    for k in range(int(round(seconds / CONTROL_DT))):
        t0 = time.perf_counter()
        sim.step_period()
        if k >= 1:
            wp.synchronize_device(sim.model.device)
            t_wall += time.perf_counter() - t0
            steps += 2 ** int(sim.clock.j)
        xs.append(sim.state.body_q.numpy()[ball, :3].astype(np.float64))
        vs.append(sim.state.body_qd.numpy()[ball].astype(np.float64))
    t = (np.arange(len(xs)) + 1) * CONTROL_DT
    return {"task": "task_football_kick", "worlds": len(kicks), "device": device, "net": net,
            "kicks": [asdict(k) for k in kicks], "foot_mps": [round(foot_speed(k.power), 2) for k in kicks],
            **score(t, np.stack(xs), np.stack(vs), net),
            "ms_per_step": 1e3 * t_wall / max(steps, 1)}


def score(t: np.ndarray, x: np.ndarray, v: np.ndarray, net: bool) -> dict:
    """Per world from the ball's centre ``x`` [K, W, 3] and twist ``v`` [K, W, 6] each period."""
    speed = np.linalg.norm(v[:, :, :3], axis=2)
    k_l = np.argmax(speed, axis=0)                                   # the period after the kick
    w_idx = np.arange(x.shape[1])
    vl, wl = v[k_l, w_idx, :3], v[k_l, w_idx, 3:]
    cross = x[:, :, 0] > LINE_BACK + M.R_BALL
    hit = cross.any(axis=0)
    k_c = np.where(hit, np.argmax(cross, axis=0), -1)
    yz = np.full((x.shape[1], 2), np.nan)
    tc = np.full(x.shape[1], np.nan)
    for w in np.flatnonzero(hit):
        k = int(k_c[w])
        a = x[max(k - 1, 0), w]
        b = x[k, w]
        f = 0.0 if b[0] == a[0] else (LINE_BACK + M.R_BALL - a[0]) / (b[0] - a[0])
        yz[w] = (a + f * (b - a))[1:]
        tc[w] = t[max(k - 1, 0)] + f * (t[k] - t[max(k - 1, 0)])
    goal = hit & (np.abs(yz[:, 0]) <= 0.5 * GOAL_WIDTH) & (yz[:, 1] <= GOAL_HEIGHT)
    check = [flight_check(x[k_l[w], w], v[k_l[w], w], tc[w] - t[k_l[w]], yz[w]) for w in range(x.shape[1])]
    end = x[-1]
    held = goal & (end[:, 0] > LINE_BACK) & (end[:, 0] < NET_X + 0.9) & (np.abs(end[:, 1]) < 0.5 * NET_WIDTH)
    return {"success": goal.tolist(),
            "launch_mps": np.round(np.linalg.norm(vl, axis=1), 2).tolist(),
            "launch_deg": np.round(np.degrees(np.arctan2(vl[:, 2], np.hypot(vl[:, 0], vl[:, 1]))), 1).tolist(),
            "launch_yaw_deg": np.round(np.degrees(np.arctan2(vl[:, 1], vl[:, 0])), 1).tolist(),
            "launch_spin_rps": np.round(wl / (2.0 * math.pi), 2).tolist(),
            "launch_period": k_l.tolist(),
            "cross_t": np.round(tc, 3).tolist(), "cross_yz": np.round(yz, 3).tolist(),
            "flight_check_mm": [None if c is None else round(c, 1) for c in check],
            "held_in_goal": held.tolist() if net else None}


def flight_check(x0: np.ndarray, v0: np.ndarray, dt_cross: float, yz: np.ndarray) -> float | None:
    """mm between the simulated line crossing and ``model.flight`` from the launch state, for flights that stay
    clear of the pitch and the frame until the line (None otherwise)."""
    if not np.isfinite(dt_cross) or dt_cross <= 0.0:
        return None
    tr, xr, _ = M.flight(x0[:3], v0[:3], v0[3:], dt_cross + 0.05, dt=2e-4)
    past = np.flatnonzero(xr[:, 0] > LINE_BACK + M.R_BALL)
    if not len(past) or xr[:past[0] + 1, 2].min() < M.R_BALL + 0.02 or abs(xr[past[0], 1]) > 0.5 * GOAL_WIDTH - 0.2 \
            or xr[past[0], 2] > GOAL_HEIGHT - 0.2:
        return None
    k = int(past[0])
    f = (LINE_BACK + M.R_BALL - xr[k - 1, 0]) / (xr[k, 0] - xr[k - 1, 0])
    want = (xr[k - 1] + f * (xr[k] - xr[k - 1]))[1:]
    return float(np.linalg.norm(want - yz) * 1e3)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--worlds", type=int, default=len(KICKS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--no-net", action="store_true")
    ap.add_argument("--seconds", type=float, default=SECONDS)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    kicks = tuple(KICKS[i % len(KICKS)] for i in range(a.worlds))
    row = run(kicks, a.device, not a.no_net, a.graph, a.seconds)
    print(json.dumps(row))
    if a.out is not None:
        with a.out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
