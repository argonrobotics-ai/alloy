# SPDX-License-Identifier: Apache-2.0
"""Example: a parallel gripper with an OpenTac pad on each finger squeezes a box, lifts it, loosens until it slides.

Run ``python -m alloy.plugins.tactile.opentac.example --out run.npz`` (CPU, one world, ~2 s of sim time), then
``python alloy/plugins/tactile/opentac/clip.py run.npz clip.mp4`` (needs matplotlib and ffmpeg) for the clip.

The scene (MuJoCo route): a carriage on a vertical slide, two fingers on horizontal slides with position servos (the
squeeze is the servo stiffness times how far the target sits inside the box), and a 150 g box on a small stand. Each
finger's inner face carries ``opentac_finger_8x6_60pin`` (15 x 21 mm, 12 bumps), read with the v3 profile. Phases:
close (0.3 s), lift 10 mm (0.5 s), hold (0.2 s), loosen (0.5 s) to a grip whose friction is below the box's weight,
and hold it there while the box slides back down onto its stand (0.5 s).

MuJoCo's box-box contact puts a few points at the corners of the overlap, not a pressure field, so on a flat face the
board sees a few loaded bumps: the corners of the box face. That is the engine's contact model, not the skin's.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from alloy._vendor import newton

from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.plugins.tactile import Tactile
from alloy.plugins.tactile.opentac.model import decode

BOARD = "opentac_finger_8x6_60pin"


@dataclass(frozen=True)
class Gripper:
    obj_half: tuple[float, float, float] = (0.007, 0.015, 0.009)     # m: 14 x 30 x 18 mm box
    obj_mass: float = 0.15                                           # kg
    pad_half: tuple[float, float, float] = (0.012, 0.004, 0.0105)   # m: finger pad, 24 x 8 x 21 mm
    pedestal_half: float = 0.005    # m: the box stands on a 10 mm static cube, so the fingers clear the ground
    finger_mass: float = 0.05
    carriage_mass: float = 0.5
    finger_ke: float = 2000.0       # N/m: a 2.5 mm squeeze is ~5 N per finger
    lift_ke: float = 2.0e4          # N/m
    open_gap: float = 0.004         # m between each pad and the box at the start
    squeeze: float = 0.0025         # m the closed target sits inside the box
    loosen_to: float = 0.00033      # m: 2 x 1.0 x 2000 N/m x 0.33 mm = 1.32 N of friction < the 1.47 N weight
    lift: float = 0.01              # m
    mu_pad: float = 1.0             # their bump's nominal mu (opentac/sim/mujoco/params.py); MuJoCo takes the max
    mu_obj: float = 0.5


PHASES = (("close", 15), ("lift", 25), ("hold", 10), ("loosen", 25), ("slide", 25))


def _quat(cols: np.ndarray) -> tuple[float, float, float, float]:
    """(x, y, z, w) of the rotation whose columns are ``cols`` (pad x, y, z in the body frame)."""
    m = np.asarray(cols, float).T
    w = math.sqrt(max(0.0, 1.0 + m[0, 0] + m[1, 1] + m[2, 2])) / 2.0
    if w > 1e-6:
        return ((m[2, 1] - m[1, 2]) / (4 * w), (m[0, 2] - m[2, 0]) / (4 * w), (m[1, 0] - m[0, 1]) / (4 * w), w)
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(max(0.0, 1.0 + m[i, i] - m[j, j] - m[k, k])) * 2.0
    q = [0.0, 0.0, 0.0, (m[k, j] - m[j, k]) / s]
    q[i], q[j], q[k] = s / 4.0, (m[j, i] + m[i, j]) / s, (m[k, i] + m[i, k]) / s
    return tuple(q)


def pad_poses(g: Gripper) -> tuple[tuple, tuple]:
    """Tactile ``origins`` and ``rotations`` for (left, right): the inner faces, pad z toward the box."""
    x, y, z = np.eye(3)
    left = _quat(np.stack([-x, z, y]))           # pad x = -world x, pad y = world z, normal +y
    right = _quat(np.stack([x, z, -y]))          # pad x = world x, pad y = world z, normal -y
    return ((0.0, g.pad_half[1], 0.0), (0.0, -g.pad_half[1], 0.0)), (left, right)


def build(b: newton.ModelBuilder, g: Gripper = Gripper()) -> None:
    """One world: carriage (lift), two fingers (left_y, right_y), and the box labelled ``object``."""
    z0, inertia = 2 * g.pedestal_half + g.obj_half[2], ((1e-4, 0, 0), (0, 1e-4, 0), (0, 0, 1e-4))
    p = g.pedestal_half
    b.add_shape_box(-1, xform=body_xform(0.0, 0.0, p), hx=p, hy=p, hz=p,
                    cfg=newton.ModelBuilder.ShapeConfig(mu=g.mu_obj))
    carriage = b.add_link(xform=body_xform(0.0, 0.0, z0), mass=g.carriage_mass, inertia=inertia, label="carriage")
    joints = [b.add_joint_prismatic(-1, carriage, axis=(0.0, 0.0, 1.0), parent_xform=body_xform(0.0, 0.0, z0),
                                    target_ke=g.lift_ke, target_kd=2.0 * math.sqrt(g.lift_ke * 0.6),
                                    effort_limit=1.0e6, label="lift")]
    pad_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=g.mu_pad)
    for side, s in (("left", -1.0), ("right", 1.0)):
        y0 = s * (g.obj_half[1] + g.pad_half[1] + g.open_gap)
        f = b.add_link(xform=body_xform(0.0, y0, z0), mass=g.finger_mass, inertia=inertia, label=f"{side}_finger")
        joints.append(b.add_joint_prismatic(carriage, f, axis=(0.0, 1.0, 0.0), parent_xform=body_xform(0.0, y0, 0.0),
                                            target_ke=g.finger_ke, target_kd=2.0 * math.sqrt(g.finger_ke * 0.05),
                                            effort_limit=1.0e6, label=f"{side}_y"))
        b.add_shape_box(f, hx=g.pad_half[0], hy=g.pad_half[1], hz=g.pad_half[2], cfg=pad_cfg)
    b.add_articulation(joints, label="gripper")
    obj = b.add_body(xform=body_xform(0.0, 0.0, z0 + 5e-4), label="object")
    hx, hy, hz = g.obj_half
    b.add_shape_box(obj, hx=hx, hy=hy, hz=hz, cfg=newton.ModelBuilder.ShapeConfig(
        density=g.obj_mass / (8 * hx * hy * hz), mu=g.mu_obj))


def targets(g: Gripper, phase: str, frac: float) -> tuple[float, float]:
    """(lift, inward finger travel) targets at ``frac`` (0-1) through ``phase``."""
    closed = g.open_gap + g.squeeze
    if phase == "close":
        return 0.0, closed * frac
    if phase == "lift":
        return g.lift * frac, closed
    if phase == "hold":
        return g.lift, closed
    if phase == "loosen":
        return g.lift, closed - (g.squeeze - g.loosen_to) * frac
    return g.lift, g.open_gap + g.loosen_to


def tactile(g: Gripper = Gripper(), board: str = BOARD, **kw: Any) -> Tactile:
    origins, rotations = pad_poses(g)
    return Tactile(bodies=("left_finger", "right_finger"), origins=origins, rotations=rotations, opentac=board,
                   **{"opentac_readout": "v3", **kw})


def simulation(worlds: int = 1, device: str = "cpu", g: Gripper = Gripper(), plugins: list | None = None) -> Any:
    scene = TestScene(build=lambda b: build(b, g), worlds=worlds, periods=1, ground_mu=g.mu_obj)
    return build_simulation(scene, plugins if plugins is not None else [tactile(g)], device, worlds=worlds)


def _columns(sim: Any) -> dict[str, np.ndarray]:
    """Target indices (joint coordinates) of each gripper joint, in every world."""
    starts = sim.model.joint_q_start.numpy()
    cols: dict[str, list[int]] = {"lift": [], "left_y": [], "right_y": []}
    for j, lab in enumerate(sim.model.joint_label):
        key = lab.split("/")[-1]
        if key in cols:
            cols[key].append(int(starts[j]))
    return {k: np.asarray(v) for k, v in cols.items()}


def set_targets(sim: Any, cols: dict[str, np.ndarray], lift: float, inward: float) -> None:
    tgt = sim.control.joint_target_q.numpy()
    tgt[cols["lift"]], tgt[cols["left_y"]], tgt[cols["right_y"]] = lift, inward, -inward
    sim.control.joint_target_q.assign(tgt.astype(np.float32))


def run(g: Gripper = Gripper(), device: str = "cpu") -> dict[str, np.ndarray]:
    """Step the four phases on one world; per period: readings of both pads, the box pose and the targets."""
    t = tactile(g)
    sim = simulation(1, device, g, [t])
    cols, obj = _columns(sim), int(sim.model.body_label.index("object")) if hasattr(sim.model, "body_label") else 3
    rows: dict[str, list] = {k: [] for k in ("counts", "board_load", "pad_force", "bump_wrench", "obj_q", "lift",
                                             "inward", "phase")}
    for pi, (phase, n) in enumerate(PHASES):
        for k in range(n):
            lift, inward = targets(g, phase, (k + 1) / n)
            set_targets(sim, cols, lift, inward)
            sim.step_period()
            r = sim.plugins.read()[t.label]
            for key in ("counts", "board_load", "pad_force", "bump_wrench"):
                rows[key].append(np.array(r[key][0], copy=True))          # on the CPU these are views
            rows["obj_q"].append(np.array(sim.state.body_q.numpy()[obj], copy=True))
            rows["lift"].append(lift), rows["inward"].append(inward), rows["phase"].append(pi)
    out = {k: np.asarray(v) for k, v in rows.items()}
    board = t._board.board
    dec = decode(board, out["counts"], "v3")
    out["decoded_shear"], out["decoded_taxel_n"] = dec["bump_shear"], dec["taxel_n"]   # [N, P, B, 2], [N, P, T]
    out["taxel_xy"], out["bump_xy"], out["taxel_bump"] = board.xy, board.bump_xy, board.taxel_bump
    out["control_dt"], out["bump_height"] = np.float64(0.02), np.float64(board.bump_height)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="opentac_grip.npz")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    out = run(device=a.device)
    np.savez_compressed(a.out, **out)
    z = out["obj_q"][:, 2]
    print(f"wrote {a.out}: {len(z)} periods; box z {z[0] * 1e3:.1f} -> max {z.max() * 1e3:.1f} -> end "
          f"{z[-1] * 1e3:.1f} mm; peak count {out['counts'].max():.0f}")


if __name__ == "__main__":
    main()
