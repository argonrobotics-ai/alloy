# SPDX-License-Identifier: Apache-2.0
"""Base placement for an arm: can it reach every key pose of a scene, with room to spare on every joint?

``reach(spec, base, poses)`` solves IK for the scene's key TCP poses (grasp, lift, pour, insert ...) from the arm's
home branch and reports per pose the IK error, the smallest joint-limit margin and pass/fail. ``place_base`` scores
candidate bases (a list, or ``grid(xs, ys, z, yaws)``) the same way in one batched IK (targets moved into each
candidate's base frame) and returns the best: most poses reached, then the largest worst-case margin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alloy.arms.rig import Rig
from alloy.arms.specs import ArmSpec
from alloy.arms.poses import compose, invert, quat_from_rpy

MARGIN_RAD = 0.20          # the kit's comfort margin: >= 0.2 rad on every joint


@dataclass(frozen=True)
class ReachReport:
    base: tuple[float, ...]
    ok: np.ndarray            # [P] reached within IK tolerance with >= margin on every joint
    pos_err_m: np.ndarray     # [P]
    ang_err_rad: np.ndarray   # [P]
    margin_rad: np.ndarray    # [P] smallest distance of any joint to its limit
    q: np.ndarray             # [P, J]
    floor_m: np.ndarray | None = None   # [P] lowest point of the arm's collision geoms above ``floor_z`` (base links
                                        # excluded); with a floor, ``ok`` also needs >= ``clearance``

    @property
    def all_ok(self) -> bool:
        return bool(self.ok.all())

    def summary(self) -> str:
        bad = np.nonzero(~self.ok)[0]
        worst = int(np.argmin(self.margin_rad))
        floor = "" if self.floor_m is None else f", lowest arm point {1e3 * self.floor_m.min():.0f} mm above the floor"
        return (f"base {tuple(round(x, 3) for x in self.base[:3])}: {int(self.ok.sum())}/{len(self.ok)} poses, "
                f"worst margin {self.margin_rad[worst]:.2f} rad (pose {worst}){floor}"
                + (f", failing poses {bad.tolist()}" if len(bad) else ""))  # fmt: skip


def grid(xs, ys, z: float = 0.0, yaws_deg=(0.0,)) -> np.ndarray:
    """``[C, 7]`` candidate bases on an x-y grid at height ``z``, each yawed by every angle in ``yaws_deg``."""
    out = [(x, y, z, *quat_from_rpy((0.0, 0.0, yaw))) for x in xs for y in ys for yaw in yaws_deg]
    return np.array(out, dtype=np.float64)


def _base7(base) -> np.ndarray:
    b = np.asarray(base, dtype=np.float64)
    return np.concatenate([b, [0.0, 0.0, 0.0, 1.0]]) if b.shape[-1] == 3 else b


def _solve_local(spec: ArmSpec, local: np.ndarray, margin: float, device: str):
    """IK of ``local [N, 7]`` TCP poses for the arm at the origin -> (ok, pos err, ang err, margin, q)."""
    rig = Rig.single(spec, (0.0, 0.0, 0.0), visuals=False)
    ik = rig.make_ik(len(local), device)
    q, err = ik.solve(local[:, None, :], rig.ready_seed(len(local)), check=False)
    lo, hi = ik.bind.lower[0], ik.bind.upper[0]
    room = np.minimum(q[:, 0] - lo, hi - q[:, 0]).min(axis=-1)
    hit = (err[:, 0, 0] <= ik.tol.pos_m) & (err[:, 0, 1] <= ik.tol.ang_rad)
    return hit & (room >= margin), err[:, 0, 0], err[:, 0, 1], room, q[:, 0]


BASE_LINKS = 2             # the first two bodies under the root (link0 / link1, base / shoulder) may sit on the floor


def floor_clearance(spec: ArmSpec, base, q: np.ndarray, floor_z: float = 0.0) -> np.ndarray:
    """``[P]`` lowest point of the arm's collision geoms (world AABBs, MuJoCo C) above ``floor_z`` at angles
    ``q [P, J]`` with the arm mounted at ``base``; the base links are left out."""
    import mujoco

    m = Rig.single(spec, base, visuals=False).mj_model()
    d = mujoco.MjData(m)
    adr = [m.joint(j).qposadr[0] for j in spec.joints]
    chain = [m.body(spec.root).id]
    for _ in range(BASE_LINKS):   # the root and the first links of the chain below it
        chain.append(next(b for b in range(m.nbody) if m.body_parentid[b] == chain[-1] and b != chain[-1]))
    skip = set(chain)
    geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] not in skip and m.geom_contype[g] | m.geom_conaffinity[g]]
    out = np.empty(len(q))
    for k, qk in enumerate(np.atleast_2d(q)):
        d.qpos[adr] = qk
        mujoco.mj_kinematics(m, d)
        lo = []
        for g in geoms:   # world AABB bottom: centre z minus the half extents projected on z
            R, c, h = d.geom_xmat[g].reshape(3, 3), m.geom_aabb[g][:3], m.geom_aabb[g][3:]
            lo.append(d.geom_xpos[g][2] + (R @ c)[2] - np.abs(R[2]) @ h)
        out[k] = min(lo) - floor_z
    return out


def reach(spec: ArmSpec, base, poses, *, margin: float = MARGIN_RAD, device: str = "cpu", floor_z: float | None = None,
          clearance: float = 0.01) -> ReachReport:  # fmt: skip
    """Key TCP ``poses [P, 7]`` (world frame) for ``spec`` mounted at ``base`` (x y z [qx qy qz qw]); with
    ``floor_z`` the arm must also stay ``clearance`` above that floor (a bench top) at every pose."""
    b = _base7(base)
    poses = np.atleast_2d(np.asarray(poses, dtype=np.float64))
    local = compose(invert(b)[None], poses)
    ok, pe, ae, room, q = _solve_local(spec, local, margin, device)
    floor = None
    if floor_z is not None:
        floor = floor_clearance(spec, b, q, floor_z)
        ok = ok & (floor >= clearance)
    return ReachReport(tuple(b), ok, pe, ae, room, q, floor)


def place_base(spec: ArmSpec, poses, candidates, *, margin: float = MARGIN_RAD, device: str = "cpu",
               floor_z: float | None = None,
               clearance: float = 0.01) -> tuple[ReachReport, list[ReachReport]]:  # fmt: skip
    """Best of ``candidates [C, 3 | 7]`` for reaching every pose of ``poses [P, 7]``; also every candidate's report.
    With ``floor_z`` the arm must clear that floor by ``clearance`` (ranked by poses reached, then the smaller of the
    joint margin and 5 x the floor clearance)."""
    cands = np.array([_base7(c) for c in np.atleast_2d(candidates)])
    poses = np.atleast_2d(np.asarray(poses, dtype=np.float64))
    local = compose(invert(cands)[:, None, :], poses[None]).reshape(-1, 7)
    ok, pe, ae, room, q = _solve_local(spec, local, margin, device)
    P = len(poses)
    reports = []
    for i, c in enumerate(cands):
        sl = slice(i * P, (i + 1) * P)
        floor = None if floor_z is None else floor_clearance(spec, c, q[sl], floor_z)
        good = ok[sl] if floor is None else ok[sl] & (floor >= clearance)
        reports.append(ReachReport(tuple(c), good, pe[sl], ae[sl], room[sl], q[sl], floor))
    score = (lambda r: float(r.margin_rad.min())) if floor_z is None else \
        (lambda r: min(float(r.margin_rad.min()), 5.0 * float(r.floor_m.min())))  # fmt: skip
    best = max(reports, key=lambda r: (int(r.ok.sum()), score(r)))
    return best, reports


__all__ = ["MARGIN_RAD", "ReachReport", "floor_clearance", "grid", "place_base", "reach"]
