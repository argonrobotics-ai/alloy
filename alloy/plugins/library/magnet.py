# SPDX-License-Identifier: Apache-2.0
"""Permanent magnets as point dipoles: forces and torques between every pair of magnets in a world.

For dipoles m_i, m_j (A·m²) at separation r = p_j - p_i (unit r̂, distance d; μ0/4π = 1e-7 T·m/A):

    F_j = 3μ0/(4π d⁴) [(m_i·r̂) m_j + (m_j·r̂) m_i + (m_i·m_j) r̂ - 5 (m_i·r̂)(m_j·r̂) r̂],   F_i = -F_j
    τ_j = m_j × B_i(p_j),   B_i(p) = μ0/(4π d³) [3 r̂ (m_i·r̂) - m_i]
    U   = μ0/(4π d³) [m_i·m_j - 3 (m_i·r̂)(m_j·r̂)]

Coaxial aligned dipoles attract with 3μ0 m_i m_j / (2π d⁴). Below ``min_distance`` the field is evaluated at
``min_distance`` and -F_j acts at p_j - r̂·min_distance, so momentum and angular momentum stay exactly balanced.
A uniform ``external_field`` B (T) adds the torque m × B (no force). A NdFeB cube of side s has m ≈ 1.03e6·s³ A·m².
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.base import ForcePlugin, Forces
from alloy.plugins.errors import PluginError
from alloy.plugins.funcs import add_wrench, body_point, body_vector, com_world, torque_wrench, wrench_at
from alloy.plugins.harness.scene import Law, LawResult, TestScene, body_xform
from alloy.plugins.npgeom import quat_rotate, transform_point

MU0_4PI = 1.0e-7          # T·m/A


@wp.kernel
def _dipole_kernel(body_q: wp.array[wp.transform], body_com: wp.array[wp.vec3], bodies: wp.array2d[wp.int32],
                   offsets: wp.array[wp.vec3], axes: wp.array[wp.vec3], moments: wp.array[wp.float32],
                   r_min: wp.float32, b_ext: wp.vec3, body_f: wp.array[wp.spatial_vector]):
    w = wp.tid()                                   # one thread per world: deterministic, no atomics
    n = bodies.shape[1]
    for i in range(n):
        bi = bodies[w, i]
        pi = body_point(body_q, bi, offsets[i])
        mi = body_vector(body_q, bi, axes[i]) * moments[i]
        ci = com_world(body_q, body_com, bi)
        add_wrench(body_f, bi, torque_wrench(wp.cross(mi, b_ext)))
        for j in range(i + 1, n):
            bj = bodies[w, j]
            if bj == bi:
                continue                           # two dipoles on one body: internal, no net effect
            pj = body_point(body_q, bj, offsets[j])
            mj = body_vector(body_q, bj, axes[j]) * moments[j]
            cj = com_world(body_q, body_com, bj)
            r = pj - pi
            d = wp.length(r)
            rh = wp.vec3(0.0, 0.0, 1.0)
            if d > 0.0:
                rh = r / d
            dd = wp.max(d, r_min)
            mir = wp.dot(mi, rh)
            mjr = wp.dot(mj, rh)
            k4 = 3.0 * MU0_4PI / (dd * dd * dd * dd)
            fj = k4 * (mir * mj + mjr * mi + wp.dot(mi, mj) * rh - 5.0 * mir * mjr * rh)
            k3 = MU0_4PI / (dd * dd * dd)
            tau_j = wp.cross(mj, k3 * (3.0 * mir * rh - mi))
            tau_i = wp.cross(mi, k3 * (3.0 * mjr * rh - mj))
            add_wrench(body_f, bj, wrench_at(fj, pj, cj) + torque_wrench(tau_j))
            add_wrench(body_f, bi, wrench_at(-fj, pj - dd * rh, ci) + torque_wrench(tau_i))


def _unit(v: Any) -> tuple[float, float, float]:
    a = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(a))
    if a.shape != (3,) or n == 0.0:
        raise PluginError(f"magnet axis {v!r} must be a nonzero (x, y, z) vector in the body frame")
    return tuple(float(x) for x in a / n)


class Magnet(ForcePlugin):
    """Point-dipole permanent magnets on rigid bodies (forces, torques and an optional uniform field).

    Example::

        mag = Magnet(bodies=("a", "b"), moments=(1.0, 1.0), axes=((1, 0, 0), (1, 0, 0)))
        sim = alloy.plugins.simulation(built, [mag])
    """

    name = "magnet"
    summary = "point-dipole magnets: pairwise forces and torques, optional uniform field"

    @dataclass
    class Config:
        bodies: tuple[str, ...] = ("magnet_a", "magnet_b")    # body label of each magnet
        moments: tuple[float, ...] = (1.0, 1.0)               # dipole moment [A·m²] (2 cm NdFeB cube ≈ 8)
        axes: tuple[tuple[float, float, float], ...] = ()     # body-frame north direction; default +z each
        offsets: tuple[tuple[float, float, float], ...] = ()  # body-frame dipole position; default origin
        min_distance: float = 2.0e-3                          # m; closer pairs are evaluated at this distance
        external_field: tuple[float, float, float] = (0.0, 0.0, 0.0)   # uniform B [T]

        def __post_init__(self) -> None:
            n = len(self.bodies)
            if n < 1 or len(self.moments) != n:
                raise ValueError(f"give one moment per body: {n} bodies, {len(self.moments)} moments")
            for name in ("axes", "offsets"):
                if len(getattr(self, name)) not in (0, n):
                    raise ValueError(f"{name} needs one (x, y, z) per body ({n}) or none")
            if self.min_distance <= 0:
                raise ValueError("min_distance must be > 0 m (use a value below your magnets' diameter)")

    @property
    def internal(self) -> bool:       # a uniform external field torques the scene from outside
        return not any(self.config.external_field)

    def setup(self, ctx: Any) -> None:
        c = self.config
        n = len(c.bodies)
        ids = np.stack([ctx.bodies(label).numpy() for label in c.bodies], axis=1)      # [W, n]
        self.bodies = ctx.array(ids.astype(np.int32), wp.int32)
        self.axes = ctx.array([_unit(a) for a in (c.axes or [(0.0, 0.0, 1.0)] * n)], wp.vec3)
        self.offsets = ctx.array(c.offsets or [(0.0, 0.0, 0.0)] * n, wp.vec3)
        self.moments = ctx.array(c.moments, wp.float32)
        self.body_com = ctx.model.body_com

    def apply(self, state: newton.State, forces: Forces, dt: float) -> None:
        c = self.config
        wp.launch(_dipole_kernel, dim=self.ctx.world_count,
                  inputs=[state.body_q, self.body_com, self.bodies, self.offsets, self.axes, self.moments,
                          float(c.min_distance), wp.vec3(*c.external_field)],
                  outputs=[forces.body_f], device=self.ctx.device)

    def energy(self, body_q: np.ndarray) -> np.ndarray:
        """``[W]`` magnetic potential energy [J] of every world at poses ``body_q [B, 7]`` (host, float64)."""
        c, ids = self.config, self.bodies.numpy()
        axes, offs = self.axes.numpy().astype(np.float64), self.offsets.numpy().astype(np.float64)
        out = np.zeros(ids.shape[0])
        for w in range(ids.shape[0]):
            p = [transform_point(body_q[b], offs[i]) for i, b in enumerate(ids[w])]
            m = [quat_rotate(body_q[b, 3:7], axes[i]) * c.moments[i] for i, b in enumerate(ids[w])]
            for i in range(len(p)):
                for j in range(i + 1, len(p)):
                    if ids[w, i] != ids[w, j]:
                        out[w] += _pair_energy(p[i], m[i], p[j], m[j], c.min_distance)
        return out

    def test_scene(self) -> TestScene:
        """One 2 cm NdFeB-density cube per configured magnet, in a row along x 3 cm apart, on a low-friction floor:
        they push and pull each other, slide, collide and twist."""
        labels = tuple(self.config.bodies)
        return TestScene(build=lambda b: _row_of_cubes(b, labels), periods=25, ground_mu=0.1)

    def laws(self) -> list[Law]:
        pair = {"bodies": ("magnet_a", "magnet_b"), "moments": (1.0, 1.0), "axes": (), "offsets": ()}
        coaxial = {**pair, "axes": ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))}
        free = {"ground": False, "gravity": (0.0, 0.0, 0.0)}
        return [
            Law("inverse_fourth", _check_inverse_fourth, TestScene(build=_pair_build(0.05, 0.01), periods=1, **free),
                "coaxial 3μ0m²/(2πd⁴), side-by-side 3μ0m²/(4πd⁴), at d and 2d", pair),
            Law("energy", _check_energy, TestScene(build=_pair_build(0.08, 0.01), periods=13, dt=5.0e-4, **free),
                "kinetic + magnetic energy constant while two magnets close", coaxial),
        ]


def _pair_energy(pi: np.ndarray, mi: np.ndarray, pj: np.ndarray, mj: np.ndarray, r_min: float) -> float:
    r = pj - pi
    d = float(np.linalg.norm(r))
    rh = r / d if d > 0 else np.array([0.0, 0.0, 1.0])
    dd = max(d, r_min)
    return MU0_4PI / dd**3 * (float(mi @ mj) - 3.0 * float(mi @ rh) * float(mj @ rh))


def _row_of_cubes(b: newton.ModelBuilder, labels: tuple[str, ...], half: float = 0.01) -> None:
    cfg = newton.ModelBuilder.ShapeConfig(density=7500.0, mu=0.1)
    for i, label in enumerate(labels):
        body = b.add_body(xform=body_xform(0.03 * i, 0.002 * (i % 2), half + 1e-4), label=label)
        b.add_shape_box(body, hx=half, hy=half, hz=half, cfg=cfg)


def _pair_build(separation: float, radius: float):
    def build(b: newton.ModelBuilder) -> None:
        cfg = newton.ModelBuilder.ShapeConfig(density=7500.0, mu=0.2)
        for label, x in (("magnet_a", 0.0), ("magnet_b", separation)):
            body = b.add_body(xform=body_xform(x, 0.0, 0.5), label=label)
            b.add_shape_sphere(body, radius=radius, cfg=cfg)
    return build


def _law_magnet(run: Any) -> Magnet:
    return run.plugin


def _check_inverse_fourth(run: Any) -> LawResult:
    mag = _law_magnet(run)
    a, b = run.bodies("magnet_a")[0], run.bodies("magnet_b")[0]
    m1, m2 = mag.config.moments[0], mag.config.moments[1]
    q0 = run.trace.body_q[0].copy()
    worst = 0.0
    for d in (0.03, 0.06):
        for axis, expect in (((0.0, 0.0, 1.0), -6.0e-7), ((1.0, 0.0, 0.0), 3.0e-7)):
            q = q0.copy()
            q[b, :3] = q[a, :3] + np.asarray(axis) * d       # dipoles point +z: z offset = coaxial, x = side by side
            f = run.probe(q)
            along = float(f[b, :3] @ np.asarray(axis))
            want = expect * m1 * m2 / d**4                   # coaxial attracts (-), side-by-side repels (+)
            worst = max(worst, abs(along - want) / abs(want))
    return LawResult.within(worst, 1.0e-4, "max relative error of the force vs 3μ0m²/(2πd⁴) and 3μ0m²/(4πd⁴)",
                            "check the dipole force formula and the r̂ direction (from i to j)")


def _check_energy(run: Any) -> LawResult:
    mag = _law_magnet(run)
    tr, m = run.trace, run.model.body_mass.numpy().astype(np.float64)
    a, b = run.bodies("magnet_a")[0], run.bodies("magnet_b")[0]
    far = np.linalg.norm(tr.body_q[:, b, :3] - tr.body_q[:, a, :3], axis=-1) >= 0.03
    keep = np.cumprod(far).astype(bool)                       # every period before they first come within 3 cm
    if keep.sum() < 5:
        return LawResult(False, math.nan, 0.02, "fewer than 5 periods before contact", "weaken the magnets")
    rows = [a, b]                                             # world 0
    ke = 0.5 * (m[rows][None] * (tr.body_qd[:, rows, :3].astype(np.float64) ** 2).sum(-1)).sum(-1)
    u = np.array([mag.energy(q)[0] for q in tr.body_q])
    e = (ke + u)[keep]
    span = float(np.abs(u[keep] - u[0]).max())
    drift = float(np.abs(e - e[0]).max()) / max(span, 1e-12)
    return LawResult.within(drift, 0.02, f"energy drift / magnetic energy released ({span:.3g} J)",
                            "forces must be the gradient of the potential; check signs and the d⁻⁴ factor")


__all__ = ["Magnet"]
