# SPDX-License-Identifier: Apache-2.0
"""Crumble: a rigid chunk turns into preallocated MPM spare particles, preserving momentum (SPEC §3.8, TOP-5).

Assembly (``add_crumble_source``): a lattice is laid inside the chunk's convex hull, with spacing ≤ the thinnest width
over ``n_min`` (≥ 7 across, §3.8; default 8). The samples are centred on the COM and **moment-matched**: a symmetric
linear map makes Σ m r rᵀ equal the body's second moment ``S = tr(I)/2·1 − I``. The lattice inertia deficit
I(1 − 1/n²) therefore drops to rounding level (spike wp10-spikes/spike4_crumble_mpm.py: 4e-3 raw at n = 7, 4e-15
matched). Each spare particle has ``m = M/N`` and radius ``a/2`` (MPM volume 8r³ = a³), and is added inactive. A
disabled weld from the world to the chunk is added for parking.

Runtime (one source per world copy):
- Trigger: manual request, or "release" (every body bond of the chunk is broken).
- Convert: ``x_i = X·(com + r_i)``, ``v_i = v + ω × (x_i − x_com)``, ``particle_qd_grad = [ω]×``, and the MPM history
  is reset as in Newton's own reset (newton@ac7f7830:newton/_src/solvers/implicit_mpm/implicit_mpm_solver_kernels.py:
  1003-1021). ACTIVE is set in the Model, in the MPM entry's view, and in place in the solver's cached flag and
  volume arrays (SCR-WP10-1).
- Retire the body: a body cannot be removed from SolverMuJoCo. It is parked: ``joint_q`` moves it ``park_offset``
  away at the same height (SolverMuJoCo pushes ``joint_q`` every step, solver_mujoco.py:4997), its velocity is
  zeroed, its parking weld is enabled at that pose, and ``body_active`` is set to 0 so the ledger and health skip
  it (SCR-WP10-2).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import warp as wp
from scipy.spatial import ConvexHull

from alloy._vendor import newton
from alloy.core.errors import AssemblyError, CapacityError
from alloy.topology.bonds import NS, register_custom_attributes
from alloy.topology.types import TopologyBudget

log = logging.getLogger("alloy.topology")

TRIGGERS = {"none": 0, "manual": 1, "release": 2}
N_MIN = 8


@dataclass(frozen=True)
class CrumbleRecord:
    particles: range                 # template-local spare particles
    park_eq: int                     # parking weld row (mujoco:equality_constraint)
    spacing: float                   # lattice spacing a [m]
    match_scale: tuple[float, float]  # (min, max) eigenvalue of the moment-matching map


def box_hull(hx: float, hy: float, hz: float) -> np.ndarray:
    return np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)


def _sqrtm(s: np.ndarray, power: float) -> np.ndarray:
    w, u = np.linalg.eigh(s)
    if w.min() <= 0.0:
        raise AssemblyError(f"crumble: second moment not positive definite (eigenvalues {w})")
    return u @ np.diag(w ** power) @ u.T


def moment_match(r: np.ndarray, m: np.ndarray, s_body: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric A with A·S·Aᵀ = S_body (S = Σ m r rᵀ); returns (A·r, eigenvalues of A)."""
    s = np.einsum("i,ij,ik->jk", m, r, r)
    half = _sqrtm(s_body, 0.5)
    a = half @ _sqrtm(half @ s @ half, -0.5) @ half
    return r @ a.T, np.linalg.eigvalsh(a)


def lattice_in_hull(points: np.ndarray, n_min: int = N_MIN, spacing: float | None = None) -> tuple[np.ndarray, float]:
    """Cell-centre lattice inside the convex hull of points; spacing ≤ thinnest width / n_min."""
    hull = ConvexHull(points)
    normals, offsets = hull.equations[:, :3], hull.equations[:, 3]
    proj = points @ normals.T
    width = float((proj.max(0) - proj.min(0)).min())
    a = min(width / n_min, spacing if spacing else np.inf)
    lo, hi = points.min(0), points.max(0)
    axes = [np.arange(lo[k] + 0.5 * a, hi[k], a) for k in range(3)]
    g = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    inside = np.all(g @ normals.T + offsets <= 1e-12 * width, axis=1)
    return g[inside], a


def sample_chunk(points: np.ndarray, mass: float, inertia: np.ndarray, *, n_min: int = N_MIN,
                 spacing: float | None = None) -> tuple[np.ndarray, float, np.ndarray]:
    """Body-frame offsets from the COM (centroid 0, second moment = the body's), spacing, matching eigenvalues."""
    g, a = lattice_in_hull(np.asarray(points, float), n_min, spacing)
    if len(g) < 4:
        raise AssemblyError(f"crumble: only {len(g)} lattice points inside the hull at spacing {a:.3g} m")
    r = g - g.mean(axis=0)
    i_b = np.asarray(inertia, float).reshape(3, 3)
    s_body = 0.5 * np.trace(i_b) * np.eye(3) - i_b
    r, eig = moment_match(r, np.full(len(r), mass / len(r)), s_body)
    return r, a, eig


def add_crumble_source(builder: newton.ModelBuilder, body: int, hull_points: np.ndarray, *, n_min: int = N_MIN,
                       spacing: float | None = None, trigger: Literal["none", "manual", "release"] = "release",
                       budget: TopologyBudget | None = None,
                       particle_attributes: Mapping[str, Any] | None = None) -> CrumbleRecord:
    """Reserve spare MPM particles for one chunk body. hull_points are in the body frame (origin), metres. Call
    after the body's shapes are added (mass and inertia come from the builder). particle_attributes go to
    every spare (e.g. ``mpm:*`` material values)."""
    from alloy._vendor.newton.solvers import mujoco as nmj

    register_custom_attributes(builder)
    mass = float(builder.body_mass[body])
    if mass <= 0.0:
        raise AssemblyError(f"crumble source body {body} has no mass")
    com = np.asarray(builder.body_com[body], float)
    r, a, eig = sample_chunk(hull_points, mass, np.asarray(builder.body_inertia[body], float), n_min=n_min,
                             spacing=spacing)
    cap = budget.spare_particles if budget is not None else len(r)
    if len(r) > cap:
        raise CapacityError(f"crumble body {body}: {len(r)} samples exceed TopologyBudget.spare_particles = {cap}")
    if eig.min() < 0.8 or eig.max() > 1.25:
        log.warning("crumble body %d: moment matching scales samples by %.3f-%.3f; the body inertia does not "
                    "match its hull", body, eig.min(), eig.max())
    xf = wp.transform(*builder.body_q[body])
    p0 = builder.particle_count
    for loc in com + r:
        attrs = {f"{NS}:crumble_body": body, f"{NS}:crumble_local": wp.vec3(*loc), **(particle_attributes or {})}
        builder.add_particle(wp.transform_point(xf, wp.vec3(*loc)), wp.vec3(0.0, 0.0, 0.0), mass / len(r),
                             radius=0.5 * a, flags=0, custom_attributes=attrs)
    eq = nmj.add_equality_weld(builder, -1, body, relpose=xf, enabled=False, label=f"crumble_park_{body}")
    for name, val in (("crumble_park_eq", eq), ("crumble_trigger", TRIGGERS[trigger])):
        attr = builder.custom_attributes[f"{NS}:{name}"]
        attr.values = attr.values or {}
        attr.values[body] = val
    log.info("crumble source body %d: %d spares at %.3g m, moment match %.4f-%.4f", body, len(r), a,
             eig.min(), eig.max())
    return CrumbleRecord(range(p0, builder.particle_count), eq, a, (float(eig.min()), float(eig.max())))
