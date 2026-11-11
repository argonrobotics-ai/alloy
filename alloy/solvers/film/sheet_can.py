# SPDX-License-Identifier: Apache-2.0
"""Law S3 of J2 metal sheets (SCR-WP7-15 P3): a can wall dented by a finger, a measured row (no closed form).

An open cylinder of a sourced metal card (a drawn tinplate can wall: r 33 mm, 60 mm tall, 0.2 mm) on the film route,
both rims pinned (the can's ends and beads), pressed radially at mid-height by a kinematic sphere (a fingertip, r 8 mm,
μ 0.5), held, released; the wall springs back and settles. The row reports the peak and kept dent at the finger, the
largest layer ε̄p, and that the run stayed finite.

**Quasi-static by mass scaling**, as S2 (sheet_dent.py): density × ``mass_scale`` with gravity off. The wall's
D dt² / (ρ t dx⁴) is about 0.85 at 10³ (D = 0.15 N m for 0.2 mm tinplate, dx 3.2 mm). Heavier walls ring: at 10⁵-10⁶
the wall moved at 3.5-10x the finger's speed and yielded during the lift (ε̄p 0.16-0.71 %), where at 10³-10⁴ the same
3 mm press stayed elastic.

**What the film can and cannot say here.** The film's membrane is scaled 1e-4 for VBD, and the strain cap at σ_y/E
carries the stretch. A finger therefore ovalizes the wall in near-inextensional bending (its curvature change about
3 δ/R²), where a real can wall also stretches and buckles locally. Plasticity is per hinge on its normal curvature
(docs/notes/plasticity-sheets.md, gaps). The row is a smoke of the J2 hinges on a curved, closed sheet with rest
curvature (the hinges' authored rest angles), not a validated dent depth.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace

import numpy as np
import warp as wp

from alloy._vendor import newton

from .emit import add_film_mesh
from .rigs import FilmRig, pin
from .sheet_dent import set_damping
from .sheet_laws import metal_spec


def can_mesh(radius: float, height: float, n_theta: int, n_z: int) -> tuple[np.ndarray, np.ndarray]:
    """An open cylinder about z, bottom rim at z = 0: n_theta x (n_z + 1) vertices (vertex i, j at angle 2π i / n_theta,
    height j H / n_z), triangles wound with outward normals, periodic in the angle."""
    th = np.arange(n_theta) * (2.0 * math.pi / n_theta)
    z = np.linspace(0.0, height, n_z + 1)
    v = np.array([(radius * math.cos(a), radius * math.sin(a), zz) for a in th for zz in z])
    idx = lambda i, j: (i % n_theta) * (n_z + 1) + j                          # noqa: E731
    tris = []
    for i in range(n_theta):
        for j in range(n_z):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            tris += [(a, b, c), (a, c, d)] if (i + j) % 2 == 0 else [(a, b, d), (b, c, d)]
    return v, np.asarray(tris, np.int64)


@dataclass
class CanRow:
    depth: float          # m, the finger's travel past first touch
    peak_dent: float      # m, the wall's largest radial inward displacement at the finger
    kept_dent: float      # m, after release and settling
    max_strain: float     # the largest layer ε̄p
    finite: bool
    steps: int
    wall_s: float


def s3_can_dent(material: str = "steel_tinplate", thickness: float = 2.0e-4, radius: float = 0.033,
                height: float = 0.06, n_theta: int = 64, n_z: int = 20, r_finger: float = 0.008,
                depth: float = 8.0e-3, speed: float = 5.0e-3, hold: float = 0.5, settle: float = 3.0,
                dt: float = 1.0e-3, iterations: int = 20, mass_scale: float = 1.0e3, bend_damping: float = 1.0e-3,
                settle_damping: float = 0.05, contact_ke: float = 1.0e5, strain_cap: float | None = None,
                device: str = "cpu") -> CanRow:
    """S3 (module doc): the finger travels ``depth`` radially in at ``speed``, holds ``hold`` s, comes back, leaves,
    and the wall settles ``settle`` s with the hinge damping raised to ``settle_damping``. ``strain_cap`` replaces
    the film's stretch cap (σ_y/E by default, a hard limit; WP7's large-cap row)."""
    t0 = time.perf_counter()
    spec = metal_spec(material, thickness)
    spec = replace(spec, density=spec.density * mass_scale).with_extras(bend_damping=bend_damping)
    if strain_cap is not None:
        spec = replace(spec, yield_stress=strain_cap * spec.youngs)            # j2_yield (the section) is separate
    v, tris = can_mesh(radius, height, n_theta, n_z)
    rims = np.flatnonzero((np.abs(v[:, 2]) < 1e-9) | (np.abs(v[:, 2] - height) < 1e-9))
    x0 = radius + spec.extras.contact_radius + r_finger                        # finger centre at first touch

    def build(b: newton.ModelBuilder) -> dict:
        p0, _ = add_film_mesh(b, v, tris, spec)
        pin(b, rims + p0)
        body = b.add_body(xform=wp.transform(wp.vec3(x0, 0.0, 0.5 * height), wp.quat_identity()), mass=0.0,
                          is_kinematic=True)
        b.add_shape_sphere(body, radius=r_finger,
                           cfg=newton.ModelBuilder.ShapeConfig(mu=0.5, ke=contact_ke, density=0.0))
        return {"p0": [p0], "body": [body]}

    rig = FilmRig(build=build, device=device, dt=dt, iterations=iterations, gravity=0.0,
                  vbd={"alloy_initial_guess": "adaptive"})
    m = rig.model
    m.soft_contact_ke, m.soft_contact_kd, m.soft_contact_mu = contact_ke, 0.0, 0.5
    rig.pipe = newton.CollisionPipeline(m, soft_contact_gap=0.002)
    rig.contacts = rig.pipe.contacts()
    tip = int(np.argmin(np.hypot(v[:, 0] - radius, v[:, 2] - 0.5 * height) + np.abs(v[:, 1])))
    dent = lambda: radius - float(np.hypot(*rig.q()[tip, :2]))                # noqa: E731
    down = np.arange(1, int(round(depth / (speed * dt))) + 1) * speed * dt
    peak, n = 0.0, 0
    for d in np.concatenate([down, np.full(int(round(hold / dt)), depth), down[::-1]]):
        _place(rig, x0 - d, 0.5 * height)
        rig.step(1)
        n += 1
        peak = max(peak, dent())
    _place(rig, x0 + 0.05, 0.5 * height)
    set_damping(rig, settle_damping)
    for _ in range(int(round(settle / dt))):
        rig.step(1)
        n += 1
    return CanRow(depth=depth, peak_dent=peak, kept_dent=dent(),
                  max_strain=float(rig.solver._j2.ebar_max.numpy().max()), finite=bool(np.isfinite(rig.q()).all()),
                  steps=n, wall_s=time.perf_counter() - t0)


def _place(rig: FilmRig, x: float, z: float) -> None:
    b = int(rig.names[0]["body"][0])
    for st in (rig.s0, rig.s1):
        q = st.body_q.numpy()
        q[b] = (x, 0.0, z, 0.0, 0.0, 0.0, 1.0)
        st.body_q.assign(q)
