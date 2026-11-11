# SPDX-License-Identifier: Apache-2.0
"""Drop test for an imported object until Alloy's physics-check tool covers it: hulls on one body, dropped onto a plane.

The body carries one convex shape per hull of ``physics.hulls.obj``; hull densities are scaled by
V_mesh / sum(V_hulls) so the body mass equals the card mass (the rigid emitter's rule for ``collision: decompose``). It starts upright
(scan pose) with its lowest point ``height`` above the plane and falls for ``duration`` seconds under MuJoCo C
(``SolverMuJoCo(use_mujoco_cpu=True)``: float64 on CPU), with the contact gains Alloy's assembler gives a dynamic
MuJoCo body (``alloy.scene.native_gains``: series stiffness ``K``, damping from restitution with the CCD resting
floor, capped for stability at ``dt``) on both the body and the plane. With MuJoCo's default solref (0.02 s) a body
sinks about 4 mm at rest, and MuJoCo averages the two geoms' solref, so a default plane alone lets flat boxes sink
5-10 mm at impact.
Pass criteria:

- finite state throughout;
- at most ``REST_PEN_TOL`` below the plane at the end (no tunnelling, no sinking); the deepest point during the
  impact is reported as ``min_clearance`` (a 2.7 kg flat box sinks about 7 mm at 0.63 m/s under Alloy's gains);
- the centre of mass never rises more than ``RISE_TOL`` above its start (no launch or explosion; the body origin
  may rise when an object tips onto its side);
- the body Newton built has the card's mass within ``MASS_TOL`` (geometry far from the body origin loses float32
  precision in the hull mass and inertia: Objaverse meshes 18-750 m off their origin came out 1.6-1,900x heavy);
- at the end the centre of mass moves slower than ``MAX_END_SPEED`` and its height varies less than
  ``MAX_REST_BAND`` over the last ``REST_WINDOW`` seconds (a dropped object comes to rest or rolls slowly; a broken
  body keeps bouncing);
- ``settled`` (information, not a failure): over the last ``REST_WINDOW`` seconds the height varies less than
  ``REST_TOL``; balls roll and tall bottles may still be tipping over.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass

import numpy as np

log = logging.getLogger("alloy.importers")

REST_PEN_TOL = 5e-4          # m, depth below the plane allowed at rest (a body that tunnels ends far below)
RISE_TOL = 5e-3              # m
REST_TOL = 1e-3              # m
REST_WINDOW = 0.3            # s
MASS_TOL = 0.01              # relative, built body mass vs card mass
MAX_END_SPEED = 0.5          # m/s, centre-of-mass speed at the end (YCB/GSO: at most 0.04)
MAX_REST_BAND = 0.05         # m, centre-of-mass height range over the last REST_WINDOW (YCB/GSO: at most 0.7 mm)
SERIES_K = 1.0e6             # N/m, two default 2e6 N/m material stiffnesses in series (SPEC D22)


@dataclass(frozen=True)
class DropResult:
    """Outcome of one drop (JSON-ready via ``to_json``)."""

    ok: bool
    reason: str
    settled: bool            # height steady over the last REST_WINDOW (information only)
    mass: float              # kg, body mass Newton built
    min_clearance: float     # m, lowest hull point over the run (negative = below the plane)
    rest_clearance: float    # m, lowest hull point at the end
    max_rise: float          # m, highest centre-of-mass height above its start
    rest_band: float         # m, centre-of-mass height range over the last REST_WINDOW
    final_speed: float       # m/s, centre-of-mass speed at the end
    seconds: float

    def to_json(self) -> dict:
        """Plain dict for ``json.dumps``.

        Example::

            json.dumps(result.to_json())
        """
        return asdict(self)


@dataclass(frozen=True)
class DropSetup:
    """Body and contact numbers of one drop."""

    density: float           # kg/m^3 on every hull (mass / sum of hull volumes)
    mass: float              # kg
    mu: float
    restitution: float
    stiffness: float = SERIES_K
    height: float = 0.02     # m, lowest point above the plane at the start
    dt: float = 1.25e-3      # s, Alloy's rigid step (library.RIGID_DT)


def _build(hulls, s: DropSetup):
    import warp as wp  # noqa: PLC0415 - engine imports only when a drop is run

    from alloy._vendor import newton  # noqa: PLC0415
    from alloy.scene.native_gains import ZETA_CCD, native_gains  # noqa: PLC0415

    ke, kd = native_gains(s.stiffness, s.mass, s.restitution, dt=s.dt, zeta_min=ZETA_CCD)
    b = newton.ModelBuilder()
    b.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=s.mu, restitution=s.restitution, ke=ke, kd=kd))
    lowest = min(float(v[:, 2].min()) for v, _ in hulls)
    body = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, s.height - lowest), wp.quat_identity()), label="object")
    for k, (v, f) in enumerate(hulls):
        mesh = newton.Mesh(np.asarray(v, np.float32), np.asarray(f, np.int32).reshape(-1))
        cfg = newton.ModelBuilder.ShapeConfig(density=s.density, mu=s.mu, restitution=s.restitution, ke=ke, kd=kd,
                                              margin=0.0)
        b.add_shape_convex_hull(body, mesh=mesh, cfg=cfg, label=f"hull_{k}")
    return b.finalize(device="cpu"), np.concatenate([v for v, _ in hulls])


def _world(points: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Body-frame ``points`` in the world under transform ``q`` = (px, py, pz, qx, qy, qz, qw)."""
    x, y, z, w = q[3:7]
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return points @ r.T + q[:3]


def _lowest(points: np.ndarray, q: np.ndarray) -> float:
    """Lowest z of body-frame ``points`` under transform ``q``."""
    return float(_world(points, q)[:, 2].min())


def _verdict(com: np.ndarray, lows: np.ndarray, dt: float) -> tuple[bool, str, float, float, float]:
    """Judge a run from the centre-of-mass track (N, 3) and the lowest hull point per step."""
    if not (np.isfinite(com).all() and np.isfinite(lows).all()):
        return False, "non-finite state", math.nan, math.nan, math.nan
    rise = float(com[:, 2].max() - com[0, 2])
    tail = com[-max(2, int(REST_WINDOW / dt)):, 2]
    band = float(tail.max() - tail.min())
    speed = float(np.linalg.norm(com[-1] - com[-2]) / dt)
    if lows[-1] < -REST_PEN_TOL:
        return False, f"sinks {-lows[-1] * 1e3:.2f} mm into the plane at rest", rise, band, speed
    if rise > RISE_TOL:
        return False, f"centre of mass rose {rise * 1e3:.1f} mm above its start", rise, band, speed
    if speed > MAX_END_SPEED:
        return False, f"still moving at {speed:.2f} m/s at the end (does not come to rest)", rise, band, speed
    if band > MAX_REST_BAND:
        return False, (f"centre of mass still bouncing over {band * 1e3:.0f} mm in the last {REST_WINDOW:g} s"), \
            rise, band, speed
    return True, "ok", rise, band, speed


def mass_issue(built: float, mass: float) -> str | None:
    """Why the body Newton built does not have the card's mass, or None when it does (within ``MASS_TOL``).

    Example::

        mass_issue(1906.0 * 0.5, 0.5)   # "Newton built a 953 kg body for a 0.5 kg card ..."
    """
    if math.isfinite(built) and abs(built / mass - 1.0) <= MASS_TOL:
        return None
    return (f"Newton built a {built:.4g} kg body for a {mass:.4g} kg card (geometry far from the body origin loses "
            "float32 precision in the hull mass and inertia: centre the mesh on its origin)")


def drop_test(hulls: list[tuple[np.ndarray, np.ndarray]], mass: float, mu: float, restitution: float,
              duration: float = 2.5, **setup) -> DropResult:
    """Drop the object's hulls (body frame, metres) onto a plane and judge the outcome.

    Example::

        _, hulls = read_hulls_obj(Path("meshes/ycb_025_mug/physics.hulls.obj"))
        r = drop_test(hulls, mass=0.118, mu=0.25, restitution=0.5)
        r.ok, r.reason
    """
    from alloy._vendor import newton  # noqa: PLC0415

    t0 = time.monotonic()
    hull_vol = sum(abs(_volume(v, f)) for v, f in hulls)
    s = DropSetup(density=mass / hull_vol, mass=mass, mu=mu, restitution=restitution, **setup)
    dt = s.dt
    model, points = _build(hulls, s)
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=True, integrator="implicitfast")
    s0, s1, ctrl = model.state(), model.state(), model.control()
    steps = int(round(duration / dt))
    track = np.zeros((steps + 1, 7))
    track[0] = s0.body_q.numpy()[0]
    for i in range(steps):
        solver.step(s0, s1, ctrl, None, dt)
        s0, s1 = s1, s0
        track[i + 1] = s0.body_q.numpy()[0]
    finite = np.isfinite(track).all(axis=1)
    lows = np.array([_lowest(points, q) if ok_q else np.nan for q, ok_q in zip(track, finite)])
    c = np.asarray(model.body_com.numpy()[0], np.float64)[None, :]
    com = np.array([_world(c, q)[0] if ok_q else np.full(3, np.nan) for q, ok_q in zip(track, finite)])
    ok, reason, rise, band, speed = _verdict(com, lows, dt)
    built = float(model.body_mass.numpy()[0])
    bad_mass = mass_issue(built, mass)
    if bad_mass is not None:
        ok, reason = False, bad_mass
    settled = bool(np.isfinite(band) and band <= REST_TOL)
    return DropResult(ok, reason, settled, built, float(np.nanmin(lows)), float(lows[-1]), rise, band, speed,
                      round(time.monotonic() - t0, 2))


def _volume(v: np.ndarray, f: np.ndarray) -> float:
    tri = np.asarray(v, np.float64)[np.asarray(f)]
    return float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)
