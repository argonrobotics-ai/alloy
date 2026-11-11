# SPDX-License-Identifier: Apache-2.0
"""``Magnetism``: applied magnetostatic forces and torques on tagged bodies (Gym-Anything coverage gap, 2026-11-06).

What it models, all as point dipoles (``field.py`` holds the formulas and the closed forms):
- **Body magnets.** Permanent dipoles on bodies (stir bars, eraser magnets, fridge magnets) push, pull and twist
  each other, with forces and torques in equal and opposite pairs.
- **Driven sources.** A stirrer's motor magnet is a dipole turning at ``rpm`` (an input per world and drive) about
  its axis, at a fixed world point or on a support body that takes the reaction.
- **Ferromagnetic plates.** A whiteboard or steel sheet is a static half-space of relative permeability μr. A
  magnet in front of it is pulled in by its image dipole, m' = κ(2(m·n)n - m) with κ = (μr - 1)/(μr + 1).
- **A uniform field**, which torques every magnet.

The library ``magnet`` plug-in is the body-to-body part alone. Use this one when a scene needs drives or plates.
The point-dipole model holds at distances of about one magnet length or more. Closer, forces are evaluated at
``min_distance`` and are an ESTIMATE; spread a flat magnet over several dipoles (``offsets``) when it sits on a
plate. Saturation of thin sheets and eddy currents are not modelled.

Example (a magnetic stirrer, a 25 mm bar 20 mm above the motor magnet)::

    mag = Magnetism(bodies=("stir_bar",), moments=(0.0675,), axes=((1, 0, 0),),
                    drives=((0.0, 0.0, -0.02),), drive_moments=(0.375,), drive_rpm=(300.0,))
    sim = alloy.plugins.simulation(built, [mag]); sim.plugins["magnetism"].set_input("rpm", 600.0)
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
from alloy.plugins.magnetism import field as F
from alloy.plugins.magnetism import kernels as K
from alloy.plugins.npgeom import quat_rotate, transform_point

Vec = tuple[float, float, float]


def _unit(v: Any, name: str) -> np.ndarray:
    a = np.asarray(v, float)
    n = float(np.linalg.norm(a))
    if a.shape != (3,) or n == 0.0:
        raise ValueError(f"{name}: a nonzero (x, y, z), got {v}")
    return a / n


def _each(values: tuple, n: int, name: str, default: Any) -> list:
    if not values:
        return [default] * n
    if len(values) not in (1, n):
        raise ValueError(f"{name}: one entry, or one per item ({n}), got {len(values)}")
    return [values[min(i, len(values) - 1)] for i in range(n)]


def _perp(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors completing ``axis`` to a right-handed frame (e1 ⊥ axis, e2 = axis × e1)."""
    seed = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = seed - np.dot(seed, axis) * axis
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(axis, e1)


class Magnetism(ForcePlugin):
    """Point-dipole magnetism: body magnets, driven rotating sources, ferromagnetic plates (image dipoles) and a
    uniform field. See the module docstring."""

    name = "magnetism"
    summary = "magnets on bodies, rotating drive magnets (stirrers), iron plates by image dipoles, uniform field"

    @dataclass
    class Config:
        bodies: tuple[str, ...] = ("stir_bar",)          # body label of each dipole (a body may carry several)
        moments: tuple[float, ...] = (0.0675,)            # A·m² per dipole (N35 25 mm stir bar ≈ 0.07, ESTIMATE)
        axes: tuple[Vec, ...] = ()                        # body-frame north per dipole; default +x
        offsets: tuple[Vec, ...] = ()                     # body-frame position per dipole; default origin
        drives: tuple[Vec, ...] = ((0.0, 0.0, -0.02),)    # drive magnet positions (world, or support-body frame)
        drive_moments: tuple[float, ...] = (0.375,)       # A·m² per drive (one for all; N35 motor magnet, ESTIMATE)
        drive_axes: tuple[Vec, ...] = ()                  # spin axis per drive; default +z
        drive_bodies: tuple[str, ...] = ()                # support body per drive; "" = the world
        drive_rpm: tuple[float, ...] = (300.0,)           # start speed per drive; input "rpm" changes it
        plates: tuple[tuple[float, float, float, float, float, float], ...] = ()   # (point, outward normal), world
        plate_half: tuple[tuple[float, float], ...] = ()  # half extents in the plate (m); 0 = unbounded
        plate_mu_r: tuple[float, ...] = (1.0e4,)          # relative permeability per plate (steel ≈ 1e3-1e4)
        min_distance: float = 2.0e-3                      # m; closer dipoles are evaluated at this distance
        external_field: Vec = (0.0, 0.0, 0.0)             # uniform B [T]

        def __post_init__(self) -> None:
            n, nd, npl = len(self.bodies), len(self.drives), len(self.plates)
            if n < 1 or len(self.moments) not in (1, n):
                raise ValueError(f"bodies: at least one, and one moment per body ({n}) or one for all")
            for name in ("axes", "offsets"):
                _each(getattr(self, name), n, name, None)
            for name in ("drive_moments", "drive_axes", "drive_bodies", "drive_rpm"):
                _each(getattr(self, name), nd, name, None)
            if nd and not self.drive_moments:
                raise ValueError("drive_moments: give the drive magnets' moments (A·m²)")
            for name in ("plate_half", "plate_mu_r"):
                _each(getattr(self, name), npl, name, None)
            if any(m <= 0 for m in (*self.moments, *self.drive_moments)) or any(u < 1 for u in self.plate_mu_r):
                raise ValueError("moments must be > 0 A·m² and plate_mu_r >= 1")
            if self.min_distance <= 0.0:
                raise ValueError("min_distance must be > 0 m (about the magnets' size)")

    @property
    def internal(self) -> bool:       # drives, plates and a uniform field act from outside the bodies
        c = self.config
        return not (c.drives or c.plates or any(c.external_field))

    # --- setup -------------------------------------------------------------------------------------------------

    def setup(self, ctx: Any) -> None:
        c, W = self.config, ctx.world_count
        n, nd = len(c.bodies), len(c.drives)
        self.bodies = ctx.array(np.stack([ctx.bodies(b).numpy() for b in c.bodies], axis=1).astype(np.int32), wp.int32)
        self.axes = ctx.array(np.array([_unit(a or (1.0, 0.0, 0.0), "axes") for a in _each(c.axes, n, "axes", None)],
                                       np.float32), wp.vec3)
        self.offsets = ctx.array(np.array(_each(c.offsets, n, "offsets", (0.0, 0.0, 0.0)), np.float32), wp.vec3)
        self.moments = ctx.array(np.array(_each(c.moments, n, "moments", None), np.float32), wp.float32)
        self.n_drives = nd
        self.drv = self._drives(ctx)
        self.rpm = self.add_input("rpm", ctx.zeros((W, max(nd, 1))))
        self.phase = self.add_state("phase", ctx.zeros((W, max(nd, 1))))
        if nd:
            self.rpm.assign(np.tile(np.array(_each(c.drive_rpm, nd, "drive_rpm", 0.0), np.float32), (W, 1)))
        self.n_plates = len(c.plates)
        self.pl = self._plates(ctx)

    def _drives(self, ctx: Any) -> K.Drives:
        c, W, nd = self.config, ctx.world_count, len(self.config.drives)
        dr = K.Drives()
        ids = np.full((W, max(nd, 1)), -1, np.int32)
        e1s, e2s = [], []
        for k, (label, axis) in enumerate(zip(_each(c.drive_bodies, nd, "drive_bodies", ""),
                                              _each(c.drive_axes, nd, "drive_axes", (0.0, 0.0, 1.0)))):
            if label:
                ids[:, k] = ctx.bodies(label).numpy()
            e1, e2 = _perp(_unit(axis, "drive_axes"))
            e1s.append(e1), e2s.append(e2)
        pad = lambda rows: np.array(rows or [(0.0, 0.0, 0.0)], np.float32)  # noqa: E731
        dr.body = ctx.array(ids, wp.int32)
        dr.pos = ctx.array(pad([tuple(p) for p in c.drives]), wp.vec3)
        dr.e1, dr.e2 = ctx.array(pad(e1s), wp.vec3), ctx.array(pad(e2s), wp.vec3)
        dr.moment = ctx.array(np.array(_each(c.drive_moments, nd, "drive_moments", 0.0) or [0.0], np.float32),
                              wp.float32)
        return dr

    def _plates(self, ctx: Any) -> K.Plates:
        c, npl = self.config, len(self.config.plates)
        rows = [(np.asarray(p[:3], float), _unit(p[3:], "plates normal")) for p in c.plates] or \
            [(np.zeros(3), np.array([0.0, 0.0, 1.0]))]
        pl = K.Plates()
        f32 = lambda a: np.array(a, np.float32)  # noqa: E731
        pl.point = ctx.array(f32([p for p, _ in rows]), wp.vec3)
        pl.normal = ctx.array(f32([n for _, n in rows]), wp.vec3)
        uv = [_perp(n) for _, n in rows]
        pl.u, pl.v = ctx.array(f32([u for u, _ in uv]), wp.vec3), ctx.array(f32([v for _, v in uv]), wp.vec3)
        pl.half = ctx.array(f32(_each(c.plate_half, npl, "plate_half", (0.0, 0.0)) or [(0.0, 0.0)]), wp.vec2)
        pl.kappa = ctx.array(f32([F.kappa(u) for u in _each(c.plate_mu_r, npl, "plate_mu_r", 1e4)] or [0.0]),
                             wp.float32)
        return pl

    def check_dt(self, dt: float) -> None:
        """Explicit forces: at the start poses, each magnet's torsional rate sqrt(m·B/I_min) times dt must stay < 1."""
        c, model = self.config, self.ctx.model
        q0, ids = model.body_q.numpy().astype(np.float64), self.bodies.numpy()[0]
        inertia, axes = model.body_inertia.numpy(), self.axes.numpy().astype(np.float64)
        offs, mom = self.offsets.numpy().astype(np.float64), self.moments.numpy().astype(np.float64)
        sources = [(transform_point(q0[b], offs[i]), quat_rotate(q0[b, 3:7], axes[i]) * mom[i], b)
                   for i, b in enumerate(ids)]
        for k, p in enumerate(c.drives):
            sources.append((np.asarray(p, float), np.array([1.0, 0, 0]) * self.drv.moment.numpy()[k], -2))
        for i, (p, m, b) in enumerate(sources[:len(ids)]):
            bmax = float(np.linalg.norm(c.external_field))
            for ps, ms, bs in sources:
                if bs != b:
                    bmax += 2e-7 * np.linalg.norm(ms) / max(np.linalg.norm(p - ps), c.min_distance) ** 3
            imin = max(float(np.linalg.eigvalsh(inertia[b]).min()), 1e-12)
            rate = math.sqrt(np.linalg.norm(m) * bmax / imin) * dt
            if rate > 1.0:
                raise PluginError(f"{self.label}: magnet on {c.bodies[i]!r} turns at sqrt(mB/I)·dt = {rate:.2f} > 1 "
                                  f"(B {bmax:.3g} T at the start): reduce dt, the moment, or move the magnets apart")

    # --- step --------------------------------------------------------------------------------------------------

    def apply(self, state: newton.State, forces: Forces, dt: float) -> None:
        c = self.config
        wp.launch(K.magnetism, dim=self.ctx.world_count, device=self.ctx.device,
                  inputs=[state.body_q, self.ctx.model.body_com, self.bodies, self.offsets, self.axes, self.moments,
                          self.drv, self.n_drives, self.phase, self.pl, self.n_plates, float(c.min_distance),
                          wp.vec3(*c.external_field)],
                  outputs=[forces.body_f])

    def update(self, state_in: newton.State, state_out: newton.State, dt: float) -> None:
        if self.n_drives:
            wp.launch(K.advance_phase, dim=self.ctx.world_count, device=self.ctx.device,
                      inputs=[self.rpm, self.n_drives, float(dt)], outputs=[self.phase])

    # --- host helpers ------------------------------------------------------------------------------------------

    def dipoles(self, body_q: np.ndarray, world: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
        """(position, moment) of each body dipole of ``world`` at poses ``body_q [B, 7]`` (float64)."""
        ids = self.bodies.numpy()[world]
        axes, offs = self.axes.numpy().astype(np.float64), self.offsets.numpy().astype(np.float64)
        return [(transform_point(body_q[b], offs[i]), quat_rotate(body_q[b, 3:7], axes[i]) * self.config.moments[
            min(i, len(self.config.moments) - 1)]) for i, b in enumerate(ids)]

    def energy(self, body_q: np.ndarray) -> np.ndarray:
        """``[W]`` body-dipole interaction energy [J] at poses ``body_q`` (pairs on different bodies, d >= r_min)."""
        ids, out = self.bodies.numpy(), np.zeros(self.ctx.world_count)
        for w in range(len(out)):
            dip = self.dipoles(body_q, w)
            for i in range(len(dip)):
                for j in range(i + 1, len(dip)):
                    if ids[w, i] != ids[w, j]:
                        out[w] += F.energy(*dip[i], *dip[j])
        return out

    def test_scene(self):
        from alloy.plugins.magnetism.laws import default_scene  # noqa: PLC0415 - laws import this module

        return default_scene()

    def laws(self):
        from alloy.plugins.magnetism.laws import magnetism_laws  # noqa: PLC0415

        return magnetism_laws()


__all__ = ["Magnetism"]
