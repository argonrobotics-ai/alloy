# SPDX-License-Identifier: Apache-2.0
"""``PressureBall``: an inflated ball's contacts with the pitch (turf in series) and with listed shapes.

The plug-in owns every rigid contact of the ball. Build the ball's sphere with
``ShapeConfig(has_shape_collision=False)`` (particles, e.g. a cloth net, still collide with it), and list the shapes
it meets in ``colliders``: boxes, spheres, capsules and cylinders, static or on bodies (a boot, posts, a wall). The
model is ``model.py``'s: pressure over the flattened patch plus the shell, the shell's dashpot, turf in series on the
pitch, a tangential spring capped at μ·F_n, and rolling resistance on the turf.

    ball = PressureBall(colliders=("boot", "post_l", "post_r", "bar"), pressure=0.8e5)
    ball.set_input("pressure", 0.6e5)          # per world (and ball): gauge Pa

Readings (state): ``touch`` [W, B, 1 + C] (duration s, peak N) of the current or last contact with the pitch
(index 0) and each collider; a negative peak marks a finished contact.
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
from alloy.plugins.sports import kernels as K
from alloy.plugins.sports import model as M

SUPPORTED = {K.SPHERE: "sphere", K.CAPSULE: "capsule", K.CYLINDER: "cylinder", K.BOX: "box"}
COLLIDE_SHAPES = 1 << 1               # newton ShapeFlags.COLLIDE_SHAPES


class PressureBall(ForcePlugin):
    """Inflated-ball contact: pressure-shell normal law, turf in series, friction, rolling resistance."""

    name = "pressure_ball"
    summary = "inflated ball (football) contact: pressure x patch area + shell, turf in series, friction, rolling"

    @dataclass
    class Config:
        bodies: tuple[str, ...] = ("ball",)
        radius: float = M.R_BALL                    # m (IFAB size 5)
        pressure: float = M.P_TEST                  # Pa gauge, every ball and world (input "pressure")
        shell_k: float = M.SHELL_K                  # N/m, the bare shell (ESTIMATE)
        shell_c: float = M.SHELL_C                  # N·s/m, the shell's loss (fitted to FIFA's 1.45 m rebound)
        tangential_k: float = 2.0e4                 # N/m, tangential contact stiffness (ESTIMATE)
        tangential_zeta: float = 1.0                # damping ratio of the tangential spring (ESTIMATE)
        colliders: tuple[str, ...] = ()             # shape labels the ball meets
        collider_mu: tuple[float, ...] = ()         # friction per collider (default 0.5 each, ESTIMATE)
        pitch: bool = True                          # the turf half-space z <= pitch_z
        pitch_z: float = 0.0
        turf_k: float = M.TURF_K                    # N/m^1.5 (ESTIMATE)
        turf_c: float = M.TURF_C                    # N·s/m (fitted to a 0.80 m rebound)
        turf_mu: float = M.TURF_MU
        turf_roll: float = M.TURF_ROLL              # rolling resistance C_rr (fitted to a 7.0 m FIFA ball roll)

        def __post_init__(self) -> None:
            if not self.bodies or self.radius <= 0.0 or min(self.pressure, self.shell_k, self.shell_c) < 0.0:
                raise ValueError("at least one ball body, radius > 0, and pressure, shell_k, shell_c >= 0")
            if self.collider_mu and len(self.collider_mu) != len(self.colliders):
                raise ValueError(f"collider_mu needs one value per collider ({len(self.colliders)}), "
                                 f"got {len(self.collider_mu)}")
            if self.pitch and (self.turf_k <= 0.0 or self.turf_c <= 0.0):
                raise ValueError("the pitch needs turf_k > 0 and turf_c > 0")
            if self.tangential_k <= 0.0:
                raise ValueError("tangential_k must be > 0")

    # --- setup ---------------------------------------------------------------------------------------------------

    def setup(self, ctx: Any) -> None:
        c, model, W = self.config, ctx.model, ctx.world_count
        ids = np.stack([ctx.bodies(b).numpy() for b in c.bodies], axis=1).astype(np.int32)
        self._check_ball_shapes(model, ids[0])
        cols = np.zeros((W, max(len(c.colliders), 1)), np.int32)
        types = model.shape_type.numpy()
        for j, label in enumerate(c.colliders):
            cols[:, j] = ctx.shapes(label).numpy()
            kind = int(types[cols[0, j]])
            if kind not in SUPPORTED:
                raise PluginError(f"{self.label}: collider {label!r} is shape type {kind}; supported: "
                                  f"{sorted(SUPPORTED.values())}")
        self.n_cols = len(c.colliders)
        self.balls = ctx.array(ids, wp.int32)
        self.cols = ctx.array(cols, wp.int32)
        mus = list(c.collider_mu) or [0.5] * self.n_cols
        self.col_mu = ctx.array(np.array(mus or [0.5], np.float32), wp.float32)
        self.ball = self._ball_params(model, int(ids[0, 0]))
        self.turf = self._pitch_params()
        nb = len(c.bodies)
        self.pressure = self.add_input("pressure", ctx.full((W, nb), float(c.pressure)))
        self.turf_u = self.add_state("turf_u", ctx.zeros((W, nb)))
        self.slip = self.add_state("slip", ctx.zeros((W, nb, 1 + self.n_cols), wp.vec3))
        self.touch = self.add_state("touch", ctx.zeros((W, nb, 1 + self.n_cols), wp.vec2))

    def _check_ball_shapes(self, model: newton.Model, bodies: np.ndarray) -> None:
        sb, flags = model.shape_body.numpy(), model.shape_flags.numpy()
        types, scale = model.shape_type.numpy(), model.shape_scale.numpy()
        for name, b in zip(self.config.bodies, bodies):
            mine = np.flatnonzero(sb == b)
            if np.any(flags[mine] & COLLIDE_SHAPES):
                raise PluginError(f"{self.label}: ball {name!r} has shapes that collide with shapes in the solver, "
                                  "but PressureBall owns its contacts: build the ball's sphere with "
                                  "ShapeConfig(has_shape_collision=False)")
            spheres = [s for s in mine if int(types[s]) == K.SPHERE]
            if spheres and abs(float(scale[spheres[0], 0]) - self.config.radius) > 1e-4:
                raise PluginError(f"{self.label}: ball {name!r} sphere radius {float(scale[spheres[0], 0]):.4f} m "
                                  f"differs from radius {self.config.radius:.4f} m: set radius to the sphere's")

    def _ball_params(self, model: newton.Model, b: int) -> K.Ball:
        c = self.config
        m = float(model.body_mass.numpy()[b])
        i_min = float(np.linalg.eigvalsh(model.body_inertia.numpy()[b]).min())
        m_eff = 1.0 / (1.0 / m + c.radius ** 2 / max(i_min, 1e-12))      # tangential effective mass
        p = K.Ball()
        p.radius, p.shell_k, p.shell_c = c.radius, c.shell_k, c.shell_c
        p.k_tan = c.tangential_k
        p.c_tan = 2.0 * c.tangential_zeta * math.sqrt(c.tangential_k * m_eff)
        self._mass, self._m_eff = m, m_eff
        return p

    def _pitch_params(self) -> K.Pitch:
        c = self.config
        t = K.Pitch()
        t.on, t.z, t.k, t.c = int(c.pitch), c.pitch_z, c.turf_k, c.turf_c
        t.mu, t.roll = c.turf_mu, c.turf_roll
        return t

    def check_dt(self, dt: float) -> None:
        """Explicit contact forces: ω·dt <= 0.5 for the normal law at the overlap of a 30 m/s impact (a hard shot)
        and for the tangential spring."""
        c = self.config
        p = float(np.max(self.pressure.numpy())) if hasattr(self, "pressure") else c.pressure
        k = M.shell_stiffness(impact_overlap(30.0, self._mass, p, c), p, c.radius, c.shell_k)
        tan = math.sqrt(c.tangential_k / self._m_eff)
        rates = {"normal": math.sqrt(k / self._mass) * dt, "tangential": tan * dt,
                 "tangential damping": 2.0 * c.tangential_zeta * tan * dt}
        bad = {k: v for k, v in rates.items() if v > 0.5}
        if bad:
            raise PluginError(f"{self.label}: dt {dt:.3g} s is too long for the ball's contact: "
                              + ", ".join(f"{k} ω·dt {v:.2f}" for k, v in bad.items())
                              + " > 0.5: use a step of 0.5 ms or less (the ball's contact lasts ~8 ms)")

    # --- step ----------------------------------------------------------------------------------------------------

    def apply(self, state: newton.State, forces: Forces, dt: float) -> None:
        m = self.ctx.model
        wp.launch(K.ball_contacts, dim=self.ctx.world_count, device=self.ctx.device,
                  inputs=[state.body_q, state.body_qd, m.body_com, self.balls, self.pressure, self.cols, self.col_mu,
                          int(self.n_cols), m.shape_body, m.shape_transform, m.shape_type, m.shape_scale, self.ball,
                          self.turf, float(dt), self.turf_u, self.slip, self.touch],
                  outputs=[forces.body_f])

    # --- harness -------------------------------------------------------------------------------------------------

    def test_scene(self) -> Any:
        from alloy.plugins.sports.laws import default_scene  # noqa: PLC0415

        return default_scene()

    def laws(self) -> list:
        from alloy.plugins.sports.laws import ball_laws  # noqa: PLC0415

        return ball_laws()


def impact_overlap(speed: float, mass: float, p: float, c: PressureBall.Config) -> float:
    """Overlap [m] where the elastic energy ∫F dδ reaches ½·m·v² (no damping): the deepest a ``speed`` impact
    presses the ball."""
    d = np.linspace(0.0, c.radius, 2001)
    f = M.shell_force(d, p, c.radius, c.shell_k)
    work = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(d))])
    return float(np.interp(0.5 * mass * speed * speed, work, d))


__all__ = ["PressureBall", "impact_overlap"]
