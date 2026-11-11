# SPDX-License-Identifier: Apache-2.0
"""A15 uniaxial Neo-Hookean compression: P = mu (lambda - lambda^-2) incompressible, here in the compressible form
the configured (mu, lambda) imply (bench §4.5 A15: within 5 % at stretch 0.8).

Reference energy: stable Neo-Hookean as Newton's VBD implements it (newton@ac7f7830:newton/_src/solvers/vbd/
particle_vbd_kernels.py:228-262): lam' = lam + mu, alpha = 1 + mu / lam', P = mu F + lam' (J - alpha) cof F.
Uniaxial stress F = diag(a, b, b), lateral P = 0 gives J = alpha - mu / (lam' a), b^2 = J / a and the nominal
axial stress P(a) = mu (a - b^2 / a) (-> mu (a - a^-2) as lam -> inf).
Set-up: a squat pad (height a quarter of its width) rests on a frictionless plane in zero gravity; the top-face
nodes carry a dead load P(target) x nodal area, eased in over T_EASE. A free cube under equal and opposite dead loads
is unstable to rigid tilt (spike [internal note]); the plane's normal reactions restore a
squat pad. Metric: |P(stretch_sim) - P_applied| / |P_applied| <= 0.05, stretch from mean top/bottom node heights.
Surface triangles and edges of the soft grid get zero stiffness so only the tet energy acts; the particle radius is
set small because Newton's default radius makes XPBD/SemiImplicit grids self-collide and explode at rest.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from alloy._vendor import newton
import numpy as np
import warp as wp
from scipy.optimize import brentq

from alloy.bench.analytic.common import BaseTwin, smoothstep
from alloy.bench.solvers import TET_SOLVERS
from alloy.bench.twin import DeviceTwinLog
from alloy.bench.types import Condition, LawSpec, Tolerance

MU_L = 1.0e4  # [Pa] shear modulus (soft rubber)
LAM_L = 9.0e4  # [Pa] Lame lambda (nu = 0.45)
DENSITY = 1000.0
WIDTH, HEIGHT = 0.04, 0.01  # [m]
CELLS = (4, 4, 2)
RADIUS = 5e-4  # [m] particle radius (contact thickness on the floor)
T_EASE = 0.25  # [s]
FLOOR_KE, FLOOR_KD = 1.0e4, 1.0  # [N/m], [N s/m] floor penalty (see on_model)
STRETCHES = (0.8, 0.9)
_SRC = "bench §4.5 A15: P = mu(lambda - lambda^-2) within 5 % at lambda = 0.8 (compressible form, SCENARIOS D06)"


def _law(dt: float, substeps: int, law_id: str, solvers: tuple[str, ...]) -> LawSpec:
    return LawSpec(id=law_id, suite="analytic", metrics={"stress_rel_err": Tolerance(abs=0.05, source=_SRC)},
                   stat={"stress_rel_err": "mean"}, dt=dt, substeps=substeps, duration_s=1.0,
                   worlds_per_condition=1, solvers=solvers, quick={"worlds": 16, "duration_s": 0.5})


LAW = _law(1e-3, 1, "A15.neohookean", ("vbd", "xpbd"))
LAW_EXPLICIT = _law(1e-3, 10, "A15.neohookean_explicit", ("semi_implicit",))  # explicit FEM needs 1e-4 s substeps
assert set(LAW.solvers) | set(LAW_EXPLICIT.solvers) == set(TET_SOLVERS)


def uniaxial_stress(a: float, mu: float = MU_L, lam: float = LAM_L) -> float:
    """Nominal axial stress [Pa] of the stable Neo-Hookean solid at axial stretch a, lateral faces free."""
    lp = lam + mu
    j = 1.0 + mu / lp - mu / (lp * a)
    return mu * (a - (j / a) / a)


def stretch_for_stress(p: float) -> float:
    return float(brentq(lambda a: uniaxial_stress(a) - p, 0.3, 3.0))


@wp.kernel
def _face_means(z: wp.array(dtype=wp.vec3), idx: wp.array(dtype=wp.int32), start: wp.array(dtype=wp.int32),
                end: wp.array(dtype=wp.int32), col: int, out: wp.array2d(dtype=wp.float64)):
    w = wp.tid()
    acc = wp.float64(0.0)
    for k in range(start[w], end[w]):
        acc += wp.float64(z[idx[k]][2])
    out[w, col] = acc / wp.float64(wp.max(end[w] - start[w], 1))


@wp.kernel
def _top_load(idx: wp.array(dtype=wp.int32), force: wp.array(dtype=float), scale: float,
              particle_f: wp.array(dtype=wp.vec3)):
    k = wp.tid()
    particle_f[idx[k]] = wp.vec3(0.0, 0.0, force[k] * scale)


def _face_weights(x: np.ndarray, tris: np.ndarray, sel: np.ndarray) -> np.ndarray:
    """Lumped nodal areas of the surface triangles whose three vertices are all in `sel`."""
    w = np.zeros(len(x))
    for tri in tris:
        if sel[tri].all():
            p = x[tri]
            w[tri] += 0.5 * np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0])) / 3.0
    return w


class NeoHookeanTwin(BaseTwin):
    law_id = LAW.id
    units = {"stress_rel_err": "1"}

    def conditions(self, quick: bool) -> tuple[Condition, ...]:
        return tuple(Condition(f"stretch{s:g}", {"stretch": s, "p_applied": uniaxial_stress(s)}) for s in STRETCHES)

    def world_gravity(self, cond: Condition) -> tuple[float, float, float]:
        return (0.0, 0.0, 0.0)

    def prepare_builder(self, b: newton.ModelBuilder, solver: str) -> None:
        b.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.0, ke=FLOOR_KE, kd=FLOOR_KD, ka=0.0))

    def build(self, b: newton.ModelBuilder, cond: Condition, sample: Mapping[str, float]) -> None:
        nx, ny, nz = CELLS
        b.add_soft_grid(pos=wp.vec3(-WIDTH / 2, -WIDTH / 2, RADIUS), rot=wp.quat_identity(), vel=wp.vec3(0.0),
                        dim_x=nx, dim_y=ny, dim_z=nz, cell_x=WIDTH / nx, cell_y=WIDTH / ny, cell_z=HEIGHT / nz,
                        density=DENSITY, k_mu=MU_L, k_lambda=LAM_L, k_damp=0.0, tri_ke=0.0, tri_ka=0.0, tri_kd=0.0,
                        add_surface_mesh_edges=False, particle_radius=RADIUS)

    def on_model(self, model: newton.Model) -> None:
        model.particle_mu = 0.0  # frictionless floor: lateral expansion is free (uniaxial stress)
        model.soft_contact_mu = 0.0
        # Floor penalty, the same for every solver: explicit SemiImplicit (particle mass ~5e-4 kg, h = 1e-4 s) needs
        # sqrt(ke/m) h < 2 and kd h / m < 2; Newton's default damping launches the pad off the floor. SemiImplicit
        # averages particle_* and shape ke/kd/ka (semi_implicit/kernels_contact.py:207-212), VBD reads soft_contact_*.
        model.soft_contact_ke, model.soft_contact_kd = FLOOR_KE, FLOOR_KD
        model.particle_ke, model.particle_kd, model.particle_ka = FLOOR_KE, FLOOR_KD, 0.0
        # Particle-particle contact in XPBD/SemiImplicit is not world-aware: overlapping worlds at the origin (F1)
        # collide with each other (2 resting pads -> NaN, spike wp6-spikes/pad_rest.py). The pad needs no self-contact;
        # Newton documents this switch (newton@ac7f7830:newton/_src/solvers/semi_implicit/solver_semi_implicit.py:211).
        if self.solver_name == "semi_implicit":
            model.particle_grid = None  # XPBD asserts a grid exists (solver_xpbd.py:712)
        x = model.particle_q.numpy().astype(np.float64)
        pw = model.particle_world.numpy()
        tris = model.tri_indices.numpy()
        top_all, bot_all, starts, ends, forces = [], [], [], [], []
        for w in range(len(self.worlds)):
            ids = np.flatnonzero(pw == w)
            z = x[ids, 2]
            sel_top = np.zeros(len(x), bool)
            sel_top[ids[np.abs(z - z.max()) < 1e-7]] = True
            area = _face_weights(x, tris, sel_top)
            top = np.flatnonzero(sel_top)
            bot = ids[np.abs(z - z.min()) < 1e-7]
            self._h0 = float(z.max() - z.min())
            starts.append(len(top_all))
            top_all += top.tolist()
            ends.append(len(top_all))
            forces += (self.worlds[w].params["p_applied"] * area[top]).tolist()  # P < 0 pushes down
            bot_all.append(bot)
        self._dev = model.device
        self._top = wp.array(np.array(top_all, np.int32), dtype=wp.int32, device=self._dev)
        self._force = wp.array(np.array(forces, np.float32), dtype=float, device=self._dev)
        self._top_rng = (wp.array(np.array(starts, np.int32), device=self._dev),
                         wp.array(np.array(ends, np.int32), device=self._dev))
        b_idx = np.concatenate(bot_all).astype(np.int32)
        b_end = np.cumsum([len(b) for b in bot_all]).astype(np.int32)
        self._bot = wp.array(b_idx, dtype=wp.int32, device=self._dev)
        self._bot_rng = (wp.array(b_end - np.array([len(b) for b in bot_all], np.int32), device=self._dev),
                         wp.array(b_end, device=self._dev))

    def drive(self, step: int, t: float, state: Any, control: Any) -> None:
        wp.launch(_top_load, dim=self._top.shape[0], inputs=[self._top, self._force, smoothstep(t / T_EASE)],
                  outputs=[state.particle_f], device=self._dev)

    def record(self, step: int, state: Any, contacts: Any, obs: Any, log: DeviceTwinLog) -> None:
        col = log.column(step)
        if col < 0:
            return
        for name, idx, (s, e) in (("z_top", self._top, self._top_rng), ("z_bot", self._bot, self._bot_rng)):
            wp.launch(_face_means, dim=len(self.worlds), inputs=[state.particle_q, idx, s, e, col],
                      outputs=[log.channel(name, wp.float64)], device=self._dev)

    def scalars(self, log: DeviceTwinLog) -> dict[str, np.ndarray]:
        t = log.times()
        late = t >= t[-1] - 0.1 * (t[-1] - T_EASE)
        stretch = np.mean(log.series("z_top")[:, late] - log.series("z_bot")[:, late], axis=1) / self._h0
        p_app = self.param("p_applied")
        p_sim = np.array([uniaxial_stress(s) if np.isfinite(s) and s > 0 else math.nan for s in stretch])
        self.monitors.update({"stretch_sim": stretch.tolist(), "stretch_target": self.param("stretch").tolist()})
        err = np.abs(p_sim - p_app) / np.abs(p_app)
        return {"stress_rel_err": np.where(np.isfinite(err), err, math.inf)}


__all__ = ["LAW", "LAW_EXPLICIT", "NeoHookeanTwin", "uniaxial_stress", "stretch_for_stress"]
