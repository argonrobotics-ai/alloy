# SPDX-License-Identifier: Apache-2.0
"""SimTwins of the sealed-liquid laws (SCR-SC10-1, ambient.py): A13.sealed_column and A13.piston_displacement.

Scene (per world): a square bore of four box walls (``n`` spacings wide, open at the bottom, z = 0 the mouth) with
a box piston on a kinematic body, the bore filled with water up to the piston face (``nz`` lattice layers), csf at a
90 deg contact angle (a flat meniscus, so the gauge pressure under the piston is -rho g H) and xsph at csf's
XSPH_MIN; ``ambient_pressure`` 500 Pa. The bore is 14 spacings (2.1 mm) at r = 0.075 mm, C18's particle size:
R/h = 3.5, csf's law range, and Bond rho g R^2 / sigma = 0.15, so a real column of this bore holds (Rayleigh-Taylor
stable); a 1 mm bore at this r is below csf's range (R/h 1.5-1.7). Walls and piston are ``sph:shape_seal`` in the
sealed worlds, not in the control. Liquid that leaves ``world_bounds`` (2 cm below the mouth) retires.

- A13.sealed_column, sealed world (piston static, 90 deg): gauge pressure on the piston from its impulse meter over the
  second half, |p / (-rho g H) - 1| <= 0.1; drift of the liquid inside the bore (mean z; inside: above one spacing
  below the mouth, where the meniscus layer sits) after the lattice settles (SETTLE_S) <= 0.1 spacing; void: 1 -
  (liquid volume inside the bore) / (bore volume under the piston) at the end <= 0.01. Control world (no seal): the
  column must drain as today: ``control`` is 1 when less than one layer (n^2 particles) left the bore. Reported,
  not gated: a sealed world at C18's 100 deg against -rho g H + 2 sigma |cos theta| / a (a the half-width: the
  convex meniscus's Laplace pressure, 4 sigma |cos theta| / w for a square bore) in ``monitors``.
- A13.piston_displacement: the piston moves down at U = 10 mm/s from ``t0``; the outflow (liquid below the mouth
  plane z = 0) grows at A U: |slope / (A U) - 1| <= 0.01, slope fitted over the push after 0.1 s. Valid for full
  mode's push (47 layers after the fit start, measured +0.18 %); quick mode's 2 layers (7.1 % low) are outside it,
  and SB-12 counts the full-mode verdict (coordinator 2026-11-05 05:48Z). (Counted below
  -2r, as ``inside`` is, the drop growing under the mouth fills that slab during the push and read 1-4 % low.)
"""

from __future__ import annotations

import math
from typing import Any

from alloy._vendor import newton
import numpy as np
import warp as wp

from alloy.bench.types import Condition

from . import bodies as bd
from . import probes as pb
from .build import add_fluid_points, wall_box
from .config import DFSPHConfig, particle_mass
from .csf import XSPH_MIN
from .twins import SPHTwin

RHO, SIGMA, G = 1000.0, 0.072, 9.81
P_AIR = 500.0  # [Pa] model ambient: ~3x the 16.5 mm column's rho g H (161 Pa)
U_PUSH = 0.01  # [m/s] piston speed of A13.piston_displacement
CHANNELS = ("jz", "inside", "zmean", "out", "below")


def add_sealed_bore(b: newton.ModelBuilder, cfg: DFSPHConfig, n: int, nz: int, seal: bool,
                    contact_angle: float | None = 0.5 * math.pi, sigma: float = SIGMA) -> dict[str, Any]:
    """Bore walls, kinematic piston (face at z = nz spacings) and the water column of one world (surface tension
    ``sigma``; walls at ``contact_angle``, None: free contact line); returns its ids."""
    s, t = cfg.spacing, cfg.h
    w, H = n * s, nz * s
    L = H + 20 * s
    for c, half in (((-0.5 * w - 0.5 * t, 0.0, 0.5 * L), (0.5 * t, 0.5 * w + t, 0.5 * L)),
                    ((0.5 * w + 0.5 * t, 0.0, 0.5 * L), (0.5 * t, 0.5 * w + t, 0.5 * L)),
                    ((0.0, -0.5 * w - 0.5 * t, 0.5 * L), (0.5 * w, 0.5 * t, 0.5 * L)),
                    ((0.0, 0.5 * w + 0.5 * t, 0.5 * L), (0.5 * w, 0.5 * t, 0.5 * L))):
        wall_box(b, c, half, contact_angle=contact_angle, seal=seal)
    body = b.add_body(xform=wp.transform((0.0, 0.0, H), wp.quat_identity()), label="piston", is_kinematic=True)
    shape = wall_box(b, (0.0, 0.0, 0.5 * t), (0.5 * w, 0.5 * w, 0.5 * t), body=body, contact_angle=contact_angle,
                     seal=seal)
    g = (np.arange(n) + 0.5) * s - 0.5 * w
    pts = np.array([(x, y, (k + 0.5) * s) for k in range(nz) for y in g for x in g])
    first = b.particle_count
    add_fluid_points(b, cfg, pts, RHO, 1.0e-3, surface_tension=sigma)
    return {"body": body, "shape": shape, "H": H, "A": w * w, "first": first, "count": len(pts)}


class SealedBoreTwin(SPHTwin):
    """Shared scene, piston drive and recording of the sealed-liquid laws (module doc)."""

    config_overrides = {"surface_tension": "csf", "viscosity": "xsph", "xsph_c": XSPH_MIN,
                        "ambient_pressure": P_AIR, "integrate_bodies": "none", "max_density_error": 1.0e-4,
                        "world_bounds": ((-0.01, -0.01, -0.02), (0.01, 0.01, 0.03))}  # drained liquid retires
    CASES: tuple[tuple[str, bool, float, float], ...] = ()  # (tag, seal, piston speed, wall contact angle [deg])

    def __init__(self, dt: float) -> None:
        super().__init__(dt)
        self.ids: list[dict[str, Any]] = []

    def conditions(self, quick: bool) -> tuple[Condition, ...]:
        nz, t0 = (20, 0.01) if quick else (110, 0.2)
        return tuple(Condition(f"bore14_{tag}_{nz}layers_r75um", {"r": 0.075e-3, "n": 14, "nz": nz, "seal": int(seal),
                                                                 "U": u, "t0": t0, "theta": th})
                     for tag, seal, u, th in self.CASES)

    def build(self, b: newton.ModelBuilder, cond: Condition, cfg: DFSPHConfig) -> None:
        p = cond.params
        self.ids.append(add_sealed_bore(b, cfg, int(p["n"]), int(p["nz"]), bool(p["seal"]),
                                        contact_angle=math.radians(float(p["theta"]))))

    def on_model(self, model: newton.Model) -> None:
        W, d = len(self.worlds), model.device
        self._vals = {c: wp.zeros(W, dtype=wp.float64, device=d) for c in CHANNELS if c != "jz"}
        self._pis = wp.array([i["shape"] for i in self.ids], dtype=wp.int32, device=d)
        self._bodies = np.array([i["body"] for i in self.ids], dtype=np.int64)
        self._vp = particle_mass(self.config, 1.0)  # particle volume [m^3]

    def drive(self, step: int, t: float, state: Any, control: Any) -> None:
        if not np.any(self.param("U") > 0.0):
            return
        z = self.param("nz") * 2.0 * self.param("r") - self.param("U") * np.maximum(t - self.param("t0"), 0.0)
        u = np.where(t >= self.param("t0"), self.param("U"), 0.0)
        bq, bqd = state.body_q.numpy(), state.body_qd.numpy()
        bq[self._bodies, 2] = z
        bqd[self._bodies] = 0.0
        bqd[self._bodies, 2] = -u
        state.body_q.assign(bq)
        state.body_qd.assign(bqd)

    def record(self, step: int, state: Any, contacts: Any, obs: Any, log_: Any) -> None:
        col = log_.column(step)
        if col < 0:
            return
        W = len(self.worlds)
        self._launch(pb.store_meter_kernel, W, [self.solver._meter_j, self._pis, 2, bd.IMP_SCALE_HOST, col,
                                                log_.channel("jz")])
        q, act = state.particle_q.numpy(), (self.model.particle_flags.numpy() & 1) != 0
        inside, zmean, out, below = np.zeros(W), np.zeros(W), np.zeros(W), np.zeros(W)
        for w, ids in enumerate(self.ids):
            sl = slice(ids["first"], ids["first"] + ids["count"])
            z, a = q[sl, 2], act[sl]
            ins = a & (z > -2.0 * float(self.config.particle_radius))  # the meniscus layer sits at the mouth
            inside[w], zmean[w] = ins.sum(), z[ins].mean() if ins.any() else 0.0
            out[w] = ids["count"] - ins.sum()
            below[w] = ids["count"] - (a & (z >= 0.0)).sum()  # below the mouth plane (or retired)
        for name, v in (("inside", inside), ("zmean", zmean), ("out", out), ("below", below)):
            self._vals[name].assign(v)
            self._launch(pb.store_column_kernel, W, [self._vals[name], col, log_.channel(name)])

    def _series(self, log_: Any) -> dict[str, np.ndarray]:
        return {c: log_.series(c).astype(np.float64) for c in CHANNELS}


class SealedColumnTwin(SealedBoreTwin):
    law_id = "A13.sealed_column"
    units = {"pressure": "1", "drift": "spacing", "void": "1", "control": "1"}
    CASES = (("sealed", True, 0.0, 90.0), ("open", False, 0.0, 90.0), ("sealed100deg", True, 0.0, 100.0))
    SETTLE_S = 0.1  # [s] the lattice column settles (first steps) before the drift is measured; quick: 1/4 of the run

    def scalars(self, log_: Any) -> dict[str, np.ndarray]:
        t, c = log_.times(), self._series(log_)
        i0 = int(np.searchsorted(t, 0.5 * t[-1]))
        i1 = int(np.searchsorted(t, min(self.SETTLE_S, 0.25 * t[-1])))
        s = 2.0 * self.param("r")
        A, H = np.array([i["A"] for i in self.ids]), np.array([i["H"] for i in self.ids])
        jz = c["jz"]
        p = (jz[:, -1] - jz[:, i0]) / (t[-1] - t[i0]) / A
        # 90 deg: -rho g H (flat meniscus); otherwise + 2 sigma |cos theta| / a, a the half-width (square bore)
        th = np.radians(self.param("theta"))
        ref = -RHO * G * H + 2.0 * SIGMA * np.abs(np.cos(th)) / (0.5 * np.sqrt(A))
        drift = (c["zmean"][:, -1] - c["zmean"][:, i1]) / s
        void = 1.0 - c["inside"][:, -1] * self._vp / (A * H)
        sealed, gated = self.param("seal") > 0, np.abs(th - 0.5 * np.pi) < 1e-9
        self.monitors.update(piston_gauge_Pa=p.tolist(), reference_Pa=ref.tolist(), rho_g_H_Pa=(RHO * G * H).tolist(),
                             pressure_ratio=(p / ref).tolist(), drift_spacings=drift.tolist(), void=void.tolist(),
                             out_particles=c["out"][:, -1].tolist(), drift_from_s=float(t[i1]),
                             cavitating=self.solver.stats.diagnostics()["cavitating"].tolist())
        g = sealed & gated  # the 90 deg sealed world is gated; 100 deg (C18's polypropylene) is reported (monitors)
        return {"pressure": np.abs(p / ref - 1.0)[g], "drift": np.abs(drift)[g],
                "void": np.maximum(void, 0.0)[g], "control": (c["out"][:, -1] < self.param("n") ** 2)[~sealed].astype(np.float64)}


class PistonDisplacementTwin(SealedBoreTwin):
    law_id = "A13.piston_displacement"
    units = {"outflow": "1"}
    CASES = (("push", True, U_PUSH, 90.0),)
    FIT_AFTER_S = 0.1  # [s] after the push starts; quick mode fits over its last 3/4

    def scalars(self, log_: Any) -> dict[str, np.ndarray]:
        t, c = log_.times(), self._series(log_)
        t0 = self.param("t0")[0]
        start = t0 + min(self.FIT_AFTER_S, 0.25 * (t[-1] - t0))
        m = t >= start
        A = np.array([i["A"] for i in self.ids])
        slope = np.array([np.polyfit(t[m], c["below"][w, m] * self._vp, 1)[0] for w in range(len(self.ids))])
        self.monitors.update(outflow_rate_m3_s=slope.tolist(), swept_rate_m3_s=(A * self.param("U")).tolist(),
                             fit_from_s=float(start))
        return {"outflow": np.abs(slope / (A * self.param("U")) - 1.0)}


__all__ = ["P_AIR", "PistonDisplacementTwin", "SealedBoreTwin", "SealedColumnTwin", "add_sealed_bore"]
