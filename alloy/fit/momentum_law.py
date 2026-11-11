# SPDX-License-Identifier: Apache-2.0
"""VBD linear momentum law (coordinator ruling SC8 ~23:30Z; WP11 until the catalogue entry is ruled, SCR-WP11-2).

Newton's second law for a free body: with zero gravity and no contacts, the total momentum of a cloth changes by the
total external impulse, P(t) = F·t, whatever its internal forces do, so the centre-of-mass acceleration is F/M. The
load is constant but non-uniform (applied to a subset of the vertices through `state.particle_f`), so a solver whose
internal forces do not cancel, or whose per-step solve leaves a net residual force, shows it directly.

Conditions (one sheet per world, at the origin so float32 positions resolve the per-step displacements):
- `free`: the same mesh with no elastic energy (tri_ke = edge_ke = 0): the harness control (mass, force applied once
  per step, no damping);
- `flat_half`, `flat_row`: a flat 10 cm grid, the force on half the sheet / on one edge row;
- `cylinder_bottom`, `cap_bottom`: a half cylinder (r 3 cm) and a spherical cap (r 4 cm, polar angle 1.2 rad), the
  force on their lowest third.
Material as SC8's repro: 0.0143 kg/m² (15 µm film), membrane 40 N/m (k·dt²/m ≫ 1, the stiff-membrane regime),
bending 1e-4 N on the flat sheet, no damping. Metric per group: |P(T) − F·T·ẑ| / (F·T) ≤ 1 % at T = duration.
Measured on the stock SolverVBD ([internal note]): free 1.0000, flat half 1.07, edge row 0.39,
cylinder and cap 2-3x; the error is the unsolved per-vertex residual of the Gauss-Seidel sweep (its sum is the
slowest-converging mode), not the element forces (they sum to zero to 1e-5 F). docs/notes/wp11-vbd-momentum.md.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from alloy._vendor import newton
import numpy as np
import warp as wp

from alloy.bench.analytic.common import BaseTwin
from alloy.bench.solvers import SolverSettings
from alloy.bench.twin import DeviceTwinLog
from alloy.bench.types import Condition, LawSpec, Tolerance
from alloy.solvers.vbd import COARSE_KEY

FORCE = 1.0e-3                 # [N] total external force, +z
AREAL_DENSITY = 0.0143         # [kg/m²]
MEMBRANE_KE, BEND_KE = 40.0, 1.0e-4
_SRC = "Newton's second law: dP/dt = sum of external forces (no gravity, no contacts); 1 % (coordinator ruling SC8)"
GROUPS = {"free": "free_err", "flat_half": "flat_err", "flat_row": "flat_err", "cylinder_bottom": "curved_err",
          "cap_bottom": "curved_err"}

LAW = LawSpec(
    id="A17.vbd_momentum", suite="analytic",
    metrics={m: Tolerance(abs=0.01, source=_SRC) for m in ("free_err", "flat_err", "curved_err")},
    stat={m: "mean" for m in ("free_err", "flat_err", "curved_err")},
    dt=1.0e-3, substeps=1, duration_s=0.3, worlds_per_condition=1, solvers=("vbd",), producer="WP11",
    quick={"worlds": 5, "duration_s": 0.1},
)


# --- meshes ------------------------------------------------------------------------------------------------------

def grid(n: int = 20, cell: float = 5e-3) -> tuple[np.ndarray, np.ndarray]:
    """Square sheet, alternating diagonals, in the z = 0 plane, corner at the origin."""
    xs = np.arange(n + 1) * cell
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    v = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    tris = []
    for i in range(n):
        for j in range(n):
            a, b, c, d = i * (n + 1) + j, (i + 1) * (n + 1) + j, (i + 1) * (n + 1) + j + 1, i * (n + 1) + j + 1
            tris += [[a, b, c], [a, c, d]] if (i + j) % 2 == 0 else [[a, b, d], [b, c, d]]
    return v, np.asarray(tris)


def cylinder(n: int = 20, radius: float = 0.03, span: float = math.pi) -> tuple[np.ndarray, np.ndarray]:
    """The grid wrapped isometrically on a cylinder (axis y) over an arc `span`."""
    v, t = grid(n, 1.0 / n)
    a = (v[:, 0] - 0.5) * span
    return np.column_stack([radius * np.sin(a), v[:, 1] * radius * span, radius * (1.0 - np.cos(a))]), t


def cap(rings: int = 10, n_theta: int = 40, radius: float = 0.04, angle: float = 1.2) -> tuple[np.ndarray, np.ndarray]:
    """Spherical cap, pole at the bottom, staggered rings."""
    v = [np.array([0.0, 0.0, -radius])]
    starts = []
    for k in range(1, rings + 1):
        ph = angle * k / rings
        starts.append(len(v))
        for i in range(n_theta):
            th = 2.0 * math.pi * (i + 0.5 * (k % 2)) / n_theta
            v.append(radius * np.array([math.sin(ph) * math.cos(th), math.sin(ph) * math.sin(th), -math.cos(ph)]))
    t = [[0, starts[0] + (i + 1) % n_theta, starts[0] + i] for i in range(n_theta)]
    for a, b in zip(starts[:-1], starts[1:]):
        for i in range(n_theta):
            i1 = (i + 1) % n_theta
            t += [[a + i, a + i1, b + i1], [a + i, b + i1, b + i]]
    return np.asarray(v), np.asarray(t)


def loaded(cond: str, v: np.ndarray) -> np.ndarray:
    """Mask of the vertices that carry the external force."""
    if cond in ("free", "flat_half"):
        return v[:, 0] < np.median(v[:, 0])
    if cond == "flat_row":
        return np.isclose(v[:, 0], v[:, 0].min())
    return v[:, 2] < v[:, 2].min() + np.ptp(v[:, 2]) / 3.0


def mesh_for(cond: str) -> tuple[np.ndarray, np.ndarray]:
    if cond.startswith("cylinder"):
        return cylinder()
    if cond.startswith("cap"):
        return cap()
    return grid()


# --- kernels -----------------------------------------------------------------------------------------------------

@wp.kernel
def _momentum(qd: wp.array(dtype=wp.vec3), mass: wp.array(dtype=float), start: wp.array(dtype=wp.int32),
              stop: wp.array(dtype=wp.int32), col: int, px: wp.array2d(dtype=wp.float64),
              py: wp.array2d(dtype=wp.float64), pz: wp.array2d(dtype=wp.float64)):
    w = wp.tid()
    p = wp.vec3d(0.0, 0.0, 0.0)
    for i in range(start[w], stop[w]):           # fixed order: deterministic
        p = p + wp.float64(mass[i]) * wp.vec3d(qd[i])
    px[w, col] = p[0]
    py[w, col] = p[1]
    pz[w, col] = p[2]


class VbdMomentumTwin(BaseTwin):
    law_id = LAW.id
    units = {"free_err": "1", "flat_err": "1", "curved_err": "1"}
    needs_contacts = False

    def __init__(self, coarse: bool | None = None) -> None:
        """`coarse` sets the `vbd` entry's `coarse_step` (None: Alloy's default, `alloy.solvers.vbd.COARSE_DEFAULT`)."""
        super().__init__()
        if coarse is not None:
            self.settings = {"vbd": SolverSettings("vbd", {"iterations": 20, COARSE_KEY: bool(coarse)})}
        self._ranges: list[tuple[int, int]] = []
        self._loads: list[np.ndarray] = []
        self._force: wp.array | None = None
        self._start: wp.array | None = None
        self._stop: wp.array | None = None

    def conditions(self, quick: bool) -> tuple[Condition, ...]:
        return tuple(Condition(name, {}) for name in GROUPS)

    def world_gravity(self, cond: Condition) -> tuple[float, float, float]:
        return (0.0, 0.0, 0.0)

    def build(self, b: newton.ModelBuilder, cond: Condition, sample: Mapping[str, float]) -> None:
        v, t = mesh_for(cond.id)
        free = cond.id == "free"
        p0 = b.particle_count
        b.add_cloth_mesh(pos=wp.vec3(0.0, 0.0, 0.0), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(),
                         vertices=v.astype(np.float32), indices=t.reshape(-1).tolist(), density=AREAL_DENSITY,
                         tri_ke=0.0 if free else MEMBRANE_KE, tri_ka=0.0, tri_kd=0.0,
                         edge_ke=0.0 if (free or not cond.id.startswith("flat")) else BEND_KE, edge_kd=0.0,
                         particle_radius=5.0e-4)
        self._ranges.append((p0, b.particle_count))
        self._loads.append(p0 + np.flatnonzero(loaded(cond.id, v)))

    def make_solver(self, model: newton.Model, solver: str) -> newton.solvers.SolverBase:
        """Alloy's VBD path as the route builds it: the `vbd` capability factory (`alloy.solvers.vbd.make_vbd`, SCR-WP11-2)
        with the bench's settings (20 iterations); the class that ran goes to the monitors."""
        from alloy.scene.builtin_plugins import builtin_capabilities

        self.model, self.solver_name = model, solver
        self.on_model(model)
        cap = next(c for c in builtin_capabilities() if c.name == "vbd")
        built = cap.factory(model, self.settings_for(solver).merged_kwargs())
        self.monitors["solver_class"] = type(built).__name__
        return built

    def on_model(self, model: newton.Model) -> None:
        f = np.zeros((model.particle_count, 3), np.float32)
        for idx in self._loads:
            f[idx, 2] = FORCE / len(idx)
        dev = model.device
        self._force = wp.array(f, dtype=wp.vec3, device=dev)
        self._start = wp.array([r[0] for r in self._ranges], dtype=wp.int32, device=dev)
        self._stop = wp.array([r[1] for r in self._ranges], dtype=wp.int32, device=dev)

    def drive(self, step: int, t: float, state: Any, control: Any) -> None:
        wp.copy(state.particle_f, self._force)       # the runner clears forces before each drive

    def record(self, step: int, state: Any, contacts: Any, obs: Any, log: DeviceTwinLog) -> None:
        col = log.column(step)
        if col < 0:
            return
        outs = [log.channel(f"p{a}", dtype=wp.float64) for a in "xyz"]
        wp.launch(_momentum, dim=len(self._ranges), inputs=[state.particle_qd, self.model.particle_mass,
                                                            self._start, self._stop, col], outputs=outs,
                  device=log.device)

    def scalars(self, log: DeviceTwinLog) -> dict[str, np.ndarray]:
        t = log.times()
        p = np.stack([log.series(f"p{a}") for a in "xyz"], -1)          # [W, T, 3]
        impulse = FORCE * t[-1]
        err = np.linalg.norm(p[:, -1] - np.array([0.0, 0.0, impulse]), axis=1) / impulse
        ratio = p[:, -1, 2] / impulse
        for cond, e, r in zip(self.worlds, err, ratio):
            self.monitors[f"{cond.id}:P_over_FT"] = float(r)
            self.monitors[f"{cond.id}:err"] = float(e)
        out: dict[str, list[float]] = {}
        for cond, e in zip(self.worlds, err):
            out.setdefault(GROUPS[cond.id], []).append(float(e))
        return {k: np.asarray(v) for k, v in out.items()}


__all__ = ["LAW", "VbdMomentumTwin", "cap", "cylinder", "grid", "loaded"]


def register() -> None:
    """Register A17.vbd_momentum with the bench (idempotent)."""
    from alloy.bench.law import register_law

    register_law(LAW, VbdMomentumTwin)


register()
