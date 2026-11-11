# SPDX-License-Identifier: Apache-2.0
"""Static squeeze twin (a stiffness law for FIT-3's k): a soft cube on a fixed base under an eased top load.

bench §4.4 R5 squeezes soft parts between plates and judges F at 10 % / 20 % strain (10 %). This fit-only twin is the
static form of the same measurement: a VBD tet cube (20 mm, 4 x 4 x 4 cells, bottom face fixed) carries a top load F
spread over its top-face particles, eased in over T_EASE (CON-51), and the observable is the settled top compression
[m] (mean over the last quarter of the run). Linear elasticity gives delta ~ F L / (E A) with a fixed-base
correction, so the fitted leaf is `elastic.youngs` (E); Poisson's ratio is held at the card's value. The law is not
registered in the bench suite (R5 belongs to WP6); it exists so FIT-3 can measure stiffness recovery on synthetic twin
data. Lame parameters: mu = E / (2 (1 + nu)), lambda = E nu / ((1 + nu)(1 - 2 nu)).

Measured (CPU, E = 50 kPa, F = 1 N, [internal note]): settled compression 1.95 mm at VBD 100
iterations, 1.836 mm at 400 iterations and at 50 iterations x 8 substeps (converged), against F L / (E A) = 1.0 mm for
free uniaxial compression; 20 iterations gives 4.63 mm and a sublinear E dependence. The 1.84x gap to the uniaxial
estimate is the 4-cell VBD discretisation, not under-iteration (not mesh-converged; the twin is used on its own data).
No closed-form start, so the search starts at the box centre (log scale).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from alloy._vendor import newton
import numpy as np
import warp as wp

from alloy.bench.analytic.common import BaseTwin, smoothstep
from alloy.bench.solvers import SolverSettings
from alloy.bench.twin import DeviceTwinLog
from alloy.bench.types import Condition, LawSpec, RigCard, RigTrial, Tolerance
from alloy.fit.twins.base import FitValues

SIDE = 0.02            # [m] cube edge
CELLS = 4
DENSITY = 30.0         # [kg/m^3] PU foam (library pu_foam_sponge)
NU = 0.3
DAMP = 1e-2            # VBD tet damping (k_damp), settles the cube within the run
T_EASE = 0.1           # [s]
LAW = LawSpec(
    id="FIT.static_squeeze", suite="rig",
    metrics={"compression": Tolerance(rel=0.10, source="bench §4.4 R5: force at 10 / 20 % strain within 10 %")},
    stat={"compression": "mean"}, dt=1.25e-3, substeps=1, duration_s=0.4, worlds_per_condition=1,
    solvers=("vbd",), producer="WP11", quick={"worlds": 16, "duration_s": 0.3},
)


def lame(E: float, nu: float) -> tuple[float, float]:
    return E / (2.0 * (1.0 + nu)), E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))


@wp.kernel
def _load_top(top: wp.array2d(dtype=wp.int32), force: wp.array(dtype=float), ease: float,
              f: wp.array(dtype=wp.vec3)):
    w, k = wp.tid()
    n = float(top.shape[1])
    f[top[w, k]] = wp.vec3(0.0, 0.0, -force[w] * ease / n)


@wp.kernel
def _rec_top(q: wp.array(dtype=wp.vec3), top: wp.array2d(dtype=wp.int32), col: int, out: wp.array2d(dtype=float)):
    w = wp.tid()
    s = float(0.0)
    for k in range(top.shape[1]):
        s += q[top[w, k]][2]
    out[w, col] = s / float(top.shape[1])


class SqueezeTwin(BaseTwin):
    """Fit twin: `youngs` per (virtual) condition; condition param `force` [N]."""

    law_id = LAW.id
    units = {"compression": "m"}
    needs_contacts = False
    fit_fields = frozenset({"youngs"})

    def __init__(self, youngs: float | None = None, nu: float = NU, cells: int = CELLS, iterations: int = 100) -> None:
        super().__init__()
        self.values = FitValues({} if youngs is None else {"youngs": float(youngs)})
        self.nu, self.cells = nu, int(cells)
        self.settings = {"vbd": SolverSettings("vbd", {"iterations": int(iterations)})}
        self._top: list[np.ndarray] = []

    def bind_values(self, cond_id: str, values: Mapping[str, float]) -> None:
        self.values.bind(cond_id, values)

    def fix_values(self, values: Mapping[str, float]) -> None:
        self.values.fix(values)

    def bind_calib(self, card: RigCard, calib: Sequence[RigTrial]) -> None:
        self.monitors["youngs_fixed"] = dict(self.values.fixed or self.values.defaults)

    def build(self, b: newton.ModelBuilder, cond: Condition, sample: Mapping[str, float]) -> None:
        k_mu, k_lambda = lame(self.values.get(cond.id, "youngs"), self.nu)
        first = b.particle_count
        n, cell = self.cells, SIDE / self.cells
        b.add_soft_grid(pos=wp.vec3(0.0, 0.0, 0.0), rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0),
                        dim_x=n, dim_y=n, dim_z=n, cell_x=cell, cell_y=cell, cell_z=cell,
                        density=DENSITY, k_mu=k_mu, k_lambda=k_lambda, k_damp=DAMP, fix_bottom=True)
        z = np.array([p[2] for p in b.particle_q[first:]])
        self._top.append(first + np.flatnonzero(np.isclose(z, z.max())))

    def world_gravity(self, cond: Condition) -> tuple[float, float, float]:
        return (0.0, 0.0, 0.0)            # the load is the applied force only

    def on_model(self, model: newton.Model) -> None:
        dev = model.device
        self._top_arr = wp.array(np.stack(self._top).astype(np.int32), dtype=wp.int32, device=dev)
        self._force = wp.array(self.param("force").astype(np.float32), dtype=float, device=dev)

    def drive(self, step: int, t: float, state: Any, control: Any) -> None:
        wp.launch(_load_top, dim=self._top_arr.shape, inputs=[self._top_arr, self._force, smoothstep(t / T_EASE)],
                  outputs=[state.particle_f], device=self.model.device)

    def record(self, step: int, state: Any, contacts: Any, obs: Any, log: DeviceTwinLog) -> None:
        col = log.column(step)
        if col >= 0:
            wp.launch(_rec_top, dim=len(self.worlds), inputs=[state.particle_q, self._top_arr, col],
                      outputs=[log.channel("ztop")], device=self.model.device)

    def scalars(self, log: DeviceTwinLog) -> dict[str, np.ndarray]:
        z = log.series("ztop").astype(np.float64)
        tail = z[:, int(0.75 * z.shape[1]):]
        return {"compression": z[:, 0] - tail.mean(axis=1)}


__all__ = ["SqueezeTwin", "LAW", "lame"]
