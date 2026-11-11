# SPDX-License-Identifier: Apache-2.0
"""J2 plasticity for VBD tetrahedra (SCR-WP7-15, P1): a soft volume that yields.

**Mechanism.** Newton's VBD evaluates each tet's elastic energy with F = Ds Dm⁻¹ and Dm⁻¹ read from `model.tet_poses`
on every evaluation, rest volume 1 / (6 det Dm⁻¹) included (newton vbd/particle_vbd_kernels.py:181-255). After each
solver step this hook:
- forms F = Ds Dm0⁻¹ from the converged positions, with Dm0⁻¹ the authored rest shape;
- runs `j2.return_3d` on Fe = F Fp⁻¹;
- writes `tet_poses` = Dm0⁻¹ Fp⁻¹.
The next step then evaluates W(Fe) over the unchanged rest volume, since det Fp = 1. Newton is unchanged.

**Which tets.** Per-tet custom attributes `plastic:yield_stress` and `plastic:hardening` (Pa, TETRAHEDRON frequency)
come from a `plastic.model: j2` card (`register_attributes`, `mark_tets`). Tets with yield_stress 0 stay elastic. With
none above 0 the hook is never built: no arrays, no kernels, bitwise the stock solver.

**State** (solver state, WP9 amendment (d)): `fp_inv` (mat33) and `ebar` (ε̄p) per tet, identity / 0 where elastic.
`tet_poses` is derived from them. `reset` restores the authored state, and `rewrite_poses` rebuilds tet_poses after a
snapshot restore.

**Readout.** `ebar` per tet, on the engine (`TetPlastic.ebar`). The API has no soft-body plastic-strain key yet.

**Small yield strains** (docs/known_gaps/plasticity-soft-tets-low-yield.md). VBD's unconverged compression of a body
under gravity, at its start and at rest, can exceed a small yield strain, and the return then keeps it as plastic flow.
A 30 mm block resting on the ground yields at σ_y / E 1 % and 3.3 %, and not at 10 %. `for_model` warns once when any
j2 tet has σ_y / E under LOW_YIELD_STRAIN.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import warp as wp

from alloy.solvers.plastic.j2 import return_3d

NAMESPACE = "plastic"
YIELD, HARDENING = "yield_stress", "hardening"
LOW_YIELD_STRAIN = 0.10   # σ_y / E under this can yield a resting body spuriously (module docstring)
GAP = "docs/known_gaps/plasticity-soft-tets-low-yield.md"
log = logging.getLogger("alloy.solvers")


def register_attributes(builder: Any) -> None:
    """Declare the per-tet plastic attributes on a ModelBuilder (idempotent)."""
    from alloy._vendor import newton  # noqa: PLC0415

    for name in (YIELD, HARDENING):
        key = f"{NAMESPACE}:{name}"
        if key in builder.custom_attributes:
            continue
        builder.add_custom_attribute(newton.ModelBuilder.CustomAttribute(
            name=name, namespace=NAMESPACE, frequency=newton.Model.AttributeFrequency.TETRAHEDRON, dtype=wp.float32,
            default=0.0))


def mark_tets(builder: Any, start: int, end: int, yield_stress: float, hardening: float = 0.0) -> None:
    """Make tets [start, end) of `builder` J2 with σ_y0 = yield_stress and H = hardening [Pa]."""
    register_attributes(builder)
    for name, value in ((YIELD, yield_stress), (HARDENING, hardening)):
        attr = builder.custom_attributes[f"{NAMESPACE}:{name}"]
        if attr.values is None:
            attr.values = {}
        for t in range(start, end):
            attr.values[t] = float(value)


@wp.kernel
def _return_tets(q: wp.array(dtype=wp.vec3), tets: wp.array2d(dtype=wp.int32), dm0: wp.array(dtype=wp.mat33),
                 mat: wp.array2d(dtype=float), sy0: wp.array(dtype=float), hmod: wp.array(dtype=float),
                 fp_inv: wp.array(dtype=wp.mat33), ebar: wp.array(dtype=float), poses: wp.array(dtype=wp.mat33)):
    t = wp.tid()
    if sy0[t] <= 0.0:
        return
    x0 = q[tets[t, 0]]
    ds = wp.matrix_from_cols(q[tets[t, 1]] - x0, q[tets[t, 2]] - x0, q[tets[t, 3]] - x0)
    fp, eb = return_3d(ds * dm0[t], fp_inv[t], ebar[t], mat[t, 0], sy0[t], hmod[t])
    fp_inv[t] = fp
    ebar[t] = eb
    poses[t] = dm0[t] * fp


@wp.kernel
def _rewrite(dm0: wp.array(dtype=wp.mat33), fp_inv: wp.array(dtype=wp.mat33), poses: wp.array(dtype=wp.mat33)):
    t = wp.tid()
    poses[t] = dm0[t] * fp_inv[t]


@wp.kernel
def _reset(world_of: wp.array(dtype=wp.int32), mask: wp.array(dtype=wp.bool), dm0: wp.array(dtype=wp.mat33),
           fp_inv: wp.array(dtype=wp.mat33), ebar: wp.array(dtype=float), poses: wp.array(dtype=wp.mat33)):
    t = wp.tid()
    w = world_of[t]
    if w >= 0 and not mask[w]:
        return
    fp_inv[t] = wp.identity(n=3, dtype=float)
    ebar[t] = 0.0
    poses[t] = dm0[t]


def lowest_yield_strain(model: Any, sy0: np.ndarray) -> float:
    """The smallest σ_y / E over the j2 tets, E from each tet's Lamé pair (k_mu, k_lambda)."""
    mat = model.tet_materials.numpy().astype(np.float64)
    mu, lam = mat[:, 0], mat[:, 1]
    youngs = np.where(lam + mu > 0.0, mu * (3.0 * lam + 2.0 * mu) / np.maximum(lam + mu, 1e-30), 0.0)
    j2 = (sy0 > 0.0) & (youngs > 0.0)
    return float((sy0[j2] / youngs[j2]).min()) if j2.any() else float("inf")


def warn_low_yield_strain(model: Any, sy0: np.ndarray) -> bool:
    """One warning when a j2 tet's σ_y / E is under LOW_YIELD_STRAIN; returns whether it warned."""
    ratio = lowest_yield_strain(model, sy0)
    if ratio >= LOW_YIELD_STRAIN:
        return False
    log.warning("J2 soft body: yield strain sigma_y/E %.2g %% is under %g %%; resting under gravity it can yield "
                "spuriously (VBD's unconverged compression), see %s", 100.0 * ratio, 100.0 * LOW_YIELD_STRAIN, GAP)
    return True


class TetPlastic:
    """The post-step J2 hook for one model's tets (module doc). Build it with `TetPlastic.for_model` (None when no tet
    yields)."""

    def __init__(self, model: Any) -> None:
        attrs = getattr(model, NAMESPACE)
        dev = model.device
        self.model = model
        self.sy0 = getattr(attrs, YIELD)
        self.hmod = getattr(attrs, HARDENING)
        self.dm0 = wp.clone(model.tet_poses)
        n = model.tet_count
        self.fp_inv = wp.array(np.repeat(np.eye(3, dtype=np.float32)[None], n, axis=0), dtype=wp.mat33, device=dev)
        self.ebar = wp.zeros(n, dtype=float, device=dev)
        self.count = int((self.sy0.numpy() > 0.0).sum())

    @classmethod
    def for_model(cls, model: Any) -> TetPlastic | None:
        attrs = getattr(model, NAMESPACE, None)
        sy0 = getattr(attrs, YIELD, None) if attrs is not None else None
        if sy0 is None or model.tet_count == 0 or not bool((sy0.numpy() > 0.0).any()):
            return None
        warn_low_yield_strain(model, sy0.numpy())
        return cls(model)

    def after_step(self, particle_q: wp.array) -> None:
        m = self.model
        wp.launch(_return_tets, dim=m.tet_count,
                  inputs=[particle_q, m.tet_indices, self.dm0, m.tet_materials, self.sy0, self.hmod, self.fp_inv,
                          self.ebar, m.tet_poses], device=m.device)

    def rewrite_poses(self) -> None:
        """tet_poses from the plastic state (after a snapshot restore)."""
        m = self.model
        wp.launch(_rewrite, dim=m.tet_count, inputs=[self.dm0, self.fp_inv, m.tet_poses], device=m.device)

    def reset(self, world_mask: wp.array | None = None) -> None:
        """Authored state for the worlds in `world_mask` (all when None)."""
        m = self.model
        if world_mask is None:
            self.ebar.zero_()
            self.fp_inv.assign(np.repeat(np.eye(3, dtype=np.float32)[None], m.tet_count, axis=0))
            wp.copy(m.tet_poses, self.dm0)
            return
        wp.launch(_reset, dim=m.tet_count, inputs=[self.tet_world(), world_mask, self.dm0, self.fp_inv, self.ebar,
                                                   m.tet_poses], device=m.device)

    def tet_world(self) -> wp.array:
        """Each tet's world (from its first particle; -1 for global)."""
        if getattr(self, "_tet_world", None) is None:
            m = self.model
            pw = m.particle_world.numpy() if getattr(m, "particle_world", None) is not None else None
            first = m.tet_indices.numpy()[:, 0]
            w = pw[first] if pw is not None else np.zeros(m.tet_count, dtype=np.int32)
            self._tet_world = wp.array(w.astype(np.int32), dtype=wp.int32, device=m.device)
        return self._tet_world

    def state_arrays(self) -> dict[str, wp.array]:
        return {"plastic_tet_fp_inv": self.fp_inv, "plastic_tet_ebar": self.ebar}
