# SPDX-License-Identifier: Apache-2.0
"""Most-mobile-axis masses of MuJoCo proxy bodies (known gap coupling-proxy-isotropic-mass; WP3 2026-11-07), and the
proxy mass of a body that cannot translate at its centre of mass on the Warp route (``ProxyMassMixin``, WP4 11-08).

A MuJoCo body coupled into another solver (VBD cloth, SPH, MPM) is a proxy there with one scalar mass,
1 / body_invweight0: MuJoCo's isotropic fit trace(W) / 3 of W, the translational block of J M^-1 J^T at the body's
centre of mass at qpos0 (a root body on its own joints takes 1 / mass instead, MuJoCo's simple-body path). A body
whose mobility differs by direction is heavier in that fit than along the direction it moves: C7's 0.56 kg pads on a
jaw slider under a 1 kg carriage get 1.33 kg, the carriage 6.36 kg against its 2.12 kg, a hinged tip 3x its mass.

The proxy keeps the fit (WP11: a proxy at the axis mass is too light along its other directions, and C7's clamp
bounced). What is sized from the proxy mass takes the body's most mobile axis instead, m_axis = 1 / lambda_max(W):
the exchange clock's interface bound sqrt(m / k_c) and the strong-pair mass (``Scheduler._pair_masses``) and the
exchange stiffness cap c·m / dt_x^2 (``ExchangeStiffnessCap.use_masses``). ``axis_mass_ratio`` gives m_axis over the
installed mass per body, so per-world masses keep their scale.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from alloy.core.log import get_logger

log = get_logger("coupling.axis_mass")


def _axis_and_installed(mj: Any) -> tuple[np.ndarray, np.ndarray]:
    """Per MuJoCo body of one world's model: (1 / lambda_max(W), 1 / body_invweight0); 0 for the world body and for
    bodies that do not move."""
    import mujoco  # noqa: PLC0415

    d = mujoco.MjData(mj)
    d.qpos[:] = mj.qpos0
    mujoco.mj_forward(mj, d)
    jacp, x = np.zeros((3, mj.nv)), np.zeros((3, mj.nv))
    axis, inst = np.zeros(mj.nbody), np.zeros(mj.nbody)
    for b in range(1, mj.nbody):
        if mj.body_invweight0[b, 0] <= 0.0:
            continue
        mujoco.mj_jacBodyCom(mj, d, jacp, None, b)
        mujoco.mj_solveM(mj, d, x, jacp)
        w = jacp @ x.T
        lam = float(np.linalg.eigvalsh(0.5 * (w + w.T))[-1])
        if lam > 0.0:
            axis[b], inst[b] = 1.0 / lam, 1.0 / float(mj.body_invweight0[b, 0])
    return axis, inst


def axis_mass_ratio(solver: Any) -> np.ndarray | None:
    """Per body of ``solver.model``: m_axis / installed proxy mass (<= 1; 1 where they agree or a body does not
    move), from the solver's MuJoCo model; None for a solver without one (no MuJoCo source)."""
    mj, table = getattr(solver, "mj_model", None), getattr(solver, "mjc_body_to_newton", None)
    if mj is None or table is None or mj.nv == 0:
        return None
    axis, inst = _axis_and_installed(mj)
    per_mjc = np.ones(mj.nbody)
    moving = inst > 0.0
    per_mjc[moving] = np.minimum(axis[moving] / inst[moving], 1.0)
    out = np.ones(int(solver.model.body_count))
    ids = table.numpy()                                    # [world, MuJoCo body] -> Newton body (-1: none)
    for row in ids:
        ok = row >= 0
        out[row[ok]] = per_mjc[np.flatnonzero(ok)]
    log.debug("axis-mass ratios: %d of %d MuJoCo bodies below 1 (min %.3f)", int((per_mjc < 1.0).sum()), mj.nbody,
              float(per_mjc.min(initial=1.0)))
    return out


def untranslatable(solver: Any) -> np.ndarray | None:
    """Per body of ``solver.model``: True where MuJoCo C's body_invweight0 has no translational term but a rotational
    one (the body cannot translate at its centre of mass: its only hinge passes through it); None for a solver
    without a MuJoCo model."""
    mj, table = getattr(solver, "mj_model", None), getattr(solver, "mjc_body_to_newton", None)
    if mj is None or table is None:
        return None
    w = np.asarray(mj.body_invweight0, dtype=np.float64).reshape(-1, 2)
    flag = (w[:, 0] <= 0.0) & (w[:, 1] > 0.0)
    out = np.zeros(int(solver.model.body_count), bool)
    for row in table.numpy():                              # [world, MuJoCo body] -> Newton body (-1: none)
        ok = (row >= 0) & flag[: row.shape[0]]
        out[row[ok]] = True
    return out


class ProxyMassMixin:
    """A MuJoCo body's proxy mass on every route as on the C route (WP9's wrist, 2026-11-08).

    MuJoCo Warp's set_const_0 gives a body that cannot translate at its centre of mass its rotational invweight0 in
    the translational slot (the degenerate-constraint fallback of mujoco_warp set_const ``_finalize_body_invweight0``).
    Newton's effective mass reads that slot as 1 / mass at the centre of mass
    (``eval_mujoco_coupling_effective_mass_block_kernel``), so on the Warp route such a body's proxy got 3·I, in kg·m²,
    as kg. WP9's 0.3 kg wrist on a hinge through its centre of mass got 1.6 g. MuJoCo C's invweight0 is 0 there, and
    Newton keeps the body's own mass (the C route). Every route takes that. Inertias are left as Newton fits them
    (from the rotational term, the same on both routes)."""

    def _eval_effective_body_inertial_properties(self, entry: Any, body_indices: Any, *,
                                                 raise_on_unsupported: bool = True) -> Any:
        out = super()._eval_effective_body_inertial_properties(           # type: ignore[misc]
            entry, body_indices, raise_on_unsupported=raise_on_unsupported)
        flag = untranslatable(entry.solver) if out is not None and out[0] else None
        if flag is None or not flag.any():
            return out
        masses, inertias = out
        local = entry.body_global_to_local.numpy()[np.asarray(body_indices.numpy(), dtype=np.int64)]
        own = entry.solver.model.body_mass.numpy()
        fixed = [float(own[li]) if li >= 0 and flag[li] else m for m, li in zip(masses, local, strict=True)]
        if fixed != masses:
            log.info("proxy masses: %d body(ies) that cannot translate at their centre of mass take their own mass "
                     "(MuJoCo Warp's invweight0 fallback read as 1/mass)", sum(a != b for a, b in zip(fixed, masses)))
        return fixed, inertias


__all__ = ["ProxyMassMixin", "axis_mass_ratio", "untranslatable"]
