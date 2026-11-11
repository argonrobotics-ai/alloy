# SPDX-License-Identifier: Apache-2.0
"""Explicit MuJoCo pair rows sized from MuJoCo's own contact metric (SCR-DC3-5 part 2, DC3 C13 latch pads).

`alloy.scene.mujoco_pairs` sizes a row's solref at assembly from the reduced *body* mass: ke = K·(1−imp)/m, so the
realised static stiffness is K when MuJoCo's contact weight (efc diagApprox, from ``body_invweight0``) is 1/m. A
body on a joint with armature breaks that. MuJoCo 3.16 counts the armature in ``body_invweight0`` under a free root
(a 0.2 g slide pad with 0.1 kg armature on a 22 g free plug: 201.5 /kg, not 5,000) and ignores it under a welded
one (5,000 /kg); a hinge counts it in the rotational weight only (`[internal note]`).
No assembly-time mass reproduces that. So the rows that touch such a body carry their K, ζ and ke cap
(`alloy:pair_k`, `alloy:pair_zeta`, `alloy:pair_kecap`). They are resized here, right after the entry is built,
from the compiled model: ke = min(K·(1−imp)·(w_a + w_b), cap), solref = (1/(ζ·√ke), ζ), with w the bodies'
translational ``body_invweight0`` and imp the row's solimp dmax (the D22 recipe of
`alloy.contact.precision.tau_for_stiffness`). Every other row, and every scene without such rows, is unchanged.

Each Newton row finds its bodies through its own two shapes (MuJoCo geom -> body), so pair order does not matter.
The router calls `fit_pair_rows` after `alloy.contact.mjw_pairs` (SCR-DC3-6), whose rows Newton's pair update then
copies to the right MuJoCo pairs; the host model is updated by pair identity.

Drawn masses (SCR-DC3-8): rows touching a body whose density is drawn per world are recorded too, with their
template reduced mass m0 and 2·dt (``alloy:pair_m0``, ``alloy:pair_tcmin``). MuJoCo's per-world ``body_invweight0`` follows the drawn masses (C13
plug: 24.3 / 16.8 / 17.2 g give 25.6 / 17.6 / 18.1 g), so an uncapped row realises K in every world. The ke cap is
the template's: a world whose reduced mass is m takes cap·m0/m, so a capped row keeps the template's force-space
stiffness m0·cap/(1−imp) whatever the draw (DC3 C13: every pad row is capped, and a 15 g plug pressed at 129.6 N and
stuck with the template rows). A light draw can then exceed the dt cap; its timeconst stays >= 2·dt, MuJoCo's own
clamp (DC3's labelled workaround used the same floor: 2.59 ms at 15 g against the 3.14 ms cap, dt 1.25 ms).
`fit_pair_rows` also wraps the solver's ``notify_model_changed``: every update that touches body inertia refits,
which covers ``Simulation.reset`` (it redraws and notifies ALL).
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from alloy.core.log import get_logger

log = get_logger("contact.pair_fit")


@dataclass(frozen=True)
class FitRows:
    """Recorded values per Newton pair row: force-space K [N/m], ζ, the ke cap at the template mass (<= 0: uncapped);
    for a row touching a drawn body its template reduced mass m0 [kg] and timeconst floor tc_min [s] (else 0)."""

    k: np.ndarray
    zeta: np.ndarray
    cap: np.ndarray
    m0: np.ndarray
    tc_min: np.ndarray

    def take(self, idx: np.ndarray) -> FitRows:
        return FitRows(*(getattr(self, f.name)[idx] for f in fields(self)))


def _rows(model) -> FitRows | None:
    """The recorded rows, or None when the scene has no row to refit."""
    alloy = getattr(model, "alloy", None)
    k = getattr(alloy, "pair_k", None)
    if k is None:
        return None
    drawn = [getattr(alloy, name, None) for name in ("pair_m0", "pair_tcmin")]
    return FitRows(_f64(k), _f64(alloy.pair_zeta), _f64(alloy.pair_kecap),
                   *(_f64(a) if a is not None else np.zeros(k.shape[0]) for a in drawn))


def _f64(a) -> np.ndarray:
    return a.numpy().astype(np.float64)


def refit_solref(rows: FitRows, imp: np.ndarray, weight: np.ndarray, mass: np.ndarray) -> np.ndarray:
    """(timeconst, dampratio) per row realising force-space K on a contact of MuJoCo weight `weight` [1/kg] (pure).
    A drawn row (m0 > 0) whose reduced mass is `mass` [kg] in this world takes the cap cap·m0/mass and a timeconst
    of at least its tc_min (module docstring); every other row keeps its cap."""
    drawn = (rows.m0 > 0.0) & (mass > 0.0)
    scale = np.where(drawn, rows.m0 / np.where(drawn, mass, 1.0), 1.0)
    ke = rows.k * (1.0 - imp) * weight
    ke = np.where(rows.cap > 0.0, np.minimum(ke, rows.cap * scale), ke)
    tc = 1.0 / (rows.zeta * np.sqrt(ke))
    return np.stack([np.where(drawn, np.maximum(tc, rows.tc_min), tc), rows.zeta], axis=1)


def _reduced_mass(model, idx: np.ndarray) -> np.ndarray:
    """Reduced mass [kg] of the dynamic bodies of Newton pair rows `idx` from the model's (drawn) body masses; 0 when
    neither body is dynamic."""
    attrs, body = model.mujoco, model.shape_body.numpy()
    mass = model.body_mass.numpy().astype(np.float64)
    inv = np.zeros(idx.size)
    for geom in (attrs.pair_geom1.numpy()[idx], attrs.pair_geom2.numpy()[idx]):
        b = body[geom]
        m = np.where(b >= 0, mass[np.maximum(b, 0)], 0.0)
        inv += np.where(m > 0.0, 1.0 / np.where(m > 0.0, m, 1.0), 0.0)
    return np.where(inv > 0.0, 1.0 / np.where(inv > 0.0, inv, 1.0), 0.0)


def _shape_bodies(solver) -> np.ndarray:
    """MuJoCo body of every Newton shape per world, shape (nworld, shape_count); -1 where a shape has no geom."""
    to_shape = solver.mjc_geom_to_newton_shape.numpy()
    to_shape = to_shape if to_shape.ndim == 2 else to_shape[None]
    body = np.asarray(solver.mj_model.geom_bodyid)
    out = np.full((to_shape.shape[0], int(solver.model.shape_count)), -1, dtype=np.int64)
    for w in range(to_shape.shape[0]):
        ok = to_shape[w] >= 0
        out[w, to_shape[w][ok]] = body[np.flatnonzero(ok)]
    return out


def _weights(solver, idx: np.ndarray, per_world: int) -> np.ndarray:
    """Sum of the two bodies' translational body_invweight0 for Newton pair rows `idx` (0 where a shape is unmapped)."""
    attrs = solver.model.mujoco
    s1, s2 = attrs.pair_geom1.numpy()[idx], attrs.pair_geom2.numpy()[idx]
    bodies = _shape_bodies(solver)
    world = (idx // per_world) % bodies.shape[0]
    b1, b2 = bodies[world, s1], bodies[world, s2]
    w = solver.mjw_model.body_invweight0.numpy()
    w = w if w.ndim == 3 else w[None]
    ww = world % w.shape[0]
    out = w[ww, np.maximum(b1, 0), 0] + w[ww, np.maximum(b2, 0), 0]
    return np.where((b1 >= 0) & (b2 >= 0), out, 0.0)


def fit_pair_rows(solver) -> int:
    """Resize the recorded pair rows of a SolverMuJoCo entry and republish; returns the number of rows resized."""
    model, mj = getattr(solver, "model", None), getattr(solver, "mj_model", None)
    rows = _rows(model) if model is not None else None
    if rows is None or mj is None or int(mj.npair) == 0 or getattr(solver, "mjc_geom_to_newton_shape", None) is None:
        return 0
    per_world = len(rows.k) // max(1, int(model.world_count))
    idx = np.flatnonzero(rows.k > 0.0)
    weight = _weights(solver, idx, per_world)
    idx, weight = idx[weight > 0.0], weight[weight > 0.0]
    if idx.size == 0:
        return 0
    attrs = model.mujoco
    solref, imp = attrs.pair_solref.numpy(), attrs.pair_solimp.numpy()[:, 1].astype(np.float64)
    solref[idx] = refit_solref(rows.take(idx), imp[idx], weight, _reduced_mass(model, idx)).astype(solref.dtype)
    attrs.pair_solref.assign(solref)
    _publish(solver, solref, per_world)
    _refit_on_notify(solver)
    log.info("pair_fit: %d pair rows sized from body_invweight0 (%d in world 0)", idx.size,
             int(np.sum(idx < per_world)))
    return int(idx.size)


def _refit_on_notify(solver) -> None:
    """Wrap the solver's notify_model_changed once: after any update touching body inertia (new masses, a reset's
    redraw), refit the recorded rows from the new body_invweight0."""
    from alloy._vendor import newton  # noqa: PLC0415

    if getattr(solver, "_alloy_pair_refit", False):
        return
    inner = solver.notify_model_changed
    inertia = int(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)

    def notify(flags) -> None:
        inner(flags)
        if int(flags) & inertia and not getattr(solver, "_alloy_pair_refitting", False):
            solver._alloy_pair_refitting = True
            try:
                fit_pair_rows(solver)
            finally:
                solver._alloy_pair_refitting = False

    solver.notify_model_changed = notify
    solver._alloy_pair_refit = True


def _publish(solver, solref: np.ndarray, per_world: int) -> None:
    """MuJoCo Warp through Newton's own pair update (rows put where it copies them by alloy.contact.mjw_pairs);
    the host model by pair identity."""
    from alloy._vendor import newton  # noqa: PLC0415

    from alloy.contact.mjw_pairs import pair_order  # noqa: PLC0415

    order = pair_order(solver, per_world)
    if order is not None:
        solver.mj_model.pair_solref[:] = solref[:per_world][order]
    if not getattr(solver, "use_mujoco_cpu", False):
        solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)


__all__ = ["FitRows", "fit_pair_rows", "refit_solref"]
