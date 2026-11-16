# SPDX-License-Identifier: Apache-2.0
"""Undo Newton's absolute inertia floor on bodies whose authored inertia is physically valid, within MuJoCo's limits.

`ModelBuilder.finalize` adds 1e-6 kg·m² to every diagonal entry of a body whose smallest principal moment is below
max(1e-6·I_max, 1e-10) (newton@ac7f7830:newton/_src/geometry/inertia.py:1081-1088, the kernel behind finalize), however
small the body. A 150 mm harness lead's 5 mm CHAIN links (I = 1.7e-10 / 3.8e-11 kg·m²) became 1.0e-6 kg·m², 5,800x:
its released chain swung at 0.324 s against the 0.156 s reference (CARD-1). 3 mm rice grains (3e-11 kg·m²) would get
30,000x. MuJoCo accepts a positive inertia that satisfies the triangle inequality (balanceinertia) down to a smallest
moment of 1e-14 kg·m² for a full tensor and 1e-15 for a diagonal one (MuJoCo 3.17: "inertia must have positive
eigenvalues in fullinertia" below that), so a body whose builder inertia is positive definite, satisfies
I1 + I2 >= I3 and clears that limit gets it back on the Model. A valid one under the limit keeps Newton's correction,
with one warning naming the body (a 0.37 mm thread segment at default density: 6.3e-16 kg·m²). The limit is MuJoCo's
alone: a body another solver steps (a clamped rod's segments on the rod solver, 4.5e-16 kg·m² for a 0.1 m copper wire
of radius 40 µm) gets its valid authored inertia back whatever its size (``mujoco_bodies``); with Newton's 1e-6 that
wire drooped 4.6 mm after 0.5 s instead of 11 mm, and not at all beside a free box. The builder is not mutated by
finalize (builder.py:15541-15544), so its values are the authored ones.

Kinematic bodies are floored before finalize instead (``floor_kinematic_inertia``): a driven body's inertia does not act
on its motion, so its principal moments are raised to Newton's floor on their own axes. Newton then corrects nothing
and warns nothing, and MuJoCo gets a tensor it accepts.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

log = logging.getLogger("alloy.scene")
TRIANGLE_RTOL = 1e-6
FLOOR_REL, FLOOR_ABS = 1e-6, 1e-10            # Newton's finalize floor: max(FLOOR_REL * I_max, FLOOR_ABS)
MJ_MIN_FULL, MJ_MIN_DIAG = 1e-14, 1e-15       # MuJoCo's smallest principal moment, full / diagonal tensor [kg m^2]
MARGIN = 1.01                                 # over a limit, for the float32 Model and the kernels' own eigensolve
_WARNED: set[str] = set()


def valid_inertia(inertia: np.ndarray) -> np.ndarray:
    """[N] bool: symmetric positive definite and I1 + I2 >= I3 (relative tolerance) for each [N, 3, 3] tensor."""
    sym = 0.5 * (inertia + np.swapaxes(inertia, 1, 2))
    eig = np.linalg.eigvalsh(sym)
    return (eig[:, 0] > 0.0) & (eig[:, 0] + eig[:, 1] >= eig[:, 2] * (1.0 - TRIANGLE_RTOL))


def mujoco_accepts(inertia: np.ndarray) -> np.ndarray:
    """[N] bool: the smallest principal moment of each [N, 3, 3] tensor clears MuJoCo's limit for its kind (diagonal
    tensors 1e-15 kg·m², full ones 1e-14), with ``MARGIN``."""
    sym = 0.5 * (inertia + np.swapaxes(inertia, 1, 2))
    diag = np.abs(sym - sym * np.eye(3)).max(axis=(1, 2)) == 0.0
    limit = np.where(diag, MJ_MIN_DIAG, MJ_MIN_FULL) * MARGIN
    return np.linalg.eigvalsh(sym)[:, 0] >= limit


def floor_kinematic_inertia(builder: newton.ModelBuilder) -> int:
    """Before ``finalize``: raise each kinematic body's principal moments (positive mass) to at least Newton's floor
    max(1e-6·I_max, 1e-10) kg·m², on the same axes (module docstring). Returns the bodies raised."""
    kinematic = int(newton.BodyFlags.KINEMATIC)
    raised = 0
    for i, flags in enumerate(builder.body_flags):
        if not int(flags) & kinematic or not float(builder.body_mass[i]) > 0.0:
            continue
        inertia = np.asarray(builder.body_inertia[i], np.float64).reshape(3, 3)
        w, v = np.linalg.eigh(0.5 * (inertia + inertia.T))
        floor = max(FLOOR_REL * max(float(w[-1]), 0.0), FLOOR_ABS) * MARGIN
        if w[0] >= floor:
            continue
        builder.body_inertia[i] = wp.mat33(*(v @ np.diag(np.maximum(w, floor)) @ v.T).ravel().tolist())
        raised += 1
    if raised:
        log.info("kinematic bodies: %d inertias raised to Newton's floor (a driven body's inertia does not act)", raised)
    return raised


def mujoco_bodies(objects: Sequence[Any], stride: int, worlds: int, count: int) -> np.ndarray:
    """[count] bool: the bodies MuJoCo may step, after ``replicate`` (global id = template id + w·``stride``): all but
    those of ``objects`` (the template's) routed to another solver's entry, such as a clamped rod's segments."""
    on = np.ones(int(count), bool)
    for t in objects:
        entry, rng = t.decision.entry, np.asarray(t.ranges.body, np.int64)
        if not entry or entry == "mujoco" or rng.size == 0:
            continue
        ids = (rng[None, :] + int(stride) * np.arange(int(worlds))[:, None]).ravel()
        on[ids[ids < on.size]] = False
    return on


def restore_authored_inertia(builder: newton.ModelBuilder, model: newton.Model,
                             mujoco: np.ndarray | None = None) -> int:
    """Write the builder's inertia back where finalize changed a valid one; returns the bodies restored. ``mujoco``:
    [N] bool, the bodies MuJoCo steps (``mujoco_bodies``; None: all). There a valid one under MuJoCo's limit keeps
    Newton's correction, with one warning naming the body."""
    if model.body_count == 0:
        return 0
    authored = np.asarray([np.asarray(m, np.float64).reshape(3, 3) for m in builder.body_inertia])
    got = model.body_inertia.numpy().astype(np.float64)
    scale = np.maximum(np.abs(authored).max(axis=(1, 2)), 1e-30)
    changed = np.abs(got - authored).max(axis=(1, 2)) > 1e-5 * scale
    valid = changed & valid_inertia(authored)
    on_mujoco = np.ones(len(authored), bool) if mujoco is None else np.asarray(mujoco, bool)
    fix = valid & (mujoco_accepts(authored) | ~on_mujoco)
    _warn_too_small(builder, authored, np.flatnonzero(valid & ~fix))
    if not fix.any():
        return 0
    got[fix] = authored[fix]
    inv = model.body_inv_inertia.numpy().astype(np.float64)
    inv[fix] = np.linalg.inv(authored[fix])
    model.body_inertia.assign(got.astype(np.float32))
    model.body_inv_inertia.assign(inv.astype(np.float32))
    log.info("restored the authored inertia of %d bodies that Newton's 1e-10 floor had raised (min moment %.3g)",
             int(fix.sum()), float(np.linalg.eigvalsh(authored[fix])[:, 0].min()))
    return int(fix.sum())


def _warn_too_small(builder: newton.ModelBuilder, authored: np.ndarray, rows: np.ndarray) -> None:
    """One warning per body name whose authored inertia is under MuJoCo's limit (Newton's correction is kept)."""
    from alloy.api import _say  # noqa: PLC0415 - the user-facing warning (alloy.api imports the scene layers)

    labels = list(getattr(builder, "body_label", []) or [])
    for i in rows.tolist():
        name = labels[i] if i < len(labels) else f"body {i}"
        if name in _WARNED:
            continue
        _WARNED.add(name)
        _say.warn("%s: its inertia (smallest moment %.3g kg·m²) is under what MuJoCo accepts (1e-14), so it steps with "
                  "Newton's floored inertia (1e-6 kg·m² added to each axis); a body this small turns far too slowly. "
                  "Fix: a larger or denser part, or kinematic=True if it is only placed or driven", name,
                  float(np.linalg.eigvalsh(0.5 * (authored[i] + authored[i].T))[0]))


__all__ = ["floor_kinematic_inertia", "mujoco_accepts", "mujoco_bodies", "restore_authored_inertia", "valid_inertia"]
