# SPDX-License-Identifier: Apache-2.0
"""Undo Newton's absolute inertia floor on bodies whose authored inertia is physically valid.

`ModelBuilder.finalize` adds 1e-6 kg·m² to every diagonal entry of a body whose smallest principal moment is below
max(1e-6·I_max, 1e-10) (newton@ac7f7830:newton/_src/geometry/inertia.py:1081-1088, the kernel behind finalize), however
small the body. A 150 mm harness lead's 5 mm CHAIN links (I = 1.7e-10 / 3.8e-11 kg·m²) became 1.0e-6 kg·m², 5,800x:
its released chain swung at 0.324 s against the 0.156 s reference (CARD-1). 3 mm rice grains (3e-11 kg·m²) would get
30,000x. MuJoCo itself accepts any positive inertia that satisfies the triangle inequality (balanceinertia), so a body
whose builder inertia is positive definite and satisfies I1 + I2 >= I3 gets it back on the Model; anything else keeps
Newton's correction. The builder is not mutated by finalize (builder.py:15541-15544), so its values are the authored ones.
"""

from __future__ import annotations

import logging

import numpy as np

from alloy._vendor import newton

log = logging.getLogger("alloy.scene")
TRIANGLE_RTOL = 1e-6


def valid_inertia(inertia: np.ndarray) -> np.ndarray:
    """[N] bool: symmetric positive definite and I1 + I2 >= I3 (relative tolerance) for each [N, 3, 3] tensor."""
    sym = 0.5 * (inertia + np.swapaxes(inertia, 1, 2))
    eig = np.linalg.eigvalsh(sym)
    return (eig[:, 0] > 0.0) & (eig[:, 0] + eig[:, 1] >= eig[:, 2] * (1.0 - TRIANGLE_RTOL))


def restore_authored_inertia(builder: newton.ModelBuilder, model: newton.Model) -> int:
    """Write the builder's inertia back where finalize changed a valid one; returns the bodies restored."""
    if model.body_count == 0:
        return 0
    authored = np.asarray([np.asarray(m, np.float64).reshape(3, 3) for m in builder.body_inertia])
    got = model.body_inertia.numpy().astype(np.float64)
    scale = np.maximum(np.abs(authored).max(axis=(1, 2)), 1e-30)
    changed = np.abs(got - authored).max(axis=(1, 2)) > 1e-5 * scale
    fix = changed & valid_inertia(authored)
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


__all__ = ["restore_authored_inertia", "valid_inertia"]
