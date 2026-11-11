# SPDX-License-Identifier: Apache-2.0
"""Regression (CARD-1, harness_lead): Newton's finalize raises any inertia whose smallest moment is below 1e-10 kg·m²
by 1e-6 on every diagonal; the assembler puts a valid authored inertia back (`alloy.scene.inertia`)."""

from __future__ import annotations

import numpy as np
import pytest

from alloy._vendor import newton
from alloy.scene.inertia import restore_authored_inertia, valid_inertia

LINK = np.diag([1.7e-10, 1.7e-10, 3.8e-11])      # a 5 mm harness-lead CHAIN link [kg·m²]
BAD = np.diag([1e-12, 1e-12, 1.0])               # breaks I1 + I2 >= I3: Newton's correction stays


def _body(b: newton.ModelBuilder, mass: float, inertia: np.ndarray) -> int:
    k = b.add_body(mass=mass, inertia=inertia)
    b.add_shape_box(k, hx=0.0025, hy=0.0025, hz=0.0025, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
    return k


def test_valid_inertia_rule():
    ok = valid_inertia(np.stack([LINK, BAD, -LINK, np.eye(3)]))
    assert ok.tolist() == [True, False, False, True]


def test_restore_authored_inertia_after_finalize():
    b = newton.ModelBuilder()
    link, bad = _body(b, 3e-4, LINK), _body(b, 1.0, BAD)
    with pytest.warns(Warning):
        model = b.finalize("cpu")
    assert model.body_inertia.numpy()[link][0, 0] > 1e-7            # Newton's floor: 1e-6, ~5,800x the link's
    assert restore_authored_inertia(b, model) == 1
    np.testing.assert_allclose(model.body_inertia.numpy()[link], LINK, rtol=1e-6)
    np.testing.assert_allclose(model.body_inv_inertia.numpy()[link], np.linalg.inv(LINK), rtol=1e-6)
    assert model.body_inertia.numpy()[bad][0, 0] > 1e-7             # invalid authored inertia keeps the correction
    assert restore_authored_inertia(b, model) == 0                  # idempotent: nothing left to restore
