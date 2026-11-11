# SPDX-License-Identifier: Apache-2.0
"""CT-7 (primitives, CPU): Newton narrow-phase separation vs the known geometric one, 12 pairs at 0-1 mm.

Target ≤ 1 µm. Also guards the measured face-face penetration bias (Newton under-reports overlap depth of flat
faces by 1.3-2.5 µm once the shapes are offset laterally by ≥ 4 nm; docs/notes/wp3-results.md CT-7b): ≤ 3 µm.
SDF pairs are GPU (texture SDFs need CUDA) and live in test_bias_gpu.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from alloy.contact.bias import Prim, stacked_separation

SEPARATIONS = [0.0, 1e-5, 1e-4, 5e-4, 1e-3]
SPHERE = Prim("sphere", (0.01,))
BOX = Prim("box", (0.02, 0.02, 0.01))
CAPSULE = Prim("capsule", (0.005, 0.01))
CYLINDER = Prim("cylinder", (0.006, 0.015))
PLANE = Prim("plane", ())

PAIRS = [(SPHERE, SPHERE), (PLANE, SPHERE), (BOX, SPHERE), (CAPSULE, SPHERE), (CYLINDER, SPHERE),
         (CAPSULE, CAPSULE), (PLANE, CAPSULE), (BOX, CAPSULE), (BOX, BOX), (PLANE, BOX), (PLANE, CYLINDER),
         (BOX, CYLINDER)]


@pytest.mark.parametrize(("lower", "upper"), PAIRS, ids=[f"{a.kind}-{b.kind}" for a, b in PAIRS])
def test_ct7_primitive_bias(device, lower, upper):
    reported = stacked_separation(lower, upper, SEPARATIONS, device=device)
    bias = np.abs(reported - np.asarray(SEPARATIONS))
    assert np.all(np.isfinite(reported)), "no contact reported within the 2.5 mm gap"
    assert bias.max() <= 1e-6, f"bias {bias.max() * 1e6:.3f} um"


@pytest.mark.parametrize("upper", [BOX, CYLINDER], ids=["box", "cylinder"])
def test_face_penetration_bias_guard(device, upper):
    """Known Newton bias (upstream issue): flat-face overlap under-reported by up to ~2.5 µm; guard at 3 µm."""
    lower = Prim("box", (0.05, 0.05, 0.01))
    seps = np.asarray([-3.8e-6, -2e-5])
    worst = max(float(np.abs(stacked_separation(lower, upper, list(seps), device=device, lateral=off) - seps).max())
                for off in (0.0, 4e-9, 1e-5, 1e-3))
    assert worst <= 3e-6, f"face penetration bias {worst * 1e6:.3f} um"
