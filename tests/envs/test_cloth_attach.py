# SPDX-License-Identifier: Apache-2.0
"""Attach-on-grasp for cloth (``alloy.envs.cloth_attach``; ``make("ClothFold-v0", grasp="attach")``): an opt-in
shortcut, not a friction grasp.

- The switch is opt-in: ClothFold's default task installs nothing; a bad value raises.
- The kernels (CPU, no physics): vertices in the box between closed fingers are held, with the anchor where they were;
  vertices outside it, or between open fingers, are not; opening the fingers lets every held vertex go; the
  after-accumulate term is the implicit spring k (anchor - x) with Hessian k I, on that colour only.
- The law (CPU, the coupled pad rig ``alloy.coupling.pinch_hold.run_hold``, the route's cloth solver): a pinched
  cloth under a load of 2x its Coulomb limit (2 x mu x 15 N) stays in the pads with attach (the friction pinch slides
  out at that load: tests/coupling/test_cloth_grasp_hold_law.py).
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy.envs import cloth_attach as CA
from alloy.envs.soft_tasks import cloth_fold_task

HALF = (0.014, 0.005, 0.032)          # the gantry finger's box half sizes + the 2 mm margin install() adds (x, z)


def test_attach_is_opt_in():
    assert cloth_fold_task().on_build is None
    assert cloth_fold_task(grasp="attach").on_build is not None
    with pytest.raises(ValueError, match="grasp"):
        cloth_fold_task(grasp="glue")


def _fingers(gap: float) -> wp.array:
    """Two finger bodies (0: left, 1: right) facing along y, centres ``gap`` apart, at z = 0.05."""
    tf = [wp.transform(wp.vec3(0.0, 0.5 * gap, 0.05), wp.quat_identity()),
          wp.transform(wp.vec3(0.0, -0.5 * gap, 0.05), wp.quat_identity())]
    return wp.array(tf, dtype=wp.transform, device="cpu")


def _update(q: np.ndarray, gap: float, held: wp.array, anchor: wp.array) -> None:
    left, right = (wp.array([b], dtype=wp.int32, device="cpu") for b in (0, 1))
    wp.launch(CA.update_hold, dim=len(q), inputs=[wp.array(q, dtype=wp.vec3, device="cpu"),
                                                  wp.zeros(len(q), dtype=wp.int32, device="cpu"), _fingers(gap),
                                                  left, right, wp.vec3(*HALF), CA.CLOSE_GAP, CA.OPEN_GAP],
              outputs=[held, anchor], device="cpu")


def test_hold_decision_and_release():
    q = np.array([[0.0, 0.0, 0.05],          # between the fingers, centre
                  [0.010, 0.002, 0.03],      # between, near the tip and the edge (inside the margin)
                  [0.0, 0.0, 0.10],          # above the fingers' length
                  [0.03, 0.0, 0.05],         # beside them, past the width
                  [0.0, 0.02, 0.05]],        # outside the left finger (beyond its centre plane)
                 np.float32)
    held = wp.zeros(len(q), dtype=wp.int32, device="cpu")
    anchor = wp.zeros(len(q), dtype=wp.vec3, device="cpu")
    _update(q, 0.09, held, anchor)                                     # open: nothing is held
    assert held.numpy().tolist() == [0, 0, 0, 0, 0]
    _update(q, 0.014, held, anchor)                                    # closed on the ridge
    assert held.numpy().tolist() == [1, 1, 0, 0, 0]
    np.testing.assert_allclose(anchor.numpy()[0], [0.0, -0.007, 0.0], atol=1e-7)   # in the left finger's frame
    _update(q, 0.025, held, anchor)                                    # between the thresholds: kept
    assert held.numpy().tolist() == [1, 1, 0, 0, 0]
    _update(q, 0.035, held, anchor)                                    # opened past OPEN_GAP: all let go
    assert held.numpy().tolist() == [0, 0, 0, 0, 0]


def test_spring_term_on_its_colour_only():
    q = np.array([[0.0, 0.001, 0.05], [0.0, -0.001, 0.05]], np.float32)
    held = wp.array([1, 1], dtype=wp.int32, device="cpu")
    anchor = wp.array([[0.0, -0.007, 0.0], [0.0, -0.007, 0.0]], dtype=wp.vec3, device="cpu")   # world (0, 0, 0.05)
    colors = wp.array([0, 1], dtype=wp.int32, device="cpu")
    forces = wp.zeros(2, dtype=wp.vec3, device="cpu")
    hess = wp.zeros(2, dtype=wp.mat33, device="cpu")
    wp.launch(CA.hold_accumulate, dim=2,
              inputs=[0, colors, wp.array(q, dtype=wp.vec3, device="cpu"), wp.zeros(2, dtype=wp.int32, device="cpu"),
                      _fingers(0.014), wp.array([0], dtype=wp.int32, device="cpu"), held, anchor, 1.0e3],
              outputs=[forces, hess], device="cpu")
    np.testing.assert_allclose(forces.numpy()[0], [0.0, -1.0, 0.0], atol=1e-5)   # k (a - x): 1e3 x -1 mm
    np.testing.assert_allclose(hess.numpy()[0], 1.0e3 * np.eye(3), atol=1e-3)
    assert not forces.numpy()[1].any() and not hess.numpy()[1].any()            # colour 1: untouched


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(3600)
def test_attach_holds_twice_the_friction_limit():
    from alloy.coupling.pinch_hold import run_hold
    from alloy.coupling.pinch_rigs import PAD_MASS

    # the rig's cheap setting (10 VBD iterations, 10x pads: a longer exchange step, 0.1 s ramp, 0.2 s hold): about 4 min
    # of one core. Measured 2026-11-10: attach 0.50 mm, in the pads (route default 40 it, 0.3 s hold: 0.47 mm);
    # friction alone at the same load and setting: the cloth falls out of the pads
    res = run_hold(2.0, hold_s=0.2, settle_s=0.1, iterations=10, pad_mass=10 * PAD_MASS, attach=True)
    assert res.finite and res.pass_through == 0
    assert not res.left_pads
    assert abs(res.slide) < 1.0e-3, res.slide        # held: under 1 mm at 2x mu N (the spring's settling)
