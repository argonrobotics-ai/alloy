# SPDX-License-Identifier: Apache-2.0
"""``suction.target_shapes`` (shared with ``adhesion``) keeps only colliding shapes: a decompose card's display mesh
(``<name>/visual``, SCR OSS-9-5 item 4; WP2 review) is no surface to seal on (CPU, no stepping)."""

from __future__ import annotations

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.plugins.context import PluginContext
from alloy.plugins.library.suction import target_shapes
from alloy.scene.emitters.common import TriMesh, shape_config
from alloy.scene.emitters.rigid import _add_display_mesh
from alloy.testing import route as T


def _model(worlds: int = 2) -> newton.Model:
    """Per world: a body ``box`` with two colliding box shapes (hulls) and its display mesh, as _emit_decomposed."""
    b = newton.ModelBuilder()
    body = b.add_body(xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()), label="box")
    cfg = shape_config(T.material("wood", **T.WOOD))
    first = b.shape_count
    for x in (-0.01, 0.01):
        b.add_shape_box(body, xform=wp.transform((x, 0.0, 0.0), wp.quat_identity()), hx=0.01, hy=0.01, hz=0.01,
                        cfg=cfg, label=f"box/hull_{int(x > 0)}")
    v = np.array([[x, y, z] for x in (-0.02, 0.02) for y in (-0.01, 0.01) for z in (-0.01, 0.01)])
    f = np.array([(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
                  (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)], np.int32)
    _add_display_mesh(b, body, TriMesh(v, f), cfg, "box", range(first, b.shape_count))
    scene = newton.ModelBuilder()
    scene.replicate(b, worlds)
    return scene.finalize(device="cpu")


def test_target_shapes_leave_out_the_display_mesh() -> None:
    m = _model()
    ctx = PluginContext(model=m, control_dt=0.01)
    _, shapes, target = target_shapes(ctx, ("box",), "suction")
    labels = [str(x).rsplit("/", 1)[-1] for x in m.shape_label]
    got = shapes.numpy()
    assert got.shape == (2, 2) and target.numpy().tolist() == [0, 0]
    assert all(labels[s].startswith("hull_") for s in got.reshape(-1))                  # box/visual left out
    assert sum(lbl == "visual" for lbl in labels) == 2                                  # it is in the model
