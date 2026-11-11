# SPDX-License-Identifier: Apache-2.0
"""SCR-SC5-1 (ruling item 4): an object card's `blade:` block reaches its ObjectHandle, so wiring can build a Blade on
the object's body. Only handles change: the model and its digest stay as before. A blade needs exactly one body."""

from __future__ import annotations

import pytest

from alloy.core.errors import AssemblyError
from alloy.scene.assembler import SceneAssembler

TWO_BODY_XML = """<mujoco model="blade2">
  <worldbody>
    <body name="handle"><freejoint/><geom type="box" size="0.05 0.01 0.01"/>
      <body name="tip" pos="0.06 0 0"><joint type="hinge" axis="0 0 1"/><geom type="box" size="0.01 0.01 0.001"/></body>
    </body>
  </worldbody>
</mujoco>
"""


def _library(*roots):
    try:
        from alloy.cards.library import CardLibrary, default_root
    except ImportError:
        pytest.skip("WP1 CardLibrary not merged")
    return CardLibrary([*roots, default_root()])


def _scene(*names):
    from alloy.cards.types import Placement, SceneCard

    return SceneCard(robots=(), objects=tuple(Placement(n, (0.1 * i, 0.0, 0.05, 0.0, 0.0, 0.0, 1.0))
                                              for i, n in enumerate(names)), control_dt=0.02)


def test_blade_spec_on_the_handle():
    lib = _library()
    built = SceneAssembler(lib).assemble(_scene("knife_blade_steel", "cube_beech_30"), "cpu", world_count=2)
    knife, cube = built.handles["knife_blade_steel"], built.handles["cube_beech_30"]
    assert knife.blade == lib.object("knife_blade_steel").blade and knife.blade is not None
    assert len(knife.template.body) == 1 and cube.blade is None


def test_blade_needs_one_body(tmp_path):
    (tmp_path / "mjcf").mkdir()
    (tmp_path / "mjcf" / "blade2.xml").write_text(TWO_BODY_XML)
    (tmp_path / "objects").mkdir()
    (tmp_path / "objects" / "wp2_test_blade2.yaml").write_text(
        "schema: alloy.object/1\nname: wp2_test_blade2\nshape:\n  mjcf:\n    path: mjcf/blade2.xml\n"
        "material: stainless_304\ntopology: {}\nblade:\n  edge_a: [0.0, 0.0, 0.0]\n  edge_b: [0.01, 0.0, 0.0]\n")
    with pytest.raises(AssemblyError, match="wp2_test_blade2"):
        SceneAssembler(_library(tmp_path)).assemble(_scene("wp2_test_blade2"), "cpu", world_count=1)
