# SPDX-License-Identifier: Apache-2.0
"""SCR-DC3-5 part 1: a precision feature on a multi-shape object covers every touched shape (a label match, or a
shape inside the feature crop), not only the object's first shape, and every touched pair is filtered from native
collision. A one-shape object keeps its single pair (library scenes keep their digests: test_library_scenes)."""

from __future__ import annotations

import pytest

from alloy.core.errors import PrecisionSpecError
from alloy.core.registry import isolated_registry
from alloy.scene.assembler import SceneAssembler, _precision_shape_pairs

PLUG = "wp2_test_pad_plug"
BORE = "block_bore_d12_2"
PLUG_XML = """<mujoco model="pad_plug">
  <worldbody>
    <body name="plug">
      <geom name="shell" type="box" size="0.01 0.01 0.01" pos="0 0 0.02"/>
      <geom name="pad_0" type="box" size="0.002 0.002 0.004" pos="0.004 0 -0.004"/>
      <geom name="pad_1" type="box" size="0.002 0.002 0.004" pos="-0.004 0 -0.004"/>
    </body>
  </worldbody>
</mujoco>
"""
CLEAR = "  clearance: {value: 0.0002, source: model, note: test}\n"
NOWHERE = "  crop:\n  - [0.5, 0.5, 0.5]\n  - [0.6, 0.6, 0.6]\n"          # body frame: holds no shape


def _library(tmp_path, feature: str):
    try:
        from alloy.cards.library import CardLibrary, default_root
    except ImportError:
        pytest.skip("WP1 CardLibrary not merged")
    (tmp_path / "mjcf").mkdir(exist_ok=True)
    (tmp_path / "mjcf" / "pad_plug.xml").write_text(PLUG_XML)
    (tmp_path / "objects").mkdir(exist_ok=True)
    (tmp_path / "objects" / f"{PLUG}.yaml").write_text(
        f"schema: alloy.object/1\nname: {PLUG}\nshape:\n  mjcf:\n    path: mjcf/pad_plug.xml\n"
        f"material: stainless_303\nprecision_partners:\n- {BORE}:bore\nprecision_features:\n{feature}")
    return CardLibrary([tmp_path, default_root()])


def _scene():
    from alloy.cards.types import Placement, SceneCard

    return SceneCard(robots=(), objects=(Placement(BORE, (0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0)),
                                         Placement(PLUG, (0.0, 0.0, 0.35, 0.0, 0.0, 0.0, 1.0))), control_dt=0.02)


def _template(tmp_path, feature: str):
    from alloy.scene.builtin_plugins import register_builtin_solvers

    with isolated_registry(empty=True):
        register_builtin_solvers()
        asm = SceneAssembler(_library(tmp_path, feature))
        plan = asm.plan(_scene())
        tmpl = asm._template(plan)
    b = tmpl.builder
    shapes = {str(b.shape_label[s]).rsplit("/", 1)[-1]: s for t in plan.objects for s in t.ranges.shape}
    return asm, plan, tmpl, shapes


def _named(b, pairs, shapes):
    name = {s: n for n, s in shapes.items()}
    return sorted((name[x], name[y]) for x, y in pairs)


def test_label_match_pairs_every_pad(tmp_path):
    asm, plan, tmpl, shapes = _template(tmp_path, f"- name: pad\n{NOWHERE}{CLEAR}")
    b = tmpl.builder
    bore = next(n for n in shapes if n not in ("shell", "pad_0", "pad_1"))
    pairs = _precision_shape_pairs(b, plan.objects)
    assert _named(b, pairs, shapes) == [(bore, "pad_0"), (bore, "pad_1")] or \
        _named(b, [(y, x) for x, y in pairs], shapes) == [(bore, "pad_0"), (bore, "pad_1")]
    for x, y in pairs:                                     # every touched pair is filtered from native collision
        assert (min(x, y), max(x, y)) in b._shape_collision_filter_pairs
    shell = shapes["shell"]
    assert not any(shell in p for p in pairs)
    built = asm._precision(plan, tmpl, "cpu")
    assert sorted({p.spec.shape_a for p in built} | {p.spec.shape_b for p in built}) == \
        sorted({shapes["pad_0"], shapes["pad_1"], shapes[bore]})


def test_crop_selects_by_shape_centre(tmp_path):
    crop = "  crop:\n  - [-0.02, -0.02, 0.01]\n  - [0.02, 0.02, 0.03]\n"      # body frame: the shell only
    _, plan, tmpl, shapes = _template(tmp_path, f"- name: tip\n{crop}{CLEAR}")
    pairs = _precision_shape_pairs(tmpl.builder, plan.objects)
    assert {s for p in pairs for s in p} - {shapes[n] for n in shapes if n not in ("shell", "pad_0", "pad_1")} == \
        {shapes["shell"]}


def test_feature_matching_no_shape_is_an_error(tmp_path):
    with pytest.raises(PrecisionSpecError, match="matches no shape label"):
        _template(tmp_path, f"- name: tip\n{NOWHERE}{CLEAR}")
