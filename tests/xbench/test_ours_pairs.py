# SPDX-License-Identifier: Apache-2.0
"""P3-2 scorer contact pairs (CPU): Newton's pipeline collides only the body pairs MuJoCo's filter lets collide."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

OURS = Path(__file__).resolve().parents[2] / "bench" / "xbench" / "ours"
sys.path.insert(0, str(OURS))

from packs_io import mujoco_shape_pairs, pack_of_target, target_model  # noqa: E402


def _body_pairs(model, pairs) -> set[tuple[str, str]]:
    names = [str(lbl).rsplit("/", 1)[-1] for lbl in model.body_label] + ["world"]
    sb = model.shape_body.numpy()
    return {tuple(sorted((names[sb[a]], names[sb[b]]))) for a, b in pairs.numpy().reshape(-1, 2)}


@pytest.mark.parametrize(("target", "filtered"), [
    ("task_t12", ("left_d405", "left_finger_a")),                    # jointless camera welded to link 6
    ("emb_e3_ur10e_2f85", ("gripper_base", "wrist_3_link")),         # gripper two fixed joints below the wrist
])
def test_scorer_pairs_are_the_packs(target, filtered):
    model = target_model(target, 2, "cpu")
    allowed = {tuple(sorted((p["body1"], p["body2"]))) for p in pack_of_target(target)["contact_body_pairs"]}
    assert filtered in _body_pairs(model, model.shape_contact_pairs)      # Newton's importer alone lets it collide
    kept = _body_pairs(model, mujoco_shape_pairs(model, target))
    assert kept == allowed and filtered not in kept


def test_targets_without_a_pack_keep_the_models_pairs():
    assert pack_of_target("no_such_target") is None
