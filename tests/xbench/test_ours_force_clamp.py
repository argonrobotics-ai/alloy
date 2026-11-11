# SPDX-License-Identifier: Apache-2.0
"""XB4-4 (CPU): the embodiment packs' ``force_clamp`` differs from forcerange x gear only for E5's spine joint, and the
Isaac Lab port's ``clamp_limit`` (engine-free; the ManiSkill helper needs sapien and is not imported here) reads it."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

OURS = Path(__file__).resolve().parents[2] / "bench" / "xbench" / "ours"
sys.path[:0] = [str(OURS / "engines" / "isaaclab"), str(OURS)]


def _emb_actuators():
    for f in sorted((OURS / "packs").glob("emb_*/emb_*.json")):
        for a in json.loads(f.read_text())["actuators"]:
            yield f.stem, a


def _composed(a):
    """forcerange x gear (None if unlimited); the joint's actuatorfrcrange is the extra clamp force_clamp may add."""
    return [x * a["gear"] for x in a["forcerange"]] if a["forcelimited"] else None


def test_force_clamp_differs_from_forcerange_only_for_e5_spine():
    diff = [(pack, a["joint"], a["force_clamp"]) for pack, a in _emb_actuators() if a["force_clamp"] != _composed(a)]
    assert sorted(diff) == [("emb_e5_fr3_duo", "franka_spine_vertical_joint", [-100.0, 100.0]),
                            ("emb_e5_fr3_duo_stiff", "franka_spine_vertical_joint", [-100.0, 100.0])]


def test_every_force_clamp_is_symmetric_for_the_ports():
    joints = pytest.importorskip("joints_t12")
    for pack, a in _emb_actuators():
        limit = joints.clamp_limit(a["force_clamp"])
        assert limit is None if a["force_clamp"] is None else limit == a["force_clamp"][1], (pack, a["name"])


def test_clamp_limit_rejects_asymmetric_clamps():
    joints = pytest.importorskip("joints_t12")
    assert joints.clamp_limit(None) is None and joints.clamp_limit([-3.0, 3.0]) == 3.0
    with pytest.raises(ValueError):
        joints.clamp_limit([-1.0, 3.0])
