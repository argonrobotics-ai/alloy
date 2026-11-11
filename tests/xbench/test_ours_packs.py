# SPDX-License-Identifier: Apache-2.0
"""P3-2 / P3-2b scene packs: every equality names its objects by their own type; actuator force clamps match the
scene (mujoco needed for those)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PACKS = Path(__file__).resolve().parents[2] / "bench" / "xbench" / "ours" / "packs"
EMB = sorted(p for p in PACKS.glob("emb_*") if (p / f"{p.name}.json").exists())


@pytest.mark.parametrize("pack", EMB, ids=[p.name for p in EMB])
def test_equalities_name_their_objects(pack):
    eqs = json.loads((pack / f"{pack.name}.json").read_text())["equality"]
    want = {"connect": ("body", "site"), "weld": ("body", "site"), "joint": ("joint",), "tendon": ("tendon",)}
    for e in eqs:
        assert e["objtype"] in want[e["type"]] and e["obj1"], e


def test_e3_four_bar_connects_follower_to_coupler():
    eqs = json.loads((PACKS / "emb_e3_ur10e_2f85" / "emb_e3_ur10e_2f85.json").read_text())["equality"]
    pairs = {(e["obj1"], e["obj2"]) for e in eqs if e["type"] == "connect"}
    assert pairs == {("gripper_right_follower", "gripper_right_coupler"),
                     ("gripper_left_follower", "gripper_left_coupler")}


ALL = sorted(p for p in PACKS.iterdir() if (p / f"{p.name}.json").exists() and (p / "scene.xml").exists())


def _clamps_from_scene(xml: Path) -> dict[str, list[float] | None]:
    """Each actuator's clamp read back from the pack's MJCF as MuJoCo compiles it (defaults resolved): actuator
    forcerange x gear when forcelimited, intersected with the joint's actuatorfrcrange when actuatorfrclimited."""
    mujoco = pytest.importorskip("mujoco")
    m = mujoco.MjModel.from_xml_path(str(xml))
    out = {}
    for a in range(m.nu):
        j = int(m.actuator_trnid[a][0])
        lo, hi = float("-inf"), float("inf")
        if m.actuator_forcelimited[a]:
            lo, hi = sorted(float(x) * float(m.actuator_gear[a][0]) for x in m.actuator_forcerange[a])
        if m.jnt_actfrclimited[j]:
            lo, hi = max(lo, float(m.jnt_actfrcrange[j][0])), min(hi, float(m.jnt_actfrcrange[j][1]))
        out[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a)] = None if hi == float("inf") else [lo, hi]
    return out


@pytest.mark.parametrize("pack", ALL, ids=[p.name for p in ALL])
def test_actuator_force_clamps_match_the_scene(pack):
    """XB4-4: t12.json said "no force limit" while scene.xml clamps every arm joint (actuatorfrcrange). The pack's
    force_clamp must equal what MuJoCo applies, read back from the pack's own scene.xml."""
    acts = json.loads((pack / f"{pack.name}.json").read_text())["actuators"]
    assert {a["name"]: a["force_clamp"] for a in acts} == _clamps_from_scene(pack / "scene.xml")


def test_t12_clamps_are_the_arm_and_finger_ranges():
    acts = json.loads((PACKS / "t12" / "t12.json").read_text())["actuators"]
    want = {**{f"{s}_m{i}": [-28.0, 28.0] for s in ("left", "right") for i in (1, 2, 3)},
            **{f"{s}_m{i}": [-10.0, 10.0] for s in ("left", "right") for i in (4, 5, 6)}}
    got = {a["name"]: a["force_clamp"] for a in acts}
    assert {k: got[k] for k in want} == want
    assert sorted(v for k, v in got.items() if k not in want) == [[-100.0, 100.0]] * 2      # finger_a, both arms
