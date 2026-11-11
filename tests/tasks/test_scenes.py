# SPDX-License-Identifier: Apache-2.0
"""Task SceneCards: every library/scenes/task_*.yaml validates, the authored cards are reproducible, subsets compile."""

from __future__ import annotations

from pathlib import Path

import mujoco
import pytest

from alloy.cards.library import CardLibrary
from alloy.cards.loader import dump_card
from alloy.tasks.mjb_convert import write_scene
from alloy.tasks.mjcf_subset import subset
from alloy.tasks.scenes import t02_card, t09_card
from alloy.testing.tasks import ROBOTS

SCENES = ROBOTS.parent / "scenes"
TASK_SCENES = sorted(p.stem for p in SCENES.glob("task_*.yaml"))


def test_task_scenes_present() -> None:
    assert {"task_t02", "task_t09", "task_t11", "task_t12", "task_t14", "task_h1"} <= set(TASK_SCENES)


@pytest.mark.parametrize("name", TASK_SCENES)
def test_task_scene_validates(name: str) -> None:
    card = CardLibrary().scene(name)  # raises CardError on any error-level issue
    assert card.robots and (ROBOTS.parent / card.robots[0].path).is_file()


@pytest.mark.parametrize(("name", "build"), [("task_t09", t09_card), ("task_t02", t02_card)])
def test_authored_cards_match_their_builders(name: str, build) -> None:  # noqa: ANN001
    text = (SCENES / f"{name}.yaml").read_text()
    body = "".join(line + "\n" for line in text.splitlines() if not line.startswith("#"))
    assert body == dump_card(build())


def test_subset_drops_bodies_and_their_references(synth_model: mujoco.MjModel, tmp_path: Path) -> None:
    xml = write_scene(synth_model, tmp_path / "robots", "synth")
    out = subset(xml, tmp_path / "robots" / "cut" / "robot.xml", "cut", {"block", "wedge_obj"})
    m = mujoco.MjModel.from_xml_path(str(out))
    assert m.npair == 0 and m.nmesh == 0 and m.nu == synth_model.nu and m.nexclude == 1
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "block") == -1


def test_t12_card_tier_matches_trace_initial_poses() -> None:
    from alloy.tasks.scenes import T12_CARDS, card_tier_scene
    from alloy.tasks.trace import load_trace
    from alloy.testing.tasks import TRACES

    tt = load_trace(TRACES / "t12")
    card = card_tier_scene(ROBOTS / "task_t12" / "scene.xml", tt.trace.initial["qpos"], T12_CARDS, "yam_bench")
    text = (SCENES / "task_t12_cards.yaml").read_text()
    assert "".join(line + "\n" for line in text.splitlines() if not line.startswith("#")) == dump_card(card)
    ref0 = tt.ref.pos[0]  # first recorded sample (50 steps in): the blocks rest, so it equals the start pose
    starts = [p.pose[:3] for p in card.objects if p.object == "block_alder_75x25x15"]
    assert max(abs(a - b) for s, r in zip(starts, ref0) for a, b in zip(s, r)) < 1e-6


def test_t02_nut_starts_at_rest_beside_the_bolt() -> None:
    """Audit LIB-4 (2026-11-10): the nut started on the bolt tip, and with no thread between the two in a scene built
    from cards it fell down the shank and hit the head at 1 m/s (PAIR_PEN_FAIL at 0.11 s, default profile). It starts
    flat on the plywood base, clear of the bolt and inside the base's edge."""
    place = {o.object: o.pose for o in t02_card().objects}
    nut, bolt, base = place["nut_m10"], place["bolt_m10x60"], place["pegboard_base_plywood"]
    half_x, half_z = 0.075, 0.00635                                            # pegboard_base_plywood's half sizes
    assert abs(nut[2] - (base[2] + half_z + 0.004)) < 1e-9                     # 8 mm nut flat on the 12.7 mm base
    assert abs(nut[0] - bolt[0]) >= 0.03 and abs(nut[0] - base[0]) + 0.01 <= half_x and nut[1] == base[1]

