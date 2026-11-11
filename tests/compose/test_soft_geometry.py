# SPDX-License-Identifier: Apache-2.0
"""OSS-22: geometry behind soft scenes, no engine run: container cavities and capacity, resting columns, piles at
the angle of repose, cable paths on the table, and the robot-only copy of a task-scene robot file."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from alloy.cards.library import CardLibrary, default_root
from alloy.compose.catalog import Catalog
from alloy.compose.fixtures import support
from alloy.compose.layout import Layout
from alloy.compose.robot_files import robot_only
from alloy.compose.soft import cable as C
from alloy.compose.soft import fill as F
from alloy.compose.soft import sheet as SH
from alloy.compose.soft.interior import measure
from alloy.compose.soft.spec import SoftSpec


@pytest.fixture(scope="module")
def cat() -> Catalog:
    return Catalog(library=CardLibrary([default_root()], assets=None))


def test_mug_cavity_capacity_level_and_column(cat):
    body = cat.body("mug_stoneware")
    it = measure(body.vertices, body.faces)
    assert it is not None and 350 < it.capacity_ml() < 480                  # a ~0.4 l mug
    assert it.floor < 0.01 and abs(it.rim - body.size[2]) < 1e-6
    level = it.level_for(200e-6)
    assert abs(it.volume_below(level) - 200e-6) < 1e-8 and it.floor < level < it.rim
    col = F.column(it, 200e-6, margin=F.sph_margin(0.075), box=False)
    assert col is not None and abs(col.volume() - 200e-6) < 1e-9
    assert col.radius < it.border and col.z0 + col.height < it.rim         # stands on the floor, inside
    for solid in ("cube_beech_30", "plate_al_100x40x6"):
        b = cat.body(solid)
        assert measure(b.vertices, b.faces) is None


def test_amounts_and_capacity_errors(cat):
    b = cat.body("jar_glass_8oz")
    it = measure(b.vertices, b.faces)
    v, how = F.amount_m3(SoftSpec("liquid", fill=0.5), it, 1000.0)
    assert abs(it.level_for(v) - (it.floor + 0.5 * it.depth)) < 1e-4 and "50%" in how
    assert F.amount_m3(SoftSpec("grain", "pile", mass_g=320.0), None, 1600.0)[0] == pytest.approx(2e-4)
    with pytest.raises(Exception) as e:
        F.check_capacity(it, 400e-6, "water", "jar_glass_8oz")
    assert "does not fit" in str(e.value) and "at most" in e.value.fix


def test_pile_cone_and_start_box():
    angle = math.radians(33.0)
    r, h = SH.pile_size(1e-4, angle)
    assert math.pi * r * r * h / 3 == pytest.approx(1e-4) and h / r == pytest.approx(math.tan(angle))
    half, height = SH.pile_start(1e-4, r)
    assert (2 * half) ** 2 * height == pytest.approx(1e-4) and half * math.sqrt(2) <= r * 1.01
    body = SH.pile_body("pile", r, h)
    assert body.zmin == pytest.approx(0.0) and body.size[2] == pytest.approx(h)


def test_cable_paths_stay_on_the_table(cat):
    layout = Layout(support("table", cat).placed, "normal", np.random.default_rng(3))
    pts = C.loose_path(layout, 0.8, 0.002)
    assert pts is not None and C.path_length(pts) == pytest.approx(0.8, rel=1e-6)
    lo, hi = layout.support.outline.min(axis=0), layout.support.outline.max(axis=0)
    assert np.all(pts[:, :2] > lo) and np.all(pts[:, :2] < hi) and np.allclose(pts[:, 2], 0.0025)
    a, b = np.array([-0.2, 0.0, 0.02]), np.array([0.2, 0.1, 0.02])
    run = C.between(layout, a, b, 0.6, 0.002, ())
    assert run is not None and np.allclose(run[0], a) and np.allclose(run[-1], b)
    assert C.path_length(run) == pytest.approx(0.6, rel=0.03)


def test_robot_only_keeps_the_arm_of_a_task_scene():
    text = (Path(default_root()) / "robots" / "emb_e2_panda" / "scene.xml").read_text(encoding="utf-8")
    new, dropped = robot_only(text)
    assert set(dropped) == {"floor", "table", "box", "cylinder", "mug"}
    assert 'name="link0"' in new and "<keyframe" not in new and 'type="plane"' not in new
    assert 'body1="link0"' in new                                      # the arm's own contact exclude stays
    assert robot_only(new) == (new, [])


def test_soft_check_reads_a_run(cat):
    """The soft extra check on a fake run: water in the mug at the start, half of it on the table at the end."""
    from types import SimpleNamespace as NS

    from alloy.check.run import RunResult, Sample
    from alloy.compose.soft.probe import Watch, soft_check

    body = cat.body("mug_stoneware")
    it = measure(body.vertices, body.faces)
    n = 40
    inside = np.column_stack([np.zeros(n), np.zeros(n), np.full(n, 0.03)])
    spilled = inside.copy()
    spilled[: n // 2] = [0.3, 0.0, 0.002]
    arr = lambda a: NS(numpy=lambda: np.asarray(a))                                       # noqa: E731
    model = NS(particle_count=n, particle_world=arr(np.zeros(n, int)), particle_mass=arr(np.full(n, 2.5e-3)),
               body_world=arr([0]), tri_count=0)
    index = NS(particle_obj=np.full(n, 1), names=["mug", "water"], bodies_of=lambda name: np.array([0]))
    pose = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    res = RunResult(samples=[Sample(t, pose, np.zeros((1, 6)), x) for t, x in ((0.0, inside), (0.5, inside),
                                                                             (1.0, spilled))],
                    control_dt=0.02, periods=50)
    fn = soft_check([Watch("fill", "water", "mug", it, body.rest, {"volume": 1e-4, "density": 1000.0})],
                    {"mug": "mug"}, np.array([[-0.6, -0.4], [0.6, -0.4], [0.6, 0.4], [-0.6, 0.4]]))
    found, metrics = fn(res, NS(model=model, world_count=1), index)
    assert fn.check_name == "soft" and metrics["water"]["in_container"] == 0.5
    assert metrics["water"]["volume_ml_simulated"] == pytest.approx(100.0)
    assert any(f.level == "error" and "inside mug" in f.message for f in found)


@pytest.mark.parametrize("sentence, card, estimated", [("a towel", "towel_cotton", False),
                                                       ("a plastic sheet", "film_", False),
                                                       ("a polyester cloth", "cloth_polyester_sheet", False),
                                                       ("a rag", "towel_cotton", True),
                                                       ("a sheet of paper", "paper_a4", False),
                                                       ("some kraft paper", "paper_kraft_sheet", False),
                                                       ("gift wrap", "paper_wrap_sheet", False),
                                                       ("scored paper", "paper_kraft_scored_sheet", False),
                                                       ("chipboard", "paper_board_sheet", False),
                                                       ("some milk in a mug", None, True)])
def test_soft_words_pick_library_cards_or_say_what_stands_in(cat, sentence, card, estimated):
    from alloy.compose.parse import parse
    from alloy.compose.soft.materials import resolve

    item = next(i for i in parse(sentence).items if i.soft is not None)
    t = resolve(item, cat)
    assert (card is None and t.material == "water") or t.card.startswith(card)
    assert bool(t.estimated) == estimated


def test_folded_sheet_is_two_layers_joined_at_the_fold():
    v, f = SH.folded_mesh(0.3, 0.2, 6e-4, 0.01)
    gap = SH.FOLD_GAP * 6e-4
    assert sorted(set(np.round(v[:, 2], 6))) == [round(-gap / 2, 6), round(gap / 2, 6)]
    assert v[:, 0].max() - v[:, 0].min() == pytest.approx((0.3 - gap) / 2)
    edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    lengths = np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1)
    assert lengths.min() == pytest.approx(gap)                            # the fold strip joins the layers
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()
    assert area == pytest.approx(0.3 * 0.2, rel=1e-6)                     # the whole sheet, nothing lost
