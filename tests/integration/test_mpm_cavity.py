# SPDX-License-Identifier: Apache-2.0
"""Fixture parts give the MPM voxel rule their fixture's inner width (``alloy.integration.mpm_cavity``, from 0.1.8).

Through 0.1.7 each part of a fixture counted by its own middle extent, so the 4 mm wall slabs of a default
``add_open_cylinder`` set a 1 mm voxel for sand inside it (a 12 mm cube: 13,824 particles, 512 at its own size). A
fixture part, static or kinematic, now gives its own middle extent or, where that is smaller, its fixture's inner
width up to 4/3 of its wall: the voxel stays within a third of the wall (a thinner wall let sand through). CPU laws:
- the default vial (24 slabs, 4 mm wall, 10 mm inner radius) with a 12 mm cube of sand: voxel 1.33 mm, 5,832
  particles, one warning naming the 9 mm wall that gives 512, the wall named in the reasons;
  ``ALLOY_MPM_CAVITY=off`` gives the 0.1.7 row (1 mm, 13,824); ``alloy.estimate`` counts what the build seeds;
- a 9 mm wall holds a 3 mm voxel: the cube's own size sets it, 512 particles, no warning;
- no static part asks for a finer voxel than through 0.1.7: a cup of 40 mm inner radius keeps its slabs' 2.92 mm, and
  its reasons do not say its wall held the voxel (its slabs' own size set it);
- a kinematic vial gets the static one's voxel (through 0.1.7 its parts were no container: 3 mm, 512);
- rice in a 60 mm cup with 2 mm walls routes grain_mpm, static and kinematic: the granular regime and the jamming
  check read the cup's width, not the voxel its wall holds (through 0.1.7 the static cup raised a jamming RouteError
  from its slabs' 8.5 mm); a fill of 1 mm grains in the default vial routes grain_mpm: its regime reads the vial's
  20 mm width (a 5 mm voxel, 5 grains), not the 1.33 mm voxel the wall holds (1.3 grains: grain_rigid);
- each fixture reports its narrower inner width and its parts' wall thickness: an open box's cavity and walls, a
  cup's diameter and radial wall (not a slab's narrower width along the rim), a block's hole and its sides.

    PYTHONPATH=. python -m pytest tests/integration/test_mpm_cavity.py -q
"""

from __future__ import annotations

import pytest

import alloy
from alloy.integration import mpm_cavity, mpm_wall

pytestmark = pytest.mark.integration


def _vial(r: float = 0.01, cube: float = 0.012, wall: float = 0.004, kinematic: bool = False) -> alloy.Scene:
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_open_cylinder("vial", inner_radius=r, depth=0.04, wall=wall, kinematic=kinematic)
    s.add_box("sand", size=(cube,) * 3, pos=(0.0, 0.0, wall + cube / 2 + 0.001), material="sand_silica_dry")
    return s


@pytest.fixture(autouse=True)
def _fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mpm_wall, "_WARNED", set())
    monkeypatch.delenv(mpm_cavity.ENV, raising=False)


@pytest.mark.parametrize("kinematic", [False, True], ids=["static", "kinematic"])
def test_default_vial_routes_at_a_third_of_its_wall(caplog: pytest.LogCaptureFixture, kinematic: bool) -> None:
    with caplog.at_level("WARNING", logger="alloy.api"):
        sim = alloy.build(_vial(kinematic=kinematic), device="cpu")
    d = sim.built.report.decisions["sand"]
    assert d.numbers["voxel"] == pytest.approx(0.004 / 3) and sim.engine.model.particle_count == 5_832
    assert (d.numbers["mpm_particles"], d.numbers["mpm_particles_free"]) == (5_832, 512)
    assert d.numbers["wall_capped"] == pytest.approx(0.004)
    assert any("1/3 of the wall of fixture part 'vial_" in r and "4 mm thick" in r for r in d.reasons), d.reasons
    warned = [r.getMessage() for r in caplog.records if "set by container part" in r.getMessage()]
    assert len(warned) == 1 and "5,832" in warned[0] and "512" in warned[0] and "9 mm thick" in warned[0], warned


def test_a_wall_three_voxels_thick_keeps_the_sands_own_size() -> None:
    sim = alloy.build(_vial(wall=0.009), device="cpu")
    d = sim.built.report.decisions["sand"]
    assert d.numbers["voxel"] == pytest.approx(0.003) and sim.engine.model.particle_count == 512
    assert "mpm_particles" not in d.numbers and "wall_capped" not in d.numbers


def test_no_part_asks_for_a_finer_voxel_than_0_1_7(monkeypatch: pytest.MonkeyPatch) -> None:
    decisions = []
    for mode in ("on", "off"):
        monkeypatch.setenv(mpm_cavity.ENV, mode)
        sim = alloy.build(_vial(r=0.04, cube=0.03), device="cpu")
        decisions.append(sim.built.report.decisions["sand"])
    voxels = [d.numbers["voxel"] for d in decisions]
    assert voxels[0] == pytest.approx(voxels[1]) == pytest.approx(0.00292, abs=1e-5)
    assert "wall_capped" not in decisions[0].numbers, decisions[0].reasons     # its slabs' own size set it
    assert not any("of the wall of fixture part" in r for r in decisions[0].reasons), decisions[0].reasons


def test_off_gives_the_0_1_7_row(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(mpm_cavity.ENV, "off")
    sim = alloy.build(_vial(), device="cpu")
    assert sim.built.report.decisions["sand"].numbers["voxel"] == pytest.approx(0.001)
    assert sim.engine.model.particle_count == 13_824


def test_estimate_counts_what_the_build_seeds() -> None:
    est = alloy.estimate(_vial(), worlds=1, seconds=0.01, device="cpu")
    assert est["counts"]["mpm_particles"] == 5_832, est["counts"]
    assert [c for c in est["caveats"] if "5,832" in c and "9 mm thick" in c], est["caveats"]


def test_each_fixture_reports_its_narrower_inner_width() -> None:
    s = alloy.Scene()
    s.add_open_box("bin", inner=(0.03, 0.06, 0.04))
    s.add_open_cylinder("cup", inner_radius=0.02, depth=0.05, pos=(0.2, 0.0, 0.0))
    s.add_block_with_hole("block", (0.1, 0.1, 0.05), hole=(0.012, 0.02), depth=0.03, pos=(0.4, 0.0, 0.0),
                          precise=False)
    c = mpm_cavity.cavities(s.items)
    assert c["bin_floor"] == c["bin_wall0"] == c["bin_wall3"] == (0.03, 0.004)
    assert c["cup_wall17"] == (0.04, 0.004) and c["cup_floor"] == (0.04, 0.004)
    assert c["block_floor"] == (0.012, 0.02) and c["block_wall0"] == (0.012, 0.044)
    assert c["block_wall2"] == (0.012, 0.04)
    assert len(c) == 5 + 25 + 5


_GRAIN_1MM = """schema: alloy.object/1
name: grain1mm_fill
shape:
  particles:
    region:
    - [-0.006, -0.006, 0.0]
    - [0.006, 0.006, 0.012]
    spacing: 0.001
    grain_diameter: 0.001
material: rice_grain
"""


def test_fine_grains_in_the_default_vial_route_grain_mpm(tmp_path) -> None:
    (tmp_path / "objects").mkdir()
    (tmp_path / "objects" / "grain1mm_fill.yaml").write_text(_GRAIN_1MM)
    s = alloy.Scene(dt=0.01, roots=[str(tmp_path)])
    s.add_ground()
    s.add_open_cylinder("vial", inner_radius=0.01, depth=0.04, wall=0.004)
    s.add_object("g", "grain1mm_fill", pos=(0.0, 0.0, 0.005))
    route = alloy.estimate(s, worlds=1, seconds=0.01, device="cpu")["routes"]["g"]
    assert route["route"] == "grain_mpm", route["reasons"]


@pytest.mark.parametrize("kinematic", [False, True], ids=["static", "kinematic"])
def test_rice_in_a_thin_walled_cup_routes_grain_mpm(kinematic: bool) -> None:
    s = alloy.Scene()
    s.add_ground()
    s.add_open_cylinder("cup", inner_radius=0.03, depth=0.06, wall=0.002, kinematic=kinematic)
    s.add_object("rice", "rice_cup_fill", pos=(0.0, 0.0, 0.004))
    assert alloy.estimate(s, worlds=1, seconds=0.01, device="cpu")["routes"]["rice"]["route"] == "grain_mpm"
