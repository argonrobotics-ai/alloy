# SPDX-License-Identifier: Apache-2.0
"""A thin container part that sets an MPM voxel warns at build with both particle counts
(``alloy.integration.mpm_wall``).

A container part that is not a fixture's (or any part with ``ALLOY_MPM_CAVITY=off``, as every part through 0.1.7)
gives its middle extent as the container's inner size, so the 4 mm slabs of a default ``add_open_cylinder`` (24
slabs, 4 mm wall) set a 1 mm voxel for sand inside it. CPU laws, with ``ALLOY_MPM_CAVITY=off``, a 12 mm cube of sand
in a vial of 10 mm inner radius:
- default walls: one warning naming ``vial_wall0`` with both counts (13,824 particles, 512 at the cube's own size),
  the same counts in the route decision and its reasons (``sim.routes()``, ``alloy.estimate(...)["routes"]``) and in
  ``alloy.estimate(...)["caveats"]``, and
  the build seeds the count the warning gives;
- the two fixes the warning names, 20 mm walls or 8 wider slabs: the cube's own size sets the voxel; no warning,
  no counts;
- a round primitive: ``particle_count`` gives the emitter's count (lattice points inside a sphere).

    PYTHONPATH=. python -m pytest tests/integration/test_mpm_wall.py -q
"""

from __future__ import annotations

import pytest

import alloy
from alloy.integration import mpm_cavity, mpm_wall

pytestmark = pytest.mark.integration

THIN, FREE = 13_824, 512           # 1 mm voxel (4 mm slabs) vs 3 mm (the 12 mm cube): 24^3 vs 8^3


def _vial(wall: float, sand: str = "box", segments: int = 24) -> alloy.Scene:
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_open_cylinder("vial", inner_radius=0.01, depth=0.04, wall=wall, segments=segments)
    if sand == "box":
        s.add_box("sand", size=(0.012, 0.012, 0.012), pos=(0.0, 0.0, wall + 0.008), material="sand_silica_dry")
    else:
        s.add_sphere("sand", 0.006, pos=(0.0, 0.0, wall + 0.008), material="sand_silica_dry")
    return s


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if "set by container part" in r.getMessage()]


@pytest.fixture(autouse=True)
def _fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mpm_wall, "_WARNED", set())
    monkeypatch.setenv(mpm_cavity.ENV, "off")                    # each part by its own size, as through 0.1.7


def test_thin_vial_warns_with_both_counts(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING", logger="alloy.api"):
        sim = alloy.build(_vial(0.004), device="cpu")
        alloy.build(_vial(0.004), device="cpu")                   # a second build of the same scene: no repeat
    warned = _warnings(caplog)
    assert len(warned) == 1, warned
    assert "'vial_wall" in warned[0] and f"{THIN:,}" in warned[0] and f"{FREE:,}" in warned[0], warned[0]
    d = sim.built.report.decisions["sand"]
    assert (d.numbers["voxel"], d.numbers["mpm_particles"], d.numbers["mpm_particles_free"]) == (0.001, THIN, FREE)
    assert sim.engine.model.particle_count == THIN                # the count the warning gives is the build's
    line = [r for r in sim.routes()["sand"]["reasons"] if "set by container part" in r]
    assert line and f"{THIN:,}" in line[0] and f"{FREE:,}" in line[0], sim.routes()["sand"]["reasons"]


def test_estimate_carries_the_counts() -> None:
    est = alloy.estimate(_vial(0.004), worlds=1, seconds=0.01, device="cpu")
    reasons, caveats = est["routes"]["sand"]["reasons"], est["caveats"]
    assert any(f"{THIN:,}" in r and f"{FREE:,}" in r for r in reasons), reasons
    assert [c for c in caveats if "'vial_wall0'" in c and f"{THIN:,}" in c and f"{FREE:,}" in c], caveats
    assert est["counts"]["mpm_particles"] == THIN


@pytest.mark.parametrize(("wall", "segments"), [(0.02, 24), (0.004, 8)], ids=["thick-wall", "wide-slabs"])
def test_wide_walls_are_silent(caplog: pytest.LogCaptureFixture, wall: float, segments: int) -> None:
    with caplog.at_level("WARNING", logger="alloy.api"):
        sim = alloy.build(_vial(wall, segments=segments), device="cpu")
    assert not _warnings(caplog)
    d = sim.built.report.decisions["sand"]
    assert d.numbers["voxel"] == pytest.approx(0.003) and "mpm_particles" not in d.numbers
    assert sim.engine.model.particle_count == FREE


def test_round_primitive_count_is_the_emitters(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING", logger="alloy.api"):
        sim = alloy.build(_vial(0.004, sand="sphere"), device="cpu")
    d = sim.built.report.decisions["sand"]
    assert d.numbers["mpm_particles"] == sim.engine.model.particle_count < THIN
    assert len(_warnings(caplog)) == 1
