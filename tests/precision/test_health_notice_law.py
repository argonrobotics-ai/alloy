# SPDX-License-Identifier: Apache-2.0
"""Law: a notice is listed and leaves the world OK (coordinator ruling 2026-11-10 02:23Z; audit PHYSA-2).

DET_DOWNGRADED ("runs on this device do not repeat bit for bit") and FALLBACK_USED ("a fallback path ran") tell the
user something and ask for no change. Before the ruling they made a world DEGRADED, so every world of every healthy
GPU rigid run was in ``degraded_worlds`` at the standard and precision profiles, and none at "fast", which repeats
no better (docs/known_gaps/determinism-gpu-rigid.md).

Laws, one per case:
- ladder: a flag word of only notices is OK; a notice beside a DEGRADED flag is DEGRADED; beside an INVALID one,
  INVALID; the notice mask is exactly the three notices (FAR_FROM_ORIGIN joined, TRAIN-FIX 4c) and shares no bit
  with the other masks;
- monitor: a downgraded run lists DET_DOWNGRADED in every world, every world reads OK, and the API summary has no
  degraded world while the row still lists the flag;
- the notice floor: the requested mode on the CPU; on a GPU at least RUN_TO_RUN, so "fast" lists the notice too;
- GPU (marked): a box pile at "standard" and at "fast" lists DET_DOWNGRADED in every world, all OK;
- the build warns once, naming the reason and the page, only when the user chose "precision" and the effective
  determinism is below the requested one (ruling 2026-11-10 03:22Z, audit WORLDS-6).
The FALLBACK_USED monitor case is tests/precision/test_mjw_chol.py::test_clamps_raise_fallback_used.
"""

from __future__ import annotations

import pytest
import warp as wp

import alloy
from alloy.api import health as api_health
from alloy.precision.determinism import notice_floor
from alloy.precision.ladder import DEGRADED_MASK, INVALID_MASK, NOTICE_MASK, severity_of
from alloy.precision.types import CheckPolicy, Determinism, HealthFlags, PrecisionConfig, Severity
from alloy.testing.precision import HealthLoop
from alloy.testing.scenes import box_on_plane

DET, FALLBACK, FAR = int(HealthFlags.DET_DOWNGRADED), int(HealthFlags.FALLBACK_USED), int(HealthFlags.FAR_FROM_ORIGIN)
NOTICES = [DET, FALLBACK, FAR, DET | FALLBACK, DET | FAR, DET | FALLBACK | FAR]


@pytest.mark.law
@pytest.mark.parametrize("bits", NOTICES)
def test_only_notices_is_ok(bits: int) -> None:
    assert severity_of(bits) == Severity.OK


@pytest.mark.law
@pytest.mark.parametrize("bits", NOTICES)
def test_a_notice_never_hides_a_real_flag(bits: int) -> None:
    assert severity_of(bits | int(HealthFlags.PEN_WARN)) == Severity.DEGRADED
    assert severity_of(bits | int(HealthFlags.JOINT_FAIL)) == Severity.DEGRADED
    assert severity_of(bits | int(HealthFlags.PEN_FAIL)) == Severity.INVALID


@pytest.mark.law
def test_the_notice_mask_is_the_three_notices_and_nothing_else() -> None:
    assert NOTICE_MASK == DET | FALLBACK | FAR
    assert NOTICE_MASK & (DEGRADED_MASK | INVALID_MASK) == 0
    assert {f.name for f in HealthFlags if NOTICE_MASK & int(f)} == set(api_health.INFO_FLAGS)


@pytest.mark.law
def test_a_downgraded_world_lists_the_notice_and_reads_ok() -> None:
    never = PrecisionConfig(determinism=Determinism.GPU_TO_GPU)        # not offered: always downgraded
    loop = HealthLoop(box_on_plane(2, device="cpu"), checks=CheckPolicy(), config=never,
                      solver_kwargs={"deterministic": wp.DeterministicMode.RUN_TO_RUN})
    loop.step(3)
    rep = loop.monitor.poll()
    assert (rep.flags & DET).all() and (rep.severity == int(Severity.OK)).all()
    summary = api_health.summary(rep, 2)
    assert summary["ok"] is True and summary["degraded_worlds"] == [] and summary["invalid_worlds"] == []
    for row in summary["worlds"]:
        assert row["severity"] == "OK" and "DET_DOWNGRADED" in row["flags"] and row["fixes"] == []
        assert "problem" not in row                                    # a notice alone is not a problem


@pytest.mark.law
@pytest.mark.parametrize("asked", list(Determinism))
def test_the_notice_floor_on_the_cpu_is_the_requested_mode(asked: Determinism) -> None:
    assert notice_floor(PrecisionConfig(determinism=asked), "cpu") == asked


@pytest.mark.gpu
@pytest.mark.law
@pytest.mark.parametrize("asked", list(Determinism))
def test_the_notice_floor_on_a_gpu_is_at_least_run_to_run(asked: Determinism) -> None:
    assert notice_floor(PrecisionConfig(determinism=asked), "cuda:0") == max(asked, Determinism.RUN_TO_RUN)


@pytest.mark.gpu
@pytest.mark.law
@pytest.mark.parametrize("profile", ["standard", "precision", "fast"])
def test_a_gpu_box_pile_lists_the_notice_in_every_world_and_reads_ok(profile: str) -> None:
    scene = alloy.Scene(worlds=4, dt=0.01)
    scene.add_ground()
    for i in range(4):
        scene.add_box(f"cube{i}", size=(0.05, 0.05, 0.05), pos=(0.004 * i, 0.003 * i, 0.03 + 0.06 * i), mass=0.1)
    sim = alloy.build(scene, device="cuda:0", profile=profile, seed=3)
    sim.run(0.5)
    health = sim.health()
    assert health["degraded_worlds"] == [] and health["invalid_worlds"] == []
    for row in health["worlds"]:
        assert row["severity"] == "OK" and row["flags"] == ["DET_DOWNGRADED"], row


class _Monitor:
    def __init__(self, asked: Determinism, got: Determinism) -> None:
        self.config, self.det_mode = PrecisionConfig(determinism=asked), got
        self.det_reasons = ("SolverMuJoCo: Warp deterministic lowering is off",)


class _Engine:
    def __init__(self, asked: Determinism, got: Determinism) -> None:
        self.health = _Monitor(asked, got)


R2R, NONE = Determinism.RUN_TO_RUN, Determinism.NONE


@pytest.mark.law
@pytest.mark.parametrize(("profile", "asked", "got", "warns"), [
    ("precision", R2R, NONE, True),          # the user asked for the careful profile and cannot have repeat runs
    ("precision", R2R, R2R, False),          # the CPU, or the ordered route: nothing to say
    ("standard", R2R, NONE, False),          # the default profile: the notice and the docs carry it, no warning
    ("fast", NONE, NONE, False),
    ("precision", Determinism.GPU_TO_GPU, R2R, False),     # runs do repeat on this device: the text would be false
])
def test_only_precision_warns_at_the_build_when_runs_cannot_repeat(profile, asked, got, warns, caplog) -> None:
    with caplog.at_level("WARNING", logger="alloy.api"):
        assert api_health.warn_downgraded(_Engine(asked, got), profile) is warns
    text = " ".join(r.getMessage() for r in caplog.records)
    assert ("will not repeat bit for bit" in text) is warns
    if warns:                                # it names the reason, the way out and the page
        assert "deterministic lowering is off" in text and 'device="cpu"' in text
        assert "docs/known_gaps/determinism-gpu-rigid.md" in text


@pytest.mark.law
def test_a_cpu_precision_build_does_not_warn(caplog) -> None:
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_sphere("ball", 0.03, pos=(0, 0, 0.3), mass=0.1)
    with caplog.at_level("WARNING", logger="alloy.api"):
        alloy.build(scene, device="cpu", profile="precision")
    assert not [r for r in caplog.records if "repeat bit for bit" in r.getMessage()]


@pytest.mark.gpu
@pytest.mark.law
@pytest.mark.parametrize(("profile", "warns"), [("precision", True), ("standard", False), ("fast", False)])
def test_a_gpu_build_warns_once_under_precision_only(profile: str, warns: bool, caplog) -> None:
    scene = alloy.Scene(worlds=2, dt=0.01)
    scene.add_ground()
    scene.add_box("cube", size=(0.05, 0.05, 0.05), pos=(0, 0, 0.1), mass=0.1)
    with caplog.at_level("WARNING", logger="alloy.api"):
        sim = alloy.build(scene, device="cuda:0", profile=profile)
        sim.step(3)
    assert len([r for r in caplog.records if "repeat bit for bit" in r.getMessage()]) == (1 if warns else 0)
