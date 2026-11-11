# SPDX-License-Identifier: Apache-2.0
"""A8.squeeze_wedge / A8.squeeze_face (alloy.contact.law_squeeze): registration, the seeded ensemble, the world layout
from WP13's scenes, the scalars on a synthetic log, and a short stepped wedge run on CPU."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from alloy.bench import law as L
from alloy.bench.runner import RunContext, build_model, plan_worlds, run_law
from alloy.bench.twin import DeviceTwinLog
from alloy.contact import law_squeeze as S


def test_registered_with_one_mm_bound() -> None:
    assert L.discover(("alloy.contact.laws",)) == {"alloy.contact.laws": None}
    ids = {s.id for s, _ in L.registered_laws()}
    assert {"A8.squeeze_wedge", "A8.squeeze_face"} <= ids
    for spec in (S.LAW_WEDGE, S.LAW_FACE):
        assert spec.metrics["peak_pen_m"].abs == pytest.approx(1e-3)
        assert spec.solvers == ("mujoco",) and spec.producer == "WP3" and spec.dt == pytest.approx(S.DT)
    assert S.LAW_WEDGE.duration_s == pytest.approx(0.255) and S.LAW_FACE.duration_s == pytest.approx(0.8)


def test_ensemble_is_seeded_and_world0_is_the_recorded_start() -> None:
    a, b = S.perturbations(16, True), S.perturbations(16, True)
    assert a == b and a[0] == {"k": 0.0} and "speed" in a[1]
    assert all(1 - S.SPREAD <= p["speed"] <= 1 + S.SPREAD for p in a[1:])
    assert all(p == {"k": float(k)} for k, p in enumerate(S.perturbations(4, False)))
    v = np.array([1.0, 0.0, 0.0])
    t = S.tilt(v, np.array([0.3, 1.0, 0.0]), 5.0)
    assert np.linalg.norm(t) == pytest.approx(1.0) and np.degrees(np.arccos(t @ v)) == pytest.approx(5.0)


def test_world_layout_from_the_scene() -> None:
    twin = S.SqueezeTwin()
    ctx = RunContext(S.LAW_WEDGE, "cpu", 0, "mujoco", True)
    model = build_model(S.LAW_WEDGE, twin, plan_worlds(S.LAW_WEDGE, twin.conditions(True), ctx, None), ctx)
    assert model.world_count == 16 and model.joint_count == 32 and model.body_count == 32
    twin.model = model
    twin.on_model(model)
    assert (twin._tip - twin._free == 7).all()             # free joint (7 coords), then the tip's 3 slides
    s0 = twin._start[0]
    assert np.allclose(s0["drive"], S.DRIVES["wedge"].velocity) and np.allclose(s0["tip_v"], s0["drive"])
    assert twin._tip_mass == pytest.approx(np.full(16, 0.3))     # the scene's tip: T12's measured 0.3 kg (WP13)


def test_scalars_take_the_deepest_world() -> None:
    twin = S.SqueezeTwin()
    twin.worlds = list(twin.conditions(True))[:3]
    log = DeviceTwinLog(3, 4, S.DT, "cpu")
    log.accumulator("peak").assign(np.array([-2e-4, -1.5e-3, -5e-4]))
    log.channel("depth").assign(np.full((3, log.n_samples), -1e-4, np.float32))
    out = twin.scalars(log)
    assert out["peak_pen_m"] == pytest.approx([1.5e-3])
    assert twin.monitors["over_1mm_fraction"] == pytest.approx(1 / 3) and twin.monitors["max_mm"] == pytest.approx(1.5)
    assert "delta_exp_mm" not in twin.monitors                    # no model bound yet: no expected depth


def test_expected_depth_is_the_rigid_tip_in_the_pair_stiffness() -> None:
    """delta_exp = sqrt(m v^2 / K_pair): 1 kg at 1.27 m/s on D22's 1e6 N/m pair is 1.27 mm; 0.3 kg, sqrt(0.3) x."""
    out = S.expected_depth(np.array([2.0, 0.5]), np.array([1.0, 0.3]), [(1.27, 0.0, 0.0), (0.0, 1.27, 0.0)], ["a", "b"])
    assert out["delta_exp_mm"]["a"] == pytest.approx(1.27, rel=1e-3)
    assert out["delta_exp_mm"]["b"] == pytest.approx(1.27 * np.sqrt(0.3), rel=1e-3)
    assert out["ratio_worst"] == pytest.approx(2.0 / 1.27, rel=1e-3) and out["tip_mass_kg"] == 1.0
    assert out["k_pair_n_per_m"] == pytest.approx(1e6) and out["k_pair_source"].startswith("default")


@pytest.mark.law
@pytest.mark.compile_heavy
def test_wedge_runs_on_cpu_and_reports_every_world() -> None:
    spec = dataclasses.replace(S.LAW_WEDGE, quick={"worlds": 2, "duration_s": 0.1})
    twin = S.SqueezeTwin()
    twin.conditions = lambda quick: tuple(S.SqueezeTwin.conditions(twin, quick))[:2]  # type: ignore[method-assign]
    res = run_law(spec, twin, None, device="cpu", quick=True)
    (v,) = res.verdicts
    assert v.verdict in ("pass", "fail") and res.monitors["finite"]
    assert set(res.monitors["peak_mm"]) == {"wedge_000", "wedge_001"} and np.isfinite(v.error)
    assert set(res.monitors["delta_exp_mm"]) == {"wedge_000", "wedge_001"} and res.monitors["ratio_worst"] > 0.0
