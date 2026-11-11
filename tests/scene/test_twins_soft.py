# SPDX-License-Identifier: Apache-2.0
"""WP2 rig-law twins R5 (squeeze), R6 (drape), R8 (granular) on their synthetic datasets, CPU quick (SPEC §6.4).

Closed-form pieces are checked without simulation; each law's `run_law` quick run is `compile_heavy` and runs once
per module (fixtures), mirroring tests/bench/test_laws_cpu.py for R3/R4.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.bench import compare as C
from alloy.bench.runner import run_law
from alloy.scene.twins import drape, granular, squeeze

pytestmark = pytest.mark.law


def _min_tur(law, data) -> float:
    """Smallest bound / U_real over (metric, condition) on the valid split (bench §4.3 TUR rule)."""
    out = math.inf
    for cond in data.conditions:
        for metric, tol in law.metrics.items():
            vals = [t.scalars[metric] for t in data.split("valid") if t.condition_id == cond.id]
            q = C.summarize(metric, "", vals, data.card.instruments[metric]["u95"])
            out = min(out, tol.bound(q.mean) / q.u95)
    return out


def _by_name(result) -> dict:
    return {v.name: v for v in result.verdicts}


# --- R5 squeeze -----------------------------------------------------------------------------------------------------

def test_r5_closed_form_limits() -> None:
    spec = squeeze.SqueezeSpec.of(squeeze.CONDITIONS[0])
    elastic = squeeze.law_metrics(spec, 30.0e3, 0.0)
    assert elastic["hysteresis"] == pytest.approx(0.0, abs=1e-9)
    # small strain: F = E·A·ε within the 10 % strain nonlinearity of the stable neo-Hookean (ν = 0.05)
    assert elastic["force_10"] == pytest.approx(30.0e3 * spec.area * 0.10, rel=0.03)
    assert elastic["slope"] == pytest.approx(30.0e3 * spec.area / spec.h, rel=0.03)
    damped = squeeze.law_metrics(spec, 30.0e3, 0.5)
    assert damped["force_10"] > elastic["force_10"] and 0.05 < damped["hysteresis"] < 0.5


def test_r5_fit_inverts_the_law_and_data_can_judge() -> None:
    spec = squeeze.SqueezeSpec.of(squeeze.CONDITIONS[1])
    m = squeeze.law_metrics(spec, 20.0e3, 0.6)
    youngs, beta = squeeze.fit_material(spec, m["force_10"], m["hysteresis"])
    assert youngs == pytest.approx(20.0e3, rel=1e-6) and beta == pytest.approx(0.6, rel=1e-6)
    data = squeeze.dataset()
    assert data.card.protocol_version == "synthetic/1" and len(data.split("valid")) == 20
    assert _min_tur(squeeze.LAW, data) >= 3.0


@pytest.fixture(scope="module")
def r5_run():
    twin = squeeze.SqueezeTwin()
    seen: list[str] = []
    bind = twin.bind_calib
    twin.bind_calib = lambda card, trials: (seen.extend(t.split for t in trials), bind(card, trials))[1]
    return run_law(squeeze.LAW, twin, squeeze.dataset(), device="cpu", quick=True), seen


@pytest.mark.compile_heavy
@pytest.mark.timeout(300)
def test_r5_twin_passes_on_synthetic_data(r5_run) -> None:
    result, seen = r5_run
    assert set(seen) == {"calib"}
    assert "error" not in result.monitors and len(result.verdicts) == 8
    assert max(result.monitors["imbalance"]) < 0.03  # top vs bottom plate reaction: quasi-static
    v = _by_name(result)
    assert all(x.verdict == "pass" for x in v.values()), {k: (x.error, x.bound, x.verdict) for k, x in v.items()}
    assert v["hysteresis@pu_foam_cube"].error < 0.01 and v["force_20@sponge"].error < 0.05 * v["force_20@sponge"].real.mean


# --- R8 granular ----------------------------------------------------------------------------------------------------

def test_r8_closed_forms_and_measurements() -> None:
    assert granular.lube_runout(1.0) == pytest.approx(1.24)
    assert granular.lube_runout(2.0) == pytest.approx(1.6 * math.sqrt(2.0))
    # a quarter cone of slope 30 deg sampled on a fine grid measures 30 deg; its 99.5 % radius is near the rim
    g = np.linspace(0.0, 0.1, 201)
    X, Y = np.meshgrid(g, g, indexing="ij")
    r = np.hypot(X, Y).ravel()
    z = np.clip(0.05 - r * math.tan(math.radians(30.0)), 0.0, None)
    cone = np.column_stack([X.ravel(), Y.ravel(), z])[r < 0.05 / math.tan(math.radians(30.0))]
    assert math.degrees(granular.repose_from_deposit(cone, 2e-3)) == pytest.approx(30.0, abs=0.5)
    assert granular.runout_from_deposit(cone, 0.025) == pytest.approx((0.0866 - 0.025) / 0.025, rel=0.02)
    assert _min_tur(granular.LAW, granular.dataset()) >= 3.0


@pytest.fixture(scope="module")
def r8_run():
    return run_law(granular.LAW, granular.GranularTwin(), granular.dataset(), device="cpu", quick=True)


@pytest.mark.slow  # R6/R8 runs take 46-93 s on CPU; their measured fails are documented
@pytest.mark.timeout(400)
def test_r8_twin_runs_and_measures(r8_run) -> None:
    assert "error" not in r8_run.monitors and r8_run.monitors["finite"]
    v = _by_name(r8_run)
    assert set(v) == {"repose_angle@sand", "runout@sand"}
    assert all(math.isfinite(x.sim.mean) and x.verdict in ("pass", "fail") for x in v.values())
    assert max(r8_run.monitors["vmax_end"]) < 0.05  # deposits settled


@pytest.mark.slow  # R6/R8 runs take 46-93 s on CPU; their measured fails are documented
@pytest.mark.timeout(400)
@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "measured CPU quick (5 mm voxels, 20x lift, APIC): repose 17.1 deg vs 32.9 deg (the 20x lift is a collapse, not "
    "quasi-static; 5x gave 24.7 deg), runout 1.60 vs Lube 2.28 at mu = tan(32.8 deg)"))
def test_r8_twin_verdict_on_cpu_quick(r8_run) -> None:
    assert r8_run.verdict == "pass"


# --- R6 drape -------------------------------------------------------------------------------------------------------

def test_r6_hinge_mapping_and_measurements() -> None:
    ell, rig = 0.01, 2.0e-5
    h = math.sqrt(3.0) / 2.0 * ell
    rest = np.array([[0.0, h, 0.0], [0.0, -h, 0.0], [-0.5 * ell, 0.0, 0.0], [0.5 * ell, 0.0, 0.0]])
    ke = drape.hinge_stiffness(rest, np.array([[0, 1, 2, 3], [-1, 1, 2, 3]]), rig)
    assert ke[0] == pytest.approx(4.0 * rig / (math.sqrt(3.0) * ell)) and ke[1] == 0.0
    verts, tris = drape.disc_mesh(0.012, seed=1)
    flat = np.column_stack([verts, np.zeros(len(verts))])
    assert drape.drape_coefficient(drape.silhouette_area(flat[:, :2], tris)) == pytest.approx(1.0, abs=0.01)
    shrunk = flat[:, :2] * (drape.R_SUPPORT / drape.R_SPECIMEN)
    assert drape.drape_coefficient(drape.silhouette_area(shrunk, tris)) == pytest.approx(0.0, abs=0.01)
    ang = np.linspace(-math.pi, math.pi, 120, endpoint=False)
    rim = (0.12 * (1.0 + 0.1 * np.cos(7 * ang)))[:, None] * np.column_stack([np.cos(ang), np.sin(ang)])
    assert drape.node_count(ang, rim) == 7


def test_r6_strip_reference_and_data() -> None:
    assert drape.strip_reach(0.5) == pytest.approx(1.0, abs=1e-3)
    # ASTM D1388: a strip of length 2c droops to a 41.5 deg chord; the elastica reach there is cos-consistent
    assert drape.strip_reach(1.96) == pytest.approx(0.742, abs=0.01)
    dcs = [drape.strip_drape_coefficient(drape.TRUTH[k]["bending_length"]) for k in ("knit", "cotton", "felt")]
    assert dcs == sorted(dcs) and 0.1 < dcs[0] and dcs[-1] < 1.0
    data = drape.synthetic_dataset()
    tol = drape.LAW.metrics["drape_coefficient"]
    for cond in data.conditions:
        vals = [t.scalars["drape_coefficient"] for t in data.split("valid") if t.condition_id == cond.id]
        q = C.summarize("dc", "1", vals, data.card.instruments["drape_coefficient"]["u95"])
        assert tol.bound(q.mean) / q.u95 >= 3.0
        assert all("node_count" not in t.scalars for t in data.trials)


@pytest.fixture(scope="module")
def r6_run():
    return run_law(drape.LAW, drape.DrapeTwin(), drape.synthetic_dataset(), device="cpu", quick=True)


@pytest.mark.slow  # R6/R8 runs take 46-93 s on CPU; their measured fails are documented
@pytest.mark.timeout(300)
def test_r6_twin_runs_and_measures(r6_run) -> None:
    assert "error" not in r6_run.monitors and r6_run.monitors["finite"]
    v = _by_name(r6_run)
    assert {k for k, x in v.items() if x.verdict == "not_run"} == {f"node_count@{c}" for c in ("cotton", "knit", "felt")}
    assert all(0.0 < d < 1.0 for d in r6_run.monitors["dc"]) and all(n >= 3 for n in r6_run.monitors["nodes"])


@pytest.mark.slow  # R6/R8 runs take 46-93 s on CPU; their measured fails are documented
@pytest.mark.timeout(300)
@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "measured CPU quick (VBD, 12 mm, 1.5 s): DC cotton 0.357 / knit 0.280 / felt 0.454 vs strip reference "
    "0.493 / 0.151 / 0.947, skirts still moving 0.3-0.8 m/s; VBD does not converge the calibrated hinge bending "
    "(ASTM strip hangs at 76-98 deg instead of 41.5 deg)"))
def test_r6_twin_dc_on_cpu_quick(r6_run) -> None:
    assert all(x.verdict == "pass" for x in r6_run.verdicts if x.name.startswith("drape_coefficient"))
