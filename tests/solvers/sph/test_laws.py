# SPDX-License-Identifier: Apache-2.0
"""WP9 laws (SPEC §5.3 A13-A14) and the R7 pour twin on synthetic data (SPEC §3.8 item 7), CPU quick runs."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.bench.law import discover, missing_laws, registered_laws
from alloy.bench.runner import run_law
from alloy.solvers.sph import laws as L
from alloy.solvers.sph.pour import PourGeometry, onset_angle_static, quasi_static_pour, retained_volume
from alloy.testing.sph import synthetic_pour_rig

SMALL_CUP = PourGeometry(cup_x=0.04, cup_y=0.04, cup_h=0.05, liquid_mass=0.032, pivot_z=0.15, gap=0.02, drop=0.04,
                         recv_x=0.1, recv_y=0.1, recv_h=0.06)


def test_laws_registered_and_discoverable():
    status = discover(("alloy.solvers.sph.laws",))
    assert status["alloy.solvers.sph.laws"] is None
    ids = {spec.id for spec, _ in registered_laws()}
    assert {"A13.hydrostatic", "A13.archimedes", "A13.resting_jitter", "A14.dam_break", "R7.pour"} <= ids
    gaps = {e.prefix for e, _ in missing_laws()}
    assert not {"A13", "A14", "R7"} & gaps
    for spec in (L.A13_HYDROSTATIC, L.A13_ARCHIMEDES, L.A13_RESTING_JITTER, L.A14_DAM_BREAK, L.R7_POUR):
        assert spec.solvers == ("sph",) and spec.producer == "WP9"
        assert math.isclose(0.02 / spec.dt, round(0.02 / spec.dt))  # D4 grid


def test_quasi_static_pour_model():
    g = SMALL_CUP
    on = onset_angle_static(g)
    v0 = g.liquid_mass / g.density
    assert retained_volume(g, np.array([on]))[0] == pytest.approx(v0, rel=1e-9)
    assert retained_volume(g, np.array([on - 0.01]))[0] > v0 > retained_volume(g, np.array([on + 0.01]))[0]
    data = synthetic_pour_rig(g, rates=(40.0,), theta_max_deg=110.0, duration_s=4.0, noise_kg=0.0, noise_deg=0.0)
    trial = data.split("valid")[0]
    assert trial.scalars["onset_angle"] == pytest.approx(math.degrees(on))
    assert trial.series["m_poured"][-1] == pytest.approx(g.liquid_mass)  # 110 deg empties a box cup
    curve = quasi_static_pour(g, data.conditions[0], trial.series["t"])
    assert np.all(np.diff(curve) >= -1e-12)


def _verdicts(res) -> dict:
    return {v.name: v for v in res.verdicts}


@pytest.mark.slow
@pytest.mark.timeout(1200)
@pytest.mark.law
def test_a13_quick_cpu():
    hydro = run_law(L.A13_HYDROSTATIC, L.HydrostaticTwin(L.HYDRO_DT), None, device="cpu", quick=True)
    assert hydro.verdict == "pass", hydro.verdicts
    draft = run_law(L.A13_ARCHIMEDES, L.ArchimedesTwin(L.DRAFT_DT), None, device="cpu", quick=True)
    assert draft.verdict == "pass", (draft.verdicts, draft.monitors)
    assert draft.monitors["draft_over_2r"][0] == pytest.approx(5.0)


@pytest.mark.slow
@pytest.mark.timeout(1800)
@pytest.mark.law
def test_resting_jitter_quick_cpu():
    """SCR-SC-C2: settled water in an 8 x 8 glass at 32 mm depth, default settings, 4 s (2 s settle): velocity RMS
    <= 3 mm/s and no growth; the resting-jitter rule tightened the density tolerance below max_density_error, without
    the density solve reaching its iteration cap; the divergence residual stays within its tolerance."""
    res = run_law(L.A13_RESTING_JITTER, L.RestingJitterTwin(L.JITTER_DT), None, device="cpu", quick=True)
    mon = res.monitors
    assert res.verdict == "pass", (res.verdicts, mon)
    assert mon["density_tol"][0] < 0.5 * 1.0e-3, mon
    assert mon["density_at_cap"][0] == 0.0 and 2.0 <= mon["density_iterations"][0] < mon["max_iterations"], mon
    assert 0.0 <= mon["divergence_residual"][0] <= 1.0e-3, mon


@pytest.mark.slow
@pytest.mark.timeout(1200)
@pytest.mark.law
def test_a14_quick_cpu_front_below_ritter():
    res = run_law(L.A14_DAM_BREAK, L.DamBreakTwin(L.DAM_DT), None, device="cpu", quick=True)
    assert res.verdict == "pass", res.verdicts
    T, Z = np.array(res.monitors["front_T"][0]), np.array(res.monitors["front_Z"][0])
    assert Z[0] == pytest.approx(1.0, abs=0.05) and np.all(np.diff(Z) >= -1e-3)  # starts at the dam, advances
    assert Z[T <= 3.0][-1] > 2.0  # and has really surged


@pytest.mark.slow
@pytest.mark.timeout(1200)
@pytest.mark.law
def test_r7_twin_on_synthetic_data():
    """The R7 pipeline end to end on a quasi-static synthetic dataset: every (metric, condition) is judged, the
    liquid is conserved (cup + receiver + spill = initial), the cup empties, and the dynamic onset is not earlier than
    the quasi-static one by more than the tolerance (inertia only delays it)."""
    data = synthetic_pour_rig(SMALL_CUP, rates=(40.0,), theta_max_deg=110.0, duration_s=4.0)
    twin = L.PourTwin(L.POUR_DT)
    res = run_law(L.R7_POUR, twin, data, device="cpu", quick=True)
    v = _verdicts(res)
    assert set(v) == {"onset_angle@tilt40", "m_poured@tilt40", "poured_mass@tilt40", "spill@tilt40"}
    assert all(x.verdict in ("pass", "fail", "inconclusive") for x in v.values())
    assert res.monitors["finite"]
    m0 = res.monitors["liquid_mass_kg"][0]
    assert res.monitors["residual_in_cup_kg"][0] < 0.02 * m0
    onset = v["onset_angle@tilt40"].sim.mean
    static = math.degrees(onset_angle_static(SMALL_CUP))
    assert static - 1.5 <= onset <= static + 12.0
    poured = v["poured_mass@tilt40"].sim.mean
    assert 0.8 * m0 < poured <= m0 * (1 + 1e-6)


@pytest.mark.gpu
@pytest.mark.law
@pytest.mark.timeout(3600)
def test_a13_a14_full_gpu():
    """GPU acceptance: A13 / LIQ-4 at H/(2r) = 50 and draft/(2r) = 10 over 5 s, A14 front at a = 50 spacings."""
    import warp as wp

    from alloy.testing.sph import write_result

    if not wp.is_cuda_available():
        pytest.skip("needs CUDA")
    out = {}
    for spec, twin in ((L.A13_HYDROSTATIC, L.HydrostaticTwin(L.HYDRO_DT)), (L.A13_ARCHIMEDES, L.ArchimedesTwin(
            L.DRAFT_DT)), (L.A14_DAM_BREAK, L.DamBreakTwin(L.DAM_DT)),
            (L.A13_RESTING_JITTER, L.RestingJitterTwin(L.JITTER_DT))):
        res = run_law(spec, twin, None, device="cuda:0", quick=False)
        out[spec.id] = {"verdict": res.verdict, "wall_s": res.wall_s, "monitors": res.monitors,
                        "metrics": {v.name: {"verdict": v.verdict, "error": v.error, "bound": v.bound}
                                    for v in res.verdicts}}
    write_result("wp9_laws_gpu", out)
    assert all(v["verdict"] != "error" for v in out.values())
