# SPDX-License-Identifier: Apache-2.0
"""WP9 liquid laws at the default 4 mm spacing (r = 2 mm; SCR-WP9-LIQRES, user decision 2026-11-08: 4 mm default,
2 mm the opt-in "super"). The registered A13 / A14 rows run at r = 1 mm (2 mm spacing); here the same twins, specs and
tolerances run at r = 2 mm: the quick conditions on the CPU, the full ones (SPEC §5.3's spacing counts, and the
r = 1 mm full scene's metres with half the spacings) on a GPU. A13.resting_jitter and R7 already run at r = 2 mm."""

from __future__ import annotations

import pytest

from alloy.bench.runner import run_law
from alloy.solvers.sph import laws as L

R_DEFAULT = 2.0e-3      # m: 4 mm spacing, liquid_accuracy "default"


@pytest.mark.slow
@pytest.mark.timeout(1200)
@pytest.mark.law
def test_a13_quick_cpu_at_4mm():
    """A13 hydrostatic (a 4 x 4 x 50 column) and Archimedes (draft 5 spacings) at 4 mm spacing: SPEC's 3 % / 5 %."""
    hydro = run_law(L.A13_HYDROSTATIC, L.HydrostaticTwin(L.HYDRO_DT, r=R_DEFAULT), None, device="cpu", quick=True)
    assert hydro.verdict == "pass", hydro.verdicts
    draft = run_law(L.A13_ARCHIMEDES, L.ArchimedesTwin(L.DRAFT_DT, r=R_DEFAULT), None, device="cpu", quick=True)
    assert draft.verdict == "pass", (draft.verdicts, draft.monitors)
    assert draft.monitors["draft_over_2r"][0] == pytest.approx(5.0)


@pytest.mark.slow
@pytest.mark.timeout(1200)
@pytest.mark.law
def test_a14_quick_cpu_at_4mm():
    """A14 dam break at 4 mm spacing (a = 12 spacings, 48 mm): no front faster than Ritter's."""
    res = run_law(L.A14_DAM_BREAK, L.DamBreakTwin(L.DAM_DT, r=R_DEFAULT), None, device="cpu", quick=True)
    assert res.verdict == "pass", res.verdicts


@pytest.mark.gpu
@pytest.mark.law
@pytest.mark.timeout(3600)
def test_laws_at_4mm_full_gpu():
    """Full mode at 4 mm: SPEC §5.3's spacing counts must pass. The same-size rows (the r = 1 mm full scenes' metres,
    half the spacings: H/(2r) = 25, draft/(2r) = 5, a = 25) are measured and written, not judged: SPEC's bounds ask
    for H/(2r) >= 50 and draft/(2r) >= 10."""
    import warp as wp

    from alloy.testing.sph import write_result

    if not wp.is_cuda_available():
        pytest.skip("needs CUDA")
    out = {}
    for same in (False, True):
        for spec, twin in ((L.A13_HYDROSTATIC, L.HydrostaticTwin(L.HYDRO_DT, r=R_DEFAULT, same_size=same)),
                           (L.A13_ARCHIMEDES, L.ArchimedesTwin(L.DRAFT_DT, r=R_DEFAULT, same_size=same)),
                           (L.A14_DAM_BREAK, L.DamBreakTwin(L.DAM_DT, r=R_DEFAULT, same_size=same))):
            res = run_law(spec, twin, None, device="cuda:0", quick=False)
            out[f"{spec.id}{twin.tag()}"] = {
                "verdict": res.verdict, "wall_s": res.wall_s, "monitors": res.monitors,
                "metrics": {v.name: {"verdict": v.verdict, "error": v.error, "bound": v.bound} for v in res.verdicts}}
    write_result("wp9_laws_4mm_gpu", out)
    assert all(v["verdict"] != "error" for v in out.values()), out
    spec_rows = {k: v["verdict"] for k, v in out.items() if not k.endswith("_samesize")}
    assert all(v == "pass" for v in spec_rows.values()), spec_rows
