# SPDX-License-Identifier: Apache-2.0
"""Soil laws on CPU (alloy.tasks.soil_laws, docs/notes/soil-design.md).

The cohesion and bulk-density mapping is tested in tests/scene/test_mpm_material.py. L1: a free-standing column
stands below H_c = 2 c tan(45° + φ/2) / (ρ g) and slumps above it. Measured (CPU, c 400 Pa): φ 30° stands at 88 mm
(H_c 88.3 mm) and slumps 3.4 / 16.9 mm at 100 / 115 mm; φ 40° stands at 110 mm (H_c 109 mm), slumps 4.3 mm at 130 mm.
L3: a rigid strip footing on φ = 0 soil rests under Prandtl's q_ult = (π + 2) c and sinks above it.
"""

from __future__ import annotations

import pytest

from alloy.tasks import soil_laws as L

CMD = "PYTHONPATH=. python -m pytest tests/tasks/test_soil_laws.py -m 'slow or not slow'"


def test_library_soil_cards_route_to_grain_mpm_with_cohesion() -> None:
    from alloy.cards.library import CardLibrary
    from alloy.scene.emitters.deformable import mpm_values

    lib = CardLibrary()
    for name, c in (("soil_sand_dry", None), ("soil_sand_damp", 4850.0), ("soil_loam_moist", 24000.0),
                    ("soil_clay_loam", 30000.0), ("soil_hardpan", 1.0e5)):
        mat = lib.material(name)
        assert mat.plastic.friction_angle is not None, name
        assert mpm_values(mat).get("yield_stress") == (pytest.approx(c) if c else None), name


@pytest.mark.slow
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("ratio, stands", [(0.85, True), (1.3, False)])
def test_l1_free_column_stands_below_hc(ratio: float, stands: bool) -> None:
    from alloy.testing.integration import record

    soil = L.Soil(30.0, 400.0)
    height = ratio * L.column_hc(soil)
    p0, p1, _ = L.run(*L.column_scene(soil, height), seconds=0.4)
    drop = L.top_drop(p0, p1)
    record("SOIL-L1", drop, "pass" if (drop < 1e-3) == stands else "fail", CMD, h_over_hc=ratio,
           hc=L.column_hc(soil), phi_deg=soil.phi_deg, cohesion=soil.cohesion)
    if stands:
        assert drop < 1e-3, f"a column at {ratio} H_c fell {drop * 1e3:.2f} mm"
    else:
        assert drop > L.VOXEL, f"a column at {ratio} H_c fell only {drop * 1e3:.2f} mm"


@pytest.mark.slow
@pytest.mark.timeout(2400)
@pytest.mark.parametrize("ratio, rests", [(0.8, True), (1.25, False)])
def test_l3_strip_footing_rests_below_prandtl(ratio: float, rests: bool) -> None:
    """Measured (CPU, c 400 Pa, q_ult 2,057 Pa, 0.5 s): 0.01 mm at 0.8 q_ult, 19.6 mm at 1.25 q_ult."""
    from alloy.testing.integration import record

    soil = L.Soil(0.0, 400.0)
    sink = L.footing_sink(*L.footing_scene(soil, ratio * L.prandtl_qult(soil)), seconds=0.5)
    record("SOIL-L3", sink, "pass" if (sink < 1e-3) == rests else "fail", CMD, p_over_qult=ratio,
           qult=L.prandtl_qult(soil), cohesion=soil.cohesion)
    if rests:
        assert sink < 1e-3, f"a footing at {ratio} q_ult sank {sink * 1e3:.2f} mm"
    else:
        assert sink > L.VOXEL, f"a footing at {ratio} q_ult sank only {sink * 1e3:.2f} mm"
