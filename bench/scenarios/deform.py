# SPDX-License-Identifier: Apache-2.0
"""Deformables family (cable, cloth, volumetric soft): D01-D06 of bench/SCENARIOS.md section 1.3.

    python -m bench.run --family deform --device cpu --ladder 1 --horizon 0.2      # Mac CPU smoke
    python -m bench.run --family deform --scenario D03_cloth_drape_cusick --device cuda:0 --ladder 1,16,256

Scenes live in helper modules (SCENARIOS.md section "Ownership" allows bench/scenarios/<key>_*.py):
deform_cable.py (D01, D02), deform_cloth.py (D03, D04, D05), deform_soft.py (D06); shared device schedules in
deform_common.py, analytic laws in deform_laws.py, meshes and measures in deform_geom.py. Notes and known problems:
bench/notes/deform.md.
"""
from __future__ import annotations

from bench.scenarios.deform_cable import SCENARIOS_CABLE
from bench.scenarios.deform_cloth import SCENARIOS_CLOTH
from bench.scenarios.deform_soft import SCENARIOS_SOFT

SCENARIOS = [*SCENARIOS_CABLE, *SCENARIOS_CLOTH, *SCENARIOS_SOFT]
