# SPDX-License-Identifier: Apache-2.0
"""R6 on measured fabrics (SCR-WP2-14, ruling 2026-11-08 14:59Z): the Petrak 2021 rig dataset, its sourced u95 and
the production-route twin (``alloy.scene.twins.drape_route``). The full verdict is an L4 run (docs/notes/wp2-results.md);
here the dataset is checked against the pack it is written from, and a CPU smoke runs the twin's plumbing."""

from __future__ import annotations

import json
import math

import pytest

from alloy.bench import compare as C
from alloy.bench.law import discover, registered_laws
from alloy.bench.runner import run_law
from alloy.scene.twins import drape_route as DR
from alloy.scene.twins import fabric_rig as FR


def _pack() -> dict:
    return json.loads(DR.PACK.read_text())


def test_the_committed_dataset_is_the_pack():
    """bench/rigs/R6.cloth_drape/petrak2021-v1 holds each fabric's KES inputs and measured DC, as the pack states."""
    data, pack = DR.dataset(), _pack()
    assert [c.id for c in data.conditions] == [f["id"] for f in pack["fabrics"]]
    for cond, fab in zip(data.conditions, pack["fabrics"]):
        assert dict(cond.params) == {k: float(fab[k]) for k in DR.FABRIC_KEYS}
    assert {t.condition_id: t.scalars["drape_coefficient"] for t in data.split("valid")} == {
        f["id"]: f["dc_measured"] for f in pack["fabrics"]}
    assert DR._pack_of(data)["rig"] == {k: pack["rig"][k] for k in ("specimen_diameter_m", "support_diameter_m")}


def test_the_u95_is_sourced_and_lets_the_rig_judge():
    """u95 = t95(4) x the largest 5-specimen SD (0.98 % DC) / sqrt(5) = 0.012: the 0.05 bar is 4.1 u95 (TUR >= 3)."""
    inst = DR.dataset().card.instruments["drape_coefficient"]
    assert inst["u95"] == pytest.approx(0.0122, abs=5e-5) and "Toshikj" in inst["source"] and "Matsudaira" in inst["note"]
    bound = DR.LAW.metrics["drape_coefficient"].bound(0.3)
    q = C.summarize("dc", "1", [0.3], inst["u95"])
    assert bound / q.u95 == pytest.approx(4.1, abs=0.05) and bound / q.u95 >= DR.LAW.min_tur


def test_without_a_source_the_dataset_says_so():
    inst = DR.measured_dataset(_pack(), u95=None).card.instruments["drape_coefficient"]
    assert "u95" not in inst and inst["note"] == DR.U95_LABEL


def test_r6_is_registered_with_the_route_twin():
    discover(("alloy.scene.laws",))
    factory = {s.id: f for s, f in registered_laws("rig")}["R6.cloth_drape"]
    assert factory is DR.DrapeRouteTwin


def test_the_report_names_the_routes_solve():
    rec = DR.DrapeRouteTwin().settings_record("vbd")
    assert rec["route"] == "cloth_vbd" and rec["kwargs"]["iterations"] == 60 and rec["protocol"]["seed"] == 7


def test_the_full_protocol_mesh_is_clean_for_every_fabric():
    """The R6 mesher at 8 mm: seed 0 starts F9 (2.42 mm sheet) in self-contact; the protocol's seed is clean."""
    worlds, edge, _, seed = DR.PROTOCOL[False]
    rigs = FR.rigs_for(_pack(), FR.HEADLINE)
    assert FR.check_mesh(FR.specimen_mesh(0.15, edge, seed), rigs) > max(FR.thickness(r.fab) for r in rigs)
    with pytest.raises(ValueError, match="self-contact"):
        FR.check_mesh(FR.specimen_mesh(0.15, edge, 0), rigs)
    assert worlds == 8


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_the_twin_runs_the_route_on_cpu_quick():
    """Plumbing on CPU (3 cm cells, 0.3 s: no verdict is meant): every fabric gets a DC verdict on the CLOTH_VBD
    route, node counts are not_run (the source has none), and the run is finite."""
    res = run_law(DR.LAW, DR.DrapeRouteTwin(), DR.dataset(), device="cpu", quick=True)
    assert "error" not in res.monitors, res.monitors.get("error")
    names = {v.name: v for v in res.verdicts}
    assert {f"drape_coefficient@{c}" for c in ("petrak2021_F1", "petrak2021_F4", "petrak2021_F9")} <= set(names)
    assert all(v.verdict == "not_run" for k, v in names.items() if k.startswith("node_count"))
    assert all(r["route"] == "cloth_vbd" for r in res.monitors["rows"]) and res.monitors["finite"]
    assert all(math.isfinite(v.sim.mean) for k, v in names.items() if k.startswith("drape_coefficient"))
