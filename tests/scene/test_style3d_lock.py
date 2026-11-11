# SPDX-License-Identifier: Apache-2.0
"""Style3D's lock warning (alloy.scene.style3d_lock; DC5 2026-11-08): a card stiff in stretch for its step (PD diagonal
over m/dt² above LOCK_RATIO) drapes slowly or stays nearly flat with health OK, and the build says so. A KES woven card
(D03M's arm A, F1: about 9,700 at 8 mm cells, 4,300 at 12 mm; the ratio grows as 1 / edge²) warns;
cotton_woven_aniso (about 70) does not."""

from __future__ import annotations

import json
import logging

import alloy
from alloy.scene import style3d_lock as L
from alloy.scene.twins import drape_route as DR
from alloy.scene.twins import fabric_rig as FR


def _ratio(sim) -> float:
    st = sim.engine.stepper
    return L.lock_ratio(st.solver, sim.engine.model, st.dt_x)


def test_a_kes_woven_card_warns(tmp_path, caplog):
    pack = json.loads(DR.PACK.read_text())
    rigs = FR.rigs_for(pack, "A", "F1")
    root = FR.write_cards(tmp_path, rigs, pack["id"])
    scene = FR.make_scene(rigs, 1, "default", root, FR.specimen_mesh(0.15, 0.012, 0), 0.09, 0, False)
    with caplog.at_level(logging.WARNING, logger="alloy.scene"):
        sim = alloy.build(scene, device="cpu")
    assert sim.routes()[rigs[0].name]["route"] == "cloth_style3d" and _ratio(sim) > 2 * L.LOCK_RATIO
    assert any("stiff in stretch" in r.getMessage() for r in caplog.records)
    sim.close()


def test_a_soft_anisotropic_card_does_not_warn(caplog):
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_cloth("napkin", size=(0.3, 0.3), pos=(0.0, 0.0, 0.1), resolution=8, material="cotton_woven_aniso")
    with caplog.at_level(logging.WARNING, logger="alloy.scene"):
        sim = alloy.build(scene, device="cpu")
    assert sim.routes()["napkin"]["route"] == "cloth_style3d" and _ratio(sim) < L.LOCK_RATIO
    assert not any("stiff in stretch" in r.getMessage() for r in caplog.records)
    sim.close()


def test_without_a_pd_matrix_or_a_step_there_is_nothing_to_judge():
    assert L.lock_ratio(object(), None, 1e-3) == 0.0
    assert L.lock_ratio(type("S", (), {"pd_diags": object()})(), None, 0.0) == 0.0
