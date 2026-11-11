# SPDX-License-Identifier: Apache-2.0
"""Every scene in the card library (WP1 demos, WP13 task scenes) assembles through the real CardLibrary from an
unrelated working directory (asset paths resolve against the library root) and steps two control periods."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.core.errors import AssemblyError
from alloy.core.registry import isolated_registry
from alloy.route.router import Router
from alloy.scene.assembler import SceneAssembler
from alloy.sim import Simulation
from alloy.testing import scene as S

# scenes the assembler refuses by design (none since WP1 0e3c0cb moved demo_cloth_drop's spring to two sheets)
UNBUILDABLE: dict[str, str] = {}
PASS = {"sag_err": -6e-4, "drop_err": -0.03, "period_err": -0.003, "finite": 1.0, "passed": 1.0}


def _lib():
    try:
        from alloy.cards.library import CardLibrary
    except ImportError:
        pytest.skip("WP1 CardLibrary not merged")
    return CardLibrary(assets=None)      # Alloy's library only: DC asset cards are tested by their owners (WP1, SCR-DC-1)


def _names() -> list[str]:
    try:
        return sorted(_lib().names("scenes"))
    except Exception:  # noqa: BLE001 - collection must not fail when WP1 is absent
        return []


#: s per scene (compile_heavy's 180 s otherwise). demo_cloth_drop's cloth runs the 40-iteration cloth default since
#: 1de04ca01: 184.6 s on a loaded host-1 (post-merge run 20261108T1545Z), so it gets 600 s as WP3's and WP12's did
TIMEOUT_S = {"demo_cloth_drop": 600}


@pytest.mark.compile_heavy
@pytest.mark.parametrize("name", [pytest.param(n, marks=pytest.mark.timeout(TIMEOUT_S.get(n, 180))) for n in _names()])
def test_library_scene_assembles_and_steps(name, tmp_path, monkeypatch, request):
    lib = _lib()
    card = lib.scene(name)
    if name in UNBUILDABLE:
        request.applymarker(pytest.mark.xfail(raises=AssemblyError, strict=True, reason=UNBUILDABLE[name]))
    monkeypatch.chdir(tmp_path)
    with isolated_registry(empty=True):
        built = SceneAssembler(lib, Router(rod_self_test=lambda rod, h: dict(PASS))).assemble(
            card, "cpu", world_count=2)
        sim = Simulation(built)
        for _ in range(2):
            sim.step_period()
        finite = bool(np.isfinite(sim.state.body_q.numpy()).all()) if built.model.body_count else True
        if built.model.particle_count:
            finite = finite and bool(np.isfinite(sim.state.particle_q.numpy()).all())
    S.record("library_scene", float(finite), "pass" if finite else "fail", scene=name,
             entries=[e.name for e in built.entries], coupler=built.coupler.kind)
    assert finite
