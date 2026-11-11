# SPDX-License-Identifier: Apache-2.0
"""SCR-WP8-8: a MuJoCo entry with precision pairs stays on MuJoCo Warp when the CPU default is the MuJoCo C route.

The C route (``alloy.route.mjc_core``, ``ALLOY_CPU_ROUTE=mjc``) cannot host appended precision rows, and its factory
cannot see them at build time, so a precision scene raised PrecisionSpecError at the MuJoCo contact binding. Wiring
now sets ``alloy:cpu_route = "warp"`` on such entries (``_mujoco_kwargs``) unless the entry chose a route itself.
"""

from __future__ import annotations

import pytest

from alloy.core.registry import isolated_registry
from alloy.testing import integration as I
from alloy.testing import scene as S

pytestmark = [pytest.mark.integration]


def _solver_names(sim) -> list[str]:
    return [type(s).__name__ for s in sim.solvers]


def test_plain_scene_takes_the_c_route(monkeypatch) -> None:
    """Control: without precision pairs the environment default reaches the factory."""
    from alloy.integration import build_simulation

    monkeypatch.setenv("ALLOY_CPU_ROUTE", "mjc")
    card, lib = S.box_on_table(worlds=1)
    with isolated_registry(empty=True):
        sim = build_simulation(I.assemble(card, lib, "cpu", 1), "standard", graph="off")
    assert "MjcCoreSolver" in _solver_names(sim)


def test_precision_entries_ask_for_mujoco_warp() -> None:
    """Wiring sets the key on every MuJoCo entry of a scene with precision pairs (SDF pairs need CUDA, so no CPU
    precision scene is built here); an entry's own choice stands."""
    import dataclasses
    from types import SimpleNamespace

    from alloy.integration.profiles import profile
    from alloy.integration.wiring import _mujoco_kwargs
    from alloy.scene.types import EntrySpec

    entry = EntrySpec("mujoco", "mujoco", (0,), (), (), (), 1, 1.0e-3, solver_kwargs={"nconmax": 8, "njmax": 32})
    built = SimpleNamespace(precision_pairs=(SimpleNamespace(ccd="off"),))
    for native in ("mujoco", "newton"):
        notes: list[str] = []
        out = _mujoco_kwargs(entry, built, native, profile("standard"), notes)
        assert out.solver_kwargs["alloy:cpu_route"] == "warp" and any("cpu_route=warp" in n for n in notes)
    own = dataclasses.replace(entry, solver_kwargs={**entry.solver_kwargs, "alloy:cpu_route": "mjc"})
    assert _mujoco_kwargs(own, built, "mujoco", profile("standard"), []).solver_kwargs["alloy:cpu_route"] == "mjc"


def test_the_warp_key_keeps_mujoco_warp_under_the_mjc_default(monkeypatch) -> None:
    """The factory honours the key the wiring sets, with ALLOY_CPU_ROUTE=mjc in the environment."""
    import dataclasses

    from alloy.integration import build_simulation

    monkeypatch.setenv("ALLOY_CPU_ROUTE", "mjc")
    card, lib = S.box_on_table(worlds=1)
    with isolated_registry(empty=True):
        built = I.assemble(card, lib, "cpu", 1)
        entries = [dataclasses.replace(e, solver_kwargs={**dict(e.solver_kwargs), "alloy:cpu_route": "warp"})
                   if e.capability == "mujoco" else e for e in built.entries]
        sim = build_simulation(dataclasses.replace(built, entries=entries), "standard", graph="off")
    from alloy._vendor import newton

    assert "MjcCoreSolver" not in _solver_names(sim)
    assert any(isinstance(s, newton.solvers.SolverMuJoCo) for s in sim.solvers), _solver_names(sim)
