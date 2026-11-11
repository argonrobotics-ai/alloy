# SPDX-License-Identifier: Apache-2.0
"""Find the liquid (SPH) solvers of a built engine and read their particle flags in the model's numbering.

The SPH solver is not always where a single-solver scene puts it:
- with a rigid entry (MuJoCo) beside it, the coupler steps it on a compact view of the model, whose own
  ``particle_flags`` the emitters and the escape box update (the full model's copy keeps its build-time flags);
- with cloth and rigid bodies too, it runs as a one-way follower (``OneWayFollower``), not in ``engine.solvers``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def refuse_solid(material: str, scene_materials: Mapping[str, Any], roots: Sequence[str | Path]) -> None:
    """Raise SceneError for ``Scene.add_liquid`` / ``add_pour`` of a material that does not flow: a card with neither
    a ``fluid`` block (water, oil: SPH) nor a paste's or grains' ``plastic`` one (MPM particles; ``flows``). Such a
    material built one rigid or soft block named like the liquid, with no warning (AUDIT-STACK ERR-6). A scene
    material made ``like=`` a library card is followed to it; an unknown name is left to the scene's own check.
    Example: ``refuse_solid("steel_carbon", {}, [])`` raises; ``refuse_solid("water", {}, [])`` returns None."""
    import yaml  # noqa: PLC0415

    from alloy.api import catalog  # noqa: PLC0415
    from alloy.api.errors import SceneError  # noqa: PLC0415

    name, seen, card = material, set(), None
    while name in scene_materials and name not in seen:
        seen.add(name)
        name = scene_materials[name].get("like") or ""
    path = catalog.card_path("materials", name, roots) if name else None
    if path is not None:
        card = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    elif material not in scene_materials:
        return
    if isinstance(card, dict) and flows(card):
        return
    raise SceneError(f"add_liquid: {material!r} is a solid material (it does not flow), and add_liquid fills a shape "
                     "with liquid particles",
                     f"give it a solid shape instead, e.g. add_box(name, size=(x, y, z), material={material!r}); or "
                     "pick a liquid: [m['name'] for m in alloy.materials() if m['kind'] == 'fluid']",
                     field="material")


def flows(card: Mapping[str, Any]) -> bool:
    """Whether a material card's stuff flows: a ``fluid`` block, grains (``plastic.friction_angle``), or a paste (a
    ``plastic.yield_stress`` that is not a sheet's or a metal's: ``_cloth_route.film_reason`` None).
    Example: ``flows({"fluid": {"viscosity": 1e-3}})`` -> True."""
    from alloy.api._cloth_route import film_reason  # noqa: PLC0415

    plastic = card.get("plastic") or {}
    if card.get("fluid") or plastic.get("friction_angle") is not None:
        return True
    return plastic.get("yield_stress") is not None and film_reason(card) is None


def all_solvers(engine: Any) -> list[Any]:
    """Every solver of ``engine``: top-level entry solvers and one-way followers, each once.
    Example: ``all_solvers(sim.engine)`` -> ``[SolverMuJoCo, SolverDFSPH]``."""
    found: list[Any] = []
    leaves = [ls.leaf.solver for ls in getattr(getattr(engine, "stepper", None), "_leaves", ())]
    for s in [*getattr(engine, "solvers", ()), *leaves]:
        if all(s is not f for f in found):
            found.append(s)
    return found


def liquid_solvers(engine: Any) -> list[Any]:
    """Every SPH solver of ``engine`` (top-level solvers and one-way followers), each once.
    Example: ``liquid_solvers(sim.engine)`` -> ``[SolverDFSPH]``."""
    return [s for s in all_solvers(engine) if hasattr(s, "add_emitter") and hasattr(s, "report")]


def mpm_grids(engine: Any) -> list[dict[str, Any]]:
    """The grid each MPM solver uses, as recorded by the engine (``solver.alloy_grid``: ``grid_type`` "sparse" or
    "dense", ``budget_cells``, and ``fallback`` when a dense grid outgrew its budget and switched to sparse); a copy,
    empty without MPM. Example: ``mpm_grids(sim.engine)`` -> ``[{"grid_type": "sparse", ...}]``."""
    import copy  # noqa: PLC0415

    return [copy.deepcopy(s.alloy_grid) for s in all_solvers(engine)
            if isinstance(getattr(s, "alloy_grid", None), dict)]


def _local_to_global(engine: Any, solver: Any) -> np.ndarray | None:
    """Model particle index of each of ``solver``'s view particles, or None when the solver runs on the model."""
    if getattr(solver, "model", engine.model) is engine.model:
        return None
    for entry in getattr(getattr(engine, "stepper", None), "_entries", {}).values():
        if getattr(entry, "view", None) is solver.model:
            return entry.particle_local_to_global.numpy()
    return None


def particle_flags(engine: Any) -> np.ndarray:
    """The model's particle flags with each liquid solver's live flags (poured, retired) in the model's numbering.
    Example: ``particle_flags(sim.engine) & int(newton.ParticleFlags.ACTIVE)``."""
    flags = engine.model.particle_flags.numpy()
    for solver in liquid_solvers(engine):
        ids = _local_to_global(engine, solver)
        if ids is not None and len(ids):
            flags[ids] = solver.model.particle_flags.numpy()[: len(ids)]
    return flags


def unseen_kinematic(engine: Any) -> int:
    """Kinematic bodies a liquid cannot collide with: when the coupler steps SPH on its own model view, it hands the
    liquid the static walls and the bodies other entries own. Kinematic bodies join the rigid solver's entry, or with
    no rigid solver (cloth and liquid only) an entry of their own (``alloy.integration.kinematic_carrier``); one
    that no entry owns is a wall the liquid passes through. Example: ``unseen_kinematic(sim.engine)`` -> ``0``."""
    from alloy._vendor import newton  # noqa: PLC0415

    on_view = [s for s in liquid_solvers(engine) if _local_to_global(engine, s) is not None]
    flags = getattr(engine.model, "body_flags", None)
    if not on_view or flags is None or engine.model.body_count == 0:
        return 0
    kinematic = (flags.numpy() & int(newton.BodyFlags.KINEMATIC)) != 0
    owner = getattr(getattr(engine, "stepper", None), "_body_owner", None)
    if owner is not None:
        kinematic &= np.asarray(owner)[: kinematic.size] < 0
    return int(kinematic.sum())
