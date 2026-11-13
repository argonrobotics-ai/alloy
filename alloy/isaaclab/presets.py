# SPDX-License-Identifier: Apache-2.0
"""``physics=alloy`` on any Isaac Lab task that offers ``physics=newton_mjwarp``, with no edit to the task.

Isaac Lab tasks list their physics choices as fields of a ``PresetCfg`` (``newton_mjwarp``, ``physx``, ...), and other
settings can be keyed by the same names (``armature = preset(default=0.0, newton_mjwarp=0.01)``). :func:`install`
teaches Isaac Lab's preset resolver one more name: where a preset has no ``alloy`` field but has ``newton_mjwarp``,
``alloy`` resolves to that value, with a ``NewtonCfg`` whose MuJoCo Warp solver settings become
:class:`~alloy.isaaclab.cfg.AlloySolverCfg` (the same fields). So ``physics=alloy`` runs the task's own Newton setup
on Alloy, and every setting the task tuned for MuJoCo Warp follows.

Installed by the ``isaaclab.tasks`` entry point (Isaac Lab's ``isaaclab train`` / ``play`` / ``zero_agent`` load it
before resolving a task) and by ``import alloy.isaaclab`` in your own script.
"""

from __future__ import annotations

import copy
import functools
import logging
from typing import Any

from ._require import NEWTON_SOLVER_API, require

log = logging.getLogger("alloy.isaaclab")

ALLOY = "alloy"
SOURCE = "newton_mjwarp"
_MARK = "_alloy_preset"


def alloy_physics(newton_cfg: Any) -> Any:
    """A copy of a ``NewtonCfg`` whose ``MJWarpSolverCfg`` became ``AlloySolverCfg`` (fields kept).

    Raises ``ValueError`` for a Newton solver other than MuJoCo Warp: Alloy's rigid route replaces that one only.
    """
    require("alloy_physics")
    from isaaclab_newton.physics import MJWarpSolverCfg

    from .cfg import AlloySolverCfg

    old = newton_cfg.solver_cfg
    if isinstance(old, AlloySolverCfg):
        return newton_cfg
    if not isinstance(old, MJWarpSolverCfg):
        raise ValueError(f"Alloy replaces Newton's MuJoCo Warp solver; this physics config uses "
                         f"{type(old).__name__}. Pick physics=newton_mjwarp's config to switch from.")
    fields = {k: copy.deepcopy(getattr(old, k)) for k in old.__dataclass_fields__ if k not in ("class_type", "solver_type")}
    new_cfg = copy.deepcopy(newton_cfg)
    new_cfg.solver_cfg = AlloySolverCfg(**fields)
    if not NEWTON_SOLVER_API:   # Isaac Lab 3.0 mirrors the solver's manager into NewtonCfg; main keeps its manager
        new_cfg.class_type = new_cfg.solver_cfg.class_type
    return new_cfg


def _is_mjwarp(value: Any) -> bool:
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

    from .cfg import AlloySolverCfg

    return (isinstance(value, NewtonCfg) and isinstance(value.solver_cfg, MJWarpSolverCfg)
            and not isinstance(value.solver_cfg, AlloySolverCfg))


def _children(cfg: Any):
    for name in getattr(cfg, "__dataclass_fields__", {}):
        yield name, getattr(cfg, name, None)


def _swap_nested(cfg: Any, seen: set[int]) -> None:
    """Every MuJoCo Warp ``NewtonCfg`` inside ``cfg`` (a copy) becomes Alloy's, at any depth."""
    if id(cfg) in seen:
        return
    seen.add(id(cfg))
    for name, child in _children(cfg):
        if _is_mjwarp(child):
            setattr(cfg, name, alloy_physics(child))
        elif hasattr(child, "__dataclass_fields__"):
            _swap_nested(child, seen)


def _alloy_value(value: Any) -> Any:
    """What ``alloy`` resolves to where a preset offers only ``newton_mjwarp``: that value, with every MuJoCo Warp
    ``NewtonCfg`` in it (the value itself, or one inside a whole ``SimulationCfg``) on Alloy."""
    if _is_mjwarp(value):
        return alloy_physics(value)
    if hasattr(value, "__dataclass_fields__"):
        value = copy.deepcopy(value)
        _swap_nested(value, set())
    return value


def stock_mujoco_warp_left(cfg: Any, path: str = "env", seen: set[int] | None = None) -> list[str]:
    """Paths in a resolved config where a MuJoCo Warp ``NewtonCfg`` is still Newton's own (empty: all on Alloy)."""
    seen = set() if seen is None else seen
    if id(cfg) in seen:
        return []
    seen.add(id(cfg))
    left = []
    for name, child in _children(cfg):
        if _is_mjwarp(child):
            left.append(f"{path}.{name}")
        elif hasattr(child, "__dataclass_fields__"):
            left += stock_mujoco_warp_left(child, f"{path}.{name}", seen)
    return left


def _requested(overrides: Any) -> bool:
    import sys

    args = sys.argv[1:] if overrides is None else overrides
    for arg in args:
        key, _, val = str(arg).partition("=")
        if key.lstrip("-") in ("physics", "presets") and ALLOY in (v.strip() for v in val.split(",")):
            return True
    return False


def _add_alloy_field(preset_obj: Any, fields_of: Any) -> None:
    cls = type(preset_obj)
    fields = fields_of(preset_obj)
    if ALLOY in fields or SOURCE not in fields:
        return
    setattr(cls, ALLOY, _alloy_value(fields[SOURCE]))  # class level: Isaac Lab's resolver reads class attributes
    log.info("physics=alloy: %s takes its %s value", cls.__name__, SOURCE)


def _unknown_alloy_error(err: ValueError) -> ValueError:
    return ValueError(f"physics=alloy works on tasks that offer physics=newton_mjwarp (Alloy replaces Newton's MuJoCo "
                      f"Warp solver), and this task does not. Its own error: {err}")


def install() -> bool:
    """Teach Isaac Lab's preset resolver the ``alloy`` name; True if it was installed now, False if already there.

    Raises ``FeatureUnavailableError`` (with the install line) when Isaac Lab is not installed, ``ImportError`` when
    Isaac Lab is there without its task package.
    """
    require("install")
    from isaaclab_tasks.utils import hydra

    if getattr(hydra._pick_alternative, _MARK, False):
        return False
    pick, register = hydra._pick_alternative, hydra.register_task

    @functools.wraps(pick)
    def pick_with_alloy(preset_obj, selected, *args, **kwargs):
        explicit = kwargs.get("explicit_name", args[1] if len(args) > 1 else None)
        if ALLOY in selected or explicit == ALLOY:
            _add_alloy_field(preset_obj, hydra._preset_fields)
        return pick(preset_obj, selected, *args, **kwargs)

    @functools.wraps(register)
    def register_with_alloy(*args, **kwargs):
        try:
            out = register(*args, **kwargs)
        except ValueError as err:
            if f"'{ALLOY}'" in str(err) or f" {ALLOY}" in str(err):
                raise _unknown_alloy_error(err) from err
            raise
        overrides = kwargs.get("overrides", args[3] if len(args) > 3 else None)
        left = stock_mujoco_warp_left(out[0]) if _requested(overrides) else []
        if left:   # never run Newton's own solver when Alloy was asked for
            raise ValueError(f"physics=alloy did not reach {', '.join(left)}: that config still names Newton's MuJoCo "
                             f"Warp solver. Set it with alloy.isaaclab.alloy_physics(...) in Python, or report the task.")
        return out

    setattr(pick_with_alloy, _MARK, True)
    hydra._pick_alternative = pick_with_alloy
    hydra.register_task = register_with_alloy
    return True
