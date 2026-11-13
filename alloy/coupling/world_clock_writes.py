# SPDX-License-Identifier: Apache-2.0
"""``alloy.build(scene, clock_worlds=n)`` and the writes that do not go through ``alloy.Sim`` (WP4's review of S2a
item 4, WP12's W1).

A resident world steps in the fine pack; its coarse slot is parked, and the mirror overwrites that slot at the end of
every period (alloy/coupling/world_clock_run.py). ``Sim``'s own writers (``set_pose``, ``settle``, ``apply_wrench``)
and ``engine.reset`` evict first; a write straight to the engine's rows would land on the parked copy and be lost
without a word. Each engine-level writer takes one of two forms:
- evict first (``evict_before_write``): the touched residents come back to the coarse pack, the write lands there as
  without the option, and a world may enter again at the next period. Used by ``alloy.envs.placement.rest_on_floor``
  (a ``Task.on_build`` hook a user may also call on any simulation).
- refuse (``refuse_world_clock``): ``alloy.envs.VectorEnv`` handed a simulation built with ``clock_worlds`` (its
  resets, randomizer, start poses, action writer and task hooks write the engine directly every step) and
  ``alloy.live.worldops.WorldOps`` (it copies a VectorEnv's world rows by slot). A VectorEnv that builds its own
  simulation never passes ``clock_worlds``; ``alloy.tasks`` scripts build their own simulations without it too.
Laws: tests/coupling/test_world_clock_writers.py.
"""

from __future__ import annotations

from typing import Any


def evict_before_write(eng: Any, worlds: Any = None) -> None:
    """Before an engine-level write to ``worlds`` (ids or a bool mask; None: every world): any resident among them
    comes back to the coarse pack with its state. Nothing without ``clock_worlds``.
    Example: ``evict_before_write(sim.engine)``."""
    clock = getattr(eng, "world_clock", None)
    if clock is not None:
        clock.evict(worlds)


def refuse_world_clock(sim: Any, who: str) -> None:
    """Raise ``EnvError`` (field ``clock_worlds``) when ``sim`` (a ``Sim`` or its engine) was built with
    ``clock_worlds``: ``who`` writes the engine's per-world rows directly, which a resident's mirror would overwrite.
    Example: ``refuse_world_clock(sim, "VectorEnv")``."""
    eng = getattr(sim, "engine", sim)
    if getattr(eng, "world_clock", None) is None:
        return
    from alloy.envs.errors import EnvError  # noqa: PLC0415  (alloy.envs imports this module)

    raise EnvError(f"{who} writes the engine's per-world state directly, and this simulation was built with "
                   "clock_worlds: a world stepping in the second simulation would lose those writes",
                   "build the simulation without clock_worlds (alloy.build(scene) or VectorEnv(scene, ...))",
                   field="clock_worlds")


__all__ = ["evict_before_write", "refuse_world_clock"]
