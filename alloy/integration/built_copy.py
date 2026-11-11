# SPDX-License-Identifier: Apache-2.0
"""BuiltScene copies that keep the assembler's extras, and the one seed draws are made at (SCR-DC3-8).

``dataclasses.replace`` on a ``BuiltScene`` copies only its dataclass fields. The assembler and the randomizer also
set attributes outside them (``alloy/scene/randomize.py``, ``alloy/scene/readback.py``):

- ``_plan``: the assembly plan; ``readback`` and ``Simulation.reset``'s redraw need it;
- ``randomized``: the per-world draws, ``{(object, field): [W] float64}``;
- ``randomized_seed``: the seed of those draws, set by a caller that redrew them (DC show's ``randomize``).
- ``mpm_memory_eager``: why the scene's MPM step runs without CUDA graphs (``mpm_memory.plan``, SCR-WP12-3);
  ``wiring._scene_graphs`` reads it, so a copy that dropped it would capture graphs that do not fit.

A copy without them still holds the drawn values in its Model (shared), but reads back no draws, cannot be read back
at all, and never redraws on reset. ``replace_built`` carries every such attribute over by reference: the copy shares
the Model, so it shares the draws that describe it.

**Draw seed.** Assembly and reset draw from one seed: ``draw_seed(built)``. It is ``built.randomized_seed`` when a
caller redrew the scene, else ``ASSEMBLY_DRAW_SEED`` = 0, the seed ``SceneAssembler`` draws at
(``alloy/scene/finish.py`` ``finish_scene``). ``IntegratedSimulation.reset`` redraws at it, not at the simulation's
``seed``, so a reset world gets back the values it was assembled with. To run other draws, redraw the built before
building the simulation: ``apply_randomization(built, s)`` and ``built.randomized_seed = s``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, TypeVar

__all__ = ["ASSEMBLY_DRAW_SEED", "draw_seed", "replace_built"]

ASSEMBLY_DRAW_SEED = 0        # alloy/scene/finish.py: finish_scene calls apply_randomization(built, seed=0)

B = TypeVar("B")


def replace_built(built: B, **changes: Any) -> B:
    """``dataclasses.replace(built, **changes)`` that keeps the attributes set outside the dataclass fields
    (``_plan``, ``randomized``, ``randomized_seed``, ...), by reference."""
    out = dataclasses.replace(built, **changes)
    fields = {f.name for f in dataclasses.fields(built)}
    for name, value in vars(built).items():
        if name not in fields and name not in vars(out):
            setattr(out, name, value)
    return out


def draw_seed(built: Any) -> int:
    """The seed ``built``'s draws were made at: ``built.randomized_seed`` if a caller redrew, else the assembler's."""
    return int(getattr(built, "randomized_seed", ASSEMBLY_DRAW_SEED))
