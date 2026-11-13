# SPDX-License-Identifier: Apache-2.0
"""Gap #10b P2, the per-world clock's driver (``world_clock_run.py``): its pure parts. Which scenes take no
per-world clock (``scene_refusal``), the period's report merged from the two packs (``merge_packs``) and the books."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from alloy.coupling.capture_passes import PER_WORLD_REPORT_FIELDS
from alloy.coupling.fine_rerun import NEVER


# Every limit true at this commit (``report()["limits"]``; the build docstring says each in short). Reviews in
# [internal note] scr/GAP-10B.md: WP4 3-6, WP10 S2, WP12 W1, W4 and its open list.
LIMITS = (
    "the sticky clock floor and the coupling mode are each simulation's, not each world's (S2c): a resident is held "
    "to both simulations' floors, a world that leaves keeps only the main simulation's",
    "an entering world takes the second simulation's substep rungs, seeded with the main simulation's stable steps; "
    "no world keeps its own step (S2c)",
    "the mirror copies each resident through the host every period (S3)",
    "while every slot is taken, a re-run ask re-runs every world of the main simulation, as without the option",
    "alloy.envs.VectorEnv and alloy.live world ops refuse a simulation built with clock_worlds; rest_on_floor and "
    "Sim.set_pose, settle, apply_wrench and reset bring a world back first; other writes to engine state bypass that",
    "a resident that fails in the second simulation is frozen by that simulation's latch and read through its mirrored "
    "copy; the main simulation's checks are not masked for parked slots; no law yet",
    "untested with a resident: a masked reset, an observer pause (sim.drive) evicting over a write to the parked slot",
    "neither simulation keeps its captured graphs across a reset, and the second recaptures after every entry; no "
    "GPU row measures both near a full device",
    "report(): steps and first_bad_step are the main simulation's (a resident's count is the second's); the books "
    "(exchanges, enters, re-run fallbacks) count over the simulation's life, a reset does not clear them",
    "exchanges_shared is an estimate (every world at the finer of the two levels, plus the main re-run exchanges), not "
    "a run",
    "every law runs on the CPU's MuJoCo Warp route; no GPU or graph-capture law yet (GPU: speed rows only)",
)


def scene_refusal(eng: Any) -> str | None:
    """Why this simulation takes no per-world clock at all (checked once, at build), or None."""
    from alloy.integration.stepper import entry_solvers  # noqa: PLC0415  (integration imports coupling)

    st = eng.stepper
    if not all(hasattr(st, name) for name in ("park_worlds", "_world_bounds", "_ladder", "_onset")):
        return "one solver steps this scene: there is no exchange clock to split"
    bad = next((type(s).__name__ for s in entry_solvers(st).values() for n in NEVER if n in type(s).__name__), None)
    if bad is not None:
        return f"a {bad} entry cannot move between simulations (liquid, MPM and the CPU's MuJoCo C route)"
    if getattr(st, "_gate", None) is not None:
        return "the contact gate's rows do not move with a world yet"
    return None


def _rows(coarse: Any, fine: Any, worlds: np.ndarray, slots: np.ndarray, sizes: tuple[int, int]) -> Any:
    """``coarse``'s value with the residents' rows from ``fine``'s slots when both are per-world arrays; else
    ``coarse``'s."""
    if coarse is None or fine is None or isinstance(coarse, (dict, str)):
        return coarse
    a, b = np.asarray(coarse), np.asarray(fine)
    if a.shape[:1] != (sizes[0],) or b.shape[:1] != (sizes[1],) or a.shape[1:] != b.shape[1:]:
        return coarse
    out = a.copy()
    out[worlds] = b[slots]
    return out.tolist() if isinstance(coarse, list) else out


def merge_packs(coarse: Any, fine: Any, worlds: np.ndarray, slots: np.ndarray, sizes: tuple[int, int]) -> Any:
    """The period's report: the coarse pack's (its dt_x and scalars), each resident's per-world rows and notes (one
    dict level deep) from its fine slot."""
    fields = {n: _rows(getattr(coarse, n), getattr(fine, n), worlds, slots, sizes) for n in PER_WORLD_REPORT_FIELDS}
    theirs = dict(getattr(fine, "notes", None) or {})
    notes: dict[str, Any] = {}
    for key, value in dict(getattr(coarse, "notes", None) or {}).items():
        other = theirs.get(key)
        if isinstance(value, dict) and isinstance(other, dict):
            notes[key] = {k: _rows(v, other.get(k), worlds, slots, sizes) for k, v in value.items()}
        else:
            notes[key] = _rows(value, other, worlds, slots, sizes)
    return dataclasses.replace(coarse, notes=notes, **fields)


@dataclasses.dataclass
class Books:
    """Exchanges over the run: taken by both packs, and what one clock shared by every world would have taken."""

    exchanges: int = 0
    shared: int = 0
    periods: int = 0
    resident_periods: int = 0
    evictions: int = 0
    refused: int = 0
    rerun_entries: int = 0
    j_coarse: int = 0
    j_fine: int = 0


__all__ = ["LIMITS", "Books", "merge_packs", "scene_refusal"]
