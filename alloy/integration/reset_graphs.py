# SPDX-License-Identifier: Apache-2.0
"""Captured period graphs across ``IntegratedSimulation.reset`` (launch item, coordinator 2026-11-10 00:56Z).

A CUDA graph bakes the addresses of the arrays its kernels read and write and every Python value passed to a launch;
the contents of those arrays are read at replay. ``reset`` used to drop every period graph, always. An RL run that
resets a few worlds almost every step then captured again almost every step: Go2, 4,096 worlds on one L4, a masked
reset plus a step took 214 ms against 50 ms for the step alone (FEAT-LOCO).

What a reset does, and what of it a captured period can notice:
- the start state goes back into the State arrays, the latch snapshot and health's per-world arrays are rewritten,
  VBD's rigid history and the proxy pipelines' contact matching are cleared for the reset worlds
  (``coupling.match_reset``): all writes into existing device arrays, read at replay. The snapshot restore and the
  matching reset also run on every rollback, which keeps its graphs;
- the re-run cooldowns (``OnsetRefiner.last``, ``PeriodLadder.last``) and the latch log are host rows that choose
  clocks and re-runs between periods; no launch inside a graph takes them;
- a scene with randomized values draws them again for the reset worlds; where a draw differs from the value the
  model holds (a world redrawn at another seed since; never on a plain reset, TRAIN-FIX 3a), it writes the model and
  calls every solver's ``notify_model_changed``. That can rebind arrays (the coupler clones ``shape_contact_pairs``
  into each entry's view; implicit MPM allocates new particle arrays) and refresh host values a launch takes by
  value. A graph cannot follow either. Code that edits the model between resets and notifies the solvers itself
  (``alloy.episode.layout``) has the same effect, and until now the next reset's drop is what made its graphs right
  again.

So ``watch`` marks the simulation whenever one of its solvers is told the model changed, by any caller, or one of
health's setters that rebind an array is called (``HEALTH_SETTERS``: each says "before any graph capture", and a late
call used to be repaired by the next reset's drop), once the simulation has stepped: before its first step nothing is
captured yet. ``after_reset`` keeps the graphs only when nothing was marked since the graphs were last dropped and
the scene has one entry solver, of a kind the law covers (``KEPT_SOLVERS``: MuJoCo Warp, or VBD cloth / soft
bodies), with no plug-in host and no topology controller. Otherwise it drops them as before, and logs why once.
``sim.reset_keeps_graphs = False`` drops them always (the law's reference run).

Law: a run with masked resets is the same bit for bit with the graphs kept as with them dropped.
- CPU, period graphs forced on (tests/integration/test_reset_graphs.py): free cubes with a world latched INVALID and
  reset, a servo arm, the fast profile's sparse checks.
- One L4 (tests/integration/test_reset_graphs_gpu.py), difference kept vs dropped / between two dropped runs, and a
  masked reset plus a step, kept / dropped: cubes in 64 worlds 0.0 / 0.0, 26.6 / 54.7 ms; the same on the fast
  profile 0.0 / 0.0, 1.7 / 13.7 ms; a cloth in 8 worlds 0.0 / 0.0, 37.8 / 131.1 ms; health records equal.
- Coupled scenes (more than one entry solver) still drop their graphs (WP0's review). The one measured, a ball
  under a cloth (MuJoCo Warp + VBD under the Scheduler), is bit for bit kept vs dropped on CPU graphs (3 worlds,
  masked resets and a full one), but on a GPU it does not repeat (two dropped runs differ by 15 mm after 24
  periods, the kept run by 14 mm), and a row inside that spread cannot show a stale binding. The row that lifts the
  refusal: a coupled scene that repeats bit for bit on the GPU, then kept vs dropped 0.0.
``sim.graph_stats()`` counts both outcomes (``reset_kept``, ``reset_dropped``).
"""

from __future__ import annotations

import logging
import weakref
from typing import Any

log = logging.getLogger("alloy.integration")

__all__ = ["HEALTH_SETTERS", "KEPT_SOLVERS", "after_reset", "keep", "watch"]

# exact class names (module doc): MuJoCo Warp as Newton's solver or Alloy's synced one, VBD for cloth and soft bodies
KEPT_SOLVERS = ("SolverMuJoCo", "SyncedMuJoCo", "SolverVBD", "SolverVBDAlloy", "SolverVBDMomentum")
HEALTH_SETTERS = ("set_cut_front", "skip_soft_contacts", "set_soft_strain", "register_depth_check")  # rebind arrays
_LOGGED: set[str] = set()


def watch(sim: Any) -> None:
    """Start the bookkeeping on ``sim`` (an IntegratedSimulation, at construction): ``reset_keeps_graphs`` on, the
    kept / dropped counts, and ``_model_notified``, set whenever ``notify_model_changed`` of the stepper or of an entry
    solver, or one of health's ``HEALTH_SETTERS``, is called after the first step. Only a simulation that steps with
    graphs (``sim.use_graph``) gets the wrappers. Each is a closure set on the instance, over a weak reference to
    ``sim``: it shadows the class's method on that instance, so a direct call counts too."""
    sim.reset_keeps_graphs, sim.reset_graphs, sim._model_notified = True, {"kept": 0, "dropped": 0}, False
    sim._notify_watched = bool(getattr(sim, "use_graph", False))
    if not sim._notify_watched:
        return
    ref, seen = weakref.ref(sim), []
    for solver in (*getattr(sim, "solvers", ()), sim.stepper):
        fn = getattr(solver, "notify_model_changed", None)
        if not callable(fn) or any(solver is s for s in seen):
            continue
        seen.append(solver)
        solver.notify_model_changed = _marking(fn, ref)
    for name in HEALTH_SETTERS:
        fn = getattr(getattr(sim, "health", None), name, None)
        if callable(fn):
            setattr(sim.health, name, _marking(fn, ref))


def _marking(fn: Any, ref: Any) -> Any:
    def marked(*args: Any, **kwargs: Any) -> Any:
        owner = ref()
        if owner is not None and owner.steps > 0:          # before the first step no graph is captured yet
            owner._model_notified = True
        return fn(*args, **kwargs)

    return marked


def keep(sim: Any) -> tuple[bool, str]:
    """(keep the period graphs across this reset, why not). Example: ``keep(sim) == (True, "")`` for a robot arm
    scene on MuJoCo Warp whose model nobody changed."""
    if not getattr(sim, "reset_keeps_graphs", True):
        return False, "reset_keeps_graphs is off"
    if not getattr(sim, "_notify_watched", False):
        return False, "notify_model_changed is not watched (graphs were turned on after the build)"
    if getattr(sim, "_model_notified", False):
        return False, "the model changed (a solver was notified, or a health setter rebound an array) since capture"
    solvers = tuple(getattr(sim, "solvers", ()))
    other = sorted({type(s).__name__ for s in solvers} - set(KEPT_SOLVERS))
    if other:
        return False, f"entry solver(s) {', '.join(other)}: outside the kept-graph law"
    if len(solvers) > 1:
        return False, "more than one entry solver (a coupled scene): outside the kept-graph law for now"
    if type(sim.stepper).__name__ == "SchedulerADMM":
        return False, "the ADMM coupler: outside the kept-graph law"
    if getattr(sim, "plugins", None) is not None:
        return False, "a plug-in host: outside the kept-graph law"
    if getattr(sim, "topology", None) is not None:
        return False, "a topology controller: outside the kept-graph law"
    return True, ""


def after_reset(sim: Any) -> bool:
    """The end of ``IntegratedSimulation.reset``: keep the period graphs (True) or drop them and the per-key node
    counts, as every reset did before (module docstring). Nothing to do without graphs (the CPU)."""
    if sim._graphs is None:
        return False
    ok, why = keep(sim)
    counts = sim.reset_graphs
    counts["kept" if ok else "dropped"] += 1
    if ok:
        return True
    sim._graphs.invalidate()
    sim.exchange_nodes.clear()
    sim._model_notified = False                # the next capture bakes the model as it is now
    if why not in _LOGGED:
        _LOGGED.add(why)
        log.info("reset drops the captured period graphs: %s", why)
    return False
