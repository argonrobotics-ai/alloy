# SPDX-License-Identifier: Apache-2.0
"""Hook-level checks: the plug-in's own hooks replayed on fixed recorded states, independent of the solver.

Whole-run repeat checks (determinism, world independence, graph replay) also see the engine; on a GPU, MuJoCo's
contacts are not run-to-run deterministic by default. These checks isolate the plug-in: from a finished run, two
consecutive recorded states feed ``before`` (``apply``) and ``after`` (``update`` / ``measure``) with the plug-in
state restored each time, and the outputs (forces and every plug-in state array) are compared:

- ``hook_determinism``: the same inputs twice give bit-identical outputs;
- ``hook_identical_worlds``: inputs equal in every world give outputs equal in every world;
- ``hook_isolated_worlds``: perturbing the inputs of worlds 1.. leaves world 0's outputs unchanged;
- ``hook_graph``: the hooks captured as a graph (CUDA or CPU) and launched give the eager outputs.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.harness.report import CheckResult
from alloy.plugins.harness.run import Run

log = logging.getLogger("alloy.plugins")

Outputs = dict[str, np.ndarray]
FIX = ("index every array with the world id, write each element from one thread, keep changing values in device "
       "arrays (add_state / add_input / ctx.time)")


def _state(run: Run, k: int, edit: Callable[[str, np.ndarray], np.ndarray] | None = None) -> newton.State:
    """A State with the trace's period-``k`` body and particle arrays (optionally edited per field)."""
    s = run.model.state()
    for name in ("body_q", "body_qd", "particle_q", "particle_qd"):
        arr = getattr(s, name, None)
        if arr is None or arr.shape[0] == 0:
            continue
        values = getattr(run.trace, name)[k].copy()
        arr.assign(np.ascontiguousarray(edit(name, values) if edit else values, dtype=np.float32))
    return s


class _HookRunner:
    """Runs the host's hooks on given states from one saved plug-in state."""

    def __init__(self, run: Run) -> None:
        self.host, self.device = run.host, run.model.device
        self.dt = float(run.sim.clock.dt)
        self.saved = {k: wp.clone(v) for k, v in self.host.state_arrays().items()}

    def restore(self) -> None:
        for k, v in self.host.state_arrays().items():
            wp.copy(v, self.saved[k])

    def hooks(self, s_in: newton.State, s_out: newton.State) -> None:
        self.host.before(s_in, None, self.dt)
        self.host.restore_forces(s_in)
        self.host.after(s_in, s_out, self.dt)

    def outputs(self) -> Outputs:
        out = {"forces/body_f": self.host.forces.body_f.numpy().copy(),
               "forces/particle_f": self.host.forces.particle_f.numpy().copy()}
        out.update({k: v.numpy().copy() for k, v in self.host.state_arrays().items()})
        return out

    def eager(self, s_in: newton.State, s_out: newton.State, *, same_state: bool = False) -> Outputs:
        self.restore()
        if same_state:
            self._world0_state()
        self.hooks(s_in, s_out)
        return self.outputs()

    def _world0_state(self) -> None:
        """Every world's plug-in state = world 0's (identical-worlds inputs)."""
        for v in self.host.state_arrays().values():
            a = v.numpy()
            a[1:] = a[:1]
            v.assign(a)

    def graph(self, s_in: newton.State, s_out: newton.State) -> Outputs:
        self.restore()
        with wp.ScopedCapture(device=self.device) as cap:
            self.hooks(s_in, s_out)
        self.restore()
        wp.capture_launch(cap.graph)
        return self.outputs()


def _rows(run: Run) -> dict[str, np.ndarray]:
    m = run.model
    return {"body": m.body_world.numpy(), "particle": m.particle_world.numpy() if m.particle_count else np.zeros(0)}


def _world_view(run: Run, out: Outputs, name: str, w: int) -> np.ndarray:
    """Output ``name`` restricted to world ``w`` (force rows of its items, or row ``w`` of a plug-in array)."""
    rows = _rows(run)
    if name == "forces/body_f":
        return out[name][rows["body"] == w]
    if name == "forces/particle_f":
        return out[name][rows["particle"] == w]
    return out[name][w]


def _max_diff(run: Run, a: Outputs, b: Outputs, wa: int | None = None, wb: int | None = None) -> float:
    diff = 0.0
    for name in a:
        x = a[name] if wa is None else _world_view(run, a, name, wa)
        y = b[name] if wb is None else _world_view(run, b, name, wb)
        if x.size:
            diff = max(diff, float(np.abs(x.astype(np.float64) - y.astype(np.float64)).max()))
    return diff


def _copy_world0(run: Run) -> Callable[[str, np.ndarray], np.ndarray]:
    rows = _rows(run)

    def edit(name: str, v: np.ndarray) -> np.ndarray:
        where = rows["body"] if name.startswith("body") else rows["particle"]
        src = v[where == 0]
        for w in range(1, run.model.world_count):
            v[where == w] = src
        return v
    return edit


def _perturb_others(run: Run, seed: int) -> Callable[[str, np.ndarray], np.ndarray]:
    rows, rng = _rows(run), np.random.default_rng(seed)

    def edit(name: str, v: np.ndarray) -> np.ndarray:
        where = rows["body"] if name.startswith("body") else rows["particle"]
        hit = where >= 1
        noise = rng.normal(0.0, 1.0e-3, v[hit].shape)
        if name == "body_q":
            noise[..., 3:] = 0.0                       # keep quaternions unit
        v[hit] = v[hit] + noise
        return v
    return edit


def hook_checks(run: Run) -> list[CheckResult]:
    """The four hook-level checks on run ``run`` (its last two recorded periods)."""
    if run.error is not None or run.host is None or not run.host.plugins:
        return [CheckResult.skip("hook_determinism", "hooks", "no finished run with plug-ins")]
    k, worlds = len(run.trace.t) - 1, run.model.world_count
    hr = _HookRunner(run)
    s_in, s_out = _state(run, max(k - 1, 0)), _state(run, k)
    first, again = hr.eager(s_in, s_out), hr.eager(s_in, s_out)
    out = [_result("hook_determinism", _max_diff(run, first, again), "same inputs twice")]
    if worlds > 1:
        same = _copy_world0(run)
        eq = hr.eager(_state(run, max(k - 1, 0), same), _state(run, k, same), same_state=True)
        out.append(_result("hook_identical_worlds", max(_max_diff(run, eq, eq, w, 0) for w in range(1, worlds)),
                           "inputs equal in every world"))
        bumped = hr.eager(_state(run, max(k - 1, 0), _perturb_others(run, 1)),
                          _state(run, k, _perturb_others(run, 2)))
        out.append(_result("hook_isolated_worlds", _max_diff(run, first, bumped, 0, 0), "worlds 1.. perturbed"))
    try:
        out.append(_result("hook_graph", _max_diff(run, first, hr.graph(s_in, s_out)), "captured hooks vs eager"))
    except Exception as err:  # noqa: BLE001 - a hook that cannot be captured fails the check
        out.append(CheckResult("hook_graph", "hooks", False, detail=f"capture failed: {err}", fix=FIX))
    hr.restore()
    return out


def _result(name: str, diff: float, what: str) -> CheckResult:
    return CheckResult(name, "hooks", diff == 0.0, diff, 0.0, f"max output difference, {what}", FIX)


__all__ = ["hook_checks"]
