# SPDX-License-Identifier: Apache-2.0
"""SC4 crash regression (coordinator ruling in [internal note]): a MuJoCo + VBD Scheduler under per-key CUDA
graph capture with forced VBD rung changes (and dt_x changes), as ``IntegratedSimulation``'s GraphCache drives it: the
first period of a key runs eagerly and captures its graph, a revisited key replays it (``cache``), or every rung change
recaptures (``recapture``). Checks over ``ALLOY_RECAPTURE_CHANGES`` (100) changes at 1, 2 and 16 worlds: no crash, no
graph or scratch memory growth once every key exists, the buffers the graphs point at never move (Newton's per-entry
contact buffers, every solver array). Captured vs eager is asserted bitwise only when the solvers claim RUN_TO_RUN
(``alloy.precision.determinism.effective_determinism``); otherwise it is recorded, not asserted. Two eager samples
agreeing by chance is no determinism claim: on L4 at 2 worlds, eager x3 came out bitwise while a recapture run
diverged from period 48 in one run and stayed bitwise over 60 periods in the next (WP10's report, scr/WP4.md 11-06).
Chunks are ``ALLOY_RECAPTURE_CHUNK`` (4) exchanges, well under WP12's node budget even at rung 64, so a crash here is
not graph size."""

from __future__ import annotations

import os

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.testing.coupling import (ClothSceneSpec, cloth_scheduler, coupling_box_on_cloth, fixed_clock_policy,
                                    record_metric)

DEV = os.environ.get("ALLOY_DEVICE", "cuda:0")
CHANGES = int(os.environ.get("ALLOY_RECAPTURE_CHANGES", "100"))
CHUNK = int(os.environ.get("ALLOY_RECAPTURE_CHUNK", "4"))
RUNGS = (1, 2, 4, 8, 16, 32, 64)
DTS = (0.625e-3, 0.3125e-3)       # dt_x keys too (SC4: pinning the rung still recaptured on dt_x changes)
FIELDS = ("body_q", "body_qd", "joint_q", "joint_qd", "particle_q", "particle_qd")
LEAK_BYTES = 1 << 20              # allowed mempool growth once every key has been captured


def schedule(n: int, seed: int = 7) -> list[tuple[float, int]]:
    """``n`` (dt_x, rung) keys, each different from the previous, every key visited, keys revisited."""
    rng = np.random.default_rng(seed)
    keys = [(dt, r) for dt in DTS for r in RUNGS]
    out = list(keys)
    while len(out) < n:
        k = keys[int(rng.integers(len(keys)))]
        if k != out[-1]:
            out.append(k)
    return out[:n]


def _copy(src, dst) -> None:
    for name in FIELDS:
        a = getattr(src, name)
        if a is not None and a.size:
            wp.copy(getattr(dst, name), a)


def _pointers(sched) -> dict[str, int]:
    """Device pointers of the buffers a captured graph bakes in, beyond the State arrays WP12 already checks."""
    out = {}
    for name, filtered in getattr(sched, "_entry_contact_buffers", {}).items():
        for attr in ("rigid_contact_count", "rigid_contact_shape0", "soft_contact_count", "soft_contact_particle"):
            arr = getattr(filtered, attr, None)
            if arr is not None:
                out[f"{name}.contacts.{attr}"] = int(arr.ptr)
    for name, entry in sched._entries.items():
        for st in ("state_0", "state_1", "state_tmp"):
            s = getattr(entry, st, None)
            for f in ("body_q", "particle_q"):
                arr = getattr(s, f, None) if s is not None else None
                if arr is not None and arr.size:
                    out[f"{name}.{st}.{f}"] = int(arr.ptr)
    for key, arr in sched.state_arrays().items():
        out[f"state.{key}"] = int(arr.ptr)
    for name, entry in sched._entries.items():
        out.update(_solver_arrays(entry.solver, f"{name}.solver"))
    return out


def _solver_arrays(obj, prefix: str, depth: int = 1) -> dict[str, int]:
    """Pointers of every non-empty wp.array attribute of a solver (and of its Newton sub-objects, one level):
    the lazily grown contact state a graph also bakes in."""
    out = {}
    for attr, val in vars(obj).items():
        if isinstance(val, wp.array):
            if val.ptr is not None and val.size:
                out[f"{prefix}.{attr}"] = int(val.ptr)
        elif depth > 0 and hasattr(val, "__dict__") and type(val).__module__.startswith("alloy._vendor.newton"):
            out.update(_solver_arrays(val, f"{prefix}.{attr}", depth - 1))
    return out


class Run:
    """One Scheduler driven through a key schedule, eagerly or with per-key graphs."""

    def __init__(self, worlds: int, mode: str):
        spec = ClothSceneSpec()
        self.model = coupling_box_on_cloth(worlds, spec, drop_offsets=[0.002 * k for k in range(worlds)], device=DEV)
        self.sched = cloth_scheduler(self.model, spec, policy=fixed_clock_policy(DTS[0]), vbd_adaptive=True)
        self.pipe = newton.CollisionPipeline(self.model)
        self.contacts = self.pipe.contacts()
        self.s0, self.s1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.s0)
        self.mode, self.graphs, self.captures, self.seen = mode, {}, 0, set()
        self.moved: list[str] = []

    def _chunk(self, dt: float) -> None:
        for _ in range(CHUNK):
            self.s0.clear_forces()
            self.pipe.collide(self.s0, self.contacts)
            self.sched.step(self.s0, self.s1, self.control, self.contacts, dt)
            _copy(self.s1, self.s0)

    def period(self, key: tuple[float, int]) -> None:
        """One period of CHUNK exchanges at ``key``; every mode advances the same sim time."""
        dt, rung = key
        self.sched.set_substeps({"vbd": rung})
        if self.mode == "eager" or key not in self.seen:
            self.seen.add(key)
            self._chunk(dt)                                 # a key's first period runs eagerly (GraphCache rule)
            if self.mode == "cache":
                self._capture(key, dt)                      # recorded only, replayed on the next visit
            return
        if self.mode == "recapture":
            self._capture(key, dt)                          # a fresh graph on every change, then launched
        self._replay(key)

    def _capture(self, key, dt: float) -> None:
        self.graphs.pop(key, None)
        with wp.ScopedCapture(device=DEV) as cap:
            self._chunk(dt)
        self.graphs[key] = (cap.graph, _pointers(self.sched))
        self.captures += 1

    def _replay(self, key) -> None:
        graph, ptrs = self.graphs[key]
        now = _pointers(self.sched)
        self.moved += [k for k, v in ptrs.items() if now.get(k) != v]
        wp.capture_launch(graph)


def _state(run: Run) -> dict[str, np.ndarray]:
    wp.synchronize_device(DEV)
    return {f: getattr(run.s0, f).numpy().copy() for f in FIELDS if getattr(run.s0, f) is not None}


def _run_to_run(run: Run) -> bool:
    """True when the scheduler's solvers claim RUN_TO_RUN on this device and world count (D9)."""
    from alloy.precision.determinism import effective_determinism
    from alloy.precision.types import Determinism, PrecisionConfig

    mode, _ = effective_determinism(PrecisionConfig(), [run.sched], int(run.model.world_count))
    return mode >= Determinism.RUN_TO_RUN


@pytest.mark.gpu
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("worlds", [1, 2, 16])
def test_forced_rung_changes_with_recapture(worlds):
    keys = schedule(CHANGES)
    rows = {}
    for mode in ("cache", "recapture"):
        run = Run(worlds, mode)
        mem = []
        for k, key in enumerate(keys):
            run.period(key)
            wp.synchronize_device(DEV)
            mem.append(int(wp.get_mempool_used_mem_current(DEV)))
        settled = mem[len(set(keys)) - 1]                     # every key captured at least once by here
        growth = max(mem[len(set(keys)):] or [settled]) - settled
        rows[mode] = {"captures": run.captures, "mem_growth_bytes": growth, "moved": sorted(set(run.moved)),
                      "state": _state(run)}
        print(f"{worlds} worlds {mode}: {run.captures} captures over {len(keys)} changes, mempool growth after all "
              f"keys {growth / 1e6:.3f} MB, moved buffers {sorted(set(run.moved))[:6]}", flush=True)
    refs = []
    for _ in range(2):                                      # eager twice: the solvers' own run-to-run spread
        eager = Run(worlds, "eager")
        for key in keys:
            eager.period(key)
        refs.append(_state(eager))
    ref = refs[0]

    def maxdq(a, b):
        return float(np.max(np.abs(a["particle_q"] - b["particle_q"])))
    spread = maxdq(refs[0], refs[1])                         # 0 when MuJoCo+VBD run bitwise here
    same = {m: all(rows[m]["state"][f].tobytes() == ref[f].tobytes() for f in ref) for m in rows}
    dq = {m: maxdq(rows[m]["state"], ref) for m in rows}
    record_metric("SC4-recapture", device=DEV, worlds=worlds, changes=len(keys), chunk=CHUNK,
                  captures={m: rows[m]["captures"] for m in rows},
                  mem_growth_bytes={m: rows[m]["mem_growth_bytes"] for m in rows},
                  moved={m: rows[m]["moved"] for m in rows}, bitwise=same, max_dq_particles_m=dq,
                  eager_vs_eager_max_dq_m=spread)
    print(f"{worlds} worlds: captured == eager bitwise {same}, max |dq| {dq}; eager vs eager {spread:.3e} m",
          flush=True)
    for m in rows:
        assert rows[m]["moved"] == [], (m, rows[m]["moved"])
        assert rows[m]["mem_growth_bytes"] <= LEAK_BYTES, (m, rows[m]["mem_growth_bytes"])
    if _run_to_run(run):                                     # the solvers claim it: capture must be bitwise too
        assert all(same.values()), dq
    elif spread > 0.0:                                       # float atomics (D9): within the solvers' own spread
        assert all(v <= 3.0 * spread for v in dq.values()), (dq, spread)
    # NONE with a zero sample spread: no determinism claim to test against; recorded above (record_metric)
