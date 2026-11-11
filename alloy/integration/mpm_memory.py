# SPDX-License-Identifier: Apache-2.0
"""GPU memory of an MPM scene, judged before the first step for the mode that will run (SCR-WP12-3).

Implicit MPM's step builds warp.fem sparse systems whose size follows the particles, and they dwarf the model's
arrays. How much depends on whether the step is captured as CUDA graphs (L4, Warp 1.18; [internal note]
notes/wp12-results/nov8-triage/ and mpm-memory/):
- graphs: capture needs Newton's rebuildable sparse grid, reserved at one cell per particle (``mpm_grid``), and
  everything on it is sized by that reservation; the graph keeps every temporary of the step alive.
  - Beside a kinematic or dynamic body, Newton's subgrid collider matrix adds 160 triplets per particle.
  - Measured: 22-28 kB per MPM particle beyond the model (dough under a press, soil_dig, soil_whisk, sand in a free
    box; nvidia-smi after capture), 24.9-31 kB in the Nov 8 rows. ``BYTES_MOVING`` = 32 kB.
  - The grid is now reserved from the cells in use (``mpm_grid``, SCR-WP12-3 a2): dough as graphs takes 150 MiB per
    world, not 390. The prediction stays at one cell per particle on purpose: with the smaller grid, 64 dough worlds
    fit as graphs but ran 3.0 s per period against 2.0 s eager (L4), so the graph/eager boundary is not moved.
  - No body in the scene: 2.7-4.2 kB in the Nov 8 rows. ``BYTES_STATIC`` = 6 kB.
- eager (no graphs): Newton's per-step sparse grid, sized by the cells in use, temporaries freed after each kernel.
  1.6-2.6 kB per particle beyond the model (2.3-3.9 kB with it), at 8 to 256 worlds. ``BYTES_EAGER`` = 3 kB.
  256 worlds of dough (4.4 M particles) peaked at 10.4 GB, 256 of sand at 11.8 GB.
- soil_dig at 64 worlds as graphs ran out inside Warp's sparse build, which went on with bad data and hit CUDA error
  700 (``sparse_guard``).

``plan(built, graph)`` runs in ``wiring.wire`` before any solver is built, on CUDA only, with the model already on the
device (its arrays are out of the free memory read). ``graph`` says whether the profile captures graphs:
- graphs asked for and predicted to fit: ``built`` comes back unchanged, so the graph path is untouched;
- graphs asked for, not fitting, eager fitting: a copy marked ``EAGER_ATTR`` (the reason) comes back; ``wire`` and
  ``wiring._scene_graphs`` then run the scene without graphs;
- the mode that would run fits in neither: ``MPMMemoryError`` (a CapacityError) names the MPM objects, the worlds,
  the predicted bytes against the free bytes, the worlds that would fit, and the fix.
The prediction is over ``HEADROOM`` of the free memory, read after a full garbage collection
(``_free_after_collect``): a simulation dropped earlier in the process (an earlier test, a notebook cell, the last
world count of a sweep) is a reference cycle whose device arrays and CUDA graphs stay allocated until Python's cycle
collector runs, which it rarely does. Until 2026-11-10 the collection ran only when the mode asked for did not fit:
a sweep's leftovers then let the graphs start in what was left, and they ran out later (SCR-OSS22-9).
``ALLOY_MPM_MEMORY_CHECK=off`` skips the plan (graphs stay as asked).
``sparse_guard`` stays as the backstop when the prediction is low.

When a step graph runs out of memory all the same, ``IntegratedSimulation`` steps down to smaller graphs, then to
none, and that period runs eagerly at once. ``room_to_step`` is called there, after the failed graph and the pool's
unused blocks went back to the driver: an MPM scene whose eager step does not fit in what is free now is refused
with the same ``MPMMemoryError``, before any Warp native runs on a full device (CUB then fails with CUDA error 101,
"invalid device ordinal", in the first prefix sum: SCR-OSS22-9, one L4, sand at 64 worlds late in a sweep).
"""

from __future__ import annotations

import gc
import logging
import os
from typing import Any

from alloy.core.errors import CapacityError

log = logging.getLogger("alloy.integration")

__all__ = ["BYTES_EAGER", "BYTES_MOVING", "BYTES_STATIC", "BYTES_STEP", "EAGER_ATTR", "HEADROOM", "MPMMemoryError",
           "eager_reason", "free_bytes", "mpm_particles", "plan", "predict", "room_to_step"]

BYTES_STATIC = 6 * 1024             # per MPM particle as graphs, no body in the scene (2.7-4.2 kB, Nov 8 rows)
BYTES_MOVING = 32 * 1024            # per MPM particle as graphs beside a kinematic / dynamic body (22-31 kB measured)
BYTES_EAGER = 3 * 1024              # per MPM particle without graphs, any scene (1.6-2.6 kB beyond the model)
BYTES_STEP = 1024                   # per MPM particle: eager scratch still to come once the solver has stepped
EAGER_ATTR = "mpm_memory_eager"     # on a BuiltScene whose MPM step runs without graphs: the reason
FIXED_BYTES = 256 << 20             # kernels, CUDA context slack, small solver arrays
HEADROOM = 0.92                     # of the free memory at build
GB = 2**30
ENV = "ALLOY_MPM_MEMORY_CHECK"
FIX = ("run fewer worlds, fewer MPM particles (a coarser spacing or a smaller region), or a GPU with more memory; "
       f"{ENV}=off skips this check (the run then fails at the first step, cleanly, if it does not fit)")


class MPMMemoryError(CapacityError):
    """An MPM scene predicted not to fit in the device's free memory: refused before the first step."""

    def __init__(self, message: str, *, need: int, free: int, fits: int) -> None:
        super().__init__(message)
        self.need, self.free, self.fits, self.fix = need, free, fits, FIX


def predict(particles: int, moving: bool = True, graph: bool = True) -> int:
    """Predicted peak device bytes of an implicit-MPM step over ``particles`` MPM particles (all worlds), beyond the
    model's arrays: captured as graphs (``moving``: the scene has a kinematic or dynamic body) or eager."""
    per = (BYTES_MOVING if moving else BYTES_STATIC) if graph else BYTES_EAGER
    return int(particles) * per + FIXED_BYTES


def mpm_particles(built: Any) -> tuple[int, list[str]]:
    """(MPM particles over all worlds, MPM object names) of a built scene's MPM entries."""
    entries = [e for e in built.entries if e.capability == "mpm"]
    names = sorted(n for n, d in built.report.decisions.items() if getattr(d, "entry", None) == "mpm")
    return sum(len(e.particles) for e in entries), names


def free_bytes(device: Any) -> int:
    """The device's free memory once Warp's mempool has handed back what it holds unused: CUDA returns pool memory
    above the release threshold at a synchronize. Without the trim, a process that already ran a large scene reads
    near 0 free (an L4 law run: 0.0 GB free after a 64-world refusal, then 21 GB after the trim)."""
    import warp as wp  # noqa: PLC0415

    wp.synchronize_device(device)
    if not wp.is_mempool_enabled(device):
        return int(device.free_memory)
    old = wp.get_mempool_release_threshold(device)
    try:
        wp.set_mempool_release_threshold(device, 0)
        wp.synchronize_device(device)
        return int(device.free_memory)
    finally:
        wp.set_mempool_release_threshold(device, old)


def eager_reason(built: Any) -> str | None:
    """Why ``plan`` turned graphs off for ``built``, or None."""
    return getattr(built, EAGER_ATTR, None)


def plan(built: Any, graph: bool) -> Any:
    """``built`` when its MPM step fits in the mode asked for (``graph``: captured), a copy marked ``EAGER_ATTR`` when
    only the eager step fits, else ``MPMMemoryError`` (module docstring)."""
    import warp as wp  # noqa: PLC0415

    device = wp.get_device(built.model.device)
    if not device.is_cuda or os.environ.get(ENV, "on").lower() == "off":
        return built
    n, names = mpm_particles(built)
    if n == 0:
        return built
    moving = int(getattr(built.model, "body_count", 0)) > 0
    free, worlds = _free_after_collect(device, free_bytes(device)), int(built.world_count)
    need_g, need_e = predict(n, moving), predict(n, graph=False)
    log.info("MPM memory: %d particles over %d worlds (%s), predicted %.2f GB as graphs, %.2f GB eager, %.2f GB "
             "free on %s", n, worlds, "beside bodies" if moving else "no bodies", need_g / GB, need_e / GB, free / GB,
             device)
    fits_e = need_e <= HEADROOM * free
    if (need_g <= HEADROOM * free) if graph else fits_e:
        return built                                    # the mode asked for fits: nothing changes
    what = f"MPM scene ({', '.join(names)}: {n:,} particles over {worlds} world(s))"
    if not fits_e:
        raise _refusal(what, device, n, worlds, free, need_g if graph else None)
    from alloy.integration.built_copy import replace_built  # noqa: PLC0415

    out = replace_built(built)
    setattr(out, EAGER_ATTR, f"{what}: as CUDA graphs its step needs about {need_g / GB:.1f} GB and "
            f"{free / GB:.1f} GB is free; it runs without graphs, about {need_e / GB:.1f} GB (SCR-WP12-3). Fewer "
            "worlds bring graphs back")
    return out


def room_to_step(built: Any, device: Any, free: int | None = None) -> None:
    """Raise ``MPMMemoryError`` when ``built`` has MPM particles and their eager step does not fit in ``device``'s
    free memory (``free``: a reading to use instead). Called after a step graph ran out of memory (module doc). The
    solver's own arrays are allocated by then and its first exchanges ran without graphs, so only the scratch still
    to come is asked for: ``BYTES_STEP`` per particle (an eager run of the sand pile at 64 worlds took 0.96 kB per
    particle from its build to its end, one T4). Nothing to do on the CPU or without MPM particles."""
    import warp as wp  # noqa: PLC0415

    device = wp.get_device(device)
    n, names = mpm_particles(built) if device.is_cuda and os.environ.get(ENV, "on").lower() != "off" else (0, [])
    if n == 0:
        return
    free = free_bytes(device) if free is None else int(free)
    need = n * BYTES_STEP
    log.info("MPM memory after a step graph did not fit: %d particles, about %.2f GB to step without graphs, %.2f GB "
             "free on %s", n, need / GB, free / GB, device)
    if need > HEADROOM * free:
        worlds = int(built.world_count)
        raise MPMMemoryError(
            f"MPM scene ({', '.join(names)}: {n:,} particles over {worlds} world(s)): its CUDA step graph ran out of "
            f"memory on {device} ({device.name}), and to step on without graphs it needs about {need / GB:.1f} GB "
            f"more ({BYTES_STEP // 1024} kB per MPM particle) where {free / GB:.1f} GB is free (other simulations' "
            f"live CUDA graphs and arrays in this process count as used). Fix: {FIX}",
            need=need, free=free, fits=int(worlds * HEADROOM * free // need))


def _free_after_collect(device: Any, free: int) -> int:
    """The device's free memory after a full garbage collection (module doc); ``free`` is the reading before it.
    A collected simulation's CUDA graphs stay reserved in Warp's graph pool until that is trimmed."""
    from alloy.parallel.counters import trim_graph_pool  # noqa: PLC0415 - WP8

    gc.collect()
    trim_graph_pool(str(device))
    after = free_bytes(device)
    if after > free:
        log.info("MPM memory: %.2f GB more free on %s after collecting dropped simulations", (after - free) / GB,
                 device)
    return after


def _refusal(what: str, device: Any, n: int, worlds: int, free: int, need_graph: int | None) -> MPMMemoryError:
    """The error for an MPM scene whose step fits in neither mode."""
    need = predict(n, graph=False)
    per_world = max(n // max(worlds, 1), 1)
    fits = max(int((HEADROOM * free - FIXED_BYTES) // (per_world * BYTES_EAGER)), 0)
    graphs = f" (as CUDA graphs, {need_graph / GB:.1f} GB)" if need_graph is not None else ""
    return MPMMemoryError(
        f"{what} does not fit on {device} ({device.name}), even without CUDA graphs: its step needs about "
        f"{need / GB:.1f} GB ({BYTES_EAGER // 1024} kB per MPM particle){graphs} and {free / GB:.1f} GB is free; about "
        f"{fits} world(s) of this scene fit (free memory leaves out live CUDA graphs of other simulations in this "
        f"process). Fix: {FIX}", need=need, free=free, fits=fits)
