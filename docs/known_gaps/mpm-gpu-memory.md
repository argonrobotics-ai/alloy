# MPM on the GPU: CUDA graphs cost ~10x the memory of an eager step; too big for graphs runs eager, too big for both is refused

Status: guarded, not fixed. A scene whose MPM step does not fit as CUDA graphs runs without them (SCR-WP12-3), and one
that fits in neither mode is refused before its first step. With the check off, a full GPU on an MPM step is a clean
error, not CUDA error 700. The graph mode's memory itself is not reduced yet (SCR
follow-ups below). Warp's unchecked allocation is written up for upstream (an issue draft, not shipped) but not posted.

## What

Implicit MPM's step builds warp.fem sparse systems every step. Their size follows the MPM particles, and depends on
how the step runs:
- **As CUDA graphs** (the default on CUDA): capture needs Newton's rebuildable sparse grid, and everything built on it
  is sized by its reservation. The graph also keeps every temporary of the step alive at once. Beside a kinematic or
  dynamic body, Newton's subgrid collider matrix adds 160 triplets per particle. Alloy reserves twice the cells the
  material fills at the start (it fills one cell per 5-9 particles; `alloy.integration.mpm_grid`, SCR-WP12-3 a2),
  and a grid that outgrows that is doubled and its period re-run (`grid_growth`).
- **Eager** (no graphs): Newton's per-step sparse grid, sized by the cells in use, temporaries freed as it goes.

When the GPU filled up inside the sparse build, Warp's native code (`wp_bsr_matrix_from_triplets_device`) did not
check its scratch allocation and went on. The next kernels failed with CUDA error 700, which leaves the context
unusable. Nov 8 reconciliation pass: soil_dig at 64 worlds on an L4, and the 1,024-world sand-fill rows.

Alloy now does two things:
- `alloy.integration.mpm_memory.plan`, in `wire`, before any solver is built:
  - graphs that fit: nothing changes;
  - graphs that do not fit, eager that does: the scene runs without graphs, with a note (`graphs off: ...`) in
    `sim.engine.parts.notes` and the log;
  - neither: MPMMemoryError (a CapacityError) with the predicted bytes against the device's free bytes, the worlds
    that would fit, and the fix.
  `ALLOY_MPM_MEMORY_CHECK=off` skips it, and graphs stay as asked.
- `alloy.integration.sparse_guard` and `alloy.integration.native_guard`: before each of Warp's native calls on an MPM
  step, they reserve its scratch through Warp's allocator on CUDA, and raise SparseMemoryError or NativeMemoryError
  instead. Covered: the sparse triplet build, NanoVDB grid builds and in-place rebuilds, radix sorts and run-length
  encodes. This is the backstop when the prediction is low or the check is off.
- When a step graph runs out of memory all the same (the prediction was low, or the device filled up after the
  build), the simulation steps down to smaller graphs, then to none, and the period it was for steps without graphs
  at once (2026-11-10, SCR-OSS22-9). Before that period it drops its captured graphs (they are captured again when
  next launched), collects simulations dropped earlier, hands the pools' unused memory back and clears the error the
  failed allocation leaves in Warp (Why, below). If the device then has no room for the eager scratch still to come
  (1 kB per MPM particle), it raises MPMMemoryError with the bytes needed and free. Before this fix that period died
  with "CUDA error 101: invalid device ordinal" in a prefix sum: sand at 64 worlds late in a speed sweep on one L4.

## Number

L4, 23 GB, Warp 1.18 (WP12's memory probes and its Nov 8 triage runs, not shipped):

| mode | per MPM particle, beyond the model | MiB per world (dough_ball under a press, 17,256 particles) |
|---|---|---|
| graphs, one cell per particle reserved (before a2), beside a body | 22-28 kB (Nov 8: 24.9-31 kB); predicted 32 kB | 398 |
| graphs, reserved from the cells in use (a2) | ~9 kB | 150 (8 worlds; 347 ms per period, was 770) |
| graphs, no body | 2.7-4.2 kB (Nov 8 rows); predicted 6 kB | - |
| graphs, no body: a pile of 200 g of sand on a table, 64 worlds, 492,032 particles (one T4, 2026-11-10) | 10.6 kB (5.2 GB beyond a 2.2 GB model); predicted 6 kB | 81 |
| eager, any scene | 1.6-2.6 kB; predicted 3 kB | 38-42 |

- What fits an L4 as graphs: soil_dig 32 worlds (15.4 GB), soil_whisk 16 worlds (11.7 GB).
- What fits eagerly, measured: dough 256 worlds (10.4 GB peak), sand fill 256 worlds (11.8 GB), soil_dig 64 worlds
  (2.8 GB).
- Speed at 8 worlds (dough, ms per 10 ms period): graphs 771, eager 353. At 256 worlds eager: 9.7 s per period
  (dough), 6.5 s (sand).

## Why

- The step's memory is Newton's implicit-MPM warp.fem systems. Under capture they are sized by the reserved grid,
  not by the cells in use. The needs are not known until the step runs, so Alloy predicts them from measured rates.
- Warp's native triplet build (`wp_bsr_matrix_from_triplets_device`, Warp 1.18) takes its scratch through
  `ScopedTemporary` without a null check. An out-of-memory there is printed, not raised, and the kernels that follow
  read null buffers.
- Warp's NanoVDB grid builder (`rebuild_count_points`, volume_builder.cu) and its run-length encode do not check
  their allocations either; its radix sort returns without raising. Before `native_guard`, on an L4 with ~2.5 GB free
  (a ballast array holding the rest), soil_dig at 64 worlds with the check off failed in the grid rebuild: "out of
  memory" printed, then `RuntimeError: Failed to create volume`, then error 700 on every later free.
- Warp's BVH and mesh builds (bvh.cu, mesh.cu) are unchecked too: a create writes through null pointers, and so does
  an LBVH rebuild. An MPM step's moving colliders refit their meshes. Since GAPS-NOW #5a (WP0, 07df81039) the guard
  reserves a create's scratch and the tree it keeps first (LBVH scratch 52 B and tree 84 B per item; a mesh's bounds
  24 B and winding data 288 B per triangle; cuBQL's terms doubled; 4 MiB for CUB). A rebuild or refit is bounded by
  the size its create recorded. The guard installs with the first MPM entry, so a scene without MPM (cloth only)
  still builds them unchecked.

Limits of the prediction (WP4's review of the first version):
- The graph rate beside a body applies whenever the scene has any body. It was measured with a moving body inside
  the material. Newton's collider matrix does not grow only near a body: it is sized by the reserved collider nodes
  (160 triplets per reserved cell, no distance test; WP0, GAPS-NOW #5a). On one L4, dough at 16 worlds as graphs, a
  press inside the dough, a press 1 m away and no press all peaked at 129.1 MiB per world. A far robot is therefore
  not charged early. The split between the rates with and without a body may be stale in that scene; it is kept,
  because a lower prediction would move scenes onto graphs without a law.
- The rate without a body is low for a pile that spreads: the sand pile above took 10.6 kB per particle as graphs,
  against 6 kB predicted. On one L4 (2026-11-10, the speed table's command) it runs at 1, 16 and 64 worlds (64: as
  graphs, 557 world-steps/s); at 256 worlds (1.97 M particles) the plan predicts 11.8 GB as graphs, they need over
  20 GB, and the step-down cannot carry it: after a graph out-of-memory the MPM step goes on with the grid sized for
  capture, so its eager periods need far more than an eager run from the start (about 6 GB here). It stops with a
  clean SparseMemoryError (the process keeps its GPU). Workaround: at most 64 worlds of it on an L4, or
  `profile="debug"`, which runs without graphs. Planned (0.1.1, physics ruling 2026-11-10): the no-body rate raised
  to 12 kB, with WP2's estimate rows re-checked, so such a scene starts without graphs.
- Above 64 worlds, MPM with a body is not validated, and the build warns (physics ruling 2026-11-11): Newton's sparse
  grid runs out of capacity (status 8) or memory. On one L4, a free 0.3 kg box on a 2 x 2 x 1 cm sand slab at 256
  worlds failed its grid rebuild with status 8 and stopped with a clean SparseMemoryError; at 1,024 worlds the runs
  stalled past 25 minutes, and once an out-of-memory in Warp's sparse transpose (`wp_bsr_transpose_device`, which the
  guard does not cover) ended in CUDA error 700. That range is also where the integer cell lookup's law has not run
  ([mpm-moving-container-many-worlds.md](mpm-moving-container-many-worlds.md)).
- The free memory is read after Warp's mempool hands back what it holds unused. It does not include the graph
  memory pool, so a process still holding another simulation's captured periods reads low.
- The check first runs a full garbage collection and trims Warp's graph pool, then reads the free memory. A
  simulation dropped earlier in the process (an earlier test, a notebook cell, a sweep's last world count) is a
  reference cycle: its device arrays and graphs stay allocated until Python's cycle collector runs, which it rarely
  does in a long process (rc3: one process running the coupling and integration GPU tests read 11.8 GB free before
  256 dough worlds, which were refused; a one-world liquid scene left 3.8 GB held after `sim.close()` on a T4).
  Until 2026-11-10 the collection ran only when the mode asked for did not fit, so a sweep's leftovers let graphs
  start in what was left. A simulation that is still referenced is not freed: drop it (`del sim`, and anything that
  holds it), or build in a fresh process.
- After any CUDA allocation failure in a process, the next Warp native built on CUB (a prefix sum, a radix sort, an
  array sum, a run-length encode) fails once in its sizing call with "CUDA error 101: invalid device ordinal",
  whatever memory is free again; kernels and allocations in between do not clear it (Warp 1.18, driver 580, one T4:
  8 orders of calls). A prefix sum then raises; a sort returns with its keys unsorted, a sum and an encode with
  their output unwritten. Alloy takes that failure with one tiny prefix sum
  (`alloy.integration.native_guard.absorb_stale_error`) where it handles an allocation failure and goes on: the
  graph step-down, and before the next guarded native after a refused one. Code of yours that catches a CUDA
  out-of-memory error and goes on in the same process should call it too. Warp's native code prints the failure it
  takes ("Warp CUDA error 101: invalid device ordinal (in function scan_device ...)") on stderr once, after the
  step-down's out-of-memory warning: that line is the absorbed error, and stepping goes on.

The guard wraps Warp's native function only on the audited Warp (1.18) with the audited argument types. On another
Warp it logs a warning and stays off, and the prediction is the only protection.

## Workaround

- For graph speed on fewer worlds: fewer worlds, or fewer particles (a coarser spacing).
- A scene refused in both modes: fewer worlds (the error says how many fit) or a bigger GPU.
- `alloy.estimate(scene, worlds)` gives the graph-mode MPM memory before a build (OSS-3, 6ea274528); for many worlds
  the eager rate (3 kB per particle) applies instead.

## Fix

- Upstream Warp: check every native scratch allocation and raise (sparse build, grid builder, run-length encode, BVH,
  mesh), and raise when a sort cannot get its buffer. Clear the CUDA runtime's error after a failed allocation, and
  check the sizing calls of the sort, the sum and the encode (not written up for upstream yet).
- Alloy: done for scenes with an MPM entry (BVH and mesh guards, GAPS-NOW #5a). Install the guard in the wiring
  as well if a cloth-only scene path ever builds them per step.
- Alloy, memory (SCR-WP12-3 follow-ups):
  - done (a2): the graph grid is reserved from the cells in use, grown on overflow. The check still predicts graphs at
    one cell per particle, on purpose: 64 dough or sand worlds then fit as graphs but ran 3.0-3.1 s per period
    against 1.8-2.0 s eager, so scenes past that size stay eager;
  - a per-scene prediction of the collider matrix (from the bodies' footprints in the grid), instead of one rate for
    any scene with a body.
