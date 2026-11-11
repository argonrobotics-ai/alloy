# Known gap: worlds of a coupled scene share one exchange clock, so one world's events change every world's steps

## What is wrong
- A coupled scene (MuJoCo + VBD cloth or soft parts under the Scheduler) picks one exchange interval dt_x and one set
  of per-entry substeps for all its worlds each control period.
- Every per-world stability estimate feeds that choice by its minimum. VBD's is 0.4 · r / |v| over its fastest
  particle. So when one world's state changes the choice, every world steps differently from then on.
- That includes a world re-run after a force jump (SCR-WP11-4). The re-run is refined to dt_x / 2^k, and its own
  later trajectory can move the clock. The other worlds are restored exactly to their first-run state, then step on
  that changed clock and leave the trajectory they would have had alone.
- Same seed, same device stays bit-for-bit (the whole scene is deterministic). Only "this world does what it would
  do by itself" does not hold.
- A re-run steps every world. The worlds that did not trip are computed again at the finer step and then put back,
  so they pay that period's compute twice. That holds for a force-jump re-run, a world's first escalation (below)
  and a contact-band re-run (WP4, R2-E3).

## Escalations (WP4, SCR-WP3-C3RAND, 2026-11-08)
- Before: a world's escalation (its energy or residual rule tripped, not a force jump) raised the clock's floor for
  good, and on the second trip switched the pairs to their strong mode, for every world. XB4-9's randomized C3
  box-on-pad (64 worlds, L4) tripped in 1-11 different worlds in each of periods 1-5. The floor went from 0.156 ms
  to 0.0195 ms with staggered coupling for all 64 worlds, and 0.5 s never finished.
- Now (`alloy.coupling.ladder`): a world's first escalation re-runs that world's period at one grid level finer per
  re-run (up to 2), for that period only. The floor and the coupling mode stay, and the next period starts on the
  clock the scene's bounds pick.
- The lasting ladder still acts in two cases: the same world trips again within the watchdog window (1 s), or a
  period ends still escalated. A resting instability needs that.
- `sim.health()["refined_periods"]` counts the refined periods per world.
- The kept worlds stay bitwise on a run without the trip
  (`tests/coupling/test_period_ladder.py::test_kept_worlds_are_bitwise_a_run_without_the_trip`): the clock is not
  moved.

## Number
- WP12's towel on a box (`tests/coupling/test_rollback_bitwise.py`, CPU, 2 worlds, C3's per-body record gains):
  - the reference's VBD stable step at P+2 is 1.59e-4 s, 1.8 % above the 2-substep threshold (dt_x 1.5625e-4 s);
  - after a refined re-run of world 1 its towel drapes about 9 % faster (max particle speed 0.69 vs 0.63 m/s), so its
    stable step is 1.46e-4 s;
  - at P+3 the scene takes 4 VBD substeps instead of 2, and the kept world 0 leaves the reference in that period.
- `test_a_refined_world_changes_the_shared_clock_for_every_world` shows this. The restore itself is exact: on the
  reference's clock the kept world stays bitwise on it (`test_a_refined_rerun_leaves_the_kept_world_on_the_reference`).
- The demonstration sits on that 1.8 % margin. Sub-ulp changes to VBD velocities move it (SCR-WP11-F32, 2026-11-08:
  velocity from the true displacement), and then the re-run world no longer crosses the threshold. The test is
  xfail(strict=False) until it has a scenario with margin. The clock is still the minimum over worlds.

## Why
- The Scheduler steps all worlds in one launch per exchange with one dt_x and one substep count per entry. Per-world
  substeps would need per-world loop bounds in every coupled kernel and graph.

## Workaround
- To compare worlds as independent experiments, keep them in separate builds (or `processes=` on the CPU, one world
  per process), or read results only from periods before any re-run (`traj.meta["health"]`, rollback counts in
  `sim.engine.rollbacks`, refined periods per world in `sim.health()["refined_periods"]`).
- Randomized worlds in one build are fine for statistics. Just do not treat one world's trajectory as what it would
  do alone.

## Planned fix
- Per-world substep counts (a per-world loop bound in the coupled kernels and their graphs), or a clock chosen from
  the unaffected worlds plus a separate pass for the refined one.
- Estimate: 1-2 weeks across the coupling layer. Not scheduled.
