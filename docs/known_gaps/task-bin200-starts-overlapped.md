# The shipped task_bin200 pile starts with 54 capsule pairs 8 mm inside each other

**What is wrong.** `library/robots/task_bin200` (200 capsules in a bin) starts from a recorded layout in which 54
pairs of capsules overlap. They push apart in the first steps, so the pile moves at the start although nothing was
dropped, and health reports the overlap: START_PEN at the start and PEN_WARN (deeper than 1 mm) until the pile has
relaxed. The world is DEGRADED during that time, never INVALID.

**The number.** 54 pairs start 7.97 mm deep (the deepest: `o158` with `o194`). Measured on the CPU's MuJoCo C route:
6.57 mm after the first step, 0.79 mm at 0.05 s, under 1 mm for good after 0.89 s, 0.02 mm at rest (4 s). The depth
never reaches 1 cm. The other bin packs (`task_bin10`, `task_bin50`, the box bins) are dropped piles that start
without overlap. Not measured: this pack on MuJoCo Warp.

**Why.** The layout is the recorded one (trace r5), kept so that the pack and its trace match the recording. The
recording's own contact model let the capsules sit that deep.

**Workaround.** Let the pile relax before measuring: `sim.settle(1.0)` runs 1 s and makes the relaxed pile the state
`sim.reset()` returns to. Or start the measurement after 0.9 s.

**Planned fix.** After the release tag the relaxed layout becomes the pack's start, and the trace and its goldens
are made again from it.
