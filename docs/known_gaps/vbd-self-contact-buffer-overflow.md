# VBD self contact drops pairs over its fixed buffers

## What is wrong
- Newton's VBD self-contact detection stores each vertex's colliding triangles and each edge's colliding edges in
  fixed per-element buffers: 32 and 64 entries (`particle_vertex_contact_buffer_size`,
  `particle_edge_contact_buffer_size`). Since 2026-11-10 Alloy sizes those rows to the scene by default and grows
  them before they drop a pair ("Rows sized to the scene" below). What is left is the limits listed there: rows at
  their cap, and the solvers that keep the fixed rows.
- Pairs beyond a buffer's capacity are dropped. Since 2026-11-08 health flags the world that dropped them:
  SELF_CONTACT_OVERFLOW, DEGRADED for the rest of the run, with `check()` still ok (`docs/guide/health.md`). Since
  2026-11-10 it also measures whether the surfaces then passed through each other, INVALID on request ("Pass-through
  after dropped pairs" below).
- The rows hold every pair in the VBD entry: within one cloth or soft body and between two of them.
- A dropped pair is never resolved, so the two surfaces can pass into each other.

## Number
- SOFT-3's skin pad on a fat pad (CPU, 1.25 ms, margin 1.38 mm), with the reach set to 7.35 mm (an uncapped 5 m/s
  motion bound, a test of the reach rule):
  - 1,890 vertices and 4,527 edges were over capacity: up to 85 of 32 and 151 of 64;
  - the skin reached 1.02 mm into the fat;
  - measured with a development overflow script (not shipped).
- At the route's reach, the same scene peaks at 14 of 32 and 24 of 64: 3.53 mm under the 0.75-edge cap (2.76 mm
  before 2026-11-08: 2 of 32, 9 of 64).
- Not measured: deep piles of folded cloth, where many layers lie within one reach.

## Why
- The buffers are preallocated per element and sized once at solver construction.
- The detector counts every pair it finds (`TriMeshCollisionInfo.*_count`) but writes only up to the capacity.
  `alloy/precision/self_contact.py` reads the overflow after every detection.

## Workaround
- Nothing to do for most scenes: rows sized to the scene are the default. `ALLOY_SELF_CONTACT_ROWS=fixed` in the
  environment (every VBD entry of the process), or `alloy_self_contact_rows="fixed"` in one VBD entry's kwargs,
  goes back to Newton's fixed rows.
- Keep the self-contact reach under one rest edge length. The route's rule caps it at 0.75.
- To check a scene, read `sim.health()`: SELF_CONTACT_OVERFLOW names the worlds that dropped pairs.

## Rows sized to the scene (the default since 2026-11-10)
- What it does (`alloy/solvers/vbd/self_contact_rows.py`, `alloy/integration/row_growth.py`):
  - each row has its own size, never under 32 / 64;
  - at the end of every control period, rows more than 75 % full grow, before they drop a pair;
  - a period in which a row did drop pairs is run again on the larger rows, so no accepted period has a dropped pair;
  - inside a period nothing is read on the host; a growth drops the captured CUDA graphs, which are captured again.
- Numbers (one world, CPU, 1 s, flat 16 x 16 cotton sheets of 0.3 m stacked 2.5 mm apart on a table; [internal note]
  `notes/gap-13b/measure/stacks.py`):

  | Sheets | Fixed 32 / 64: peak pairs | Fixed: dropped | Sized rows: peak pairs | Sized: dropped | Row memory |
  |---|---|---|---|---|---|
  | 4 | 24 / 45 | 0 | 24 / 45 | 0 | 1.8 MiB, unchanged |
  | 8 | 32 / 60 | 0 | 32 / 60 | 0 | 3.7 -> 5.5 MiB (grown once, before a drop) |
  | 12 | 40 / 76 | up to 8 / 12 per row, flag raised | 48 / 90 | 0, no flag | 5.5 -> 8.6 MiB (3 growths, 1 period re-run) |

  - A sheet folded over a bar and a sheet draped over a post peak at 8 / 15: no growth, no array replaced, the same mean
    height to the last printed digit and the same wall time.
  - A scene whose rows stay under 75 % full steps bit for bit as with fixed rows (CPU law).
  - Wall time of the 12-sheet run: 5 % more than with fixed rows.
- Memory: 8 bytes per pair slot. A fixed vertex row is 256 B and a fixed edge row 512 B: about 1.8 kB per cloth
  particle, for every world. Growth is per row, so only crowded rows cost more.
- Limits (what is still wrong):
  - a row stops at 256 pairs, and a solver's rows at 4 times their fixed size in bytes (at least 256 MiB;
    `alloy_self_contact_row_bytes` in the entry's kwargs raises it). Past either, pairs drop as before, the build
    logs it once with the fix, and health raises SELF_CONTACT_OVERFLOW;
  - not covered: a solver whose rows belong to a collision pipeline (rigs built with `collision_pipeline`), and
    Newton's deterministic modes (their record arrays are sized for 32 / 64): both keep the fixed rows, and the
    solver logs it;
  - the triangle-to-vertex rows are not grown. Health's watch reads them too, so the flag rises if a pile fills
    them; in the stacks above they did not (no flag with sized rows);
  - a rig that drives the solver itself must run its first step outside a captured graph and detect self contact
    in every step, as Alloy's routes do;
  - with `rerun_worlds` (the side simulation that re-runs worlds) the side simulation's rows keep their build size:
    they do not grow, and a pair dropped inside it triggers no re-run;
  - a re-run starts from the period's start, so a run that grew differs from one that had the larger rows from the
    start only by the pairs the first attempt dropped; the first attempt is discarded;
  - a grown row is sized past its need (overshoot): growth at least doubles a row, so the largest rows ended at
    144 / 256 slots for a peak of 56 / 91 pairs (12 sheets, GPU). Tighter growth is open;
  - growing costs time once, in the periods that grow: a growth captures the CUDA graphs again, and a period that
    dropped pairs runs twice. On the 12-sheet, 16-world GPU run below that was 19 s more over 1 s of simulation
    (53 s against 34 s for the first half second); later periods ran at the fixed rows' speed;
  - `alloy.estimate` does not count the self-contact rows in its memory: 64 worlds of the 12-sheet stack hold about
    354 MiB of rows before growth and 591 MiB after (scaled from 2 worlds, AUDIT-STACK), against an estimate of
    146 MiB for the whole scene;
  - sized rows keep the pairs a crowded row used to drop; they do not stop sheets crossing in general. 12 or 20
    sheets falling together from 5 cm apart end with layers passed through each other in both modes, with no row
    over 24 / 46 pairs (no pair dropped; a separate gap, PHYSA-10);
  - these keep the fixed 32 / 64 rows: the side simulation of `rerun_worlds`, Newton's deterministic modes, rows
    owned by a collision pipeline (above), and every `SolverVBD` an Alloy module builds itself instead of through a
    scene's VBD entry (`alloy.solvers.vbd.make_vbd`): the cucumber-slicing job (`alloy/tasks/job_slice_sim.py`),
    the coupling and topology rigs, the test rigs.
- On a GPU (one L4, CUDA graphs on, 1 s, the same stacks in many worlds; [internal note] `notes/gap-13b/vmgpu/REPORT.md`):

  | Scene, worlds | Fixed 32 / 64 | Sized rows | Row memory | Warm world-steps/s, fixed -> sized |
  |---|---|---|---|---|
  | 4 sheets, 64 | no drop | no growth | 118 MiB, unchanged | 134 -> 134 |
  | folded sheet, 64 | no drop | no growth | 65 MiB, unchanged | 2,705 -> 2,634 |
  | 12 sheets, 16 | flag raised | no flag: 7 growths, 4 periods re-run | 89 -> 157 MiB | 191 -> 190 |
  | 8 sheets, 256 | flag raised | no flag: 4 growths, 1 period re-run | 945 -> 1,513 MiB | 103 -> 111 |

  - The periods that grew cost time once: the first half second of the 12-sheet run took 53 s against 34 s (re-runs
    and graphs captured again); the second half took the same 34 s.
  - GPU cloth that touches itself does not repeat bit for bit (`docs/guide/worlds.md`): two fixed-row runs of the
    4-sheet stack differ by 6 um, and a sized-row run differs from a fixed-row one by the same 6 um. The bit for bit
    law is a CPU law.
  - Not measured: the GPU's total memory at the peak (the table is the rows' own bytes).
- The default since 2026-11-10 (coordinator ruling 04:34Z): it changes nothing below the fill mark (CPU bit for bit,
  law R3) and replaces dropped pairs where a scene is crowded (law R1).

## Pass-through after dropped pairs (opt-in since 2026-11-10)
- Dropped pairs are often harmless, so SELF_CONTACT_OVERFLOW alone stays DEGRADED. On request health also measures
  whether they let surfaces pass through each other (`alloy/precision/self_contact_cross.py`), after every
  self-contact detection (16 per control period at `dt=0.01`):
  - every particle of a row that dropped a pair is marked (the vertex, an edge's two ends), for the rest of the run;
  - a world has passed through when two triangles that share no vertex intersect and one has a marked vertex, or
    when a marked vertex went through a triangle's inside to its other side since the detection before (two flat
    layers swap sides this way without any triangle pair ever intersecting).
  - `ALLOY_SELF_CONTACT_CROSS` in the environment at build (`CheckPolicy.self_contact_cross` at the engine layer):
    `off` (the default: no meter at all, every step as before), `report` (the count is in the world's health row,
    `self_contact.crossed_pairs`; the world stays DEGRADED) or `invalid` (such a world is INVALID, frozen as a blown
    cloth is, and `sim.check()` names it: "passed through").
- Laws (CPU, `tests/precision/test_health_self_contact_cross.py`; world 1's rows shrunk after build, world 0 keeps
  its rows as the control):
  - two sheets stacked on the ground, rows of one pair: world 1 drops up to 14 pairs a row, nothing passes through,
    the sheets rest 2.0 mm apart: DEGRADED with the switch on;
  - a sheet dropped tilted onto one draped over a stand, rows of no pair: world 1 crosses from step 9 (72 triangle
    pairs): INVALID and named with the switch on; world 0 never crosses;
  - the stack with a 1 kg upper sheet, rows of no pair: it sinks flat through the lower one from step 4 with no
    triangle pair intersecting; only the side-change clause sees it (it read 0 before that clause).
- Miss test (two sheets, 6 ways of loading them, rows of 0 and 1 pair, 100 steps; [internal note]
  `notes/gap-13a/vm/miss_test.py`): all 6 runs with rows of no pair passed through, and each was caught by the step
  its layers inverted; no run with rows of one pair swapped or crossed; the control world never.
- Cost with the meter on (CPU, 4 worlds of two 20 x 20 sheets): 0.4 % per step when nothing drops (no marked
  vertex: no search), about 2 % with a quarter of the particles marked. Off (the default): none, and no step
  changes (CPU law: on and off step bit for bit).
- Scenes (CPU, the switch on `invalid`, [internal note] `notes/gap-13a/sweep.md`): of 22 library scenes, examples and
  cloth tasks run with the fixed 32 / 64 rows, only SOFT-3's skin pad on a fat pad (3 pairs dropped) and 8 or 12
  stacked sheets (8 and 16) dropped pairs; none passed through (crossed 0, DEGRADED as before, no new INVALID). With
  the default rows sized to the scene, those three drop nothing and the meter has nothing to judge.
- Not seen by design: sheets that pass through each other with every pair kept (no row dropped), as in PHYSA-10
  above; an auditor's sheet dropped onto another at full rows ended with 5 vertex columns up to 1.3 mm below the
  lower sheet's, health OK.
- Limits:
  - only surfaces whose rows dropped pairs are watched: sheets that pass through each other with every pair kept
    (12 or 20 sheets falling together, PHYSA-10, above) are not seen;
  - a particle that moved more than the self-contact query radius between two detections (a reset, a teleport) is
    not judged for that move; a vertex that went round a free sheet edge inside one such short move would read as a
    pass-through (not seen in the laws or the sweep);
  - a period that re-runs on grown rows puts the marks and counts back to its start, with the dropped count; an
    engine rollback of some worlds (a coupled re-run, a watchdog's recover) puts their counts back with health's
    arrays, and clears their marks and skips their side changes at the next detection: a pass that was thrown away
    leaves no crossing, at the cost of a possible miss;
  - the side-change clause needs a detection every step (Alloy's routes detect twice per step); with a longer
    detection interval it can only miss;
  - not run on a GPU yet: the meter is launches only on arrays made at attach, as the overflow count it extends.

## Planned fix
- Done (2026-11-08): the SELF_CONTACT_OVERFLOW health flag, DEGRADED. In a calm two-sheet stack with rows shrunk to one
  pair, one edge dropped up to 14 pairs and the sheets still lay 2.0 mm apart: dropped pairs do not always mean pass
  through, so the flag does not make a world INVALID.
- Done (2026-11-10): rows sized to the scene, above, opt-in first, then the default.
- Done (2026-11-10, opt-in): the pass-through law and meter above; INVALID with `ALLOY_SELF_CONTACT_CROSS=invalid`.
- Open: INVALID by default (needs the sweep on more scenes, a GPU run and a ruling); tighter growth (the overshoot);
  growing rows in the side simulation of `rerun_worlds`.
