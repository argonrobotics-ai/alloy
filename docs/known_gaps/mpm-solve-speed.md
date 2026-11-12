# MPM at many worlds: the rheology solve sets the speed (256 dough worlds ~7.6 s per 10 ms period on an L4)

Status: open. The solve tolerance is 3e-4 by default since 2026-11-08 (22 % faster for dough), after a law
re-pass. A faster solver is planned after launch (2026-11-08).

## What

An implicit-MPM step (pastes, dough, sand and soil) solves the material's stress with Newton's projected Gauss-Seidel
until the residual is under a tolerance or 250 iterations. That solve is over half of an eager dough period at 256
worlds (57 % at the default tolerance, 64 % at 1e-4). More worlds cost proportionally more: the GPU is already full of
work at 256.

## Number

L4, 256 worlds, eager periods (the memory check's choice at this size), 10 ms periods (the rheology-default rows, not
shipped, 2026-11-08):

| scene | tolerance 1e-4 (before) | tolerance 3e-4 (default) | memory per world |
|---|---|---|---|
| dough ball pressed by a plate | 9.78 s | 7.60 s (-22 %) | 37 MiB |
| sand fill | 6.73 s | 6.69 s (-1 %) | 45 MiB |

Where a period goes (same L4, 256 dough worlds eager; Newton's own phase timers, which
synchronize, so a period reads about 1 s slower than untimed):

| phase | tolerance 3e-4 | tolerance 1e-4 |
|---|---|---|
| whole period | 8.64 s | 10.44 s |
| strain (rheology) solve | 4.91 s (57 %) | 6.71 s (64 %) |
| allocate grid | 1.13 s | within 2 % |
| elasticity | 1.05 s | within 2 % |
| rasterize collider | 0.60 s | within 2 % |
| scratchpad, strain matrix, the rest | 0.31, 0.14, each other under 0.12 s | within 2 % |

The solve's loop (16 solves, counted on the host) averages 203 sweeps at 1e-4, with 9 of 16 at the cap. At 3e-4 it
averages 130, with 2 of 16 at the cap. So the 22 % from the looser tolerance is all the solve's.

One world on the CPU: a pressed dough ball takes 210 iterations per solve on average and often hits the 250 cap
(0.2 s, 160 solves); at 3e-4 it takes 166. The sand fill converges in fewer iterations, so the tolerance barely
changes its time.

## Why

A Gauss-Seidel sweep moves information about one cell. A body many cells across therefore needs on the order of the
square of its width in sweeps before its bulk mode settles. Newton's other solvers do not help:
- the struct-of-arrays Gauss-Seidel is bitwise the same and no faster;
- Jacobi always hits the cap;
- conjugate residual followed by Gauss-Seidel takes as many iterations and is 2.1x slower;
- Newton refuses conjugate gradient with contacts.

## Workaround

- The default tolerance is 3e-4, a 3x looser residual than Newton's 1e-4. `ALLOY_MPM_TOLERANCE` changes it when a
  scene is built (`alloy.integration.mpm_solve`): `ALLOY_MPM_TOLERANCE=1e-4` for the tighter solve. A scene's own MPM
  solver arguments win over both.
  - Effect: a pressed dough ball's particles move by mean 0.07 mm, max 0.52 mm against 1e-4, 2.4x the solver's own
    floor (1e-4 against 1e-5: mean 0.03, max 0.22 mm).
  - Validation: every MPM law and the dough, paste and soil tasks pass at it within their own bands.
  - Accepted range: 1e-5 to 3e-4. Looser values are refused, because every solve would stop unconverged.
- Fewer worlds per GPU also help, since time scales with worlds once the GPU is busy. The CPU runs one world per core
  (`processes=`).
- Not offered: Newton's older trilinear collider basis (`collider_basis="Q1"`). It ran 256 dough worlds in 6.14 s
  (-37 %) and sand in 4.90 s (-27 %), but it failed its contact laws (Newton moved to the quadratic S2 basis for
  contact quality):
  - a paste layer below its stop thickness slid 28 mm (law: under 1 mm);
  - a strip footing at 0.8 of Prandtl's bearing load sank 60 mm (law: under 1 mm);
  - the soil cone factor fell to 4.3-5.3 (law: 5-35, measured 7.5-11.9 past one radius);
  - the paste slump read 0.321 against Pashias' 0.235 (law: within 0.07);
  - the dig scoop lifted 2.7 g instead of 8.0 g and poured none into the bucket.

## A robot or a free body beside MPM (robot + sand)

A Panda plunging its fingers into a 4 cm tray of dry sand and dragging them (35,000 grains), L4, CUDA graphs, per 10 ms
control period while the fingers are in the sand, one build (main, 2026-11-09):

| `alloy.build(mpm_coupling=...)` | 1 world | 64 worlds |
|---|---|---|
| "standard": 2 passes per exchange, 1.25 ms exchanges | 0.365 s | 13.24 s |
| "auto" (the default; robot-only scenes): 1 pass, 1.25 ms | 0.188 s (1.9x) | 7.02 s (1.9x) |
| "fast": 1 pass, 2.5 ms (opt-in) | 0.107 s (3.4x) | 3.83 s (3.5x) |

The fast row comes from the commit that added it (cd4c7e4c1; the same engine code for fast). Before the proxy-body fix
below, rc2-1109 took 1.336 s (1 world) and 18.84 s (64 worlds) for the same scene.

- **Fixed (default since 2026-11-09):** the MPM solver never wrote its proxy bodies to its output, so the coupler read
  a moving robot's travel as a coupling error and halved the exchange step for the rest of the run
  (`alloy.integration.mpm_bodies`; `ALLOY_MPM_BODY_CARRY=off` restores the old output).
- **auto** (the default; `alloy.integration.mpm_clock`, `mpm_robots`): one coupling pass per exchange instead of two,
  at the standard exchange step, when every coupled pair ends in MPM, the scene has a servo-driven robot, and every
  body that starts within 2 cm of the material is that robot's link or welded; otherwise standard. Against standard on
  the rows above: finger end height within 0.03 mm; the sand surface within 0.19 mm on average (95 % of 5 mm columns
  within 0.8 mm; two standard runs differ by 0.03 mm), no grain spilled either way; no re-run or escalation. It is
  decided at build: a free body that reaches the sand later also runs one pass. A 2 cm steel block dropped 4 cm onto
  sand rests within 0.5 mm of standard and moves under 1 cm/s from 0.3 s (tests/integration/test_mpm_clock.py).
- **fast**: also 2.5 ms exchanges. A robot held by its servos shows no change (finger end height within 0.2 mm), but a
  free body does: the dropped block rests 3.3 mm deeper, and a heavy cup resting on sand bobs by about 15 µm (up to
  7 mm/s) for about 0.3 s. Use it for robots and tools in sand or paste, not for free objects that must rest on it.
- `sim.routes()` names the setting and why on the MPM object (`"mpm_coupling"`); `ALLOY_MPM_COUPLING` overrides it for
  a whole run.
- Where the time goes, 64 worlds in contact (standard): MPM 90 % (MuJoCo 10 %); in MPM, the strain solve 72 %,
  elasticity 7 %, collider rasterization 6 %, grid allocation 5 %.

## Planned fix

- **Tolerance 3e-4 as the default**: done (2026-11-08), in its own commit after the law re-pass above.
- **A faster solver, measured** (an Anderson-acceleration probe). Anderson-accelerated Gauss-Seidel takes 1.6-2.4x
  fewer sweeps per solve on the CPU (1 and 2 worlds; the dough within mean 0.06 mm of GS). On the L4 at 256 worlds the
  loop drops from 130 to 84 sweeps, but the untimed periods are equal (7.56 against 7.59 s). Not merged.
- **Converged worlds are skipped: done** (5fd256eaf). Newton's Gauss-Seidel sweep skips the worlds that the
  per-world stop has frozen, so results are bitwise the same. L4, 256 dough worlds:
  - presses at spread heights, so the worlds stop at different sweeps: 7.87 against 10.10 s per period
    (strain solve 4.16 against 6.39 s);
  - identical worlds: no change.
- **After the solve:** grid allocation, elasticity and the collider rasterization take about 2.8 s of a timed 8.6 s
  period together.
- **Not planned**, by decision: an iteration cap of 100, tolerance 1e-3, or half-precision solves.
  Each leaves the solve unconverged or below its own resolution.
