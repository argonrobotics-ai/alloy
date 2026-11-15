---
owner: Alloy maintainers
---

# A light folded cloth keeps creeping under the default cloth accuracy

## What is wrong
- Since 2026-11-14 a folded cloth under the default cloth accuracy (and "super") comes to rest: a towel folded in
  thirds settles within its gates and stops moving. What is left is a sheet that is light per vertex:
  - a towel at a tenth of its weight still creeps after the fold, at every cloth-on-cloth stiffness tried;
  - the cloth-on-cloth stiffness follows the median particle mass of the cloth and soft bodies on the VBD route
    (the scene's cloth solver), not each sheet's own, so a sheet more than about 9x lighter per vertex than that
    median may not rest folded.
- On the GPU, a repeat run of a fold is another draw, because self contact sums with atomics. On the CPU every run
  repeats bit for bit.
- A single sheet resting on a body is not affected. A sheet draped on a stand settles.

## Number
Measured on the thirds fold (`alloy.tasks.laundry_fold.towel_rig("thirds")`): the library terry towel at bath-towel
size, 700 x 450 mm, with self contact, dt 1.25 ms, one world, CPU (bitwise). Each run's start is nudged by up to
1 µm in x-y per particle, from its seed; seed 0 is the law's own start. The extent is measured at the law's point,
0.75 s after the last release; the speed is the mean particle speed over the last 0.5 s of a 1.5 s settle after it.
A towel at rest reads 0.003-0.006 m/s. Data: 2026-11-14, main 50cd06a51; the rows are in the development notes (not
shipped).

| solve | starts | normal extent at the law's point | over 25 mm | mean speed |
|---|---|---|---|---|
| default since 2026-11-14 (cloth-on-cloth ke capped at 400 m / dt², no collision damping) | 8 | 12.6-12.7 mm | 0 of 8 | 0.001 m/s on all 8 |
| default before 2026-11-14 (cloth-on-cloth ke 2e6 N/m, the towel's bulk K; kd 10 N s/m) | 8 | 14.6-30.5 mm | 2 of 8 | 0.003-0.061 m/s |
| cap by hand, towel at 0.1x mass | 8 | 26.8-27.1 mm | 8 of 8 | 0.018-0.020 m/s |
| cap by hand, towel at 10x mass | 8 | 12.5-20.6 mm | 0 of 8 | 0.000-0.003 m/s |

- Under the default since 2026-11-14 every fold gate passes on all 8 starts (three-ply share 0.965-0.967, stretch
  0.11 %, no crossings), and the stack keeps settling to 9.3 mm 1.5 s later.
- The 0.1x and 10x rows were run with the cap set by hand (2026-11-12 sweep, main bd9e6b412, self-contact ke 2e3 and
  2e5 N/m with no damping: 360 m / dt² at those masses).
- Cost: a sheet that folds onto itself steps 1.1-1.3x slower than before (a T-shirt's sleeve fold on one L4: 1.26x
  at 64 worlds, 1.12x at 256; 1.27x on the CPU); a sheet that does not, 1.00-1.04x (a drape at 1-256 worlds).

## Why
- VBD collides cloth with itself at the particle contact stiffness of cloth on bodies, which Alloy's route sets to
  the material's bulk stiffness (2e6 N/m for the towel), and damps every contact on approach (10 N s/m). For the
  plies of a fold that stiffness is about 36,000 m / dt², which the cloth solve's fixed 40 iterations do not
  converge, and the approach damping keeps feeding the jitter.
- Since 2026-11-14 the cloth solver (`alloy/solvers/vbd/base.py`, `SelfContactGains`) runs cloth-on-cloth contact
  at its own stiffness, capped at 400 m / dt² with m the median particle mass of the solver's cloth and soft bodies,
  and with no collision damping; contact with bodies keeps its values. The cap is one number for all of the
  solver's sheets (Newton's self-contact kernels take one stiffness per model), so a sheet far lighter per vertex
  than the median gets a cap that is, for it, many times 400 m / dt². At 3,600 m / dt² the bath towel still rested;
  at 10x its mass and 3,600 it did not on 4 of 8 starts.
- The 0.1x towel creeps at every cloth-on-cloth stiffness tried (2e3-2e5 N/m): a separate limit, not the cap.
- The default also holds self-contact particles out of the Chebyshev step (since 2026-11-09): without that a fold
  kept bouncing.

## Workaround
- Keep the per-vertex masses of the cloth in one scene within about 9x of each other: a coarser mesh for a light
  sheet (heavier vertices) or a finer one for a heavy sheet. This follows from the rule above; it is not measured.
- In the engine layer (tasks, rigs), the cap and the damping are VBD solver kwargs: `alloy_self_contact_ke_ratio`
  (None: the model's stiffness) and `alloy_self_contact_kd`.
- Score a light sheet's fold on its time average, or after a longer settle, and check particle speeds before trusting
  one frame.

## Planned fix
- A per-sheet cap (each particle's own sheet's mass) needs a per-particle stiffness inside Newton's self-contact
  kernels; it is not scheduled.
- The 0.1x towel's creep: not yet understood.

The thirds fold's 25 mm extent law passes under the default since 2026-11-14 on all 8 starts, and is a plain law
since then (`tests/tasks/test_laundry_fold.py`; it was an expected failure while the towel was still moving).
