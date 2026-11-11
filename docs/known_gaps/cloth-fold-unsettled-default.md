# A folded cloth keeps moving under the default cloth accuracy

## What is wrong
- When cloth plies rest on each other (a towel folded in thirds, a sheet folded in half), the stack does not come to
  rest under the default cloth accuracy. After the grippers let go, the plies keep creeping, where they should settle
  within a fraction of a second. Since the default holds self-contact particles out of the Chebyshev step
  (2026-11-09), the stack's height stays under the task's gate, but the plies still move at 0.03-0.06 m/s.
- How high the stack ends up depends on the draw. A 1 µm nudge to the start changes whether the task's 25 mm
  normal-extent gate passes. On the GPU, a repeat run is another draw, because self-contact sums with atomics.
- A single sheet resting on a body is not affected. A sheet draped on a stand settles: drape coefficient 0.728 at
  0.36 % stretch.

## Number
Measured on the thirds fold (`alloy.tasks.laundry_fold.towel_rig("thirds")`): the library terry towel at bath-towel
size, 700 x 450 mm, with self contact, dt 1.25 ms, one world.
- Each run's start is nudged by up to 1 µm in x-y per particle, from its seed. Seed 0 is the law's own start.
- The extent is measured at the law's point, 0.75 s after the last release.
- The speed is the mean particle speed 1.5 s later.
- Data: WP11, 2026-11-09, main 7f7e9a937 and F32 3057f60a7. Measured with a development script; its
  rows for the spread, settle and proto runs are in the development notes (not shipped).

| solve | runs | normal extent at the law's point | over 25 mm | mean speed 1.5 s later |
|---|---|---|---|---|
| default (adaptive start, Chebyshev rho 0.9, 40 iterations) | 20, CPU + GPU | 12.7-39.0 mm | 8 of 20 (main alone 3 of 10) | 0.065-0.103 m/s (2 runs) |
| default at 100 iterations | 2, GPU | 28.9-34.5 mm | 2 of 2 | 0.075-0.122 m/s |
| Chebyshev off (adaptive start, 40 iterations) | 5, GPU | 14.0-24.8 mm | 0 of 5 | 0.003-0.059 m/s |
| prototype: self-contact particles not extrapolated | 4, GPU | 17.3-29.7 mm | 1 of 4 | 0.035-0.073 m/s |
| WP2's CPU sweep 2, the default before 2026-11-09 | 8, CPU | 20.6-38.1 mm | 5 of 8 | 0.078-0.093 m/s |
| WP2's CPU sweep 2, self-contact hold (the default since) | 8, CPU | 14.1-22.2 mm | 0 of 8 | 0.033-0.064 m/s |

WP2's sweeps ([internal note] `notes/wp2-fold`, `vm_out2a`, `vm_out2b`) use the same starts and measure the speed as the
mean over the last 0.5 s of a 1.5 s settle.
- Since the float32 velocity fix (the VBD velocity no longer rounds to ulp(x) / dt; a cloth falls at g at every
  substep), the same 8 starts under the hold read 2 of 8 over 25 mm, at 26.0 and 30.5 mm; two starts are at rest.
  The thirds fold's 25 mm law is therefore an expected failure that may pass, with a hard guard at 35 mm
  (`tests/tasks/test_laundry_fold.py`).
- 1.5 s after the law's point, 1 of 8 starts is over 25 mm under the hold, against 6 of 8 before.
- While the plies move, a single edge can spike. At the law's point the largest edge elongation is under 3 % on most
  starts, but one start under the hold read 27 % for that instant (the fold law allows 30 %) and 0.5 % 1.5 s later.
  Chebyshev off and the wider holds showed 9-14 % spikes too; the default before 2026-11-09 showed up to 6 %.

On one GPU start under the default, the extent swings between 31 and 57 mm for the whole 1.5 s. Particle speeds peak
at 0.5-2.5 m/s throughout. A three-ply stack at rest is about 14 mm high.

## Why
The cause is partly known.
- The default's Chebyshev step (`alloy/solvers/vbd/accel.py`) extrapolates each particle's iterate. It skips
  particles held by a body, shape or attachment contact (`AccelState.mark_held`), but it still extrapolates
  particles held only by cloth-on-cloth contact, which are the fold's plies. The extrapolation over-relaxes there,
  and the plies keep moving. With Chebyshev off, no run fails the gate.
- That is not the whole cause. Holding self-contact particles out of the extrapolation (WP11's prototype) helps
  less than turning Chebyshev off. And with Chebyshev off, two of five starts still move at 0.03-0.06 m/s after
  1.5 s.
- That leftover jitter is VBD self contact's own. WP2's CPU sweep (4 starts per solve, 2026-11-09) found it in stock
  VBD at 10 iterations, with no Alloy change: 0.067-0.089 m/s, no start at rest. A towel at rest reads 0.003-0.006
  m/s. The adaptive start damps it rather than causing it. With Chebyshev off, 2 of 4 adaptive starts came to rest,
  against 1 of 4 inertial starts at 40 iterations.
- More iterations do not help: 100 iterations under the default jitter as much as 40.
- VBD's collision damping feeds it. Self contact and cloth-on-body contact damp only on approach (`soft_contact_kd`,
  Newton's default 10 N s/m). WP2 restarted a rested thirds fold for 0.5 s (one start, CPU), measuring the mean speed
  over the last 0.25 s:
  - under the old default: 0.061 m/s, and 0.022 at kd 0;
  - under the hold: 0.021 m/s, and 0.0015 at kd 0, at rest;
  - with Chebyshev off: 0.006.
  Once a fold jitters, kd 0 does not stop it (0.057-0.061 m/s), and kd 100 makes it worse (0.113).

## Workaround
- In the engine layer (tasks, rigs), turn the Chebyshev step off: the VBD solver kwarg `alloy_chebyshev_rho=None`,
  e.g. `towel_rig("thirds", vbd={"alloy_chebyshev_rho": None})`. That gave 0 of 5 over the gate, at the default's
  40 iterations. Some starts still creep (Why).
- The self-contact hold (VBD solver kwarg `alloy_chebyshev_hold_self_contact=True`) is in `cloth_accuracy` "default"
  and "super" since 2026-11-09: 0 of 8 starts over the gate, against 5 of 8 before. The plies still move (Number).
- Do not turn collision damping down for a whole fold (`model.soft_contact_kd`): a rested stack calms, but the fold
  itself then fails its gates on some starts (Planned fix, step 3).
- `cloth_accuracy="fast"` also turns Chebyshev off, but it runs 10 iterations (12 % stretch on a drape), and it has not
  been measured on a fold.
- Score a fold's height on its time average, or after a longer settle, and check particle speeds before trusting
  one frame.

## Planned fix
WP2 owns the fix, in the accelerated cloth solve; it follows the 2026-11-09 launch.
1. Separate the jitter left with Chebyshev off from what Chebyshev adds. Done: the leftover is self contact's own
   (Why).
2. Fix both. Done for Chebyshev: the hold is the default since 2026-11-09. It passed the thirds-fold seed spread
   and the drape laws, CPU and L4. Wider holds did no better: one or three edge rings, 1 of 8 over the
   gate each; the whole sheet while it touches itself, 2 of 8.
3. Self contact's own jitter: collision damping is the lead (Why), but turning it down is not the fix. WP2's CPU
   sweep 3 ran the whole thirds fold from 8 starts ([internal note] `notes/wp2-fold/vm_out3`):

   | solve | fails a fold gate | at rest 1.5 s later (under 0.006 m/s) | mean speed |
   |---|---|---|---|
   | hold (the default) | 0 of 8 | 0 of 8 | 0.033-0.064 m/s |
   | hold, kd 0.01 | 1 of 8 (extent 29 mm) | 3 of 8 | 0.000-0.057 m/s |
   | hold, kd 0 | 3 of 8 (extent 37 mm; three-ply share twice) | 1 of 8 | 0.000-0.057 m/s |
   | no hold, kd 0.01 | 3 of 8 (extent) | 1 of 8 | 0.002-0.065 m/s |
   | no hold, kd 0 | 4 of 8 (hems outside three times; extent) | 1 of 8 | 0.001-0.065 m/s |

   Less damping lets some starts rest, but the moving fold needs it: the hems land off the stack. Next: damping that
   stays on during the fold and does not feed a resting stack.

The thirds-fold extent law stays xfail (strict=False), citing this page, until the plies come to rest.
