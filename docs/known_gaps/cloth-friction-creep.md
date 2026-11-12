# Cloth held by friction creeps out of a pinch below the Coulomb limit

## What is wrong
- Cloth held between rigid pads or fingers by friction alone (`cloth_vbd` route) slides out slowly under a steady
  load, even when the load is well under the Coulomb limit mu N. A real pinch below that limit does not move.
- The creep speed does not depend on the step size. It grows with the load: at half the limit it is about 0.3 x
  `friction_epsilon` (2.9 mm/s with the default 1e-2 m/s), and at 0.9 of the limit about 0.7 x (6.8 mm/s).
- Over seconds that is centimetres: a corner held for a 10 s carry at half the limit slides about 3 cm.
- Nothing flags it: `sim.health()` reads OK.

## Number
CPU, plain VBD (`SolverVBDAlloy`), main 8921e7718 (2026-11-10). A 4 x 12 cm cloth strip (1 cm cells, 1 g vertices,
no gravity) hangs between two static pads that press each of its top 15 vertices with 1 N per pad (mu 0.3, so a
limit of 0.6 N per vertex). A downward load on those vertices, ramped on over 0.1 s, at a fraction of the limit.
Creep speed over the second half of a 1 s run (development notes, not shipped):

| Load / limit | `friction_epsilon` | Iterations | Stock creep | Smoothed law | With the stick |
|---|---|---|---|---|---|
| 0.25 | 1e-2 | 10 | 1.34 mm/s | 1.34 mm/s | 0.0000 mm/s |
| 0.5 | 1e-2 | 10 | 2.92 mm/s | 2.93 mm/s | 0.0000 mm/s |
| 0.7 | 1e-2 | 10 | 4.65 mm/s | 4.52 mm/s | 0.0000 mm/s |
| 0.9 | 1e-2 | 40 | 6.82 mm/s (0.5 s run), then off the pads | 6.84 mm/s | 0.0000 mm/s |
| 0.98 | 1e-2 | 40 | off the pads in 1 s | 8.59 mm/s | 0.0000 mm/s |
| 0.5 | 1e-3 | 10 | 0.36 mm/s | 0.29 mm/s | 0.0000 mm/s |

- At 0.9 the stock pinch crept about 7 mm in the first second, off the bottom of the pads, and the strip fell. With
  the stick its rows moved 3 um in all (the strip's own stretch) and then not at all.
- With `friction_epsilon` 1e-3 the friction term is ten times stiffer: at 40 iterations the stick holds 0.9 of the
  limit but slips out at 0.98, as stock does (the solve does not converge so close to the limit; `rows_it40.txt`).
- Past the limit (1.3x) both slide out at the same kinetic friction: the drops after 1 s agree within 0.2 %.
- Coupled, per-vertex contacts (a gripper's two MuJoCo pads at 15 N on a hanging double towel, the lagged coupler, 10
  iterations, a load of 0.3 of the limit on the 10-12 particles between the pads; a development
  script, not shipped): stock creeps 1.21 mm/s, every clamped particle drifting; with the stick
  7 of 12 stay within 0.03 mm and a few at the pads' edges shift both ways, a mean of 0.19 mm/s. The force the
  coupler hands the pads equals the load in both (3.84 N against 3.82 N).
- Coupled, full-surface contacts as a gripper's fingers get them by default (a pad rig, `run_hold`: two MuJoCo
  pads at 15 N, the route's VBD at 40 iterations, `friction_epsilon` 1e-2, 1 s hold; CPU): stock at half the limit
  slides 2.99 mm and creeps 2.97 mm/s; with the stick 0.000 mm/s at 0.5 and at 0.9 of the limit (0.07 mm once, while
  the load comes on), and past it (1.2x) the cloth slides out of the pads as with stock.

## Why
- VBD's friction is IPC's smoothed Coulomb law on the slip within one step, measured from the step's start. Below a
  slip of `friction_epsilon` x dt the friction grows smoothly from zero, so holding a load T needs a slip u with
  mu N (2 - u / eps_u) u / eps_u = T in every step: a creep of friction_epsilon (1 - sqrt(1 - T / (mu N))).
- The smoothing keeps the solve stable. A smaller `friction_epsilon` creeps less but stiffens the friction term.

## Workaround
- Opt in to stick friction: `body_particle_friction_stick=True` in the cloth's VBD entry kwargs, or
  `ALLOY_FRICTION_STICK=1` in the environment for every VBD entry of a process (`alloy/solvers/vbd/stick.py`).
  - Each cloth vertex keeps an anchor on each of up to three shapes it touches, and its friction slip is measured
    from where it stuck, not from the step's start. The anchor moves only by the slip past eps_u, so a held vertex
    stays put below the limit and slides with the same kinetic friction past it. A full-surface contact (a gripper
    finger's edge or face against the cloth's faces) reads its corners' anchors, mixed by their weights.
  - An anchor outlives its contact by 0.02 s, so a finger that chatters (lifts off and touches again) keeps it.
  - No vendored change: the anchors shift the contact records' body velocity, so the coupler's feedback to a gripper
    sees the same friction as the cloth.
  - A sheet on a moving plate goes with it as with stock friction (a 10 cm sheet on a kinematic plate, CPU;
    development notes, not shipped): driven at 0.05 m/s it lags 0.33 mm (stock 0.35), at 0.3 m/s
    11.3 mm (stock 11.3), and closes to 0 after the stop; tilted to half the friction angle it moves 0.29 mm while
    tilting (stock slides 3.7 mm downhill) and then holds (stock creeps 3 mm/s). An early version of the stick let a
    driven sheet fall behind at `friction_epsilon` (10 mm/s); a law pins that.
- Limits of the stick:
  - a fourth shape on one vertex keeps the smoothed friction on that contact;
  - at 10 VBD iterations a load past about 0.7 of the limit that comes on suddenly slips out with or without the stick
    (the friction solve does not converge from a far first guess); at the cloth route's 40 iterations it holds.
- Without the stick: a lower `friction_epsilon` in the entry kwargs (1e-3 creeps ten times slower), or keep holds
  short.

## Planned fix
- The stick by default for cloth, after the grasp law and the cloth environments pass with it (a planned decision).
