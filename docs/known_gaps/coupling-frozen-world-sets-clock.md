# A frozen world still sets a coupled scene's clock, so a start overlap can make each control step take minutes

## What is wrong
- Health freezes an INVALID world by putting it back to its period-start state after every exchange. The solvers still
  step it from there on every exchange, and in a coupled scene (cloth or a soft body beside a free rigid body or a
  robot: MuJoCo and VBD under the Scheduler) the shared exchange clock still reads it.
- A world that froze at its start because a sheet or soft body began inside a rigid part (or the ground) is the bad
  case. Every exchange pushes the same vertices out again, and the clock reads their speed: its contact-band bound
  and VBD's stable step halve the exchange step. At the finer step the same depth is undone in less time, so the
  speed reads higher again. Within three control steps the clock is at its finest step (`dt / 1024`) and VBD's
  substeps climb toward their top rung (64), on a world whose result is thrown away.
- The clock is shared, so every world of the scene steps at that rate (Number, 4 worlds).
- Without a free rigid body or robot there is no coupled clock: the frozen world then costs a bounded amount.
- A start overlap this shallow is not named: health's start-overlap check (START_PEN) judges rigid contacts, not
  cloth or soft-body vertices. The latch warning gives only indices ("particle 810 / shape 5").

## Number
One L4 and the CPU (Warp route), main bd9e6b412 (2026-11-11). The reproduction: a 15 cm `cotton_woven` sheet (10 g,
resolution 30) centred at z 0.103 over `add_open_box(inner=(0.10, 0.10, 0.10), wall=0.008)`, whose rim is at
0.108 (the cavity floor is one wall higher than `pos`), so the sheet starts 5 mm inside the four walls; a 27 mm
`steel_carbon` ball (80 g) at z 0.36; `dt=0.01`.
- With the ball: INVALID at the first check (PAIR_PEN_FAIL, 4 mm into a wall; ENERGY_JUMP; vertices leaving the
  walls at 10.9 m/s). Per control step:

  | step | L4 wall | CPU wall | exchanges | VBD substeps | what the clock read |
  |---|---|---|---|---|---|
  | 0 | 3.8 s | 89 s | 128 | 1 | the ball's interface bound, 1.0e-4 s |
  | 1 | 20.5 s | 829 s | 256 | 4 | band bound 4.6e-5 s (vertices at 11 m/s) |
  | 2 | 154.5 s | (stopped) | 1,024 | 8 | band bound 3.0e-6 s (about 170 m/s), finer than the finest step |
  | 3 | over 2,500 s (cut) | | 1,024 | 64 | bound 5.7e-7 s; 270,857 graph nodes an exchange, over the budget: eager |

  sample-sweep measured the same on rc2 (3.7, 20.2, 153.6 s) and rc6 (step 3 over 600 s).
- Without the ball (cloth alone, no coupled clock): INVALID at step 1 as well (CLOTH_STRETCH, ENERGY_JUMP), at a
  steady 0.77 s per step on the L4. The 0.15 s per step first reported as this scene's baseline was also a frozen world.
- The same scene with the sheet at z 0.115 (7 mm above the rim): no flags; 0.06 s per step without the ball, and a
  steady 2.0 s per step with it (128 exchanges of 15 ms in CUDA graphs: the documented clock for a body of 80 g
  beside cloth, `docs/guide/deformables.md`, Cost).
- 4 worlds on the L4, the ball in each: every sheet at 0.115, a steady 2.48 s per step (128 exchanges). World 0's
  sheet at 0.103 and worlds 1-3 at 0.115: 4.3 s, then 24.0 s (256 exchanges, VBD 4), then step 2 at 1,024 exchanges
  with VBD 16, whose exchange (68,688 graph nodes) is over the graph budget and runs eagerly: it had run over 19 min
  when the spot VM was reclaimed. Worlds 1-3 stayed healthy and paid all of it.
- A soft body shows it too. A 15 x 15 x 5 cm `silicone_40a` pad placed flush on the ground (its vertices then start
  one particle radius, 2.5 mm, inside it), with the same ball resting 2 mm above it, latched at step 0 (`dt=0.005`).
  Its steps grew to 37 s, 140 s and 265 s on the CPU. On the L4 they took 56, 64 and 42 s, then 5.7 s each. The same
  pad and ball placed clear step in 0.02-0.14 s on the L4.

## Why
- The latch's source is the period start (`alloy/integration/simulation.py`, Latch source), restored after every
  exchange; the frozen world is still a world of the Scheduler, so its estimators and contact band feed the clock
  (`SchedulerMixin.choose_clock`). Re-stepping a start overlap from the same state every exchange makes the
  exit-speed kick of `cloth-contact-exit-kick.md` grow as the step shrinks.
- Parked worlds (`alloy/coupling/parked.py`) do exactly what a frozen world needs (no clock bound, no
  re-run asks, held). Without `ALLOY_PARK_LATCHED` only `clock_worlds=` parks worlds; the health latch does not.

## Workaround
- Opt-in: `ALLOY_PARK_LATCHED=1` in the environment, before the build (`alloy/integration/park_latched.py`). At
  the end of the period a world latched in, it is parked in the Scheduler as parked worlds are: it
  holds no clock bound, asks for no re-run and is held. The substeps it raised drop at once. A reset unparks it.
  Turn it on for a coupled scene (cloth or a soft body beside a robot or a free rigid body) run with many
  worlds or for long, when frozen worlds are not reset at once. Nothing changes until a world freezes. After
  that the other worlds step on their own clock, so their results differ from a run without it. Scenes built
  with `clock_worlds=` are left alone (their driver parks worlds itself).
- With it, wall time per control step on the same L4:

  | scene | without it | with it |
  |---|---|---|
  | 1 world, frozen at step 0 | 3.8, 20.5, 154.5 s, then over 2,500 s | 3.8 s, then under 3 s (0.5 s once captured) |
  | 4 worlds, world 0 frozen | 4.3, 24.0 s, then over 19 min | 4.3, 3.4, 15.7 s, then 2.5 s |
  | 1,024 worlds, world 0 frozen | 119 s, then over 37 min (cut) | 116 s, then 2.5, 11.3, 11.3, 6.7 s, then 2.1 s |

  - All 4 worlds healthy take 2.5 s a step. All 1,024 take 2.0 s after the same graph warm-up (3.2, 10.9, 11.2,
    6.8 s), with the option and without it: with no world frozen it costs nothing measurable.
  - The 1,024 worlds: a 9 cm sheet (resolution 8, 5 g) over a 6 cm open box with an 8 mm wall, and a free 1 kg ball,
    in each world; world 0's sheet starts 5 mm inside the walls. Without it, step 1 runs 1,024 exchanges at 64 VBD
    substeps, eagerly (158,231 graph nodes an exchange).
- Read `sim.check()["ok"]` or `sim.health()` after the first step. A world INVALID at step 0 or 1 is almost always a
  start overlap: move the object clear and build again.
- Place cloth and soft bodies clear of what they rest on: a sheet above a box's rim (`add_open_box`'s rim is
  `pos` z + wall + depth), a soft body one particle radius above the ground or a support (VBD solids rest one radius
  above what they touch; the radius is 1/5 of the tet mesh's shortest cell edge).
- Without the option, a world that froze later still costs every world while the clock reads it: `sim.reset()` brings
  it back to its start; `alloy.build(..., clock_worlds=n)` moves the neediest worlds onto a clock of their own.

## Planned fix
- The option as the default, after a decision: it changes the other worlds' steps only after a world
  has frozen.
- START_PEN for cloth and soft-body vertices, naming the object and the part, so a start overlap is reported as one.
