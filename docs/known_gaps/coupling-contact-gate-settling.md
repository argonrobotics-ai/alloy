# Known gap: with the opt-in contact gate, cloth resting on fixed geometry while the robot is away settles at a coarser step

## What is wrong
- The contact gate is experimental and not advertised (coordinator 2026-11-08). Its loss against today's clock in the
  scene where it releases the clock is fixed (Speed below).
- The contact gate (`ClockPolicy.contact_gate`, off by default; `alloy/coupling/contact_gate.py`) releases the
  exchange clock while a robot is away from the cloth. Today's clock instead holds the lightest robot link's bound
  the whole run.
- While released, VBD steps the cloth at the coarser exchange step, because VBD's own step follows the exchange step.
- Cloth that rests on fixed geometry meanwhile (a table that is part of the robot's file, the floor) settles slightly
  differently than under today's clock.
- When the robot then touches the cloth, the pair is armed and runs today's clock exactly. The small offset carried
  into contact grows there like any other perturbation.
- There is no drift caused by the contact itself: from one state, the gated run equals today's clock bitwise (CPU).

## Number
- Scene: L4 GPU, 4 worlds, robot_cloth_liquid (a Panda arm reaches a 0.3 m towel lying on its table, beside a water box;
  built by a development script of mixed-material speed scenes, not shipped). Control period 10 ms, 30 timed periods;
  the arm touches the towel at period ~25.
- The gate runs 8 exchanges per period for the 24 periods before the arm arrives; today's clock runs 128.
- When the arm arrives the towel is 0.6 mm off at its centre. Three runs of today's clock agree there to 0.00 mm.
- Final towel offset against today's clock (main 16281ba5e, one gated run against 2 runs of today's clock):
  1.7-2.1 mm at the centre and 8.1-8.8 mm RMS over the particles.
- Today's run-to-run floor, from 3 pairs of today's runs: 1.6-2.5 mm centre, 7.0-9.8 mm RMS.
- The water moves 0.04 mm at its centre and 2.4 mm RMS over its particles (floor 0).
- Before wp4's MuJoCo -> VBD -> SPH chain (3dddcdfaa) the same scene ended 4.5-9.4 mm off at the centre against a
  2.0-4.9 mm floor.
- Scenes where the arm is near the cloth from the start keep today's clock and stay within the floor:
  robot_cloth (towel draped on the arm) and robot_cloth_drop (towel dropped onto a swinging arm).

## Why
- VBD is implicit: how the towel settles on the table depends on its step. Released, the step is 0.16-0.6 ms
  (1.25 ms exchange, 2-8 substeps); under today's clock it is 78 us.
- The gate keeps today's clock only for coupled pairs in or near contact, as designed. A fixed body cannot hold the
  clock, so cloth touching it does not arm the pair.

## Workaround
Leave the gate off (the default) when the cloth's settled shape before contact matters at the 1-2 mm level.

## Speed (the chain's pass loop, fixed)
- Since wp4's MuJoCo -> VBD -> SPH chain (3dddcdfaa), each released 1.25 ms exchange ran the chain's coupling pass
  loop to k_max: 8 passes. The cause (speed-now S2, 2026-11-09): the released arm's lagged MuJoCo -> VBD pair still
  counted in its world's residuals. A lagged pair's velocity residual is the link's lag behind its source, a dt_x,
  the same in every pass: 0.012-0.035 m/s against tol 0.01 at 1.25 ms (0.005-0.007 at 78 us), with no force on any
  proxy. Short worlds re-ran the period, the captured passes rose to 8 and the exchange graph grew past the
  65,536-node budget, so it ran eagerly.
- Fix: a released proxy adds no residual (`alloy.coupling.residuals.ResidualInputs.gated`). The gate released it
  because it cannot touch within the period, and the safety net re-runs any period in which it did.
  `tests/coupling/test_contact_gate_residual.py`: with nothing touching, the state is bitwise the same in 2 passes
  instead of 8.
- Speed after the fix (rc2-1109, 64 worlds, one L4, the towel falling beside water in robot_cloth_liquid, 6 timed
  periods): 6.8 s per 10 ms period against today's clock's 17.5 s, so 2.6x faster. Before the fix the gate took
  65 s. The result rows for `gatefix`, `base` and `gate` are in a speed run's output (not shipped).

## Planned fix
- Probe (WP10, coordinator 2026-11-08), report only: keeping VBD's fine step while the gate is released removes
  the towel's offset (0.00 mm when the arm arrives), but costs more: 41-62 s per period at 64 worlds against 9.3 s.
  The water still moves 2.4 mm RMS.
- The settling offset is within today's run-to-run floor at the end of the measured scene (Number). Default-on is
  speed-now S2's call, on its laws (the speed-now plan, not shipped).
