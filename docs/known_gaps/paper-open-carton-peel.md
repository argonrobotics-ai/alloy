# Known gap: the open-carton peel throws a light body in some worlds (1 in 40 with the task's caps)

## What
- `alloy/tasks/gift_carton_tape.py` (open-carton): the hook's tine holds the tape's tab (WP13's Bonds grip) and peels
  the seam tape off at ~180 deg. In some worlds, at the moment the arm goes taut or in the first ~0.5 s of peeling, a
  light body is launched past 20 m/s: the hook's wrist or palm, or the -y major (31 g). That world goes INVALID
  (`sim.check()` reports it) and fails the task.
- Which worlds fail follows the VBD colouring: Newton colours identical worlds differently, so identical worlds
  diverge.

## Number
- Without the caps (the default tine, 0.9 mm over the held row): 3 of 24 GPU worlds (16 worlds x 1.0 s of the peel:
  1; the task on 8 worlds: 2).
- The first merge's tine, 0.1 mm over the row: the hook was launched in 2-9 of 16 worlds at the taut moment, on every
  tree tried.
- With the coupler's exchange caps (c 4, c_d 0.5, which the task sets): 1 of 40 (two 16-world probes 0 and 0, the
  task 1 of 8).
- The task (NVIDIA L4, 8 worlds x 31.25 s): 6 of 8 succeed with or without the caps (the taped-carton measurements in
  the development notes, not shipped).

## Why
- A rigid body's contact force from the film reaches it one exchange late (lagged coupling). Where the tape folds and
  snaps taut under the tine, the film's strain projection is unconverged and that lagged force spikes for one
  exchange, on the tine or, through Bonds' pull at the front, on a flap.
- With the tine 0.1 mm over the held row, the folded arm was pinched between the tine and the board (the film
  collides at 0.5 mm); 0.9 mm removes the pinch. The later spikes remain.
- The coupler's exchange caps (described in their design note, not shipped) limit the lagged contact stiffness and
  damping and stop most of them. They are off by default: on by default for every film's coupled bodies (route c,
  22248a172) they slowed the bag task's carton-off-the-mouth GPU test from 24 min to over 3 h, so they are opt-in again.
  The task sets them for its own build (`gift_carton_tape.exchange_caps`, route b), and `sim.routes()` names them where
  they are on.

## Workaround
- Run more worlds and keep those `sim.check()` passes: an INVALID world is flagged and frozen, never silently wrong.
- The task builds with the caps (c 4, c_d 0.5): `with gift_carton_tape.exchange_caps(4.0, 0.5): alloy.build(...)`
  does the same for another scene with a film coupled to light bodies. The engine layers take them too:
  `FilmConfig(exchange_caps=(4.0, 0.5))` for the film entry, or an `IntegrationProfile` with
  `scheduler_policy=SchedulerPolicy(exchange_stiffness_c=4.0, exchange_damping_c=0.5)`, passed to
  `alloy.integration.wiring.build_simulation`.

## Planned fix
- A public way to set the caps (`alloy.build`) is a post-launch SCR with OSS-3 (coordinator 2026-11-08 15:16Z: no
  new public knob before launch). Caps on by default need the bag slowdown understood first (SC8).
- The rest (1 in 40) needs a converged film solve where the arm snaps taut.
