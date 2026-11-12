# A pour can splash over its container's rim, much more beside cloth, and health stays OK

## What
- A stream from `add_pour` landing in an open container can throw part of the liquid over the rim. Beside cloth, in a
  scene with no free rigid body, it is a fifth of the pour or more: 20 % of a 6 ml pour and 23 % of a 20 ml pour
  ended outside their static boxes in the scene below.
- All of it goes over the rim. No particle passed through a wall in these runs: the walls hold.
- Alone (liquid and static objects only), 4 % of the 6 ml pour went over the rim and the 20 ml pour stayed in. With a
  free rigid body anywhere in the scene, both stayed in.
- Health stays OK, and `sim.check()["ok"]` is True. Only liquid that leaves the scene's bounds counts as escaped
  (`retired_particles`); liquid that leaves a container does not.

## Number
CPU, 0.6 s simulated. Two static `add_open_box` containers, 8 x 8 x 6 cm and 10 x 10 x 8 cm inside with 5 mm walls.
Pours of 6 ml (100 particles) and 20 ml (294 particles) at 0.1 l/s, from 12 cm above the floor. The cloth is a 10 cm
sheet lying 30 cm away. A particle counts as through a wall when its path in one step crossed wall solid.

| scene | 6 ml outside | 20 ml outside | through a wall |
|---|---|---|---|
| cloth, no free rigid body | 20 of 100 (20 %) | 68 of 294 (23 %) | 0 |
| cloth and a free 2 cm ball | 0 | 0 | 0 |
| no cloth | 4 of 100 (4 %) | 0 | 0 |

## Why
- A stream landing on a floor needs small liquid steps while it lands. Beside cloth, the liquid steps in substeps of
  the coupled scene's exchange step. It takes their size from its own stability limit with no further margin, and
  that is too coarse for the landing.
- A free rigid body makes every exchange much shorter, so the liquid's steps shrink with it, which hides the problem.
- Alone, the liquid steps at its stability limit from the last step's top speed. That leaves a small loss over the
  rim at the landing.

## Workaround
- Add a free rigid body to the scene, for example a small ball placed out of the way. It costs the scene's step rate.
- Or count the liquid that stays in the container yourself, from `"<name>.points"` and `"<name>.active"` in
  `sim.state()`, instead of trusting a clean health report.

## Planned fix
Planned for a later release: beside cloth the liquid takes a quarter of its stability limit per substep (in the scene
above that kept every particle in both boxes), with a law that a pour keeps its particles in an open box alone,
beside cloth and beside a free body. The 4 % alone is studied as well.
