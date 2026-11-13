# A pour alone can splash a few percent over its container's rim, and health stays OK

## What
- A stream from `add_pour` landing in an open container can throw part of the liquid over the rim. With the liquid
  alone (liquid and static objects only), 3 % of a 6 ml pour went over the rim in the scene below; a 20 ml pour
  stayed in.
- Beside cloth with no free rigid body it is a fifth of the pour or more through 0.1.1 (20 % of the 6 ml pour, 24 % of
  the 20 ml pour). After 0.1.1 both stay in.
- All of it goes over the rim. No particle passed through a wall in these runs: the walls hold.
- Health stays OK, and `sim.check()["ok"]` is True. Only liquid that leaves the scene's bounds counts as escaped
  (`retired_particles`); liquid that leaves a container does not.

## Number
CPU, 0.6 s simulated. Two static `add_open_box` containers, 8 x 8 x 6 cm and 10 x 10 x 8 cm inside with 5 mm walls.
Pours of 6 ml (94 particles) and 20 ml (312 particles) at 0.1 l/s, from 12 cm above the floor. The cloth is a 10 cm
sheet lying 30 cm away. A particle counts as through a wall when its path in one step crossed wall solid. The same
counts on one L4.

| scene | 6 ml outside | 20 ml outside | through a wall |
|---|---|---|---|
| no cloth | 3 of 94 (3 %) | 0 | 0 |
| cloth, no free rigid body, through 0.1.1 | 19 of 94 (20 %) | 74 of 312 (24 %) | 0 |
| cloth, no free rigid body, after 0.1.1 | 0 | 0 | 0 |
| cloth and a free 2 cm ball | 0 | 0 | 0 |

Cost of the change made after 0.1.1, one world on one L4:
- the two pours beside cloth: 79.5 -> 110 wall seconds per simulated second, because the liquid takes 3.3x the steps
  while it pours;
- a cup of water at rest beside cloth: 9.05 -> 9.78 wall seconds per simulated second, with the same number of liquid
  steps;
- a 10 cm cotton sheet (resolution 10) dropped 2 cm onto 0.8 l of water in a static 13 x 13 cm box: 33 -> 88 s per
  10 ms step after first touch;
- the same scene with a `paper_80gsm` sheet: 2.3x over its first 0.3 s.

## Why
- A stream landing on a floor needs small liquid steps while it lands.
- Alone, the liquid steps at one fixed step chosen at build from an assumed contact speed of 0.5 m/s. A stream from
  12 cm lands at about 1.5 m/s, three times faster than that step allows.
- Beside cloth, the liquid used to step at its own stability limit, measured from its top speed, with no margin.
  After 0.1.1 it takes a quarter of that limit while it moves fast; at rest it keeps its largest step.

## Workaround
- Count the liquid that stays in the container yourself, from `"<name>.points"` and `"<name>.active"` in
  `sim.state()`, instead of trusting a clean health report.

## Planned fix
Planned, with no release named: the liquid alone chooses its step while it runs, as it does beside other solvers,
with a law that a pour keeps its particles in an open box alone.
