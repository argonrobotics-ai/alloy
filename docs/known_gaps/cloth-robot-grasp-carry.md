# A robot's friction pinch on cloth creeps, and lets go below its friction limit

**What is wrong.** Cloth held between two gripper fingers by friction alone slides slowly out of the pinch under any
steady load, and falls out before the load reaches the pinch's Coulomb limit (two faces x mu x the grip force). A
task that lifts, carries or folds cloth by a pinch can lose it. The library task that folds a towel,
`ClothFold-v0`, is marked experimental (`docs/guide/envs.md`).

**The number.** Two pads squeeze a hanging double layer of cloth at 15 N, through the robot coupling and the shipped
cloth solver (40 iterations), and a load is hung on the cloth's bottom row (CPU, 1 s hold;
`alloy.coupling.pinch_hold.run_hold`):

| load / Coulomb limit | 0.5 | 0.75 | 0.9 | 1.2 |
|---|---|---|---|---|
| slide speed, default friction | 2.97 mm/s | 5.05 mm/s | falls out | slides out |
| with stick friction (opt-in) | 0.000 mm/s | | 0.000 mm/s | slides out |

A real pinch holds still below the limit and slides above it.

**Why.** The cloth solver's friction is smoothed Coulomb friction: the slip is measured from where the cloth was at
the start of each step, so a held vertex under a steady load slips a little every step, at friction_epsilon x
(1 - sqrt(1 - load / limit)): 2.93 mm/s at half the limit, against 2.97 measured. Near the limit the friction solve
does not converge, and the cloth falls out early. [cloth-friction-creep.md](cloth-friction-creep.md) has the
solver-level law.

**Workaround.**
- Stick friction, opt-in: `ALLOY_FRICTION_STICK=1` in the environment, or `body_particle_friction_stick=True` in the
  cloth entry's solver options ([cloth-friction-creep.md](cloth-friction-creep.md)). On the rig above it holds the
  pinch still to 0.9 of its limit, with the full-surface contacts robot fingers get. Its effect on `ClothFold-v0`'s
  fold count is not measured yet.
- `make("ClothFold-v0", grasp="attach")`: an opt-in shortcut, not a friction grasp. While the fingers are closed on
  the towel, the towel between them is held to a finger until they open, as cloth benchmarks grasp
  (`alloy.envs.cloth_attach`). It is one-way: the gripper does not feel the towel's weight. On the rig above it holds
  twice the friction limit (0.5 mm of give).
- Keep a friction grasp short: the creep is mm per second, so a lift and fold of a few seconds holds (ClothFold's pinch
  held the towel in every world of every run).
- Judge the grasp by the cloth, not the fingers: record the sheet's points and check that they follow the gripper.

**Planned fix.** Stick friction is opt-in for now; whether it becomes the default is not decided yet
([cloth-friction-creep.md](cloth-friction-creep.md)). No date yet.
