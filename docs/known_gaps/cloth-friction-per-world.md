# Known gap: per-world and per-pair friction on cloth and soft bodies covers contacts with shapes only

## What
`scene.randomize(name, "friction", lo, hi)` and `scene.set_friction(a, b, mu)` reach the contacts of a cloth, film
sheet or soft body with rigid objects, kinematic objects, the ground and robot links, per world. What they do not
reach:

1. **Friction of a sheet with itself and between two cloths or soft bodies.** It is one value for the whole scene,
   the same in every world: the heaviest deformable's friction, and for one with a randomized friction the lowest
   value of its range.
2. **Scenes that use neither call keep the older rule.** With several cloths or soft bodies of different friction and
   no `randomize` or `set_friction` on any of them, every one of them takes the heaviest one's friction (the build
   report says so). As soon as one of them has a randomized friction or a `set_friction` pair, each takes its own.
3. **The mix with the other object's friction is a geometric mean**, sqrt(cloth friction x object friction), not the
   harmonic mean rigid pairs use, and a library pair card for a cloth material is not read. `set_friction` gives the
   exact value.
4. **`rerun_worlds` and `clock_worlds`:** in a scene with randomized cloth friction a world whose values differ from
   its slot's is not moved to the second simulation, so the options do little there.

Checked on the CPU only; not run on a GPU yet.

## Number
A 10 cm, 40 g cotton sheet on a kinematic tray (tray friction 0.25) that is pulled 0.40 m in 0.25 s, CPU,
`cloth_accuracy="fast"`; the sheet's mean x afterwards:

| Case | Before | Now |
|---|---|---|
| `friction=1.2` | 0.285 m | 0.285 m (same bits) |
| `friction=1.2` and `set_friction("towel", "tray", 0.0005)` | 0.285 m | 0.0004 m |
| `randomize("towel", "friction", 0.0005, 1.5)`, 4 worlds drawing 0.055 / 0.48 / 0.37 / 1.32 | 0.008 m in all four | 0.081 / 0.224 / 0.202 / 0.290 m |
| that sheet at 1.2 beside a heavier rag randomized in [0.0005, 1.5] | 0.008 m | 0.285 m, as alone |

## Why
The cloth solver (Newton's VBD) takes one friction scalar for all particles and mixes it with each shape's own value
when it builds a contact. Alloy now rewrites that per-contact value from each world's draw and the scene's pairs
(`alloy/solvers/vbd/pair_mu.py`), only in scenes that ask for it, so every other cloth scene steps as before. The
self-contact and cloth-on-cloth kernels read the scalar directly and have no per-contact value to rewrite.

## Workaround
- For cloth-on-cloth friction, give every cloth in the scene the friction you want there with `friction=` (no
  randomization of it), and randomize the rigid objects' friction instead.
- To make each of several cloths use its own friction without randomizing, add one `scene.set_friction` pair for one
  of them (for example with the ground).

## Planned fix
Per-world friction in the self-contact kernels needs a change in the vendored solver; not scheduled. Moving the two
side simulations' friction rows with a world is a small follow-up.
