# Liquid poured on a fabric sheet makes the world INVALID; cloth takes up no water

## What is wrong
Two separate limits.

1. **A stream landing on a fabric sheet breaks the world.** This happens with self-contact on, which is the
   default. Within a few hundredths of a second after the stream lands, health flags ENERGY_JUMP, and often
   SPEED_FAIL too. The world goes INVALID and is frozen (`sim.check()["ok"]` is False). The same scene with
   `add_cloth(..., self_contact=False)` stays OK, and so does a film sheet (`film_ldpe`, route `cloth_film`; paper
   sheets not measured). In those runs the water rests on the sheet and none passes it.
2. **Cloth takes no water.** There is no absorption, no wetting and no seepage. A sheet gains no weight from the
   water on it, and its friction and drape do not change when wet. Real untreated cotton wets at once, soaks up
   about 1-3 times its own weight and lets water through once saturated. In the simulation every fabric acts as a
   watertight sheet.

## Number
The scene: a 0.2 m `cotton_woven` sheet (2 mm, about 12 g, 20 mm cells) resting on the rim of a static open box.
A 20 ml `add_pour` at 0.5 m/s starts 5 cm above it; the stream lands at about 0.06 s. CPU, `Scene(dt=0.005)`, one run
per row (the CPU repeats bit for bit).

| variant | first INVALID |
|---|---|
| as above (self-contact on, the default) | 0.080 s, ENERGY_JUMP + SPEED_FAIL (21 m/s) |
| `self_contact=False` | none to 0.6 s; all 244 particles rest on the sheet |
| `material="film_ldpe"`, 20 mm or 5 mm cells | none to 0.6 s |
| sheet mass 4 g, 120 g or 360 g (`mass=`) | 0.075-0.115 s |
| `dt` 2.5 ms or 1 ms | 0.077 s / 0.087 s |
| cells 10 mm or 40 mm; `cloth_accuracy="super"`; `liquid_accuracy="super"` | 0.075-0.115 s |
| stream 0.2 m/s | 0.105 s |
| sheet on a full block top instead of a rim | 0.39 s, SPEED_FAIL |

- A towel draped over a static block and a 2 m/s stream also went INVALID soon after the stream landed.
- A world that fails does not recover: run to 3 s, the default row stays frozen from 0.080 s.
- In one development run with the freeze turned off, 1.2 % of the poured particles ended below the sheet, 9 cm
  from its border. They were carried through within one 2.5 ms step while the sheet under the stream moved at
  3-4 m/s.
- Only this rim scene was measured with `self_contact=False`, for 0.6 s. A drape, a fold, a long pour and a robot
  holding the cloth were not.

## Why
The failure needs both the sheet's self-contact and liquid on the sheet. The same sheet with no liquid, or with the
pour landing beside the box, stays OK with self-contact on. Changing mass, time step, mesh, accuracy settings or the
number of coupling passes does not remove it; turning self-contact off does. The exact mechanism is not yet found.

Separately, the liquid sees the cloth as a zero-thickness two-sided wall. Nothing sweeps a fast particle's path
against the sheet, and a particle that has crossed stays on the far side. That is how a sheet moving fast lets
particles through.

There is no absorption model, so limit 2 is simply missing.

## Workaround
- Pour onto a sheet that never folds with `add_cloth(..., self_contact=False)`. It is also about 1.4x cheaper per
  step on a GPU (`docs/guide/deformables.md`).
- Use a film sheet (`material="film_ldpe"`) where a watertight sheet is what you want.
- Check `sim.check()["ok"]` after any pour onto cloth: a frozen world looks calm.
- To show a wet towel's weight, add the water's mass to the cloth (`add_cloth(mass=...)`) and leave the liquid out:
  the added mass stands in for the water.

## Planned fix
Not in this release. In this order:
1. Find and fix the self-contact failure under a stream, with laws: a pouch of fabric holds a poured volume for
   10 s, and a drop landing on one cell does not pass. The stream's momentum on the sheet must match rho Q v.
2. No pass-through: the cloth's thickness and a swept test for the liquid against the sheet.
3. An opt-in absorption model: particles taken into a per-vertex water mass up to the fabric's measured capacity,
   with that mass added to the cloth.

Wet friction and wet bending need measurements of a named fabric wet and dry before they get a model. No date yet.
