---
owner: Alloy maintainers
---

# Cloth takes up no water, and liquid on cloth steps slowly

## What is wrong
1. **Cloth takes no water.** There is no absorption, no wetting and no seepage. A sheet gains no weight from the
   water on it, and its friction and drape do not change when wet. Real untreated cotton wets at once, soaks up
   about 1-3 times its own weight and lets water through once saturated. In the simulation every fabric acts as a
   watertight sheet.
2. **Liquid on cloth is slow.** Once water lies on a fabric sheet, the coupled step costs several hundred times
   real time on one CPU core and thousands of times on a GPU (Number).
3. **No free pouch.** A sheet laid over an opening bridges it and sags only 6-10 mm, so poured water runs off its
   sides. The API has no way to pin a cloth's edge or start it in a curved shape. To hold water on a sheet, pinch
   its border between two static parts (Workaround).

Fixed in this release: a stream poured onto a fabric sheet with self-contact on (the default) no longer makes the
world INVALID. Until now the cloth solver's self-contact treated the liquid's particles as parts of the cloth. Water
resting 1.15 mm above the sheet, inside the 2 mm contact margin, pushed the sheet away with forces of up to about
10 kN on a 0.3 g vertex. The world went INVALID 0.02-0.05 s after the stream landed. Self-contact now leaves other
objects' particles out.

## Number
CPU rows, `Scene(dt=0.005)`, one world unless stated:

| scene | before the fix | with the fix |
|---|---|---|
| a 0.2 m `cotton_woven` sheet (about 36 g, 20 mm cells) on a box rim, 20 ml poured at 0.5 m/s | INVALID at 0.125 s (the stream reaches the sheet at 0.08 s) | OK to 0.6 s in a measurement run (the law checks 0.3 s), no particle through the sheet |
| a 1 ml jet on one 40 mm cell of a sheet on a table, 0.5 / 1 / 2 / 4 m/s | INVALID at 1, 2 and 4 m/s | OK at all four, no particle through |
| a cloth-floor tank: 60 ml held 10 s (0.2 m sheet, border pinched 1.5 mm under a frame) | INVALID during the pour, with 250 of 938 particles poured | 10 s hold: 894 of 938 particles held, 44 left under the frame, none through or under the sheet. 1 s hold: 893 held |
| 1,024 worlds of a 0.24 m sheet on a box rim, 20 ml, one L4 | 503 of 1,024 INVALID at 0.08 s, 660 at 0.105 s (stopped there after 3 hours) | 1,024 of 1,024 OK to 0.26 s, no water under the sheet |

- Film sheets (`film_ldpe`, paper and metal sheets: the film solver) behave the same before and after. They keep
  the old self-contact detection, which in the film rows measured listed no pair with the liquid.
- Granular and paste particles (sand, dough: the MPM solver) are left out of the cloth's self-contact by the same
  rule. That is not measured, because cloth beside sand is refused at build.
- With `self_contact=False` the sheet steps the same as before. With the fix, a flat sheet under the pour steps
  up to 0.64 mm apart with self-contact on and off over 0.3 s, though it lists no self-contact pair. Two candidate
  causes, not yet separated: the self-contact motion limits, which are on whenever self-contact is, and the rim
  contact sharing their motion reference.
- Cost, wall seconds per simulated second once water is on the sheet, with the fix; one run each:
  - one CPU core:
    - the tank: 370-440 (a cloud VM core);
    - the rim pour: about 460 (a busy shared host);
  - the same two scenes with a film sheet: 80-240;
  - one L4, the 1,024-world scene: 12,700 s for 1,024 worlds together (12 s per world). Before the fix the
    same scene ran 97,600 s per simulated second, until 0.105 s.
- Cost per 10 ms control step, one world on one L4: a 10 cm cotton sheet dropped 2 cm onto 0.8 l of water in a
  13 x 13 cm box, over the first 0.15 s after it touches:
  - with the fix: 7.3 s per step (median 4.2, the most 16.7), falling to 1-3 s once the sheet settles;
  - before the fix: 29 s (median 30, the most 56), still rising;
  - a paper sheet: 5.0 s.
  The slow steps are bound by the host launching the cloth solver's kernels: the GPU sits at 11-15 %.

## Why
Absorption is not modelled.

The cost is not broken down yet. Two parts are seen so far:
- Once the cloth and the liquid touch, their exchange runs up to 8 passes per period, against 1 before.
- On a GPU the cloth solver's steps are launched one by one, and the host doing the launching is the bottleneck.

The liquid still sees a cloth as a zero-thickness two-sided wall, with no swept test for a fast particle. What does
hold:
- At each step the liquid is kept on the side of the sheet where the step began, against the sheet faces found
  within 1.5 particle spacings at the step's start.
- A particle can skip a face only by moving farther than that in one step, and the liquid's time-step limit holds
  it to about 0.2 spacings per step.

In the rows above no particle crossed a sheet. Impacts far above 4 m/s and sheets that fold under a stream were
not measured.

## Workaround
- To hold water on a sheet, pinch its border between a box rim and a static frame whose bottom is less than one
  particle (4 mm) above the sheet. At 1.5 mm the 60 ml stayed. At 2.5 mm a stream leaves under the frame.
- To show a wet towel's weight, add the water's mass to the cloth (`add_cloth(mass=...)`) and leave the liquid out:
  the added mass stands in for the water.
- For speed, use a film sheet where a watertight sheet is what you want. Pass `self_contact=False` for a sheet
  that never folds (about 1.4x cheaper per step on a GPU).
- `ALLOY_VBD_FOREIGN_SELF_CONTACT=off` restores the old self-contact, which makes a pour onto cloth INVALID; for
  comparisons only.

## Planned fix
1. An opt-in absorption model: particles taken into a per-vertex water mass up to the fabric's capacity, with that
   mass added to the cloth. Design and laws are drafted; no date yet.
2. Cost: a captured graph for the coupled cloth step on GPUs, and fewer exchange passes once the sheet is at rest.
3. A swept liquid test against cloth for fast impacts, with its own law.

Wet friction and wet bending need measurements of a named fabric wet and dry before they get a model.
