---
owner: Alloy maintainers
---

# Cloth takes up water only when asked, and liquid on cloth steps slowly

## What is wrong
1. **Cloth takes up water only when asked, and coarsely.** By default a sheet takes no water: no absorption, no
   wetting and no seepage, so every fabric acts as a watertight sheet. Real untreated cotton wets at once, soaks up
   about 1-3 times its own weight and lets water through once saturated.
   `add_cloth(absorb={"capacity": kg/m², "rate": kg/m²/s, "wicking": m²/s})` (opt-in) takes liquid resting on the
   sheet into a water mass per vertex, up to capacity x area. The sheet weighs that water, and wicking spreads it in
   the sheet's plane. Its limits:
   - The values are yours: no fabric card carries them yet. Derived from published figures, not measured here:
     cotton holds about 0.35-0.96 kg/m², a terry towel 0.65-2.2.
   - Particles are taken whole (64 mg at the default 4 mm spacing). A vertex fills in steps of a particle's share
     and can stop up to one particle short of full. A particle is taken only when every vertex of its face has room
     for its share, so a vertex whose capacity x area is under a third of a particle (21 mg at the default spacing)
     takes up little or nothing. At 0.6 kg/m² some vertices fall under it at cells under about 10 mm, half the
     sheet under about 7 mm and all of it at 5 mm. The build warns with the count, naming the sheet and the fix: a
     coarser sheet, a higher capacity or `liquid_accuracy="super"`.
   - A particle is taken one step after it touches the sheet. Uptake runs at the rate on average. In each step only
     one group of faces that share no vertex takes water.
   - A full sheet lets nothing through (no seepage). Wet friction, wet drape and drying are not modelled.
   - Film and paper sheets refuse it. The ADMM coupler refuses it (the proxy coupler takes it), and so does the
     per-world fine pack.
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

- With `absorb` (test values, not fabric data), CPU, one world, the rim scene of the first row unless stated:

  | scene | result |
  |---|---|
  | 20 ml pour, capacity 0.6 kg/m², rate 2 kg/m²/s, 0.6 s | 224 of 312 particles taken up (14.3 g); nearly all the rest ran off the sheet (3 still on it); health OK |
  | the same at rate 20, wicking 2e-4 m²/s / none, 1.5 s | 261 / 272 taken up; with wicking the water reaches all 144 vertices, without it 111 |
  | the cloth-floor tank, 60 ml, capacity 0.6, rate 20 | every vertex of the opening within one particle of capacity, none over; one particle is 39-78 % of a vertex's capacity here, so the lowest fill is 0.42 (mean 0.95) |
  | the tank, capacity 5, rate 1 | uptake 14.3 g/s against rate x wetted area 16.6 g/s (0.86: vertices at the water's edge are wet only part of the time); no vertex faster than its rate |
  | the wet sheet against a dry twin weighing its dry mass plus the water, per vertex | deepest sag 11.76 mm against 11.44 mm (2.8 %) |

  Cost, the rim pour, steps after the water lands, one quiet CPU machine, one run each: median 0.75 s per 5 ms
  control step with absorb against 1.57 s without (means 1.8 s and 2.3 s; the first 45 steps cost the same). Taken-up
  water leaves the liquid, which fits later steps being cheaper.
- With `absorb` on one L4 (capacity 0.6, rate 2), one run each unless stated:
  - the rim pour: two runs took up the same 220 particles into the same water; 16 identical worlds took up 220
    each; the CPU took up 224 by the same time, about 2 % more (the liquid's CPU and GPU steps are the same model,
    not the same bits). Uptake repeats only as far as the rest of the scene does: on a GPU a sheet that folds or
    piles may not repeat (AGENTS.md section 6), and then neither does its uptake;
  - 1,024 worlds of a 0.24 m sheet on a box rim, each sheet starting within 2 mm of centre, 20 ml: none INVALID
    with absorb or without, and 123-161 particles taken up per world by 0.26 s, uptake still rising;
  - cost: none measurable at one world (2.9 s mean per step with absorb and without, 3 runs each); 1,024 worlds about
    the same (one run each: medians 54.4 s per step with absorb, 55.4 s without).

## Why
Absorption is opt-in and coarse (What is wrong, 1). By default it is not modelled.

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
  the added mass stands in for the water. To have the sheet take up liquid as it lands, pass `absorb=` (opt-in;
  What is wrong, 1).
- For speed, use a film sheet where a watertight sheet is what you want. Pass `self_contact=False` for a sheet
  that never folds (about 1.4x cheaper per step on a GPU).
- `ALLOY_VBD_FOREIGN_SELF_CONTACT=off` restores the old self-contact, which makes a pour onto cloth INVALID; for
  comparisons only.

## Planned fix
1. Absorption: water values on the fabric cards (cotton, towel; tagged catalogue or fit, with ranges), seepage
   through a full sheet, and uptake finer than whole particles.
2. Cost: a captured graph for the coupled cloth step on GPUs, and fewer exchange passes once the sheet is at rest.
3. A swept liquid test against cloth for fast impacts, with its own law.

Wet friction and wet bending need measurements of a named fabric wet and dry before they get a model.
