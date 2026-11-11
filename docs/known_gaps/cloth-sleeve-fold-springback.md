# Known gap: folded T-shirt sleeves do not stay folded at the default mesh and contact stiffness

## What
- In the fold-a-shirt task (`alloy.tasks.laundry_tee`), two hands swing the cuffs of a flat T-shirt round its side
  seams onto the body, hold, and let go. The T-shirt is two cotton panels sewn at the side, sleeve and shoulder
  seams.
- As the hands let go, both sleeves lie on the body.
- With the default rig (`TeeRig()`: 20 mm cells, the cloth route's contact stiffness) they then spring part-way
  back off it and never lie flat. The task's settled gates fail.
- With the fine rig (`FINE_TEE`: 10 mm cells, particle contact stiffness 2e4 N/m, no contact damping) they stay:
  every gate passes. That rig is a GPU run (9,300 particles: about 7 min on one L4, about 11 h on one CPU core).
- So this is not the fabric springing back. It is the mesh and the contact, see Why.

## Number
One L4, plain-weave cotton (h 0.975 mm), 2.5 s swing, 0.5 s hold, the start nudged 1 µm per seed, settled gates:

| rig | starts that pass | sleeve inside the body (gate >= 90 %) | placement RMS (gate <= 45 mm) | normal-gap p90 (gate <= 25 mm) |
|---|---|---|---|---|
| default: 20 mm cells, contact 2e6 N/m | 0 of 6 | 8-92 % | 48-204 mm | 20-94 mm |
| 20 mm cells, contact 2e3 N/m, no contact damping | 0 of 2 | 59-100 % | 76-103 mm | 26-29 mm |
| 10 mm cells, contact 2e3 to 2e5 N/m, no contact damping (`FINE_TEE`: 2e4) | 18 of 18 | 100 % | 2-6 mm | 14-17 mm |
| the same on `cotton_single_jersey` (2e3 N/m) | 7 of 8 | 100 % | 3-16 mm | 14-27 mm |
| 12.5 mm cells, jersey | 0 of 1 | 100 % | 10-12 mm | 21-29 mm |
| 10 mm cells, contact 2e3 or 2e4 N/m with the contact damping left on | 0 of 2 | 8-78 % | 100-220 mm | 1-27 mm |
| 10 mm cells, the route's contact | 0 of 1 | 13-100 % | 15-213 mm | 5-33 mm |

0 cloth crossings in every run with the self-contact hold (the cloth default). Stretch under 10 %.
On the CPU the default rig reads the same way (2,360 particles: inside 41-100 %, 75-120 mm, 50-54 mm).

## Why
- **The mesh.** Cells of 20 mm cannot crease tighter than about a 2 cm radius. While the hands still hold the cuff,
  the sleeve stands in a loop 4-6 cm high beside the fold line. On release that loop throws a plain-weave sleeve
  back out (inside 4-7 % with the plies otherwise at rest), and a jersey sleeve's cuff snaps back 6-8 cm.
  At 10 mm cells the loop is under 2 cm and the sleeve stays where it was laid.
- **The contact.** The route takes the particle contact stiffness from the material's bulk stiffness, 2e6 N/m here,
  against a vertex of 46 mg. The four plies then never come to rest: 3 s after release the typical particle moves
  at 3-5 cm/s and the sleeves at 0.1-0.5 m/s, mostly up and down, on plain weave and jersey alike, with the
  acceleration off and at 120 iterations too. Particles over 0.25 m/s, of 2,360: 199-331 at 2e6 N/m, 86 at 2e5,
  14-30 at 2e4, 0-4 at 2e3. The contact's damping (Newton's 10) alone keeps a 2e3 N/m fold moving; with it off the
  fold rests under 1 cm/s. It is the pair that rests a fold, not the stiffness alone: on WP2's towel folded in
  thirds (CPU, 8 starts, [internal note] `scr/WP2.md`, sweep 4) the route's 2e6 N/m with damping 10 ended 0.03-0.06 m/s
  short of rest on every start, 2e4 with the damping on failed a gate on 4 of 8 (the worst setting), and 2e3 with
  the damping off rested on all 8 (0.001 m/s, extent 11.9-12.1 mm, 0 crossings).
- Not the cause: the card's bending. An earlier version of this page blamed the plain weave's bending rigidity
  and the missing bending hysteresis. The jersey card with hysteresis (gap #4 slice 1) keeps a sleeve inside the
  body better at 20 mm cells (93-100 %) but does not make it lie flat, and plain weave passes on the fine rig.

## Workaround
- Use the fine rig's settings for a garment fold: cells of 10 mm or less, and on the engine model
  `soft_contact_ke = 2e4`, `soft_contact_kd = 0` (`alloy.tasks.laundry_tee.build` shows where). There is no public
  API setting for the particle contact stiffness yet.
- Or score the fold as the hands let go (`TeeVerdict.release_gates`), or keep the cuffs held until a later step
  presses the sleeves flat.

## Planned fix
- The route's particle contact stiffness for cloth on cloth: a default from the fabric's own compression, not the
  bulk stiffness (the unsettled-fold gap, docs/known_gaps/cloth-fold-unsettled-default.md). When that lands the
  fine rig needs only its mesh.
- `tests/tasks/test_laundry_tee.py` holds both: the fine rig's fold as a GPU law, and the default rig's settled
  gates as a strict expected failure.
