# Cloth and soft solids with self contact move at most about 3 m/s

## What is wrong
- In a VBD entry with self contact (`add_cloth`'s default, a cloth card with `features.self_contact`, two or more
  cloth or soft-solid objects), nothing moves faster than a cap. A sheet dropped from 1 m reaches 3 m/s and then falls
  at a constant 3 m/s instead of 4.4 m/s at the floor.
- On fine meshes the cap is lower (see Number).
- Before 2026-11-08 the cap was much lower: 0.82 m/s for the 0.6 mm towel card and 2.72 m/s for a 2 mm sheet at the
  1.25 ms step.

## Number
- **Rule:** cap = 0.85 (margin + gap) / step, where margin is the contact thickness h and the step is the cloth's
  physics step. The route (`alloy.solvers.vbd.reach`) picks the gap from the step so that the cap is 3 m/s:
  gap = max(0, min(3 step / 0.85, 0.75 ℓ) - h), with ℓ the median rest edge length. A finer step shrinks the gap and
  the cap stays 3 m/s.
- The cap rises above 3 m/s only when the gap is 0, that is when the step is under 0.85 h / 3 m/s: 0.57 ms for a
  2 mm sheet. Below that it is 0.85 h / step.
- On fine meshes, where 0.75 ℓ binds, it is lower than 3 m/s: at the 1.25 ms step that is a median edge under
  5.9 mm.
- **A control dt above 1.25 ms does not change the step:** `Scene(dt=...)` of 10, 5 or 2.5 ms still steps the cloth at
  1.25 ms (800 steps/s).
- **Measured (CPU, 0.3 m sheet, resolution 16, 0.03 kg, dropped from 1.6 m, fall speed fitted over 0.35 to 0.4 s;
  free fall is 3.68 m/s at the middle of that window, 3.92 m/s at 0.4 s):**
  - 2 mm sheet, Scene dt 10 ms, step 1.25 ms: 3.00 m/s (cap 3);
  - 2 mm, dt 1.25 ms, step 1.25 ms: 3.00 m/s (cap 3);
  - 2 mm, dt 0.625 ms, step 0.625 ms: 3.00 m/s (cap 3, gap 0.2 mm);
  - 2 mm, dt 0.5 ms, step 0.5 ms: 3.40 m/s (cap 3.4, gap 0);
  - 2 mm, dt 0.25 ms, step 0.25 ms: 3.58 m/s (cap 6.8, so not capped; 3 % under free fall's 3.68, which is
    float32 rounding of the velocity at fine steps);
  - 4 mm, dt 1.25 ms: 3.00 m/s (cap 3); 4 mm, dt 0.625 ms: 3.67 m/s (cap 5.4: free fall);
  - 2 mm, 5 mm cells (resolution 61, median edge 7.5 mm), dt 1.25 ms: 3.00 m/s (cap 3; 0.75 ℓ does not bind);
  - `self_contact=False`, dt 10 ms: 3.70 m/s (free fall).
- **Earlier numbers:** with the old reach the cap was 0.816 and 2.72 m/s (development scripts, not shipped).
- **Fine meshes, where 0.75 ℓ binds:**
  - SOFT-3's skin/fat pads (ℓ 4.7 mm, margin 1.38 mm): 2.4 m/s;
  - a 2 mm-cell sheet at h 0.6 mm: about 1 m/s.

## Why
- Newton's penetration-free truncation (DAT) caps every particle's motion between self-contact detections at
  0.5 · 0.85 · (margin + gap). A step detects twice.
- A larger gap lifts the cap, but it lists more candidate pairs: the cost grows with the reach.
- Past about a cell, the pairs overflow Newton's fixed per-vertex buffers, and the overflowing pairs are dropped
  (docs/known_gaps/vbd-self-contact-buffer-overflow.md).
- The 3 m/s bound and the 0.75-edge cap were chosen by measurement:
  - at 5 m/s a towel's fast pinch ran 9x slower and its half fold failed;
  - at 3 m/s, 0 crossings, the same fold quality, 1.5-2.8x the old cost on towels, and 3.35x faster on WP2's 4 mm drape.

## Workaround
- `Scene(dt=...)` alone does not raise the cap. Halving the control dt does not double it: at the route's reach the cap
  is 3 m/s at every step down to 0.57 ms (2 mm sheet).
- A thicker sheet with a fine enough step: a 4 mm sheet at a 0.625 ms step (`dt=0.000625`) fell freely (3.67 m/s,
  cap 5.4). A 4 mm sheet at 1.25 ms stayed at 3.00 m/s.
- A 2 mm sheet needs a step under 0.57 ms: 3.40 m/s at 0.5 ms, not capped at 0.25 ms.
- A finer step takes more steps per simulated second: 2 x at 0.625 ms and 5 x at 0.25 ms, against 1.25 ms.
- Self contact off (`self_contact=False`) removes the cap (3.70 m/s, free fall), and with it layer-on-layer contact.
- Engine layer, not public API: an explicit `particle_self_contact_gap` in the VBD entry's kwargs wins over the rule
  (keep margin + gap under one rest edge length, or pairs overflow), and `alloy_self_contact_speed` sets the rule's
  motion bound. No speed was measured for either.

## Planned fix
- None planned in the reach rule itself.
- A detection inside the iterations would lift the cap at less cost than a larger reach. Newton's ITERATIONS schedule
  does that; it is not measured yet (WP11).
