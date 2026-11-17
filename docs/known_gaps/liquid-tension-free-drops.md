---
owner: Alloy maintainers
---
# With surface tension on, a free blob of water oscillates more and more

## What
- With `alloy.Scene(liquid_tension=True)`, a free blob of a thin liquid such as water (several particles across,
  in flight or in zero gravity) that oscillates between a long and a flat shape grows the oscillation instead of
  settling back to round.
- Water at rest does not: in a tub, a narrow vial or on a plate it settles with tension on as it does without.
- A thick liquid (0.05 Pa·s measured) is not affected: its oscillation dies away around round.

## Number
- Free water drops with tension, kicked into the oscillation: the amplitude grew 18 % to 2.4x over 15 periods (drops
  7 to 14 particles in radius, larger drops grow less).
- Water at rest, 5 s with tension on: kinetic energy fell from 2.9e-5 to 3.9e-8 J/kg in an 80 x 80 mm tub at 4 mm
  particles and from 2.2e-5 to 1.3e-8 at 2 mm; a 15 mm vial and a puddle on glass settled too.

## Why
- The surface force is computed per particle from the local curvature. Over one oscillation it does a little more
  work on the liquid than the surface energy gives back, so a free oscillation gains energy each period.
- In liquid at rest nothing oscillates, and the velocity smoothing that runs with tension damps what starts.

## Workaround
- Leave tension off (the default) for scenes whose result depends on how a free blob of water oscillates.
- A thicker liquid's oscillation settles normally.

## Planned fix
None scheduled. The fix needs a surface term in the pressure solve that keeps the surface layer together.
