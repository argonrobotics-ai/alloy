# Style3D cloth stiff in stretch for its step drapes slowly or stays flat

## What is wrong
- A cloth card with `elastic.cloth` (warp and weft stretch, shear, bending: it routes to CLOTH_STYLE3D) whose stretch
  stiffness is large for its mass, mesh and step barely bends out of plane: it creeps (about 2 mm/s) and stays nearly
  flat, while `sim.check()` and `sim.health()` read OK.
- Style3D solves each step with a fixed matrix built from the stretch and bending stiffness. When its diagonal
  outweighs the inertia term m/dt² by more than about a thousand per vertex, each iteration releases only a little of
  the out-of-plane motion.

## Number
- Woven fabrics from KES-F data on a Cusick drape rig (Petrak 2021 F1 / F4 / F9, `alloy.scene.twins.fabric_rig` arm A,
  8 mm cells, 1.25 ms steps): ratio 9,700 (F1) and 2,900 (F9). They drape to DC 0.98-0.99, against measured 0.22-0.65.
- Its own levers barely help (DC5, rim sag at 0.7 s):

  | change | rim sag at 0.7 s |
  |---|---|
  | as built | -1.0 mm |
  | 40 PCG iterations | no change |
  | 40 nonlinear iterations | -3.8 mm (3.5x the wall time) |
  | dt / 8 | -11 mm (7.5x the wall time) |
  | stretch ÷10 (ratio about 1,000) | still locked, DC 0.949 |
  | stretch ÷100 (ratio about 100) | drapes |

- The ratio grows as dt² / (areal mass x cell size²): a finer mesh or a lighter fabric locks sooner.
- The library's `napkin_cotton_aniso` builds at about 1,300, so it warns. Dropped 6.5 cm onto a 12 cm ball, it still
  hung 27 mm above the floor at 1.5 s; a 30 cm sheet of `cotton_woven` or `cotton_woven_aniso` (ratio 70) reached the
  floor. The napkin's own size differs, so this is a sign, not a measured lock.

## Why
Style3D solves each step with a fixed matrix built once from the stretch and bending stiffness (projective dynamics).
When its diagonal outweighs the inertia term m/dt² by about a thousand per vertex, each of its iterations releases only
a little out-of-plane motion, so the sheet creeps instead of falling. DC5 ruled out the card's units, gravity, the
clamp, bending and contact (bending ÷1000 leaves the sag unchanged). The build warns, with the ratio: "a Style3D
cloth (an elastic.cloth card) is stiff in stretch for its step" (`alloy.scene.style3d_lock`, threshold 1,000).

## Workaround
- For a woven fabric's drape, give the card no `elastic.cloth`. It routes to the VBD cloth, which drapes it, within
  its own isotropy limit (`cloth-woven-drape-isotropic.md`).
- Check one drape against a measurement before trusting a Style3D sheet's shape.
- A coarser mesh lowers the ratio (cell size squared), at the cost of resolution.

## Planned fix
None planned. A disclosed tensile cap on the Style3D route needs a coordinator ruling (DC5: F9 at 0.651 against 0.647
with warp and weft capped, F1 0.347 against 0.222). Sources: DC5's probe notes and ratio script, and
DC5's write-up of 2026-11-08 (not shipped).
