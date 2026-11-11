# Cloth sheets falling onto each other can pass through each other

## What is wrong
- Sheets dropped onto each other can end with the layer order broken: parts of an upper sheet below the sheet under
  it. `add_cloth` sheets collide with each other (and each with itself) through VBD's self contact, which should keep
  them apart.
- It was seen with several sheets released together over the ground, so each lands on the ones that landed before it.
- Nothing flags it: `sim.health()` does not read INVALID, no contact was dropped for lack of room (no
  SELF_CONTACT_OVERFLOW), and `sim.check()["ok"]` stays True.
- Sheets placed 2.5 mm apart and left to settle kept their order over the 0.15-0.24 s measured.

## Number
CPU, the default cloth settings, main 03616d08c (2026-11-10). Cotton sheets of 16 x 16 vertices, 2 mm thick, 5 cm
apart in height, released together over a ground plane. Measured per vertex column (the same grid vertex in every
sheet) as the height difference to the sheet below; a column is crossed when that is 0 or less ([internal note]
`audit/stack/work/review/gap13b`, RESULTS.md section 3, finding PHYSA-10):

| Sheets | Time reached | Crossed columns | Smallest gap | Self-contact rows used |
|---|---|---|---|---|
| 12 | 1.36 s | 27 | -10.5 mm | 24 / 46 of 32 / 64 |
| 20 | 0.95-0.99 s | 70-108 | -5.3 to -15.3 mm | 24 / 40 of 32 / 64 |
| 12, placed 2.5 mm apart | 0.24 s | 0 | 2.0 mm | - |
| 20, placed 2.5 mm apart | 0.15 s | 0 | 1.98 mm | - |

- The same with the self-contact rows sized to the scene (`docs/known_gaps/vbd-self-contact-buffer-overflow.md`):
  no pair was dropped in either mode.
- The fastest particle at the end of the 12-sheet run moved at 1.05 m/s, under the 3 m/s self-contact cap
  (`docs/known_gaps/vbd-self-contact-speed-cap.md`). Speeds at the moments of impact were not recorded.

## Why
- Not traced yet. Ruled out by the rows above: dropped contact pairs (the rows never filled) and contact rows sized
  too small (the scene-sized rows give the same crossings).
- Open: how the sheets' impacts on the sheets below interact with the motion cap and the truncation that keeps
  self contact penetration-free, and whether the cloth's step or iteration count changes it.

## Workaround
- None verified.
- To check a scene, compare neighbouring sheets' heights yourself (record `"<sheet>.points"` and take, per grid
  vertex, the height difference to the sheet below); health will not report it.
- A stack that only needs to rest can be placed instead of dropped: sheets 2.5 mm apart kept their order over the
  short runs above (not checked over seconds).

## Planned fix
- Trace the crossings on the 12-sheet drop (which pairs cross, at what relative speed, before or after the
  truncation), then fix in the cloth self-contact path. Not before the 0.1.0 tag.
