---
owner: Alloy maintainers
---
# Liquid in narrow openings and thin streams needs finer particles than the default

## What
- Liquid runs at 4 mm particle spacing by default (`liquid_accuracy="default"`). `"super"` gives 2 mm spacing, at
  about 16x the work, and `"ultra"` 1.5 mm, at about 50x.
- An opening narrower than 8 particles across is too coarse: under 32 mm at default, 16 mm at super, 12 mm at ultra.
  That covers a cup, a bottle mouth, a hole or a funnel neck the liquid sits in or must pass through. Liquid poured
  into it can partly miss.
- A pour whose nozzle is under 2 particles across is a column of single drops, not a stream (under 8 mm at default),
  and most of it scatters.
- A stream poured from a tilted container thins to single particles before it reaches a narrow receiver, at every
  setting: from a 50 mm beaker into a 15 mm vial, 2-4 % of what left the beaker landed in the vial at 1.5 mm. Surface
  tension (`alloy.Scene(liquid_tension=True)`) keeps the stream together: 48-56 % at 1.5 mm.
- The build warns once about each opening and stream. The warning names the item, its width in particles, and the
  cheapest setting that resolves it, or else a wider opening, or a pour with a higher `rate` or lower `speed`. It
  never refines silently. It checks `add_open_box`, `add_open_cylinder`, `add_block_with_hole` and cards with a
  `passage_width`; an `add_mesh` container is not checked.

## Number
Probes through `alloy.Scene` (development scripts, not shipped):
- **Water at rest in round cups (`add_open_cylinder`), 40 mm deep, 0.6 s, CPU:** healthy in every cup measured,
  12-32 mm at 4 mm, 8-24 mm at 2 mm and 8-24 mm at 1.5 mm; none left the cup, and the surface rose at most 2.3 mm.
  A cylinder of liquid fills whole rings of particles and adds layers on top: these held 95.5-101.9 % of the volume
  asked, their columns up to 20 % taller than asked (40 % in the 12 mm cup at 4 mm, 3 particles across).
- **Pour (half the cup's volume, the stream half its width), share kept in the cup, one L4:** 99.7-100 % in 12-32 mm
  cups at 4 mm (88.6 % in 12 mm before this release); 98.5-99.9 % at 2 mm; 98.1-100 % at 1.5 mm.
- **Stream into a 70 mm cup, share that lands:** 36 % at 1 particle across (4 or 2 mm); 99.0-100 % at 2-5 across.
- **Tilt pour into a 15 mm vial** (a 50 mm beaker tilted to 64 deg about its lip, one L4; five runs each, the vial
  moved up to 0.5 mm between them): 2-5 % of the poured liquid in the vial at 2 mm and 2-4 % at 1.5 mm without
  surface tension; 52-57 % and 48-56 % with it.

## Why
- SPH resolves the pressure near a wall over a kernel of 2 particle spacings each side. With fewer than about 8
  particles across, the kernels of opposite walls overlap, and the stream through the opening has no bulk.
- A thin stream breaks into a column of particles with no neighbours to hold it together; surface tension holds it.

## Workaround
- Build with the setting the warning names (super for openings of 16-32 mm, ultra for 12-16 mm), or widen the
  opening, or widen the pour's stream (`rate` up, `speed` down).
- For a stream into a narrow receiver, turn surface tension on.

## Planned fix
None scheduled. Refining the particles only near an opening would cost less than a finer setting everywhere.
