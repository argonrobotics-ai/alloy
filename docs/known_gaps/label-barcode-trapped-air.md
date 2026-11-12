# Known gap: trapped air under a label is scored at 20 mm², not the task's 3 mm² inside a barcode

## What is wrong
- The `bonds` plug-in's trapped-air pockets (`Bonds.Config.trapped_air`, `alloy/plugins/bonds/air.py`) resolve a
  pocket to the label's mesh cells. The task's label criteria ask for no trapped air larger than 20 mm² over the label and
  no larger than 3 mm² inside its barcode.
- Label stock (60 µm BOPP) behaves at 2 mm cells, where one cell is 4 mm². So the 20 mm² criterion is scored (about
  5 cells), and the 3 mm² barcode criterion is not: a 3 mm² pocket is smaller than one cell.
- Each label task's result says `barcode trapped-air: not scored (needs < 1.5 mm cells; 60 µm stock freezes below
  that)` (`air.BARCODE_NOT_SCORED`), and the task counts as partial on that criterion. No binary proxy stands in for
  it.

## Number
B2 resolution check (a write-up of 2026-11-08, not shipped), 60 µm BOPP (E 2 GPa ESTIMATE), film route as shipped
(dt 1 ms, 10 VBD iterations), CPU, built near z = 0:
- a 20 mm cantilever sags 0.96-1.07x plate theory at 2 mm cells; 1.5 mm cells sag about 20 % stiff and ring;
  1 mm cells are iteration-dependent and settle 3x too stiff (0.08 mm for 0.25); 0.5 mm cells do not move;
- a pad press at 2x the bond pressure bonds the whole label at 2 and 1.5 mm cells and nothing at 1 mm;
- scoring 3 mm² needs cells of about 0.7 mm or finer.

## Why
The film route's bending stiffness converges ever more slowly as cells shrink: stiffness relative to inertia grows as
D/ℓ⁴, so at fixed iterations a fine sheet of stiff stock is effectively frozen. Built far from the origin, float32
positions freeze it sooner (the ulp at 0.5 m swallows each step's move), so label scenes are built near z = 0.

## Workaround
- Score the 20 mm² criterion at 2 mm cells (the pocket rig's laws T-1 and T-2 run there) and report the barcode
  criterion as not scored.
- A thinner or softer stock, or a barcode region meshed finer, does not help today: the stock itself freezes.

## Planned fix
A film-solver item (the film route): bending that converges at sub-millimetre cells for stiff
stock (a multigrid or Chebyshev-accelerated bending solve, or a hinge model whose stiffness does not grow as D/ℓ⁴ per
iteration), checked by the B2 resolution probe (a 60 µm BOPP cantilever within 5 % of plate theory at 0.7 mm cells,
and a pad press that bonds there). Then the pocket model scores the 3 mm² criterion unchanged. Not scheduled.
