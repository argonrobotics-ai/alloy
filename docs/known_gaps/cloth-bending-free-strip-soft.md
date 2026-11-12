# A free strip of cloth bends about 35 % too soft along a mesh axis

## What is wrong
- A cloth sheet (`add_cloth`, the VBD cloth route) that bends freely along one of its mesh axes bends as if its
  rigidity were about 0.64 of its material card's `elastic.bending`.
- A strip clamped level at one end droops too far under its own weight (the textile cantilever test). At a free
  length L of 1.5 bending lengths on a 12-cell mesh the tip hangs 0.57 L below the clamp. Hinges of exactly the
  card's rigidity on that mesh would give 0.43 L (+34 %). The smooth elastica gives 0.37 L; the 14 % between the
  two is the mesh's clamp, which any hinge model has.
- So free cloth bending along a mesh axis bends more than its card says: an edge that hangs square to the grid
  droops further. Bends across the mesh diagonals are not measured as a droop; by the hinge energy of a uniform bend
  they hold about 1.1 of the card's rigidity (slightly stiff). Whether this carries part of the woven drape gap
  (docs/known_gaps/cloth-woven-drape-isotropic.md) is not measured.
- The strip is at rest and `sim.check()` is ok.
- Not affected, as far as measured: plates supported on their edges. The route's hinge map is fitted there, and
  converged square plates sag 0.99-1.02 x the Navier value. That is the route's existing coverage, not a new
  measurement.

## Number
The scene (`tests/solvers/vbd/test_fabric_bending_law.py`): a strip along the mesh's x axis, the two vertex rows of
its first cell held, 12 cells free, free length L = 1.5 c with the bending length c = (D / (w g))^(1/3). D is the
card's `elastic.bending` and w the areal mass. It hangs in still air (a drag that is critical for the first bending
mode and zero at rest) for 0.4 s. CPU, one world, dt 0.01 s, default cloth accuracy, read at rest (under 1 mm/s).
References, from the card's D, its areal weight and g only: the heavy elastica D θ'' = -w g (L - s) cos θ, clamped
level, tip drop 0.3735 L; and the exact rest of the mesh's own hinge chain (rigid links, a hinge of D / cell at
every node), 0.4261 L at 12 links and 0.3998 L at 24.

| strip | tip drop / L | against the hinge chain |
|---|---|---|
| `cotton_woven` (D 1.98e-5 N m, 300 g/m², c 18.9 mm), 30.7 x 7.1 mm, 3 cells wide | 0.5705 | +33.9 % |
| `cotton_woven`, 12 cells wide | 0.5658 | +32.8 % |
| `cotton_woven`, 36 cells wide | 0.5653 | +32.7 % |
| `cotton_woven`, 24 cells along the free length | 0.5535 | +38.4 % (chain 0.3998) |

- `cloth_polyester_woven` (D 3.43e-6 N m, 158 g/m², c 13.0 mm) gives 0.5696. It is the same strip in bending
  lengths, so it shows the result does not depend on the card, not a second measurement of the error.
- `cloth_accuracy="super"` gives the same drop within 0.0002 L (measured on a 16 x 4 cell mesh).
- On Newton 1.8, the engine's next vendored update, the two fabrics give 0.5706 and 0.5697.
- Read as one rigidity on the chain, the drop is that of 0.64 D at 12 cells and 0.62 D at 24.
- On the chain, half the card's rigidity would give 0.6506 L and twice would give 0.2371 L.

## Why
- It is the rest state of the hinge energy, not the solve. More iterations give the same drop, and the hinge moments
  balance the weight's moment at every column of the mesh (0.99-1.04).
- The route writes each hinge as `edge_ke = 0.75 D L / (A1 + A2)`. On the alternating right-triangle grid only the
  hinges across the bend carry a bend along a mesh axis, so they hold 0.75 D. The 0.75 is fitted to plate sag, where
  the diagonal hinges carry the rest. The chain at 0.75 D gives 0.5204 L: most of the gap is this, by design.
- That is not all of it: the engine's 0.5705 L is the chain at 0.64 D. With the map at 1.0 (the hinges scaled by
  4/3) the tip hangs 0.48 L at 12 cells and 0.46 L at 24, against the chain's 0.43 and 0.40.
- In the rest state the vertex rows zig-zag across the width (about 12 µm on the 7 mm strip), and the strip turns
  more at each column than the hinges there read: 4 % more at the clamp and 15-22 % more at the next four columns.
  This was measured on one mesh. That the zig-zag is the cause of the rest is not proven.

## Workaround
- None in `add_cloth` itself: it has no bending setting.
- For a sheet that bends freely along a mesh axis only (a strip, an edge that hangs square to the grid): copy the
  material card into your own root (`<root>/materials/<name>.yaml`, `alloy.Scene(roots=[<root>])`) with
  `elastic.bending` x 1/0.6. Measured on the cotton cantilever above, on that mesh only: the tip hangs 0.408 L,
  4 % under the chain of the true rigidity.
- The cost: the hinge stiffness is linear in the card's value, so plates, which are right today, then come out
  1.67x too stiff, and a bend across the mesh diagonals holds about 1.9 of the card's rigidity (by the hinge
  energy, not a measured droop). Do not use it for a sheet that drapes in two directions.

## Planned fix
- A hinge map that holds in free cylindrical bending as well as in plate sag. Not scheduled.
- Until then `tests/solvers/vbd/test_fabric_bending_law.py` keeps the hinge-chain law as a strict expected
  failure and pins the present droop within 3 %, so a change of the hinge stiffness of about 5 % either way fails.
