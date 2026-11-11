# Known gap: `cut_model: copies` on a coarse mesh (pieces with one node layer, no kerf, what `.points` holds)

## What
With `topology: {cut_model: copies}` a knife cuts a soft object along any path. A cut tet's material goes to two
fragment tets of the whole tet's shape. Each fragment keeps the mesh nodes on its own side of the cut and takes a
copy vertex for every node on the far side. Those copies are not material: they lie outside the piece and nothing
touches them. That gives five limits, all largest on a coarse mesh.

1. **A piece needs two layers of mesh nodes.** A piece left with one layer of nodes has nothing under its cut side,
   so it tips toward the knife and leans through it.
2. **The knife opens no kerf between nodes.** Only mesh nodes meet the blade's faces. Where no node lies within the
   blade's thickness, the cut faces stay inside the blade.
3. **`"<name>.points"` holds the copies too.** They sit on their mesh node until a cut wakes them, then some lie
   outside the material. A bounding box or centroid taken over `.points` after a cut is wrong.
4. **A cut piece can creep on the ground** at the soft solver's default accuracy.
5. **A drawing from `alloy.export` frames shows each fragment as its whole tet**, so the two pieces overlap across
   the cut by up to one cell, and the cut face itself is not drawn.

Also: one knife per world. A second blade in a world raises `ValueError` at build, naming both blades.

## Number
All on the CPU, on the example's 20 x 15 x 15 mm agar block (50 kPa) of 96 tets (cells 5 x 7.5 x 7.5 mm) and its
1.2 mm knife at 0.05 m/s (`examples/topology/cut_anywhere.py`; main 6f5da6ef2 with this page's commit).

| Limit | Case | Value |
|---|---|---|
| 1 | knife plane 1.7 mm off the middle y plane (2 cells across): the 5.8 mm piece has one node layer | its top outer nodes move from y 7.5 mm to 2.3 mm by 0.4 s (the knife's plane is at 1.7 mm); `sim.check()` ok |
| 1 | knife plane 1.7 mm off the middle x plane (4 cells across): both pieces have two layers or more | every node within 0.1 mm of its settled height, none sideways; 24 tets cut; max penetration 1 um |
| 2 | that x cut, knife still in the block | the pieces have parted 0.09 mm (blade 1.2 mm) |
| 3 | that block | 135 points for 45 mesh nodes; in the y cut a copy read 2.0 mm below the ground plane |
| 4 | that x cut, the 8.3 mm piece (1.8 g), after the knife stops | 0.56 mm/s for at least 2.7 s at 10 solver iterations; 0.41 at 20, 0.21 at 40, 0.007 at 160 (set on the engine's solver in a probe) |
| 4 | the 11.7 mm piece (2.8 g) | 0.005 mm/s |

## Why
- 1, 3: a copy stands in for a node of the other piece, so it carries no ground or blade contact (a far copy pushed
  by the blade shoved material at 0.39 m/s in an off-node stroke, `alloy/topology/vnode_split.py`). A piece is held
  up by its own mesh nodes only.
- 2: contact is between particles (mesh nodes) and shapes. A cut face between nodes has no collider.
- 4: the soft solver runs 10 iterations per step and its momentum correction is off while a body touches the ground
  (`alloy/solvers/vbd/solver.py`, Scope). The leftover is a small steady push on a piece whose mass hangs past its
  last supported node row. The probe raised the count on the engine's solver; whether a scene setting
  (`cloth_accuracy`) reaches a soft body's solve was not tried.
- 5: the frame's `mesh_tets` are the fragment tets; where the cut plane crosses them is not exported.

## Workaround
- Place the cut, or choose the tet budget, so that each piece keeps at least two layers of mesh nodes across its
  thinnest direction. In the example the knife crosses the block's long axis (5 mm cells), not its short one.
- For a kerf of the blade's thickness use cells no larger than the blade is thick. That is not measured here.
- Read a cut object's mesh nodes, not all of `.points`:
  `nodes = frame_export(sim, [0])[0].topology.mesh_vertex_active["<name>"]` taken before any cut is True for the
  mesh nodes (`from alloy.export import frame_export`); index `.points` or the frame's `meshes["<name>"]` with it.
- `frame.topology_version >> 40` is the number of tets cut so far.
- Do not read a cut piece's rest position to better than 1 mm per second of simulated time.

## Planned fix
None scheduled. A cut-face collider (2) and ground contact for the material under a copy (1) are the two changes
that would remove the mesh-size limits; exporting each split tet's cut plane would fix the drawing (5).
