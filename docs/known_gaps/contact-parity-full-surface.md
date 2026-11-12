# Known gap: full-surface mesh contacts can take a wrong inside/outside sign from Warp's ray parity

## What is wrong
- On a watertight mesh, Newton signs soft contacts by Warp's ray parity (`wp.mesh_query_point_sign_parity`, Warp
  1.18.0). At degenerate points, where the ray grazes a shared edge or vertex, parity gives the wrong side: inside for a
  point in a cup's cavity, or outside for a point inside its wall.
- Alloy re-signs the per-particle rows (Newton's `create_soft_contacts`) after each collide with a perturbed-parity vote
  (`alloy.contact.mesh_sign`, 2026-11-07). That covers static meshes and every mesh outside a destination's
  full-surface set.
- Not re-signed: the full-surface records of `soft_contacts_mesh` that VBD-family destinations take on moving meshes
  (robot links, grippers, tools; `ProxyContactPipeline(full_surface=True)`). Their vertex-face records take the sign
  from the same parity (`_mesh_partition_sign`), and so do the edge-penetration checks (`_is_inside`). A record with
  the wrong sign has its INSIDE flag, and so its normal, the wrong way round. The soft particle is then pushed toward
  the far side of the surface for that substep, or a penetration-recovery record is added for a particle outside the
  mesh.

## Number
- A trace on `open_cylinder(0.03, 0.06, 0.004)`: one wrong point (cavity, 4.3 mm from the inner wall,
  n.(x - p) = -4.257 mm). 0 errors in 200,000 random cavity points. Moving one ulp off the point gives the right sign.
- So it is rare, a measure-zero sheet hit by a float32 position. It has not been seen on a full-surface record, but it
  has not been looked for there either.

## Why
- Ray parity counts surface crossings along one ray. A ray through a shared edge or vertex counts that crossing twice,
  or not at all.
- The full-surface pass emits feature records (vertex-face, face-vertex, edge-edge) with sign flags. Those flags drive
  the separate geometry evaluation and `mesh_contact_valid`, so flipping a normal after the fact, as the per-particle
  fix does, would leave the flag and the recovery records inconsistent.

## Workaround
- None needed in most scenes. If a VBD contact on a moving mesh kicks the cloth the wrong way for one substep, at a
  fixed position that does not repeat one ulp away, suspect this gap.
- To rule it out, give that shape per-vertex contact: leave it out of `full_surface_shapes`. Its rows then go through
  the re-signed per-particle path.

## Planned fix
- Re-sign the full-surface vertex-face records the same way. After the mesh block is detected, run the perturbed-parity
  vote at each record's particle on a parity-signed mesh. Where it disagrees, toggle the record's INSIDE flag, drop a
  recovery record whose particle the vote puts outside, and re-run the block's evaluation (it is pure per record).
- The edge-penetration check (`_is_inside`) needs the same vote, which means a copy of that kernel, like the vertex
  query in `alloy.contact.mesh_cull`.
- Estimate: 1.5-2 days, including a planted degenerate point on a moving mesh as the law and a GPU check that the
  records are otherwise identical. Not scheduled; upstream reporting is on hold.
