# Pastes and viscous liquids run slow along the ground (MPM, default grid)

**What is wrong.** Liquids and pastes on the MPM paste route (anything thicker than 1 Pa·s, or with a yield stress:
honey, ketchup, mayonnaise, the food pastes) flow slower than their card viscosity says along the ground, or along
any floor that lies midway between two of the grid's node planes. The bottom one or two particle layers move at about
half their real speed; above them the paste shears at its card viscosity. The cause is the MPM grid's floor offset,
[mpm-sparse-floor-offset](mpm-sparse-floor-offset.md); this page gives the paste-flow numbers.

**The number.** A Newtonian film on an incline between frictionless side walls, 20 mm thick, 10 deg, on the ground;
depth-averaged speed against the analytic mean at its own thickness:

| Film | Sparse grid (default) | Dense grid |
|---|---|---|
| 20 Pa·s | 0.82 (0.821 at a 5 mm voxel, 0.817 at 2.5 mm) | 0.970 |
| 200 Pa·s | 0.874 | 0.995 |
| 20 Pa·s, 30 deg | 0.927 | |
| Bingham, 50 Pa yield, 50 Pa·s, 30 mm, 30 deg (law L4) | 0.995 | 1.055 |

The whole film carries a near-constant speed deficit from its base: 2.2-2.7 mm/s at 20 Pa·s, at 10 or 30 deg and at
5 or 2.5 mm voxels, so slow films lose the largest share. On the dense grid the Bingham film of law L4 runs 5.5 %
fast (its 0.995 on the sparse grid included the base's slowdown). Spreading, slumping and stopping on a slope are set
by the yield stress and are not affected.

**Why.** The default sparse grid puts its node planes half a voxel off the origin, so the ground lies exactly between
two and the boundary condition holds the bottom layers back over half a cell. The dense grid puts the planes on the
ground; it is right for flows along the ground, but through 0.1.2 a floor thinner than a voxel resting on the ground
(a 4 mm container floor) leaked there. Every MPM step now ends by pushing particles back out of colliders, and the
dense grid then holds the floor measured ([mpm-sparse-floor-offset.md](mpm-sparse-floor-offset.md)); fills in
containers keep the sparse grid, the default.

**Workaround.** For a flow along the ground with no floor thinner than a voxel (a film, a pour speed, a paste spreading
on the ground plane), ask for the dense grid: `ALLOY_MPM_GRID=dense`, or
`with alloy.integration.mpm_grid.grid_override("dense"):` around the build, in a run without CUDA graphs (the
`"debug"` profile). The film laws (L4, L5) do this. Otherwise expect films on the ground to run slow by about 2-3 mm/s
at 20 Pa·s: 18 % for a 20 mm film at 10 deg, 7 % at 30 deg. Liquids under 1 Pa·s route to SPH and are not affected.

**The fix and when.** Sparse stays the default (2026-11-07). A sparse grid that places walls between node planes
correctly, or takes a half-voxel offset, is a Newton fix, after launch
([mpm-sparse-floor-offset](mpm-sparse-floor-offset.md)). It also feeds [paste-shear-thinning](paste-shear-thinning.md).
