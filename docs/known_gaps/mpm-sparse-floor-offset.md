# MPM floors between grid planes: slow flows on the ground (sparse) or leaks through thin floors (dense)

**What is wrong.** Newton's MPM grids handle a wall well only when it lies on, or well inside, a plane of grid nodes.
Alloy runs the sparse grid by default. Its node planes sit at half voxels, (k + 1/2) x voxel, so the ground (z = 0)
lies exactly midway between two of them. Material resting directly on the ground then shears too little in its
bottom one or two particle layers: pastes, viscous liquids and other flows along the ground run slow. Newton's dense
grid has its planes on whole voxels, so the ground case is right. But a floor thinner than a voxel that rests on the
ground (a 4 mm container floor) then has no node plane inside it, and particles leak through. The sparse grid holds
such floors. Fills in containers are the common case, so sparse is the default (2026-11-07). Sparse
entries warn once when their material starts within a voxel of a ground plane.

**The number.** CPU, 5 mm voxel.

| Case | Sparse (default) | Dense (`ALLOY_MPM_GRID=dense`) |
|---|---|---|
| Newtonian film on the ground, 20 mm, 10 deg, 20 Pa·s: speed / analytic | 0.821 | 0.970 |
| same, 200 Pa·s | 0.874 | 0.995 |
| honey_box_150ml in liquid_box_56 (4 mm floor) after 0.3 s: lowest particle | 5.24 mm (rests on the floor) | -0.24 mm (leaks) |

A container that moved at many GPU worlds lost sand into or through its floor at every floor thickness measured
(4, 6 and 10 mm): a separate fault, not this one, fixed for 64 worlds and below
([mpm-moving-container-many-worlds.md](mpm-moving-container-many-worlds.md)).

On the sparse grid the film ratio is the same at a 2.5 mm voxel (0.811-0.817) and at half the time step (0.821). A
floor raised 1 mm (0.2 voxel) off the midplane gives 1.016. Rows and probes: the development scripts and result rows of
a development ticket (not shipped).

**Why.** Newton applies a collider at grid nodes: a node inside it takes the wall's velocity. A wall halfway between
two planes couples the lowest particles to the wall too strongly; a floor with no node inside it has nothing to hold
material out. Sparse: `grid_offset = 0.5` in `SolverImplicitMPM`. Dense: planes at `floor(bbox_min / voxel)`.

**Workaround.** For a flow along the ground that has no floor thinner than a voxel (a film or pour speed, a paste
spreading on the ground plane), ask for the dense grid: `ALLOY_MPM_GRID=dense`, an entry's `grid_type="dense"`, or
`with alloy.integration.mpm_grid.grid_override("dense"):` around the build (the film law tests do this). Dense needs
a run without CUDA graph capture: under graphs the entry keeps Newton's persistent sparse grid. Build containers with
floors at least one voxel thick (a container that moved at many GPU worlds was a separate fault:
[mpm-moving-container-many-worlds.md](mpm-moving-container-many-worlds.md)). The run's solvers record the grid in use as `solver.alloy_grid`. A dense grid that
would need more than 4 cells per particle is built sparse when its start box is that large, with one warning. If it
grows that large mid-run (particles thrown far apart), a one-world entry switches to sparse with one warning, and an
entry of separate worlds raises `MPMGridOverflowError`: on CUDA Newton's next step failed to rebind its collision
fields after such a switch (64 worlds, 2026-11-07), and on the CPU separate worlds have no sparse grid.

**The fix and when.** Newton should place walls between node planes correctly, or at least let the sparse grid take a
half-voxel translation (warp.fem's Nanogrid already takes `translation`). It is reported upstream as a Newton bug;
the fix comes with a Newton pin bump, after launch. The GPU cost of dense against sparse at 64-1,024 worlds is a row
of the Nov 8 reconciliation pass.
