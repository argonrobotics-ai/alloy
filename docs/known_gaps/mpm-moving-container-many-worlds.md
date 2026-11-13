# MPM: at many GPU worlds, sand and paste sank into bodies they touched (fixed for 64 worlds and below)

Status: fixed for 64 worlds and below (0.1.0). Alloy now picks a particle's grid cell in integers
(`alloy.integration.nanogrid_lookup`, which replaces one Warp function at run time: `alloy/_vendor/VENDORED.md`).
Above 64 worlds, MPM with a body is not validated: Newton's sparse grid runs out of capacity (status 8) or memory
before the fix's law can run there, and such a build warns (`alloy.integration.mpm_moving`).

## What

- Before the fix, at many worlds on a GPU, MPM grains (sand, soil, paste) sank into bodies they touched, and
  `sim.check()` stayed ok. Affected: a container that a drive moves, a kinematic tool or press, a robot's link, and
  a free box resting on the material. A sand slab on the ground alone held (an audit): static shapes did not show
  it.
- How much grew with the world's place in the batch: world 0 of 64 read like a lone world, and the last worlds were
  the worst. Sand in a lifted open box sank into or through its floor at every floor thickness measured (4, 6 and
  10 mm, against a 5 mm voxel).
- With the fix, at 64 worlds every world reads as one world does, in each scene re-run (Number).
- A build with MPM material and any body that is not welded to the world (free, kinematic or an articulation), on a
  GPU at more than 64 worlds, warns.

## Number

One L4, 2026-11-11. Bodies in the material: the largest number of grains more than 1 mm inside the body at any
control period, per world. Before: main; after: the integer lookup (the law runs, not shipped).

| scene (grains per world) | 1 world | 64 worlds before: world 0 / last 8 mean / worlds with any | 64 worlds after |
|---|---|---|---|
| kinematic blade ploughing a sand slab (16,128) | 4 | 4 / 66 / 64 | 4 in every world |
| blade: worlds with grains through it (behind its face) | 0 | 57 of 64 | 0 |
| kinematic plate pressing a dough block (6,912) | 0 | 1 / 32 / 64 | 0 in every world |
| a free 0.3 kg box resting on a sand slab (an audit) | 0 | 45 of 64 (at most 5) | 0 in every world |
| plate, every world's grid offset shifted 131,072 voxels | | about 633 in every world, world 0 included | 0 in every world |
| servo gantry's finger pushed into sand and swept (13,824) | 11 | 12 / 76 / 64 | 12 in every world (an audit) |

- 1 world is bit for bit with Warp's own lookup (it has no offsets, and that branch is unchanged).
- Newton's and Alloy's GPU MPM laws pass with the fix (17 tests).
- The lifted box, with the fix (an audit, 6 mm floor, 64 worlds): 0, 0 and 0 worlds at 0.099, 0.100 and
  0.101 m/s (before: 42, 0 and 13), the 64 worlds within 0.007 mm of each other; a 128-world lift sank none.
- The lifted box before the fix: sand_silica_dry 4 x 4 x 2 cm in an open box 6 x 6 x 5 cm inside, at rest for
  0.05 s, then lifted for 0.15 s at the speed shown, and read 0.05 s after the lift stops. Worlds out of 64 with a
  grain centre within 1.5 mm of the floor's mid-plane or below it:

| floor | 0.099 m/s | 0.0995 | 0.100 | 0.1005 | 0.101 |
|---|---|---|---|---|---|
| 4 mm | 0 | | 43 | | 0 |
| 6 mm | 42 | 0 | 0 | 32 | 13 |
| 10 mm | 52 | 56 | 24 | 0 | 0 |

- 1 world read 0 at all five speeds (6 mm floor) and at 0.099, 0.100 and 0.101 (4 mm). The deepest grain went up to
  15.7 mm below the floor's bottom face (6 mm floor at 0.101).
- Above 64 worlds (a free box on a 2 x 2 x 1 cm slab, with the fix and without): at 256 worlds the grid rebuild
  failed with status 8 and the step ended in SparseMemoryError; at 1,024 worlds the runs stalled past 25 minutes, and
  once ended in CUDA error 700 ([mpm-gpu-memory.md](mpm-gpu-memory.md)).

## Why

- Alloy steps the worlds of an MPM entry in one shared sparse grid (Warp's `fem.Nanogrid` with environments). Each
  world's cells sit at an integer offset along one axis: world w is about 22 w voxels out at this size.
- Warp's cell lookup for such a grid added the world's offset to the particle's local grid coordinate in float32 and
  then took the floor. Near a cell face that floor can land in the neighbouring cell, and the error grows with the
  offset: 1e-4 voxel at 1,400 voxels, 0.016 voxel at 131,072. The material's contact with a body that moves, or rests
  on it, turned those misplaced grains into grains inside the body.
- The fix takes the floor of the local coordinate, then adds the integer offset, so every world's lookup is exact
  wherever it sits in the grid.
- It was not the floor's thickness, which is what [mpm-sparse-floor-offset.md](mpm-sparse-floor-offset.md) covers,
  and not a race: every run repeated to the digit.

## Workaround

- 64 worlds or fewer per GPU: none needed.
- Above 64 worlds with a body: build 64 worlds or fewer per GPU, or check the lowest grain against the floor or tool
  yourself (`sim.check()` does not flag a sunk grain).
- On another Warp than the pinned 1.18.0, or an edited lookup, the fix stays off with a logged warning and the old
  fault returns: there build 8 worlds or fewer, or step on the CPU.

## Planned fix

- Done (0.1.0): the integer cell lookup.
- Above 64 worlds: the sparse grid's capacity and memory ([mpm-gpu-memory.md](mpm-gpu-memory.md)), then the fix's law
  (per-world counts flat in the offset) at 256 and 1,024 worlds.
- A health check that flags MPM particles inside a collider.
- Upstream Warp: the same lookup in integers (upstream posting is on hold).
