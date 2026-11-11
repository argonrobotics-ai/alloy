# MPM: on the dense grid, a world that outgrows the grid's box couples the worlds by one rounding step

Status: open, narrow. Upstream fix proposed (Newton: integer cell origin, cell size exactly the voxel).

## What

Implicit MPM's dense grid is one box shared by every world (Newton `Grid3D` with one environment per world). A
particle's cell and weights come from `(x - lo) / cell_size`, with `cell_size = (hi - lo) / res`. When the box
changes, both round differently, so every world's arithmetic moves by about one float32 ulp.

Alloy keeps a multi-world dense grid's box fixed: the first step's box plus 8 voxels each side
(`alloy.integration.mpm_worlds`). While every particle stays inside it, each world's trajectory is bitwise
independent of the others (`tests/integration/test_mpm_world_independence.py`). When a particle in any world leaves
the box, it grows for every world. A warning is logged once, and at that step the worlds couple by about one ulp.

The per-world sparse grids (the default on CUDA) index voxels from the origin and do not have this coupling.

CPU worlds split over worker processes (`alloy.build(scene, device="cpu", processes=N)`) hold the whole scene's
start box in every worker (`mpm_worlds.start_box` / `hold_box`, WP8's 9a1bd07da), so a world steps the same in
any split. When a particle leaves the held box, only its own worker's box grows. From then on that worker's
worlds differ by about one ulp from a run with another split, or with one process.

## Number

- Newton's own dense grid, which follows every world's particles each step: melting world 1 next door moved world 0
  by 1.5e-8 m with a fixed iteration count and 3.1e-8 m with each world's solve stopping on its own. That is
  melt_chocolate's scene, CPU, 2 worlds, 6 periods.
- With the held box, 0 (bitwise), while no particle leaves it.
- Three paste worlds in 1, 2 and 3 worker processes: bitwise the same
  (`tests/parallel/test_cpu_worlds.py::test_a_paste_world_steps_the_same_in_any_process`). Before the shared start
  box, the split moved a world by 2.2e-5 m at 2 worlds per process (mayonnaise paste, 0.05 s), and by 3.6 mm at one
  world per process, which built Newton's sparse single-world grid.

## Why

A box that holds still for some worlds while others outgrow it has to depend on them. A box that never moves costs
memory with no bound: Newton marks active cells in an array over the whole box every step.

## Workaround

- Use the sparse grid (the default; on CUDA it is per world).
- On the CPU, where Newton builds no per-world sparse grid, start the material near where it will stay, or set
  `grid_padding` so the first box has room for the motion.

## Related

Newton sets two material switches from every world's particles when the solver is built: the compliant path
(`min_young_modulus < inf`) and hardening (`max_hardening > 0`). A world of rigid-plastic particles next to an elastic
one moved by 1.9e-5 m in a plain-Newton test. Alloy's material cards always give a finite E, so the compliant path is
on in every Alloy MPM scene. Worlds that differ in whether they harden would still couple.

## Planned fix

In Newton: a dense grid that locates cells by `floor(x / voxel)` minus an integer origin, with its cell size exactly
the voxel. Alloy's write-up for upstream (an issue text and a repro script, with the per-world stop) is not shipped.
