---
owner: Alloy maintainers
---

# MPM material through container walls thinner than about three voxels

**What is wrong.** Sand, soil and pastes (Newton's MPM) can pass through a container wall that is only a voxel or two
thick when they lean on it: a container tilted with `sim.drive`, or material pushed against its side. `sim.check()`
stays ok, so the loss looks like a correct result. From 0.1.8 a fixture's walls (`add_open_box`, `add_open_cylinder`,
`add_block_with_hole`, static or kinematic) keep the voxel of the material inside within a third of the wall: in the
cases measured below, 0 or 1 grain then crossed. What that rule leaves:

- A fixture whose parts are wide gives their own size, as through 0.1.7. Its voxel can then be coarser than a third
  of its wall. The open box measured below (4 mm walls, 5 mm voxel) held. The rule never asks for a finer voxel than
  a part's own size gave: a cup of 30 mm radius with 2 mm walls keeps the 2.1 mm voxel its 8.5 mm slabs give, about
  one voxel per wall. Kinematic, with a 30 mm cube of sand, tilted to pour (CPU, one run): 2 of 21,952 grains
  crossed, where through 0.1.7 (5 mm voxel) 12 of 1,728 did.
- Containers that are not fixtures give their own size, as through 0.1.7: parts placed with `add_box`, meshes
  (`add_mesh`, library objects such as `liquid_box_56`: 4 mm walls, its shipped fills at a 5 mm voxel) and free
  rigid containers. Not measured under load.
- Saved cards do not keep a part's fixture. `alloy.load` and `alloy.check` of a saved scene route its fixtures by the
  0.1.7 rule, with another voxel and particle count than `alloy.build(scene)`.
- The rule does not look at distance: a fixture anywhere in the scene sets the voxel of every MPM fill, as static parts
  did through 0.1.7, and from 0.1.8 kinematic fixtures do too. A kinematic thin-walled fixture can raise the count of
  a fill it never touches: sand in an open box beside a kinematic default vial 0.5 m away takes 27,000 particles, was
  512. A kinematic cup of 30 mm radius with 2 mm walls above a 200 x 200 x 30 mm sand bed in a static tray now fails to
  build with `CapacityError` (989,632 particles, over the 200,000 budget); through 0.1.7 it built with 76,800. The
  build warns first, naming the cup. The fixes: a thicker wall, fewer `segments=` (wider slabs), or
  `ALLOY_MPM_CAVITY=off`.
- Granular fills (cards with a grain diameter, such as `rice_cup_fill`) read a fixture's inner width as the narrowest
  opening they must pass. Under about 12 grain diameters (36 mm for its 3 mm rice), a fill of more than 2,000 grains
  fails to build with a jamming `RouteError`: any fixture, open boxes and cylinders, static or kinematic, not only
  `add_block_with_hole`'s hole. `rice_cup_fill` in a 30 mm-wide open box built through 0.1.7 and fails now; over
  `add_block_with_hole`, holes of 20, 30 and 36 mm fail and 40 mm builds.

**The number.** The default vial: `add_open_cylinder(inner_radius=0.01, depth=0.04)` (24 slabs, 4 mm wall) holding a
12 mm cube of `sand_silica_dry`. A grain counts once it has crossed the far face of any part (beyond a wall slab's
outer face within that slab's width and height, or under the floor), at any control period, unless it left through
the mouth first: grains that cross fall away or flicker back, so a count at the end misses them. Two other counting
rules (anywhere outside the vial's outer outline below the lip; 0-5 mm past a face) gave the same count in every row
checked with them. Sideways: the static vial, with gravity turned to -x after a 0.5 s settle, then 1.5 s. Tilt: a
kinematic vial settled 2 s, turned 90 degrees in 1 s with `sim.drive` about its outer lip on the +x side, then held
1 s, one of two ways:
- outward: the vial 5 cm above the ground, `angular_velocity=(0, -pi/2, 0)`. The bottom swings out past the lip; the
  mouth ends facing -x and the sand on the -x wall.
- pouring: `angular_velocity=(0, pi/2, 0)`. The bottom rises over the lip; the mouth ends facing +x and the sand on
  the +x wall. Rows from a second, independent script (vial on the ground), tipped toward 0 and 45 degrees.

CPU and one NVIDIA L4, 2026-11-17 to 2026-11-19; one seed per row.

| Voxel | Particles | Tilt outward, grains crossed (CPU / L4) | Tilt pouring (CPU: 0 / 45 deg) | Sideways (CPU / L4) | Settled fill height (CPU / L4) |
|---|---|---|---|---|---|
| 1 mm (0.1.7, static vial: its slabs' size) | 13,824 | not run / 0 | not run | not run / 1 | 5.87 / 5.87 mm |
| 1.33 mm (0.1.8: a third of the wall) | 5,832 | 0 / 0 | 0 / 1 | 0 / 0 | 6.02 / 6.01 mm |
| 2 mm (half the wall) | 1,728 | 0 / 3 | not run | 14 / 19 | 6.87 / 6.84 mm |
| 3 mm (the cube's own size; 0.1.7, kinematic vial) | 512 | 2 / 1 | 0 / 1 | 16 / not run | 8.87 / 8.84 mm |

At a 3 mm voxel, by wall (CPU): sideways, 16 grains crossed a 4 mm wall, 5 a 4.5 mm one, none a 6 or 8 mm one;
tilted outward, 2 crossed the 4 mm wall, 3 the 6 mm one (during the hold) and none the 8 mm one.

Other rows:
- A 20 mm cube in a kinematic vial of 15 mm radius (4 mm wall). Through 0.1.7 it ran at its own 5 mm voxel (512
  particles). Tilted outward, 75 grains (15 %) crossed a wall, 14 of them still outside at the end (CPU: the same count
  by all three rules, and the same counts on two machines, 2026-11-17 and 2026-11-19); they crossed the slabs 30 to 45
  degrees either side of the one the sand rests on, during the hold. Tilted to pour, 1 or 2 crossed (2 with the second
  script toward 0, 7.5 and 45 degrees; 1 with the first). From 0.1.8 it runs at 1.33 mm (27,000 particles): none
  crossed, outward (CPU and L4) or pouring (CPU, the three directions), at 24-30x the CPU run time.
- A default vial with 1 mm walls, sideways, at the voxel the rule gives (0.73 mm, 32,768 particles: its slabs' own
  size, as through 0.1.7): none crossed (L4).
- An open box (`add_open_box`, 30 mm square inside, 4 mm walls), 20 mm cube at its own 5 mm voxel: none crossed
  tilted or sideways (CPU). Its walls lie along the grid, where the vial's slabs sit at angles to it.

**Why.** Newton's MPM holds material out of a collider through the grid nodes inside it, and every Alloy MPM step
ends by pushing a particle found inside a collider to the nearest surface. A wall a voxel or two thick has few nodes
inside it. Under load a particle can be carried past the wall's middle, and the nearest surface is then the far one.
The crossings measured came through the vial's slabs, which sit at angles to the grid.

**Workaround.** Make walls at least three voxels thick. The build warns when a container's parts raise the particle
count 4x or more, and names the wall thickness that gives the material its own voxel (9 mm for the default vial's
cube). `sim.routes()` lists each MPM object's voxel and the part that set it, and says when a fixture's wall held it
to a third of the wall. A container that is not a fixture: measure its loss with your own load before you trust a
fill. `ALLOY_MPM_CAVITY=off` restores the 0.1.7 rule (each static part's own size; kinematic fixtures not counted),
for comparisons.

**The fix and when.** The 0.1.8 rule above. A health flag for material that leaves its container through a wall is a
separate proposal, not scheduled.
