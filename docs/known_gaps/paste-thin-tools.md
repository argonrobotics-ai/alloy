# A tool thinner than a paste's grid cell drags it in pulses (whisk wires, skewers)

**What is wrong.** An MPM paste is stepped on a grid, `voxel = min(d_feature / 4, 5 mm)`, and Newton's MPM colliders
are SDF or mesh shapes sampled at the grid nodes. A rod, wire or skewer thinner than a cell touches nodes only part
of the time, so it pushes the paste in pulses as it crosses them, and its mean drag depends on how long you average.
A rod two cells wide (10 mm at the default 5 mm voxel) drags the rigid-plastic force. A spoon, spatula or knife blade
is wider than that.

In the 1109 release (rc2-1109) a thin tool beside a paste fails the build instead: the voxel rule took a moving rod's
diameter as a container size, so the voxel became d/4 and a rod under about 8 mm asked for more MPM particles than the
budget (CapacityError). Main since 1a23be349 builds every rod at the paste's own voxel.

**The number.** CPU. Mayonnaise (Bingham, yield stress τ_y 96 Pa) 30 mm deep in a 120 x 60 mm tub; a vertical
kinematic rod of diameter d, its tip 10 mm above the floor (immersed L = 20 mm), dragged sideways at 20 mm/s; force
from the MPM collider impulses on the rod, averaged over the second half of the drag. A rigid-plastic medium resists
F = N_p τ_y d L with N_p = 9.14 (smooth) to 11.94 (rough) (Randolph & Houlsby 1984, a laterally loaded pile).

| d | Main (1a23be349), 5 mm voxel | rc2-1109 (before the fix) |
|---|---|---|
| 10 mm | N_p 9.53 (0.5 s drag; 9.6 over 0.3 s) | voxel 2.5 mm, N_p 7.1 |
| 5 mm | N_p 15.3 (15.5 over 0.3 s): acts about a cell wide | BuildError: 830,208 MPM particles > budget 200,000 |
| 1.5 mm | N_p 19.8 over 0.5 s, 8.5 over 0.3 s: pulses, 50 ms means 0-0.14 N | BuildError: 31,097,760 particles |

The probe is a development drag script (env `D` in mm, `T` the drag time); rows on the fix are a
development results file (neither shipped).

**Why.** On a 5 mm grid a sub-cell rod is seen by the few nodes it covers at each moment. Gym-Anything handles thin
stirrers with marker segments and a cylinder drag law, which Alloy does not have.

**Workaround.** For a mixing or stirring force that matters, model a whisk or skewer as rods or paddles at least two
cells (10 mm) wide; a 10 mm rod drags within the rigid-plastic band. On rc2-1109, a tool under about 8 mm beside a
paste does not build: use a tool whose middle extent is 20 mm or more (a 10 x 20 mm paddle).

**The fix and when.** A drag-marker coupling for thin rigid tools in MPM (a coupling feature, not a paste one), after
the 1109 release. The voxel rule is fixed on main (1a23be349).
