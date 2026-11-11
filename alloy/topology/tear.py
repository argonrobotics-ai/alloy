# SPDX-License-Identifier: Apache-2.0
"""Break tests for particle bonds: CLOTH_EDGE (tear), SPRING, and the fracture/finish tests of TET_CUT (§3.8).

The strain of a seam bond is the material strain next to it: the largest principal engineering strain over the
triangles (CLOTH_EDGE) or tets (TET_CUT) incident to either twin. A pre-split line therefore opens when the
material at that point fails, whatever the seam's orientation (TOP-4 pulls along and across the lines). The
twin spring has rest length 0, so its force ``ke·|x_a − x_b|`` is the traction the seam carries at that vertex.
A SPRING bond uses its own stretch ``(l − l0)/l0``. A TET_CUT bond in the cutting phase also breaks when its
cut-damaged stiffness reaches 0 (cut.py).
"""

from __future__ import annotations

import warp as wp

from alloy.topology.geometry import tet_stretch_range, tri_stretch_range
from alloy.topology.types import BondKind

K_SPRING = wp.constant(int(BondKind.SPRING))
K_CLOTH = wp.constant(int(BondKind.CLOTH_EDGE))
K_TET = wp.constant(int(BondKind.TET_CUT))
PHASE_CUTTING = wp.constant(1)


@wp.func
def max_tri_strain(start: int, end: int, elem: wp.array[wp.int32], tri_indices: wp.array2d[wp.int32],
                   tri_poses: wp.array[wp.mat22], x: wp.array[wp.vec3]) -> float:
    st = float(0.0)
    for k in range(start, end):
        t = elem[k]
        r = tri_stretch_range(x[tri_indices[t, 0]], x[tri_indices[t, 1]], x[tri_indices[t, 2]], tri_poses[t])
        st = wp.max(st, r[0] - 1.0)
    return st


@wp.func
def tet_strain_range(start: int, end: int, elem: wp.array[wp.int32], tet_indices: wp.array2d[wp.int32],
                     tet_poses: wp.array[wp.mat33], x: wp.array[wp.vec3]) -> wp.vec2:
    """(max tensile, max compressive) engineering strain over the listed tets."""
    tens = float(0.0)
    comp = float(0.0)
    for k in range(start, end):
        t = elem[k]
        r = tet_stretch_range(x[tet_indices[t, 0]], x[tet_indices[t, 1]], x[tet_indices[t, 2]],
                              x[tet_indices[t, 3]], tet_poses[t])
        tens = wp.max(tens, r[0] - 1.0)
        comp = wp.max(comp, 1.0 - r[1])
    return wp.vec2(tens, comp)


@wp.kernel
def eval_particle_bonds(
    kind: wp.array[wp.int8], a: wp.array[wp.int32], b: wp.array[wp.int32], intact: wp.array[wp.int8],
    break_force: wp.array[wp.float32], break_strain: wp.array[wp.float32], spring: wp.array[wp.int32],
    rest: wp.array[wp.float32], phase: wp.array[wp.int8], elem_start: wp.array[wp.int32], elem: wp.array[wp.int32],
    tri_indices: wp.array2d[wp.int32], tri_poses: wp.array[wp.mat22], tet_indices: wp.array2d[wp.int32],
    tet_poses: wp.array[wp.mat33], x: wp.array[wp.vec3], spring_stiffness: wp.array[wp.float32],
    ext_force: wp.array[wp.float32], ext_threshold: wp.array[wp.float32],
    want: wp.array[wp.int8], strain_out: wp.array[wp.float32], force_out: wp.array[wp.float32],
):
    """``ext_force`` / ``ext_threshold`` [bonds]: a force and break force an entry solver supplies (the film's seams,
    alloy/solvers/film/seams.py); -1 = none, the spring's ke·|l - rest| against ``break_force`` as before."""
    i = wp.tid()
    k = wp.int32(kind[i])
    if intact[i] == 0 or (k != K_SPRING and k != K_CLOTH and k != K_TET):
        return
    l = wp.length(x[a[i]] - x[b[i]])
    ke = spring_stiffness[spring[i]]
    force = ke * wp.abs(l - rest[i])
    limit = break_force[i]
    if ext_force[i] >= 0.0:
        force = ext_force[i]
        limit = ext_threshold[i]
    strain = float(0.0)
    if k == K_CLOTH:
        strain = max_tri_strain(elem_start[i], elem_start[i + 1], elem, tri_indices, tri_poses, x)
    elif k == K_TET:
        strain = tet_strain_range(elem_start[i], elem_start[i + 1], elem, tet_indices, tet_poses, x)[0]
    elif rest[i] > 0.0:
        strain = (l - rest[i]) / rest[i]
    strain_out[i] = strain
    force_out[i] = force
    brk = strain >= break_strain[i] or force >= limit
    if k == K_TET and phase[i] == PHASE_CUTTING and ke <= 0.0:
        brk = True
    if brk:
        want[i] = wp.int8(1)
