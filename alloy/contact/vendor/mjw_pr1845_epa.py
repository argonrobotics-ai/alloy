# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Modified and vendored by Alloy (WP3) from google-deepmind/mujoco_warp PR #1845 by almightychang (Sam, Joochul Chang),
# open, head 54f56caf58b83a4bafefc53ae89e0445f0a90ac8: its first six EPA commits (8cffc23 dd30420 b934fc6 7d069ca
# e8cf449 a5e624b) cherry-picked onto mujoco_warp/_src/collision_gjk.py at Alloy's pin
# f34c6dd52813852d0f1a2ab6d1f4d21ef1f0829f (v3.17.0; first merged onto 86e2fae0, 3.16.0: on v3.17.0 the six apply
# cleanly and leave every definition kept here unchanged, 2026-11-09). Left out: 54f56ca (it turns a touching cylinder /
# box pair into a 1.1 mm contact) and a3b060a (multicontact's four-point pick). Alloy's change: only the definitions
# those commits change, and those that name them, are kept, verbatim and in the file's order; every other name comes
# from the pinned collision_gjk. This file: EPA's loop, epa_phase and ccd.
# Installed by alloy.contact.mjw_epa; drop it once the MuJoCo Warp pin includes the fix.
# License: LICENSE-mujoco_warp (Apache-2.0).
# ==============================================================================

from typing import Tuple

import warp as wp

from alloy._vendor.mujoco_warp._src.collision_core import Geom
from alloy._vendor.mujoco_warp._src.types import GeomType
from alloy._vendor.mujoco_warp._src.types import OverflowType

from alloy._vendor.mujoco_warp._src.collision_gjk import FLOAT_MAX
from alloy._vendor.mujoco_warp._src.collision_gjk import GJKResult
from alloy._vendor.mujoco_warp._src.collision_gjk import MIN_EPATOL
from alloy._vendor.mujoco_warp._src.collision_gjk import _delete_face
from alloy._vendor.mujoco_warp._src.collision_gjk import _discrete_geoms
from alloy._vendor.mujoco_warp._src.collision_gjk import _get_edge
from alloy._vendor.mujoco_warp._src.collision_gjk import _get_face_verts
from alloy._vendor.mujoco_warp._src.collision_gjk import _gjk_simplex_to_epa
from alloy._vendor.mujoco_warp._src.collision_gjk import _invalidate_face
from alloy._vendor.mujoco_warp._src.collision_gjk import _is_face_deleted
from alloy._vendor.mujoco_warp._src.collision_gjk import _is_invalid_face
from alloy._vendor.mujoco_warp._src.collision_gjk import _supports_multiccd
from alloy._vendor.mujoco_warp._src.collision_gjk import _tri_affine_coord
from alloy._vendor.mujoco_warp._src.collision_gjk import gjk_phase
from alloy._vendor.mujoco_warp._src.collision_gjk import support
from alloy.contact.vendor.mjw_pr1845_polytope import Polytope
from alloy.contact.vendor.mjw_pr1845_polytope import _attach_face
from alloy.contact.vendor.mjw_pr1845_polytope import _epa_support
from alloy.contact.vendor.mjw_pr1845_polytope import _polytope2
from alloy.contact.vendor.mjw_pr1845_polytope import _polytope3
from alloy.contact.vendor.mjw_pr1845_polytope import _polytope4

# TODO(team): improve compile time to enable backward pass
wp.set_module_options({"enable_backward": False})


@wp.func
def _add_edge(pt: Polytope, e1: int, e2: int) -> int:
  n = pt.nhorizon

  if n < 0:
    return -1

  # an edge is directed as in its deleted face, and is interior if the other face across it, which
  # holds it reversed, is deleted too
  edge = e1 + (e2 << 10)
  twin = e2 + (e1 << 10)

  for i in range(n):
    if twin == pt.horizon[i]:
      pt.horizon[i] = pt.horizon[n - 1]
      return n - 1

  # out of memory, force EPA to return early without contact
  if n == pt.horizon.shape[0]:
    return -1

  pt.horizon[n] = edge
  return n + 1


@wp.func
def _touches_horizon(pt: Polytope, face: wp.vec3i) -> bool:
  for j in range(3):
    twin = face[(j + 1) % 3] + (face[j] << 10)
    for i in range(pt.nhorizon):
      if pt.horizon[i] == twin:
        return True
  return False


@wp.func
def _epa_witness(
  pt: Polytope, geom1: Geom, geom2: Geom, geomtype1: int, geomtype2: int, face_idx: int
) -> Tuple[wp.vec3, wp.vec3, float]:
  face = _get_face_verts(pt.face[face_idx])
  # compute affine coordinates for witness points on plane defined by face
  v1 = pt.vert[2 * face[0]] - pt.vert[2 * face[0] + 1]
  v2 = pt.vert[2 * face[1]] - pt.vert[2 * face[1] + 1]
  v3 = pt.vert[2 * face[2]] - pt.vert[2 * face[2] + 1]

  coordinates = _tri_affine_coord(v1, v2, v3, pt.face_dist[face_idx] * pt.face_normal[face_idx])
  l1 = coordinates[0]
  l2 = coordinates[1]
  l3 = coordinates[2]

  # face on geom 2
  v1 = pt.vert[2 * face[0] + 1]
  v2 = pt.vert[2 * face[1] + 1]
  v3 = pt.vert[2 * face[2] + 1]
  x2 = wp.vec3()
  x2[0] = v1[0] * l1 + v2[0] * l2 + v3[0] * l3
  x2[1] = v1[1] * l1 + v2[1] * l2 + v3[1] * l3
  x2[2] = v1[2] * l1 + v2[2] * l2 + v3[2] * l3

  # correct witness points for hfield geoms
  i1 = pt.vert_index[2 * face[0]]
  i2 = pt.vert_index[2 * face[1]]
  i3 = pt.vert_index[2 * face[2]]
  if geomtype1 == GeomType.HFIELD and (i1 != i2 or i1 != i3):
    # TODO(kbayes): Fix case where geom2 is near bottom of height field or "extreme" prism heights
    n = wp.vec3(0.0, 0.0, 1.0)

    # height field prism vertices
    a = geom1.polyvert[3]
    b = geom1.polyvert[4]
    c = geom1.polyvert[5]

    # TODO(kbayes): Support cases where geom2 is larger than the height field
    if geomtype2 == GeomType.CAPSULE or geomtype2 == GeomType.SPHERE:
      radius = geom2.size[0]
      margin = geom2.margin
      geom2.margin = 0.0
      geom2.size = wp.vec3(0.0, geom2.size[1], geom2.size[2])
      sp = support(geom2, geomtype2, x2)
      x2 = sp.point - (0.5 * margin + radius) * n
      geom2.size[0] = radius
      geom2.margin = margin
    else:
      x2 = wp.normalize(x2)
      sp = support(geom2, geomtype2, x2)
      x2 = sp.point

    coordinates2 = _tri_affine_coord(a, b, c, x2)
    if coordinates2[0] > 0.0 and coordinates2[1] > 0.0 and coordinates2[2] > 0.0:
      x1 = coordinates2[0] * a + coordinates2[1] * b + coordinates2[2] * c
    else:
      p = c
      p = wp.where(coordinates2[1] > 0, b, p)
      p = wp.where(coordinates2[0] > 0, a, p)
      x1 = x2 - wp.dot(x2 - p, n) * n
    return x1, x2, -wp.norm_l2(x1 - x2)

  # face on geom 1
  v1 = pt.vert[2 * face[0]]
  v2 = pt.vert[2 * face[1]]
  v3 = pt.vert[2 * face[2]]
  x1 = wp.vec3()
  x1[0] = v1[0] * l1 + v2[0] * l2 + v3[0] * l3
  x1[1] = v1[1] * l1 + v2[1] * l2 + v3[1] * l3
  x1[2] = v1[2] * l1 + v2[2] * l2 + v3[2] * l3

  return x1, x2, -pt.face_dist[face_idx]


@wp.func
def _epa(
  # In:
  tolerance: float,
  epa_iterations: int,
  pt: Polytope,
  geom1: Geom,
  geom2: Geom,
  geomtype1: int,
  geomtype2: int,
  is_discrete: bool,
  warn_overflow: bool,
  worldid: int,
  # Data out:
  overflow_out: wp.array[int],
) -> Tuple[float, wp.vec3, wp.vec3, int]:
  """Recover penetration data from two geoms in contact given an initial polytope."""
  upper = FLOAT_MAX
  idx = int(-1)
  pidx = int(-1)
  epsilon = wp.where(is_discrete, MIN_EPATOL, tolerance)
  nvalid = pt.nface  # number of potential faces for expanding the polytope

  # the face vertices are encoded in 10-bits that index the vertex array,
  # so iterations must be cap to limit the number of generated vertices
  # (one new vertex per iteration)
  epa_iterations = wp.min(epa_iterations, 1000)
  for _ in range(epa_iterations):
    pidx = idx
    idx = int(-1)
    lower = float(FLOAT_MAX)

    # find the face closest to the origin (lower bound for penetration depth); the origin may lie on
    # or, by rounding, slightly outside a face, which only makes its distance zero or negative
    for i in range(pt.nface):
      if not _is_invalid_face(pt.face[i]) and pt.face_dist[i] < lower:
        idx = i
        lower = pt.face_dist[i]

    # face not valid, return previous face
    if lower > upper or idx < 0:
      idx = pidx
      break

    # compute support point w from the closest face's normal
    wi = pt.nvert
    face_normal = pt.face_normal[idx]
    i1, i2 = _epa_support(pt, wi, geom1, geom2, geomtype1, geomtype2, face_normal)
    w = pt.vert[2 * wi] - pt.vert[2 * wi + 1]
    geom1.index = i1
    geom2.index = i2
    pt.nvert += 1

    # upper bound for kth iteration
    upper = wp.min(upper, wp.dot(face_normal, w))

    if upper - lower < epsilon:
      break

    # check if vertex wi is a repeated support point
    if is_discrete:
      found_repeated = bool(False)
      for i in range(pt.nvert - 1):
        if pt.vert_index[2 * i] == pt.vert_index[2 * wi] and pt.vert_index[2 * i + 1] == pt.vert_index[2 * wi + 1]:
          found_repeated = True
          break
      if found_repeated:
        break

    nvalid -= 1
    pt.face[idx] = _delete_face(pt.face[idx])
    face = _get_face_verts(pt.face[idx])
    pt.nhorizon = _add_edge(pt, face[0], face[1])
    pt.nhorizon = _add_edge(pt, face[1], face[2])
    pt.nhorizon = _add_edge(pt, face[2], face[0])

    # delete only the visible faces connected to the closest one, so the deleted region stays one
    # piece and the horizon one loop; a face listed before the face connecting it waits a pass
    revisit = bool(True)
    while revisit and pt.nhorizon >= 0:
      revisit = bool(False)
      skipped = bool(False)
      for i in range(pt.nface):
        if _is_face_deleted(pt.face[i]):
          continue

        if wp.dot(pt.face_normal[i], w) > pt.face_dist[i]:
          face = _get_face_verts(pt.face[i])
          if _touches_horizon(pt, face):
            nvalid = wp.where(_is_invalid_face(pt.face[i]), nvalid, nvalid - 1)
            pt.face[i] = _delete_face(pt.face[i])
            pt.nhorizon = _add_edge(pt, face[0], face[1])
            pt.nhorizon = _add_edge(pt, face[1], face[2])
            pt.nhorizon = _add_edge(pt, face[2], face[0])
            revisit = skipped
          else:
            skipped = bool(True)

    # the polytope cannot grow past this point, so the closest face so far is the answer
    if pt.nhorizon == -1:
      if warn_overflow:
        wp.printf(
          "Warning: EPA horizon = %d isn't large enough.\n"
          "To disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.EPA_HORIZON (or = 0 for all)\n",
          pt.horizon.shape[0],
        )
      wp.atomic_or(overflow_out, worldid, OverflowType.EPA_HORIZON)
      break

    # insert w as new vertex and attach faces along the horizon
    attached = bool(True)
    for i in range(pt.nhorizon):
      edge = _get_edge(pt.horizon[i])
      attached = _attach_face(pt, pt.nface, wi, edge[0], edge[1])
      if not attached:
        break

      pt.nface += 1

      # a new face closer than the lower bound only exists by rounding, when the origin is on the
      # boundary, and then it is the answer, so only faces beyond the upper bound are pruned
      if pt.face_dist[pt.nface - 1] <= upper:
        nvalid += 1
      else:
        pt.face[pt.nface - 1] = _invalidate_face(pt.face[pt.nface - 1])

    # no face candidates left, or a horizon edge is collinear with w or the face buffer is full
    if nvalid == 0 or not attached:
      break

    # clear horizon
    pt.nhorizon = 0

  # return from valid face
  if idx > -1:
    x1, x2, dist = _epa_witness(pt, geom1, geom2, geomtype1, geomtype2, idx)
    return dist, x1, x2, idx
  return 0.0, wp.vec3(), wp.vec3(), -1


@wp.func
def epa_phase(
  # In:
  tolerance: float,
  epa_iterations: int,
  result: GJKResult,
  geom1: Geom,
  geom2: Geom,
  geomtype1: int,
  geomtype2: int,
  vert: wp.array[wp.vec3],
  vert_index: wp.array[int],
  face: wp.array[int],
  face_normal: wp.array[wp.vec3],
  face_dist: wp.array[float],
  horizon: wp.array[int],
  warn_overflow: bool,
  worldid: int,
  # Data out:
  overflow_out: wp.array[int],
) -> Tuple[float, int, wp.vec3, wp.vec3, int]:
  """Run EPA given GJK result. Returns (dist, ncontact, x1, x2, multiccd_idx)."""
  pt = Polytope()
  pt.nface = 0
  pt.nvert = 0
  pt.nhorizon = 0
  pt.vert = vert
  pt.vert_index = vert_index
  pt.face = face
  pt.face_normal = face_normal
  pt.face_dist = face_dist
  pt.horizon = horizon

  if result.dim == 2:
    new_result = _polytope2(
      pt,
      result.simplex,
      result.simplex1,
      result.simplex2,
      result.simplex_index1,
      result.simplex_index2,
      geom1,
      geom2,
      geomtype1,
      geomtype2,
    )
    result.simplex = new_result.simplex
    result.simplex1 = new_result.simplex1
    result.simplex2 = new_result.simplex2
    result.simplex_index1 = new_result.simplex_index1
    result.simplex_index2 = new_result.simplex_index2
    result.dim = 3
  elif result.dim == 4:
    pt, new_result = _polytope4(
      pt,
      result.simplex,
      result.simplex1,
      result.simplex2,
      result.simplex_index1,
      result.simplex_index2,
    )
    if pt.status == -1:
      result.simplex = new_result.simplex
      result.simplex1 = new_result.simplex1
      result.simplex2 = new_result.simplex2
      result.simplex_index1 = new_result.simplex_index1
      result.simplex_index2 = new_result.simplex_index2
      result.dim = 3

  # polytope2 always continues here, and polytope4 may fall back here
  if result.dim == 3:
    pt = _polytope3(
      pt,
      result.dist,
      result.simplex,
      result.simplex1,
      result.simplex2,
      result.simplex_index1,
      result.simplex_index2,
      geom1,
      geom2,
      geomtype1,
      geomtype2,
    )

  # no starting polytope with volume (e.g. flat support), so there is no depth to recover
  if pt.status:
    return result.dist, 1, result.x1, result.x2, -1

  is_discrete = _discrete_geoms(geomtype1, geomtype2) and (geom1.margin == 0.0 and geom2.margin == 0.0)
  dist, x1, x2, idx = _epa(
    tolerance, epa_iterations, pt, geom1, geom2, geomtype1, geomtype2, is_discrete, warn_overflow, worldid, overflow_out
  )
  if idx == -1:
    return FLOAT_MAX, 0, wp.vec3(), wp.vec3(), -1

  if not _supports_multiccd(geom1, geom2, geomtype1, geomtype2):
    idx = -1

  return dist, 1, x1, x2, idx


@wp.func
def ccd(
  # In:
  tolerance: float,
  cutoff: float,
  gjk_iterations: int,
  epa_iterations: int,
  geom1: Geom,
  geom2: Geom,
  geomtype1: int,
  geomtype2: int,
  x_1: wp.vec3,
  x_2: wp.vec3,
  vert: wp.array[wp.vec3],
  vert_index: wp.array[int],
  face: wp.array[int],
  face_normal: wp.array[wp.vec3],
  face_dist: wp.array[float],
  horizon: wp.array[int],
  warn_overflow: bool,
  worldid: int,
  # Data out:
  overflow_out: wp.array[int],
) -> Tuple[float, int, wp.vec3, wp.vec3, int]:
  """General convex collision detection via GJK/EPA."""
  needs_epa, dist, ncontact, x1, x2, result, geom1, geom2 = gjk_phase(
    tolerance, cutoff, gjk_iterations, geom1, geom2, geomtype1, geomtype2, x_1, x_2
  )
  if not needs_epa:
    idx = -1
    if dist < 0.0 and result.dim > 0 and _supports_multiccd(geom1, geom2, geomtype1, geomtype2):
      idx = _gjk_simplex_to_epa(result, vert, vert_index, face)
    return dist, ncontact, x1, x2, idx
  return epa_phase(
    tolerance,
    epa_iterations,
    result,
    geom1,
    geom2,
    geomtype1,
    geomtype2,
    vert,
    vert_index,
    face,
    face_normal,
    face_dist,
    horizon,
    warn_overflow,
    worldid,
    overflow_out,
  )
