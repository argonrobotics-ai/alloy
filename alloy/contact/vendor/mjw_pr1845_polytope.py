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
# from the pinned collision_gjk. This file: the polytope.
# Installed by alloy.contact.mjw_epa; drop it once the MuJoCo Warp pin includes the fix.
# License: LICENSE-mujoco_warp (Apache-2.0).
# ==============================================================================

from typing import Tuple

import warp as wp

from alloy._vendor.mujoco_warp._src.collision_core import Geom
from alloy._vendor.mujoco_warp._src.types import mat43

from alloy._vendor.mujoco_warp._src.collision_gjk import FLOAT_MAX
from alloy._vendor.mujoco_warp._src.collision_gjk import GJKResult
from alloy._vendor.mujoco_warp._src.collision_gjk import MINVAL
from alloy._vendor.mujoco_warp._src.collision_gjk import _det3
from alloy._vendor.mujoco_warp._src.collision_gjk import _test_tetra
from alloy._vendor.mujoco_warp._src.collision_gjk import _tri_point_intersect
from alloy._vendor.mujoco_warp._src.collision_gjk import support

# TODO(team): improve compile time to enable backward pass
wp.set_module_options({"enable_backward": False})


@wp.struct
class Polytope:
  status: int

  # vertices in polytope (packed geom1 followed by geom2)
  vert: wp.array[wp.vec3]
  vert_index: wp.array[int]
  nvert: int

  # +1 if the faces are wound counterclockwise seen from outside, -1 if clockwise
  orient: float

  # faces in polytope
  # 10 bits per each vertex index, while the last significant bits are for
  # invalid and deleted face
  face: wp.array[int]
  face_normal: wp.array[wp.vec3]
  face_dist: wp.array[float]
  nface: int

  # edges that make up the horizon when adding new vertices to polytope
  horizon: wp.array[int]
  nhorizon: int


@wp.func
def _attach_face(pt: Polytope, idx: int, v1: int, v2: int, v3: int) -> bool:
  """Attaches a face by its outward plane, which does not depend on where the origin is.

  The face is wound like the polytope, so its outward side follows from its winding. Returns False
  if the face has no area or the polytope is out of memory.
  """
  if pt.nface == pt.face.shape[0]:
    return False

  p1 = pt.vert[2 * v1] - pt.vert[2 * v1 + 1]
  p2 = pt.vert[2 * v2] - pt.vert[2 * v2 + 1]
  p3 = pt.vert[2 * v3] - pt.vert[2 * v3 + 1]
  n = wp.cross(p2 - p1, p3 - p1)
  norm = wp.norm_l2(n)
  if norm < MINVAL:
    return False
  n = pt.orient * n / norm

  pt.face[idx] = v1 + (v2 << 10) + (v3 << 20)
  pt.face_normal[idx] = n
  pt.face_dist[idx] = wp.dot(n, p1)
  return True


@wp.func
def _epa_support(
  pt: Polytope, idx: int, geom1: Geom, geom2: Geom, geom1_type: int, geom2_type: int, dir: wp.vec3
) -> Tuple[int, int]:
  sp = support(geom1, geom1_type, dir)
  pt.vert[2 * idx] = sp.point
  pt.vert_index[2 * idx] = sp.vertex_index
  index1 = sp.cached_index

  sp = support(geom2, geom2_type, -dir)
  pt.vert[2 * idx + 1] = sp.point
  pt.vert_index[2 * idx + 1] = sp.vertex_index
  index2 = sp.cached_index

  return index1, index2


@wp.func
def _replace_simplex3(pt: Polytope, v1: int, v2: int, v3: int) -> GJKResult:
  result = GJKResult()

  # reset GJK simplex
  simplex1 = mat43()
  simplex1[0] = pt.vert[2 * v1]
  simplex1[1] = pt.vert[2 * v2]
  simplex1[2] = pt.vert[2 * v3]

  simplex2 = mat43()
  simplex2[0] = pt.vert[2 * v1 + 1]
  simplex2[1] = pt.vert[2 * v2 + 1]
  simplex2[2] = pt.vert[2 * v3 + 1]

  simplex = mat43()
  simplex[0] = simplex1[0] - simplex2[0]
  simplex[1] = simplex1[1] - simplex2[1]
  simplex[2] = simplex1[2] - simplex2[2]

  simplex_index1 = wp.vec4i()
  simplex_index1[0] = pt.vert_index[2 * v1]
  simplex_index1[1] = pt.vert_index[2 * v2]
  simplex_index1[2] = pt.vert_index[2 * v3]

  simplex_index2 = wp.vec4i()
  simplex_index2[0] = pt.vert_index[2 * v1 + 1]
  simplex_index2[1] = pt.vert_index[2 * v2 + 1]
  simplex_index2[2] = pt.vert_index[2 * v3 + 1]

  result.simplex = simplex
  result.simplex1 = simplex1
  result.simplex2 = simplex2
  result.simplex_index1 = simplex_index1
  result.simplex_index2 = simplex_index2

  return result


@wp.func
def _polytope2(
  # In:
  pt: Polytope,
  simplex: mat43,
  simplex1: mat43,
  simplex2: mat43,
  simplex_index1: wp.vec4i,
  simplex_index2: wp.vec4i,
  geom1: Geom,
  geom2: Geom,
  geomtype1: int,
  geomtype2: int,
) -> GJKResult:
  """Grow a 1-simplex from GJK into a 2-simplex for polytope3 with one support point."""
  diff = simplex[1] - simplex[0]

  # find component with smallest magnitude (so cross product is largest)
  value = FLOAT_MAX
  index = 0
  for i in range(3):
    if wp.abs(diff[i]) < value:
      value = wp.abs(diff[i])
      index = i

  # cross product with best coordinate axis
  e = wp.vec3(0.0, 0.0, 0.0)
  e[index] = 1.0
  d = wp.cross(e, diff)

  pt.vert[0] = simplex1[0]
  pt.vert[1] = simplex2[0]
  pt.vert[2] = simplex1[1]
  pt.vert[3] = simplex2[1]

  pt.vert_index[0] = simplex_index1[0]
  pt.vert_index[1] = simplex_index2[0]
  pt.vert_index[2] = simplex_index1[1]
  pt.vert_index[3] = simplex_index2[1]

  _epa_support(pt, 2, geom1, geom2, geomtype1, geomtype2, d / wp.norm_l2(d))
  return _replace_simplex3(pt, 0, 1, 2)


@wp.func
def _polytope3(
  # In:
  pt: Polytope,
  dist: float,
  simplex: mat43,
  simplex1: mat43,
  simplex2: mat43,
  simplex_index1: wp.vec4i,
  simplex_index2: wp.vec4i,
  geom1: Geom,
  geom2: Geom,
  geomtype1: int,
  geomtype2: int,
) -> Polytope:
  """Create polytope for EPA given a 2-simplex from GJK."""
  # get normals in both directions
  n = wp.cross(simplex[1] - simplex[0], simplex[2] - simplex[0])
  norm = wp.norm_l2(n)
  if norm < MINVAL:
    pt.status = 2
    return pt
  n = n / norm

  pt.vert[0] = simplex1[0]
  pt.vert[1] = simplex2[0]
  pt.vert[2] = simplex1[1]
  pt.vert[3] = simplex2[1]
  pt.vert[4] = simplex1[2]
  pt.vert[5] = simplex2[2]

  pt.vert_index[0] = simplex_index1[0]
  pt.vert_index[1] = simplex_index2[0]
  pt.vert_index[2] = simplex_index1[1]
  pt.vert_index[3] = simplex_index2[1]
  pt.vert_index[4] = simplex_index1[2]
  pt.vert_index[5] = simplex_index2[2]

  _epa_support(pt, 3, geom1, geom2, geomtype1, geomtype2, -n)
  _epa_support(pt, 4, geom1, geom2, geomtype1, geomtype2, n)

  v1 = simplex[0]
  v2 = simplex[1]
  v3 = simplex[2]
  v4 = pt.vert[6] - pt.vert[7]
  v5 = pt.vert[8] - pt.vert[9]

  # check that v4 is not contained in the 2-simplex
  if _tri_point_intersect(v1, v2, v3, v4):
    pt.status = 3
    return pt

  # check that v5 is not contained in the 2-simplex
  if _tri_point_intersect(v1, v2, v3, v5):
    pt.status = 4
    return pt

  # if origin does not lie on simplex then we need to check that the hexahedron contains the
  # origin
  if dist > 1e-5 and not _test_tetra(v1, v2, v3, v4) and not _test_tetra(v1, v2, v3, v5):
    pt.status = 5
    return pt

  # create hexahedron for EPA, whose faces are wound consistently, so the volume across the
  # 2-simplex decides which way they face
  pt.orient = wp.sign(_det3(v2 - v1, v3 - v1, v5 - v4))
  if not _attach_face(pt, 0, 4, 0, 1):
    pt.status = 6
    return pt
  if not _attach_face(pt, 1, 4, 2, 0):
    pt.status = 7
    return pt
  if not _attach_face(pt, 2, 4, 1, 2):
    pt.status = 8
    return pt
  if not _attach_face(pt, 3, 3, 1, 0):
    pt.status = 9
    return pt
  if not _attach_face(pt, 4, 3, 0, 2):
    pt.status = 10
    return pt
  if not _attach_face(pt, 5, 3, 2, 1):
    pt.status = 11
    return pt

  # set polytope counts
  pt.nvert = 5
  pt.nface = 6
  pt.status = 0
  return pt


@wp.func
def _polytope4(
  # In:
  pt: Polytope,
  simplex: mat43,
  simplex1: mat43,
  simplex2: mat43,
  simplex_index1: wp.vec4i,
  simplex_index2: wp.vec4i,
) -> Tuple[Polytope, GJKResult]:
  """Create polytope for EPA given a 3-simplex from GJK."""
  pt.vert[0] = simplex1[0]
  pt.vert[1] = simplex2[0]
  pt.vert[2] = simplex1[1]
  pt.vert[3] = simplex2[1]
  pt.vert[4] = simplex1[2]
  pt.vert[5] = simplex2[2]
  pt.vert[6] = simplex1[3]
  pt.vert[7] = simplex2[3]

  pt.vert_index[0] = simplex_index1[0]
  pt.vert_index[1] = simplex_index2[0]
  pt.vert_index[2] = simplex_index1[1]
  pt.vert_index[3] = simplex_index2[1]
  pt.vert_index[4] = simplex_index1[2]
  pt.vert_index[5] = simplex_index2[2]
  pt.vert_index[6] = simplex_index1[3]
  pt.vert_index[7] = simplex_index2[3]

  # the faces are wound consistently, so the volume of the tetrahedron decides which way they face;
  # if a face is degenerate, replace the 3-simplex with a 2-simplex
  pt.orient = wp.sign(_det3(simplex[1] - simplex[0], simplex[2] - simplex[0], simplex[0] - simplex[3]))
  if not _attach_face(pt, 0, 0, 1, 2):
    pt.status = -1
    return pt, _replace_simplex3(pt, 0, 1, 2)

  if not _attach_face(pt, 1, 0, 3, 1):
    pt.status = -1
    return pt, _replace_simplex3(pt, 0, 3, 1)

  if not _attach_face(pt, 2, 0, 2, 3):
    pt.status = -1
    return pt, _replace_simplex3(pt, 0, 2, 3)

  if not _attach_face(pt, 3, 3, 2, 1):
    pt.status = -1
    return pt, _replace_simplex3(pt, 3, 2, 1)

  # set polytope counts
  pt.nvert = 4
  pt.nface = 4
  pt.status = 0
  return pt, GJKResult()
