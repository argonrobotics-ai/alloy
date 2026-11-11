# SPDX-License-Identifier: Apache-2.0
"""Host-side builders: light tree (scene/light_tree.cpp measures, orientation-bound merge) and LightScene assembly."""
import numpy as np
import warp as wp

from .common import KIND_AREA, KIND_BACKGROUND, KIND_DISTANT
from .tree import LightScene, TreeEmitter, TreeNode

LEAF_INNER, LEAF_LEAF, LEAF_DISTANT = 0, 1, 2
_FLT_MAX = 3.4028235e38


def _v(x):
    return np.array([x[0], x[1], x[2]], dtype=np.float64)


class _Measure:
    """LightTreeMeasure: bbox, orientation bounds (axis, theta_o, theta_e), energy."""

    def __init__(self, bmin=None, bmax=None, axis=None, theta_o=0.0, theta_e=0.0, energy=0.0):
        self.bmin = np.full(3, np.inf) if bmin is None else bmin
        self.bmax = np.full(3, -np.inf) if bmax is None else bmax
        self.axis = np.array([0, 0, 1.0]) if axis is None else axis
        self.theta_o, self.theta_e, self.energy = theta_o, theta_e, energy

    def add(self, o):
        r = _Measure(np.minimum(self.bmin, o.bmin), np.maximum(self.bmax, o.bmax), energy=self.energy + o.energy)
        r.axis, r.theta_o, r.theta_e = _merge_cones(self, o)
        return r


def _merge_cones(ca, cb):
    """scene/light_tree.cpp merge(OrientationBounds, OrientationBounds)."""
    a, b = (cb, ca) if cb.theta_o > ca.theta_o else (ca, cb)
    cos_ab = float(np.dot(a.axis, b.axis))
    theta_d = float(np.arccos(np.clip(cos_ab, -1, 1)))
    theta_e = max(a.theta_e, b.theta_e)
    if a.theta_o + 5e-4 >= min(np.pi, theta_d + b.theta_o):
        return a.axis, a.theta_o, theta_e
    theta_o = (theta_d + a.theta_o + b.theta_o) * 0.5
    if theta_o >= np.pi:
        return a.axis, np.pi, theta_e
    if cos_ab < -0.9995:
        t = np.cross(a.axis, [1.0, 0, 0] if abs(a.axis[0]) < 0.9 else [0, 1.0, 0])
        return t / np.linalg.norm(t), theta_o, theta_e
    theta_r = theta_o - a.theta_o
    ortho = b.axis - a.axis * cos_ab
    ortho = ortho / max(np.linalg.norm(ortho), 1e-30)
    return a.axis * np.cos(theta_r) + ortho * np.sin(theta_r), theta_o, theta_e


def _area_measure(a):
    co, u, v, d = _v(a.co), _v(a.axis_u), _v(a.axis_v), _v(a.dir)
    hu, hv = 0.5 * a.len_u * u, 0.5 * a.len_v * v
    corners = [co + hu + hv, co + hu - hv, co - hu + hv, co - hu - hv]
    half_spread = np.pi / 2 if a.tan_half_spread >= 1e30 else float(np.arctan(a.tan_half_spread))
    area = a.len_u * a.len_v
    e = float(np.mean(np.abs(_v(a.strength)))) / np.pi * (1.0 if a.normalized else area)
    return _Measure(np.min(corners, 0), np.max(corners, 0), d, 0.0, half_spread, e)


def _bg_measure(world):
    return _Measure(axis=np.array([0, 0, 1.0]), theta_o=np.pi, theta_e=0.0, energy=1.0 * world.average_radiance * np.pi)


def _sun_measure(s):
    return _Measure(axis=_v(s.co), theta_o=0.0, theta_e=s.angle, energy=float(np.mean(np.abs(_v(s.strength)))))


def _node(m, type_, left=0, right=0, first=0, num=0):
    n = TreeNode()
    n.bbox_min = wp.vec3(*(m.bmin if np.all(np.isfinite(m.bmin)) else np.zeros(3)))
    n.bbox_max = wp.vec3(*(m.bmax if np.all(np.isfinite(m.bmax)) else np.zeros(3)))
    n.axis = wp.vec3(*m.axis)
    n.theta_o, n.theta_e, n.energy = float(m.theta_o), float(m.theta_e), float(m.energy)
    n.type, n.left, n.right, n.first_emitter, n.num_emitters = type_, left, right, first, num
    return n


def _build_local(meas, nodes, trails, idx, depth, trail):
    """Recursive median split of the area lights idx (stable sort along the widest centroid axis)."""
    ms = meas[idx[0]]
    for i in idx[1:]:
        ms = ms.add(meas[i])
    me = len(nodes)
    nodes.append(None)
    if len(idx) == 1:
        trails[idx[0]] = trail
        nodes[me] = _node(ms, LEAF_LEAF, first=idx[0], num=1)
        return me, ms
    cent = np.array([0.5 * (meas[i].bmin + meas[i].bmax) for i in idx])
    dim = int(np.argmax(cent.max(0) - cent.min(0)))
    order = [idx[j] for j in np.argsort(cent[:, dim], kind="stable")]
    h = len(order) // 2
    li, lm = _build_local(meas, nodes, trails, order[:h], depth + 1, trail)
    ri, rm = _build_local(meas, nodes, trails, order[h:], depth + 1, trail | (1 << depth))
    nodes[me] = _node(ms, LEAF_INNER, left=li, right=ri, num=-1)  # inner: num_emitters = -1 (scene/light.cpp:554)
    return me, ms


def _make_emitters(all_meas, k, has_world, trails):
    """TreeEmitter list: area lights (first k), background (if the world exists), then suns."""
    ems = []
    for i, m in enumerate(all_meas):
        e = TreeEmitter()
        e.kind = KIND_AREA if i < k else (KIND_BACKGROUND if has_world and i == k else KIND_DISTANT)
        e.idx = i if i < k else (0 if e.kind == KIND_BACKGROUND else i - k - (1 if has_world else 0))
        e.energy, e.theta_o, e.theta_e, e.bit_trail = float(m.energy), float(m.theta_o), float(m.theta_e), trails[i]
        ems.append(e)
    return ems


def build_tree(areas, world, suns):
    """One world: returns (node list, emitter list) following scene/light_tree.cpp's top-level layout.
    Emitter order: area lights, background (if world), suns."""
    k = len(areas)
    meas = [_area_measure(a) for a in areas]
    dist_meas = ([_bg_measure(world)] if world is not None else []) + [_sun_measure(s) for s in suns]
    n_em = k + len(dist_meas)
    nodes, trails = [], [0] * n_em

    if k == 0 and not dist_meas:  # nothing to sample: one empty leaf
        return [_node(_Measure(), LEAF_DISTANT, first=0, num=0)], []
    root = len(nodes)
    nodes.append(None)
    root_meas = _Measure()
    if k > 0:
        li, lm = _build_local(meas, nodes, trails, list(range(k)), 1, 0)
        root_meas = lm
    dm = None
    for m in dist_meas:
        dm = m if dm is None else dm.add(m)
    if dm is None:
        dm = _Measure()
    ri = len(nodes)
    nodes.append(_node(dm, LEAF_DISTANT, first=k, num=len(dist_meas)))
    for i in range(k, n_em):
        trails[i] = 1
    if k == 0:
        nodes[root] = nodes.pop()      # distant-only: the leaf is the root
        nodes[root].first_emitter = 0
    else:
        nodes[root] = _node(root_meas.add(dm) if dist_meas else root_meas, LEAF_INNER, left=li, right=ri, num=-1)
    ems = _make_emitters(meas + dist_meas, k, world is not None, trails)
    return nodes, ems


def build_light_scene(areas, world=None, suns=None, device=None):
    """areas: list (per world) of lists of AreaLight; suns: same with DistantLight; world: WorldMap or None.
    Every world must have the same number of lights. Returns a LightScene (use with `w` = world index)."""
    nw = len(areas)
    suns = suns if suns is not None else [[] for _ in range(nw)]
    k, s = len(areas[0]), len(suns[0])
    assert all(len(a) == k for a in areas) and all(len(x) == s for x in suns)
    # a world with MIS off (Cycles' sampling method NONE) is no light: escaped rays see it at full weight
    light_world = world if world is not None and int(world.use_mis) != 0 else None
    trees = [build_tree(areas[w], light_world, suns[w]) for w in range(nw)]
    nn = max(len(t[0]) for t in trees)
    assert all(len(t[0]) == nn for t in trees), "worlds must share the tree topology"
    sc = LightScene()
    if k:
        sc.areas = struct_array([a for row in areas for a in row], _area_struct(), (nw, k), device)
    else:
        sc.areas = wp.zeros((nw, 0), dtype=_area_struct(), device=device)
    if s:
        sc.suns = struct_array([x for row in suns for x in row], _sun_struct(), (nw, s), device)
    else:
        sc.suns = wp.zeros((nw, 0), dtype=_sun_struct(), device=device)
    if world is not None:
        sc.world = world
        sc.has_world = 1
    else:
        sc.world = _dummy_world(device)
        sc.has_world = 0
    sc.world_light = int(light_world is not None)
    sc.n_areas, sc.n_suns = k, s
    sc.nodes = struct_array([n for t in trees for n in t[0]], TreeNode, (nw, nn), device)
    ne = len(trees[0][1])
    sc.emitters = (struct_array([e for t in trees for e in t[1]], TreeEmitter, (nw, ne), device) if ne else
                   wp.zeros((nw, 1), dtype=TreeEmitter, device=device))
    sc.n_emitters = len(trees[0][1])
    return sc


def _struct_of(x):
    return type(x)


def _area_struct():
    from .area import AreaLight
    return AreaLight


def _sun_struct():
    from .world import DistantLight
    return DistantLight


def _dummy_world(device):
    from .world import WorldMap
    w = WorldMap()
    w.tex = wp.zeros((1, 1), dtype=wp.vec3, device=device)
    w.cond = wp.zeros((1, 2), dtype=wp.vec2, device=device)
    w.marg = wp.zeros(2, dtype=wp.vec2, device=device)
    w.width = w.height = 1
    w.use_mis = 0
    w.w2t = wp.mat33(*np.eye(3).flatten())
    w.t2w = wp.mat33(*np.eye(3).flatten())
    return w


def struct_array(items, dtype, shape, device=None):
    """wp.array of struct instances (Warp cannot build these from Python lists): fill a zero array through its
    numpy structured view."""
    arr = wp.zeros(len(items), dtype=dtype, device="cpu")
    npa = arr.numpy()
    names = npa.dtype.names
    for i, it in enumerate(items):
        for nme in names:
            v = getattr(it, nme)
            npa[nme][i] = np.array(v) if not isinstance(v, (int, float)) else v
    return wp.array(npa, dtype=dtype, device=device).reshape(shape)
