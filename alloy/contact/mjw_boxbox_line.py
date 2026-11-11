# SPDX-License-Identifier: Apache-2.0
"""Line contacts for box-box pairs on MuJoCo Warp's primitive collider (BUL-0014, coordinator ruling 2026-11-04 17:26Z).

MuJoCo's ``mjc_BoxBox`` (and MuJoCo Warp's port ``box_box``) picks the separating axis of least overlap. When an
edge-cross axis wins, it emits one closest-point contact between the two edges. For a box lying across another box's
edge (the sheet's top edge under T12's block_3, BUL-0014), the winning edge axis is nearly a face normal and the edge
lies along that face. The single point sits at one end of the overlap, the box pivots about it and the edge cuts in
(upstream package 2: 1 contact at x = 340.8 mm while the edge is inside the box from 345 to 415 mm).

``box_box_line_wrapper`` keeps MuJoCo Warp's output except when ``box_box`` returns exactly one contact and the two
boxes' support features along its normal are an edge and a face, or two parallel edges. Then it emits the overlap
segment's two endpoints instead:
- edge on a face: the edge clipped against the face rectangle and the face's plane band (distance <= margin + gap);
  normal = the face normal; distance = each endpoint's height over the face plane; position halfway to the plane;
- parallel edges: the overlap of the two edges along their direction, the band clipped the same way; normal =
  ``box_box``'s; distance = each endpoint pair's separation along it.
A feature is read from the box axes nearly perpendicular to the normal (|n . axis| < ``FEATURE_SIN``): two = a face,
one = an edge, none = a vertex. A vertex, crossing edges or a segment shorter than ``SEGMENT_MIN`` of the boxes'
scale keep MuJoCo's single contact. Face-face cases are MuJoCo's (``box_box`` already clips the incident face).

The new contacts go through MuJoCo Warp's ``write_contact``: they count against ``naconmax`` and overflow as usual.

Scope (coordinator ruling 2026-11-04 20:08Z, option 3): on by default for every MuJoCo entry, per solver.
``enable_line_boxbox(solver)`` puts the wrapper in MuJoCo Warp's primitive collision table for (BOX, BOX) around each
of that solver's steps (graph capture included) and puts MuJoCo's entry back after, so other solvers in the process
are unaffected. It needs the primitive collider (``NATIVECCD`` disabled, ``alloy.contact.mjw_boxbox``). An entry
built from any object in ``LINE_OPT_OUT`` keeps MuJoCo's single edge contact (``line_default``). The list is empty:
DC3's QSFP latch, its only entries (opted out at the 20:08Z ruling), passes every row on line contacts (DC3's A/B,
main 52b9f01a). An entry kwarg
``{LINE_KEY: bool}`` overrides (the A8 squeeze twins set True). ``ALLOY_BOXBOX_LINE=0`` is a debug override only
(MuJoCo's routine everywhere); no entry, profile or row may depend on it. ``install_line_boxbox()`` sets it
process-wide (scripts). MuJoCo's source is not changed; the MuJoCo C path keeps ``mjc_BoxBox``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import warp as wp
from alloy._vendor.mujoco_warp._src.collision_core import Geom, write_contact
from alloy._vendor.mujoco_warp._src.collision_primitive_core import box_box
from alloy._vendor.mujoco_warp._src.math import make_frame
from alloy._vendor.mujoco_warp._src.types import vec5

log = logging.getLogger("alloy.contact")

FEATURE_SIN = 0.05     # |n . axis| below this: the axis lies in the contact plane (about 2.9 deg)
SEGMENT_MIN = 1.0e-4   # shortest segment that becomes two contacts, as a fraction of the two boxes' summed half sizes
_FEATURE_SIN = wp.constant(FEATURE_SIN)
_SEGMENT_MIN = wp.constant(SEGMENT_MIN)
_EMPTY = wp.constant(wp.vec2(1.0, -1.0))
ENV = "ALLOY_BOXBOX_LINE"


@wp.func
def _axis(rot: wp.mat33, k: int) -> wp.vec3:
    return wp.vec3(rot[0, k], rot[1, k], rot[2, k])


@wp.func
def _feature(rot: wp.mat33, d: wp.vec3) -> wp.vec2i:
    """Support feature of a box in direction ``d``: (kind, axis), kind 0 vertex, 1 edge along ``axis``, 2 face with
    normal ``axis``."""
    n_perp = int(0)
    perp = int(-1)
    other = int(-1)
    for k in range(3):
        if wp.abs(wp.dot(_axis(rot, k), d)) < _FEATURE_SIN:
            n_perp += 1
            perp = k
        else:
            other = k
    if n_perp == 1:
        return wp.vec2i(1, perp)
    if n_perp == 2:
        return wp.vec2i(2, other)
    return wp.vec2i(0, -1)


@wp.func
def _support_edge(pos: wp.vec3, rot: wp.mat33, size: wp.vec3, d: wp.vec3, a: int) -> wp.vec3:
    """Centre of the box's edge along axis ``a`` that is furthest in direction ``d``."""
    base = pos
    for j in range(3):
        if j != a:
            r = _axis(rot, j)
            s = wp.where(wp.dot(r, d) >= 0.0, 1.0, -1.0)
            base = base + s * size[j] * r
    return base


@wp.func
def _clip(span: wp.vec2, alpha: float, beta: float, bound: float) -> wp.vec2:
    """Restrict the parameter interval ``span`` to alpha + beta t <= bound (an empty span has x > y)."""
    if wp.abs(beta) < 1.0e-12:
        if alpha > bound:
            return _EMPTY
        return span
    t = (bound - alpha) / beta
    if beta > 0.0:
        return wp.vec2(span[0], wp.min(span[1], t))
    return wp.vec2(wp.max(span[0], t), span[1])


@wp.func
def _edge_on_face(
    face_pos: wp.vec3, face_rot: wp.mat33, face_size: wp.vec3, fa: int, toward_edge: wp.vec3,
    edge_pos: wp.vec3, edge_rot: wp.mat33, edge_size: wp.vec3, ea: int, margin: float,
):
    """The edge (axis ``ea`` of the edge box, the one nearest the face box) clipped against the face (normal axis
    ``fa`` of the face box, facing the edge box) and the band height <= margin. Returns (span, base, e, f, cf):
    endpoints are base + t e for t in span; f the outward face normal, cf the face centre."""
    r = _axis(face_rot, fa)
    f = wp.where(wp.dot(r, toward_edge) >= 0.0, r, -r)
    cf = face_pos + face_size[fa] * f
    u = _axis(face_rot, (fa + 1) % 3)
    v = _axis(face_rot, (fa + 2) % 3)
    hu = face_size[(fa + 1) % 3]
    hv = face_size[(fa + 2) % 3]
    base = _support_edge(edge_pos, edge_rot, edge_size, -toward_edge, ea)
    e = _axis(edge_rot, ea)
    w = base - cf
    span = wp.vec2(-edge_size[ea], edge_size[ea])
    span = _clip(span, wp.dot(w, u), wp.dot(e, u), hu)
    span = _clip(span, -wp.dot(w, u), -wp.dot(e, u), hu)
    span = _clip(span, wp.dot(w, v), wp.dot(e, v), hv)
    span = _clip(span, -wp.dot(w, v), -wp.dot(e, v), hv)
    span = _clip(span, wp.dot(w, f), wp.dot(e, f), margin)
    return span, base, e, f, cf


@wp.func
def box_box_line(
    box1_pos: wp.vec3, box1_rot: wp.mat33, box1_size: wp.vec3,
    box2_pos: wp.vec3, box2_rot: wp.mat33, box2_size: wp.vec3,
    normal: wp.vec3, margin: float,
):
    """Two contacts spanning an edge-on-face or parallel-edge overlap along ``normal`` (box1 -> box2), or count 0
    when the features are anything else. Returns (count, dist0, dist1, pos0, pos1, normal)."""
    zero = wp.vec3(0.0)
    f1 = _feature(box1_rot, normal)
    f2 = _feature(box2_rot, -normal)
    seg_min = _SEGMENT_MIN * (box1_size[0] + box1_size[1] + box1_size[2] + box2_size[0] + box2_size[1] + box2_size[2])

    if (f1[0] == 2 and f2[0] == 1) or (f1[0] == 1 and f2[0] == 2):
        span = wp.vec2(0.0)
        base = zero
        e = zero
        f = zero
        cf = zero
        sign = float(1.0)
        if f1[0] == 2:  # face of box1, edge of box2: the face normal points box1 -> box2
            span, base, e, f, cf = _edge_on_face(box1_pos, box1_rot, box1_size, f1[1], normal,
                                                 box2_pos, box2_rot, box2_size, f2[1], margin)
        else:  # face of box2, edge of box1: the face normal points box2 -> box1
            span, base, e, f, cf = _edge_on_face(box2_pos, box2_rot, box2_size, f2[1], -normal,
                                                 box1_pos, box1_rot, box1_size, f1[1], margin)
            sign = -1.0
        if span[1] - span[0] <= seg_min:
            return 0, 0.0, 0.0, zero, zero, zero
        q0 = base + span[0] * e
        q1 = base + span[1] * e
        d0 = wp.dot(q0 - cf, f)
        d1 = wp.dot(q1 - cf, f)
        return 2, d0, d1, q0 - 0.5 * d0 * f, q1 - 0.5 * d1 * f, sign * f

    if f1[0] == 1 and f2[0] == 1:
        e = _axis(box1_rot, f1[1])
        e2 = _axis(box2_rot, f2[1])
        if wp.abs(wp.dot(e, e2)) < wp.sqrt(1.0 - _FEATURE_SIN * _FEATURE_SIN):
            return 0, 0.0, 0.0, zero, zero, zero  # crossing edges: MuJoCo's closest point is the contact
        e2 = wp.where(wp.dot(e, e2) >= 0.0, e2, -e2)
        b1 = _support_edge(box1_pos, box1_rot, box1_size, normal, f1[1])
        b2 = _support_edge(box2_pos, box2_rot, box2_size, -normal, f2[1])
        c1 = wp.dot(b1, e)
        c2 = wp.dot(b2, e)
        # parameter w = position along e; the overlap of the two edges' intervals
        span = wp.vec2(wp.max(c1 - box1_size[f1[1]], c2 - box2_size[f2[1]]),
                       wp.min(c1 + box1_size[f1[1]], c2 + box2_size[f2[1]]))
        # separation along the normal at w: (b2 + (w - c2) e2 - b1 - (w - c1) e) . n, linear in w
        s0 = wp.dot(b2 - c2 * e2 - b1 + c1 * e, normal)
        s1 = wp.dot(e2 - e, normal)
        span = _clip(span, s0, s1, margin)
        if span[1] - span[0] <= seg_min:
            return 0, 0.0, 0.0, zero, zero, zero
        p0a = b1 + (span[0] - c1) * e
        p0b = b2 + (span[0] - c2) * e2
        p1a = b1 + (span[1] - c1) * e
        p1b = b2 + (span[1] - c2) * e2
        return 2, wp.dot(p0b - p0a, normal), wp.dot(p1b - p1a, normal), 0.5 * (p0a + p0b), 0.5 * (p1a + p1b), normal

    return 0, 0.0, 0.0, zero, zero, zero


@wp.func
def box_box_line_wrapper(
    # Data in:
    naconmax_in: int,
    # In:
    box1: Geom,
    box2: Geom,
    worldid: int,
    margin: float,
    gap: float,
    condim: int,
    friction: vec5,
    solref: wp.vec2,
    solreffriction: wp.vec2,
    solimp: vec5,
    adhesion: float,
    geoms: wp.vec2i,
    pairid: wp.vec2i,
    # Data out:
    contact_dist_out: wp.array[float],
    contact_pos_out: wp.array[wp.vec3],
    contact_frame_out: wp.array[wp.mat33],
    contact_includemargin_out: wp.array[float],
    contact_friction_out: wp.array[vec5],
    contact_solref_out: wp.array[wp.vec2],
    contact_solreffriction_out: wp.array[wp.vec2],
    contact_solimp_out: wp.array[vec5],
    contact_dim_out: wp.array[int],
    contact_geom_out: wp.array[wp.vec2i],
    contact_efc_address_out: wp.array2d[int],
    contact_worldid_out: wp.array[int],
    contact_type_out: wp.array[int],
    contact_geomcollisionid_out: wp.array[int],
    contact_adhesion_out: wp.array[float],
    nacon_out: wp.array[int],
):
    """MuJoCo Warp's ``box_box_wrapper`` with ``box_box_line`` replacing a single edge contact by the overlap
    segment's endpoints (same arguments, same writer)."""
    ncon, dist, pos, normal = box_box(box1.pos, box1.rot, box1.size, box2.pos, box2.rot, box2.size, margin + gap)
    if ncon == 0:
        return
    if ncon == 1:
        n_line, d0, d1, p0, p1, n = box_box_line(box1.pos, box1.rot, box1.size, box2.pos, box2.rot, box2.size,
                                                 normal, margin + gap)
        if n_line == 2:
            ncon = 2
            dist[0] = d0
            dist[1] = d1
            pos[0] = p0
            pos[1] = p1
            normal = n
    frame = make_frame(normal)
    for i in range(ncon):
        write_contact(
            naconmax_in, i, dist[i], pos[i], frame, margin, gap, condim, friction, solref, solreffriction, solimp,
            adhesion, geoms, pairid, worldid, contact_dist_out, contact_pos_out, contact_frame_out,
            contact_includemargin_out, contact_friction_out, contact_solref_out, contact_solreffriction_out,
            contact_solimp_out, contact_dim_out, contact_geom_out, contact_efc_address_out, contact_worldid_out,
            contact_type_out, contact_geomcollisionid_out, contact_adhesion_out, nacon_out,
        )


LINE_KEY = "alloy:boxbox_line"   # MuJoCo entry kwarg: True / False overrides line_default for that entry
LINE_DEFAULT = True                # coordinator ruling 2026-11-04 20:08Z (option 3)
# BUL-0014 opt-out (object card ids; entries built from any of them keep MuJoCo's single edge contact;
# tests/contact/test_mjw_boxbox_line.py keeps this list from spreading). Empty since DC3's QSFP A/B (52b9f01a).
LINE_OPT_OUT: tuple[str, ...] = ()
_ORIGINAL: dict[str, Any] = {}


def _table() -> tuple[dict, Any]:
    from alloy._vendor.mujoco_warp._src import collision_primitive as cp  # noqa: PLC0415
    from alloy._vendor.mujoco_warp._src.types import GeomType  # noqa: PLC0415

    return cp._PRIMITIVE_COLLISIONS, (GeomType.BOX, GeomType.BOX)


def model_objects(model: Any) -> frozenset[str]:
    """Object ids a Newton model was assembled from: the first component of its body labels ("<object>/worldbody/...")."""
    return frozenset(str(lab).split("/", 1)[0] for lab in getattr(model, "body_label", ()) or ())


def line_default(objects: Any) -> bool:
    """Line contacts for an entry built from ``objects`` (ids): LINE_DEFAULT unless one of them is in LINE_OPT_OUT."""
    return LINE_DEFAULT and not any(o in LINE_OPT_OUT for o in objects)


def line_boxbox_on() -> bool:
    """Whether MuJoCo Warp's primitive (BOX, BOX) entry is ``box_box_line_wrapper``."""
    table, key = _table()
    return table.get(key) is box_box_line_wrapper


def install_line_boxbox() -> bool:
    """Put ``box_box_line_wrapper`` in MuJoCo Warp's primitive collision table (process-wide, idempotent; kernels
    built after this use it). False when ``ALLOY_BOXBOX_LINE=0`` keeps MuJoCo's routine."""
    if os.environ.get(ENV, "1") == "0":
        return False
    table, key = _table()
    if table.get(key) is not box_box_line_wrapper:
        _ORIGINAL["box_box"] = _ORIGINAL.get("box_box") or table.get(key)
        table[key] = box_box_line_wrapper
        log.info("mjw_boxbox_line: box-box edge-on-face / parallel-edge contacts emit the overlap segment (BUL-0014)")
    return True


def enable_line_boxbox(solver: Any, reduce4: bool = False) -> bool:
    """Line contacts on this solver's box-box pairs: its MuJoCo Warp step runs with ``box_box_line_wrapper`` in the
    primitive table, then MuJoCo's entry is put back. ``reduce4`` (the fast rigid mode's ``REDUCE_KEY``) puts
    ``mjw_boxbox_reduce.box_box_reduce_wrapper`` there instead: the same line contacts, face pairs cut to 4 points.
    Call before the first step or graph capture. False when ``ALLOY_BOXBOX_LINE=0`` or the solver has no MuJoCo Warp
    step."""
    if os.environ.get(ENV, "1") == "0" or getattr(solver, "use_mujoco_cpu", False):
        return False
    orig = getattr(solver, "_mujoco_warp_step", None)
    if orig is None:
        return False
    if getattr(solver, "_alloy_boxbox_line", False):
        if bool(getattr(solver, "_alloy_boxbox_reduce4", False)) != bool(reduce4):   # WP0 / WP13 review
            raise RuntimeError(f"box-box contacts on this solver are already installed with reduce4="
                               f"{not reduce4}; a second call cannot switch them to reduce4={bool(reduce4)}")
        return True
    from alloy.contact.mjw_boxbox_reduce import box_box_reduce_wrapper  # noqa: PLC0415 - imports this module

    wrapper = box_box_reduce_wrapper if reduce4 else box_box_line_wrapper
    table, key = _table()
    ours = (box_box_line_wrapper, box_box_reduce_wrapper)
    _ORIGINAL.setdefault("box_box", table.get(key) if table.get(key) not in ours else None)

    def step(*args: Any, **kwargs: Any) -> Any:
        prev = table.get(key)
        table[key] = wrapper
        try:
            return orig(*args, **kwargs)
        finally:
            table[key] = prev

    solver._mujoco_warp_step = step
    solver._alloy_boxbox_line = True
    solver._alloy_boxbox_reduce4 = bool(reduce4)
    log.info("mjw_boxbox_line: box-box edge-on-face / parallel-edge contacts emit the overlap segment on this solver%s",
             "; face pairs reduced to 4 points (fast rigid mode)" if reduce4 else "")
    return True


def restore_mujoco_boxbox() -> None:
    """MuJoCo Warp's own (BOX, BOX) entry back (tests, A/B runs)."""
    table, key = _table()
    if _ORIGINAL.get("box_box") is not None:
        table[key] = _ORIGINAL["box_box"]


__all__ = ["FEATURE_SIN", "LINE_DEFAULT", "LINE_KEY", "LINE_OPT_OUT", "SEGMENT_MIN", "line_default", "model_objects", "box_box_line", "box_box_line_wrapper",
           "enable_line_boxbox", "install_line_boxbox", "line_boxbox_on", "restore_mujoco_boxbox"]
