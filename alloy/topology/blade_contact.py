# SPDX-License-Identifier: Apache-2.0
"""Side-aware blade contact for pre-split twins (SCR-SC5-1; WP10 owns it after review).

Pre-split puts the two copies of a seam vertex at one point: twin a on the -n side, twin b on the +n side, with n the
TET_CUT bond's seam normal (``topology:bond_normal``, from a's region to b's). Newton's particle-vs-shape contact gives
both copies the same point and normal, so a blade pushes the pair to one side of itself and never opens the seam.
``BladeContact.adjust_contacts`` runs after ``collide`` and before the solver step. It rewrites every soft contact
between an *active* twin and a shape on a blade body into a contact on the blade face on the twin's own side:

- n is the current seam normal ``normalize(F^-T n0)``, with F from one tet that holds twin a (Newton's
  ``tet_poses`` is ``Dm^-1``, ``Dm = [x1-x0, x2-x0, x3-x0]``, so ``F = Ds Dm^-1``);
- the face normal is the thinnest axis of the blade body's thinnest box, in the body frame (or body x, the
  ``blade_pose`` convention, for a blade without a box), and the half-thickness is that box's thinnest half-extent
  unless given explicitly;
- which face (SCR-JOB1-2, rulings 2026-11-04 22:03Z and 2026-11-05 02:00Z; engine item 9 step 1, 2026-11-06):
  - the face on the twin's current side of the blade's mid-plane, anywhere on the blade. Newton's closest point at
    the edge is the box's bottom corner, whose diagonal normal let a just-cut twin 0.73 mm into the blade;
  - the far face only while the blade is cutting the pair: the twin is within the cut-front cap 2h + r of the edge
    segment (h the half-thickness, r the larger twin radius), the pair is still less than 2h + r apart, and the edge
    is advancing into the twin (its velocity relative to the blade point there, along the spine from the edge into the
    blade, is >= 0, with the blade's ``body_qd``; a blade without a box has no spine and skips that test). So an opened
    pair is pushed, not pulled through the blade (JOB1's sweep), and an edge moving out of a round does not carry a
    joined twin across;
  - or, for a parted pair, at a re-entry (gap #9, 2026-11-10): the twin is within that cap, ahead of the edge (where
    no face stands yet), and the edge is coming down on it (over 1 mm/s along the spine and no slower than the blade
    moves across it: ``edge_closing``). An edge that re-enters an opened cut off its seam then wedges the twin to
    its own side by the ramp below, where the stock record was the blade's bottom pressing straight down on it.
    ``reentry_count`` / ``reentry_cross`` count these records per world, and ``BladeContactConfig.reentry=False``
    is the rule before;
- how deep (item 9 step 1): ahead of the edge along the spine, where the face does not reach yet, the record's
  penetration is capped at h + r + edge_radius - (the twin's distance to the edge segment). The twin ramps onto its
  face as the edge nears, instead of jumping h + r in one step when the cut model first damages its bond 1-1.6 mm
  ahead of the edge (1.1-1.3 m/s kicks; WP7's cut_block_stroke class: S1 1,371 -> 112 of 2048 episodes). Records
  with no allowed depth keep Newton's contact. At and behind the edge the face record is full;
- side s = side(twin) * sign(n . face normal), side(a) = -1, side(b) = +1; the contact point is the twin's position
  projected onto the plane at s * (half-thickness - r) (gap #9, WP15 2026-11-08: a seam twin is a point of the cut
  surface, so it rests on the face itself and the blade opens a kerf of its own thickness; with the plane at the face
  a 1 mm blade held its twins 3 mm apart, and the pinch's friction dragged the cut faces along a sawing stroke), the
  normal is s * face normal (world), the shape becomes the face box (its margin and friction), and the body velocity
  and rigid primitive ids are cleared (an analytic record).

Active twin rule (default, SCR-SC5-1 acceptance "intact twins keep the stock contact"): the bond is TET_CUT, its
seam normal is aligned with the face (``|n . face| >= min_alignment``), and it is broken or damaged (``intact == 0``
or ``damage > 0``). ``edge_gate=True`` also activates intact, undamaged twins whose midpoint is within
``edge_radius + reach + particle_radius`` of the blade edge segment. On SC5's repro both rules cut 98/98
(``docs/notes/wp10-blade-contact.md``). The edge gate does not depend on the cut damage model, so re-check it when
the clean-room cut damage lands (ruling item 5). Keep ``reach`` small: at 2 mm, intact twins ahead of the edge are
held in the blade's mid-plane by their spring, face friction drags them down, and the repro slice was crushed
(29/98).

Edge/face soft-contact records (more than one particle slot) are rewritten only when every non-negative slot is an
active twin and all slots resolve to the same side; the point is then the barycentric point. Otherwise they keep the
stock contact. A particle in several TET_CUT bonds (seam lines meeting) uses its bond best aligned with the face.
Blades act only on bonds of their own world (``body_world`` vs the bond's world; -1 is global).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.core.errors import AssemblyError
from alloy.topology.blade_contact_kernels import (SEAM_ALIGNMENT, BladeTable, TwinTable, rewrite_blade_contacts,
                                                  seam_normals_kernel)
from alloy.topology.bonds import bond_column, has_bonds
from alloy.topology.types import Blade, BondKind, BondSet

log = logging.getLogger("alloy.topology")


@dataclass(frozen=True)
class BladeContactConfig:
    """``edge_gate``: also activate intact, undamaged twins near the edge (off by default; see the module docstring).
    ``reach`` [m]: with the edge gate, the twin midpoint to edge segment distance, on top of edge radius + particle
    radius, under which such a pair becomes active (0 and 0.5 mm cut the repro; 2 mm crushed it). ``min_alignment``:
    skip when |seam normal . face normal| is lower."""

    reach: float = 0.0
    min_alignment: float = SEAM_ALIGNMENT
    edge_gate: bool = False           # True: intact twins within reach of the edge are active too
    reentry: bool = True              # False: the far-face rule before the re-entry case (pair distance only)

    def __post_init__(self) -> None:
        if not (self.reach >= 0.0 and np.isfinite(self.reach)):
            raise AssemblyError(f"BladeContactConfig.reach must be finite and >= 0, got {self.reach}")
        if not 0.0 <= self.min_alignment <= 1.0:
            raise AssemblyError(f"BladeContactConfig.min_alignment must be in [0, 1], got {self.min_alignment}")


@dataclass(frozen=True)
class BladeFace:
    """One blade's face geometry, body frame."""

    body: int
    world: int
    face_shape: int          # box whose thin axis is the face normal; -1 = none (body x, explicit half-thickness)
    center: np.ndarray       # [3] a point on the blade mid-plane
    axis: np.ndarray         # [3] unit face normal
    half: float              # half-thickness [m]


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate v by unit quaternions q (x, y, z, w) on the last axis; broadcast over leading axes."""
    u, w = q[..., :3], q[..., 3:4]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def blade_box(model: Any, blade: Blade) -> int:
    """The blade's own shape if it is a box on its body, else the body's box with the smallest half-extent; -1."""
    body = model.shape_body.numpy()
    kind = model.shape_type.numpy()
    scale = model.shape_scale.numpy()
    is_box = (body == blade.body) & (kind == int(newton.GeoType.BOX))
    if 0 <= blade.shape < len(body) and is_box[blade.shape]:
        return int(blade.shape)
    ids = np.flatnonzero(is_box)
    return int(ids[np.argmin(scale[ids].min(axis=1))]) if len(ids) else -1


def blade_axis(model: Any, blade: Blade) -> np.ndarray:
    """[3] unit face normal of a blade, body frame: the thin axis of its box (``blade_box``), or body x (the
    ``blade_pose`` convention) for a blade without one. The cut damage's alignment gate reads it."""
    box = blade_box(model, blade)
    if box < 0:
        return np.array([1.0, 0.0, 0.0])
    q = model.shape_transform.numpy()[box].astype(np.float64)[3:7]
    axis = quat_rotate(q, np.eye(3)[int(np.argmin(model.shape_scale.numpy()[box]))])
    return axis / np.linalg.norm(axis)


def blade_face(model: Any, blade: Blade, half_thickness: float | None = None) -> BladeFace:
    """Face normal, mid-plane point and half-thickness of a blade (see the module docstring)."""
    if blade.body < 0:
        raise AssemblyError(f"BladeContact: blade body {blade.body} is static; a blade must be on a body")
    world = int(model.body_world.numpy()[blade.body]) if getattr(model, "body_world", None) is not None else -1
    box = blade_box(model, blade)
    if box < 0:
        if half_thickness is None:
            raise AssemblyError(f"BladeContact: blade body {blade.body} has no box shape; pass half_thickness")
        log.warning("BladeContact: blade body %d has no box; face normal = body x, half-thickness %.3g m",
                    blade.body, half_thickness)
        mid = 0.5 * (np.asarray(blade.edge_a, np.float64) + np.asarray(blade.edge_b, np.float64))
        return BladeFace(blade.body, world, -1, mid, np.array([1.0, 0.0, 0.0]), float(half_thickness))
    xf = model.shape_transform.numpy()[box].astype(np.float64)
    ext = model.shape_scale.numpy()[box].astype(np.float64)
    k = int(np.argmin(ext))
    axis = quat_rotate(xf[3:7], np.eye(3)[k])
    half = float(ext[k]) if half_thickness is None else float(half_thickness)
    return BladeFace(blade.body, world, box, xf[:3], axis / np.linalg.norm(axis), half)


def blade_spine(face: BladeFace, blade: Blade) -> np.ndarray:
    """Unit vector in the face plane, normal to the edge, from the edge toward the blade's mid-plane point (body
    frame); zeros when the edge is degenerate or the point lies on the edge line (no box: the edge-advancing test of
    SCR-JOB1-2 is then off)."""
    ea, eb = np.asarray(blade.edge_a, np.float64), np.asarray(blade.edge_b, np.float64)
    e = eb - ea
    s = np.cross(face.axis, e)
    if np.linalg.norm(e) < 1e-12 or np.linalg.norm(s) < 1e-12:
        return np.zeros(3)
    s /= np.linalg.norm(s)
    side = float(s @ (face.center - 0.5 * (ea + eb)))
    return np.zeros(3) if abs(side) < 1e-9 else s * np.sign(side)


def _twin_csr(bonds: BondSet, particle_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """TET_CUT bond ids, and per particle (CSR start [P+1], bond list) the TET_CUT bonds it is a twin of."""
    kind = bonds.kind.numpy()
    cut = np.flatnonzero(kind == int(BondKind.TET_CUT)).astype(np.int32)
    owners = np.concatenate([bonds.a.numpy()[cut], bonds.b.numpy()[cut]]).astype(np.int64)
    ids = np.concatenate([cut, cut])
    order = np.argsort(owners, kind="stable")
    start = np.zeros(max(particle_count, int(owners.max(initial=-1)) + 1) + 1, np.int32)
    np.add.at(start, owners + 1, 1)
    return cut, np.cumsum(start).astype(np.int32), ids[order].astype(np.int32)


def side_a_tets(tet_indices: np.ndarray, particle_a: np.ndarray) -> np.ndarray:
    """Per bond, the lowest-index tet holding twin a (-1 when none): its F carries the seam normal."""
    first = np.full(int(max(particle_a.max(initial=-1), tet_indices.max(initial=-1))) + 1, -1, np.int64)
    if len(tet_indices):
        verts, pos = np.unique(tet_indices.ravel(), return_index=True)
        first[verts] = pos // 4
    out = np.full(len(particle_a), -1, np.int32)
    ok = particle_a >= 0
    out[ok] = first[particle_a[ok]]
    return out


class BladeContact:
    """Rewrites blade soft contacts of active pre-split twins onto the blade face of their own side.

    ``model`` is a ``newton.Model`` or a coupled-solver ``ModelView`` (duck-typed). ``half_thickness`` is per blade
    (None entries: from the blade box). ``adjust_contacts`` is graph-capturable: one fixed-size launch over
    ``contacts.soft_contact_max``, no host readback. With no blades or no TET_CUT bonds it writes nothing."""

    def __init__(self, model: Any, bonds: BondSet, blades: Sequence[Blade], *,
                 half_thickness: Sequence[float | None] | None = None,
                 config: BladeContactConfig | None = None, cut_alignment: float | None = None) -> None:
        """``cut_alignment``: the cutter's gate (``TopologyController.Config.cut_alignment``) where the caller has
        one. A gated cutter may not cut seams this contact will not open, so it must be >= ``min_alignment``."""
        self.model, self.bonds, self.blades = model, bonds, list(blades)
        self.config = config or BladeContactConfig()
        if cut_alignment is not None and cut_alignment < self.config.min_alignment:
            raise AssemblyError(f"BladeContact: the cutter's cut_alignment {cut_alignment} is under min_alignment "
                                f"{self.config.min_alignment}: it would cut seams whose twins keep the stock contact")
        self.device = model.device
        worlds = max(int(getattr(model, "world_count", 1) or 1), 1)
        # diagnostics, not simulation state (never snapshotted): per world, the records the re-entry case rewrote
        # since ``reset_reentry`` and the farthest such twin stood across the mid-plane from its own side [m]
        self.reentry_count = wp.zeros(worlds, dtype=wp.int32, device=self.device)
        self.reentry_cross = wp.zeros(worlds, dtype=wp.float32, device=self.device)
        halves = list(half_thickness) if half_thickness is not None else [None] * len(self.blades)
        if len(halves) != len(self.blades):
            raise AssemblyError(f"BladeContact: {len(halves)} half_thickness values for {len(self.blades)} blades")
        cut, start, plist = _twin_csr(bonds, int(model.particle_count)) if bonds.count else (np.zeros(0),) * 3
        self.cut_bonds = int(len(cut))
        self.enabled = bool(self.blades) and self.cut_bonds > 0 and has_bonds(model)
        if not self.enabled:
            log.info("BladeContact: %d blades, %d TET_CUT bonds; adjust_contacts is a no-op",
                     len(self.blades), self.cut_bonds)
            return
        self.faces = [blade_face(model, b, h) for b, h in zip(self.blades, halves, strict=True)]
        self._blades = self._blade_table()
        self._twins = self._twin_table(start, plist)
        log.info("BladeContact: %d blades (half-thickness %s m; spine %s), %d TET_CUT bonds, reach %.3g m, edge gate "
                 "%s", len(self.blades), ", ".join(f"{f.half:.3g}" for f in self.faces),
                 ", ".join(str(np.round(blade_spine(f, b), 3).tolist()) for f, b in zip(self.faces, self.blades)),
                 self.cut_bonds, self.config.reach, self.config.edge_gate)

    def _arr(self, x: Any, dtype: Any) -> wp.array:
        return wp.array(np.asarray(x), dtype=dtype, device=self.device)

    def _blade_table(self) -> BladeTable:
        t = BladeTable()
        f, b = self.faces, self.blades
        t.body = self._arr([x.body for x in f], wp.int32)
        t.world = self._arr([x.world for x in f], wp.int32)
        t.face_shape = self._arr([x.face_shape for x in f], wp.int32)
        t.center = self._arr(np.stack([x.center for x in f]).astype(np.float32), wp.vec3)
        t.axis = self._arr(np.stack([x.axis for x in f]).astype(np.float32), wp.vec3)
        t.half = self._arr([x.half for x in f], wp.float32)
        t.edge_a = self._arr(np.asarray([x.edge_a for x in b], np.float32), wp.vec3)
        t.edge_b = self._arr(np.asarray([x.edge_b for x in b], np.float32), wp.vec3)
        t.edge_radius = self._arr([x.edge_radius for x in b], wp.float32)
        spine = np.stack([blade_spine(x, y) for x, y in zip(f, b, strict=True)])
        t.spine = self._arr(spine.astype(np.float32), wp.vec3)
        t.com = self._arr(self.model.body_com.numpy()[[x.body for x in f]].astype(np.float32), wp.vec3)
        shape_blade = np.full(int(self.model.shape_count), -1, np.int32)
        shape_body = self.model.shape_body.numpy()
        for j, face in enumerate(f):
            if (shape_blade[shape_body == face.body] >= 0).any():
                log.warning("BladeContact: two blades on body %d; the later one wins", face.body)
            shape_blade[shape_body == face.body] = j
        t.shape_blade = self._arr(shape_blade, wp.int32)
        return t

    def _twin_table(self, start: np.ndarray, plist: np.ndarray) -> TwinTable:
        t = TwinTable()
        m = self.model
        a = self.bonds.a.numpy().astype(np.int64)
        is_cut = self.bonds.kind.numpy() == int(BondKind.TET_CUT)
        tet = side_a_tets(m.tet_indices.numpy() if m.tet_count else np.zeros((0, 4), np.int64), np.where(is_cut, a, -1))
        t.start, t.bonds = self._arr(start, wp.int32), self._arr(plist, wp.int32)
        t.a, t.b, t.world = self.bonds.a, self.bonds.b, self.bonds.world
        t.normal = bond_column(m, "normal")
        t.tet = self._arr(tet, wp.int32)
        t.tet_indices, t.tet_poses = m.tet_indices, m.tet_poses
        t.radius = m.particle_radius
        t.intact, t.damage = self.bonds.intact, self.bonds.damage
        log.debug("BladeContact: %d of %d TET_CUT bonds have no side-a tet (rest normal)",
                  int((tet[is_cut] < 0).sum()), self.cut_bonds)
        return t

    def reset_reentry(self) -> None:
        """Zero the re-entry diagnostics (``reentry_count``, ``reentry_cross``)."""
        self.reentry_count.zero_()
        self.reentry_cross.zero_()

    def seam_normals(self, state: newton.State) -> wp.array:
        """[B] current seam normal per bond (F^-T n0 for TET_CUT bonds with a side-a tet; diagnostics, tests)."""
        out = wp.zeros(self.bonds.count, dtype=wp.vec3, device=self.device)
        if self.enabled:
            wp.launch(seam_normals_kernel, dim=self.bonds.count, device=self.device,
                      inputs=[self._twins, state.particle_q], outputs=[out])
        return out

    def adjust_contacts(self, state: newton.State, contacts: newton.Contacts | None) -> None:
        """Rewrite this step's blade contacts of active twins (after ``collide``, before the solver step)."""
        if not self.enabled or contacts is None or contacts.soft_contact_max == 0:
            return
        tw = self._twins
        tw.intact, tw.damage = self.bonds.intact, self.bonds.damage     # follow a reassigned BondSet array
        c = self.config
        wp.launch(rewrite_blade_contacts, dim=contacts.soft_contact_max, device=self.device, inputs=[
            self._blades, tw, state.body_q, state.body_qd, state.particle_q, state.particle_qd, float(c.reach),
            float(c.min_alignment), int(c.edge_gate), int(c.reentry), contacts.soft_contact_count,
            contacts.soft_contact_max,
            contacts.soft_contact_indices, contacts.soft_contact_barycentric, contacts.soft_contact_shape,
            contacts.soft_contact_body_pos, contacts.soft_contact_body_vel, contacts.soft_contact_normal,
            contacts.soft_contact_rigid_indices, self.reentry_count, self.reentry_cross])
