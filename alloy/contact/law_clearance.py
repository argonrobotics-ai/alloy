# SPDX-License-Identifier: Apache-2.0
"""A8.clearance: centred gap and maximum tilt of the CON-46 peg in its bore, kinematic (bench §4 A8, SPEC §5.3).

The 12.0 mm peg and the 12.2 mm blind bore are 96-gons in the same angular phase (radial clearance uniform to
0.05 µm), sdf mode (texture SDF, voxel 50 µm). Poses are written straight into ``body_q`` and collided once; no solver
steps, so the law runs on ``solver="none"``. A row's separation is ``n·(x1 − x0) − (margin0 + margin1)``, the quantity
every solver consumes (as CT-5/CT-7).

- Centred gap: peg axis on the bore axis at several depths; every world's minimum separation must be within 1 µm of
  the radial clearance (D − d)/2 = 0.100 mm.
- Max tilt: a peg tilted by θ toward φ, its axis crossing the bore-top plane at offset s along φ and its bottom face
  ``depth`` below that crossing (along the axis), touches the bore at two points: its side at the top edge (+φ) and
  its bottom rim on the −φ wall. Their axial distance is l = depth + (d/2)·tanθ, and a non-penetrating s exists iff
  W(θ) = D − (d/2)(cosθ + secθ) − depth·sinθ ≥ 0, i.e. iff d·cosθ + l·sinθ ≤ D (the closed form). W, the width of the
  feasible-shift interval, is measured as g₊ + g₋: the minimum reported separations on the +φ and −φ sides with the
  peg at the middle of that interval (|g| ≪ the 0.5 mm gap band, so both sides report rows; the sum does not depend
  on s to 1e-4). θ_max(φ) is the root of a line fitted to W over a θ grid around the closed form. A free/blocked grid
  over (θ, s) as in CT-5 quantises θ_max by (s step)/(dW/dθ ≈ l): 8 µm / 10 mm ≈ 0.05°, too coarse for 0.01°.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alloy.bench import compare as C
from alloy.bench.types import Condition, MetricVerdict
from alloy.contact.bias import _world_points
from alloy.contact.pipeline import AlloyContactLayer
from alloy.core.log import get_logger
from alloy.testing.contact import PegHoleSpec, peg_hole_scene

log = get_logger("contact.laws")

COLLIDE_DT = 1.0e-3          # s; poses carry no velocity, so the speculative band is the authored gap
START_GAP_M = 5.0e-3         # scene build pose (overwritten): peg well above the bore bottom


# ---------------------------------------------------------------------------------------------- closed forms ---

def closed_form_tilt(d: float, D: float, l: float) -> float:
    """θ [rad] solving d·cosθ + l·sinθ = D (the smaller root; d < D < √(d² + l²))."""
    hyp = math.hypot(d, l)
    if not d < D < hyp:
        raise ValueError(f"no tilt root for d={d}, D={D}, l={l} (need d < D < sqrt(d^2 + l^2))")
    return math.atan2(l, d) - math.acos(D / hyp)


def free_width(d: float, D: float, depth: float, theta: float) -> float:
    """W(θ) = D − (d/2)(cosθ + secθ) − depth·sinθ [m]: width of the non-penetrating shift interval (module doc)."""
    return D - 0.5 * d * (math.cos(theta) + 1.0 / math.cos(theta)) - depth * math.sin(theta)


def contact_length(d: float, depth: float, theta: float) -> float:
    """Axial distance l between the bottom-rim and top-edge contacts [m]: depth + (d/2)·tanθ."""
    return depth + 0.5 * d * math.tan(theta)


def tilt_at_depth(d: float, D: float, depth: float) -> float:
    """θ_max [rad] for a bottom face ``depth`` below the axis' top-plane crossing: the root of ``free_width``
    (equivalently d·cosθ + l·sinθ = D with l = ``contact_length``)."""
    lo, hi = 0.0, math.pi / 4
    if free_width(d, D, depth, lo) <= 0.0 or free_width(d, D, depth, hi) >= 0.0:
        raise ValueError(f"no tilt root in [0, 45 deg] for d={d}, D={D}, depth={depth}")
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if free_width(d, D, depth, mid) > 0.0 else (lo, mid)
    return 0.5 * (lo + hi)


def mid_shift(g: PegHoleSpec, theta: float, depth: float) -> float:
    """Middle of the feasible shift interval [−R_b + depth·sinθ + r·cosθ, R_b − r/cosθ] [m] (may be empty)."""
    r, rb = 0.5 * g.peg_d, 0.5 * g.bore_d
    return 0.5 * ((-rb + depth * math.sin(theta) + r * math.cos(theta)) + (rb - r / math.cos(theta)))


def tilted_pose(g: PegHoleSpec, phi: float, theta: float, shift: float, depth: float) -> np.ndarray:
    """Peg body pose [x y z qx qy qz qw]: axis tilted θ toward φ, crossing the top plane at ``shift`` along φ, bottom
    face ``depth`` below that crossing along the axis."""
    axis = np.array([math.cos(phi) * math.sin(theta), math.sin(phi) * math.sin(theta), math.cos(theta)])
    top = np.array([shift * math.cos(phi), shift * math.sin(phi), g.height])
    centre = top - depth * axis + 0.5 * g.peg_len * axis
    k = np.array([-math.sin(phi), math.cos(phi), 0.0])            # rotates +Z toward +φ
    return np.r_[centre, k * math.sin(0.5 * theta), math.cos(0.5 * theta)]


# ------------------------------------------------------------------------------------------ contact readout ---

@dataclass(frozen=True, eq=False)
class ContactRows:
    """Host copy of the layer's rows: world, separation [m] and world-frame midpoint of the two contact points."""

    world: np.ndarray
    sep: np.ndarray
    mid: np.ndarray

    def world_min(self, worlds: int, mask: np.ndarray | None = None) -> np.ndarray:
        """Per-world minimum separation over the (masked) rows; inf where a world has none."""
        keep = np.ones(self.sep.shape, bool) if mask is None else mask
        out = np.full(worlds, np.inf)
        np.minimum.at(out, self.world[keep], self.sep[keep])
        return out

    def side_gaps(self, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(g₊, g₋) per world: minimum separation of rows on the +φ / −φ side of the bore axis (``phi`` per world)."""
        p = phi[self.world]
        along = self.mid[:, 0] * np.cos(p) + self.mid[:, 1] * np.sin(p)
        return self.world_min(len(phi), along > 0.0), self.world_min(len(phi), along < 0.0)


def contact_rows(model: Any, state: Any, contacts: Any) -> ContactRows:
    """Read the layer's contact buffer (host sync; the law collides once)."""
    n = int(contacts.rigid_contact_count.numpy()[0])
    s0, s1 = contacts.rigid_contact_shape0.numpy()[:n], contacts.rigid_contact_shape1.numpy()[:n]
    nrm = contacts.rigid_contact_normal.numpy()[:n].astype(np.float64)
    margin = (contacts.rigid_contact_margin0.numpy()[:n] + contacts.rigid_contact_margin1.numpy()[:n]).astype(float)
    body, world = model.shape_body.numpy(), model.shape_world.numpy()
    bq = state.body_q.numpy().astype(np.float64)
    x0 = _world_points(body[s0], contacts.rigid_contact_point0.numpy()[:n], bq)
    x1 = _world_points(body[s1], contacts.rigid_contact_point1.numpy()[:n], bq)
    sep = np.einsum("ij,ij->i", nrm, x1 - x0) - margin
    return ContactRows(np.maximum(world[s0], world[s1]).astype(np.int64), sep, 0.5 * (x0 + x1))


def line_root(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """(root, rms residual) of a least-squares line through finite (x, y); NaN root if it cannot be fitted or the
    line does not decrease (W must shrink as the tilt grows)."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return math.nan, math.nan
    a, b = np.polyfit(x[ok], y[ok], 1)
    rms = float(np.sqrt(np.mean((np.polyval((a, b), x[ok]) - y[ok]) ** 2)))
    return (float(-b / a) if a < 0.0 else math.nan), rms


# ------------------------------------------------------------------------------------------------- the twin ---

@dataclass(frozen=True)
class ClearancePlan:
    """World layout: centred worlds first, then ``directions`` × ``tilts`` tilted worlds (θ inner)."""

    directions: int
    tilts: int
    depth: float = 10.0e-3                                   # m, the CT-5 engagement
    centred_depths: tuple[float, ...] = (5.0e-3, 10.0e-3, 15.0e-3)
    span: tuple[float, float] = (0.8, 1.2)                   # θ grid as a fraction of the closed-form θ_max
    geometry: PegHoleSpec = field(default_factory=PegHoleSpec)

    @classmethod
    def for_run(cls, quick: bool) -> ClearancePlan:
        return cls(directions=8, tilts=9) if quick else cls(directions=36, tilts=17)

    @property
    def theta_closed(self) -> float:
        g = self.geometry
        return tilt_at_depth(g.peg_d, g.bore_d, self.depth)

    def phis(self) -> np.ndarray:
        return 2.0 * math.pi * np.arange(self.directions) / self.directions

    def thetas(self) -> np.ndarray:
        return self.theta_closed * np.linspace(self.span[0], self.span[1], self.tilts)

    def poses(self) -> tuple[np.ndarray, np.ndarray]:
        """([N, 7] peg poses, [N] φ per world; 0 for the centred worlds)."""
        g = self.geometry
        out = [tilted_pose(g, 0.0, 0.0, 0.0, dep) for dep in self.centred_depths]
        phi = [0.0] * len(self.centred_depths)
        for p in self.phis():
            for th in self.thetas():
                out.append(tilted_pose(g, float(p), float(th), mid_shift(g, float(th), self.depth), self.depth))
                phi.append(float(p))
        return np.asarray(out, np.float32), np.asarray(phi, np.float64)


class ClearanceTwin:
    """A8.clearance (module docstring): one collide of a pose grid, judged directly (``evaluate`` hook)."""

    law_id = "A8.clearance"
    units: Mapping[str, str] = {"gap_err_m": "m", "tilt_err_deg": "deg"}
    contacts_per_world = 64

    def __init__(self) -> None:
        self.monitors: dict[str, Any] = {}

    def conditions(self, quick: bool) -> tuple[Condition, ...]:
        plan = ClearancePlan.for_run(quick)
        return tuple(Condition(f"phi{math.degrees(p):.0f}", {"phi": float(p), "depth": plan.depth})
                     for p in plan.phis())

    def collide_plan(self, plan: ClearancePlan, poses: np.ndarray, device: str) -> ContactRows:
        """Build ``peg_hole_scene`` (sdf mode, CUDA), write one peg pose per world and collide once."""
        scene = peg_hole_scene(len(poses), device=device, mode="sdf", geometry=plan.geometry, start_gap=START_GAP_M)
        layer = AlloyContactLayer(scene.model, [scene.spec], native="mujoco",
                                  precision_contacts_per_world=self.contacts_per_world)
        state = scene.model.state()
        q = state.body_q.numpy()
        q[scene.peg_bodies] = poses
        state.body_q.assign(q)
        layer.collide(state, layer.contacts(), dt=COLLIDE_DT)
        self.monitors["overflow"] = layer.overflow_report()
        self.monitors["max_penetration_m"] = scene.spec.resolved_max_penetration()
        return contact_rows(scene.model, state, layer.contacts())

    def tilt_errors(self, rows: ContactRows, plan: ClearancePlan, phi: np.ndarray) -> np.ndarray:
        """|θ_max(φ) − θ_closed| [deg] per direction from the root of W = g₊ + g₋ over the θ grid."""
        g_plus, g_minus = rows.side_gaps(phi)
        n0, th = len(plan.centred_depths), plan.thetas()
        width = (g_plus + g_minus)[n0:].reshape(plan.directions, plan.tilts)
        fits = [line_root(th, w) for w in width]
        theta_max = np.array([f[0] for f in fits])
        th_c = plan.theta_closed
        self.monitors.update({
            "theta_closed_deg": math.degrees(th_c), "depth_m": plan.depth,
            "l_contact_m": contact_length(plan.geometry.peg_d, plan.depth, th_c),
            "theta_max_deg": np.degrees(theta_max).tolist(), "width_fit_rms_m": [f[1] for f in fits],
            "width_at_closed_m": [float(np.interp(th_c, th, w)) for w in width]})
        return np.abs(np.degrees(theta_max - th_c))

    def evaluate(self, ctx: Any) -> tuple[MetricVerdict, ...]:
        plan = ClearancePlan.for_run(ctx.quick)
        poses, phi = plan.poses()
        rows = self.collide_plan(plan, poses, ctx.device)
        n0, radial = len(plan.centred_depths), plan.geometry.radial_clearance
        gaps = rows.world_min(len(phi))[:n0]
        gap_err = np.abs(gaps - radial)
        tilt_err = self.tilt_errors(rows, plan, phi)
        self.monitors.update({"centred_gap_m": gaps.tolist(), "radial_clearance_m": radial,
                              "worlds": len(phi), "mode": "sdf"})
        log.info("A8.clearance: centred gaps %s m, tilt errors %s deg", np.round(gaps, 8).tolist(),
                 np.round(tilt_err, 5).tolist())
        spec = ctx.spec
        return (C.compare_errors("gap_err_m", "m", gap_err, spec.metrics["gap_err_m"], None, spec.stat["gap_err_m"]),
                C.compare_errors("tilt_err_deg", "deg", tilt_err, spec.metrics["tilt_err_deg"], None,
                                 spec.stat["tilt_err_deg"]))


__all__ = ["ClearanceTwin", "ClearancePlan", "ContactRows", "contact_rows", "closed_form_tilt", "tilt_at_depth",
           "free_width", "contact_length", "mid_shift", "tilted_pose", "line_root"]
