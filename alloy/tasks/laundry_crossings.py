# SPDX-License-Identifier: Apache-2.0
"""CT-13 on a fold (Gym-Anything laundry gap, REPORT.md section 2 cloth row; coordinator 2026-11-06 18:40Z): a towel
card on the cloth route with ``features.self_contact`` (WP2's entry-wide VBD self contact, SCR-OSS22-4) lies on a
plywood table top; one long edge is carried round a fold line onto the rest of the towel and let go. Folds:
``half`` (long edge to long edge) or ``thirds`` (two flaps, the second over the first). Counted every ``every``
seconds: edge-through-triangle crossings of the towel with itself (``fold_checks.self_intersections``, CT-13: 0) and
the towel's depth into the table top.

The carried edge row is kinematic while it moves (mass 0, its positions written before every step) and gets its mass
back when it lets go. The VBD entry, its solver kwargs and its step come from the route; ``vbd`` overrides kwargs.

Run: ``python -m alloy.tasks.laundry_crossings [--folds half|thirds] [--vbd '{"k": v}'] [--out F]``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.tasks.fold_checks import self_intersections, table_penetration

TABLE = "table_top_plywood"            # static, half-extents 0.6 x 0.4 x 0.009 m: its top at z = 0 below
TABLE_HALF_Z = 0.009


@dataclass(frozen=True)
class FoldRig:
    card: str = "towel_cotton"
    folds: str = "half"                 # "half" or "thirds"
    self_contact: bool = True           # the card's features.self_contact (the route's switch)
    fold_s: float = 1.0                 # s per carried flap, round the fold line
    settle_s: float = 0.5               # s after each release
    landing_gap: float = 1.5            # carried edge lands this many sheet thicknesses above the layers below
    apex: float = 0.5                   # swing height / fold radius (0.95, C7's near-taut swing, lifts a light towel's
    #                                     far half with the flap and turns the whole towel over)
    inset: float = 0.0                  # carry the row this far inside the edge [m] (C7 pinches 12 mm in: the strip
    #                                     beyond it folds back on itself, a crease as sharp as the mesh allows)
    every: float = 0.05                 # s between crossing counts
    vbd: dict[str, Any] = field(default_factory=dict)      # solver kwargs over the route's (None drops one)
    size: tuple[float, float] | None = None                 # sheet (x, y) [m] over the card's (x the long axis)
    grasp: str = "edge"                 # "edge": the whole row; "corners": two patches at its ends (two hands)
    corner_inset: float = 0.03          # [m] patch centres in from the short edges ("corners")
    patch: float = 0.015                # [m] patch half-width along x ("corners")
    thickness: float | None = None      # [m] the sheet's (contact) thickness over the card's; it bends as its fabric
    particles: int | None = None        # the card's particle budget (sets the cell)
    mass_scale: float = 1.0             # every particle's mass x this (a thicker sheet weighs density x thickness)


@dataclass(frozen=True)
class CrossingResult:
    crossings_max: int                  # most self crossings in any counted frame
    crossings_end: int
    first_crossing_t: float | None      # s
    table_pen: float                    # deepest particle sphere into the table top [m]
    edge_error: float                   # carried edge's distance from its target line at the end, worst point [m]
    y_span: float                       # towel's width across the fold lines at the end (half: ~1/2, thirds: ~1/3) [m]
    y_span0: float
    finite: bool
    frames: list[tuple[float, int]]     # (t, crossings)
    solver_kwargs: dict[str, Any]
    dt: float
    particles: int
    cell: float
    wall_s: float


class _CardOverride:
    """A CardLibrary that serves changed object cards (and material cards) by name (in memory; nothing written)."""

    def __new__(cls, cards: dict[str, Any], materials: dict[str, Any] | None = None):
        from alloy.cards.library import CardLibrary

        mats = dict(materials or {})

        class Lib(CardLibrary):
            def object(self, name, *a, **k):
                return cards.get(name) or super().object(name, *a, **k)

            def peek_object(self, name):
                return cards.get(name) or super().peek_object(name)

            def material(self, name, *a, **k):
                return mats.get(name) or super().material(name, *a, **k)

            def has_material(self, name):
                return name in mats or super().has_material(name)

        return Lib()


def build(rig: FoldRig, device: str = "cpu") -> tuple[newton.Model, dict[str, Any], float, float]:
    """The table + towel scene through the card route: (model, VBD solver kwargs, entry dt, sheet thickness h)."""
    from alloy.cards import types as ct
    from alloy.cards.library import CardLibrary
    from alloy.core.registry import isolated_registry
    from alloy.testing import integration as I

    base = CardLibrary().object(rig.card)
    card = dataclasses.replace(base, features=dataclasses.replace(base.features, self_contact=rig.self_contact))
    if rig.thickness is not None:
        f = card.features
        card = dataclasses.replace(card, features=dataclasses.replace(
            f, thickness=dataclasses.replace(f.thickness, value=rig.thickness)))
    if rig.particles is not None:
        card = dataclasses.replace(card, budget=dataclasses.replace(card.budget, particles=rig.particles))
    if rig.size is not None:
        sz = card.shape.size
        card = dataclasses.replace(card, shape=dataclasses.replace(card.shape, size=(0.5 * rig.size[0],
                                                                                     0.5 * rig.size[1], sz[2])))
    lib = _CardOverride({rig.card: card})
    h = float(card.features.thickness.value)
    scene = ct.SceneCard(robots=(), objects=(ct.Placement(TABLE, (0.0, 0.0, -TABLE_HALF_Z, 0.0, 0.0, 0.0, 1.0)),
                                             ct.Placement(rig.card, (0.0, 0.0, 0.5 * h, 0.0, 0.0, 0.0, 1.0))))
    with isolated_registry(empty=True):
        built = I.assemble(scene, lib, device, 1)
    if rig.mass_scale != 1.0:
        m = built.model
        _set_mass(m, np.arange(m.particle_count), m.particle_mass.numpy() * rig.mass_scale)
    (vbd,) = [e for e in built.entries if e.capability == "vbd"]
    kwargs = {k: v for k, v in {**dict(vbd.solver_kwargs), **rig.vbd}.items() if v is not None}   # None: drop
    return built.model, kwargs, float(vbd.target_dt), h


@dataclass
class _Flap:
    """One carried flap: its edge row swings round the fold line y = ``y_fold`` (an axis along x) onto the layers."""
    rows: np.ndarray                    # particle ids of the carried edge
    start: np.ndarray                   # their positions [K, 3] at the start of the swing
    y_fold: float
    z_land: float                       # mid-surface height the edge lands at
    apex: float = 0.5                   # swing height / fold radius

    def at(self, s: float) -> np.ndarray:
        """Edge positions at swing fraction s in [0, 1]: a half ellipse round the fold line, ``apex`` of the fold
        radius high, landing at ``z_land`` on the far side."""
        th = math.pi * min(max(s, 0.0), 1.0)
        r = self.start[:, 1] - self.y_fold                          # signed fold radius per point
        z0 = self.start[:, 2]
        out = self.start.copy()
        out[:, 1] = self.y_fold + r * math.cos(th)
        out[:, 2] = z0 + self.apex * np.abs(r) * math.sin(th) + (self.z_land - z0) * (1.0 - math.cos(th)) / 2.0
        return out


def _flaps(q: np.ndarray, rig: FoldRig, h: float) -> list[tuple[np.ndarray, float, int]]:
    """(edge particle ids, fold line y, layers under the landing) per flap; the towel's long axis is x."""
    y0, y1 = float(q[:, 1].min()), float(q[:, 1].max())
    rows_y = np.unique(np.round(q[:, 1], 6))
    tol = 1.0e-6 + 0.25 * (y1 - y0) / max(len(rows_y) - 1, 1)
    y_lo = float(rows_y[np.argmin(np.abs(rows_y - (y0 + rig.inset)))])     # nearest grid row to the pinch line
    y_hi = float(rows_y[np.argmin(np.abs(rows_y - (y1 - rig.inset)))])
    lo, hi = np.flatnonzero(np.abs(q[:, 1] - y_lo) < tol), np.flatnonzero(np.abs(q[:, 1] - y_hi) < tol)
    if rig.grasp == "corners":
        lo, hi = _corners(q, lo, rig), _corners(q, hi, rig)
    if rig.folds == "half":
        return [(lo, 0.5 * (y0 + y1), 1)]
    if rig.folds == "thirds":
        w = (y1 - y0) / 3.0
        return [(lo, y0 + w, 1), (hi, y1 - w, 2)]
    raise ValueError(f"folds: {rig.folds!r} (half or thirds)")


def _corners(q: np.ndarray, row: np.ndarray, rig: FoldRig) -> np.ndarray:
    """The row's particles within ``rig.patch`` of the two points ``rig.corner_inset`` in from its ends."""
    x = q[row, 0]
    ends = (float(x.min()) + rig.corner_inset, float(x.max()) - rig.corner_inset)
    keep = (np.abs(x - ends[0]) <= rig.patch) | (np.abs(x - ends[1]) <= rig.patch)
    return row[keep]


def _set_mass(model: newton.Model, ids: np.ndarray, mass: np.ndarray) -> None:
    m, im = model.particle_mass.numpy(), model.particle_inv_mass.numpy()
    m[ids] = mass
    im[ids] = np.where(mass > 0.0, 1.0 / np.maximum(mass, 1e-30), 0.0)
    model.particle_mass.assign(m)
    model.particle_inv_mass.assign(im)


class _Stepper:
    """VBD (Alloy's factory, the route's kwargs) + particle-shape contacts against the table."""

    def __init__(self, model: newton.Model, kwargs: dict[str, Any], dt: float) -> None:
        from alloy.solvers.vbd import make_vbd

        self.model, self.dt = model, dt
        self.solver = make_vbd(model, kwargs)
        self.s0, self.s1 = model.state(), model.state()
        self.pipe = newton.CollisionPipeline(model)
        self.contacts = self.pipe.contacts()

    def step(self, drive: tuple[np.ndarray, np.ndarray] | None = None) -> None:
        if drive is not None:
            ids, pos = drive
            q, v = self.s0.particle_q.numpy(), self.s0.particle_qd.numpy()
            v[ids] = (pos - q[ids]) / self.dt
            q[ids] = pos
            self.s0.particle_q.assign(q)
            self.s0.particle_qd.assign(v)
        self.s0.clear_forces()
        self.pipe.collide(self.s0, self.contacts)
        self.solver.step(self.s0, self.s1, None, self.contacts, self.dt)
        self.s0, self.s1 = self.s1, self.s0


def _count(q: np.ndarray, tri: np.ndarray) -> int:
    return int(self_intersections(q, tri))


def _swing(st: _Stepper, flap: _Flap, steps: int, rig: FoldRig, tri: np.ndarray, log: list) -> None:
    """Carry the flap's edge round its fold line over ``steps`` steps, counting crossings every ``rig.every``."""
    every = max(int(round(rig.every / st.dt)), 1)
    for k in range(1, steps + 1):
        st.step((flap.rows, flap.at(k / steps)))
        if k % every == 0:
            _record(st, tri, log, every)


def _record(st: _Stepper, tri: np.ndarray, log: list, every: int) -> None:
    t = (log[-1][0] if log else 0.0) + every * st.dt
    log.append((round(t, 6), _count(st.s0.particle_q.numpy(), tri)))


def _settle(st: _Stepper, steps: int, rig: FoldRig, tri: np.ndarray, log: list) -> None:
    every = max(int(round(rig.every / st.dt)), 1)
    for k in range(1, steps + 1):
        st.step()
        if k % every == 0:
            _record(st, tri, log, every)


@dataclass
class FoldState:
    """Positions at the start and the end of a fold, the triangles, the sheet thickness, the crossing log."""
    q0: np.ndarray
    q: np.ndarray
    tri: np.ndarray
    h: float
    log: list
    edge_error: float
    dt: float
    kwargs: dict[str, Any]


def simulate(rig: FoldRig, device: str = "cpu") -> FoldState:
    """Fold the towel per ``rig``: carry each flap's grasp round its fold line, let go, settle."""
    model, kwargs, dt, h = build(rig, device)
    q0, tri = model.particle_q.numpy(), model.tri_indices.numpy()
    mass0 = model.particle_mass.numpy().copy()
    plan = _flaps(q0, rig, h)
    for rows, _, _ in plan:
        _set_mass(model, rows, np.zeros(len(rows)))                 # grasped particles are kinematic from the start
    st = _Stepper(model, kwargs, dt)
    log: list = []
    swing, settle = int(round(rig.fold_s / dt)), int(round(rig.settle_s / dt))
    for rows, y_fold, layers in plan:
        start = st.s0.particle_q.numpy()[rows]
        flap = _Flap(rows, start, y_fold, float(start[:, 2].mean()) + layers * rig.landing_gap * h, rig.apex)
        _swing(st, flap, swing, rig, tri, log)
        target = flap.at(1.0)
        _set_mass(model, rows, mass0[rows])                         # let go
        _settle(st, settle, rig, tri, log)
    q = st.s0.particle_q.numpy()
    edge_err = float(np.abs(q[plan[-1][0]][:, 1] - target[:, 1]).max())
    return FoldState(q0, q, tri, h, log, edge_err, dt, kwargs)


def run(rig: FoldRig = FoldRig(), device: str = "cpu", save_q: str | None = None) -> CrossingResult:
    """CT-13: fold per ``rig`` and count self crossings on the way; ``save_q``: npz of the start and end positions
    and the triangles."""
    t_wall = time.time()
    f = simulate(rig, device)
    if save_q:
        np.savez(save_q, q0=f.q0, q=f.q, tri=f.tri)
    counts = [c for _, c in f.log]
    return CrossingResult(crossings_max=max(counts, default=0), crossings_end=counts[-1] if counts else 0,
                          first_crossing_t=next((t for t, c in f.log if c > 0), None),
                          table_pen=float(table_penetration(f.q, 0.5 * f.h)), edge_error=f.edge_error,
                          y_span=float(np.ptp(f.q[:, 1])), y_span0=float(np.ptp(f.q0[:, 1])),
                          finite=bool(np.isfinite(f.q).all()), frames=f.log,
                          solver_kwargs={k: v for k, v in f.kwargs.items() if isinstance(v, (int, float, bool))},
                          dt=f.dt, particles=len(f.q0), cell=_cell(f.q0, f.tri), wall_s=round(time.time() - t_wall, 1))


def _cell(q: np.ndarray, tri: np.ndarray) -> float:
    e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]]])
    return float(np.median(np.linalg.norm(q[e[:, 0]] - q[e[:, 1]], axis=1)))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--folds", default="half", choices=("half", "thirds"))
    p.add_argument("--card", default=FoldRig.card)
    p.add_argument("--no-self-contact", action="store_true")
    p.add_argument("--fold-s", type=float, default=FoldRig.fold_s)
    p.add_argument("--inset", type=float, default=FoldRig.inset)
    p.add_argument("--apex", type=float, default=FoldRig.apex)
    p.add_argument("--settle-s", type=float, default=FoldRig.settle_s)
    p.add_argument("--vbd", default="{}", help="JSON solver kwargs over the route's")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None, help="append the result as a JSON line")
    p.add_argument("--save-q", default=None, help="npz of the start / end positions")
    a = p.parse_args(argv)
    rig = FoldRig(card=a.card, folds=a.folds, self_contact=not a.no_self_contact, fold_s=a.fold_s, inset=a.inset,
                  apex=a.apex, settle_s=a.settle_s, vbd=json.loads(a.vbd))
    res = dataclasses.asdict(run(rig, a.device, a.save_q))
    row = json.dumps({"rig": dataclasses.asdict(rig), **res}, default=float)
    print(row)
    if a.out:
        with open(a.out, "a") as f:
            f.write(row + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
