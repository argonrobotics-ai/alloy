# SPDX-License-Identifier: Apache-2.0
"""CT-13 under C7's grip: two pad pairs (kinematic boxes) come down on a towel near one long edge, close on it and
gather a ridge, carry it round the fold line, set it down, open and lift away. The squeezed ridge tip and the curled
corners are where C7's crossings were ([internal note] clips/C7/C7.md: 25-101 crease-local crossings per folded world).

The towel is ``laundry_crossings``' (card route, ``features.self_contact``, plywood top); the pads are kinematic
object cards whose poses and velocities are written before every VBD step (only VBD steps: the pads go where they
are told). Crossings are counted every ``every`` seconds; the result also says whether the ridge came along (the
towel's top row within the pads' reach at the top of the swing).

Run: ``python -m alloy.tasks.laundry_pinch [--folds half] [--rig '{"closed_gap": 1.5}'] [--out F]``.
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

from alloy.tasks.laundry_crossings import TABLE, TABLE_HALF_Z, _CardOverride, _count, _Stepper

PAD = "laundry_pad"
PAD_HALF = (0.010, 0.004, 0.008)       # [m] x (width), y (jaw axis), z (height); C7's pads are ~20 x 16 mm faces


@dataclass(frozen=True)
class PinchRig:
    card: str = "towel_cotton"
    size: tuple[float, float] | None = None
    folds: str = "half"                 # one flap round the middle ("half") or round the first third ("third")
    inset: float = 0.012                # pinch line from the long edge [m] (C7 12 mm)
    corner_inset: float = 0.04          # pinch centres from the short edges [m]
    open_gap: float = 0.024             # pads' inner faces apart while open [m]
    closed_gap: float = 2.0             # ... while closed [sheet thicknesses]: two layers of the ridge
    press: float = 0.5                  # pad bottoms this many h into the towel's top while they close
    apex: float = 0.5                   # swing height / fold radius
    phases: tuple[float, ...] = (0.2, 0.4, 1.0, 0.3, 0.3, 0.5)    # s: down, close, swing, open, up, settle
    every: float = 0.05
    vbd: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PinchResult:
    crossings_max: int
    crossings_end: int
    first_crossing_t: float | None
    frames: list[tuple[float, int]]
    carried: bool                       # towel within the pads' reach at the top of the swing
    finite: bool
    dt: float
    wall_s: float


def build(rig: PinchRig, device: str = "cpu"):
    """Table, towel and four kinematic pads (two pairs) through the card route: (model, kwargs, dt, h)."""
    from alloy.cards import types as ct
    from alloy.cards.library import CardLibrary
    from alloy.core.registry import isolated_registry
    from alloy.testing import integration as I

    lib0 = CardLibrary()
    base = lib0.object(rig.card)
    towel = dataclasses.replace(base, features=dataclasses.replace(base.features, self_contact=True))
    if rig.size is not None:
        towel = dataclasses.replace(towel, shape=dataclasses.replace(
            towel.shape, size=(0.5 * rig.size[0], 0.5 * rig.size[1], towel.shape.size[2])))
    table = lib0.object(TABLE)
    pad = dataclasses.replace(table, name=PAD, motion="kinematic",
                              shape=dataclasses.replace(table.shape, size=PAD_HALF))
    h = float(towel.features.thickness.value)
    sx, sy = (2.0 * towel.shape.size[0], 2.0 * towel.shape.size[1])
    pads = [ct.Placement(PAD, (x, y, 0.05, 0.0, 0.0, 0.0, 1.0)) for x, y in _pad_xy(rig, sx, sy, open_=True)]
    scene = ct.SceneCard(robots=(), objects=(ct.Placement(TABLE, (0.0, 0.0, -TABLE_HALF_Z, 0.0, 0.0, 0.0, 1.0)),
                                             ct.Placement(rig.card, (0.0, 0.0, 0.5 * h, 0.0, 0.0, 0.0, 1.0)),
                                             *pads))
    with isolated_registry(empty=True):
        built = I.assemble(scene, _CardOverride({rig.card: towel, PAD: pad}), device, 1)
    (vbd,) = [e for e in built.entries if e.capability == "vbd"]
    kwargs = {k: v for k, v in {**dict(vbd.solver_kwargs), **rig.vbd}.items() if v is not None}
    _make_kinematic(built.model)
    return built.model, kwargs, float(vbd.target_dt), h


def _make_kinematic(model) -> None:
    """Zero the pads' inverse mass and inertia: VBD then never moves them (the card route gives kinematic bodies
    their geometric mass for the rigid entry, which this rig does not step)."""
    for name in ("body_inv_mass", "body_mass"):
        arr = getattr(model, name)
        arr.assign(np.zeros(arr.shape, dtype=np.float32))
    for name in ("body_inv_inertia", "body_inertia"):
        arr = getattr(model, name)
        arr.assign(np.zeros((arr.shape[0], 3, 3), dtype=np.float32))


def _pad_xy(rig: PinchRig, sx: float, sy: float, open_: bool, gap: float | None = None) -> list[tuple[float, float]]:
    """Pad centres (x, y): two pairs at x = +-(sx/2 - corner_inset) astride the pinch line y = -sy/2 + inset."""
    g = rig.open_gap if open_ else gap
    yc, off = -0.5 * sy + rig.inset, 0.5 * g + PAD_HALF[1]
    xs = (-0.5 * sx + rig.corner_inset, 0.5 * sx - rig.corner_inset)
    return [(x, yc + s * off) for x in xs for s in (-1.0, 1.0)]


@dataclass
class _Pads:
    """Kinematic pad bodies: (body ids) and a pose writer."""
    ids: np.ndarray
    rest: np.ndarray                    # [4, 3] pad centres at build

    def put(self, st: _Stepper, pos: np.ndarray) -> None:
        q, qd = st.s0.body_q.numpy(), st.s0.body_qd.numpy()
        qd[self.ids, :3] = (pos - q[self.ids, :3]) / st.dt
        qd[self.ids, 3:] = 0.0
        q[self.ids, :3] = pos
        q[self.ids, 3:] = (0.0, 0.0, 0.0, 1.0)
        st.s0.body_q.assign(q)
        st.s0.body_qd.assign(qd)


def _path(rig: PinchRig, sx: float, sy: float, h: float) -> list[tuple[float, Any]]:
    """(duration, f(s) -> pad centres [4, 3]) per phase."""
    z_up, z_on = 0.05, h + PAD_HALF[2] - rig.press * h
    shut = np.array([(x, y, z_on) for x, y in _pad_xy(rig, sx, sy, False, rig.closed_gap * h)])
    opened = np.array([(x, y, z_on) for x, y in _pad_xy(rig, sx, sy, True)])
    y_fold = -0.5 * sy + (0.5 if rig.folds == "half" else 1.0 / 3.0) * sy
    r = shut[:, 1] - y_fold
    z_land = z_on + 2.0 * h
    lerp = lambda a, b: (lambda s: a + (b - a) * s)                                      # noqa: E731

    def swing(s: float) -> np.ndarray:
        th = math.pi * s
        out = shut.copy()
        out[:, 1] = y_fold + r * math.cos(th)
        out[:, 2] = z_on + rig.apex * np.abs(r) * math.sin(th) + (z_land - z_on) * (1.0 - math.cos(th)) / 2.0
        return out

    landed = swing(1.0)
    land_open = landed.copy()
    land_open[:, 1] = y_fold - (opened[:, 1] - y_fold)
    lifted = land_open + np.array([0.0, 0.0, z_up])
    above = opened.copy()
    above[:, 2] = z_up
    d = rig.phases
    return [(d[0], lerp(above, opened)), (d[1], lerp(opened, shut)), (d[2], swing),
            (d[3], lerp(landed, land_open)), (d[4], lerp(land_open, lifted)), (d[5], lambda s: lifted)]


def run(rig: PinchRig = PinchRig(), device: str = "cpu") -> PinchResult:
    t0 = time.time()
    model, kwargs, dt, h = build(rig, device)
    q0, tri = model.particle_q.numpy(), model.tri_indices.numpy()
    sx, sy = float(np.ptp(q0[:, 0])), float(np.ptp(q0[:, 1]))
    rest = model.body_q.numpy()[:, :3]
    order = [int(np.argmin(np.linalg.norm(rest[:, :2] - np.asarray(xy), axis=1)))
             for xy in _pad_xy(rig, sx, sy, open_=True)]               # bodies in _pad_xy's order
    pads = _Pads(np.asarray(order), rest[order].copy())
    st, log, carried = _Stepper(model, kwargs, dt), [], False
    every = max(int(round(rig.every / dt)), 1)
    k = 0
    for i, (dur, f) in enumerate(_path(rig, sx, sy, h)):
        n = max(int(round(dur / dt)), 1)
        for j in range(1, n + 1):
            pads.put(st, f(j / n))
            st.step()
            k += 1
            if k % every == 0:
                log.append((round(k * dt, 6), _count(st.s0.particle_q.numpy(), tri)))
            if i == 2 and j == n // 2:
                carried = _carried(st.s0.particle_q.numpy(), f(0.5))
    q = st.s0.particle_q.numpy()
    counts = [c for _, c in log]
    return PinchResult(max(counts, default=0), counts[-1] if counts else 0,
                       next((t for t, c in log if c > 0), None), log, carried, bool(np.isfinite(q).all()), dt,
                       round(time.time() - t0, 1))


def _carried(q: np.ndarray, pads: np.ndarray) -> bool:
    """Some towel particles within a pad size of each pair's centre at the top of the swing."""
    reach = float(np.linalg.norm(PAD_HALF))
    centres = 0.5 * (pads[0::2] + pads[1::2])
    return all(bool((np.linalg.norm(q - c, axis=1) < reach).any()) for c in centres)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--rig", default="{}", help="JSON PinchRig fields")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    over = json.loads(a.rig)
    for key in ("size", "phases"):
        if key in over:
            over[key] = tuple(over[key])
    rig = PinchRig(**over)
    row = json.dumps({"rig": dataclasses.asdict(rig), **dataclasses.asdict(run(rig, a.device))}, default=float)
    print(row)
    if a.out:
        with open(a.out, "a") as fh:
            fh.write(row + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
