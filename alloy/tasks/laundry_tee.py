# SPDX-License-Identifier: Apache-2.0
"""task_laundry_tee_sleeves: fold both short sleeves of a T-shirt onto its body (Gym-Anything laundry gap,
``laundry/tee/sleeve_tuck_fold``; coordinator 2026-11-06 18:40Z, the fold-a-shirt half of the ask).

The T-shirt is ``laundry_garment.make_tee``: two panels with the library's plain-weave cotton sheet card's material
and thickness, sewn along the side, sleeve and shoulder seams, lying flat on a plywood table top. It goes through the
cloth route with ``features.self_contact`` (WP2's VBD contact: the panels rest h apart). Two hands, one per cuff
(both panels, a cell deep), swing the cuffs at the same time round fold lines at the side seams onto the body. They
hold, let go, and the shirt settles. CT-13's crossings are counted throughout. Both hands at once: one at a time
dragged the light shirt across the table (mu 0.2).

Gates, from their evaluator (laundry report p.29 Table 19, T-shirt row), as read here:
- each sleeve: >= 90 % of its particles inside the body's live convex hull (15 mm margin);
- placement RMS <= 45 mm: each sleeve particle against its mirror image over the fold line (their "nearest projected
  RMS" to the goal);
- 90th-percentile normal gap <= 25 mm: a sleeve particle's height over the body particle under it;
- common: no crossing at the end, stretch <= 0.30, finite.

Two rigs. ``FINE_TEE`` (10 mm cells, particle contact stiffness 2e4 N/m with no contact damping) passes every
gate: the folded sleeves stay on the body, 2-6 mm RMS from their goal, 14-17 mm high, at rest (GPU, plain weave 18
of 18 nudged starts at 2e3-2e5 N/m; jersey 7 of 8). It is a GPU rig: 9,300 particles are about 11 h on one CPU core.
The default ``TeeRig()`` (20 mm cells, the route's contact stiffness 2e6 N/m) is the CPU law's: its sleeves reach
the body without crossing, and after release they do not stay (known gap
docs/known_gaps/cloth-sleeve-fold-springback.md). Two causes, neither the fabric's bending:
- 20 mm cells cannot crease tighter than about a 2 cm radius, so the held sleeve stands in a loop 4-6 cm high by the
  fold line; on release the loop throws a plain-weave sleeve back out, and a jersey's cuff snaps back 6-8 cm;
- at 2e6 N/m against a 46 mg vertex the four plies never come to rest (typical particle 3-5 cm/s, sleeves 0.1-0.5
  m/s, 3 s after release). At 2e4 N/m and below they rest; the contact's damping (kd 10) alone keeps them moving.
Their own report says of their shirt sleeve fold: "not shown to stay after release".

Solve: the cloth route's VBD setting, which holds self-contact plies out of the Chebyshev step by default (WP2's
hold): without it the shirt's plies keep moving and the sleeve law is a draw (7 of 12 nudged GPU starts passed;
with the hold 6 of 6). ``TeeRig.vbd`` overrides the route's kwargs.

Run: ``python -m alloy.tasks.laundry_tee [--rig '{"hold_s": 0.5}'] [--out F] [--save-q F.npz]``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

from alloy.tasks.laundry_crossings import TABLE, TABLE_HALF_Z, _CardOverride, _count, _set_mass, _Stepper
from alloy.tasks.laundry_garment import Tee, TeeShape, make_tee, write_obj

CARD = "cloth_cotton_sheet"            # plain-weave cotton, 175 g/m2, h 0.975 mm (theirs: jersey, 160 g/m2)
NAME = "tee_cotton"


@dataclass(frozen=True)
class TeeRig:
    cell: float = 0.02                  # [m] mesh spacing
    sleeve: float = 0.15                # [m] under-seam past the side seam: the folded tips land 66 mm apart
    fold_in: float = 0.0                # [m] fold lines this far inside the side seams (< 0: out in the sleeve)
    apex: float = 0.4                   # swing height / fold radius
    fold_s: float = 2.5
    hold_s: float = 0.5                 # the hands stay at the landing before letting go
    settle_s: float = 0.75
    landing_gap: float = 1.5            # [h] over the front panel
    every: float = 0.05
    vbd: dict[str, Any] = field(default_factory=dict)
    material: Any = None                # a MaterialCard for the shirt instead of CARD's (e.g. a jersey with KES 2HB)
    contact_ke: float | None = None     # [N/m] particle contact stiffness (None: the route's, the material's)
    contact_kd: float | None = None     # particle contact damping (None: Newton's 10)


FINE_TEE = TeeRig(cell=0.01, contact_ke=2.0e4, contact_kd=0.0)      # the rig whose fold stays (module docstring)


@dataclass(frozen=True)
class TeeVerdict:
    success: bool                       # every gate on the settled shirt
    gates: dict[str, bool]
    metrics: dict[str, float]
    release_gates: dict[str, bool]      # the same gates as the hands let go
    release_metrics: dict[str, float]
    frames: list[tuple[float, int]]
    particles: int
    wall_s: float


@dataclass
class _SideFlap:
    """A cuff swung round the line x = ``x_fold`` (along y) onto the body. The grasp turns rigidly: each point's
    offset (r, dz) from the fold line and the grasp's mean height turns with it, so the two panels stay one h apart
    and come down turned over. Only the height of the swing is scaled (``apex``)."""
    rows: np.ndarray
    start: np.ndarray
    x_fold: float
    z_land: float
    apex: float

    def at(self, s: float) -> np.ndarray:
        th = math.pi * min(max(s, 0.0), 1.0)
        r = self.start[:, 0] - self.x_fold
        zc = float(self.start[:, 2].mean())
        dz = self.start[:, 2] - zc
        out = self.start.copy()
        out[:, 0] = self.x_fold + r * math.cos(th) - np.sign(r) * dz * math.sin(th)
        out[:, 2] = (zc + self.apex * np.abs(r) * math.sin(th) + (self.z_land - zc) * (1.0 - math.cos(th)) / 2.0
                     + dz * math.cos(th))
        return out


def build(rig: TeeRig, workdir: Path, device: str = "cpu"):
    """Table + T-shirt through the card route: (model, VBD kwargs, dt, h, Tee)."""
    from alloy.cards import types as ct
    from alloy.cards.library import CardLibrary
    from alloy.core.registry import isolated_registry
    from alloy.testing import integration as I

    base = CardLibrary().object(CARD)
    h = float(base.features.thickness.value)
    tee = make_tee(h, rig.cell, s=TeeShape(sleeve=rig.sleeve))
    path = workdir / f"{NAME}.obj"
    sha = write_obj(tee, path)
    card = dataclasses.replace(base, name=NAME, shape=ct.MeshSource(str(path), sha),
                               features=dataclasses.replace(base.features, self_contact=True),
                               budget=dataclasses.replace(base.budget, particles=len(tee.vertices)))
    mats = {}
    if rig.material is not None:
        card, mats = dataclasses.replace(card, material=rig.material.name), {rig.material.name: rig.material}
    y_mid = 0.5 * (tee.vertices[:, 1].min() + tee.vertices[:, 1].max())
    scene = ct.SceneCard(robots=(), objects=(ct.Placement(TABLE, (0.0, 0.0, -TABLE_HALF_Z, 0.0, 0.0, 0.0, 1.0)),
                                             ct.Placement(NAME, (0.0, -y_mid, 0.0, 0.0, 0.0, 0.0, 1.0))))
    with isolated_registry(empty=True):
        built = I.assemble(scene, _CardOverride({NAME: card}, mats), device, 1)
    (vbd,) = [e for e in built.entries if e.capability == "vbd"]
    kwargs = {k: v for k, v in {**dict(vbd.solver_kwargs), **rig.vbd}.items() if v is not None}
    if rig.contact_ke is not None:          # VBD reads both off the model: cloth on cloth and cloth on the table
        built.model.soft_contact_ke = float(rig.contact_ke)
    if rig.contact_kd is not None:
        built.model.soft_contact_kd = float(rig.contact_kd)
    return built.model, kwargs, float(vbd.target_dt), h, tee


def _plan(q0: np.ndarray, tee: Tee, rig: TeeRig, h: float) -> list[tuple[np.ndarray, float, float]]:
    """(cuff grasp ids, fold line x, landing height of the grasp's middle) per sleeve, left then right."""
    w = 0.5 * tee.shape.width
    z_land = float(q0[:, 2].max()) + (rig.landing_gap + 1.0) * h
    return [(tee.cuffs[0], -w + rig.fold_in, z_land), (tee.cuffs[1], w - rig.fold_in, z_land)]


class _Run:
    """The stepping loop with the crossing log."""

    def __init__(self, st: _Stepper, tri: np.ndarray, every: float) -> None:
        self.st, self.tri, self.log, self.k = st, tri, [], 0
        self.every = max(int(round(every / st.dt)), 1)

    def steps(self, n: int, drive=None) -> None:
        for j in range(1, n + 1):
            self.st.step(None if drive is None else drive(j / n))
            self.k += 1
            if self.k % self.every == 0:
                self.log.append((round(self.k * self.st.dt, 6), _count(self.st.s0.particle_q.numpy(), self.tri)))


def simulate(rig: TeeRig, workdir: Path, device: str = "cpu") -> dict[str, Any]:
    """Both cuffs swing at once (the pulls on the body cancel), hold, let go; the shirt settles."""
    model, kwargs, dt, h, tee = build(rig, workdir, device)
    q0, tri = model.particle_q.numpy(), model.tri_indices.numpy()
    mass0 = model.particle_mass.numpy().copy()
    plan = _plan(q0, tee, rig, h)
    rows = np.concatenate([r for r, _, _ in plan])
    _set_mass(model, rows, np.zeros(len(rows)))
    run = _Run(_Stepper(model, kwargs, dt), tri, rig.every)
    flaps = [_SideFlap(r, run.st.s0.particle_q.numpy()[r], x, z, rig.apex) for r, x, z in plan]
    run.steps(int(round(rig.fold_s / dt)), lambda s: (rows, np.concatenate([f.at(s) for f in flaps])))
    landed = np.concatenate([f.at(1.0) for f in flaps])
    run.steps(int(round(rig.hold_s / dt)), lambda s: (rows, landed))
    q_release = run.st.s0.particle_q.numpy().copy()        # a CPU numpy() is a view of the state's buffer
    _set_mass(model, rows, mass0[rows])
    run.steps(int(round(rig.settle_s / dt)))
    return {"q0": q0, "q": run.st.s0.particle_q.numpy(), "q_release": q_release, "tri": tri, "h": h, "log": run.log,
            "tee": tee, "plan": plan}


def _sleeve_gates(f: dict[str, Any], q: np.ndarray, side: str, x_fold: float) -> tuple[dict[str, bool],
                                                                                     dict[str, float]]:
    q0, tee = f["q0"], f["tee"]
    body = np.flatnonzero(tee.region == 0)
    sl = np.flatnonzero(tee.region == (1 if side == "left" else 2))
    eq = ConvexHull(q[body, :2]).equations
    inside = float(np.all(q[sl, :2] @ eq[:, :2].T + eq[:, 2] <= 0.015, axis=1).mean())
    goal = q0[sl, :2].copy()
    goal[:, 0] = 2.0 * x_fold - goal[:, 0]
    rms = float(np.sqrt(np.mean(np.sum((q[sl, :2] - goal) ** 2, axis=1))))
    _, nn = cKDTree(q[body, :2]).query(q[sl, :2])
    gap = float(np.percentile(q[sl, 2] - q[body[nn], 2], 90))
    return ({f"{side}_inside": inside >= 0.90, f"{side}_rms": rms <= 0.045, f"{side}_gap": gap <= 0.025},
            {f"{side}_inside": inside, f"{side}_rms": rms, f"{side}_gap_p90": gap})


def gates(f: dict[str, Any], q: np.ndarray | None = None) -> tuple[dict[str, bool], dict[str, float]]:
    """The task's gates on ``q`` (default: the settled shirt); crossings are counted on ``q`` itself."""
    from alloy.tasks.laundry_fold import stretch

    q = f["q"] if q is None else q
    g, m = {}, {}
    for side, (_, x_fold, _) in zip(("left", "right"), f["plan"]):
        gs, ms = _sleeve_gates(f, q, side, x_fold)
        g.update(gs)
        m.update(ms)
    end = _count(q, f["tri"])
    st = stretch(f["q0"], q, f["tri"])
    m.update(crossings_end=end, crossings_max=max([c for _, c in f["log"]] + [end]), stretch=st)
    g.update(no_crossing_end=end == 0, stretch=st <= 0.30, finite=bool(np.isfinite(q).all()))
    return g, m


def run(rig: TeeRig = TeeRig(), device: str = "cpu", save_q: str | None = None) -> TeeVerdict:
    t0 = time.time()
    with tempfile.TemporaryDirectory() as d:
        f = simulate(rig, Path(d), device)
    if save_q:
        np.savez(save_q, q0=f["q0"], q=f["q"], tri=f["tri"], region=f["tee"].region)
    g, m = gates(f)
    gr, mr = gates(f, f["q_release"])
    return TeeVerdict(bool(all(g.values())), {k: bool(v) for k, v in g.items()}, m,
                      {k: bool(v) for k, v in gr.items()}, mr, f["log"], len(f["q0"]), round(time.time() - t0, 1))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--rig", default="{}", help="JSON TeeRig fields")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None, help="append the verdict as a JSON line")
    p.add_argument("--save-q", default=None, help="npz of the start / end positions")
    a = p.parse_args(argv)
    rig = TeeRig(**json.loads(a.rig))
    row = json.dumps({"rig": dataclasses.asdict(rig), **dataclasses.asdict(run(rig, a.device, a.save_q))},
                     default=float)
    print(row)
    if a.out:
        with open(a.out, "a") as fh:
            fh.write(row + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
