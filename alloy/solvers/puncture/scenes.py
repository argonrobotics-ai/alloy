# SPDX-License-Identifier: Apache-2.0
"""Small puncture scenes for laws, tests and benches: a silicone block (pinned base) and a straight kinematic needle.

``Block`` holds the block, needle and material; ``build_block`` returns (model, NeedleSpec); ``Drive`` steps a
SolverPuncture with a prescribed needle height per step and records one row per step (``ROW`` names the columns).
Material numbers here are the law defaults (design note §7); clips set their own.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import warp as wp

from alloy._vendor import newton

from .config import NeedleSpec, TissueMaterial, set_tet_material
from .thread import isolate_thread_colours

ROW = ("t", "tip_depth", "axial", "fax", "track", "anchors", "offset", "flags", "punctures", "length", "net", "E",
       "W", "D", "R")


@dataclass(frozen=True)
class Thread:
    """Suture string: particles every ``spacing``; EA and EI go to the solver's chain solve."""

    count: int = 20
    spacing: float = 0.5e-3
    diameter: float = 0.22e-3
    density: float = 1140.0       # nylon 6/6 (AmesWeb 1.135 g/cm3; design note §7)
    EA: float = 76.0              # [N]
    damping: float = 0.0
    flen: float = 50.0            # thread-tissue friction per length [N/m]
    EI: float = 3.4e-7            # [N·m²] (solid nylon rod E 3 GPa at 0.22 mm; a braid is limper)
    stopper: bool = False


@dataclass(frozen=True)
class Block:
    size: float = 8e-3            # x, y extent [m]
    thick: float = 6e-3           # z extent [m]
    cell: float = 1e-3
    E: float = 2.6e5              # [Pa]
    nu: float = 0.45
    rho: float = 1070.0
    damp: float = 0.0             # VBD tet damping [Pa·s]
    gravity: float = 9.81
    material: TissueMaterial = TissueMaterial(fp=1.0, fcut=0.2, flen=50.0)
    gap: float = 0.5e-3           # needle tip height above the top face at t = 0
    x_off: float = 0.3e-3         # needle axis offset from the block centre (off the grid lines)
    wire_radius: float = 0.35e-3
    fix_sides: bool = False       # pin the side faces too (for through-and-out insertions; the base is open then)
    needle_length: float = 0.03
    from_below: bool = False      # needle under the block pointing up (the thread hangs below the swage)
    thread: Thread | None = None  # a suture string from the swage, straight up along the needle axis


    def lame(self) -> tuple[float, float]:
        return self.E / (2 * (1 + self.nu)), self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))


def build_block(bl: Block, worlds: int = 1, extra: Callable[[newton.ModelBuilder], None] | None = None
                ) -> tuple[newton.Model, list[NeedleSpec]]:
    """Block of ``worlds`` copies (one world each) with one straight needle per world, tip down along -z.
    ``extra(builder)`` adds more to each world after the needle and thread (e.g. an obstacle body)."""
    tpl = newton.ModelBuilder(gravity=(0.0, 0.0, -bl.gravity))
    n, nz = round(bl.size / bl.cell), round(bl.thick / bl.cell)
    mu, lam = bl.lame()
    tpl.add_soft_grid(pos=wp.vec3(-0.5 * bl.size, -0.5 * bl.size, 0.0), rot=wp.quat_identity(), vel=wp.vec3(0.0),
                      dim_x=n, dim_y=n, dim_z=nz, cell_x=bl.cell, cell_y=bl.cell, cell_z=bl.cell, density=bl.rho,
                      k_mu=mu, k_lambda=lam, k_damp=bl.damp, fix_bottom=not bl.fix_sides, fix_left=bl.fix_sides,
                      fix_right=bl.fix_sides, particle_radius=0.25 * bl.cell)
    set_tet_material(tpl, range(len(tpl.tet_indices)), bl.material)
    up = -1.0 if bl.from_below else 1.0          # +z: the needle hangs tip-down above the block
    z_tip = -bl.gap if bl.from_below else bl.thick + bl.gap
    body = tpl.add_body(xform=wp.transform(wp.vec3(bl.x_off, 0.0, z_tip), wp.quat_identity()),
                        mass=0.0, is_kinematic=True)
    t0 = len(tpl.particle_q)
    if bl.thread is not None:
        _add_thread(tpl, bl.thread, (bl.x_off, 0.0, z_tip + up * bl.needle_length), up)
    if extra is not None:
        extra(tpl)
    if worlds == 1:
        b = tpl
    else:
        b = newton.ModelBuilder(gravity=(0.0, 0.0, -bl.gravity))
        b.replicate(tpl, worlds)
    b.color()
    if bl.thread is not None:
        per_p = len(b.particle_q) // max(worlds, 1)
        isolate_thread_colours(b, [(t0 + w * per_p, bl.thread.count) for w in range(worlds)])
    m = b.finalize()
    per = m.body_count // max(worlds, 1)
    pper = m.particle_count // max(worlds, 1)
    th = bl.thread
    specs = [NeedleSpec(body=body + w * per, length=bl.needle_length, wire_radius=bl.wire_radius,
                        origin=(0.0, 0.0, 0.0), e1=(0.0, 0.0, up), world=w,
                        thread_start=t0 + w * pper if th else -1, thread_count=th.count if th else 0,
                        stopper=bool(th and th.stopper), thread_flen=th.flen if th else 0.0,
                        thread_ei=th.EI if th else 0.0, thread_ea=th.EA if th else 0.0) for w in range(worlds)]
    return m, specs


def _add_thread(b: newton.ModelBuilder, th: Thread, swage: tuple[float, float, float], up: float) -> None:
    """Particles from the swage straight on along the needle axis (``up`` = +1: upward); stretch and bending are the
    solver's chain solve, particle 0 is tied to the swage by its SWAGE anchor."""
    m = th.density * np.pi * (0.5 * th.diameter) ** 2 * th.spacing
    first = len(b.particle_q)
    for k in range(th.count):
        b.add_particle(pos=wp.vec3(swage[0], swage[1], swage[2] + up * k * th.spacing), vel=wp.vec3(0.0), mass=m,
                       radius=0.5 * th.diameter)



class Drive:
    """Steps ``solver`` with every needle advanced depth_of(t) along its tip axis from its start (z0 − depth, or
    z0 + depth with ``up_axis``; one-way: poses set before the step)."""

    def __init__(self, model: newton.Model, solver, dt: float, up_axis: bool = False) -> None:
        self.m, self.s, self.dt = model, solver, dt
        self.sign = 1.0 if up_axis else -1.0
        self.s0, self.s1 = model.state(), model.state()
        self.X = self.s0.body_q.numpy().copy()
        self.z0 = self.X[:, 2].copy()
        self.k = 0
        self.rows: list[list[float]] = []

    def _pose(self, st, depth: float) -> None:
        q = self.X.copy()
        q[:, 2] = self.z0 + self.sign * depth
        st.body_q.assign(q)

    def step(self, depth_of: Callable[[float], float], gap: float, record: bool = True) -> None:
        t0, t1 = self.k * self.dt, (self.k + 1) * self.dt
        self._pose(self.s0, depth_of(t0))
        self._pose(self.s1, depth_of(t1))
        self.s.step(self.s0, self.s1, None, None, self.dt)
        self.s0, self.s1 = self.s1, self.s0
        self.k += 1
        if record:
            self.rows.append(self.row(t1, depth_of(t1) - gap))

    def row(self, t: float, tip_depth: float, n: int = 0) -> list[float]:
        s, ld, tr = self.s, self.s.loads, self.s.tracks
        lg = s.ledger.lg if s.ledger is not None else None
        led = [float(lg.net_prev.numpy()[0] - lg.net0.numpy()[0])] + [float(a.numpy()[0]) for a in
                                                                       (lg.e, lg.w, lg.d, lg.r)] if lg else [0.0] * 5
        return [t, tip_depth, float(ld.axial.numpy()[n]), float(tr.fax.numpy()[n]), int(tr.state.numpy()[n]),
                int(ld.anchors.numpy()[n]), float(ld.offset.numpy()[0]), int(tr.flags.numpy()[n]),
                int(tr.punctures.numpy()[n]), float(ld.length.numpy()[n])] + led

    def table(self) -> np.ndarray:
        return np.asarray(self.rows, dtype=np.float64)
