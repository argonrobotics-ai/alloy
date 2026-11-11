# SPDX-License-Identifier: Apache-2.0
"""Meshes and a small stepping rig for the film laws (laws.py) and spikes: no router, no coupler.

``FilmRig`` builds one or more identical worlds from a ``build(b)`` callback (film meshes via ``add_film_mesh``,
pins via ``pin``), steps ``SolverFilm`` at a fixed dt, and keeps the per-world ledger. Everything is CPU-small by
default; the laws pass ``device="cuda:0"`` and full sizes in the slow/GPU tests.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import warp as wp

from alloy._vendor import newton

from .config import FilmConfig
from .solver import SolverFilm


def grid_mesh(nx: int, ny: int, dx: float, plane: str = "xy") -> tuple[np.ndarray, np.ndarray]:
    """(nx+1)·(ny+1) vertices, alternating-diagonal triangles (no preferred shear direction). ``plane`` "xy" lies
    flat (x along nx, y along ny); "xz" hangs (x along nx, −z along ny)."""
    xs, ys = np.meshgrid(np.arange(nx + 1) * dx, np.arange(ny + 1) * dx, indexing="ij")
    z = np.zeros(xs.size)
    v = np.stack([xs.ravel(), ys.ravel(), z], 1) if plane == "xy" else np.stack([xs.ravel(), z, -ys.ravel()], 1)
    idx = lambda i, j: i * (ny + 1) + j  # noqa: E731
    t = []
    for i in range(nx):
        for j in range(ny):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            t += [(a, b, c), (a, c, d)] if (i + j) % 2 == 0 else [(a, b, d), (b, c, d)]
    return v, np.asarray(t, np.int64)


def frustum_mesh(r_bottom: float, r_top: float, height: float, n_theta: int = 48, n_rings: int = 6,
                 n_wall: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Open-top frustum (a basket coffee filter): a bottom disk of ``n_rings`` rings plus a wall of ``n_wall``
    rows, ``n_theta`` around; bottom at z = 0."""
    verts = [np.zeros(3)]
    rings = []
    for k in range(1, n_rings + 1):
        r = r_bottom * k / n_rings
        rings.append(len(verts))
        verts += [np.array([r * math.cos(a), r * math.sin(a), 0.0]) for a in np.arange(n_theta) * 2 * math.pi / n_theta]
    for k in range(1, n_wall + 1):
        r = r_bottom + (r_top - r_bottom) * k / n_wall
        rings.append(len(verts))
        z = height * k / n_wall
        verts += [np.array([r * math.cos(a), r * math.sin(a), z]) for a in np.arange(n_theta) * 2 * math.pi / n_theta]
    tris = [(0, rings[0] + i, rings[0] + (i + 1) % n_theta) for i in range(n_theta)]
    for a, b in zip(rings[:-1], rings[1:]):
        for i in range(n_theta):
            i1 = (i + 1) % n_theta
            tris += [(a + i, b + i, b + i1), (a + i, b + i1, a + i1)]
    return np.asarray(verts), np.asarray(tris, np.int64)


@dataclass
class FilmRig:
    build: Callable[[newton.ModelBuilder], dict]          # adds one world's content; returns named particle ids
    worlds: int = 1
    device: str = "cpu"
    dt: float = 1.0e-3
    iterations: int = 10
    film: FilmConfig = field(default_factory=FilmConfig)
    gravity: float = -9.81
    ground: bool = False                                   # a ground plane at z = 0 (collision pipeline contacts)
    vbd: dict = field(default_factory=dict)                # extra SolverVBD kwargs (self contact, friction, ...)

    def __post_init__(self) -> None:
        b = newton.ModelBuilder(gravity=(0.0, 0.0, float(self.gravity)))
        from .config import register_custom_attributes

        register_custom_attributes(b)
        self.names: list[dict] = []
        for _ in range(self.worlds):
            b.begin_world()
            self.names.append({k: np.asarray(v) for k, v in self.build(b).items()})
            b.end_world()
        if self.ground:
            b.add_ground_plane()
        b.color()
        self.model = b.finalize(device=self.device)
        self.solver = SolverFilm(self.model, iterations=self.iterations, film=self.film, **self.vbd)
        self.s0, self.s1 = self.model.state(), self.model.state()
        self.pipe = newton.CollisionPipeline(self.model) if self.ground else None
        self.contacts = self.pipe.contacts() if self.pipe is not None else None
        self.t = 0.0

    def step(self, n: int = 1, before: Callable[[FilmRig], None] | None = None) -> None:
        for _ in range(n):
            self.s0.clear_forces()
            if before is not None:
                before(self)
            if self.pipe is not None:
                self.pipe.collide(self.s0, self.contacts)
            self.solver.step(self.s0, self.s1, None, self.contacts, self.dt)
            self.s0, self.s1 = self.s1, self.s0
            self.t += self.dt

    def run(self, seconds: float, every: float | None = None, probe: Callable[[FilmRig], object] | None = None,
            before: Callable[[FilmRig], None] | None = None) -> list:
        """Steps ``seconds``; with ``every`` and ``probe``, returns probe results every ``every`` seconds."""
        out = []
        n = int(round(seconds / self.dt))
        stride = max(1, int(round(every / self.dt))) if every else n + 1
        for k in range(n):
            self.step(1, before)
            if probe is not None and (k + 1) % stride == 0:
                out.append(probe(self))
        return out

    def q(self) -> np.ndarray:
        return self.s0.particle_q.numpy()

    def qd(self) -> np.ndarray:
        return self.s0.particle_qd.numpy()


def pin(b: newton.ModelBuilder, ids) -> None:
    """Make particles kinematic (mass 0, inactive flag)."""
    for p in ids:
        b.particle_mass[int(p)] = 0.0
        b.particle_flags[int(p)] = 0
