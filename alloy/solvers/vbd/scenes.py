# SPDX-License-Identifier: Apache-2.0
"""Small check scenes for the VBD coarse step (tests/solvers/vbd and the SCR-WP11-2 acceptance runs).

- `cap_load`: a stiff spherical cap in zero gravity, a constant force on its lowest third (A17's worst case);
- `sheet_rest`: a flat sheet resting on the ground under gravity (the ruling's "stays at rest" check);
- `stack`: a smaller sheet resting on a fully pinned sheet 5 cm above the ground, self-contact on: its only support
  is a self-contact pair with another component (out of the ground's contact margin);
- `pinned`: a sheet hanging from one pinned edge row under gravity (a pinned component keeps the stock path);
- `drop`: a flat sheet falling from 2 cm onto the ground (free fall, then contact).
`worlds` copies the scene side by side (one component per sheet per copy).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

from alloy._vendor import newton

DENSITY, MEMBRANE_KE, BEND_KE = 0.0143, 40.0, 1.0e-4        # 15 µm film (A17), stiff membrane regime
RADIUS = 5.0e-4
STACK_Z = 0.05
FORCE = 1.0e-3
KINDS = ("cap_load", "sheet_rest", "stack", "pinned", "drop")


@dataclass
class CheckScene:
    kind: str
    model: newton.Model
    pipeline: newton.CollisionPipeline | None
    force: wp.array | None                  # copied into state.particle_f before every step
    sheets: list[tuple[int, int]]           # particle ranges, one per sheet
    self_contact: bool = False


def grid(n: int = 12, cell: float = 5.0e-3) -> tuple[np.ndarray, np.ndarray]:
    """Flat n x n cell grid in z = 0, alternating diagonals, centred on the origin."""
    xs = (np.arange(n + 1) - 0.5 * n) * cell
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    v = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    t = []
    for i in range(n):
        for j in range(n):
            a, b, c, d = i * (n + 1) + j, (i + 1) * (n + 1) + j, (i + 1) * (n + 1) + j + 1, i * (n + 1) + j + 1
            t += [[a, b, c], [a, c, d]] if (i + j) % 2 == 0 else [[a, b, d], [b, c, d]]
    return v, np.asarray(t)


def cap(rings: int = 8, n_theta: int = 24, radius: float = 0.04, angle: float = 1.2) -> tuple[np.ndarray, np.ndarray]:
    """Spherical cap, pole at the bottom, staggered rings."""
    v = [np.array([0.0, 0.0, -radius])]
    starts = []
    for k in range(1, rings + 1):
        ph, starts = angle * k / rings, starts + [len(v)]
        for i in range(n_theta):
            th = 2.0 * math.pi * (i + 0.5 * (k % 2)) / n_theta
            v.append(radius * np.array([math.sin(ph) * math.cos(th), math.sin(ph) * math.sin(th), -math.cos(ph)]))
    t = [[0, starts[0] + (i + 1) % n_theta, starts[0] + i] for i in range(n_theta)]
    for a, b in zip(starts[:-1], starts[1:]):
        for i in range(n_theta):
            i1 = (i + 1) % n_theta
            t += [[a + i, a + i1, b + i1], [a + i, b + i1, b + i]]
    return np.asarray(v), np.asarray(t)


def _add_sheet(b: newton.ModelBuilder, v: np.ndarray, t: np.ndarray, offset, bend: float = BEND_KE) -> tuple[int, int]:
    p0 = b.particle_count
    b.add_cloth_mesh(pos=wp.vec3(*offset), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(), vertices=v.astype(np.float32),
                     indices=t.reshape(-1).tolist(), density=DENSITY, tri_ke=MEMBRANE_KE, tri_ka=0.0, tri_kd=0.0,
                     edge_ke=bend, edge_kd=0.0, particle_radius=RADIUS)
    return p0, b.particle_count


def _populate(b: newton.ModelBuilder, kind: str, x0: float) -> list[tuple[int, int]]:
    v, t = grid()
    if kind == "cap_load":
        cv, ct = cap()
        return [_add_sheet(b, cv, ct, (x0, 0.0, 0.0), bend=0.0)]
    if kind == "stack":
        return [_add_sheet(b, v, t, (x0, 0.0, STACK_Z)), _add_sheet(b, v * 0.8, t, (x0, 0.0, STACK_Z + RADIUS))]
    z = {"sheet_rest": RADIUS, "pinned": 0.05, "drop": 0.02}[kind]
    return [_add_sheet(b, v, t, (x0, 0.0, z))]


def build(kind: str, device: str = "cpu", worlds: int = 1) -> CheckScene:
    """One of `KINDS`, `worlds` copies 10 cm apart along x."""
    if kind not in KINDS:
        raise ValueError(f"unknown check scene {kind!r}; known: {KINDS}")
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0) if kind == "cap_load" else (0.0, 0.0, -9.81))
    if kind != "cap_load":
        b.add_ground_plane()
    sheets = [s for w in range(worlds) for s in _populate(b, kind, 0.1 * w)]
    _pin(b, kind, sheets)
    b.color()
    model = b.finalize(device=device)
    model.soft_contact_ke, model.soft_contact_kd, model.soft_contact_mu = 1.0e2, 1.0e-3, 0.5
    pipe = None if kind == "cap_load" else newton.CollisionPipeline(model)
    return CheckScene(kind, model, pipe, _load(model, sheets) if kind == "cap_load" else None, sheets,
                      self_contact=kind == "stack")


def _pin(b: newton.ModelBuilder, kind: str, sheets: list[tuple[int, int]]) -> None:
    """`pinned`: one edge row of each sheet; `stack`: the whole lower sheet of each copy (m = 0)."""
    q = np.asarray(b.particle_q)
    if kind == "pinned":
        rows = [p0 + np.flatnonzero(np.isclose(q[p0:p1, 0], q[p0:p1, 0].min())) for p0, p1 in sheets]
    elif kind == "stack":
        rows = [np.arange(p0, p1) for p0, p1 in sheets[0::2]]
    else:
        return
    for i in np.concatenate(rows):
        b.particle_mass[int(i)] = 0.0


def _load(model: newton.Model, sheets: list[tuple[int, int]]) -> wp.array:
    q = model.particle_q.numpy()
    f = np.zeros((model.particle_count, 3), np.float32)
    for p0, p1 in sheets:
        z = q[p0:p1, 2]
        sel = p0 + np.flatnonzero(z < z.min() + np.ptp(z) / 3.0)
        f[sel, 2] = FORCE / len(sel)
    return wp.array(f, dtype=wp.vec3, device=model.device)


def solver_kwargs(scene: CheckScene, iterations: int = 10) -> dict:
    """SolverVBD kwargs for the scene (self-contact radii sized to the particle radius for `stack`)."""
    kw: dict = {"iterations": iterations}
    if scene.self_contact:
        kw.update(particle_enable_self_contact=True, particle_self_contact_radius=RADIUS,
                  particle_self_contact_margin=2.0 * RADIUS)
    return kw


def run(scene: CheckScene, solver: newton.solvers.SolverVBD, steps: int, dt: float = 1.0e-3) -> newton.State:
    """Step `steps` times (collide, force, step); returns the last state."""
    m = scene.model
    s0, s1, ctrl = m.state(), m.state(), m.control()
    contacts = scene.pipeline.contacts() if scene.pipeline is not None else None
    for _ in range(steps):
        s0.clear_forces()
        if scene.force is not None:
            wp.copy(s0.particle_f, scene.force)
        if scene.pipeline is not None:
            scene.pipeline.collide(s0, contacts)
        solver.step(s0, s1, ctrl, contacts, dt)
        s0, s1 = s1, s0
    return s0


def momentum(scene: CheckScene, state: newton.State) -> np.ndarray:
    """Linear momentum per sheet, [S, 3]."""
    m, qd = scene.model.particle_mass.numpy(), state.particle_qd.numpy()
    return np.array([(m[a:b, None] * qd[a:b]).sum(0) for a, b in scene.sheets])


__all__ = ["KINDS", "CheckScene", "build", "cap", "grid", "momentum", "run", "solver_kwargs"]
