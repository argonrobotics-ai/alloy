# SPDX-License-Identifier: Apache-2.0
"""Scenes for the SPH-VBD coupler (engine item 3, soft_walls.py): soft tet boxes in a pool, the VBD <-> SPH Scheduler.

Newton's ``add_soft_grid`` lumps a full cell of mass on every vertex (an n^3-cell grid weighs (n+1)^3 cells) and
defaults the particle radius to 0.1 m, which makes VBD's particle-shape contacts touch every wall within 10 cm;
``add_soft_cube`` corrects both, so the cube weighs rho V and collides at ``radius``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from alloy._vendor import newton
import numpy as np
import warp as wp

from .build import add_box_tank, add_fluid_points, lattice_points
from .config import DFSPHConfig, register_custom_attributes


def lame(E: float, nu: float) -> tuple[float, float]:
    """(mu, lambda) of Young's modulus E [Pa] and Poisson's ratio nu."""
    return E / (2.0 * (1.0 + nu)), E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))


def add_soft_box(b: newton.ModelBuilder, lo, size, cells, rho: float, *, E: float = 5.0e4, nu: float = 0.45,
                 radius: float = 0.5e-3, k_damp: float = 0.0) -> list[int]:
    """A tet-grid box (corner ``lo``, ``size`` [m] and ``cells`` per axis) of density rho; returns its particle ids."""
    mu, lam = lame(E, nu)
    n = [int(c) for c in cells]
    cell = [float(size[k]) / n[k] for k in range(3)]
    lump = n[0] * n[1] * n[2] / ((n[0] + 1) * (n[1] + 1) * (n[2] + 1))  # Newton puts a full cell on every vertex
    p0 = b.particle_count
    b.add_soft_grid(pos=wp.vec3(*map(float, lo)), rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0), dim_x=n[0],
                    dim_y=n[1], dim_z=n[2], cell_x=cell[0], cell_y=cell[1], cell_z=cell[2], density=rho * lump,
                    k_mu=mu, k_lambda=lam, k_damp=k_damp, particle_radius=radius)
    return list(range(p0, b.particle_count))


def add_soft_cube(b: newton.ModelBuilder, lo, edge: float, ncell: int, rho: float, **kw) -> list[int]:
    """A tet-grid cube: ``add_soft_box`` with equal sides."""
    return add_soft_box(b, lo, (edge, edge, edge), (ncell, ncell, ncell), rho, **kw)


@dataclass
class SoftPool:
    model: newton.Model
    config: DFSPHConfig
    liquid: range
    soft: list[int] = field(default_factory=list)  # all particles of the soft cube
    surface: list[int] = field(default_factory=list)  # its surface vertices (the proxies)
    shape: int = -1  # the rigid twin's box shape
    lo: np.ndarray = None  # box corner [m]
    size: np.ndarray = None  # box sides [m]

    @property
    def edge(self) -> float:
        return float(self.size[0])


def cube_pool(mode: str = "soft", *, n: int = 16, depth: int = 24, cube_cells=6, cube_at=(5, 5, 9),
              rho_s: float = 1000.0, ncell=3, cfg: DFSPHConfig | None = None, device: str | None = None,
              E: float = 5.0e4) -> SoftPool:
    """A pool n x n x depth lattice cells with a box (``cube_cells`` lattice cells per side, an int for a cube, corner
    at lattice cell ``cube_at``) as a soft tet grid (mode "soft", ``ncell`` tet cells per side) or a static box shape
    ("rigid"); liquid cells inside it are left out. A box reaching above the pool is a floater."""
    cfg = cfg or DFSPHConfig(particle_radius=1.0e-3)
    s = cfg.spacing
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    register_custom_attributes(b)
    b.begin_world()
    add_box_tank(b, (0.0, 0.0, 0.0), (n * s, n * s, (depth + 2 * float(np.max(cube_cells))) * s), wall=cfg.h)
    lo = np.asarray(cube_at, float) * s
    size = np.broadcast_to(np.asarray(cube_cells, float), (3,)) * s
    cells = np.broadcast_to(np.asarray(ncell, int), (3,))
    pts = lattice_points((0.0, 0.0, 0.0), (n, n, depth), s)
    inside = np.all((pts > lo) & (pts < lo + size), axis=1)
    liq = add_fluid_points(b, cfg, pts[~inside])
    out = SoftPool(None, cfg, liq, lo=lo, size=size)
    if mode == "soft":
        out.soft = add_soft_box(b, lo, size, cells, rho_s, E=E, radius=0.25 * s)
    else:
        c = lo + 0.5 * size
        out.shape = b.add_shape_box(-1, xform=wp.transform(wp.vec3(*map(float, c)), wp.quat_identity()),
                                    hx=size[0] / 2, hy=size[1] / 2, hz=size[2] / 2)
    b.end_world()
    if mode == "soft":
        b.color()
    out.model = b.finalize(device=device)
    if mode == "soft":
        tri = out.model.tri_indices.numpy()
        out.surface = sorted({int(v) for t in tri for v in t})
    return out


def vbd_sph_scheduler(pool: SoftPool, *, mode: str = "staggered", sph_rung: int = 2, vbd_iterations: int = 20,
                      added_mass: float | None = None, policy=None, sph_kwargs: dict | None = None):
    """Two-entry Scheduler: VBD owns the soft cube, SPH the liquid, the cube's surface vertices are particle proxies
    VBD -> SPH (soft walls). VBD is the `vbd` capability's solver (``make_vbd``: SolverVBDMomentum, whose coarse step
    keeps the external impulse exact; stock SolverVBD gains momentum, SCR-WP11-2). Staggered pairs relax with Aitken
    (WP4: the strong mode for a fluid pair). ``sph_kwargs`` go to SolverDFSPH (soft_inertia, soft_half_thickness)."""
    from alloy.core.newton_compat import SolverCoupledProxy
    from alloy.solvers.vbd import make_vbd
    from alloy.coupling.scheduler import Scheduler
    from alloy.coupling.types import AlloyEntry, PairPolicy, SchedulerPolicy

    from .solver import SolverDFSPH

    cfg = pool.config
    e_v = SolverCoupledProxy.Entry(name="vbd", particles=pool.soft,
                                   solver=lambda v: make_vbd(v, {"iterations": vbd_iterations}))
    e_s = SolverCoupledProxy.Entry(name="sph", particles=list(pool.liquid), substeps=sph_rung,
                                   solver=lambda v: SolverDFSPH(v, cfg, **(sph_kwargs or {})))
    relax = "aitken" if mode == "staggered" else "fixed"
    px = SolverCoupledProxy.Proxy(source="vbd", destination="sph", particles=pool.surface, mode=mode,
                                  proxy_relaxation_mode=relax)
    return Scheduler(pool.model, [AlloyEntry(e_v, adaptive=False, capability="vbd"),
                                  AlloyEntry(e_s, initial_rung=sph_rung, adaptive=False, capability="sph")],
                     [PairPolicy("vbd", "sph", mode=mode, proxy=px, added_mass=added_mass)],
                     policy or SchedulerPolicy())


def box_frame(rest: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Best-fit rigid frame of a soft body (Kabsch): (R, c, c0) with x ~ R (rest - c0) + c."""
    c0, c = rest.mean(axis=0), x.mean(axis=0)
    H = (rest - c0).T @ (x - c)
    U, _S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, c, c0


def inside_box(q: np.ndarray, rest: np.ndarray, x: np.ndarray, shrink: float = 0.01) -> int:
    """Points inside a soft box (rest vertex positions ``rest``, current ``x``), in its best-fit frame, the rest box
    shrunk by ``shrink`` of each side."""
    R, c, c0 = box_frame(rest, x)
    local = (q - c) @ R + c0
    lo, hi = rest.min(axis=0), rest.max(axis=0)
    m = shrink * (hi - lo)
    return int(np.all((local > lo + m) & (local < hi - m), axis=1).sum())


def tilt_deg(rest: np.ndarray, x: np.ndarray) -> float:
    """Angle [deg] between the body's z axis and the world's z (its best-fit frame)."""
    R, _c, _c0 = box_frame(rest, x)
    return float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))


SOFT_CONTACT_GAP = 1.0e-3  # [m] about two soft-vertex radii


def soft_pipeline(model: newton.Model, gap: float = SOFT_CONTACT_GAP) -> newton.CollisionPipeline:
    """Collision pipeline for the soft bodies' wall contacts with a small gap. Newton's default 1 cm gap gives
    records to every soft vertex within 1 cm of a wall, and SolverVBDMomentum skips its exact momentum step for a
    component with any record (stock VBD then gains momentum: a floating box read Fz 1.074 W, draft 1.077)."""
    return newton.CollisionPipeline(model, soft_contact_gap=gap)


def elastic_energy(model: newton.Model, q: np.ndarray) -> float:
    """VBD's stable neo-Hookean energy [J] of every tet at positions q, zero at rest (Newton's
    evaluate_volumetric_neo_hookean_force_and_hessian: psi = mu/2 (I_C - 3) + lam'/2 (J - alpha)^2, lam' = lam + mu,
    alpha = 1 + mu / lam'; the rest value mu^2 / (2 lam') subtracted)."""
    if int(model.tet_count) == 0:
        return 0.0
    tet = model.tet_indices.numpy()
    Dm_inv = model.tet_poses.numpy().astype(np.float64)
    mat = model.tet_materials.numpy().astype(np.float64)
    x = q.astype(np.float64)
    Ds = np.stack([x[tet[:, 1]] - x[tet[:, 0]], x[tet[:, 2]] - x[tet[:, 0]], x[tet[:, 3]] - x[tet[:, 0]]], axis=2)
    F = Ds @ Dm_inv
    J = np.linalg.det(F)
    mu, lam = mat[:, 0], mat[:, 1] + mat[:, 0]
    alpha = 1.0 + mu / lam
    psi = 0.5 * mu * ((F**2).sum(axis=(1, 2)) - 3.0) + 0.5 * lam * (J - alpha) ** 2 - mu**2 / (2.0 * lam)
    vol = 1.0 / (6.0 * np.abs(np.linalg.det(Dm_inv)))
    return float((vol * psi).sum())
