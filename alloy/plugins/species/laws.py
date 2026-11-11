# SPDX-License-Identifier: Apache-2.0
"""Test scenes and laws of the species plug-in (plugin.py), CPU-sized.

- ``conserved_exactly``: a two-species block diffusing; per world and species, sum(amounts) = initial + in - out in
  integers at every period (bound 0).
- ``diffusion_first_mode``: the same block, a step in species a along x. With no-flux ends the first cosine mode of
  the concentration decays as exp(-D pi^2 t / L^2) (L the block's length); the measured rate over 0.5 s against it.
- ``labels_without_diffusion``: diffusivity 0 and the liquid collapsing in a tank under gravity (a small dam break):
  every particle keeps its amounts bit for bit while it moves (advection is exact) and the totals hold.
"""

from __future__ import annotations

import numpy as np

from alloy.plugins.harness.scene import Law, LawResult, TestScene
from alloy.solvers.sph.build import add_box_tank, add_fluid_points, lattice_points
from alloy.solvers.sph.config import DFSPHConfig, register_custom_attributes

R = 1.0e-3                  # particle radius [m]
SP = 2.0 * R                # lattice spacing
BLOCK = (6, 6, 6)           # cells
D_LAW = 5.0e-5              # effective diffusivity of the laws [m^2/s]
C0 = 1000.0                 # start concentration [unit/m^3]
HALF = BLOCK[0] * SP / 2.0
SPH = {"particle_radius": R}


def _block(b) -> None:
    """A free 12 mm liquid cube at the origin (zero gravity: it stays put)."""
    register_custom_attributes(b)
    add_fluid_points(b, DFSPHConfig(particle_radius=R), lattice_points((0.0, 0.0, 0.0), BLOCK, SP))


def _dam(b) -> None:
    """A 16 x 8 x 12 mm tank with a 6 x 8 x 6 cell liquid block in one end: it collapses and runs along the floor."""
    register_custom_attributes(b)
    cfg = DFSPHConfig(particle_radius=R)
    add_box_tank(b, (0.0, 0.0, 0.0), (8 * SP, 4 * SP, 6 * SP), wall=cfg.h)
    add_fluid_points(b, cfg, lattice_points((0.0, 0.0, 0.0), (3, 4, 3), SP))


def block_scene(periods: int = 25) -> TestScene:
    return TestScene(build=_block, worlds=3, periods=periods, solver="sph", solver_kwargs=SPH, ground=False,
                     gravity=(0.0, 0.0, 0.0))


def dam_scene(periods: int = 15) -> TestScene:
    return TestScene(build=_dam, worlds=3, periods=periods, solver="sph", solver_kwargs=SPH, ground=False)


def split_config(diffusivity: float = D_LAW) -> dict:
    """Two species: a in the block's left half (x < 6 mm), b in the right half, both at C0."""
    big = 1.0
    return {"species": ("a", "b"), "diffusivity": (diffusivity, diffusivity),
            "regions": ((-big, -big, -big, HALF, big, big, C0, 0.0), (HALF, -big, -big, big, big, big, 0.0, C0))}


def _conserved(run) -> LawResult:
    err = np.asarray(run.trace.plugin(run.plugin.label, "conservation_error"))
    worst = int(np.abs(err).max())
    return LawResult.within(worst, 0.0, f"largest |sum - (initial + in - out)| = {worst} fixed-point steps over "
                            f"{err.shape[0]} periods", "pair fluxes must be applied with exactly opposite signs")


def _first_mode(run) -> LawResult:
    conc = np.asarray(run.trace.plugin(run.plugin.label, "conc"))[:, 0, :, 0]      # [K, N] species a, world 0
    ids = run.plugin.particles()[0]
    x = np.asarray(run.trace.particle_q)[:, ids, 0]                                 # [K, N]
    L = BLOCK[0] * SP
    phi = np.cos(np.pi * x / L)
    amp = (conc * phi).sum(axis=1) / (phi * phi).sum(axis=1)
    t = np.asarray(run.trace.t)
    rate = -np.polyfit(t[1:], np.log(np.abs(amp[1:])), 1)[0]
    want = run.plugin.config.diffusivity[0] * np.pi**2 / L**2
    return LawResult.within(rate / want - 1.0, 0.15, f"first-mode decay {rate:.3f} 1/s vs D pi^2/L^2 {want:.3f} 1/s",
                            "check the Laplacian's factor (2 D) and the kernel gradient's sign")


def _labels(run) -> LawResult:
    amt = np.asarray(run.trace.plugin(run.plugin.label, "amount"))                 # [K, W, N, S]
    moved = float(np.abs(np.asarray(run.trace.particle_q)[-1] - np.asarray(run.trace.particle_q)[0]).max())
    worst = int(np.abs(amt - amt[0]).max())
    return LawResult.within(worst, 0.0, f"largest change of any particle's amount {worst} (particles moved up to "
                            f"{moved * 1e3:.1f} mm)", "without diffusion nothing may change a particle's amount")


def species_laws() -> list[Law]:
    return [
        Law("conserved_exactly", _conserved, scene=block_scene(), config=split_config(),
            doc="sum(amounts) = initial + in - out, exactly, every period"),
        Law("diffusion_first_mode", _first_mode, scene=block_scene(), config=split_config(),
            doc="the step's first cosine mode decays at D pi^2 / L^2 (15 %: SPH Laplacian on a 6-cell block)"),
        Law("labels_without_diffusion", _labels, scene=dam_scene(), config=split_config(0.0),
            doc="labels move with the liquid exactly; no change without diffusion"),
    ]
