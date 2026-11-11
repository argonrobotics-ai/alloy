# SPDX-License-Identifier: Apache-2.0
"""Structure and solids in the film's rheology (design note §3.2, "Rheology").

- **Thixotropy (Moore).** A structure parameter ``lam`` in [0, 1] per cell, ``dlam/dt = (1 - lam)/t_r - k_b·γ̇·lam``
  with ``γ̇ = 3|q̄|/h²``, stepped exactly for the step's shear (piecewise-constant γ̇):
  ``lam ← lam_ss + (lam - lam_ss)·exp(-(1/t_r + k_b·γ̇)·dt)``, ``lam_ss = 1/(1 + k_b·γ̇·t_r)``. An applicator pass
  breaks the structure (lam → 0, P:796): a ceiling tool or pad over the cell, or a store tool's nip. The yield stress
  is ``lam`` times the at-rest yield (the material's, or the per-cell input that setting writes, design §4.8), so a
  rolled coat levels, then the rebuilt structure stops its sag.
- **Solids (Krieger-Dougherty).** With a ``solids`` species (volume per area, ``phi_s = m/h``), the viscosity is the
  material's (or the per-cell input's) times ``((1 - phi_s/phi_m)/(1 - phi_ref/phi_m))^(-2.5·phi_m)``, at least 1:
  the card's viscosity holds at ``phi_ref`` (``krieger_ref``: a paint card's as-applied phi_0; 0, the solvent's), and
  drying stiffens the film and sets the wet edge. It only raises the viscosity, and the face friction is implicit, so
  the step bound holds.

Both are per-cell factors the face kernel's ``face_rheology`` reads (``Fluid.lam``, ``Fluid.vfac``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import warp as wp

from alloy.plugins.errors import PluginError
from alloy.plugins.surface_flow.cell_kernels import Species
from alloy.plugins.surface_flow.kernels import Cells, Faces, Fluid, ToolField, touch
from alloy.plugins.surface_flow.links import cell_dir

PHI_CAP = 0.999                   # phi_s/phi_m never reaches 1: the factor stays finite (6.3e4 at phi_m 0.64)


@dataclass(frozen=True)
class Thixotropy:
    """Moore's structure kinetics: ``rebuild`` t_r [s] at rest, ``breakdown`` k_b [-] under shear."""

    rebuild: float
    breakdown: float = 0.0

    def __post_init__(self) -> None:
        if not (math.isfinite(self.rebuild) and self.rebuild > 0.0):
            raise PluginError(f"thixotropy: rebuild must be finite and > 0 s, got {self.rebuild}")
        if not (math.isfinite(self.breakdown) and self.breakdown >= 0.0):
            raise PluginError(f"thixotropy: breakdown must be finite and >= 0, got {self.breakdown}")


def thixotropy_spec(spec: Thixotropy | dict | None) -> Thixotropy | None:
    if spec is None or isinstance(spec, Thixotropy):
        return spec
    if isinstance(spec, dict):
        return Thixotropy(**spec)
    raise PluginError(f"thixotropy: expected a Thixotropy or a dict, got {spec!r}")


def setup_rheology(flow: Any, ctx: Any, W: int, C: int) -> None:
    """The structure and solids factors on ``flow.fluid`` ((1, 1) and off unless configured)."""
    c, fl = flow.config, flow.fluid
    thixo = flow.thixo = thixotropy_spec(c.thixotropy)
    fl.thixo = 1 if thixo is not None else 0
    fl.lam = flow.add_state("lam", ctx.full((W, C), 1.0, wp.float32)) if thixo else ctx.full((1, 1), 1.0, wp.float32)
    fl.krieger = 0
    flow.solids_index, flow.phi_m, flow.phi_ref = -1, 0.0, 0.0
    if c.krieger is not None:
        phi_m, phi_ref = float(c.krieger), float(c.krieger_ref)
        if not 0.0 < phi_m <= 1.0:
            raise PluginError(f"{flow.label}: krieger (phi_m) must be in (0, 1], got {phi_m}")
        if not 0.0 <= phi_ref < phi_m:
            raise PluginError(f"{flow.label}: krieger_ref (the solids fraction where the viscosity is the card's) must "
                              f"be in [0, phi_m = {phi_m}), got {phi_ref}")
        if "solids" not in tuple(c.species):
            raise PluginError(f"{flow.label}: krieger needs a 'solids' species (Config.species), the solids volume "
                              f"per area; got {tuple(c.species)}")
        fl.krieger = 1
        flow.solids_index, flow.phi_m, flow.phi_ref = tuple(c.species).index("solids"), phi_m, phi_ref
    fl.vfac = ctx.full((W, C) if fl.krieger else (1, 1), 1.0, wp.float32)


@wp.func
def cell_shear_rate(cells: Cells, faces: Faces, q: wp.array2d[wp.float32], h: wp.array2d[wp.float64], w: int,
                    c: int, h_wet: wp.float64) -> float:
    """``γ̇ = 3|q̄|/max(h, h_wet)²``, ``q̄`` the mean of the cell's face discharges (two faces per axis)."""
    fs = cells.face[c]
    qv = wp.vec3(0.0)
    for s in range(4):
        f = fs[s]
        if f >= 0:
            qv += q[w, f] * cell_dir(faces, f, c)
    hh = float(wp.max(h[w, c], h_wet))
    return 3.0 * 0.5 * wp.length(qv) / (hh * hh)


@wp.kernel
def solids_kernel(cells: Cells, tick: wp.array[wp.int32], stride: int, on: wp.array2d[wp.int32],
                  h: wp.array2d[wp.float64], sp: Species, k_s: int, phi_m: float, phi_ref: float,
                  h_wet: wp.float64, vfac: wp.array2d[wp.float32]):
    """The Krieger-Dougherty factor of each cell's solids at the substep's start (the face kernel's viscosity),
    relative to ``phi_ref``'s and at least 1."""
    w, c = wp.tid()
    if tick[w] % stride != 0 or on[w, cells.tile[c]] == 0:
        return
    phi = float(0.0)
    if h[w, c] > h_wet:
        phi = float(sp.mass[w, c, k_s] / h[w, c])
    x = wp.min(wp.max(phi / phi_m, 0.0), PHI_CAP)
    vfac[w, c] = wp.max(wp.pow((1.0 - x) / (1.0 - phi_ref / phi_m), -2.5 * phi_m), 1.0)


@wp.kernel
def structure_kernel(cells: Cells, faces: Faces, tick: wp.array[wp.int32], stride: int, on: wp.array2d[wp.int32],
                     fl: Fluid, h: wp.array2d[wp.float64], base: wp.array2d[wp.float32], q: wp.array2d[wp.float32],
                     tf: ToolField, t_r: float, k_b: float, dt: float):
    """Moore's kinetics over the substep at its moved discharge (exact for its γ̇); a tool over the cell breaks it."""
    w, c = wp.tid()
    if tick[w] % stride != 0 or on[w, cells.tile[c]] == 0:
        return
    if tf.on != 0 and (tf.press[w, c] > 0.0 or (tf.cover[w, c] > 0.0 and touch(tf, h, base, w, c) > 0.0)):
        fl.lam[w, c] = 0.0                                       # an applicator pass (P:796)
        return
    rate = k_b * cell_shear_rate(cells, faces, q, h, w, c, fl.h_wet)
    r = 1.0 / t_r + rate
    lam_ss = (1.0 / t_r) / r
    fl.lam[w, c] = lam_ss + (fl.lam[w, c] - lam_ss) * wp.exp(-r * dt)


__all__ = ["PHI_CAP", "Thixotropy", "cell_shear_rate", "setup_rheology", "solids_kernel", "structure_kernel",
           "thixotropy_spec"]
