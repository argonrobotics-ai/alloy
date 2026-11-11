# SPDX-License-Identifier: Apache-2.0
"""SolverRheologyMPM: Newton's implicit MPM with a Herschel-Bulkley viscosity per particle (docs/notes/
viscoplastic-design.md §2-§3).

At every ``step()``, before the stock solve, the Herschel-Bulkley particles get
eta = K * max(gdot, gdot_min)^(n - 1), clamped to [eta_min, K * gdot_min^(n - 1)], from the step-start shear rate
gdot = sqrt(2 D':D') of ``state_in.mpm.particle_qd_grad`` (D' the deviatoric strain rate). eta is a pure function of
the step-start state, so a rolled-back or re-stepped period gives the same eta (WP12's review). The value goes into the
viscosity array the solver's ``MaterialParameters`` holds, which Newton reads per particle every step
(implicit_mpm_solver_kernels.get_yield_parameters); the yield stress keeps the plug rigid, so eta only acts where the
paste flows. With no Herschel-Bulkley particle the builder keeps the stock solver (builtin_plugins._mpm).

The shear rate is that of the strain-rate tensor averaged over the particle's neighbours in its own world
(``RheologyConfig.smooth``: two voxels; a warp HashGrid with world groups). Each particle's APIC gradient carries
~0.5-2 1/s of noise in every component, and with n < 1 that noise, read per particle, made a film's plug fluid
(fitted 1/n 1.60-1.68 for n 0.4 on the dense grid). Smoothed, on 600 mm films on the dense grid: 0.99 / 0.87 of
analytic on 3 / 4.5 deg, 1/n 2.16-2.21 at 5 and 2.5 mm voxels (true 2.5); the remaining gap:
docs/known_gaps/paste-shear-thinning.md. The yield stress holds (L3). A Bingham film (n = 1: eta = K whatever the
average) is 0.995 of analytic on the sparse grid and 1.055 on the dense grid, where the sparse grid's floor offset no
longer slows its base (L4)."""

from __future__ import annotations

import logging

import numpy as np
import warp as wp
from alloy._vendor.newton.solvers import SolverImplicitMPM

from alloy.solvers.mpm_rheology.config import RHEO, RheologyConfig
from alloy.solvers.sph.grid_descriptor import GridDescriptor
from alloy.solvers.mpm_rheology.kernels import (gather_strain_rates, herschel_bulkley_viscosity,
                                                herschel_bulkley_viscosity_smoothed)

log = logging.getLogger("alloy.solvers.mpm_rheology")


def hb_particles(model) -> np.ndarray:
    """Indices of the particles with a Herschel-Bulkley consistency (> 0) on ``model`` (a Model or ModelView)."""
    ns = getattr(model, RHEO, None)
    k = getattr(ns, "consistency", None) if ns is not None else None
    return np.zeros(0, np.int32) if k is None else np.flatnonzero(k.numpy() > 0.0).astype(np.int32)


class SolverRheologyMPM(SolverImplicitMPM):
    """``SolverImplicitMPM`` plus the per-step Herschel-Bulkley viscosity update (capability ``mpm``)."""

    def __init__(self, model, config=None, rheology: RheologyConfig | None = None, **kw) -> None:
        super().__init__(model, config, **kw)
        self.rheology = rheology or RheologyConfig()
        ns = getattr(model, RHEO)
        idx = hb_particles(model)
        dev = model.device
        self._hb = wp.array(idx, dtype=wp.int32, device=dev)
        self._k, self._n = ns.consistency, ns.flow_index
        self._grid = None
        if len(idx) and self.rheology.smooth > 0.0:
            n = len(idx)
            world = getattr(model, "particle_world", None)
            self._world = world if world is not None else wp.zeros(model.particle_count, dtype=wp.int32, device=dev)
            self._points = wp.zeros(n, dtype=wp.vec3, device=dev)
            self._dev = wp.zeros(n, dtype=wp.mat33, device=dev)
            self._groups = wp.zeros(n, dtype=wp.int32, device=dev)
            self._grid = wp.HashGrid(128, 128, 128, device=dev)
            self._grid.reserve(n, with_groups=True)                 # graph capture reuses its buffers
            self._grid_desc = GridDescriptor(self._grid)              # and its descriptor (sph/grid_descriptor.py)
        log.info("SolverRheologyMPM: %d Herschel-Bulkley particles (gdot_min %.3g 1/s, strain rate averaged over "
                 "%.2g voxels)", len(idx), self.rheology.gdot_min, self.rheology.smooth)

    @property
    def _visc(self) -> wp.array:
        """The array the rheology solve reads, fetched each step (a notify may re-point it)."""
        return self._mpm_model.material_parameters.viscosity

    def step(self, state_in, state_out, control, contacts, dt: float) -> None:
        n, cfg, dev = self._hb.shape[0], self.rheology, self.model.device
        if n and self._grid is None:
            wp.launch(herschel_bulkley_viscosity, dim=n,
                      inputs=[self._hb, state_in.mpm.particle_qd_grad, self._k, self._n, cfg.gdot_min, cfg.eta_min],
                      outputs=[self._visc], device=dev)
        elif n:
            radius = cfg.smooth * float(self.voxel_size)
            wp.launch(gather_strain_rates, dim=n,
                      inputs=[self._hb, state_in.particle_q, state_in.mpm.particle_qd_grad, self._world],
                      outputs=[self._points, self._dev, self._groups], device=dev)
            self._grid_desc.build(self._points, radius, groups=self._groups)
            wp.launch(herschel_bulkley_viscosity_smoothed, dim=n,
                      inputs=[self._grid.id, self._hb, self._points, self._dev, self._groups, radius, self._k,
                              self._n, cfg.gdot_min, cfg.eta_min],
                      outputs=[self._visc], device=dev)
        super().step(state_in, state_out, control, contacts, dt)
