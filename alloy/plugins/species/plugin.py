# SPDX-License-Identifier: Apache-2.0
"""Species carried by particles: per-particle amounts of named species on SPH liquids (and any particles), moving with
them, optionally diffusing between neighbours, conserved exactly (Gym-Anything gap: mixing, titration; coordinator
2026-11-06).

What it does every exchange, after the solver (a state-only plug-in, ``update``):
- **Advection.** Amounts belong to particles, so they move with the liquid exactly (Lagrangian labels).
- **Inflow and outflow.** A particle that becomes active (an emitter or pour) takes the ``inflow`` composition and
  the amount is counted in; one that becomes inactive (a sink, the escape box) gives its amounts to the out ledger.
- **Diffusion** (species with ``diffusivity`` > 0): SPH Laplacian exchange between neighbours within ``radius``,
  pairwise antisymmetric in int64 fixed point (kernels.py). The neighbours come from the plug-in's own HashGrid,
  guarded across CUDA graph replays by WP9's ``GridDescriptor`` (warp 1.18 replays every captured float32 grid build
  with the last host-built grid's descriptor: an SPH solver's, a camera's, another sim's). Setup builds the grid
  once (and ``attach`` again when it sets the support), so the guard's one host read, its check of warp's layout,
  runs there: hooks only launch kernels.
- **Conservation.** Per world and species, sum(amounts) = initial + in - out holds exactly (integers); ``read()``
  reports the difference (``conservation_error``, always 0).

Which particles. SPH particles (``model.sph.phase`` >= 0: liquid and spares) when the model has the SPH attributes,
else every particle (MPM grains, labels only). Each world must hold the same number. Volume per particle: mass /
rest density (``model.sph.rest_density``; ``density`` otherwise).

Units. Concentrations [unit/m^3], amounts [unit] = concentration x particle volume. For the acid-base readout the
unit is mol, so mol/m^3 = mmol/L.

Diffusivity is an effective, sub-resolution value. Molecular diffusion (~1e-9 m^2/s) would need hours to cross a 2 mm
particle; in a stirred liquid, eddies mix at scales below the particle spacing much faster. The tasks set it.

Example::

    sp = Species(species=("red", "blue"), diffusivity=(2e-5, 2e-5),
                 regions=((0, 0, 0, 0.02, 0.04, 0.03, 1.0, 0.0), (0.02, 0, 0, 0.04, 0.04, 0.03, 0.0, 1.0)))
    sim = alloy.plugins.simulation(built, [sp])
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from alloy.plugins.base import ForcePlugin
from alloy.plugins.errors import PluginError

from . import chem
from . import kernels as kn

GRID_DIM = 128                 # cells per axis of the diffusion neighbour grid


@dataclass
class SpeciesConfig:
    species: tuple = ("a", "b")          # species names
    diffusivity: tuple = (0.0, 0.0)      # effective diffusivity per species [m^2/s]; 0 = labels only
    regions: tuple = ()                  # ((x0, y0, z0, x1, y1, z1, c_1, ..., c_S), ...) start conc. [unit/m^3]
    inflow: tuple = ()                   # concentrations of particles that become active [unit/m^3]; () = zeros
    radius: float = 0.0                  # neighbour radius [m]; 0 = the SPH solver's support h (4 radii w/o SPH)
    density: float = 1000.0              # [kg/m^3] particle volume = mass / density without SPH attributes
    acid: str = ""                       # acid-base readout: strong acid species (mol)
    base: str = ""                       # strong base species (mol)
    indicator: str = ""                  # indicator species; coloured where present
    pka: float = chem.PHENOLPHTHALEIN_PKA   # indicator pKa

    def __post_init__(self) -> None:
        S = len(self.species)
        if S < 1 or len(set(self.species)) != S:
            raise ValueError(f"species must be distinct names, got {self.species!r}")
        if len(self.diffusivity) != S or min(self.diffusivity) < 0.0:
            raise ValueError(f"diffusivity needs one value >= 0 per species ({S}), got {self.diffusivity!r}")
        for r in self.regions:
            if len(r) != 6 + S:
                raise ValueError(f"each region is (x0, y0, z0, x1, y1, z1) plus {S} concentrations, got {r!r}")
        if self.inflow and len(self.inflow) != S:
            raise ValueError(f"inflow needs {S} concentrations, got {self.inflow!r}")
        for name in (self.acid, self.base, self.indicator):
            if name and name not in self.species:
                raise ValueError(f"{name!r} is not a species; species: {self.species!r}")
        if bool(self.acid) != bool(self.base):
            raise ValueError("the acid-base readout needs both acid and base")


class Species(ForcePlugin):
    """Species carried by particles: Lagrangian labels, exact inflow/outflow ledger, SPH diffusion between
    neighbours, acid-base pH and indicator colour. Example: Species(species=("red", "blue"),
    diffusivity=(2e-5, 2e-5), regions=((...red box..., 1, 0), (...blue box..., 0, 1)))."""

    name = "species"
    summary = "species amounts carried by particles: advection, exact inflow/outflow, SPH diffusion, pH"
    Config = SpeciesConfig

    def setup(self, ctx) -> None:
        cfg, m, W = self.config, ctx.model, ctx.world_count
        S = len(cfg.species)
        P = int(m.particle_count)
        sph = getattr(m, "sph", None)
        tracked = np.ones(P, bool) if sph is None else sph.phase.numpy() >= 0
        world = m.particle_world.numpy()
        ids = [np.nonzero(tracked & (world == w))[0] for w in range(W)]
        N = len(ids[0]) if ids else 0
        if N == 0 or any(len(x) != N for x in ids):
            raise PluginError(f"{self.label}: every world needs the same number (>= 1) of particles, got "
                              f"{[len(x) for x in ids]}; build the liquid inside each world")
        idx = np.stack(ids).astype(np.int32)
        rho = sph.rest_density.numpy() if sph is not None else np.full(P, cfg.density, np.float32)
        vol = (m.particle_mass.numpy() / np.maximum(rho, 1.0e-12)).astype(np.float32)
        slot = np.full(P, -1, np.int32)
        slot[idx.reshape(-1)] = np.tile(np.arange(N, dtype=np.int32), W)
        active = (m.particle_flags.numpy()[idx] & 1) != 0
        amount = self._initial(m.particle_q.numpy()[idx], vol[idx], active)
        self.idx_host, self.vol_host = idx, vol
        self.idx, self.slot = ctx.array(idx, wp.int32), ctx.array(slot, wp.int32)
        self.volume = ctx.array(vol, wp.float32)
        self.amount = self.add_state("amount", ctx.array(amount, wp.int64))
        self.active = self.add_state("active", ctx.array(active.astype(np.int32), wp.int32))
        self.ledger_in = self.add_state("ledger_in", ctx.zeros((W, S), wp.int64))
        self.ledger_out = self.add_state("ledger_out", ctx.zeros((W, S), wp.int64))
        self.total0 = amount.sum(axis=1)
        self.delta = ctx.zeros((W, N, S), wp.int64)
        self.group = ctx.full(max(P, 1), -1, wp.int32)
        self.diff = ctx.array(np.asarray(cfg.diffusivity, np.float64), wp.float64)
        self.inflow = ctx.array(np.asarray(cfg.inflow or (0.0,) * S, np.float64), wp.float64)
        r = float(m.particle_radius.numpy()[idx[0, 0]]) if cfg.radius <= 0.0 else float(cfg.radius)
        self.h = 4.0 * r if cfg.radius <= 0.0 else r
        self.diffuses = max(cfg.diffusivity) > 0.0
        if self.diffuses:
            from alloy.solvers.sph.grid_descriptor import GridDescriptor  # noqa: PLC0415 - keeps imports light

            self.grid = wp.HashGrid(GRID_DIM, GRID_DIM, GRID_DIM, device=ctx.device)
            self.grid.reserve(max(P, 1), with_groups=True)
            self.desc = GridDescriptor(self.grid, GRID_DIM, ctx.device)   # graph replays keep this grid's descriptor
            self._arm_grid(ctx)

    def _arm_grid(self, ctx) -> None:
        """One build as ``update`` builds (the groups, the support ``h``), outside the step: the guard checks warp's
        layout here (a host read; under the harness's strict hooks ``.numpy()`` in ``update`` raises) and keeps the
        descriptor, so even a first build inside a graph capture is guarded."""
        W, N = ctx.world_count, self.idx_host.shape[1]
        wp.launch(kn.groups_kernel, dim=(W, N), device=ctx.device, inputs=[self.idx, self.active, self.group])
        self.desc.build(ctx.model.particle_q, self.h, groups=self.group)

    def _initial(self, x: np.ndarray, vol: np.ndarray, active: np.ndarray) -> np.ndarray:
        """Fixed-point start amounts [W, N, S]: the first region containing each active particle."""
        S = len(self.config.species)
        conc = np.zeros(x.shape[:2] + (S,), np.float64)
        free = np.ones(x.shape[:2], bool)
        for r in self.config.regions:
            lo, hi, c = np.asarray(r[:3]), np.asarray(r[3:6]), np.asarray(r[6:], np.float64)
            inside = free & np.all((x >= lo) & (x <= hi), axis=-1)
            conc[inside] = c
            free &= ~inside
        conc[~active] = 0.0
        return np.round(conc * vol[..., None].astype(np.float64) * kn.AMT_SCALE_HOST).astype(np.int64)

    def attach(self, stepper) -> None:
        """Called by ``PluginHost.install``: with ``radius`` 0, diffuse on the SPH solver's own support (DFSPHConfig.h
        = support_factor x particle radius) rather than the 4-radius default."""
        if self.config.radius > 0.0:
            return
        from alloy.integration.stepper import entry_solvers  # noqa: PLC0415 - keeps `import alloy.plugins` light
        from alloy.solvers.sph.solver import SolverDFSPH  # noqa: PLC0415

        hs = sorted({float(s.config.h) for s in entry_solvers(stepper).values() if isinstance(s, SolverDFSPH)})
        if len(hs) > 1:
            raise PluginError(f"{self.label}: SPH entries with different supports {hs} m: set SpeciesConfig.radius")
        if hs and hs[0] != self.h:
            self.h = hs[0]
            if getattr(self, "desc", None) is not None:     # set up, with a diffusion grid
                self._arm_grid(self.ctx)                     # the descriptor at the new support

    def check_dt(self, dt: float) -> None:
        d = max(self.config.diffusivity)
        if d > 0.0 and dt * d / self.h**2 > 0.1:
            raise PluginError(f"{self.label}: explicit diffusion needs dt D / h^2 <= 0.1, got {dt * d / self.h**2:.3g}"
                              f" (dt {dt:g} s, D {d:g} m^2/s, h {self.h:g} m): lower the diffusivity or the step")

    def update(self, state_in, state_out, dt) -> None:
        del state_in
        ctx, W, N = self.ctx, self.ctx.world_count, self.idx_host.shape[1]
        wp.launch(kn.bookkeeping_kernel, dim=(W, N), device=ctx.device,
                  inputs=[self.idx, ctx.model.particle_flags, self.volume, self.inflow, self.active, self.amount,
                          self.ledger_in, self.ledger_out])
        if not self.diffuses:
            return
        wp.launch(kn.groups_kernel, dim=(W, N), device=ctx.device, inputs=[self.idx, self.active, self.group])
        self.grid.build(state_out.particle_q, self.h, groups=self.group)
        self.desc.after_build(state_out.particle_q.shape[0], self.h)
        wp.launch(kn.diffusion_kernel, dim=(W, N), device=ctx.device,
                  inputs=[self.grid.id, state_out.particle_q, self.idx, self.slot, self.group, self.volume,
                          self.diff, float(self.h), float(0.01 * self.h * self.h), float(dt), self.amount,
                          self.delta])
        wp.launch(kn.apply_kernel, dim=(W, N, len(self.config.species)), device=ctx.device,
                  inputs=[self.delta, self.amount])

    # --- readouts (host) -------------------------------------------------------------------------------------------

    def particles(self) -> np.ndarray:
        """[W, N] global particle ids, in the order of every per-particle readout."""
        return self.idx_host

    def read(self) -> dict[str, np.ndarray]:
        """State plus: ``conc`` [W, N, S] (unit/m^3), ``total`` [W, S] (unit), ``conservation_error`` [W, S] (fixed
        point steps, 0 when exact), with the acid-base readout ``ph`` [W, N] and ``colour`` [W, N] (indicator's
        coloured fraction where the indicator is present, else 0)."""
        out = super().read()
        amt = out["amount"]
        vol = self.vol_host[self.idx_host].astype(np.float64)
        out["conc"] = amt.astype(np.float64) / kn.AMT_SCALE_HOST / vol[..., None]
        out["total"] = amt.sum(axis=1).astype(np.float64) / kn.AMT_SCALE_HOST
        out["conservation_error"] = amt.sum(axis=1) - (self.total0 + out["ledger_in"] - out["ledger_out"])
        cfg = self.config
        if cfg.acid:
            sp = list(cfg.species)
            c = out["conc"] / 1000.0  # mol/m^3 -> mol/L
            out["ph"] = chem.ph(c[..., sp.index(cfg.acid)], c[..., sp.index(cfg.base)])
            col = chem.coloured_fraction(out["ph"], cfg.pka)
            if cfg.indicator:
                col = np.where(c[..., sp.index(cfg.indicator)] > 0.0, col, 0.0)
            out["colour"] = np.where(out["active"] != 0, col, 0.0)
        return out

    def test_scene(self):
        from .laws import block_scene  # noqa: PLC0415

        return block_scene()

    def laws(self):
        from .laws import species_laws  # noqa: PLC0415

        return species_laws()
