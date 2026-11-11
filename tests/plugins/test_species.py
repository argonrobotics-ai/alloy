# SPDX-License-Identifier: Apache-2.0
"""Species carried by particles (alloy/plugins/species): chemistry, config checks, the harness laws and the exact
inflow/outflow ledger (CPU)."""

from __future__ import annotations

import numpy as np
import pytest

import alloy.plugins as ap
from alloy.plugins.harness.scene import build_simulation
from alloy.plugins.species import Species, SpeciesConfig, chem
from alloy.plugins.species.kernels import AMT_SCALE_HOST
from alloy.plugins.species.laws import block_scene, split_config

pytestmark = pytest.mark.law


def test_ph_of_strong_acid_and_base():
    assert chem.ph(0.01, 0.0) == pytest.approx(2.0, abs=1e-9)
    assert chem.ph(0.0, 0.01) == pytest.approx(12.0, abs=1e-9)
    assert chem.ph(0.01, 0.01) == pytest.approx(7.0, abs=1e-12)
    assert chem.ph(0.0, 1.0) == pytest.approx(14.0, abs=1e-9)         # excess base: no cancellation
    assert 6.9 < chem.ph(1e-8, 0.0) < 7.0                             # water's own H+ dominates


def test_indicator_colour_and_equivalence():
    assert chem.coloured_fraction(chem.PHENOLPHTHALEIN_PKA) == pytest.approx(0.5)
    assert chem.coloured_fraction(chem.PINK_PH) == pytest.approx(1.0 / (1.0 + 10.0 ** 1.1))
    assert chem.coloured_fraction(7.0) < 0.01 < 0.97 < chem.coloured_fraction(11.0)
    assert chem.equivalence_volume(0.01, 25e-6, 0.1) == pytest.approx(2.5e-6)


@pytest.mark.parametrize("kwargs", [
    {"species": ("a", "a")},
    {"diffusivity": (1e-5,)},
    {"diffusivity": (1e-5, -1.0)},
    {"regions": ((0, 0, 0, 1, 1, 1, 1.0),)},
    {"inflow": (1.0,)},
    {"acid": "a"},
    {"acid": "a", "base": "zz"},
])
def test_config_is_validated(kwargs):
    with pytest.raises(ValueError):
        SpeciesConfig(**kwargs)


@pytest.mark.compile_heavy
def test_species_quick_check_passes_every_law():
    report = ap.check("alloy.plugins.species.plugin:Species", quick=True)
    assert report.passed, report.summary()
    laws = {c.name for c in report.checks if c.group == "law"}
    assert {"conserved_exactly", "diffusion_first_mode", "labels_without_diffusion"} <= laws


@pytest.mark.compile_heavy
def test_the_grid_guard_reads_the_host_in_setup_not_in_the_step(monkeypatch):
    """The diffusion grid's guard (WP9's GridDescriptor) checks warp's layout once, a host read: the plug-in's setup
    builds the grid for it, so no hook reads back (rc2-1109's CUDA red alloy-job3-f34a8f: ``.numpy()`` in update()
    under the harness's strict hooks failed conserved_exactly and diffusion_first_mode). The guard is off on the
    CPU, so this turns it on over the CPU grid (the same HashGrid_t) and runs the quick check's laws."""
    import warp as wp  # noqa: PLC0415

    from alloy.solvers.sph import grid_descriptor as gd  # noqa: PLC0415

    made = []

    class OnCpu(gd.GridDescriptor):
        def __init__(self, grid, dim=None, device=None):
            super().__init__(grid, dim, device)
            n = gd.descriptor_bytes(wp.float32)
            self.on = True
            self.view = wp.array(ptr=int(grid.id), dtype=wp.uint8, shape=(n,), device=grid.device, copy=False)
            self.saved = wp.zeros(n, dtype=wp.uint8, device=grid.device)
            made.append(self)

    monkeypatch.setattr(gd, "GridDescriptor", OnCpu)
    report = ap.check("alloy.plugins.species.plugin:Species", quick=True)
    assert report.passed, report.summary()
    assert made and all(d.ready and d.on for d in made)               # armed: warp's layout read as expected


def _set_active(sim, ids: np.ndarray, on: bool) -> None:
    f = sim.model.particle_flags.numpy()
    f[ids] = f[ids] | 1 if on else f[ids] & ~1
    sim.model.particle_flags.assign(f)


@pytest.mark.compile_heavy
def test_inflow_and_outflow_are_booked_exactly():
    """Five particles of world 0 leave (sink) and come back (emitter): their amounts go to the out ledger, they come
    back with the inflow composition, every world's totals hold exactly and the other worlds see nothing."""
    plug = Species(**(split_config() | {"inflow": (0.0, 500.0)}))
    sim = build_simulation(block_scene(), [plug], "cpu")
    sim.step_period()
    ids = plug.particles()[0][:5]
    before = plug.read()["amount"][0, :5].copy()
    _set_active(sim, ids, False)
    sim.step_period()
    r = plug.read()
    assert np.array_equal(r["ledger_out"][0], before.sum(axis=0))
    assert not r["amount"][0, :5].any()
    _set_active(sim, ids, True)
    sim.step_period()
    r = plug.read()
    vol = plug.vol_host[ids].astype(np.float64)
    want = np.round(500.0 * vol * AMT_SCALE_HOST).astype(np.int64).sum()
    assert r["ledger_in"][0].tolist() == [0, want]
    assert not r["ledger_in"][1:].any() and not r["ledger_out"][1:].any()
    assert not np.abs(r["conservation_error"]).any()


def test_diffusion_support_follows_the_sph_solver():
    """With radius 0 the plug-in diffuses on the SPH solver's own support (support_factor x particle radius)."""
    from types import SimpleNamespace  # noqa: PLC0415

    from alloy.solvers.sph.config import DFSPHConfig  # noqa: PLC0415
    from alloy.solvers.sph.solver import SolverDFSPH  # noqa: PLC0415

    plug = Species(**split_config())
    plug.h = 0.008                                      # setup's 4-radius default for r = 2 mm
    sph = object.__new__(SolverDFSPH)                   # a stand-in entry: only its config is read
    sph.config = DFSPHConfig(particle_radius=0.002, support_factor=3.0)
    plug.attach(SimpleNamespace(solver=sph))
    assert plug.h == pytest.approx(0.006)
