# SPDX-License-Identifier: Apache-2.0
"""Heat v3 (SCR-WP8-14, docs/notes/heat-v3-design.md): grid materials and their H -> T map on host and device, graded
grids, GridThermal config, size and step pins for a GA-sized coupon, the laws G-1 to G-6 and D-1 to D-7 (deposition,
the cap, the pool), the harness checks and a CUDA smoke."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from alloy.plugins.errors import PluginError
from alloy.plugins.harness.check import _law, check, fresh
from alloy.plugins.harness.scene import build_simulation
from alloy.plugins.thermal import grid_cap as C
from alloy.plugins.thermal import grid_kernels as K
from alloy.plugins.thermal.grid import Config, GridThermal, estimate, graded, occupancy, substeps
from alloy.plugins.thermal.grid_deposit_laws import deposit_laws
from alloy.plugins.thermal.grid_laws import BUSY, busy_scene, grid_laws, neumann_lambda, tee
from alloy.plugins.thermal.grid_materials import MATERIALS, material
from alloy.plugins.thermal.sensor import Thermal

LAWS = {law.name: law for law in grid_laws() + deposit_laws()}

GA_COUPON = Config(x=graded(-0.01, 0.16, 5e-4, band=(0.0, 0.15)), y=graded(-0.05, 0.05, 2.5e-4, band=(-0.01, 0.016)),
                   z=graded(-0.01, 0.05, 2.5e-4, band=(-0.01, 0.016)),
                   boxes=((-0.01, -0.05, -0.01, 0.16, 0.05, 0.0), (0.0, -0.01, 0.0, 0.15, 0.0, 0.05)), parts=(0, 1),
                   sink=("z-",))


@wp.kernel
def _lookup(tb: K.Table, h: wp.array(dtype=float), out: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    out[i] = K.lookup(tb, h[i])


def _device_lookup(name: str, h: np.ndarray) -> np.ndarray:
    m = material(name)
    tt, hh, kk = m.table()
    tb = K.Table()
    tb.t, tb.h, tb.k = (wp.array(v.astype(np.float32), dtype=float, device="cpu") for v in (tt, hh, kk))
    tb.count, tb.h_liquidus, tb.h_boil = len(tt), m.h_liquidus, m.h_boil
    tb.c_low = float(m.c_solid(np.asarray(tt[0])))
    out = wp.zeros(len(h), dtype=wp.vec3, device="cpu")
    wp.launch(_lookup, dim=len(h), inputs=[tb, wp.array(h.astype(np.float32), dtype=float, device="cpu")],
              outputs=[out], device="cpu")
    return out.numpy()


def test_steel_round_trip_mushy_zone_and_liquid_conductivity():
    m = material("steel_a36")
    temps = np.array([250.0, 293.15, 1000.0, 1008.15, 1745.15, 1765.15, 1785.15, 2500.0])
    back, phi = m.temperature(m.enthalpy(temps))
    np.testing.assert_allclose(back, temps, atol=1e-6)
    np.testing.assert_allclose(phi, [0, 0, 0, 0, 0, 0.5, 1, 1], atol=1e-12)
    assert float(m.enthalpy(m.t_solidus)) == 0.0 and m.h_liquidus == pytest.approx(2.47e5 + 650.0 * 40.0)
    assert float(m.conductivity(m.h_liquidus)) == pytest.approx(420.0)
    assert float(m.conductivity(0.5 * m.h_liquidus)) == pytest.approx(0.5 * (27.3 + 420.0))
    # EN 1993-1-2's 735 °C peak: 5000 J/(kg·K) on both sides of the breakpoint
    assert float(m._c(273.15 + 735.0)) == pytest.approx(5000.0)


def test_pure_substance_plateau_ends_on_host_and_device():
    """WP13: with T_l = T_s, H = 0 is solid and H = L liquid, both at T_s, no division by T_l − T_s."""
    m = material("law_pure")
    h = np.array([0.0, 0.5 * m.latent, m.latent, -600.0, m.latent + 600.0])
    t, phi = m.temperature(h)
    np.testing.assert_allclose(t, [1000.0, 1000.0, 1000.0, 999.0, 1001.0])
    np.testing.assert_allclose(phi, [0.0, 0.5, 1.0, 0.0, 1.0])
    dev = _device_lookup("law_pure", h)
    np.testing.assert_allclose(dev[:, 0], t, atol=1e-3)
    np.testing.assert_allclose(dev[:, 2], phi, atol=1e-6)
    assert np.all(np.isfinite(dev))


def test_device_lookup_matches_host_on_steel():
    m = material("steel_a36")
    h = m.enthalpy(np.linspace(250.0, 3100.0, 997))
    dev = _device_lookup("steel_a36", h)
    t, phi = m.temperature(h)
    np.testing.assert_allclose(dev[:, 0], t, atol=0.05)
    np.testing.assert_allclose(dev[:, 1], m.conductivity(h), rtol=1e-4)
    np.testing.assert_allclose(dev[:, 2], phi, atol=1e-5)


def test_every_material_cites_a_source():
    for m in MATERIALS.values():
        assert m.source and np.all(np.diff(m.table()[1]) > 0.0)
    assert "ESTIMATE" in material("steel_a36").source
    with pytest.raises(KeyError, match="steel_a36"):
        material("unobtainium")


def test_graded_edges():
    e = np.asarray(graded(-0.01, 0.05, 5e-4, band=(0.0, 0.02), growth=1.2, coarse=3e-3))
    d = np.diff(e)
    assert e[0] == -0.01 and e[-1] == 0.05 and np.all(d > 0.0)
    inner = (e[:-1] >= 0.0) & (e[1:] <= 0.02 + 1e-12)
    np.testing.assert_allclose(d[inner], 5e-4)
    assert np.max(np.maximum(d[1:] / d[:-1], d[:-1] / d[1:])) <= 1.2 + 1e-9 and d.max() <= 3e-3
    with pytest.raises(ValueError):
        graded(0.0, 1.0, 0.1, band=(0.5, 2.0))


def test_config_errors():
    with pytest.raises(ValueError, match="edges"):
        Config(x=(0.0,))
    with pytest.raises(ValueError, match="boxes"):
        Config(boxes=((0.0, 0.0, 0.0, -1.0, 1.0, 1.0),))
    with pytest.raises(ValueError, match="sink"):
        Config(sink=("bottom",))
    with pytest.raises(KeyError):
        Config(material="unobtainium")
    with pytest.raises(ValueError, match="deposit_efficiency"):
        Config(deposit_efficiency=1.5)
    with pytest.raises(ValueError, match="bead_norm"):
        Config(bead_norm=0.5)
    with pytest.raises(ValueError, match="bead_model"):
        Config(bead_model="blob")
    with pytest.raises(ValueError, match="counter-clockwise"):
        Config(joint_rays=((0.0, 1.0), (1.0, 0.0)))
    with pytest.raises(ValueError, match="parts"):
        Config(**{**tee(1e-3, 0.01), "parts": (0,)})


def test_tee_occupancy_and_default_plate():
    cfg = Config(**tee(1e-3, 0.01))
    cell = occupancy(cfg)
    assert set(np.unique(cell)) == {0, 1, 2}
    assert np.count_nonzero(cell == 2) == 10 * 6 * 13    # bracket 10 x 6 x 20 mm: 6 band cells + 7 graded in z
    plate = Config()
    assert estimate(plate)["metal"] == 20 * 20 * 5


def test_ga_coupon_size_and_substeps_are_pinned():
    """A GA-sized coupon (0.5 mm along the joint, 0.25 mm across in the band, 1.2x growth): 5.76 M cells, ~0.3 GB
    per world, 7 explicit steps per 1 ms robot tick (GA: 7.18 M cells, 6 steps of 0.169 ms; report_welding §3-4)."""
    est = estimate(GA_COUPON)
    assert est["cells"] == 5_757_072 and est["bytes_per_world"] == pytest.approx(2.99e8, rel=0.01)
    assert substeps(GA_COUPON, 1e-3) == 7
    for law, cells, steps in (("ledger", 19_000, 12), ("rosenthal", 70_848, 4), ("stefan", 48, 7)):
        cfg = Config(**LAWS[law].config)
        assert (estimate(cfg)["cells"], substeps(cfg, 0.02)) == (cells, steps), law


def test_max_cells_refuses_before_allocating():
    with pytest.raises(PluginError, match="max_cells"):
        build_simulation(busy_scene(), [GridThermal(**{**BUSY, "max_cells": 100})], "cpu")


def test_exclusive_refuses_a_body_heated_twice():
    host = SimpleNamespace(plugins={"thermal": Thermal(bodies=("plate",)), "grid": GridThermal(body="plate")})
    with pytest.raises(PluginError, match="count twice"):
        GridThermal.exclusive(host)
    GridThermal.exclusive(SimpleNamespace(plugins={"grid": GridThermal(body="plate")}))


def test_neumann_lambda():
    assert neumann_lambda(1.2) == pytest.approx(0.6655, abs=1e-4)


@pytest.mark.law
@pytest.mark.timeout(900)
@pytest.mark.parametrize("name", sorted(LAWS))
def test_grid_law(name):
    law = LAWS[name]
    res = _law(law, None, [fresh(GridThermal(**BUSY), law.config)], "cpu")
    assert res.passed, (res.value, res.detail, res.fix)


@pytest.mark.timeout(600)
def test_harness_checks_pass():
    report = check(GridThermal(**BUSY), laws=False, speed=False)
    assert report.passed, [(c.name, c.value, c.detail) for c in report.failures()]


@pytest.mark.gpu
@pytest.mark.timeout(900)
def test_ledger_and_rosenthal_on_cuda():
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    for name in ("ledger", "rosenthal"):
        law = LAWS[name]
        res = _law(law, None, [fresh(GridThermal(**BUSY), law.config)], "cuda:0")
        assert res.passed, (name, res.value, res.detail)


def test_the_pool_flow_warns_on_cells_too_coarse_for_its_dent(caplog):
    """The flow's dent has the arc pressure's σ (GA: 1.79 mm): cells across the joint coarser than a third of it do
    not resolve it (the short CPU fillet at 1 mm cells left a 2 mm section), so the config says so."""
    law = LAWS["pool_dent"]
    coarse = {**law.config, "y": graded(-0.012, 0.012, 1e-3), "z": graded(-0.006, 0.012, 1e-3)}
    with caplog.at_level("WARNING", logger="alloy.plugins"):
        Config(**law.config)
        assert not [r for r in caplog.records if "pool flow" in r.getMessage()]
        Config(**coarse)
    assert len([r for r in caplog.records if "pool flow" in r.getMessage()]) == 1


def test_cells_of_one_band_are_equal_only_within_volume_tol():
    """Why liquid cells move between cells "of equal volume" within VOLUME_TOL, not exactly (SCR-WP8-14 slice 8): the
    float32 edge widths of one uniform band differ by a few 1e-6, and an exact test refused most moves."""
    from alloy.plugins.thermal.grid_cap import VOLUME_TOL

    for cell in (2.5e-4, 5e-4):
        edges = np.asarray(graded(-0.012, 0.012, cell, band=(-0.004, 0.008)), np.float32)
        w = np.diff(edges)
        band = w[np.abs(w / cell - 1.0) < 1e-3]
        spread = float(band.max() / band.min() - 1.0)
        assert len(band) >= 20 and 1.0e-6 < spread < 0.1 * VOLUME_TOL, (cell, spread)


@wp.kernel
def _carry_one(cell: wp.array2d(dtype=wp.int32), h: wp.array2d(dtype=float), lo: wp.array2d(dtype=float),
               peak: wp.array2d(dtype=float), mass: wp.array(dtype=float)):
    C.carry(cell, h, lo, peak, mass, 0, 0, 1, 3)


def test_a_moved_cell_keeps_its_energy_by_the_ledgers_masses():
    """``carry`` scales H by m_src / m_dst (the ledger's float32 cell masses), so Σ m H does not change when the two
    cells' volumes differ by float32 rounding (WP13's review: the kernel's width ratio would leave ~1e-5)."""
    mass = np.array([1.0, 1.0 + 3.7e-6], np.float32)
    h, lo = np.array([[1.2345678e6, 0.0]], np.float32), np.array([[0.0123, 0.0]], np.float32)
    arr = [wp.array(a, dtype=float, device="cpu") for a in (h, lo, np.array([[1900.0, 300.0]], np.float32))]
    cell = wp.array(np.array([[3, 0]], np.int32), dtype=wp.int32, device="cpu")
    wp.launch(_carry_one, dim=1, inputs=[cell, *arr, wp.array(mass, dtype=float, device="cpu")], device="cpu")
    before = (float(h[0, 0]) + float(lo[0, 0])) * float(mass[0])
    after = (float(arr[0].numpy()[0, 1]) + float(arr[1].numpy()[0, 1])) * float(mass[1])
    assert abs(after / before - 1.0) < 1e-12
    assert cell.numpy().tolist() == [[0, 3]] and arr[2].numpy()[0, 1] == 1900.0
