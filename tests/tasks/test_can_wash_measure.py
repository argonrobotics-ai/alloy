# SPDX-License-Identifier: Apache-2.0
"""rinse-garbage-can measurements (alloy/tasks/can_wash_measure.py) on synthetic film, adsorbed and tank arrays."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.tasks.can_wash_measure import CanProbe, free_ml, soap_kg, tilt_deg, world_normal_z

AREA = np.array([1e-4, 2e-4, 3e-4, 4e-4])          # 4 film cells; the can owns 0-2, its inner region 0-1
PROBE = CanProbe(cells=np.array([0, 1, 2]), inner=np.array([0, 1]), tank=1, soap=0, soap_net=0)


def _quat(axis, deg):
    s = math.sin(math.radians(deg) / 2.0)
    return np.array([[axis[0] * s, axis[1] * s, axis[2] * s, math.cos(math.radians(deg) / 2.0)]])


def test_soap_counts_film_adsorbed_and_the_cavity_tank():
    film = np.zeros((1, 4, 2))
    film[0, :, 0] = 1e-6                            # kg/m² of soap; species 1 is soil
    film[0, :, 1] = 9.0
    out = {"species": film, "adsorbed": np.full((1, 4, 2), 0.5e-6)}
    tank = np.zeros((1, 3, 1))
    tank[0, 1, 0] = 2e-9                            # kg dissolved in the can's cavity (node 1)
    want = 1.5e-6 * AREA[:3].sum() + 2e-9           # cell 3 is not the can's
    assert soap_kg(out, AREA, tank, PROBE)[0] == pytest.approx(want)
    del out["adsorbed"]                             # a flow without adsorption, a network without the species
    assert soap_kg(out, AREA, None, PROBE)[0] == pytest.approx(1e-6 * AREA[:3].sum())


@pytest.mark.parametrize("axis, deg", [((1, 0, 0), 0.0), ((1, 0, 0), 90.0), ((0, 1, 0), 30.0), ((1, 1, 0), 105.0)])
def test_tilt_is_the_axis_angle_from_vertical(axis, deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    assert tilt_deg(_quat(a, deg))[0] == pytest.approx(deg, abs=1e-9)
    nz = world_normal_z(np.array([[0.0, 0.0, 1.0]]), _quat(a, deg))[0, 0]
    assert nz == pytest.approx(math.cos(math.radians(deg)))


def test_world_normals_turn_with_the_can():
    up_y90 = _quat((0, 1, 0), 90.0)                 # +90° about y takes body x to world -z
    n = world_normal_z(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), up_y90)
    assert n[0] == pytest.approx([-1.0, 0.0], abs=1e-12)


def test_free_water_is_the_pool_plus_the_inner_film_above_what_clings():
    depth = np.array([[1.0e-3, 0.3e-3, 5e-3, 5e-3]])   # floor cells 0-1; cells 2-3 not inner
    level = np.array([[1.0, 1.0]])
    got = free_ml(depth, level, AREA, np.array([20e-6]), PROBE)[0]
    assert got == pytest.approx(20.0 + 0.5e-3 * AREA[0] * 1e6)   # LLDPE level floor clings 0.5 mm


@pytest.mark.compile_heavy
def test_soap_reads_the_flow_as_its_own_ledger():
    """On a real SurfaceFlow (WP15's adsorption plate, CPU): the can's detergent read from ``read()`` equals the
    flow's own species_mass + adsorbed_mass over the chart, after the wall has taken some up."""
    from alloy.plugins.harness.scene import TestScene, build_simulation  # noqa: PLC0415
    from alloy.plugins.surface_flow import FilmMaterial, PlaneChart, SurfaceFlow  # noqa: PLC0415
    from alloy.plugins.surface_flow import laws as sl  # noqa: PLC0415
    from alloy.plugins.surface_flow.adsorption import Adsorption  # noqa: PLC0415

    from alloy.tasks.can_wash_measure import GAMMA_M, soap_cells  # noqa: PLC0415

    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(4, 4)),), material=FilmMaterial("flat", surface_tension=0.0),
                       max_depth=2.0e-3, species=("soap",), max_stride=1,
                       adsorption=Adsorption("soap", K=20.0, k_d=0.05, capacity=GAMMA_M))

    def init(sim) -> None:
        flow.set_depth(1.0e-3)
        flow.set_species(0.75 * 1.0e-3, "soap")          # their wash strength, 0.75 kg/m³, in a 1 mm film
        flow.set_adsorbed(1.0e-7, "soap")
    sim = build_simulation(TestScene(build=sl.plate_scene(0.1, 0.1), init=init, ground=False, worlds=1,
                                     gravity=(0.0, 0.0, -9.81), dt=1.0e-2), [flow], "cpu")
    for _ in range(5):
        sim.step_period()
    out, area = flow.read(), flow.tables.area_np
    cells = np.arange(len(area))[flow.tables.cells_of(0)]          # cells_of gives a slice
    probe = CanProbe(cells=cells, inner=cells, tank=0, soap=0, soap_net=-1)
    assert out["adsorbed"][0, :, 0].max() > 1.0e-7                       # the wall took some up
    want = float(out["species_mass"][0, 0] + out["adsorbed_mass"][0, 0])
    assert soap_kg(out, area, None, probe)[0] == pytest.approx(want, rel=1e-12)
    assert soap_cells(out, probe).shape == out["species"].shape[:2]
