# SPDX-License-Identifier: Apache-2.0
"""Surface flow, a species mass clipped at zero (cell_kernels.carry_species): a donor's concentration that takes a
cell below zero is clipped before the drains and pools take, and booked in ``species_clipped`` as the water's clip,
so ``species_out`` is what left and equals what its destinations book (the drops' shed, the drains and pools: WP9's
identity) while the ledger stays exact (CPU)."""

from __future__ import annotations

import numpy as np
import warp as wp

from alloy.plugins.harness.scene import TestScene, build_simulation
from alloy.plugins.surface_flow import FilmMaterial, PlaneChart, SurfaceFlow
from alloy.plugins.surface_flow import cell_kernels as CK
from tests.plugins.test_surface_flow import _plate

AREA = 1.0e-4                   # m²: one cell
M0 = 1.0e-6                     # kg/m²: each cell's start load, 1e-10 kg


@wp.kernel
def _carry(sp: CK.Species, book: CK.Book, dm: wp.array[CK.vec4d], edge: wp.array[CK.vec4d],
           moved: wp.array[wp.float64], left: wp.array[wp.float64]):
    c = wp.tid()
    CK.carry_species(sp, book, 0, c, wp.float64(AREA), wp.float64(0.01), wp.float64(100.0), 1, dm[c], edge[c],
                     moved[c], left[c], wp.float64(0.0))


def _species(cells: int) -> CK.Species:
    sp = CK.Species()
    sp.count = 1
    sp.mass = wp.array(np.full((1, cells, 1), M0), dtype=wp.float64, device="cpu")
    sp.conc, sp.src, sp.supplied, sp.out, sp.shed, sp.drained, sp.clipped = (
        wp.zeros((1, cells, 1), dtype=wp.float64, device="cpu") for _ in range(7))
    sp.src_conc = wp.zeros(1, dtype=wp.float64, device="cpu")
    return sp


def test_a_clipped_cell_books_the_clip_apart_and_its_destinations_match_out():
    """Four cells, each starting with 1e-10 kg:
    - cell 0: 3e-10 kg leaves over an open edge with no drain;
    - cell 1: the same over a drained edge;
    - cell 2: the same, and the cell is under a pool that takes half its film;
    - cell 3: under the same pool, losing only 0.5e-10 kg.
    The first three clip 2e-10 kg into ``clipped``. ``out`` books the 3e-10 that left, and it equals ``shed`` plus
    ``drained`` (the old rule booked 1e-10 in ``out`` against 3e-10 shed, and gave cell 2's pool -1e-10 kg). Cell 3
    is untouched by the rule. Every cell's mass + out + clipped - supplied stays at its start, and no drain or pool
    gets a negative mass."""
    n = 4
    sp = _species(n)
    book = CK.Book()
    book.drain_of = wp.array(np.array([-1, 0, -1, -1], np.int32), dtype=wp.int32, device="cpu")
    book.flooded = wp.array(np.array([[-1, -1, 0, 0]], np.int32), dtype=wp.int32, device="cpu")
    lost = np.array([3.0e-10, 3.0e-10, 3.0e-10, 0.5e-10])
    dm = wp.array([CK.vec4d(-x, 0.0, 0.0, 0.0) for x in lost], dtype=CK.vec4d, device="cpu")
    edge = wp.array([CK.vec4d(x, 0.0, 0.0, 0.0) for x in lost], dtype=CK.vec4d, device="cpu")
    moved = wp.array(np.array([0.0, 0.0, 1.0e-9, 1.0e-9]), dtype=wp.float64, device="cpu")
    left = wp.array(np.array([0.0, 0.0, 0.5e-9, 0.5e-9]), dtype=wp.float64, device="cpu")
    wp.launch(_carry, dim=n, inputs=[sp, book, dm, edge, moved, left], device="cpu")
    mass, out, shed, drained, clipped, supplied = (a.numpy()[0, :, 0] * f for a, f in (
        (sp.mass, AREA), (sp.out, 1.0), (sp.shed, 1.0), (sp.drained, AREA), (sp.clipped, 1.0), (sp.supplied, 1.0)))
    start = M0 * AREA
    np.testing.assert_allclose(clipped, [start - 3.0e-10] * 3 + [0.0], rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(out, [3.0e-10, 3.0e-10, 3.0e-10, 0.5e-10 + 0.25e-10], rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(out, shed + drained, rtol=1e-12, atol=0.0)              # WP9's identity
    np.testing.assert_allclose(drained, [0.0, 3.0e-10, 0.0, 0.25e-10], rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(mass, [0.0, 0.0, 0.0, 0.25e-10], rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(mass + out + clipped - supplied, start, rtol=1e-12, atol=0.0)
    assert (drained >= 0.0).all() and (mass >= 0.0).all()


def test_the_species_ledger_reads_the_clip():
    """``read()`` gives ``species_clipped`` and ``species_ledger`` = mass + out + clipped - supplied: a clip booked by
    hand moves the ledger by exactly it."""
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(4, 4)),), material=FilmMaterial("w", surface_tension=0.0),
                       max_depth=2.0e-3, species=("soap",))
    build_simulation(TestScene(build=_plate(False), ground=False, gravity=(0.0, 0.0, 0.0), worlds=1, dt=2.0e-3),
                     [flow], "cpu")
    before = flow.read()
    assert np.array_equal(before["species_clipped"], np.zeros_like(before["species_clipped"]))
    clip = np.zeros(flow.sp.clipped.shape)
    clip[0, 5, 0] = -3.0e-11
    flow.sp.clipped.assign(clip)
    after = flow.read()
    assert after["species_ledger"][0, 0] - before["species_ledger"][0, 0] == -3.0e-11
