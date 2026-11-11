# SPDX-License-Identifier: Apache-2.0
"""Surface-flow sponges with porous pads (WP15's porous pads, 0fd62b505; WP15's review of the lumped films, must-fix 2;
WP9). CPU, harness: the take is the film the pad would plough, so it does not depend on the film's stride or the
exchange step; what is over the sponge's capacity goes back to the cells it came from."""

from __future__ import annotations

import pytest

from alloy._vendor import newton

from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.plugins.surface_flow import PlaneChart, SurfaceFlow
from alloy.plugins.surface_flow.applicators import Applicator
from alloy.plugins.surface_flow.lumped_specs import Lumped, Pores

TOOL_HALF = (0.02, 0.04, 0.01)
H0 = 5.0e-4


def _scene(underside: float):
    def build(b) -> None:
        b.add_shape_box(-1, xform=body_xform(0.0, 0.0, -0.01), hx=0.1, hy=0.05, hz=0.01, label="plate")
        xf = body_xform(0.0, 0.0, underside + 0.002 + TOOL_HALF[2])
        sponge = b.add_link(xform=xf, label="sponge")
        b.add_articulation([b.add_joint_prismatic(-1, sponge, parent_xform=xf, axis=newton.Axis.X)])
        b.add_shape_box(sponge, hx=TOOL_HALF[0], hy=TOOL_HALF[1], hz=TOOL_HALF[2],
                        cfg=newton.ModelBuilder.ShapeConfig(density=2.0e6))
    return build


def _run(material: str, gap: float, max_stride: int, dt: float, seconds: float, water: float = 0.0, soap: float = 0.0,
         film_soap: float = 0.0):
    pores = Pores(area=0.04 * 0.07, tool="sponge")
    pad = Applicator(body="sponge", shape="rect", size=(TOOL_HALF[0], 0.035), center=(0.0, 0.0, -TOOL_HALF[2] - 0.002),
                     stiffness=pores.modulus / pores.height, porous=True)
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(40, 10)),), material=material, max_stride=max_stride,
                       tools=(pad,), max_depth=2.0e-3, species=("soap",), lumped=(Lumped("sponge", pores=pores),))

    def init(sim) -> None:
        flow.set_depth(H0)
        flow.set_species(film_soap * H0, "soap")
        flow.lumped.set_water(water)
        flow.lumped.set_species(soap * water, "soap")
    sim = build_simulation(TestScene(build=_scene(gap), init=init, ground=False, gravity=(0.0, 0.0, -9.81), worlds=1,
                                     dt=dt, control_dt=dt), [flow], "cpu")
    for _ in range(int(round(seconds / dt))):
        sim.step_period()
    return flow, pores


@pytest.mark.compile_heavy
def test_a_porous_pad_takes_a_moving_film_alike_at_any_stride_and_step():
    """WP15's must-fix 2: an empty sponge whose porous pad rests 0.1 mm over a 0.5 mm latex film takes up the film it
    would plough while the film flows back under it. Over 0.128 s it takes the same at stride 1 and at the stride the
    film picks under max_stride 8 (4), and at 2 and 1 ms exchanges (1e-4; 1.1e-5 measured; the touch take before:
    7.1e-8 m³ at stride 1, 2.1e-7 at stride 4, 4.5e-8 at 1 ms); the film's ledger closes with to_lumps."""
    takes = {}
    for key, ms, dt in (("s1", 1, 2.0e-3), ("s8", 8, 2.0e-3), ("dt1", 1, 1.0e-3)):
        flow, _ = _run("latex_paint", 1.0e-4, ms, dt, 0.128)
        r, f = flow.lumped.read(), flow.read()
        takes[key] = r["lump_uptake"][0, 0]
        if key == "s8":
            assert flow.stride > 1
        assert f["ledger"][0] == pytest.approx(H0 * flow.tables.area_np.sum(), rel=1e-12, abs=0.0)
        assert r["to_lumps_volume"][0] == pytest.approx(takes[key], rel=1e-12, abs=0.0)
    assert takes["s1"] > 0.0
    assert takes["s8"] == pytest.approx(takes["s1"], rel=1e-4, abs=0.0)
    assert takes["dt1"] == pytest.approx(takes["s1"], rel=1e-4, abs=0.0)


@pytest.mark.compile_heavy
def test_what_a_full_sponge_cannot_hold_goes_back_to_the_cells():
    """A sponge 0.2 mL short of full of 5 kg/m³ wash rests its porous pad 0.2 mm over a 0.5 mm film of 2 kg/m³: its
    pad absorbs over 4x its room in the first exchange with the film's soap; the sponge keeps its capacity and the
    rest goes back to the cells it came from with the sponge's mixed soap, (5·w0 + 2·uptake)/(w0 + uptake) kg/m³
    (1e-12; WP15's rounds 5 and 6: returns of water alone, or ignoring the uptake's soap, passed before), booked
    squeezed and as the film's supplied; the sponge's and the film's books close."""
    w0, c, cf = Pores(area=0.04 * 0.07).free - 2.0e-7, 5.0, 2.0
    flow, pores = _run("water", 2.0e-4, 1, 2.0e-3, 2.0e-3, water=w0, soap=c, film_soap=cf)
    r, f = flow.lumped.read(), flow.read()
    up, sq = r["lump_uptake"][0, 0], r["lump_squeezed"][0, 0]
    sq_sp, up_sp = r["lump_species_squeezed"][0, 0, 0], r["lump_species_uptake"][0, 0, 0]
    assert up_sp == pytest.approx(cf * up, rel=1e-12, abs=0.0)
    assert sq_sp == pytest.approx(sq * (c * w0 + cf * up) / (w0 + up), rel=1e-12, abs=0.0)
    assert r["lump_species"][0, 0, 0] == pytest.approx(c * w0 + up_sp - sq_sp, rel=1e-12, abs=0.0)
    film_soap = f["species_supplied"][0].sum(axis=0)[0] + float((flow.sp.src.numpy()[0, :, 0]
                                                                 * flow.tables.area_np).sum())
    assert film_soap == pytest.approx(sq_sp, rel=1e-12, abs=0.0)                 # the film's supplied is the return
    assert abs(r["lump_species_ledger"][0, 0, 0]) <= 1e-14 * (c * w0 + up_sp)             # 2e-15 of it measured
    assert up > 4.0 * 2.0e-7 and r["lump_water"][0, 0] == pytest.approx(pores.free, rel=1e-12, abs=0.0)
    assert sq == pytest.approx(up - 2.0e-7, rel=1e-12, abs=0.0)
    assert f["supplied_volume"][0] + f["pending_volume"][0] == pytest.approx(sq, rel=1e-12, abs=0.0)
    assert f["ledger"][0] == pytest.approx(H0 * flow.tables.area_np.sum(), rel=1e-12, abs=0.0)
    assert r["lump_water_set"][0, 0] == pores.free - 2.0e-7 and abs(r["lump_ledger"][0, 0]) <= 1e-15 * pores.free
