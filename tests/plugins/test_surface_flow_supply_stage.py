# SPDX-License-Identifier: Apache-2.0
"""Stage drips (supply_stage.py; WP6's patch-flush sink, WP15's stores) on CPU: what a film stage releases lands where
it falls (a bucket's tank, a chart, a shape, or lost), one exchange in flight; refusals; its tool's own shapes;
species."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from alloy.plugins.errors import PluginError
from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.plugins.surface_flow import PlaneChart, SurfaceFlow
from alloy.plugins.surface_flow.charts import RevolutionChart
from alloy.plugins.surface_flow import laws as sl
from alloy.plugins.surface_flow.supply import Pool
from alloy.plugins.surface_flow.supply_kernels import CHART as HIT_CHART
from alloy.plugins.surface_flow.supply_stage import check_stage, stage_entries
from alloy.plugins.surface_flow.supply_stub import CHART, SLAB, StubFilm, slab_scene, stage_arrays

DT = 0.02                                               # one exchange a period
TANK_RTOL = 1e-9          # a tank's volume is κ·(P - P_ref), P absolute: a small head keeps its rounding
BUCKET = (0.5, 0.0, 0.05)                               # a bucket's floor, off the slab


def _bucket(species: bool = False) -> dict:
    sp = {"species": (0.0,)} if species else {}
    net = {"nodes": [{"name": "bucket", "medium": "tank", "volume": 0.0, "area": 0.04, **sp}], "edges": [],
           "pools": [Pool("bucket", floor=BUCKET, half=(0.1, 0.1))]}
    if species:
        net["species"] = ("soap",)
    return net


def _sim(p, build, ground: bool = False):
    return build_simulation(TestScene(build=build, ground=ground, worlds=2, dt=DT, control_dt=DT), [p], "cpu")


def _release(p, v, pos, vel=None, sp=None) -> None:
    """The stage's requests for this exchange: v [W, E] m³ at world points pos [E, 3] (velocities vel [E, 3])."""
    W, E = np.shape(v)
    p.src_drip_v.assign(np.asarray(v, np.float64))
    p.src_drip_p.assign(np.tile(np.asarray(pos, np.float32)[None], (W, 1, 1)))
    p.src_drip_u.assign(np.tile(np.asarray(vel if vel is not None else np.zeros((E, 3)), np.float32)[None], (W, 1, 1)))
    if sp is not None:
        p.src_drip_sp.assign(np.asarray(sp, np.float64))


def _chord_landing(x0: float, z0: float, vx: float, z_hit: float, reach: float = 1.5, segments: int = 4) -> float:
    """x where the caster's chords of the path from (x0, z0) at vx (gravity 9.81) first reach z_hit."""
    t_end = 2.0 * reach / (vx + np.sqrt(vx * vx + 2.0 * 9.81 * reach))
    ts = np.linspace(0.0, t_end, segments + 1)
    x, z = x0 + vx * ts, z0 - 0.5 * 9.81 * ts * ts
    k = int(np.argmax(z < z_hit)) - 1
    return float(x[k] + (x[k + 1] - x[k]) * (z[k] - z_hit) / (z[k] - z[k + 1]))


def _flight(supply) -> np.ndarray:
    """[W] m³ the stages' drips have in flight (their columns of the discharge table, < 0 out)."""
    q = supply.nozzle_q.numpy()[:, supply.stage_col0:supply.stage_col0 + supply.n_stage]
    return -q.sum(axis=1) * DT


def test_stage_drips_land_in_a_bucket_on_a_chart_or_are_lost():
    """Three tools release for three exchanges (different amounts per world): one over the slab, one over a bucket's
    pool, one over nothing. Every exchange staged = sent + in flight; at the end the slab holds the first's, the
    bucket's tank the second's (to_tank), the third's is lost, and nothing is refused."""
    p = StubFilm(charts=(CHART,), network=_bucket(), stage=3)
    sim = _sim(p, slab_scene())
    pos = ((0.05, 0.0, 0.2), (BUCKET[0], 0.0, 0.3), (3.0, 0.0, 0.3))
    total = np.zeros((2, 3))
    for k in range(4):
        v = np.array([[1e-5, 2e-5, 3e-5], [4e-5, 1e-5, 5e-6]]) * (k + 1) if k < 3 else np.zeros((2, 3))
        _release(p, v, pos)
        total += v
        sim.step_period()
        r = p.supply.read()
        assert np.allclose(r["staged"], total.sum(axis=1), rtol=1e-12, atol=0.0)
        assert np.allclose(r["staged"], r["sent"] + _flight(p.supply), rtol=1e-12, atol=0.0)
        assert (p.src_drip_v.numpy() == 0.0).all()                              # consumed each exchange
    r = p.supply.read()
    assert (_flight(p.supply) == 0.0).all() and (r["refused"] == 0.0).all()
    assert np.allclose(p.supplied.numpy().sum(axis=1), total[:, 0], rtol=1e-12, atol=0.0)
    assert np.allclose(r["to_tank"], total[:, 1], rtol=1e-12, atol=0.0)
    assert np.allclose(r["amount"][:, 0], total[:, 1], rtol=TANK_RTOL, atol=0.0)
    assert np.allclose(r["lost"], total[:, 2], rtol=1e-12, atol=0.0)
    assert (r["caught"] == 0.0).all()


def test_a_negative_release_is_refused_and_a_drip_starts_at_its_point_and_velocity():
    """A negative volume is refused (and goes nowhere); a drip thrown sideways at 1 m/s from 0.2 m over the slab lands
    where the caster's path from that point (lifted 1 mm) and velocity meets the slab: its 32 chords over the 1.5 m
    reach, 0.12 mm short of the parabola (4 chords: 8.6 mm, WP15). A tool pressed 0.1 mm into the slab
    still lays its drip on it."""
    p = StubFilm(charts=(CHART,), network=_bucket(), stage=1)
    sim = _sim(p, slab_scene())
    _release(p, [[-1e-5], [-2e-5]], ((0.0, 0.0, 0.2),))
    sim.step_period()
    sim.step_period()
    r = p.supply.read()
    assert np.array_equal(r["refused"], [1e-5, 2e-5]) and (r["staged"] == 0.0).all() and (r["sent"] == 0.0).all()
    _release(p, [[1e-5], [1e-5]], ((-0.15, 0.0, 0.2),), vel=((1.0, 0.0, 0.0),))
    sim.step_period()
    sim.step_period()
    hit = p.supply.hit["p"].numpy()[:, p.supply.ray_starts[0]]
    assert np.allclose(hit[:, 0], _chord_landing(-0.15, 0.201, 1.0, 0.02, segments=32), rtol=0.0, atol=1e-6)
    assert np.allclose(hit[:, 0], -0.15 + np.sqrt(2.0 * 0.181 / 9.81), rtol=0.0, atol=2e-4)     # the parabola
    assert (p.supply.hit["kind"].numpy()[:, p.supply.ray_starts[0]] == HIT_CHART).all()   # (hit: the chart's frame)
    assert np.allclose(p.supplied.numpy().sum(axis=1), 1e-5, rtol=1e-12, atol=0.0)
    _release(p, [[2e-5], [2e-5]], ((0.05, 0.0, 0.02 - 1e-4),))                   # 0.1 mm inside the slab
    sim.step_period()
    sim.step_period()
    assert np.allclose(p.supplied.numpy().sum(axis=1), 3e-5, rtol=1e-12, atol=0.0)


def _tool(b) -> None:
    """The slab scene and a tool (a box on a fixed link) around the drips' exit point."""
    slab_scene()(b)
    link = b.add_link(xform=body_xform(0.0, 0.0, 0.2), label="tool")
    b.add_shape_box(link, hx=0.03, hy=0.03, hz=0.03)
    b.add_articulation([b.add_joint_fixed(-1, link, parent_xform=body_xform(0.0, 0.0, 0.2))])


@pytest.mark.parametrize("own", [False, True])
def test_a_drip_passes_its_own_tools_shapes(own):
    """A drip released inside its tool's box: with ``src_drip_body`` naming the tool it passes the box to the slab and
    pushes the tool not at all; with no body the box catches it (and its momentum)."""
    p = StubFilm(charts=(CHART,), network=_bucket(), stage=1, stage_bodies=("tool",) if own else ())
    sim = _sim(p, _tool)
    _release(p, [[1e-5], [1e-5]], ((0.0, 0.0, 0.2),), vel=((0.5, 0.0, 0.0),))
    sim.step_period()
    forces = SimpleNamespace(body_f=wp.zeros(sim.model.body_count, dtype=wp.spatial_vector, device="cpu"))
    p.supply.apply(sim.model.state(), forces, DT)                              # in flight: it pushes no tool
    f = forces.body_f.numpy()[p.ctx.bodies("tool").numpy()]
    assert (f == 0.0).all() if own else (np.abs(f).sum(axis=1) > 0.0).all()   # (caught: it strikes the box)
    sim.step_period()
    r = p.supply.read()
    landed, caught = p.supplied.numpy().sum(axis=1), r["caught"]
    assert np.allclose(landed if own else caught, 1e-5, rtol=1e-12, atol=0.0)
    assert ((caught if own else landed) == 0.0).all()


def test_the_stage_registers_all_its_arrays_with_their_shapes():
    """All of the arrays or none; each with its shape and type (a PluginError names the wrong one)."""
    ctx = SimpleNamespace(world_count=2, zeros=lambda s, t: wp.zeros(s, dtype=t, device="cpu"),
                          full=lambda s, v, t: wp.full(s, v, dtype=t, device="cpu"))
    film = SimpleNamespace(**stage_arrays(ctx, 3, n_film=2))
    assert stage_entries(SimpleNamespace()) == 0 and stage_entries(film) == 3
    check_stage(ctx, film, 2)
    with pytest.raises(PluginError, match="src_drip_sp"):
        check_stage(ctx, film, 1)
    film.src_drip_u = wp.zeros((2, 3), dtype=wp.vec3f, device="cpu")
    film.src_drip_v = wp.zeros((2, 3), dtype=wp.float32, device="cpu")
    with pytest.raises(PluginError, match="src_drip_v"):
        check_stage(ctx, film, 2)
    del film.src_drip_p
    with pytest.raises(PluginError, match="src_drip_p"):
        stage_entries(film)


class _StagedFlow(SurfaceFlow):
    """WP15's film with a stage that registers three drip entries before the supply's setup (as its stores will)."""

    name = "sf_stage_test"

    def setup(self, ctx) -> None:
        for name, a in stage_arrays(ctx, 3, n_film=len(self.config.species)).items():
            setattr(self, name, self.add_state(name, a))
        super().setup(ctx)


def test_stage_drips_carry_their_species_onto_the_film_and_into_a_tank():
    """Drips carrying soap and a dye onto a walled plate and into a bucket whose network carries soap only: the film
    gains each species that landed on it (the dye too: a species the network lacks), the tank its entry's soap (20
    kg/m³ there, 10 on the plate: each entry its own) and the dye goes away; the species ledger's staged = sent +
    to_tank + away; soap released with no water is refused; the film's species ledger closes."""
    flow = _StagedFlow(charts=(PlaneChart("plate", cells=(20, 20), edges=("wall",) * 4),), material="water",
                       species=("soap", "dye"), network=_bucket(species=True))
    sim = build_simulation(TestScene(build=sl.plate_scene(0.2, 0.2), ground=False, worlds=2, dt=DT, control_dt=DT),
                           [flow], "cpu")
    r0 = flow.read()
    v = np.array([[1e-5, 2e-5, 0.0], [3e-5, 1e-5, 0.0]])
    sp = np.stack([v * (10.0, 20.0, 0.0), 5.0 * v], axis=2)                       # [W, E, (soap, dye)]
    sp[:, 2, 0] = (1e-6, 2e-6)                                                    # soap with no water
    _release(flow, v, ((0.0, 0.0, 0.1), (BUCKET[0], 0.0, 0.3), (0.0, 0.0, 0.1)), sp=sp)
    for _ in range(3):
        sim.step_period()
    r = flow.read()
    gain = r["species_mass"] - r0["species_mass"]
    assert np.allclose(gain[:, 0], 10.0 * v[:, 0], rtol=1e-9, atol=0.0)
    assert np.allclose(gain[:, 1], 5.0 * v[:, 0], rtol=1e-9, atol=0.0)          # the network lacks it: still lands
    assert np.allclose(flow.supply.fluid.species.numpy()[:, 0, 0], 20.0 * v[:, 1], rtol=TANK_RTOL, atol=0.0)
    sent, tank, away = (r[f"supply_species_{k}"] for k in ("sent", "to_tank", "away"))
    assert np.allclose(sent, gain, rtol=1e-12, atol=0.0)                         # onto the plate
    assert np.allclose(tank[:, 0], 20.0 * v[:, 1], rtol=1e-12, atol=0.0) and (tank[:, 1] == 0.0).all()
    assert np.allclose(away[:, 1], 5.0 * v[:, 1], rtol=1e-12, atol=0.0) and (away[:, 0] == 0.0).all()
    assert np.allclose(r["supply_species_staged"], sent + tank + away, rtol=1e-12, atol=0.0)
    assert np.allclose(r["supply_species_staged"], sp[:, :2].sum(axis=1), rtol=1e-12, atol=0.0)
    assert np.array_equal(r["supply_species_refused"][:, 0], [1e-6, 2e-6])
    assert np.abs(r["species_ledger"] - r0["species_ledger"]).max() <= 1e-12 * r["species_mass"].max()
    assert np.allclose(r["supply_staged"], v.sum(axis=1), rtol=1e-12, atol=0.0)


def test_a_drop_lands_whole_in_its_cell_on_a_standing_film():
    """A 10 µL drop at a 2 ms exchange (5 mL/s over it) onto 0.3 mm of standing film: all of it in the cell it
    strikes, nothing swept (through the jet's jump it swept the film aside and left cells dry, WP15)."""
    p = StubFilm(charts=(CHART,), network=_bucket(), stage=1)
    sim = build_simulation(TestScene(build=slab_scene(), ground=False, worlds=2, dt=2e-3, control_dt=2e-3), [p], "cpu")
    p.h.assign(np.full(p.h.shape, 3e-4))
    _release(p, [[1e-8], [1e-8]], ((0.0, 0.0, 0.1),))
    sim.step_period()
    sim.step_period()
    dh = p.h.numpy() - 3e-4
    area = p.tables.area.numpy()
    assert ((dh != 0.0).sum(axis=1) == 1).all() and np.allclose((dh * area).sum(axis=1), 1e-8, rtol=1e-9, atol=0.0)
    assert (p.supply.read()["swept"] == 0.0).all()


def _colander(b) -> None:
    """The slab scene and a colander (30 mm radius, 100 mm deep) on a fixed link, its floor 0.25 m up."""
    slab_scene()(b)
    link = b.add_link(xform=body_xform(0.0, 0.0, 0.25), label="colander")
    b.add_shape_box(link, xform=body_xform(0.0, 0.0, -0.003), hx=0.04, hy=0.04, hz=0.002)
    b.add_articulation([b.add_joint_fixed(-1, link, parent_xform=body_xform(0.0, 0.0, 0.25))])


def test_stage_drips_beside_a_vessels_drips_keep_their_own_columns():
    """A dripping colander over the slab and a tool releasing over nothing: the tool's columns come after the
    vessel's in the discharge table, so the colander's drips land on the slab (holes = supplied + in flight) and the
    tool's are lost (staged = lost + in flight), every exchange."""
    a = np.pi * 0.03 ** 2
    net = {"nodes": [{"name": "colander", "medium": "tank", "volume": 0.05 * a, "area": a}], "edges": [],
           "pools": [Pool("colander", body="colander", cavity=((0.03, 0.0), (0.03, 0.1)), holes=((0.0, 2e-5),),
                          drip_rays=4)]}
    p = StubFilm(charts=(CHART,), network=net, stage=1)
    sim = _sim(p, _colander)
    total = np.zeros(2)
    for k in range(4):
        v = np.array([[1e-5], [3e-5]]) * (k < 3)
        _release(p, v, ((3.0, 0.0, 0.3),))
        total += v[:, 0]
        sim.step_period()
        r = p.supply.read()
        vessel = p.supply.drip_q.numpy()[:, 0] * DT
        assert (r["holes"] > 0.0).all()
        assert np.allclose(r["holes"], p.supplied.numpy().sum(axis=1) + vessel, rtol=1e-12, atol=0.0)
        assert np.allclose(r["staged"], r["lost"] + _flight(p.supply), rtol=1e-12, atol=0.0)
        assert np.allclose(r["staged"], total, rtol=1e-12, atol=0.0)
    assert np.allclose(p.supply.read()["lost"], total, rtol=1e-12, atol=0.0)


CELL = 0.01                                             # the slab's and the plate's cells


def _ga_cone(xy, corner, n: int, v: float) -> tuple[np.ndarray, float]:
    """GA's gob (concrete_finishing/coupling.py:978-986 with _lay): rad = max(2, ceil((3v/π)^(1/3)/cell)) cells about
    ``xy``, weights max(1 - r/rad, 0) over the lattice continued past an n x n chart's edges: the volume each chart
    cell gets (i-major, as the chart's cells) and the share past the edges."""
    rad = max(2, int(np.ceil((3.0 * v / np.pi) ** (1.0 / 3.0) / CELL)))
    centre = (np.asarray(xy) - corner) / CELL
    i0, j0 = int(centre[0]) - rad, int(centre[1]) - rad
    ii, jj = np.meshgrid(np.arange(i0, i0 + 2 * rad + 1), np.arange(j0, j0 + 2 * rad + 1), indexing="ij")
    w = np.maximum(1.0 - np.hypot(ii + 0.5 - centre[0], jj + 0.5 - centre[1]) / rad, 0.0)
    share = w * v / w.sum()
    inside = (ii >= 0) & (ii < n) & (jj >= 0) & (jj < n)
    out = np.zeros(n * n)
    out[ii[inside] * n + jj[inside]] = share[inside]
    return out, float(share[~inside].sum())


def _land(v, xy, chart=CHART):
    """One stage drip per world (v [W] m³ at xy [W, 2], 0.1 m over the slab, its chart ``chart``) cast and laid: the
    stub film's cells' supplied volume [W, C] and the supply's read."""
    p = StubFilm(charts=(chart,), network=_bucket(), stage=1)
    sim = _sim(p, slab_scene())
    p.src_drip_v.assign(np.asarray(v, np.float64)[:, None])
    p.src_drip_p.assign(np.asarray([[(x, y, 0.1)] for x, y in xy], np.float32))
    sim.step_period()
    sim.step_period()
    return p.supplied.numpy(), p.supply.read()


def test_a_trowels_load_lands_as_gas_cone():
    """WP6's margin trowel releases 38 mL (world 1: 12 mL) 3 and 4 mm off a cell's centre on 10 mm cells: every cell
    gets GA's share of the cone of R = (3V/π)^(1/3) rounded up to whole cells (4 and 3), the volume exactly, nothing
    lost."""
    v, xy = np.array([3.8e-5, 1.2e-5]), np.array([[0.0031, -0.0042], [0.0031, -0.0042]])
    got, r = _land(v, xy)
    for w in range(2):
        want, off = _ga_cone(xy[w], -SLAB[0], 40, v[w])
        assert off == 0.0 and np.array_equal(got[w] > 0.0, want > 0.0) and (got[w] > 0.0).sum() > 20
        assert np.allclose(got[w], want, rtol=1e-5, atol=1e-6 * want.max())
    assert np.allclose(got.sum(axis=1), v, rtol=1e-12, atol=0.0) and (r["lost"] == 0.0).all()


def test_a_drip_at_the_charts_edge_books_what_falls_past_it_lost():
    """38 mL 5 mm inside the slab's edge (world 1: 12 mL, 2 mm inside a corner): the cells inside get GA's share of the
    whole cone (no thicker than a whole one) and the rest is lost: sent = supplied + lost (1e-12)."""
    v, xy = np.array([3.8e-5, 1.2e-5]), np.array([[0.195, 0.0031], [0.198, -0.198]])
    got, r = _land(v, xy)
    for w in range(2):
        want, off = _ga_cone(xy[w], -SLAB[0], 40, v[w])
        assert 0.3 * v[w] < off < 0.8 * v[w] and np.array_equal(got[w] > 0.0, want > 0.0)
        assert np.allclose(got[w], want, rtol=1e-5, atol=1e-6 * want.max())
        assert r["lost"][w] == pytest.approx(off, rel=1e-5)
    assert np.allclose(r["sent"], got.sum(axis=1) + r["lost"], rtol=1e-12, atol=0.0)
    assert np.allclose(r["staged"], v, rtol=1e-12, atol=0.0)


def test_a_drop_up_to_one_cell_lands_whole_and_past_it_spreads_over_two():
    """A drop with (3V/π)^(1/3) a thousandth under one cell lands whole in its cell; a thousandth over spreads over
    GA's two-cell cone."""
    v1 = np.pi * CELL ** 3 / 3.0
    v, xy = np.array([0.999 * v1, 1.001 * v1]), np.array([[0.0031, -0.0042], [0.0031, -0.0042]])
    got, _ = _land(v, xy)
    want, _ = _ga_cone(xy[1], -SLAB[0], 40, v[1])
    assert (got[0] > 0.0).sum() == 1 and np.array_equal(got[1] > 0.0, want > 0.0) and (want > 0.0).sum() >= 9
    assert np.allclose(got.sum(axis=1), v, rtol=1e-12, atol=0.0)


def test_on_oblong_cells_the_cone_rounds_to_the_smaller_step():
    """10 x 5 mm cells: 38 mL's R = 33 mm rounds up to 7 steps of 5 mm (35 mm, not 4 of 10 mm), each cell weighed by
    its centre's distance in metres; every cell gets its share (1e-5) and the whole volume lands (1e-12)."""
    v, xy = np.array([3.8e-5, 3.8e-5]), np.array([[0.0031, -0.0042], [-0.0123, 0.0071]])
    got, _ = _land(v, xy, PlaneChart("slab", cells=(40, 80), edges=CHART.edges))
    x, y = np.meshgrid(-0.2 + (np.arange(40) + 0.5) * 0.01, -0.2 + (np.arange(80) + 0.5) * 0.005, indexing="ij")
    for w in range(2):
        wt = np.maximum(1.0 - np.hypot(x - xy[w, 0], y - xy[w, 1]).ravel() / 0.035, 0.0)
        want = v[w] * wt / wt.sum()
        assert np.array_equal(got[w] > 0.0, want > 0.0) and np.allclose(got[w], want, rtol=1e-5, atol=1e-6 * want.max())
    assert np.allclose(got.sum(axis=1), v, rtol=1e-12, atol=0.0)


def test_a_trowels_load_on_a_tub_floor_is_all_laid():
    """WP15's probe: 38 mL onto a tub's floor (a revolution chart from r = 21 mm, 128 columns, 2 mm rows) at r 30 mm,
    θ 0.01: all of it laid (1e-12), nothing lost (a plane's off-chart lattice read on the revolution chart lost 1.2 %)."""
    tub = RevolutionChart(profile=((0.021, 0.0), (0.15, 0.0), (0.15, 0.1), (0.153, 0.1), (0.153, 0.0)),
                          n_theta=128, ds=2e-3, frame=(0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 1.0))
    p = StubFilm(charts=(CHART, tub), network=_bucket(), stage=1)
    sim = _sim(p, slab_scene(slab_x=1.0, crate=(-1.0, 0.0)))
    _release(p, [[3.8e-5], [3.8e-5]], ((0.03 * np.cos(0.01), 0.03 * np.sin(0.01), 0.3),))
    sim.step_period()
    sim.step_period()
    got = p.supplied.numpy()[:, p.tables.cells_of(1)]
    assert ((got > 0.0).sum(axis=1) > 50).all() and np.allclose(got.sum(axis=1), 3.8e-5, rtol=1e-12, atol=0.0)
    assert (p.supply.read()["lost"] == 0.0).all()


def test_a_soapy_drip_past_the_edge_loses_its_soap_with_its_water():
    """On WP15's film, 38 mL of 10 kg/m³ soap 5 mm inside the plate's edge: the film gains the soap of what it laid,
    the species ledger's lost the rest; staged = sent + lost; the film's species ledger closes."""
    flow = _StagedFlow(charts=(PlaneChart("plate", cells=(20, 20), edges=("wall",) * 4),), material="water",
                       species=("soap", "dye"), network=_bucket(species=True))
    sim = build_simulation(TestScene(build=sl.plate_scene(0.2, 0.2), ground=False, worlds=2, dt=DT, control_dt=DT),
                           [flow], "cpu")
    r0 = flow.read()
    v = np.array([[3.8e-5, 0.0, 0.0], [3.8e-5, 0.0, 0.0]])
    _release(flow, v, ((0.095, 0.0, 0.1), (0.0, 0.0, 0.1), (0.0, 0.0, 0.1)),
             sp=np.stack([10.0 * v, np.zeros_like(v)], axis=2))
    for _ in range(3):
        sim.step_period()
    r = flow.read()
    lost = r["supply_lost"]
    _, off = _ga_cone((0.095, 0.0), -0.1, 20, 3.8e-5)
    assert np.allclose(lost, off, rtol=1e-5, atol=0.0)
    gain = r["species_mass"] - r0["species_mass"]
    assert np.allclose(gain[:, 0], 10.0 * (3.8e-5 - lost), rtol=1e-9, atol=0.0)
    sent, sp_lost = r["supply_species_sent"], r["supply_species_lost"]
    assert np.allclose(sp_lost[:, 0], 10.0 * lost, rtol=1e-12, atol=0.0) and np.allclose(sent, gain, rtol=1e-9, atol=0)
    assert np.allclose(r["supply_species_staged"], sent + sp_lost, rtol=1e-12, atol=0.0)
    assert np.abs(r["species_ledger"] - r0["species_ledger"]).max() <= 1e-12 * r["species_mass"].max()


def test_a_drip_into_a_cup_spreads_over_its_floor_and_stays_on_it():
    """3 mL onto a cup's floor (a revolution chart, 2 mm rows) 15 mm from the floor-wall fold, on its θ = 0 seam: its
    cone (R 16 mm) spreads over the floor's cells within its fold span, across the seam, all of it laid there (nothing
    lost, none on the wall or outside)."""
    cup = RevolutionChart(profile=((0.006, 0.0), (0.035, 0.0), (0.035, 0.14), (0.038, 0.14), (0.038, 0.0)),
                          n_theta=32, ds=2e-3, frame=(0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 1.0))
    p = StubFilm(charts=(CHART, cup), network=_bucket(), stage=1)
    sim = _sim(p, slab_scene(slab_x=1.0, crate=(-1.0, 0.0)))
    _release(p, [[3e-6], [3e-6]], ((0.02, 0.0, 0.3),))
    sim.step_period()
    sim.step_period()
    got = p.supplied.numpy()[:, p.tables.cells_of(1)].reshape(2, *p.tables.shapes[1])
    floor = (np.arange(got.shape[1]) + 0.5) * 2e-3 < 0.029                      # rows on the floor (29 mm of profile)
    assert ((got > 0.0).sum(axis=(1, 2)) > 20).all() and (got[:, ~floor] == 0.0).all()
    assert (got[:, :, 0] > 0.0).any(axis=1).all() and (got[:, :, -1] > 0.0).any(axis=1).all()     # across the seam
    assert np.allclose(got.sum(axis=(1, 2)), 3e-6, rtol=1e-12, atol=0.0) and (p.supply.read()["lost"] == 0.0).all()
