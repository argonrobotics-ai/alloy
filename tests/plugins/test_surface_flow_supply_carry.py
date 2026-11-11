# SPDX-License-Identifier: Apache-2.0
"""A vessel's water as mass on its body (supply_carry.py, G2: JOB3's can tipped and emptied) on the stub film, CPU: the
body's mass, centre and inertia in the model's arrays during apply, per world, and the dry body between exchanges; the
solver moves the body by them (a lift and a couple on a free glass; MuJoCo Warp and Alloy's MuJoCo C route); a spinning
glass emptied keeps its velocity; the solvers that take run-time masses (the rest refused)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
import alloy._vendor.newton.solvers as ns

from alloy.plugins.base import ForcePlugin
from alloy.plugins.errors import PluginError
from alloy.plugins.funcs import add_wrench
from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.plugins.host import inertia_pushes
from alloy.plugins.surface_flow.charts import PlaneChart
from alloy.plugins.surface_flow.supply import Pool
from alloy.plugins.surface_flow.supply_stub import StubFilm

G, RHO = 9.81, 998.0
FAR = PlaneChart("far", cells=(2, 2))
R, HT = 0.03, 0.1                                      # the glass's inside: 30 mm radius, 100 mm deep
BALL = (0.0, 0.0, 0.05, 0.03, 1000.0)                   # its dry mass: a sphere (centre, radius, density) on the body
M0 = 4.0 / 3.0 * math.pi * BALL[3] ** 3 * BALL[4]      # 0.113 kg
DEPTH = (0.06, 0.03)                                   # each world's water (0.17 and 0.085 kg)
FIELDS = ("body_mass", "body_inv_mass", "body_com", "body_inertia", "body_inv_inertia")


@wp.kernel
def _push(b: wp.array[wp.int32], f: wp.vec3, tau: wp.vec3, body_f: wp.array[wp.spatial_vector]):
    w = wp.tid()
    add_wrench(body_f, b[w], wp.spatial_vector(f[0], f[1], f[2], tau[0], tau[1], tau[2]))   # at the centre of mass


class _Push(ForcePlugin):
    """A constant force and couple on the glass at its centre of mass."""

    name = "push_test"

    def __init__(self, force=(0.0, 0.0, 0.0), torque=(0.0, 0.0, 0.0)) -> None:
        super().__init__()
        self.f, self.tau = wp.vec3(*force), wp.vec3(*torque)

    def setup(self, ctx) -> None:
        self.b = ctx.bodies("glass")

    def apply(self, state, forces, dt) -> None:
        wp.launch(_push, dim=self.ctx.world_count, inputs=[self.b, self.f, self.tau], outputs=[forces.body_f],
                  device=self.ctx.device)


@wp.kernel
def _look(b: wp.array[wp.int32], mass: wp.array[wp.float32], inv_mass: wp.array[wp.float32], com: wp.array[wp.vec3],
          inertia: wp.array[wp.mat33], inv_inertia: wp.array[wp.mat33], out_m: wp.array[wp.vec2],
          out_c: wp.array[wp.vec3], out_i: wp.array[wp.mat33], out_ii: wp.array[wp.mat33]):
    w = wp.tid()
    out_m[w] = wp.vec2(mass[b[w]], inv_mass[b[w]])
    out_c[w] = com[b[w]]
    out_i[w] = inertia[b[w]]
    out_ii[w] = inv_inertia[b[w]]


class _Look(_Push):
    """A plug-in after the flow (a hand): the glass's mass, centre and inertia as it reads them in apply."""

    name = "look_test"

    def setup(self, ctx) -> None:
        super().setup(ctx)
        W = ctx.world_count
        self.m, self.c = ctx.zeros(W, wp.vec2), ctx.zeros(W, wp.vec3)
        self.i, self.ii = ctx.zeros(W, wp.mat33), ctx.zeros(W, wp.mat33)

    def apply(self, state, forces, dt) -> None:
        m = self.ctx.model
        wp.launch(_look, dim=self.ctx.world_count,
                  inputs=[self.b, m.body_mass, m.body_inv_mass, m.body_com, m.body_inertia, m.body_inv_inertia],
                  outputs=[self.m, self.c, self.i, self.ii], device=self.ctx.device)


def _build(tilt: float = 0.0):
    q = (math.sin(tilt / 2.0), 0.0, 0.0, math.cos(tilt / 2.0))

    def build(b) -> None:
        b.add_shape_box(-1, xform=body_xform(2.0, 0.0, 0.01), hx=0.05, hy=0.05, hz=0.01, label="far")
        glass = b.add_body(xform=body_xform(0.0, 0.0, 1.0, q), label="glass")
        b.add_shape_sphere(glass, xform=body_xform(*BALL[:3]), radius=BALL[3],
                           cfg=newton.ModelBuilder.ShapeConfig(density=BALL[4]))
    return build


def _set_depths(p: StubFilm, depths) -> None:
    """Each world's glass tank holding depths[w] of water (κ·ΔP = V)."""
    f, k = p.supply.fluid, p.supply.net.node_index("glass")
    pr = f.pressure.numpy()
    kappa = p.supply.net.tank_kappa(f.gravity)["glass"]
    vol = p.supply.fluid.amount(pr, f.volume.numpy())[:, k]
    pr[:, k] += (np.asarray(depths) * math.pi * R * R - vol) / kappa
    f.pressure.assign(pr)


def _sim(carry: str, push: _Push | None = None, graph: bool = False, tilt: float = 0.0, spin: float = 0.0):
    """The glass on a free body holding each world's DEPTH of water (set before the first exchange), turning at
    ``spin`` rad/s about x."""
    A = math.pi * R * R
    net = {"nodes": [{"name": "glass", "medium": "tank", "volume": A * DEPTH[0], "area": A}], "edges": [],
           "pools": [Pool("glass", body="glass", cavity=((R, 0.0), (R, HT)), carry=carry)]}
    p = StubFilm(charts=(FAR,), network=net)
    sim = build_simulation(TestScene(build=_build(tilt), ground=False, worlds=2),
                           [p] + ([push] if push else []), "cpu", graph=graph)
    _set_depths(p, DEPTH)
    if spin:
        jqd, bqd = sim.state.joint_qd.numpy(), sim.state.body_qd.numpy()
        jqd.reshape(-1, 6)[:, 3] = spin
        bqd[:, 3] = spin
        sim.state.joint_qd.assign(jqd)
        sim.state.body_qd.assign(bqd)
    return p, sim, int(p.supply.pbody.numpy()[0, 0]), int(p.supply.pbody.numpy()[1, 0])


def _combine(cw: np.ndarray, depth: float) -> tuple[float, np.ndarray, np.ndarray]:
    """Mass, centre and inertia about it (body frame) of the dry sphere and the water (ρ·π·R²·depth) at ``cw``."""
    mw = RHO * math.pi * R * R * depth
    c0 = np.array(BALL[:3])
    m = M0 + mw
    c = (M0 * c0 + mw * cw) / m
    pt = lambda mass, d: mass * (d @ d * np.eye(3) - np.outer(d, d))      # noqa: E731
    return m, c, 0.4 * M0 * BALL[3] ** 2 * np.eye(3) + pt(M0, c0 - c) + pt(mw, cw - c)


def _expected(depth: float) -> tuple[float, np.ndarray, np.ndarray]:
    """The upright glass: its water's centroid at half its depth."""
    return _combine(np.array([0.0, 0.0, 0.5 * depth]), depth)


def test_a_carried_pool_is_its_bodys_mass_centre_and_inertia_per_world():
    """0.17 kg of water in world 0, 0.085 kg in world 1, on a 0.113 kg glass: a plug-in after the flow reads in apply
    the dry sphere's mass, centre and inertia plus the water at its centroid (1e-6; the inverse to 1e-6), carried_kg
    reads ρ·V of the tank (1e-12); between exchanges the model is the dry body's, bit for bit."""
    look = _Look()
    p, sim, b0, b1 = _sim("mass", look)
    dry = {f: getattr(sim.model, f).numpy().copy() for f in FIELDS}
    sim.step_period()
    for w in range(2):
        want_m, want_c, want_i = _expected(DEPTH[w])
        assert look.m.numpy()[w][0] == pytest.approx(want_m, rel=1e-6)
        assert look.m.numpy()[w][1] == pytest.approx(1.0 / want_m, rel=1e-6)
        assert np.allclose(look.c.numpy()[w], want_c, rtol=0.0, atol=1e-6)
        assert np.allclose(look.i.numpy()[w], want_i, rtol=1e-5, atol=1e-9)
        assert np.allclose(look.ii.numpy()[w] @ look.i.numpy()[w], np.eye(3), rtol=0.0, atol=1e-6)
        r = p.supply.read()
        assert r["carried_kg"][w, 0] == pytest.approx(RHO * r["amount"][w, p.supply.net.node_index("glass")], rel=1e-12)
    for f in FIELDS:
        assert np.array_equal(getattr(sim.model, f).numpy(), dry[f]), f


def test_a_tipped_glass_carries_its_water_at_the_tilted_centroid():
    """The glass tipped 30° about x: its water's centroid lies at y = -tanθ·R²/(4·h0), z = h0/2 + tan²θ·R²/(8·h0)
    (body frame; the surface crosses the wall, not the floor or the rim), so the combined centre is (2e-5 m: the slices'
    chords) and the inertia's yz term is the water's."""
    th = math.radians(30.0)
    look = _Look()
    _, sim, _, _ = _sim("mass", look, tilt=th)
    sim.step_period()
    t = math.tan(th)
    for w, h0 in enumerate(DEPTH):
        cw = np.array([0.0, -t * R * R / (4.0 * h0), 0.5 * h0 + t * t * R * R / (8.0 * h0)])
        _, want_c, want_i = _combine(cw, h0)
        assert np.allclose(look.c.numpy()[w], want_c, rtol=0.0, atol=2e-5)
        assert look.i.numpy()[w][1, 2] == pytest.approx(want_i[1, 2], rel=2e-2)


@pytest.mark.parametrize("carry", ["mass", "weight"])
def test_a_lifted_glass_accelerates_as_its_whole_mass(carry):
    """A 3 N lift on the glass falling free: carried as mass it accelerates at F/(m0 + m_w) - g in each world (the
    solver's mass: 1e-5); with the default weight force at (F - m_w·g)/m0 - g, the dry mass carrying the water."""
    p, sim, b0, b1 = _sim(carry, _Push(force=(0.0, 0.0, 3.0)))
    n = 10
    for _ in range(n):
        sim.step_period()
    t = n * sim.clock.control_dt
    for w, b in ((0, b0), (1, b1)):
        mw = RHO * math.pi * R * R * DEPTH[w]
        a = 3.0 / (M0 + mw) - G if carry == "mass" else (3.0 - mw * G) / M0 - G
        assert sim.state.body_qd.numpy()[b][2] / t == pytest.approx(a, rel=1e-5)


def test_on_alloys_mujoco_c_route_each_world_carries_its_own_water(monkeypatch):
    """ALLOY_CPU_ROUTE=mjc (the CPU default: one MjModel per world): the lift gives F/(m0 + m_w) - g in each world
    (1e-5; WP15's probe: the template model took the masses, so the water vanished)."""
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "mjc")
    p, sim, b0, b1 = _sim("mass", _Push(force=(0.0, 0.0, 3.0)))
    assert any(isinstance(getattr(s, "worlds", None), list) for s in sim.stepper.solvers())
    n = 10
    for _ in range(n):
        sim.step_period()
    t = n * sim.clock.control_dt
    for w, b in ((0, b0), (1, b1)):
        a = 3.0 / (M0 + RHO * math.pi * R * R * DEPTH[w]) - G
        assert sim.state.body_qd.numpy()[b][2] / t == pytest.approx(a, rel=1e-5)


def _tub_and_lid(b) -> None:
    """WP0's probe (contact_drop.py): a free tub of two box plates whose dry centre is its body origin (mass 0.1 kg,
    massless plates: MuJoCo compiles body_sameframe 1) and a free lid of two boxes touching its wall's top."""
    b.add_shape_box(-1, xform=body_xform(2.0, 0.0, 0.01), hx=0.05, hy=0.05, hz=0.01, label="far")
    none = newton.ModelBuilder.ShapeConfig(density=0.0)
    tub = b.add_body(xform=body_xform(0.0, 0.0, 0.0), mass=0.1, com=(0.0, 0.0, 0.0),
                     inertia=((1e-4, 0.0, 0.0), (0.0, 1e-4, 0.0), (0.0, 0.0, 1e-4)), label="tub")
    b.add_shape_box(tub, xform=body_xform(0.0, 0.0, 0.005), hx=0.06, hy=0.06, hz=0.005, cfg=none)
    b.add_shape_box(tub, xform=body_xform(0.055, 0.0, 0.1), hx=0.005, hy=0.06, hz=0.1, cfg=none)
    cfg = newton.ModelBuilder.ShapeConfig(density=500.0)
    lid = b.add_body(xform=body_xform(0.07, 0.0, 0.19), label="lid")
    b.add_shape_box(lid, hx=0.02, hy=0.02, hz=0.02, cfg=cfg)
    b.add_shape_box(lid, xform=body_xform(0.0, 0.1, 0.0), hx=0.02, hy=0.02, hz=0.02, cfg=cfg)


def _tub_lid_contacts(m) -> tuple[int, np.ndarray]:
    """MuJoCo C at the build pose of world model m: the tub-lid contacts and the tub's xipos."""
    import mujoco  # noqa: PLC0415

    tub, lid = (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("tub", "lid"))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    n = sum(1 for c in d.contact[:d.ncon] if {m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]} == {tub, lid})
    return n, d.xipos[tub].copy()


def test_on_the_mujoco_c_route_a_wet_vessel_takes_its_centre_and_keeps_its_contacts(monkeypatch):
    """ALLOY_CPU_ROUTE=mjc, WP0's case: 0.3 kg of water 18 cm under the 0.1 kg tub's origin (its dry centre; its
    world models marked body_sameframe 1, as MuJoCo compiles a body whose inertial frame is its own, and then ignores
    ipos) moves the centre 13.6 cm down. After the push each world's MjModel puts the tub's inertial frame at the wet
    centre (xipos, 5e-5 m) and its midphase boxes in it: the lid keeps every contact it had at the build pose
    (without the refit they drop to 0; without clearing sameframe xipos stays at the origin)."""
    import mujoco  # noqa: PLC0415

    monkeypatch.setenv("ALLOY_CPU_ROUTE", "mjc")
    net = {"nodes": [{"name": "tub", "medium": "tank", "volume": 3e-4, "area": math.pi * 0.05 ** 2}], "edges": [],
           "pools": [Pool("tub", body="tub", floor=(0.0, 0.0, -0.2), cavity=((0.05, 0.0), (0.05, 0.1)), carry="mass")]}
    p = StubFilm(charts=(FAR,), network=net)
    sim = build_simulation(TestScene(build=_tub_and_lid, ground=False, worlds=2, gravity=(0.0, 0.0, 0.0)), [p], "cpu")
    p.supply.gravity = np.array([0.0, 0.0, -G])                              # the water lies on its floor
    s = next(x for x in sim.stepper.solvers() if isinstance(getattr(x, "worlds", None), list))
    for m in s.worlds:
        m.body_sameframe[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "tub")] = 1
    n0 = [_tub_lid_contacts(m)[0] for m in s.worlds]
    sim.step_period()
    kg = p.supply.read()["carried_kg"][0, 0]
    zc = kg * (-0.2 + 0.5 * kg / RHO / (math.pi * 0.05 ** 2)) / (0.1 + kg)
    for m, n in zip(s.worlds, n0):
        got, xipos = _tub_lid_contacts(m)
        assert n > 0 and got == n and xipos == pytest.approx([0.0, 0.0, zc], abs=5e-5), (n, got, xipos, zc)


def test_a_couple_turns_the_glass_by_its_waters_inertia():
    """A 2 mN·m couple about x on the glass falling free (gravity turns nothing about the centre): it turns at
    α = τ / I_xx about the combined centre, the water's parallel-axis term included (2e-3 over one 20 ms period:
    6 mrad, the water's centroid barely moves; without the term α is 19 % larger in world 0)."""
    p, sim, b0, b1 = _sim("mass", _Push(torque=(2e-3, 0.0, 0.0)))
    sim.step_period()
    for w, b in ((0, b0), (1, b1)):
        _, _, want_i = _expected(DEPTH[w])
        assert sim.state.body_qd.numpy()[b][3] / sim.clock.control_dt == pytest.approx(2e-3 / want_i[0, 0], rel=2e-3)


def test_a_spinning_glass_emptied_keeps_its_velocity():
    """The glass turning at 5 rad/s from a still dry centre: over the first period its wet centre falls free, its
    velocity ω × R·(c_wet - c_dry) at the start plus g·T (1e-4 m/s). WP15's probe C: then world 1's water goes (its
    centre moves 12 mm back): its dry centre's velocity after the next period is the one before plus g·T (1e-4 m/s;
    without moving the velocities with the centre it was kicked by ω × Δc, 0.06 m/s)."""
    p, sim, b0, b1 = _sim("mass", spin=5.0)
    q0 = sim.state.body_q.numpy()[b0].astype(float)
    sim.step_period()
    T, dc = sim.clock.control_dt, _expected(DEPTH[0])[1] - np.array(BALL[:3])
    rot = lambda q, v: np.array(wp.quat_rotate(wp.quat(*q[3:7]), wp.vec3(*v)), float)     # noqa: E731
    qd0 = sim.state.body_qd.numpy()[b0].astype(float)
    start = np.cross([5.0, 0.0, 0.0], rot(q0, dc))
    now = qd0[:3] + np.cross(qd0[3:], rot(sim.state.body_q.numpy()[b0].astype(float), dc))
    assert np.allclose(now, start + np.array([0.0, 0.0, -G]) * T, rtol=0.0, atol=1e-4)
    before = sim.state.body_qd.numpy()[b1].astype(float)
    _set_depths(p, (DEPTH[0], 0.0))
    sim.step_period()
    after = sim.state.body_qd.numpy()[b1].astype(float)
    assert np.allclose(after[:3], before[:3] + np.array([0.0, 0.0, -G]) * sim.clock.control_dt, rtol=0.0, atol=1e-4)


def test_a_captured_period_replays_the_carried_mass_bit_for_bit():
    """The masses set in apply and pushed into MuJoCo Warp replay in a captured graph exactly as eager (CPU graphs)."""
    runs = []
    for graph in (False, True):
        _, sim, b0, b1 = _sim("mass", _Push(force=(0.0, 0.0, 3.0), torque=(2e-3, 0.0, 0.0)), graph=graph, spin=1.0)
        for _ in range(3):
            sim.step_period()
        runs.append((sim.state.body_q.numpy()[[b0, b1]], sim.state.body_qd.numpy()[[b0, b1]]))
    assert np.array_equal(runs[0][0], runs[1][0]) and np.array_equal(runs[0][1], runs[1][1])


def _bare(cls, **attrs):
    s = object.__new__(cls)
    for k, v in attrs.items():
        setattr(s, k, v)
    return s


def test_only_solvers_that_take_run_time_masses_are_allowed():
    """XPBD and semi-implicit need no push, Featherstone gets its public notify, MuJoCo Warp the per-world update;
    Newton's one-model MuJoCo C path with W > 1, a SolverMuJoCo without the hook, an unknown solver and a coupled
    stepper are refused."""
    assert inertia_pushes([_bare(ns.SolverXPBD), _bare(ns.SolverSemiImplicit)], 2, ["flow"]) == []
    assert len(inertia_pushes([_bare(ns.SolverFeatherstone)], 2, ["flow"])) == 1
    assert len(inertia_pushes([_bare(ns.SolverMuJoCo, use_mujoco_cpu=False)], 2, ["flow"])) == 1
    assert len(inertia_pushes([_bare(ns.SolverMuJoCo, use_mujoco_cpu=True)], 1, ["flow"])) == 1
    with pytest.raises(PluginError, match="one model for every world"):
        inertia_pushes([_bare(ns.SolverMuJoCo, use_mujoco_cpu=True)], 2, ["flow"])
    with pytest.raises(PluginError, match="no _update_model_inertial_properties"):
        inertia_pushes([_bare(ns.SolverMuJoCo, use_mujoco_cpu=False, _update_model_inertial_properties=None)], 2,
                       ["flow"])
    with pytest.raises(PluginError, match="not known to take"):
        inertia_pushes([_bare(ns.SolverVBD)], 2, ["flow"])
    with pytest.raises(PluginError, match="coupled stepper"):
        inertia_pushes([_bare(ns.SolverXPBD, _entries={})], 2, ["flow"])


def test_carry_mass_needs_a_cavity_on_a_body():
    with pytest.raises(PluginError, match="carry='mass' needs a cavity on a body"):
        Pool("sink", carry="mass")
    with pytest.raises(PluginError, match="carry is 'weight' or 'mass'"):
        Pool("glass", body="glass", cavity=((R, 0.0), (R, HT)), carry="inertia")
