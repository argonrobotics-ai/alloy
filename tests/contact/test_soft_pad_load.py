# SPDX-License-Identifier: Apache-2.0
"""A rigid box resting on a soft pad loads it (C3 box_on_pad, XB4-5; ``alloy.contact.particle_gains``).

Through ``alloy.Scene`` a soft solid's VBD records against a MuJoCo-owned box read the box's MuJoCo native gains
(acceleration-space, kd 364 "N·s/m" at ζ 0.7) plus half of Newton's default soft kd. VBD's lagged reaction on the box
was then 1,300x the pad's impulse: XB4-5's 0.515 kg platen hovered at the contact distance on an unloaded pad
(stretch 0.998 for 0.900) or was thrown off. The particle destination now reads the card's force-space K, and the
records against a MuJoCo body share a spring and damper held to the exchange clock's bound for that body (per-body:
ω·dt and c·dt/m <= 0.5). A heavy ball on a coarse cloth (a few records) rests without bouncing or sinking."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import brentq

import alloy
from alloy._vendor import newton
from alloy.bench.analytic.neohookean import uniaxial_stress
from alloy.cards.types import Budget, Primitive
from alloy.contact import particle_gains as PG
from alloy.scene.emitters.deformable import box_mesh, soft_particle_radius

PAD, BOX = (0.04, 0.04, 0.01), (0.06, 0.06, 0.02)
BOX_MASS = 0.5153                       # kg (XB4-5's platen)
PAD_MAT, E, NU, RHO = "tissue_fat_silicone", 3.0e4, 0.45, 1070.0     # the library card's values (Ecoflex 00-30)
DT, SECONDS = 0.0025, 0.04


def _arr(a):
    return SimpleNamespace(numpy=lambda a=np.asarray(a): a)


def test_force_gains_name_the_native_gain_shapes():
    """The shapes ``apply_native_gains`` rewrote: card objects on dynamic bodies of the MuJoCo entry, in every world;
    not static, kinematic, massless, robot, untagged or other-entry shapes. K is the material's own (2x the table's
    series diagonal), Newton's soft ke where the table has none."""
    kin = int(newton.BodyFlags.KINEMATIC)
    # bodies: 0 box w0, 1 kinematic tray, 2 robot link, 3 massless, 4 VBD-owned body, 5 box w1
    model = SimpleNamespace(shape_count=9, body_count=6, soft_contact_ke=1.0e3,
                            shape_body=_arr([-1, 0, 1, 2, 3, 4, 5, 0, 0]),
                            body_mass=_arr([0.5, 1.0, 1.0, 0.0, 1.0, 0.5]),
                            body_flags=_arr([0, kin, 0, 0, 0, 0]),
                            alloy=SimpleNamespace(shape_material=_arr([0, 1, 1, 2, 1, 1, 1, 3, -1])))
    table = SimpleNamespace(materials=("plywood", "beech", "robot:arm", "rubber@puck"),
                            stiffness=_arr(np.diag([1e6, 2e6, 1e6, np.inf]) / 2.0))
    built = SimpleNamespace(model=model, pair_table=table,
                            entries=[SimpleNamespace(capability="mujoco", bodies=(0, 1, 2, 3, 5)),
                                     SimpleNamespace(capability="vbd", bodies=(4,))])
    gains = PG.force_gains(built)
    assert gains.shapes.tolist() == [1, 6, 7] and gains.model_shapes == 9
    assert gains.ke.tolist() == [2e6, 2e6, 1e3]


def _expected_stretch() -> float:
    """Uniaxial Neo-Hookean stretch under the box plus half the pad (C3's P_eff), the pad's Lamé values."""
    mu, lam = E / (2 * (1 + NU)), E * NU / ((1 + NU) * (1 - 2 * NU))
    p = -(BOX_MASS + 0.5 * RHO * np.prod(PAD)) * 9.81 / (PAD[0] * PAD[1])
    return float(brentq(lambda a: uniaxial_stress(a, mu, lam) - p, 0.3, 1.0))


@pytest.fixture(scope="module")
def run():
    """C3 through alloy.Scene, both bodies placed at rest at their contact distance (the pad's particle radius r, the
    route's rule): the box's weight comes on suddenly, so the pad's first compression peaks past the static stretch
    (0.888 near 22 ms on the CPU's Warp route for the static 0.903) and rebounds."""
    r = soft_particle_radius(*box_mesh(Primitive(kind="box", size=tuple(0.5 * x for x in PAD)), Budget().tets))
    scene = alloy.Scene(dt=DT)
    scene.add_ground()
    scene.add_box("pad", size=PAD, pos=(0.0, 0.0, r + PAD[2] / 2), material=PAD_MAT)
    z0 = r + PAD[2] + r + BOX[2] / 2
    scene.add_box("box", size=BOX, pos=(0.0, 0.0, z0), mass=BOX_MASS, material="wood_beech")
    sim = alloy.build(scene, device="cpu")
    assert sim.routes()["pad"]["route"] == "soft_vbd"
    assert float(sim.engine.model.particle_radius.numpy()[0]) == pytest.approx(r, rel=1e-5)
    traj = sim.run(SECONDS, record=["box", "pad"]).world(0)
    return SimpleNamespace(sim=sim, traj=traj, z0=z0)


@pytest.mark.law
@pytest.mark.compile_heavy                 # the module fixture builds and runs the coupled scene (~50 s warm on CPU)
def test_box_loads_the_pad(run):
    """The pad carries the box: within 40 ms it compresses past the static stretch, by less than the undamped step
    response's 2x, and the box never rises above its start. Before: the box hovered or was thrown off, stretch >=
    0.993."""
    pts, box_z = run.traj["pad.points"], run.traj["box.pos"][:, 2]
    rest = pts[0]
    top, bot = rest[:, 2] > rest[:, 2].max() - 1e-5, rest[:, 2] < rest[:, 2].min() + 1e-5
    stretch = (pts[:, top, 2].mean(axis=1) - pts[:, bot, 2].mean(axis=1)) / (rest[top, 2].mean() - rest[bot, 2].mean())
    expected = _expected_stretch()
    assert 0.88 < expected < 0.92
    assert expected - 0.5 * (1.0 - expected) < stretch.min() < expected, (stretch.round(4).tolist(), expected)
    assert box_z.max() < run.z0 + 1e-4, box_z.round(5).tolist()
    assert run.sim.check()["ok"], run.sim.check()["problems"]


@pytest.mark.compile_heavy
def test_vbd_reads_card_gains_and_record_gains(run):
    """VBD's view has the card's K and kd 0 on the box; the Model keeps MuJoCo's native gains; VBD's records on the
    box take their per-body share of the clock bound (bound² m / (dt² n), bound m / (dt n)) and those on the ground
    keep the soft mix (VBD's averages)."""
    eng = run.sim.engine
    vbd = eng.parts.solvers["vbd"]
    box = int(np.flatnonzero(eng.model.shape_body.numpy() >= 0)[0])
    assert vbd.model.shape_material_ke.numpy()[box] == pytest.approx(2.0e6)
    assert vbd.model.shape_material_kd.numpy()[box] == 0.0
    assert eng.model.shape_material_kd.numpy()[box] > 1.0                   # MuJoCo's native kd [1/s]
    assert eng.model.shape_material_ke.numpy()[box] != pytest.approx(2.0e6)
    contacts = eng.stepper.get_proxy_contacts("mujoco", "vbd")
    n = int(contacts.soft_contact_count.numpy()[0])
    shape = contacts.soft_contact_shape.numpy()[:n]
    kd = vbd.body_particle_contact_material_kd.numpy()[:n]
    ke = vbd.body_particle_contact_penalty_k.numpy()[:n]
    on_box = shape == box
    body = int(vbd.model.shape_body.numpy()[box])                          # the box's id in the (compacted) view
    m = 1.0 / float(vbd.body_inv_mass_effective.numpy()[body])
    records = int(vbd.body_particle_contact_counts.numpy()[body])
    dt = float(eng._last_coupling.dt_x)
    bound = 0.5                                                            # ClockPolicy: safety 0.5 x omega_dt_max 1
    assert np.any(on_box) and records >= int(on_box.sum())
    assert np.allclose(kd[on_box], bound * m / (dt * records), rtol=1e-5, atol=0.0)
    soft = vbd.model
    mix_box = 0.5 * (float(soft.soft_contact_ke) + 2.0e6)
    assert np.allclose(ke[on_box], min(mix_box, bound * bound * m / (dt * dt * records)), rtol=1e-5, atol=0.0)
    floor = int(np.flatnonzero(eng.model.shape_body.numpy() < 0)[0])
    ground = shape == floor
    assert np.any(ground) and np.allclose(kd[ground], 0.5 * float(soft.soft_contact_kd), atol=0.0)
    mix = 0.5 * (float(soft.soft_contact_ke) + float(soft.shape_material_ke.numpy()[floor]))
    assert np.allclose(ke[ground], mix, rtol=1e-6, atol=0.0)


@pytest.mark.law
@pytest.mark.compile_heavy
@pytest.mark.timeout(900)    # 40-iteration cloth default (1de04ca01): ~250 s on a loaded VM, over compile_heavy's 180 s
def test_heavy_ball_rests_on_a_coarse_cloth():
    """Few records under a heavy body get stiff, damped shares: a 0.5 kg steel ball dropped 1 mm onto a cloth with 3 cm
    vertex spacing, lying on the ground, rests at its contact distance (0.05 mm in) from 0.05 s on. With a fixed
    per-record 1e4 N/m and 1 N·s/m (the first version of this fix) it still bounces ±0.6 mm at 0.05-0.08 s. ~50 s on
    the CPU (exchange interval 1/6400 s)."""
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_cloth("sheet", size=(0.3, 0.3), pos=(0.0, 0.0, 0.001), resolution=11, mass=0.05, self_contact=False)
    scene.add_sphere("ball", 0.03, pos=(0.0, 0.0, 0.033), mass=0.5, material="steel_carbon")       # 1 mm up
    sim = alloy.build(scene, device="cpu")
    traj = sim.run(0.08, record=["ball", "sheet"]).world(0)
    r = float(sim.engine.model.particle_radius.numpy()[0])
    t, z, pts = traj["t"], traj["ball.pos"][:, 2] - 0.03, traj["sheet.points"]
    under = np.linalg.norm(pts[:, :, :2] - traj["ball.pos"][:, None, :2], axis=2) < 0.03
    top = np.array([pts[i, under[i], 2].max() for i in range(len(t))])
    gap = z - top - r                                                     # the ball's bottom above its contact distance
    late = t >= 0.05
    assert np.all(np.abs(gap[late]) < 2e-4), gap.round(6).tolist()
    assert sim.check()["ok"], sim.check()["problems"]
