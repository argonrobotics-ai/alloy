# SPDX-License-Identifier: Apache-2.0
"""Film grip law (SCR-WP13-GRIP-1, alloy/solvers/film/grip.py; rows: docs/notes/film-grip-c16.md). CPU.

Rig: an HDPE 15 um strip (4 x 12 cm, 1 cm cells, vertex mass x25) hangs at z = 0.5 m. Two static pads overlap
the film's 0.5 mm contact radius by ``pen`` on either side of its top three rows, so each pinched vertex feels
ke·pen from each pad. Its friction capacity is 2·mu·ke·pen; the whole pinch's is 15 vertices x 2 pads x mu·ke·pen.
Law: the pinch holds a load within mu·N and slips beyond it. Before the fix, the strain projection walked a loaded
strip out of any pinch, and Newton's zero-slip friction gap let it creep (module doc of grip.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.solvers.film.config import FilmConfig
from alloy.solvers.film.emit import add_film_mesh
from alloy.solvers.film.materials import HDPE_15UM
from alloy.solvers.film.rigs import grid_mesh
from alloy.solvers.film.solver import SolverFilm

Z0, NX, NY, DX = 0.5, 4, 12, 0.01
KE, MU = 1.0e4, 0.3
OFF = {"projection_friction": False, "static_friction_hessian": False}


@dataclass(frozen=True)
class Pinch:
    pen: float = 1.0e-4                 # pad overlap of the contact radius (m): ke·pen per pad per pinched vertex
    mass_x: float = 25.0
    dt: float = 5.0e-4
    pads: bool = True
    far: float = 0.0                    # > 0: pads moved this far away in y (contacts exist, none touch)

    @property
    def capacity(self) -> float:
        return 3 * (NX + 1) * 2 * MU * KE * self.pen


class Rig:
    def __init__(self, p: Pinch, **film) -> None:
        b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        v, t = grid_mesh(NX, NY, DX, plane="xz")
        p0, _ = add_film_mesh(b, v, t, HDPE_15UM, pos=(0.0, 0.0, Z0))
        n = b.particle_count - p0
        for i in range(p0, p0 + n):
            b.particle_mass[i] *= p.mass_x
        r = HDPE_15UM.extras.contact_radius
        cfg = newton.ModelBuilder.ShapeConfig(ke=KE, kd=0.0, mu=MU, margin=0.0, gap=0.002)
        for side in (+1, -1) if p.pads else ():
            y = side * (r - p.pen + 0.005 + p.far)
            b.add_shape_box(-1, xform=wp.transform((0.5 * NX * DX, y, Z0 - 0.01), wp.quat_identity()),
                            hx=0.5 * NX * DX + 0.01, hy=0.005, hz=0.015, cfg=cfg)
        b.color()
        self.model = m = b.finalize(device="cpu")
        m.soft_contact_ke, m.soft_contact_kd, m.soft_contact_mu = KE, 0.0, MU
        q = m.particle_q.numpy()
        self.top = np.flatnonzero(np.abs(q[:, 2] - q[:, 2].max()) < 1e-6)
        self.bottom = np.flatnonzero(np.abs(q[:, 2] - q[:, 2].min()) < 1e-6)
        self.weight = float(m.particle_mass.numpy().sum() * 9.81)
        self.solver = SolverFilm(m, iterations=10, film=FilmConfig(**film), friction_epsilon=1.0e-3)
        self.s0, self.s1 = m.state(), m.state()
        self.pipe = newton.CollisionPipeline(m, soft_contact_gap=0.002) if p.pads else None
        self.contacts = self.pipe.contacts() if self.pipe is not None else None
        self.z_top0 = float(q[self.top, 2].mean())
        self.dt, self.k = p.dt, 0

    def step(self, pull: float = 0.0) -> None:
        self.s0.clear_forces()
        if pull > 0.0:
            f = np.zeros((self.model.particle_count, 3), np.float32)
            f[self.bottom, 2] = -pull / len(self.bottom)
            self.s0.particle_f.assign(f)
        if self.pipe is not None:
            self.pipe.collide(self.s0, self.contacts)
        self.solver.step(self.s0, self.s1, None, self.contacts, self.dt)
        self.s0, self.s1 = self.s1, self.s0
        self.k += 1

    def slide(self) -> float:
        """Top row's drop from its start [m] (positive: down, out of the pinch)."""
        return self.z_top0 - float(self.s0.particle_q.numpy()[self.top, 2].mean())


def _hold(seconds: float, **film) -> list[float]:
    """Slides of the heavy pinch (pen 1e-4: capacity 9 N vs a 0.017 N strip) every 0.25 s."""
    rig, out = Rig(Pinch(), **film), []
    for k in range(int(round(seconds / rig.dt))):
        rig.step()
        if (k + 1) % int(round(0.25 / rig.dt)) == 0:
            out.append(rig.slide())
    return out


def test_newton_internals_grip_uses_exist():
    """Canary (WP4 review): grip.py imports these Newton internals; a rename upstream turns the grip terms off (with
    a warning) instead of breaking film imports, and this test names the cause."""
    from alloy.solvers.film import grip, solver

    assert solver.FilmGrip is grip.FilmGrip
    from alloy._vendor.newton._src.geometry.soft_contacts_mesh import (_make_soft_contact_filter_data,  # noqa: F401
                                                         _soft_contact_force_enabled, _SoftContactFilterData)
    from alloy._vendor.newton._src.solvers.vbd.rigid_vbd_kernels import (_evaluate_rigid_soft_contact_force_norm,  # noqa: F401
                                                           compute_projected_isotropic_friction)


def test_newton_friction_still_drops_its_hessian_at_zero_slip():
    """Canary (WP4 review): the static_friction_hessian slot fills Newton's zero Hessian at exactly zero slip. When
    upstream adds the limit mu·f_n·2/eps_u itself, this fails: remove the slot (it would double the stiffness)."""
    from alloy._vendor.newton._src.solvers.vbd.rigid_vbd_kernels import compute_projected_isotropic_friction as friction

    @wp.kernel
    def _friction_at_zero_slip(out: wp.array(dtype=wp.mat33)):
        f, K = friction(0.3, 1.0, wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0), 1.0e-6)
        out[0] = K

    out = wp.zeros(1, dtype=wp.mat33, device="cpu")
    wp.launch(_friction_at_zero_slip, dim=1, outputs=[out], device="cpu")
    assert np.abs(out.numpy()).max() == 0.0


def test_free_film_steps_bitwise_as_before():
    """Grip terms act only on film vertices in body contact: with no pads, and with the pads 1 mm clear of the film
    (contact records inside the 2 mm gap, no penetration), both settings give the same bits."""
    for p in (Pinch(pads=False), Pinch(far=0.001)):
        runs = []
        for film in ({}, OFF):
            rig = Rig(p, **film)
            for _ in range(60):
                rig.step()
            runs.append(rig.s0.particle_q.numpy().copy())
        assert np.array_equal(runs[0], runs[1]), p


def test_loaded_pinch_holds_quick():
    """Law (hold), the gate's short form: 0.25 s. Fixed: no slide; stock: 3.45 mm already (walking out)."""
    assert abs(_hold(0.25)[-1]) < 5.0e-5
    assert _hold(0.25, **OFF)[-1] > 2.0e-3


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_loaded_pinch_holds():
    """Law (hold): a strip loaded past its strain cap stays in a pinch with 500x friction margin; before the fix it
    walked out (stock row in the note: 10.2 mm at 0.5 s, out by 1 s)."""
    s = _hold(1.0)
    assert max(abs(x) for x in s) < 5.0e-5, s
    stock = _hold(0.5, **OFF)
    assert stock[-1] > 5.0e-3, stock


@pytest.mark.slow
@pytest.mark.timeout(600)
@pytest.mark.parametrize("only", ["projection_friction", "static_friction_hessian"])
def test_each_fix_alone_lets_the_pinch_go(only):
    """Both gaps are real: projection friction alone still creeps (Newton's zero-slip Hessian), the Hessian alone
    still walks out (the projection)."""
    s = _hold(1.0, **{**OFF, only: True})
    assert s[-1] > 1.0e-3, (only, s)


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_pinch_slips_beyond_mu_n():
    """Law (slip): a light pinch (pen 1e-6: capacity 0.09 N) under a pull ramped on the bottom row over 2 s, after a
    0.3 s settle. The top row starts to slide (0.1 mm) when the total load is within 10 % of mu·N (measured 1.004)
    and leaves the 20 mm pinch (slide > 20 mm) before 1.5 x (kinetic = static friction here: the excess accelerates
    it)."""
    p = Pinch(pen=1.0e-6)
    rig = Rig(p)
    settle, ramp, top = 0.3, 2.0, 0.15
    onset, out_at = None, None
    for k in range(int(round((settle + ramp) / rig.dt))):
        pull = top * min(1.0, max(0.0, (k * rig.dt - settle) / ramp))
        rig.step(pull)
        ratio, s = (pull + rig.weight) / p.capacity, rig.slide()
        if onset is None and s > 1.0e-4:
            onset = ratio
        if s > 2.0e-2:
            out_at = ratio
            break
    assert onset is not None and 0.9 <= onset <= 1.1, onset
    assert out_at is not None and out_at <= 1.5, out_at


def _drag_mu(k_pull: float = 3.0, pull_s: float = 0.2, **film) -> float:
    """Implied kinetic mu of a strip (12 x 4 cm, x25) lying on a static box (mu 0.3) and dragged by k_pull·mu·M·g on
    its front row for ``pull_s`` after a 0.1 s settle: mu_k = (F - M·a)/(M·g) from its centre of mass (SOFT-1's
    review probe, [internal note] notes/soft1/film_drag_probe.py)."""
    dt, r = 5.0e-4, HDPE_15UM.extras.contact_radius
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    v, t = grid_mesh(12, 4, 0.01, plane="xy")
    p0, _ = add_film_mesh(b, v, t, HDPE_15UM, pos=(0.0, 0.0, r))
    for i in range(p0, b.particle_count):
        b.particle_mass[i] *= 25.0
    cfg = newton.ModelBuilder.ShapeConfig(ke=KE, kd=0.0, mu=MU, margin=0.0, gap=0.002)
    b.add_shape_box(-1, xform=wp.transform((0.3, 0.0, -0.01), wp.quat_identity()), hx=0.6, hy=0.3, hz=0.01, cfg=cfg)
    b.color()
    m = b.finalize(device="cpu")
    m.soft_contact_ke, m.soft_contact_kd, m.soft_contact_mu = KE, 0.0, MU
    solver = SolverFilm(m, iterations=10, film=FilmConfig(**film), friction_epsilon=1.0e-3)
    s0, s1 = m.state(), m.state()
    pipe = newton.CollisionPipeline(m, soft_contact_gap=0.002)
    con = pipe.contacts()
    q, mass = m.particle_q.numpy(), m.particle_mass.numpy()
    M = float(mass.sum())
    f = np.zeros((m.particle_count, 3), np.float32)
    front = np.flatnonzero(np.abs(q[:, 0] - q[:, 0].max()) < 1e-6)
    f[front, 0] = k_pull * MU * M * 9.81 / len(front)
    xs = []
    for k in range(int(round((0.1 + pull_s) / dt))):
        s0.clear_forces()
        if k * dt >= 0.1:
            s0.particle_f.assign(f)
        if abs(k * dt - 0.1) < 0.5 * dt:
            xs.append(float((s0.particle_q.numpy()[:, 0] * mass).sum() / M))
        pipe.collide(s0, con)
        solver.step(s0, s1, None, con, dt)
        s0, s1 = s1, s0
    a = 2.0 * (float((s0.particle_q.numpy()[:, 0] * mass).sum() / M) - xs[0]) / pull_s ** 2
    return (k_pull * MU * 9.81 - a) / 9.81


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_dragged_film_feels_coulomb_friction():
    """Law (kinetic): a film dragged across a table at 3·mu·M·g feels mu_k within 10 % of the card (0.3). The clamp
    first gave the projection a second full mu·f_n on top of VBD's saturated friction: mu_k 0.40 (SOFT-1 review); a
    sliding vertex now gives the projection none."""
    assert _drag_mu() == pytest.approx(MU, rel=0.10)


# --- coupled: WP4's two-pad gripper (MuJoCo servos, lagged Scheduler) on a SolverFilm strip, as C16's fingers ---

def _coupled_model(fix_top: bool, film_top: float, mass_x: float = 25.0) -> newton.Model:
    """alloy.coupling.pinch_rigs.gripper_model with its cloth replaced by a single HDPE 15 um film strip (4 x 8 cm,
    1 cm cells, x``mass_x``) whose top 20 mm the closed pads grip; only its x = 0 column lies within the pads."""
    import math

    from alloy._vendor.newton.solvers import SolverMuJoCo

    from alloy.coupling.pinch_rigs import GRIP_KP, LIFT_KE, PAD_HALF, PAD_MASS, PAD_Z, SHAPE_KE, mesh_box
    from alloy.solvers.film.config import register_custom_attributes

    b = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(b)
    register_custom_attributes(b)
    b.default_shape_cfg.ke, b.default_shape_cfg.kd, b.default_shape_cfg.mu = SHAPE_KE, 1.0e3, 0.4
    lift = b.add_link(xform=wp.transform((0.0, 0.0, PAD_Z + 0.06), wp.quat_identity()), label="carriage")
    cfg = b.default_shape_cfg.copy()
    cfg.density = 1.0 / 8.0e-6
    b.add_shape_box(lift, hx=0.01, hy=0.01, hz=0.01, cfg=cfg)
    joints = [b.add_joint_prismatic(-1, lift, axis=(0.0, 0.0, 1.0), label="lift", target_ke=LIFT_KE,
                                    parent_xform=wp.transform((0.0, 0.0, PAD_Z + 0.06), wp.quat_identity()),
                                    target_kd=2.0 * math.sqrt(LIFT_KE), actuator_mode=newton.JointTargetMode.POSITION)]
    pcfg = b.default_shape_cfg.copy()
    pcfg.density = PAD_MASS / (8 * PAD_HALF[0] * PAD_HALF[1] * PAD_HALF[2])
    r = HDPE_15UM.extras.contact_radius
    for sign, label in ((-1.0, "pad_a"), (1.0, "pad_b")):
        y = sign * (r + PAD_HALF[1])
        pad = b.add_link(xform=wp.transform((0.0, y, PAD_Z), wp.quat_identity()), label=label)
        b.add_shape_mesh(pad, mesh=mesh_box(PAD_HALF), cfg=pcfg)
        joints.append(b.add_joint_prismatic(lift, pad, axis=(0.0, 1.0, 0.0), label=label + "_slide",
                                            parent_xform=wp.transform((0.0, y, -0.06), wp.quat_identity()),
                                            target_ke=GRIP_KP, target_kd=2.0 * math.sqrt(GRIP_KP * PAD_MASS),
                                            actuator_mode=newton.JointTargetMode.POSITION))
    b.add_articulation(joints)
    v, t = grid_mesh(4, 8, 0.01, plane="xz")
    p0, _ = add_film_mesh(b, v, t, HDPE_15UM, pos=(-0.02, 0.0, film_top))
    for i in range(p0, b.particle_count):
        b.particle_mass[i] *= mass_x
        if fix_top and abs(b.particle_q[i][2] - film_top) < 1e-6:
            b.particle_mass[i], b.particle_flags[i] = 0.0, 0
    top = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(top)
    register_custom_attributes(top)
    top.add_world(b)
    top.color()
    m = top.finalize(device="cpu")
    m.soft_contact_ke, m.soft_contact_kd, m.soft_contact_mu = 1.0e4, 1.0e-2, 0.3
    return m


def _coupled_scheduler(m: newton.Model, **film):
    from alloy._vendor.newton.solvers import SolverMuJoCo

    from alloy.core.newton_compat import SolverCoupledProxy
    from alloy.coupling.proxy_contacts import ProxyContactPipeline
    from alloy.coupling.scheduler import Scheduler
    from alloy.coupling.types import AlloyEntry, PairPolicy, SchedulerPolicy

    e_m = SolverCoupledProxy.Entry(name="mjc", solver=lambda v: SolverMuJoCo(model=v, use_mujoco_contacts=False,
                                                                             njmax=50),
                                   bodies=list(range(m.body_count)), joints=list(range(m.joint_count)))
    e_v = SolverCoupledProxy.Entry(name="vbd", particles=list(range(m.particle_count)),
                                   solver=lambda v: SolverFilm(v, iterations=10, film=FilmConfig(**film),
                                                               friction_epsilon=1.0e-3,
                                                               rigid_body_particle_contact_buffer_size=2048))
    proxy = SolverCoupledProxy.Proxy(source="mjc", destination="vbd", bodies=list(range(m.body_count)),
                                     mode="lagged", proxy_relaxation_mode="fixed", collide_interval=1,
                                     collision_pipeline=ProxyContactPipeline(full_surface=False,
                                                                             soft_contact_gap=0.002))
    return Scheduler(m, [AlloyEntry(e_m, adaptive=False), AlloyEntry(e_v, adaptive=False)],
                     [PairPolicy("mjc", "vbd", mode="lagged", proxy=proxy)], SchedulerPolicy())


def _coupled_slide(lift: float = 0.03, hold: float = 1.0, **film) -> float:
    """Pads close at 15 N (top row fixed for the 0.4 s squeeze, then a free model from that state, as the film pinch
    law), settle 0.5 s, lift ``lift`` at 0.05 m/s, hold. Returns how far the pinched column's top vertex fell relative
    to the pads from the lift's start to the hold's end [m]."""
    from alloy.coupling.pinch_rigs import PAD_HALF, PAD_Z, PinchPhase, _run_script

    top_z, face = PAD_Z + PAD_HALF[2], HDPE_15UM.extras.contact_radius
    m1 = _coupled_model(True, top_z)
    s1 = _run_script(_coupled_scheduler(m1, **film), m1, (PinchPhase("squeeze", 0.4, 0.0, True),), lambda *a: None,
                     15.0, half_gap=face)
    m2 = _coupled_model(False, top_z)
    for name in ("particle_q", "particle_qd", "joint_q"):
        getattr(m2, name).assign(getattr(s1, name).numpy())
    m2.joint_qd.zero_()
    q = m2.particle_q.numpy()
    col = int(np.argmax(np.where(np.abs(q[:, 0]) < 1e-6, q[:, 2], -np.inf)))          # x = 0 column, top vertex
    rec: dict = {}

    def observe(ph, k, s):
        rec[ph.name] = float(s.particle_q.numpy()[col, 2]) - float(s.body_q.numpy()[1, 2])

    t_lift = 0.5 + lift / 0.05
    script = (PinchPhase("settle", 0.5, 0.0, True), PinchPhase("lift", t_lift, lift, True, ramp=lift / 0.05),
              PinchPhase("hold", t_lift + hold, lift, True))
    _run_script(_coupled_scheduler(m2, **film), m2, script, observe, 15.0, half_gap=face)
    return rec["settle"] - rec["hold"]


@pytest.mark.slow
@pytest.mark.timeout(1200)
def test_coupled_pinch_holds():
    """Law (coupled): the pinched column keeps its place in the MuJoCo pads through a 30 mm lift and a 1 s hold
    (measured 0.00 mm over 50 mm + 2 s); stock walks out of the pads (the strip falls)."""
    assert abs(_coupled_slide()) < 5.0e-4
    assert _coupled_slide(**OFF) > 2.0e-2
