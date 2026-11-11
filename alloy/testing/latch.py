# SPDX-License-Identifier: Apache-2.0
"""Generic hinge-tab latch rig for ``alloy.latch`` tests (and a template for bench twins, e.g. DC6's force rig).

RJ45-like scale, ESTIMATE geometry (the rig checks the mechanism, not a product): a 20 x 11.68 x 6.6 mm carrier on
a prescribed-motion stage along +x, a 14 mm polycarbonate tab hinged on its top 2 mm behind the nose and rising
15 deg toward the rear, a 6 mm wide shoulder on the tab (vertical rear catch face, 30 deg nose-side ramp), and a
static keeper: two lips either side of a 3.6 mm slot (the narrow lever passes, the shoulder hooks behind them) and
a back wall that stops the nose. Insertion presses the tab (ramp under the lips), the shoulder clears the lips'
deep face and the tab springs up (latched); pulling loads the catch face; pressing the tab >= its release angle
lets the carrier out.

The ramp face stands ``ramp_proud`` above the shoulder's top edge. Flush (0), the lips' entry edge arrives at the
shoulder's nose-side face still a contact depth inside the ramp, box-box contact turns its normal to that face and
the carrier stops there until the stage forces it past: 5.24 N at dt/4 and dt/8 against 3.3 N at dt and dt/2, which
step over it.

Contacts on touched faces use the D22 precision setting at the design step (timeconst 2·design_dt, damping ratio
1, flat solimp 0.99: ``alloy.contact.shadow.set_flat_solimp``), held fixed when the step is halved (CON-62).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.latch.builder import LatchHandles, add_latch
from alloy.latch.design import cantilever_hinge
from alloy.latch.spec import ROOT, TOUCHED, Box, Catch, LatchSpec, Member, Spring
from alloy.latch.state import catch_state, compose

PC_FLEX_MODULUS = 2.35e9      # Pa, Makrolon 2407 flexural modulus (alloy-assets sources/eth/dc1-sources.md M1)


def _rot_y(a: float) -> tuple[float, float, float, float]:
    return (0.0, math.sin(0.5 * a), 0.0, math.cos(0.5 * a))


@dataclass(frozen=True)
class TabRig:
    """Rig numbers (m, rad, kg, N)."""
    carrier_half: tuple[float, float, float] = (0.010, 0.00584, 0.0033)
    carrier_mass: float = 0.005
    hinge_x: float = 0.008
    lever: tuple[float, float, float] = (0.014, 0.0030, 0.0010)      # length, width, thickness
    angle: float = math.radians(15.0)
    shoulder_s: float = 0.007                                       # along the lever from the hinge
    shoulder: tuple[float, float, float] = (0.0016, 0.0060, 0.0012)  # length, width, height above the lever
    ramp_angle: float = math.radians(30.0)
    ramp_proud: float = 1.0e-4                                      # ramp face above the shoulder's top edge
    tab_mass: float = 6.0e-5
    beam_length: float = 0.010                                      # cantilever length for k_theta
    preload_frac: float = 0.1                                       # of rate x travel
    travel: float = math.radians(15.0)
    overlap: float = 0.0010                                         # catch overlap above the lips' lower face
    slot: float = 0.0036
    depth: float = 0.016                                            # back wall (nose stop) from the opening
    play: float = 0.0005
    mu: float = 0.38                                                # PC on steel (sources M2)
    design_dt: float = 6.25e-4
    imp: float = 0.99
    start_gap: float = 0.003                                        # nose outside the opening at q_stage = 0
    stage_ke: float = 1500.0
    stage_kd: float = 5.0

    @property
    def rate(self) -> float:
        return cantilever_hinge(PC_FLEX_MODULUS, self.lever[1], self.lever[2], self.beam_length)

    def frames(self) -> dict[str, np.ndarray]:
        """Key points in the carrier frame at rest: shoulder rear-face x, shoulder top z."""
        a, (L, w, t), (sl, sw, sh) = self.angle, self.lever, self.shoulder
        u, n = np.array([-math.cos(a), 0.0, math.sin(a)]), np.array([math.sin(a), 0.0, math.cos(a)])
        hinge = np.array([self.hinge_x, 0.0, self.carrier_half[2]])
        base = hinge + self.shoulder_s * u + t * n
        return {"hinge": hinge, "u": u, "n": n, "base": base,
                "rear_x": np.array(base[0] - 0.5 * sl), "top_z": np.array(base[2] + sh)}

    @property
    def lip_x(self) -> float:
        """Deep face of the lips (world x); the opening is at x = 0."""
        latched_origin = self.depth - self.carrier_half[0]
        return float(latched_origin + self.frames()["rear_x"] - self.play)

    @property
    def lip_z(self) -> float:
        return float(self.frames()["top_z"] - self.overlap)


def tab_spec(r: TabRig) -> LatchSpec:
    f = r.frames()
    a, (L, w, t), (sl, sw, sh) = r.angle, r.lever, r.shoulder
    u, n = f["u"], f["n"]
    s_local = r.shoulder_s * u + t * n                          # shoulder base, member frame (origin at hinge)
    embed = 0.5 * t
    lever = Box("rig_tab_lever", (0.5 * L, 0.5 * w, 0.5 * t), tuple(0.5 * L * u + 0.5 * t * n), _rot_y(a), "body")
    catch = Box("rig_tab_catch", (0.5 * sl, 0.5 * sw, 0.5 * (sh + embed)),
                (s_local[0], 0.0, s_local[2] + 0.5 * (sh - embed)), role="catch")
    b = r.ramp_angle
    rl, rt = sh / math.sin(b), 0.0008
    corner = np.array([s_local[0] + 0.5 * sl, 0.0, s_local[2] + sh])
    dirv, ntop = np.array([math.cos(b), 0.0, -math.sin(b)]), np.array([math.sin(b), 0.0, math.cos(b)])
    ramp_c = corner + 0.5 * rl * dirv + (r.ramp_proud - 0.5 * rt) * ntop
    ramp = Box("rig_tab_ramp", (0.5 * rl, 0.5 * sw, 0.5 * rt), tuple(ramp_c), _rot_y(b), "ramp")
    k = r.rate
    tab = Member("tab", "hinge", tuple(f["hinge"]), (0.0, -1.0, 0.0), r.travel,
                 Spring(k, r.preload_frac * k * r.travel, 1e-5), (lever, catch, ramp), r.tab_mass)
    body = Box("rig_carrier_body", r.carrier_half, role="body")
    nose = Box("rig_carrier_nose", (0.0005, r.carrier_half[1] * 0.9, r.carrier_half[2] * 0.9),
               (r.carrier_half[0] - 0.0005, 0.0, 0.0), role="pad")
    return LatchSpec("rig_carrier", (body, nose), r.carrier_mass, (tab,),
                     catches=(Catch("tab_catch", "tab", "rig_tab_catch", (0.0, 0.0, 1.0)),), design_dt=r.design_dt)


def keeper_spec(r: TabRig) -> LatchSpec:
    y0, y1 = 0.5 * r.slot, 0.5 * r.shoulder[1] + 0.0015
    lip_h = 0.0015
    lips = tuple(Box(f"rig_keeper_lip_{s}", (0.5 * r.lip_x, 0.5 * (y1 - y0), 0.5 * lip_h),
                     (0.5 * r.lip_x, sg * 0.5 * (y0 + y1), r.lip_z + 0.5 * lip_h), role="catch")
                 for s, sg in (("l", 1.0), ("r", -1.0)))
    back = Box("rig_keeper_back", (0.001, 0.008, 0.006), (r.depth + 0.001, 0.0, 0.0), role="pad")
    return LatchSpec("rig_keeper", (*lips, back), 0.1, static=True, design_dt=r.design_dt)


def touched_cfg(r: TabRig):
    """ShapeConfig factory: D22-style stiff contact on touched roles, the same friction everywhere."""
    tau = 2.0 * r.design_dt

    def cfg_for(box: Box) -> newton.ModelBuilder.ShapeConfig:
        stiff = box.role in TOUCHED
        return newton.ModelBuilder.ShapeConfig(density=0.0, mu=r.mu, ke=1.0 / tau**2 if stiff else 2.5e3,
                                               kd=2.0 / tau if stiff else 100.0, margin=0.0, gap=0.002)
    return cfg_for


@dataclass(frozen=True)
class CatchProbe:
    """Which boxes the geometric latch state compares (catch on a carrier part, keeper on a keeper part)."""
    carrier_part: str
    catch: Box
    keeper_part: str
    keeper: Box
    insert: tuple[float, float, float] = (1.0, 0.0, 0.0)      # keeper frame
    lift: tuple[float, float, float] = (0.0, 0.0, 1.0)


@dataclass
class Rig:
    """A built rig: model, the carrier on its stage, the keeper, stage drive gains, insertion travel, the probe."""
    model: newton.Model
    carrier: LatchHandles
    keeper: LatchHandles
    stage_ke: float
    stage_kd: float
    travel_in: float              # stage target at full insertion (m from the start)
    probe: CatchProbe
    member: str = "tab"           # the carrier member a finger presses


def build_rig(r: TabRig, worlds: int = 1, device: str = "cpu") -> Rig:

    w = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(w)
    cfg = touched_cfg(r)
    x0 = -(r.carrier_half[0] + r.start_gap)
    ts, ks = tab_spec(r), keeper_spec(r)
    car = add_latch(w, ts, wp.transform(wp.vec3(x0, 0.0, 0.0), wp.quat_identity()), mount="slide",
                    cfg_for=cfg, stage=(r.stage_ke, r.stage_kd))
    kee = add_latch(w, ks, cfg_for=cfg)
    stiffen(w, ((car, ts), (kee, ks)), r.imp)
    probe = CatchProbe("tab", ts.box("rig_tab_catch")[1], ROOT, ks.box("rig_keeper_lip_l")[1])
    return Rig(replicate(w, worlds, device), car, kee, r.stage_ke, r.stage_kd, r.start_gap + r.depth, probe)


def stiffen(b: newton.ModelBuilder, parts, imp: float) -> None:
    """Flat solimp ``imp`` on every touched-role shape of the (handles, spec) pairs (D22 precision setting)."""
    from alloy.contact.shadow import set_flat_solimp

    for h, spec in parts:
        for part in (ROOT, *[m.name for m in spec.members]):
            for box in spec.part_shapes(part):
                if box.role in TOUCHED:
                    set_flat_solimp(b, h.shapes[box.name], imp)


def replicate(w: newton.ModelBuilder, worlds: int, device: str) -> newton.Model:
    top = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(top)
    top.replicate(w, worlds)
    return top.finalize(device=device)


RIG_SOLVER = dict(use_mujoco_contacts=True, integrator="implicitfast", cone="elliptic", impratio=300.0,
                  iterations=100, ls_iterations=50, nconmax=64, njmax=256)   # MuJoCo's Newton solver (default)


class RigRun:
    """Steps a rig with SolverMuJoCo (``RIG_SOLVER`` + ``solver_kw``); the stage follows a target (m along its
    axis from the start)."""

    def __init__(self, rig: Rig, dt: float, solver_kw: dict | None = None) -> None:
        kw = {**RIG_SOLVER, **(solver_kw or {})}
        self.rig, self.dt, m = rig, dt, rig.model
        self.solver = newton.solvers.SolverMuJoCo(m, **kw)
        self.s0, self.s1, self.ctrl = m.state(), m.state(), m.control()
        newton.eval_fk(m, m.joint_q, m.joint_qd, self.s0)
        self.contacts = None
        self.nq = m.joint_coord_count // m.world_count
        self.nv = m.joint_dof_count // m.world_count
        self.nb = m.body_count // m.world_count
        self.t = 0.0

    def _idx(self, base: int, stride: int) -> list[int]:
        return [base + k * stride for k in range(self.rig.model.world_count)]

    def set_inputs(self, target: float | np.ndarray, torque: float | np.ndarray = 0.0) -> None:
        h = self.rig.carrier
        tq = self.ctrl.joint_target_q.numpy()
        tq[self._idx(h.stage_coord, self.nq)] = target
        self.ctrl.joint_target_q.assign(tq)
        f = np.zeros(self.rig.model.joint_dof_count, np.float32)
        f[self._idx(h.dofs[self.rig.member], self.nv)] = torque
        self.ctrl.joint_f.assign(f)

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            self.solver.step(self.s0, self.s1, self.ctrl, self.contacts, self.dt)
            self.s0, self.s1 = self.s1, self.s0
            self.t += self.dt

    def overflow(self) -> int:
        """MuJoCo Warp's per-world overflow bitmask, OR-ed (0 = none; HealthMonitor does not read it, DC6)."""
        ov = getattr(getattr(self.solver, "mjw_data", None), "overflow", None)
        return int(np.bitwise_or.reduce(ov.numpy().ravel())) if ov is not None else 0

    def read(self) -> dict[str, np.ndarray]:
        """Stage x [W], stage force [W] (drive spring term, as a load cell reads it), member angle [W], latch
        engaged [W], finite, overflow. ``v`` is the stage speed."""
        rig, h = self.rig, self.rig.carrier
        q, qd = self.s0.joint_q.numpy(), self.s0.joint_qd.numpy()
        tq = self.ctrl.joint_target_q.numpy()
        idx = self._idx(h.stage_coord, self.nq)
        x, v = q[idx], qd[self._idx(h.stage_dof, self.nv)]
        force = rig.stage_ke * (tq[idx] - x)          # load-cell reading: the drive spring's deflection
        tab = q[self._idx(h.coords[rig.member], self.nq)]
        return {"x": x, "force": force, "v": v, "tab": tab, "engaged": self.engaged(), "finite": np.isfinite(q).all(),
                "overflow": self.overflow()}

    def engaged(self) -> np.ndarray:
        """Latch state from geometry (``alloy.latch.state``) for the rig's catch probe, per world."""
        p, rig = self.rig.probe, self.rig
        bq = self.s0.body_q.numpy().reshape(rig.model.world_count, -1, 7)
        cb, kb = rig.carrier.bodies[p.carrier_part] % self.nb, rig.keeper.bodies[p.keeper_part] % self.nb
        cc, cR = compose(bq[:, cb], p.catch.pos, p.catch.quat)
        kc, kR = compose(bq[:, kb], p.keeper.pos, p.keeper.quat)
        return catch_state(cc, cR, p.catch.half, kc, kR, p.keeper.half, p.insert, p.lift).engaged


# ------------------------------------------------------------------------------------------- procedures

@dataclass(frozen=True)
class Trace:
    x: np.ndarray            # [T, W] stage position (m)
    force: np.ndarray        # [T, W] stage force (N, + pushes in)
    tab: np.ndarray          # [T, W] tab angle (rad)
    engaged: np.ndarray      # [T, W] latch state from geometry
    finite: bool
    overflow: int = 0        # MuJoCo Warp overflow bits seen
    sample_dt: float = 0.0   # s between rows

    def filtered(self, window_s: float = 5.0e-3) -> np.ndarray:
        """[T, W] force, moving average over ``window_s`` (a force gauge's bandwidth; excludes one-step spikes)."""
        n = max(1, int(round(window_s / self.sample_dt))) if self.sample_dt > 0 else 1
        if n == 1 or len(self.force) < n:
            return self.force
        k = np.ones(n) / n
        return np.stack([np.convolve(self.force[:, w], k, mode="same") for w in range(self.force.shape[1])], axis=1)

    @property
    def peak(self) -> np.ndarray:
        """[W] peak of the 5 ms filtered force."""
        return self.filtered().max(axis=0)

    def peak_before(self, x_limit: float) -> np.ndarray:
        """[W] peak filtered force while the stage is short of ``x_limit`` (the insertion peak, not the push
        against the stop)."""
        f = np.where(self.x < x_limit, self.filtered(), -np.inf)
        return f.max(axis=0)


def _record(run: RigRun, n: int, target_of, torque: float, every: int) -> Trace:
    rows = []
    for i in range(n):
        run.set_inputs(target_of(run), torque)
        run.step(1)
        if i % every == 0 or i == n - 1:
            rows.append(run.read())
    stack = {k: np.stack([r[k] for r in rows]) for k in ("x", "force", "tab", "engaged")}
    return Trace(**stack, finite=bool(all(r["finite"] for r in rows)),
                 overflow=int(np.bitwise_or.reduce([r["overflow"] for r in rows])), sample_dt=every * run.dt)


def insert(run: RigRun, speed: float = 0.010, overtravel: float = 0.005, every: int = 1,
           relax_s: float = 0.05) -> Trace:
    """Drive the stage in at ``speed`` to ``overtravel`` past the nose stop (a hand pushing home: stage_ke x
    overtravel at the stop), hold 0.05 s, then relax the drive onto the reached position for ``relax_s``."""
    x_end = run.rig.travel_in + overtravel
    t0 = run.t
    n = int((x_end / speed + 0.05) / run.dt)
    push = _record(run, n, lambda rr: min(speed * (rr.t - t0), x_end), 0.0, every)
    x_home = run.read()["x"]
    rest = _record(run, max(1, int(relax_s / run.dt)), lambda rr: x_home, 0.0, every)
    return Trace(*(np.concatenate([getattr(push, k), getattr(rest, k)]) for k in ("x", "force", "tab", "engaged")),
                 finite=push.finite and rest.finite, overflow=push.overflow | rest.overflow, sample_dt=push.sample_dt)


def pull(run: RigRun, force: float, ramp_s: float = 0.5, hold_s: float = 0.2, torque: float = 0.0,
         every: int = 4) -> Trace:
    """Ramp the stage force outward to ``force`` N over ``ramp_s`` (drive target behind the carrier), then hold."""
    x0, t0 = run.read()["x"], run.t
    n = int((ramp_s + hold_s) / run.dt)

    def target(rr: RigRun) -> np.ndarray:
        f = force * min(1.0, (rr.t - t0) / ramp_s)
        return x0 - f / rr.rig.stage_ke
    return _record(run, n, target, torque, every)


def press(run: RigRun, torque: float, settle_s: float = 0.1, every: int = 4) -> Trace:
    """Hold the stage where it is and press the tab with ``torque`` N·m (a finger on the lever)."""
    x0 = run.read()["x"]
    return _record(run, int(settle_s / run.dt), lambda rr: x0, torque, every)


# ------------------------------------------------------------------------------------------- preloaded pad rig

@dataclass(frozen=True)
class PadRig:
    """A pair of preloaded slide pads on a carrier opened by a static blade (DC3's C13 contact, ESTIMATE geometry from
    DC3's ``C13Dims``): each pad rests on its closed stop at ``gap/2`` from the blade axis, a 12 deg lead-in at its
    front cams it open by ``(blade_t - gap)/2`` as the carrier moves +x onto the blade."""
    blade_t: float = 2.0e-3
    gap: float = 1.6e-3
    pad_t: float = 1.5e-3
    pad_h: float = 4.0e-3
    pad_len: float = 6.0e-3
    lead_len: float = 2.8e-3
    lead_open: float = 0.6e-3
    rate: float = 1000.0
    preload: float = 6.84
    damping: float = 5.0
    travel: float = 0.8e-3
    pad_mass: float = 2.0e-3
    carrier_mass: float = 0.022
    mu: float = 0.5
    engage: float = 5.0e-3          # blade length past the pads' lead at the seat
    start_gap: float = 1.0e-3
    stage_ke: float = 5000.0        # DC3's crosshead
    stage_kd: float = 50.0
    design_dt: float = 1.25e-3
    imp: float = 0.99
    regularise: str = "armature"
    stop_stiffness: float | None = 1.0e5
    snap_step: float | None = 1.07e-4

    @property
    def opening(self) -> float:
        return 0.5 * (self.blade_t - self.gap)

    def peak_estimate(self) -> float:
        """Wedge law on both pads at the lead (CON-61): 2 x N x (tan a + mu) / (1 - mu tan a)."""
        a = math.atan2(self.lead_open, self.lead_len)
        n = self.preload + self.rate * self.opening
        return 2.0 * n * (math.tan(a) + self.mu) / (1.0 - self.mu * math.tan(a))


def _pad_member(p: PadRig, side: float) -> Member:
    """Pad on +y (side 1) or -y (-1): member frame on the face plane, +y = away from the blade (opening)."""
    a = math.atan2(p.lead_open, p.lead_len)
    face = Box(f"pad_{'l' if side > 0 else 'r'}_face", (0.5 * p.pad_len, 0.5 * p.pad_t, 0.5 * p.pad_h),
               (-0.5 * p.pad_len, 0.5 * p.pad_t, 0.0), role="pad")
    L = math.hypot(p.lead_len, p.lead_open)
    n = np.array([-math.sin(a), math.cos(a)])                      # outward normal of the sloped face (x, y)
    c = np.array([0.5 * p.lead_len, 0.5 * p.lead_open]) + 0.5 * p.pad_t * n
    lead = Box(f"pad_{'l' if side > 0 else 'r'}_lead", (0.5 * L, 0.5 * p.pad_t, 0.5 * p.pad_h), (c[0], c[1], 0.0),
               (0.0, 0.0, math.sin(0.5 * a), math.cos(0.5 * a)), role="ramp")
    if side < 0:                                                  # mirror y: positions and the slope's sign
        face = Box(face.name, face.half, (face.pos[0], -face.pos[1], 0.0), role="pad")
        lead = Box(lead.name, lead.half, (c[0], -c[1], 0.0), (0.0, 0.0, -math.sin(0.5 * a), math.cos(0.5 * a)),
                   role="ramp")
    return Member(f"pad_{'l' if side > 0 else 'r'}", "slide", (0.0, side * 0.5 * p.gap, 0.0), (0.0, side, 0.0),
                  p.travel, Spring(p.rate, p.preload, p.damping), (face, lead), p.pad_mass,
                  regularise=p.regularise, stop_stiffness=p.stop_stiffness, snap_step=p.snap_step)


def pad_carrier_spec(p: PadRig) -> LatchSpec:
    """The carrier (the C13 plug): a body behind the pads and the two pads; front of the leads at x = lead_len."""
    body = Box("pad_carrier_body", (0.004, 0.006, 0.004), (-p.pad_len - 0.006, 0.0, 0.0))
    return LatchSpec("pad_carrier", (body,), p.carrier_mass, (_pad_member(p, 1.0), _pad_member(p, -1.0)),
                     design_dt=p.design_dt)


def pad_blade_spec(p: PadRig) -> LatchSpec:
    """The static blade (C14 pin) from its tip at x = 0 forward (no seat stop: the stage drives past the full
    engagement, and seating means reaching it)."""
    blade = Box("pad_blade", (0.010, 0.5 * p.blade_t, 0.003), (0.010, 0.0, 0.0), role="guide")
    return LatchSpec("pad_blade", (blade,), 0.1, static=True, design_dt=p.design_dt)


def build_pad_rig(p: PadRig, worlds: int = 1, device: str = "cpu") -> Rig:
    """Carrier on a stage along +x, its lead front ``start_gap`` before the blade tip; seated after ``travel_in``."""
    w = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(w)
    tau = 2.0 * p.design_dt

    def cfg_for(box: Box) -> newton.ModelBuilder.ShapeConfig:
        stiff = box.role in TOUCHED
        return newton.ModelBuilder.ShapeConfig(density=0.0, mu=p.mu, ke=1.0 / tau**2 if stiff else 2.5e3,
                                               kd=2.0 / tau if stiff else 100.0, margin=0.0, gap=0.002)
    cs, bs = pad_carrier_spec(p), pad_blade_spec(p)
    x0 = -(p.lead_len + p.start_gap)
    car = add_latch(w, cs, wp.transform(wp.vec3(x0, 0.0, 0.0), wp.quat_identity()), mount="slide", cfg_for=cfg_for,
                    stage=(p.stage_ke, p.stage_kd))
    bla = add_latch(w, bs, cfg_for=cfg_for)
    stiffen(w, ((car, cs), (bla, bs)), p.imp)
    probe = CatchProbe("pad_l", cs.box("pad_l_face")[1], ROOT, bs.box("pad_blade")[1], (1.0, 0.0, 0.0),
                       (0.0, 1.0, 0.0))
    return Rig(replicate(w, worlds, device), car, bla, p.stage_ke, p.stage_kd,
               p.start_gap + p.lead_len + p.engage, probe, member="pad_l")
