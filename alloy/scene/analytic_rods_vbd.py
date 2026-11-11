# SPDX-License-Identifier: Apache-2.0
"""Analytic rod laws on the ROD_VBD route (SCR-WP9-ROD, producer WP9; WP2 reviews).

The same conditions, readings and bounds as WP2's CHAIN laws. Only the rod changes: it is built by the ROD_VBD
emitter (`alloy.scene.emitters.rod_vbd.build_rod_vbd`) and stepped by `alloy.solvers.rod.SolverVBDRod`, so each law
judges what that route emits.
- The law cards are isotropic (G J = E I / 1.3). Their EA is set to E A from the card's E I (`vbd_card`): the CHAIN
  law card's EA is a placeholder, since CHAIN links do not stretch.
- Newton refuses an uncoloured VBD model, and the bench colours only particle models, so `emit` colours the builder
  after each rod.

**A11.torsion.rod_vbd** is A11.torsion (`analytic_torsion`) on this route: T L / (G J) within 1 % at 0.5 and 1.5 rad.
- Segments keep their physical axial inertia: no armature. That is what R1 fixed. Before it, VBD's inertial target
  took the end torque, and the twist read 2 atan(x / 2) / x of T L / (G J), with x = dt^2 T / I_axis (0.51 / 0.21 on
  a 12-segment probe).
- The tracked off-axis point is placed at world +z of the tip's axis at build, since segments lie along their local
  z here; CHAIN links lie along local x.

**A11.cantilever.rod_vbd** is A11.cantilever on this route: the clamped self-weight cantilever (gamma 0.5 and 5,
30 segments, settled with the self-test's damping). Its tip drop must match the exact static drop of the same
discrete chain within 1 % (`rodref.static_tip_drop`), with the elastica gap as a monitor. Points are tracked by
their world position at build (`local_point`).

**A11.hanging_chain.rod_vbd** and **A11.catenary.rod_vbd** are WP2's chain laws on this route, with pinned ends
(`build_rod_vbd(pin=)`: world ball joints). Their cards bend almost freely (gamma 1e6), so EA is set stiff
(`CHAIN_EA`, a nylon-like 3 GPa over the section) for an inextensible chain. The catenary settles with CHAIN's
2e-4 N m s/rad per bend and twist slot.

**A10.capstan.rod_vbd** is WP2's capstan (`analytic_capstan`) on this route.
- The rope is the same: the router's link count, the same polygon over the drum, and dead end loads at a fixed
  ratio r of exp(mu theta).
- Contacts are Newton's (CollisionPipeline), with VBD's penalty contact at `K_CONTACT` and Coulomb friction mu on the
  rope and the drum. SolverVBDRod adds them to each chain's solve.
- Slip is judged by speed, as on CHAIN.
- It steps at CHAIN's 5e-4 s with 4 iterations. Before the rod solve took tension's geometric stiffness (R3c), the
  sliding worlds at mu 0.4 diverged there, and the law ran at 2.5e-4 s with 8 iterations.
- The holding worlds creep up to 14 mm/s at mu 0.4, ratio 0.99 (WP2's review). That is the viscous zone of VBD's
  friction_epsilon 1e-2, well under the 72 mm/s slip threshold.

**`torsion_case`** is XB4-10's rod-twist hook ([internal note] `xb4/mujoco4/RERUN.md`: `rerun.py twist`, scored at
T = 1e-4, 5e-4, 2e-3 N m, closed at |theta G J / (T L) - 1| <= 1.7 %; MuJoCo 4.0: 0.983) on this route.
- The rod dict is the A11.cantilever gamma5 rod MuJoCo ran.
- Root clamped, gravity off, the torque held about +x on the free end with the law's settle damping.
- theta is the tip segment's rotation relative to the root's about the axis, unwrapped step by step.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.bench.solvers import SolverSettings
from alloy.bench.twin import WorldRanges
from alloy.bench.types import Condition
from alloy.route.types import RodCard
from alloy.scene import analytic_torsion as AT
from alloy.scene import analytic_capstan as AC
from alloy.scene import analytic_rods as AR
from alloy.scene.analytic_rods import H0, SETTLE_TAU, ei_for_gamma, rod_card
from alloy.scene.emitters.chain import ChainHandles, link_frames
from alloy.scene.emitters.rod_vbd import build_rod_vbd
from alloy.solvers.rod import SolverVBDRod

HOLD_MAX_S, STATIC_TAIL_S = 10.0, 0.5   # s: torsion_case's longest hold, and the tail that must be static
STATIC_REL = 1e-6                       # static: theta moves by under this share over the tail

ITERATIONS = 1                  # SolverVBDRod iterations per step (WP2's D8 rows: 1 is enough for one chain)

LAW_TORSION_VBD = dataclasses.replace(AT.LAW_TORSION, id="A11.torsion.rod_vbd", solvers=("vbd",), producer="WP9")
LAW_CANTILEVER_VBD = dataclasses.replace(AR.LAW_CANTILEVER, id="A11.cantilever.rod_vbd", solvers=("vbd",),
                                         producer="WP9")
LAW_HANGING_VBD = dataclasses.replace(AR.LAW_HANGING, id="A11.hanging_chain.rod_vbd", solvers=("vbd",), producer="WP9")
LAW_CATENARY_VBD = dataclasses.replace(AR.LAW_CATENARY, id="A11.catenary.rod_vbd", solvers=("vbd",), producer="WP9")
LAW_CAPSTAN_VBD = dataclasses.replace(AC.LAW, id="A10.capstan.rod_vbd", solvers=("vbd",), producer="WP9")
CAPSTAN_ITERATIONS = 4            # SolverVBDRod iterations with the drum's contact blocks
CHAIN_EA_MODULUS = 3.0e9          # Pa: the chain laws' inextensible axis (their EI is near zero)


def vbd_card(rod: RodCard, chain: bool = False) -> RodCard:
    """The law card with EA = E A, E from its E I (module docstring); `chain`: a stiff axis for the chain laws."""
    EA = CHAIN_EA_MODULUS * math.pi * rod.radius ** 2 if chain else rod.EI * 4.0 / rod.radius ** 2
    return dataclasses.replace(rod, EA=EA)


def local_offset(q_xyzw: Sequence[float], world: Sequence[float]) -> tuple[float, float, float]:
    """A world-frame offset in the body frame of rotation q (x y z w): R(q)^T v."""
    x, y, z, w = (float(c) for c in q_xyzw)
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return tuple(float(v) for v in R.T @ np.asarray(world, np.float64))


def local_point(b: newton.ModelBuilder, body: int, world: Sequence[float]) -> tuple[float, float, float]:
    """A world point at build in `body`'s frame."""
    tf = b.body_q[body]
    off = np.asarray(world, np.float64) - np.asarray(tf.p, np.float64)
    return local_offset(tf.q, off)


class VBDRodTwin:
    """Mixin over a CHAIN law twin (`analytic_rods.RodLawTwin`): rods from the ROD_VBD emitter, SolverVBDRod."""

    settings: Mapping[str, SolverSettings] = {"vbd": SolverSettings("vbd", {"iterations": ITERATIONS})}

    chain_card = False                # the chain laws: stiff EA (module docstring)
    pins: str | None = None           # build_rod_vbd(pin=) for opts["pin"]

    def emit(self, b: newton.ModelBuilder, rod: RodCard, points: Any, **opts: Any) -> ChainHandles:
        h = build_rod_vbd(b, vbd_card(rod, self.chain_card), points, clamp="start" if opts.get("clamp") else None,
                          pin=self.pins if opts.get("pin") else None, label=f"law{len(self.chains)}")
        self.nodes = link_frames(points, rod.length, int(rod.links))[0]
        b.color()
        self.chains.append(h)
        self.rods.append(rod)
        return h

    def make_solver(self, model: newton.Model, solver: str) -> newton.solvers.SolverBase:
        self.model, self.solver_name = model, solver
        self.ranges = WorldRanges.bodies(model) if model.body_count else None
        self._first_body = self.ranges.start if self.ranges is not None else None
        self.on_model(model)
        return SolverVBDRod(model, **self.settings_for(solver).merged_kwargs())


class TorsionVBDTwin(VBDRodTwin, AT.TorsionTwin):
    law_id = LAW_TORSION_VBD.id

    def build(self, b: newton.ModelBuilder, cond: Condition, sample: Mapping[str, float]) -> None:
        rod = rod_card(AT.L, ei_for_gamma(AT.L, AT.GAMMA), AT.LINKS, damping_ratio=0.5 * SETTLE_TAU)
        h = self.emit(b, rod, [(0.0, 0.0, H0), (AT.L, 0.0, H0)], clamp=True)
        tip = h.bodies[-1]
        off = local_offset(b.body_q[tip].q, (0.0, 0.0, AT.OFFSET))
        self.track([(tip, off), (tip, (0.0, 0.0, 0.0))])
        torque = float(cond.params["theta"]) * rod.GJ / rod.length
        self._torques.append(torque)
        self._expected.append(torque * rod.length / rod.GJ)
        self._length_ratio.append(float(sum(h.lengths)) / rod.length)


def card_from_dict(rod: Mapping[str, float], damping_ratio: float = 0.5 * SETTLE_TAU) -> RodCard:
    """RERUN.md's rod dict (length_m, EI_Nm2, GJ_Nm2, links, mass_per_length_kg_m, radius_m) as a ROD_VBD card."""
    r, lin, EI, L = (float(rod[k]) for k in ("radius_m", "mass_per_length_kg_m", "EI_Nm2", "length_m"))
    return RodCard(length=L, radius=r, density=lin / (math.pi * r * r), EI=EI, GJ=float(rod["GJ_Nm2"]),
                   EA=EI * 4.0 / (r * r), links=int(rod["links"]), damping_ratio=damping_ratio,
                   gamma=lin * 9.81 * L ** 3 / EI)


def axial_twist(q_root: np.ndarray, q_tip: np.ndarray) -> float:
    """Rotation of the tip segment relative to the root's about the segments' local z (the rod axis), in (-pi, pi]."""
    x0, y0, z0, w0 = (float(c) for c in q_root[3:7])
    x1, y1, z1, w1 = (float(c) for c in q_tip[3:7])
    rz = w0 * z1 - x0 * y1 + y0 * x1 - z0 * w1                 # z and w of conj(q_root) * q_tip
    rw = w0 * w1 + x0 * x1 + y0 * y1 + z0 * z1
    return 2.0 * math.atan2(rz, rw) if rw >= 0.0 else 2.0 * math.atan2(-rz, -rw)


def torsion_case(rod: Mapping[str, float], torque_nm: float, device: str = "cpu") -> dict[str, Any]:
    """XB4-10's hook (module docstring): {"theta_rad", "ok", "route", ...}. Holds the torque in 0.5 s chunks until
    theta moves by under 1e-6 of itself over one (at most 10 s); ok when finite and static."""
    card = card_from_dict(rod)
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    h = build_rod_vbd(b, card, [(0.0, 0.0, H0), (card.length, 0.0, H0)], clamp="start", label="twist")
    b.color()
    model = b.finalize(device=device)
    solver = SolverVBDRod(model, iterations=ITERATIONS)
    s0, s1, ctl = model.state(), model.state(), model.control()
    f = np.zeros((model.body_count, 6), np.float32)
    f[h.bodies[-1], 3] = float(torque_nm)
    dt = LAW_TORSION_VBD.dt
    tail = int(round(STATIC_TAIL_S / dt))
    theta, prev, t, spread = 0.0, 0.0, 0.0, math.inf
    while t < HOLD_MAX_S and not spread <= STATIC_REL * abs(theta):
        start = theta
        for _ in range(tail):
            s0.clear_forces()
            s0.body_f.assign(f)
            solver.step(s0, s1, ctl, None, dt)
            s0, s1 = s1, s0
            q = s0.body_q.numpy()
            cur = axial_twist(q[h.bodies[0]], q[h.bodies[-1]])
            theta += math.remainder(cur - prev, 2.0 * math.pi)
            prev = cur
        t += tail * dt
        spread = abs(theta - start)
    finite = bool(np.isfinite(s0.body_q.numpy()).all())
    return {"theta_rad": theta, "ok": finite and spread <= STATIC_REL * abs(theta), "route": "ROD_VBD (SolverVBDRod)",
            "ratio": theta * card.GJ / (float(torque_nm) * card.length), "held_s": t, "last_0p5s_change_rad": spread,
            "segments": card.links}


class CantileverVBDTwin(VBDRodTwin, AR.CantileverTwin):
    law_id = LAW_CANTILEVER_VBD.id

    def build(self, b: newton.ModelBuilder, cond: Condition, sample: Mapping[str, float]) -> None:
        rod = rod_card(self.L, ei_for_gamma(self.L, cond.params["gamma"]), int(cond.params["links"]),
                       damping_ratio=0.5 * SETTLE_TAU)
        h = self.emit(b, rod, [(0.0, 0.0, H0), (self.L, 0.0, H0)], clamp=True)
        self.track([(h.bodies[-1], local_point(b, h.bodies[-1], (self.L, 0.0, H0)))])


class HangingChainVBDTwin(VBDRodTwin, AR.HangingChainTwin):
    law_id = LAW_HANGING_VBD.id
    chain_card, pins = True, "start"

    def build(self, b: newton.ModelBuilder, cond: Condition, sample: Mapping[str, float]) -> None:
        L = cond.params["length"]
        h = self.emit(b, rod_card(L, ei_for_gamma(L, 1e6), self.LINKS), AR.mode_points(L, 0.01 * L), pin=True)
        self.track([(h.bodies[-1], local_point(b, h.bodies[-1], self.nodes[-1]))])


class CatenaryVBDTwin(VBDRodTwin, AR.CatenaryTwin):
    law_id = LAW_CATENARY_VBD.id
    chain_card, pins = True, "both"

    def build(self, b: newton.ModelBuilder, cond: Condition, sample: Mapping[str, float]) -> None:
        S, L = cond.params["span"], self.L
        d = math.sqrt(max(0.25 * L * L - 0.25 * S * S, 0.0))
        h = self.emit(b, rod_card(L, ei_for_gamma(L, 1e6), int(cond.params["links"])),
                      [(-0.5 * S, 0.0, H0), (0.0, 0.0, H0 - d), (0.5 * S, 0.0, H0)], pin=True)
        for j in h.joints:                                  # bend and twist slots: CHAIN's settle damping
            q = b.joint_qd_start[j]
            for k in range(3, 6):
                b.joint_target_kd[q + k] = self.SETTLE_DAMPING
        mid = len(h.bodies) // 2 - 1                        # node L/2 = the end of segment N/2 - 1 (N even)
        self.track([(h.bodies[mid], local_point(b, h.bodies[mid], self.nodes[mid + 1])),
                    (h.bodies[-1], local_point(b, h.bodies[-1], self.nodes[-1]))])


class CapstanVBDTwin(VBDRodTwin, AC.CapstanTwin):
    law_id = LAW_CAPSTAN_VBD.id
    dt = LAW_CAPSTAN_VBD.dt
    needs_contacts = True
    settings: Mapping[str, SolverSettings] = {"vbd": SolverSettings("vbd", {"iterations": CAPSTAN_ITERATIONS})}

    def settings_for(self, solver: str) -> SolverSettings:
        return self.settings[solver]

    def build(self, b: newton.ModelBuilder, cond: Condition, sample: Mapping[str, float]) -> None:
        w, mu = AC.wrap(), cond.params["mu"]
        rod = rod_card(w.length, AC.EI, w.links, radius=AC.ROPE_R, density=AC.DENSITY)
        cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=mu, ke=AC.K_CONTACT, kd=0.0, mu_torsional=0.0,
                                              mu_rolling=0.0, gap=0.002)
        axis_y = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * math.pi)
        b.add_shape_capsule(-1, radius=AC.DRUM_R, half_height=0.05, xform=wp.transform(wp.vec3(), axis_y), cfg=cfg)
        pts = AC.wrap_points()
        h = build_rod_vbd(b, vbd_card(rod, chain=True), pts, cfg=cfg, label=f"law{len(self.chains)}")
        b.color()
        self.chains.append(h)
        self.rods.append(rod)
        nodes = link_frames(pts, rod.length, int(rod.links))[0]
        self._loads += [AC.T_IN, AC.T_IN * sample["ratio"] * math.exp(mu * math.pi)]
        self.track([(h.bodies[0], local_point(b, h.bodies[0], nodes[0])),
                    (h.bodies[-1], local_point(b, h.bodies[-1], nodes[-1]))])


__all__ = ["CantileverVBDTwin", "CapstanVBDTwin", "CatenaryVBDTwin", "HangingChainVBDTwin", "ITERATIONS",
           "LAW_CANTILEVER_VBD", "LAW_CAPSTAN_VBD", "LAW_CATENARY_VBD", "LAW_HANGING_VBD", "LAW_TORSION_VBD",
           "TorsionVBDTwin", "VBDRodTwin", "axial_twist", "card_from_dict", "local_offset", "local_point",
           "torsion_case", "vbd_card"]
