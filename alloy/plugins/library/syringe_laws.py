# SPDX-License-Identifier: Apache-2.0
"""Law scenes and checks of the ``syringe`` plug-in (SCR-WP8-11 Part A: G-1..G-6 with bodies, the null case).

Every scene builds a barrel fixed to the world with its head along ``head`` and a 2 g plunger on prismatic joint
``syringe_plunger`` whose armature is the needle's liquid column (``inertance``). A ``push`` input presses the
plunger toward the head; gravity -z sets "up". Closed forms read the plunger's mass and the plug-in's parameters back.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.fluid import P_ATM, Edge
from alloy.plugins.harness.scene import Law, LawResult, TestScene
from alloy.plugins.npgeom import quat_rotate
from alloy.plugins.library.syringe import Syringe, Vial, inertance, needle

G = 9.81
BORE = 8.59e-3                           # m, BD 3 mL
N25, N18 = (0.26e-3, 0.016), (0.84e-3, 0.038)   # needle (bore, length) [m]: 25 G x 16 mm, 18 G x 38 mm blunt fill
PLUNGER = 2.0e-3                         # kg
X_FULL = 0.0523                          # m: 3.1 mL in the barrel (2.8 mL water + 0.3 mL air)


def rig(head: tuple[float, float, float], x0: float, armature: float, stroke: float = 0.06, drive_kd: float = 0.0):
    """build(b) for a barrel with its head along ``head`` (world) and the plunger at travel x0 (limits [0, stroke]);
    ``drive_kd`` > 0: a velocity drive on the plunger joint (a finger at a set speed, MuJoCo's implicit damper)."""
    h = np.asarray(head, np.float64) / np.linalg.norm(head)
    axis = -h                                            # the joint axis: from the head toward the thumb rest
    z = np.array([0.0, 0.0, 1.0])
    rot = wp.quat_between_vectors(wp.vec3(*z), wp.vec3(*axis)) if abs(axis @ z + 1.0) > 1e-9 else \
        wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi)

    def build(b: newton.ModelBuilder) -> None:
        base = wp.transform(wp.vec3(0.0, 0.0, 0.3), rot)
        barrel = b.add_link(xform=base, label="syringe_barrel")
        b.add_shape_box(barrel, xform=wp.transform(wp.vec3(0.0, 0.0, 0.5 * stroke), wp.quat_identity()),
                        hx=0.006, hy=0.006, hz=0.5 * stroke, cfg=newton.ModelBuilder.ShapeConfig(density=1000.0))
        tip = wp.transform_multiply(base, wp.transform(wp.vec3(0.0, 0.0, x0), wp.quat_identity()))
        plunger = b.add_link(xform=tip, label="syringe_plunger_body")
        hz = 0.004
        dens = PLUNGER / (8.0 * 0.004 * 0.004 * hz)
        b.add_shape_box(plunger, hx=0.004, hy=0.004, hz=hz, cfg=newton.ModelBuilder.ShapeConfig(density=dens))
        j0 = b.add_joint_fixed(-1, barrel, parent_xform=base, label="syringe_mount")
        j1 = b.add_joint_prismatic(barrel, plunger, parent_xform=wp.transform(wp.vec3(0.0, 0.0, x0),
                                                                              wp.quat_identity()),
                                   axis=(0.0, 0.0, 1.0), limit_lower=-x0, limit_upper=stroke - x0, armature=armature,
                                   target_ke=0.0, target_kd=drive_kd, label="syringe_plunger")
        b.add_articulation([j0, j1], label="syringe")
    return build


DRIVE_KD = 1.0e4                          # N·s/m: the finger's velocity drive (stiff against the needle's ~480)
SPEED = 5.0e-3                           # m/s into the barrel: 0.29 mL/s


def scene(head, x0: float, armature: float, periods: int, gravity=(0.0, 0.0, -G), speed: float = 0.0,
          dt: float = 2.0e-3) -> TestScene:
    """``speed`` > 0: the plunger driven into the barrel at about that speed (its drive's target velocity)."""
    def init(sim: Any) -> None:
        if speed:
            j = sim.model.joint_qd_start.numpy()[[k for k, lab in enumerate(sim.model.joint_label)
                                                  if lab.endswith("syringe_plunger")]]
            qd = sim.control.joint_target_qd.numpy()
            qd[j] = -speed
            sim.control.joint_target_qd.assign(qd)
    return TestScene(build=rig(head, x0, armature, drive_kd=DRIVE_KD if speed else 0.0), periods=periods,
                     ground=False, gravity=gravity, init=init, dt=dt)


def purge_scene(plug: Any, head=(0.0, 0.0, 1.0), periods: int = 50) -> TestScene:
    """Head ``head``, 2.8 mL of water and 0.3 mL of air, a 25 G needle to the air, the finger pushing at 2 mm/s."""
    return scene(head, X_FULL, inertance(BORE, *N25), periods, speed=SPEED)


def _syr(**kw) -> Syringe:
    return Syringe("syringe", "syringe_plunger", bore=BORE, stroke=0.06, **kw)


PURGE = {"syringes": (_syr(x0=X_FULL, liquid=2.8e-6),), "vials": (),
         "ports": (Edge("nozzle", "port", "syringe", reservoir=P_ATM, **needle(*N25)),), "openings": ()}


def capped(n: float) -> dict:
    return {**PURGE, "syringes": (_syr(x0=X_FULL, liquid=2.8e-6, polytropic=n),),
            "openings": (("nozzle", 0.0),)}


VIAL_P0, VIAL_HEAD = 1.5 * P_ATM, 5.0e-6
X_EMPTY = 0.002                          # m: the plunger near the head (0.19 mL with the dead space)
DRAW = {"syringes": (_syr(x0=X_EMPTY),),
        "vials": (Vial("vial", volume=10e-6, liquid=5e-6, radius=10.75e-3, up=-1.0, pressure=VIAL_P0),),
        "ports": (Edge("needle", "port", "syringe", "vial", depth_b=0.005, **needle(*N18)),), "openings": ()}


def _trace(run: Any, name: str) -> np.ndarray:
    return run.trace.plugin(run.plugin.label, name).astype(np.float64)


def _plunger_weight_along_axis(run: Any) -> float:
    """m·g along the joint axis (+: pulling the plunger out), from the model."""
    mass = float(run.model.body_mass.numpy()[int(run.bodies("syringe_plunger_body")[0])])
    rot = run.model.body_q.numpy()[int(run.bodies("syringe_barrel")[0])][3:7].astype(np.float64)
    axis = quat_rotate(rot, np.array([0.0, 0.0, 1.0]))              # the barrel's local z: the joint axis
    return mass * float(np.asarray(run.model.gravity.numpy(), np.float64).reshape(-1, 3)[0] @ axis)


def _boyle(run: Any) -> LawResult:
    """G-1: capped, the finger pushing at 2 mm/s: every period P = P_atm·(V_g0/V_g)^n at the measured travel (water's
    compression and the dead space included), 0.5 %; the plunger force is (P - P_atm)·A, 1 %."""
    s = run.plugin.config.syringes[0]
    p, x = _trace(run, "pressure")[1:, 0, 0], _trace(run, "travel")[1:, 0, 0]
    f = _trace(run, "force")[1:, 0, 0]
    vg0 = s.volume(s.x0) - s.liquid
    want = np.array([_capped_p(s.dead + s.area * xi, s.liquid, vg0, s.polytropic, s.bulk_modulus) for xi in x])
    err_p = float(np.max(np.abs(p / want - 1.0)))
    big = np.abs(want - P_ATM) > 1e3
    err_f = float(np.max(np.abs(f[big] / ((want[big] - P_ATM) * s.area) - 1.0))) if big.any() else math.inf
    err = max(err_p / 5e-3, err_f / 1e-2)
    return LawResult(err <= 1.0, err, 1.0, f"P up to {p.max():.6g} Pa, worst {err_p:.2e} off P·V^n (n = "
                     f"{s.polytropic}); force worst {err_f:.2e} off (P - P_atm)·A", "the pocket's Boyle pressure")


def _capped_p(v: float, liq: float, vg0: float, n: float, k: float) -> float:
    """Host bisection: L·(1 - (P - P_atm)/K) + V_g0·(P_atm/P)^(1/n) = V."""
    lo, hi = 1.0e3, 1.0e9
    for _ in range(200):
        p = 0.5 * (lo + hi)
        lo, hi = (p, hi) if liq * (1 - (p - P_ATM) / k) + vg0 * (P_ATM / p) ** (1 / n) > v else (lo, p)
    return 0.5 * (lo + hi)


def _purge(run: Any) -> LawResult:
    """G-2: head up, the finger pushing: the air ledger when liquid first leaves is the start pocket (at P_atm, 1 %);
    then P - P_atm = A·v/G at the plunger's own speed v (Poiseuille through the 25 G needle, 2 %)."""
    s = run.plugin.config.syringes[0]
    air = run.plugin.air_out_ml("nozzle")[0]
    pocket = (s.volume(s.x0) - s.liquid) * 1e6
    liq = -_trace(run, "ledger")[:, 0, 0]
    t, x = np.asarray(run.trace.t, np.float64), _trace(run, "travel")[:, 0, 0]
    p = _trace(run, "pressure")[:, 0, 0]
    k = int(np.argmax(liq > 0))
    late = slice(max(k + 5, len(t) // 2), None)
    speed = -np.polyfit(t[late], x[late], 1)[0]
    want = s.area * speed / needle(*N25)["conductance"]
    dp = float(np.mean(p[late])) - P_ATM
    err = max(abs(air / pocket - 1.0) / 1e-2, abs(dp / want - 1.0) / 2e-2) if k > 0 else math.inf
    return LawResult(err <= 1.0, err, 1.0, f"air out {air:.5g} mL (pocket {pocket:.5g}); liquid from period {k}; "
                     f"P - P_atm {dp:.5g} Pa at {speed * 1e3:.4g} mm/s (Poiseuille {want:.5g})",
                     "the air above the nozzle first, then the needle's laminar law")


def _nozzle_down(run: Any) -> LawResult:
    """G-4: head down: liquid covers the nozzle the whole run, the air ledger stays exactly 0 while liquid leaves."""
    air = run.plugin.air_out_ml("nozzle")[0]
    liq = -_trace(run, "ledger")[-1, 0, 0]
    ok = air == 0.0 and liq > 1e-8
    return LawResult(ok, abs(air), 0.0, f"air out {air} mL, liquid out {liq * 1e6:.4g} mL", "liquid at the port only")


def _draw(run: Any) -> LawResult:
    """G-3: an inverted vial at 1.5 atm with a 5 mL headspace fills a free syringe until the vial reaches the
    syringe's rest pressure P_atm - m·g_axis/A: drawn = V_h·(P0/P_rest - 1) (0.5 %)."""
    s = run.plugin.config.syringes[0]
    p_rest = P_ATM - _plunger_weight_along_axis(run) / s.area
    drawn = _trace(run, "liquid")[-1, 0, 0]
    want = VIAL_HEAD * (VIAL_P0 / p_rest - 1.0)
    p_vial = _trace(run, "pressure")[-1, 0, 1]
    err = max(abs(drawn / want - 1.0), abs(p_vial / p_rest - 1.0))
    return LawResult(err <= 5e-3, err, 5e-3, f"drawn {drawn * 1e6:.5g} mL (want {want * 1e6:.5g}); vial "
                     f"{p_vial:.6g} Pa (want {p_rest:.6g})", "Boyle in the vial's headspace")


def _books(run: Any) -> LawResult:
    """G-5: liquid and gas amounts (syringe + vial) constant every period in every world, 1e-12 relative."""
    worst = 0.0
    for name in ("liquid", "gas"):
        tot = _trace(run, name).sum(axis=2)
        worst = max(worst, float(np.max(np.abs(tot / tot[0] - 1.0))))
    return LawResult(worst <= 1e-12, worst, 1e-12, f"worst drift {worst:.2e}", "exact books per port")


def _energy(run: Any) -> LawResult:
    """G-6 (n = 1): work on the contents -∫(P - P_atm)·dV = the pocket's and the water's stored exergy + the
    needle's throttling (gas a·ln(P_up/P_down), liquid ∫Q·ΔP dt), 1 %."""
    s = run.plugin.config.syringes[0]
    p, a = _trace(run, "pressure")[-1, 0, 0], _trace(run, "gas")[-1, 0, 0]
    vg = a / p
    stored = (a * math.log(p / P_ATM) - a + P_ATM * vg if a > 0 else 0.0) + \
        _trace(run, "liquid")[-1, 0, 0] * (p - P_ATM) ** 2 / (2 * s.bulk_modulus)
    stored0 = 0.0                                        # start: everything at P_atm
    work = -_trace(run, "node_work")[-1, 0, 0]
    loss = _trace(run, "work")[-1, 0, 0]
    err = abs(work - (stored - stored0 + loss)) / max(abs(work), 1e-12)
    return LawResult(err <= 1e-2, err, 1e-2, f"plunger work {work:.5g} J, stored {stored:.4g} J, throttled "
                     f"{loss:.5g} J", "isothermal exergy ledger")


def _null(run: Any) -> LawResult:
    f = float(np.max(np.abs(_trace(run, "force")[1:])))          # frame 0 is before the first exchange
    dp = float(np.max(np.abs(_trace(run, "pressure")[1:] - P_ATM)))
    x = float(np.ptp(_trace(run, "travel")[1:]))
    ok = dp <= 1.0 and f <= 1e-4 and x <= 1e-6                     # float32 poses: the travel reads to ~1e-8 m
    return LawResult(ok, dp, 1.0, f"max |force| {f:.2e} N, |P - P_atm| {dp:.2e} Pa, travel range {x:.2e} m",
                     "a closed syringe at P_atm, no push, no gravity: nothing moves")


def syringe_laws(plug: Any) -> list[Law]:
    up, down = (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)
    i25, i18 = inertance(BORE, *N25), inertance(BORE, *N18)
    return [
        Law("G-1_capped_isothermal", _boyle, scene(up, X_FULL, i25, 40, speed=SPEED), "capped, n = 1: P·V = const "
            "at the travel, 0.5 %; force (P - P_atm)·A, 1 %", capped(1.0)),
        Law("G-1_capped_adiabatic", _boyle, scene(up, X_FULL, i25, 40, speed=SPEED), "capped, n = 1.4: P·V^1.4 = "
            "const at the travel, 0.5 %", capped(1.4)),
        Law("G-2_purge_air_first", _purge, purge_scene(plug, up, 100), "head up: the pocket out first (1 %), then "
            "Poiseuille (2 %)", PURGE),
        Law("G-3_vial_equal_volume", _draw, scene(up, X_EMPTY, i18, 100, dt=5e-4), "an inverted vial at 1.5 atm "
            "fills a free syringe: Boyle, 0.5 %", DRAW),
        Law("G-4_nozzle_down_no_air", _nozzle_down, purge_scene(plug, down, 30), "head down: no air leaves", PURGE),
        Law("G-5_books", _books, scene(up, X_EMPTY, i18, 30, dt=5e-4), "amounts constant to 1e-12", DRAW),
        Law("G-6_energy_ledger", _energy, purge_scene(plug, up, 100), "work = stored + throttled, 1 %", PURGE),
        Law("null_closed_at_atmosphere", _null, scene(up, X_FULL, i25, 10, gravity=(0.0, 0.0, 0.0)), "no drive, "
            "capped, no gravity: no force", {**PURGE, "openings": (("nozzle", 0.0),)}),
    ]


__all__ = ["DRAW", "PURGE", "capped", "purge_scene", "rig", "scene", "syringe_laws"]
