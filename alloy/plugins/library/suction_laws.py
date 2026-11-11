# SPDX-License-Identifier: Apache-2.0
"""Law scenes and checks of the ``suction`` plug-in (A-SUC-1..4, the null case, the amount ledger).

Every scene has a rigid ``SuctionCup`` (label "suction") fixed to the world and a free box "box". The cup faces down
(lip plane at z = LIP_Z) unless a scene turns it sideways. Closed forms use the plug-in's own parameters: ΔP·A with
P_ss from ``Suction.steady_pressure`` and A = π·R². Bands are two-sided where a threshold is judged (holds at 0.95,
lets go at 1.05).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from alloy._vendor import newton

from alloy.plugins.harness.scene import Law, LawResult, TestScene, body_xform
from alloy.plugins.library.suction import SuctionCup

G = 9.81
BOX = (0.04, 0.04, 0.02)               # box half extents [m]
MOUNT_Z = 0.30                         # hanging scenes: mount frame height
LIP_DROP = 0.025                       # mount origin to lip plane (mount half height 5 mm + cup height 20 mm)
HOLD_TOL, DROP_MIN = 1.0e-3, 5.0e-3    # m: held = moved less than this; let go = moved more than this
SIDE_Q = (0.0, -math.sqrt(0.5), 0.0, math.sqrt(0.5))   # cup local -z (the lip) along world +x
LIP = "box"
CONTACT_KE = 1.0e5                     # lip and box contact stiffness (mass-normalised in MuJoCo): sinks ~0.2 mm
CONTACT_KD = 2.0 * CONTACT_KE ** 0.5    # critical damping (MuJoCo dampratio ~1; Newton's default kd 100 gives ~0.15 and
                                        # the contacts chatter at ~100 Hz under a steady load: WP13 11-06)
# Coulomb stick for the shear law: impratio 100 and the elliptic cone. With critically damped contacts the box sticks
# below μ·N (0.02 mm in 0.5 s, no noslip needed); the earlier "creep" was the kd-100 contact chatter
FRICTION = {"impratio": 100.0, "cone": "elliptic"}


def f_max(plug: Any, seal: float = 1.0) -> float:
    """(P_atm - P_ss)·π·R²·s with the pump on."""
    return (plug.config.atmosphere - plug.steady_pressure(seal)) * plug.area * seal


def _box(b: newton.ModelBuilder, mass: float, center: tuple[float, float, float], half=BOX, mu=None) -> None:
    body = b.add_body(xform=body_xform(*center), label="box")
    cfg = newton.ModelBuilder.ShapeConfig(density=mass / (8.0 * half[0] * half[1] * half[2]), ke=CONTACT_KE,
                                          kd=CONTACT_KD)
    if mu is not None:
        cfg.mu = mu
    b.add_shape_box(body, hx=half[0], hy=half[1], hz=half[2], cfg=cfg)


def _inputs(label: str, pump: float, vent: float = 0.0):
    def init(sim: Any) -> None:
        if label in sim.plugins.plugins:                 # the speed baseline runs without the plug-in
            sim.plugins[label].set_input("pump", pump)
            sim.plugins[label].set_input("vent", vent)
    return init


def hold_scene(plug: Any, load: float, half: tuple[float, float, float] = BOX, pump: float = 1.0, vent: float = 0.0,
               periods: int = 25, mass: float | None = None) -> TestScene:
    """A box (half extents ``half``) hanging centred under a fixed cup, its top on the lip plane, weight =
    load·F_max (or ``mass``)."""
    m = mass if mass is not None else load * f_max(plug) / G
    lip = MOUNT_Z - LIP_DROP

    def build(b: newton.ModelBuilder) -> None:
        SuctionCup(mount="fixed", stroke=0.0, contact_ke=CONTACT_KE, contact_kd=CONTACT_KD, lip_shape=LIP).build(b, body_xform(0.0, 0.0, MOUNT_Z), "suction")
        _box(b, m, (0.0, 0.0, lip - half[2]), half=half)

    return TestScene(build=build, init=_inputs(plug.label, pump, vent), periods=periods, ground=False)


def rest_scene(plug: Any, gap: float = 0.0, periods: int = 25, mass: float = 0.2) -> TestScene:
    """A box of ``mass`` on the ground under a fixed cup, ``gap`` below the lip (heavier than the pressure force: it
    stays on the ground and the seal stays whole)."""
    top = 2.0 * BOX[2]

    def build(b: newton.ModelBuilder) -> None:
        SuctionCup(mount="fixed", stroke=0.0, contact_ke=CONTACT_KE, contact_kd=CONTACT_KD, lip_shape=LIP).build(b, body_xform(0.0, 0.0, top + gap + LIP_DROP), "suction")
        _box(b, mass, (0.0, 0.0, BOX[2]))

    return TestScene(build=build, init=_inputs(plug.label, 1.0), periods=periods)


def side_scene(plug: Any, load: float, mu: float = 0.5, periods: int = 25) -> TestScene:
    """The cup axis horizontal (+x); a box hangs on the lip by friction: weight = load·μ·F_max."""
    m = load * mu * f_max(plug) / G
    half = (0.01, 0.04, 0.04)

    def build(b: newton.ModelBuilder) -> None:
        SuctionCup(mount="fixed", stroke=0.0, mu=mu, contact_ke=CONTACT_KE, contact_kd=CONTACT_KD, lip_shape=LIP).build(b, body_xform(0.0, 0.0, MOUNT_Z, q=SIDE_Q), "suction")
        _box(b, m, (LIP_DROP + half[0], 0.0, MOUNT_Z), half=half, mu=mu)

    return TestScene(build=build, init=_inputs(plug.label, 1.0), periods=periods, ground=False,
                     solver_kwargs=FRICTION)


def _box_drop(run: Any) -> np.ndarray:
    b = int(run.bodies("box")[0])
    z = run.trace.body_q[:, b, 2].astype(np.float64)
    return z[0] - z                                         # [K], downward displacement of world 0's box


def _held(run: Any, want: bool, what: str) -> LawResult:
    d = float(np.max(np.abs(_box_drop(run))))
    if want:
        return LawResult.within(d, HOLD_TOL, f"{what}: box moved {d * 1e3:.3g} mm (held: < {HOLD_TOL * 1e3:g} mm)",
                                "force = (P_atm - P)·π·R²·s on the box toward the cup")
    return LawResult(d > DROP_MIN, d, DROP_MIN, f"{what}: box moved {d * 1e3:.3g} mm (let go: > {DROP_MIN * 1e3:g} mm)",
                     "no hold above the pressure force: check the seal fraction and the force scaling")


def _shear(load: float):
    """Coulomb with the suction preload: below μ·N the box holds (moves < HOLD_TOL, no acceleration), above it slides
    at a = g·(1 - 1/load) (kinetic friction = the same μ·N). The acceleration is fitted over periods 3-12, while the
    seal is whole."""
    def check(run: Any) -> LawResult:
        t = np.asarray(run.trace.t, np.float64)
        z = -_box_drop(run)
        sl = slice(3, 13)
        c2, c1, _ = np.polyfit(t[sl], z[sl], 2)
        a, creep = -2.0 * c2, -c1 - 2.0 * c2 * float(t[sl].mean())
        want = G * (1.0 - 1.0 / load) if load > 1.0 else 0.0
        scale = G * (1.0 - 1.0 / 1.05)                   # the 1.05 case's slide acceleration
        err = (a - want) / scale
        what = "stick" if load <= 1.0 else f"slide at g·(1 - 1/{load:g}) = {want:.3g} m/s²"
        moved = float(np.max(np.abs(z - z[0])))
        ok = abs(err) <= 0.2 and (load > 1.0 or moved < HOLD_TOL)
        return LawResult(ok, err, 0.2, f"{load:g}·μ·ΔP·A: acceleration {a:.3g} m/s² ({what}); moved "
                         f"{moved * 1e3:.3g} mm, drift {creep * 1e3:.3g} mm/s",
                         "friction normal load = (P_atm - P)·π·R²·s: the force acts along the cup axis")
    return check


def _series(run: Any, name: str) -> np.ndarray:
    return run.trace.plugin(run.plugin.label, name).astype(np.float64)


def _check_evacuation(run: Any) -> LawResult:
    plug, c = run.plugin, run.plugin.config
    p = _series(run, "pressure")[:, 0, 0]
    t = np.asarray(run.trace.t, np.float64)
    p_ss = plug.steady_pressure(1.0)
    want = p_ss + (c.atmosphere - p_ss) * np.exp(-(c.pump_speed + c.leak) * t / c.dead_volume)
    dp, dw = c.atmosphere - p, c.atmosphere - want
    err = float(np.max(np.abs(dp[1:] - dw[1:]) / dw[1:]))
    return LawResult.within(err, 0.01, f"P_atm - P vs the exponential, worst {err:.2e} relative over "
                            f"{len(t) - 1} periods (τ = {c.dead_volume / (c.pump_speed + c.leak) * 1e3:.3g} ms)",
                            "V·dP/dt = S·(P_ult - P) + C_0·(P_atm - P) on a sealed lip, stepped exactly")


def _cross_time(run: Any, level: float) -> float:
    """First time the plug-in's force falls below ``level`` (linear between periods), or inf."""
    f, t = _series(run, "force")[:, 0], np.asarray(run.trace.t, np.float64)
    k = np.nonzero(f[1:] < level)[0] + 1                 # sample 0 is the state before the first step (force 0)
    if len(k) == 0:
        return math.inf
    i = int(k[0])
    if i == 1:
        return float(t[1])
    return float(t[i - 1] + (t[i] - t[i - 1]) * (f[i - 1] - level) / (f[i - 1] - f[i]))


def _drop_time_law(seal: float, conductance: str):
    def check(run: Any) -> LawResult:
        c, m = run.plugin.config, float(run.model.body_mass.numpy()[int(run.bodies("box")[0])])
        s = float(_series(run, "seal")[1, 0])
        cond = c.leak + c.open_leak * (1.0 - s) + (c.vent if conductance == "vent" else 0.0)
        f0 = (c.atmosphere - c.initial_pressure) * run.plugin.area * s
        want = c.dead_volume / cond * math.log(f0 / (m * G))
        got = _cross_time(run, m * G)
        dt_p = float(run.trace.t[1] - run.trace.t[0])
        moved = float(_box_drop(run)[-1])
        ok = abs(s - seal) < 1e-6 and abs(got - want) <= 2.0 * dt_p and moved > DROP_MIN
        return LawResult(ok, got - want, 2.0 * dt_p,
                         f"s {s:.4f} (geometry {seal:.4f}); hold < m·g at {got:.3f} s vs (V/C)·ln(F0/mg) {want:.3f} s; "
                         f"box fell {moved * 1e3:.3g} mm", "leak C_0 + C_open·(1 - s) (+ vent) on V, force ∝ s")
    return check


def _check_null(run: Any) -> LawResult:
    f, s = _series(run, "force"), _series(run, "seal")
    worst = float(np.max(np.abs(f)) + np.max(np.abs(s)))
    return LawResult(worst == 0.0, worst, 0.0, f"cup 5 mm above the box: |force| + |seal| max {worst:.3g}",
                     "a lip point seals only within seal_gap of a target surface")


def _check_balance(run: Any) -> LawResult:
    p, v, led = _series(run, "pressure")[:, :, 0], _series(run, "volume")[:, :, 0], _series(run, "ledger")
    amount = p * v                                       # Pa·m³ (gas)
    err = float(np.max(np.abs(amount - amount[0] - led.sum(axis=2))))
    return LawResult.within(err, 1e-9 * float(amount[0].max()), f"Pa·m³ unaccounted, worst {err:.3g}",
                            "every edge's throughput goes into the chamber amount and its ledger")


def suction_laws(plug: Any) -> list[Law]:
    """A-SUC-1..4, null, amount balance. Pre-evacuated scenes start at P_ss (``initial_pressure``)."""
    p_ss = plug.steady_pressure(1.0)
    held = {"initial_pressure": p_ss}
    # a box narrower than the lip (half-width 0.75·R in x, centred: no peel moment): the 6 lip points at
    # |cos φ| >= 0.924 hang 3.5 mm past its edges, the other 10 seal
    s_edge = 10.0 / 16.0
    narrow = (0.75 * plug.config.radius, BOX[1], BOX[2])
    m_edge = 0.5
    p1 = plug.config.atmosphere - 1.5 * m_edge * G / (plug.area * s_edge)
    return [
        Law("A-SUC-1_holds_below", lambda r: _held(r, True, "0.95·ΔP·A"), hold_scene(plug, 0.95),
            "a sealed cup holds m·g = 0.95·(P_atm - P_ss)·π·R²", held),
        Law("A-SUC-1_lets_go_above", lambda r: _held(r, False, "1.05·ΔP·A"), hold_scene(plug, 1.05),
            "and lets go at 1.05", held),
        Law("A-SUC-2_evacuation", _check_evacuation, rest_scene(plug, periods=20, mass=5.0 * f_max(plug) / G),
            "P(t) = P_ss + (P_atm - P_ss)·exp(-(S + C)·t/V), 1 % on P_atm - P"),
        Law("A-SUC-3_shear_holds_below", _shear(0.95), side_scene(plug, 0.95),
            "cup axis horizontal: friction holds m·g = 0.95·μ·ΔP·A (no acceleration)", held),
        Law("A-SUC-3_shear_slides_above", _shear(1.05), side_scene(plug, 1.05),
            "and slides at 1.05 with a = g·(1 - 1/1.05)", held),
        Law("A-SUC-4_seal_loss_drop_time", _drop_time_law(s_edge, "leak"),
            hold_scene(plug, 0.0, half=narrow, pump=0.0, periods=30, mass=m_edge),
            "lip over both edges of a narrow box (s = 10/16), pump off: the hold falls below m·g at (V/C)·ln(F0/mg)",
            {"initial_pressure": p1, "open_leak": 1.3e-5}),
        Law("A-SUC-4_vent_drop_time", _drop_time_law(1.0, "vent"),
            hold_scene(plug, 0.0, pump=0.0, vent=1.0, periods=30, mass=m_edge),
            "sealed, pump off, vent open: the hold falls below m·g at (V/C)·ln(F0/mg)",
            {"initial_pressure": plug.config.atmosphere - 1.5 * m_edge * G / plug.area, "vent": 6.0e-6}),
        Law("null_out_of_reach", _check_null, rest_scene(plug, gap=5.0e-3), "cup 5 mm above the box: no force"),
        Law("amount_balance", _check_balance, None, "chamber amount change = the edges' ledgers"),
    ]


__all__ = ["f_max", "hold_scene", "rest_scene", "side_scene", "suction_laws"]
