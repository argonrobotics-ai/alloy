# SPDX-License-Identifier: Apache-2.0
"""The ``Thermal`` plug-in's laws (heat v1, plan section 4), on small scenes.

- **cooling (T-1):** a floating hot box (no contacts) follows T_amb + (T0 - T_amb)·exp(-hA·t/C), within 1 % of
  (T - T_amb).
- **conduction_energy (T-2):** two stacked boxes, convection off: C_a·T_a + C_b·T_b is conserved within 1e-6
  (relative).
- **conduction_rate (T-2):** their difference decays as exp(-∫G·(1/C_a + 1/C_b) dt), with G the recorded contact
  conductance, within 5 %.
- **friction_heat (T-3):** a box slid to rest on a plate fixed to the world. The heat gained, Σ C·ΔT, equals the
  kinetic energy lost, ½·m·v0², within 1 %. This is independent of the plug-in's own friction power.
  - The scene steps at dt ≤ 5e-4 s. The heat falls short of the kinetic energy by an amount that shrinks with the
    step, the engine's time discretisation: 1.9 % at 1.25 ms, 1.2 % at 0.625 ms and 0.16 % at 0.3125 ms (CPU,
    [internal note]).
- **motor (T-4):** a constant torque. The winding temperature follows the exact two-node solution within 1 % of its
  rise, and the torque limit is the derating line.
- **external_heat:** a closed floating box (convection off) given Q through the ``external_heat`` input rises by
  exactly Q/C (within 1e-9, relative), and ``external_heat_total`` reads Q.
- **null:** at ambient with no sources, temperatures stay at ambient (within 1e-9 K).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from alloy._vendor import newton

from alloy.plugins.harness.scene import Law, LawResult, TestScene, body_xform

AMB = 293.15
BOX = 0.01
V0 = 0.5
COOL = {"initial": (353.15,), "specific_heat": (100.0,), "convection": 100.0}
STACK = {"bodies": ("a", "b"), "initial": (353.15, AMB), "specific_heat": (100.0, 100.0), "convection": 0.0,
         "contact_conductance": 1.0e4, "friction_heat": False}
SLIDE = {"bodies": ("plate", "box"), "specific_heat": (1.0, 1.0), "convection": 0.0, "contact_conductance": 0.0}
MOTOR = {"motors": ("m0",), "torque_constant": 0.1, "winding_resistance": 0.1, "winding_capacity": 1.0,
         "housing_capacity": 2.0, "r_winding_housing": 0.5, "r_housing_ambient": 1.0, "t_derate": 303.15,
         "t_max": 333.15, "torque_max": 10.0}
MOTOR_TORQUE = 2.0
EXT = {"specific_heat": (100.0,), "convection": 0.0}
EXT_Q_UJ = 5_000_000                                           # 5 J, put in once at the start


def _cfg(density: float, mu: float = 0.3) -> newton.ModelBuilder.ShapeConfig:
    return newton.ModelBuilder.ShapeConfig(density=density, mu=mu)


def _box(b: newton.ModelBuilder, label: str, x: float, z: float, mass: float, half=(BOX, BOX, BOX)) -> int:
    body = b.add_body(xform=body_xform(x, 0.0, z), label=label)
    b.add_shape_box(body, hx=half[0], hy=half[1], hz=half[2], cfg=_cfg(mass / (8 * half[0] * half[1] * half[2])))
    return body


def _build_float(b: newton.ModelBuilder) -> None:
    _box(b, "box", 0.0, 0.5, 0.008)


def _build_stack(b: newton.ModelBuilder) -> None:
    _box(b, "a", 0.0, BOX + 5e-4, 0.008)
    _box(b, "b", 0.0, 3 * BOX + 1e-3, 0.008)


def _build_slide(b: newton.ModelBuilder) -> None:
    half = (0.04, 0.03, 0.005)                         # a plate fixed to the world: no slip, no ground contact
    plate = b.add_link(xform=body_xform(0.0, 0.0, 0.005), label="plate")
    b.add_shape_box(plate, hx=half[0], hy=half[1], hz=half[2], cfg=_cfg(1.0 / (8 * half[0] * half[1] * half[2])))
    b.add_articulation([b.add_joint_fixed(parent=-1, child=plate, parent_xform=body_xform(0.0, 0.0, 0.005))])
    _box(b, "box", -0.025, 0.01 + BOX + 1e-3, 0.05)


def _push_box(sim: Any) -> None:
    m = sim.model
    qd = sim.state.joint_qd.numpy()
    child, start = m.joint_child.numpy(), m.joint_qd_start.numpy()
    boxes = set(next(p for p in sim.plugins.plugins.values() if p.name == "thermal").ctx.bodies("box").numpy())
    for j, body in enumerate(child):
        if int(body) in boxes:
            qd[start[j]] = V0
    sim.state.joint_qd.assign(qd)
    newton.eval_fk(m, sim.state.joint_q, sim.state.joint_qd, sim.state)


def _set_torque(sim: Any) -> None:
    for p in sim.plugins.plugins.values():
        if p.name == "thermal":
            p.set_input("motor_torque", MOTOR_TORQUE)


def _put_heat(sim: Any) -> None:
    for p in sim.plugins.plugins.values():
        if p.name == "thermal":
            p.set_input("external_heat", EXT_Q_UJ)


def float_scene() -> TestScene:
    """A box floating in still air, with no contacts (the default test scene)."""
    return TestScene(build=_build_float, periods=25, ground=False, gravity=(0.0, 0.0, 0.0))


def _series(run: Any, name: str) -> np.ndarray:
    return np.asarray(run.trace.plugin(run.plugin.label, name), np.float64)


def _caps(run: Any) -> np.ndarray:
    return run.plugin._capacity.numpy()[: run.plugin.B]


def _check_cooling(run: Any) -> LawResult:
    temp, t = _series(run, "temperature")[:, 0, 0], np.asarray(run.trace.t)
    c, ha = float(_caps(run)[0]), float(run.plugin._h_area.numpy()[0])
    want = AMB + (temp[0] - AMB) * np.exp(-ha * t / c)
    err = float(np.abs((temp - want) / (want - AMB)).max())
    return LawResult.within(err, 0.01, f"T(end) {temp[-1]:.4f} K vs {want[-1]:.4f} K (hA/C = {ha / c:.4g}/s)",
                            "convection must be h·A·(T - T_amb), with A from the collision shapes")


def _check_energy(run: Any) -> LawResult:
    temp, c = _series(run, "temperature")[:, 0, :], _caps(run)
    e = temp @ c
    rel = float(np.abs(e - e[0]).max() / abs(e[0]))
    return LawResult.within(rel, 1e-6, f"Σ C·T drifts {rel:.2e} (relative)", "conduction must move heat, not make it")


def _check_rate(run: Any) -> LawResult:
    temp, t = _series(run, "temperature")[:, 0, :], np.asarray(run.trace.t)
    g = _series(run, "conductance")[:, 0, 0]
    c = _caps(run)
    k0 = 3                                                     # after the top box has landed
    rate = g * (1.0 / c[0] + 1.0 / c[1])
    integ = np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(t))])
    dtemp = temp[:, 0] - temp[:, 1]
    want = dtemp[k0] * np.exp(-(integ - integ[k0]))
    err = float(np.abs((dtemp[k0:] - want[k0:]) / dtemp[k0]).max())
    return LawResult.within(err, 0.05, f"ΔT(end) {dtemp[-1]:.4f} K vs {want[-1]:.4f} K, G {g[-1]:.3g} W/K",
                            "contact conduction must be G·(T_j - T_i) on both sides")


def _check_friction(run: Any) -> LawResult:
    temp, c = _series(run, "temperature")[:, 0, :], _caps(run)
    heat = float((temp[-1] - temp[0]) @ c)
    m = float(run.model.body_mass.numpy()[run.bodies("box")[0]])
    v_end = float(np.linalg.norm(run.trace.body_qd[-1, run.bodies("box")[0], :3]))
    ke = 0.5 * m * (V0 ** 2 - v_end ** 2)
    return LawResult.within((heat - ke) / ke, 0.01, f"Σ C·ΔT {heat * 1e3:.4f} mJ vs ½·m·Δv² {ke * 1e3:.4f} mJ",
                            "friction heat must be |f_t · v_slip| at the contact")


def _two_node(t: np.ndarray, p: dict, power: float) -> np.ndarray:
    a = np.array([[-1.0 / (p["winding_capacity"] * p["r_winding_housing"]),
                   1.0 / (p["winding_capacity"] * p["r_winding_housing"])],
                  [1.0 / (p["housing_capacity"] * p["r_winding_housing"]),
                   -(1.0 / p["r_winding_housing"] + 1.0 / p["r_housing_ambient"]) / p["housing_capacity"]]])
    b = np.array([power / p["winding_capacity"], 0.0])
    lam, vec = np.linalg.eig(a)
    xs = -np.linalg.solve(a, b)                                # steady state (rise above ambient)
    coef = np.linalg.solve(vec, -xs)
    return np.real(np.array([xs + vec @ (coef * np.exp(lam * ti)) for ti in t]))[:, 0]


def _check_motor(run: Any) -> LawResult:
    t = np.asarray(run.trace.t)
    tw, lim = _series(run, "winding")[:, 0, 0], _series(run, "torque_limit")[:, 0, 0]
    power = (MOTOR_TORQUE / MOTOR["torque_constant"]) ** 2 * MOTOR["winding_resistance"]
    want = AMB + _two_node(t, MOTOR, power)
    err = float(np.abs(tw[1:] - want[1:]).max() / (want[-1] - AMB))
    frac = np.clip((MOTOR["t_max"] - tw[-1]) / (MOTOR["t_max"] - MOTOR["t_derate"]), 0.0, 1.0)
    lim_err = abs(lim[-1] - MOTOR["torque_max"] * frac)
    return LawResult.within(err + lim_err, 0.01, f"winding {tw[-1]:.3f} K vs {want[-1]:.3f} K; limit {lim[-1]:.3f} "
                            f"N·m vs {MOTOR['torque_max'] * frac:.3f}", "the two-node motor model or the derating line")


def _check_external(run: Any) -> LawResult:
    temp, total = _series(run, "temperature")[:, 0, 0], _series(run, "external_heat_total")[:, 0, 0]
    q = EXT_Q_UJ * 1e-6
    want = AMB + q / float(_caps(run)[0])
    err = max(abs(temp[-1] - want) / (want - AMB), abs(total[-1] - q) / q)
    return LawResult.within(err, 1e-9, f"T(end) {temp[-1]:.9f} K vs {want:.9f} K; total {total[-1]:.9f} J",
                            "external heat must enter the energy once, in full")


def _check_null(run: Any) -> LawResult:
    d = float(np.abs(_series(run, "temperature") - AMB).max())
    return LawResult.within(d, 1e-9, f"max |T - T_amb| {d:.3g} K with no sources", "no source must make no heat")


def thermal_laws() -> list[Law]:
    """The laws above, each on its scene."""
    stack = TestScene(build=_build_stack, periods=25, ground_mu=1.0)
    slide = TestScene(build=_build_slide, periods=25, ground_mu=1.0, init=_push_box, dt=5e-4)   # see the module doc
    motor = TestScene(build=_build_float, periods=25, ground=False, gravity=(0.0, 0.0, 0.0), init=_set_torque)
    heated = TestScene(build=_build_float, periods=25, ground=False, gravity=(0.0, 0.0, 0.0), init=_put_heat)
    return [
        Law("cooling", _check_cooling, config=COOL, doc="Newton cooling of a lone body"),
        Law("conduction_energy", _check_energy, scene=stack, config=STACK, doc="conduction conserves Σ C·T"),
        Law("conduction_rate", _check_rate, scene=stack, config=STACK, doc="ΔT decays at G·(1/C_a + 1/C_b)"),
        Law("friction_heat", _check_friction, scene=slide, config=SLIDE, doc="heat = kinetic energy lost"),
        Law("motor", _check_motor, config=MOTOR, scene=motor, doc="two-node winding/housing + derating"),
        Law("external_heat", _check_external, scene=heated, config=EXT, doc="ΔT = Q/C for heat from other plug-ins"),
        Law("null", _check_null, doc="no sources, no heat"),
    ]
