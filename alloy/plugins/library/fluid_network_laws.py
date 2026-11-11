# SPDX-License-Identifier: Apache-2.0
"""Law scenes and checks of the ``fluid_network`` plug-in (H-1..H-6, the vane stall, the amount balance, the null
case).

Every scene builds a vertical ``HydraulicCylinder`` (label "cyl": barrel fixed to the world, rod on joint "cyl_slide",
axis +z) so the harness has bodies; the network and the actuators come from each law's config override. Gravity loads
the rod (the plate plus 11.7 g of steel; the checks read the mass back from the model). Closed forms use the plug-in's
own parameters; bands are two-sided where a threshold is judged.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from alloy._vendor import newton

from alloy.plugins.fluid import P_ATM, Edge, Node, laminar_conductance, stored_energy
from alloy.plugins.fluid.actuators import Cylinder, Vane
from alloy.plugins.fluid.kernels import RHO0_R_T0
from alloy.plugins.harness.scene import Law, LawResult, TestScene, body_xform
from alloy.plugins.library.hydraulic import HydraulicCylinder

G = 9.81
ROD_STEEL = 0.0117                      # kg, HydraulicCylinder's rod shape (the plate adds plate_mass)
SUPPLY, BACK = 6.0e5, 3.0e5             # Pa abs: stall scene supply (cap) and back pressure (rod side)
BIG_C = 1.0e-8                          # m³/(s·Pa): a valve wide enough to hold a moving chamber at its reservoir
STALL_FC = 4.0                          # N, Coulomb seal friction in the stall scenes


def _scene(plug: Any, mass: float, x0: float, travel: float, periods: int, gravity=(0.0, 0.0, -G)) -> TestScene:
    """The cylinder at piston travel x0 of ``travel`` (joint limits [-x0, travel - x0]), rod mass ``mass``."""
    def build(b: newton.ModelBuilder) -> None:
        HydraulicCylinder(stroke=travel - x0, retract=x0, plate_mass=max(mass - ROD_STEEL, 1e-3)).build(
            b, body_xform(0.0, 0.0, 0.0), "cyl")

    return TestScene(build=build, periods=periods, ground=False, gravity=gravity)


def stall_cylinder(fc: float = STALL_FC) -> Cylinder:
    return Cylinder("cyl", joint="cyl_slide", a="cap", b="rod", bore=0.02, rod=0.02 / math.sqrt(2.0), stroke=0.1,
                    x0=0.05, coulomb=fc)


STALL = {"nodes": (Node("cap", pressure=SUPPLY), Node("rod", pressure=BACK)),
         "edges": (Edge("supply_cap", "orifice_gas", "cap", reservoir=SUPPLY, sonic_conductance=BIG_C),
                   Edge("back_rod", "orifice_gas", "rod", reservoir=BACK, sonic_conductance=BIG_C)),
         "actuators": (stall_cylinder(),), "openings": ()}


def stall_force(fc: float = 0.0) -> float:
    return stall_cylinder(fc).force(SUPPLY, BACK)


def stall_scene(plug: Any, load: float, periods: int = 20) -> TestScene:
    """Double-acting cylinder at mid stroke, cap at 6 bar, rod side at 3 bar; rod weight = load·F_stall."""
    return _scene(plug, load * stall_force() / G, 0.05, 0.1, periods)


def _travel(run: Any) -> np.ndarray:
    return run.trace.plugin(run.plugin.label, "travel").astype(np.float64)[:, 0, 0]


def _rod_mass(run: Any) -> float:
    return float(run.model.body_mass.numpy()[int(run.bodies("cyl_rod")[0])])


def _stall(direction: int):
    """direction +1: weight 0.98·(F - fc), the rod rises at (F - fc - m·g)/m; -1: 1.02·(F + fc), it sinks at
    (F + fc - m·g)/m. Fitted acceleration over periods 4-19 (the Coulomb term saturated), 20 % of the 2 % scale."""
    def check(run: Any) -> LawResult:
        t, x = np.asarray(run.trace.t, np.float64), _travel(run)
        m = _rod_mass(run)
        f = stall_force() - direction * STALL_FC
        want = (f - m * G) / m
        sl = slice(4, None)
        a = 2.0 * np.polyfit(t[sl], x[sl], 2)[0]
        err = abs(a - want) / (0.02 * G)
        ok = np.sign(a) == direction and err <= 0.2
        p = run.trace.plugin(run.plugin.label, "pressure")[-1, 0]
        return LawResult(bool(ok), float(a - want), 0.2 * 0.02 * G,
                         f"rod {'rises' if a > 0 else 'sinks'} at {a:.4g} m/s² (want {want:.4g}: F = P_A·A_A - P_B·A_B - "
                         f"P_atm·(A_A - A_B) {'-' if direction > 0 else '+'} fc = {f:.5g} N on {m:.4g} kg); P_A "
                         f"{p[0] / 1e5:.4f} bar, P_B {p[1] / 1e5:.4f} bar",
                         "cylinder force from both chambers' pressures and areas, friction against the motion")
    return check


def closed_scene(plug: Any, periods: int = 60) -> TestScene:
    """Single-acting, closed cap at P_atm at travel 80 mm; a 3 kg rod compresses it to equilibrium."""
    return _scene(plug, 3.0, 0.08, 0.1, periods)


def closed_config(n: float) -> dict:
    return {"nodes": (Node("cap", pressure=P_ATM, polytropic=n),), "edges": (),
            "actuators": (Cylinder("cyl", joint="cyl_slide", a="cap", bore=0.02, stroke=0.1, x0=0.08, viscous=100.0),),
            "openings": ()}


def _polytropic(run: Any) -> LawResult:
    act, n = run.plugin.config.actuators[0], run.plugin.config.nodes[0].polytropic
    m = _rod_mass(run)
    p_eq = P_ATM + m * G / act.area_a
    v0 = act.volumes(act.x0)[0]
    x_eq = (v0 * (P_ATM / p_eq) ** (1.0 / n) - act.dead_a) / act.area_a
    x = _travel(run)
    p = run.trace.plugin(run.plugin.label, "pressure").astype(np.float64)[:, 0, 0]
    v = run.trace.plugin(run.plugin.label, "volume").astype(np.float64)[:, 0, 0]
    inv = float(np.max(np.abs(p * v ** n / (p[0] * v[0] ** n) - 1.0)))
    err = abs(x[-1] - x_eq) / x_eq
    return LawResult(err <= 0.01 and inv <= 1e-9, err, 0.01,
                     f"n = {n:g}: rest travel {x[-1] * 1e3:.3f} mm vs {x_eq * 1e3:.3f} mm from P·V^n = const at "
                     f"P_atm + m·g/A ({err:.2e} relative); P·V^n drift {inv:.1e}",
                     "closed chamber: exact P·(V_old/V_new)^n step; force (P - P_atm)·A")


H1_LIQUID = {"nodes": (Node("mid", medium="liquid", volume=1.0e-4),),
             "edges": (Edge("in", "orifice", "mid", reservoir=50.0e5, cd_area=1.0e-6, density=850.0),
                       Edge("out", "orifice", "mid", reservoir=P_ATM, cd_area=2.0e-6, density=850.0)),
             "actuators": (), "openings": ()}
H1_GAS = {"nodes": (Node("mid", volume=1.0e-6),),
          "edges": (Edge("in", "orifice_gas", "mid", reservoir=SUPPLY, sonic_conductance=1.0e-9, critical_ratio=0.3),
                    Edge("out", "orifice_gas", "mid", reservoir=P_ATM, sonic_conductance=4.0e-9, critical_ratio=0.3)),
          "actuators": (), "openings": ()}
H2_LINE = (2.0e-3, 1.0, 0.05)           # d [m], L [m], μ [Pa·s]
H2 = {"nodes": (Node("mid", medium="liquid", volume=1.0e-7),),
      "edges": (Edge("pump", "source", "mid", flow=1.0e-6),
                Edge("line", "linear", "mid", reservoir=P_ATM, conductance=laminar_conductance(*H2_LINE))),
      "actuators": (), "openings": ()}


def _steady_flows(run: Any, k: int = 5) -> tuple[float, np.ndarray]:
    """(node pressure, each edge's flow into the node) over the last k periods."""
    led = run.trace.plugin(run.plugin.label, "ledger").astype(np.float64)[:, 0]
    t = np.asarray(run.trace.t, np.float64)
    p = float(run.trace.plugin(run.plugin.label, "pressure")[-1, 0, 0])
    return p, (led[-1] - led[-1 - k]) / (t[-1] - t[-1 - k])


def _h1_liquid(run: Any) -> LawResult:
    e_in, e_out = run.plugin.config.edges
    p, q = _steady_flows(run)
    want = np.array([e_in.cd_area * math.sqrt(2.0 * (e_in.reservoir - p) / e_in.density),
                     -e_out.cd_area * math.sqrt(2.0 * (p - e_out.reservoir) / e_out.density)])
    k = (e_out.cd_area / e_in.cd_area) ** 2
    p_want = (e_in.reservoir + k * e_out.reservoir) / (1.0 + k)
    err = max(float(np.max(np.abs(q[:2] / want - 1.0))), abs(p / p_want - 1.0))
    return LawResult.within(err, 0.02, f"node {p / 1e5:.4f} bar (closed form {p_want / 1e5:.4f}); flows "
                            f"{q[0]:.4e} / {q[1]:.4e} m³/s vs C_d·A·sqrt(2ΔP/ρ); worst {err:.1e}",
                            "liquid orifice g = C_d·A·sqrt(2/ρ)/sqrt(|ΔP|)")


def iso6358(c: float, b: float, p_up: float, p_down: float) -> float:
    """ISO 6358 throughput [Pa·m³/s] at T0."""
    r = p_down / p_up
    psi = 1.0 if r <= b else math.sqrt(max(1.0 - ((r - b) / (1.0 - b)) ** 2, 0.0))
    return c * p_up * psi * RHO0_R_T0


def _h1_gas(run: Any) -> LawResult:
    e_in, e_out = run.plugin.config.edges
    p, q = _steady_flows(run)
    lo, hi = e_out.reservoir, e_in.reservoir                # bisection on Q_in(P) = Q_out(P)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if iso6358(e_in.sonic_conductance, e_in.critical_ratio, e_in.reservoir, mid) > \
                iso6358(e_out.sonic_conductance, e_out.critical_ratio, mid, e_out.reservoir):
            lo = mid
        else:
            hi = mid
    want = np.array([iso6358(e_in.sonic_conductance, e_in.critical_ratio, e_in.reservoir, p),
                     -iso6358(e_out.sonic_conductance, e_out.critical_ratio, p, e_out.reservoir)])
    err = max(float(np.max(np.abs(q[:2] / want - 1.0))), abs(p / lo - 1.0))
    r_in, r_out = p / e_in.reservoir, e_out.reservoir / p
    return LawResult.within(err, 0.02, f"node {p / 1e5:.4f} bar (balance {lo / 1e5:.4f}); in r {r_in:.3f} "
                            f"({'choked' if r_in <= e_in.critical_ratio else 'subsonic'}), out r {r_out:.3f} "
                            f"({'choked' if r_out <= e_out.critical_ratio else 'subsonic'}); flows vs ISO 6358, worst "
                            f"{err:.1e}", "ISO 6358: Q = C·p_up·ψ(r)·ρ0·R·T0, linearised per step")


def _h2(run: Any) -> LawResult:
    d, length, mu = H2_LINE
    q = run.plugin.config.edges[0].flow
    p, _ = _steady_flows(run)
    want = 128.0 * mu * length * q / (math.pi * d ** 4)
    err = abs((p - P_ATM) / want - 1.0)
    return LawResult.within(err, 0.02, f"ΔP {(p - P_ATM) / 1e5:.5f} bar vs 128·μ·L·Q/(π·d⁴) {want / 1e5:.5f} bar",
                            "laminar_conductance = π·d⁴/(128·μ·L) on a linear edge")


def _balance(run: Any) -> LawResult:
    fl = run.plugin.fluid
    p, v = (run.trace.plugin(run.plugin.label, n).astype(np.float64)[:, 0] for n in ("pressure", "volume"))
    amount = fl.amount(p, v).sum(axis=1)
    led = run.trace.plugin(run.plugin.label, "ledger").astype(np.float64)[:, 0].sum(axis=1)
    err = float(np.max(np.abs(amount - amount[0] - led)))
    return LawResult.within(err, 1e-9 * float(np.abs(amount).max()), f"amount unaccounted, worst {err:.3g}",
                            "every edge's amount goes into its node; a volume change keeps the amount (n = 1)")


def _null(run: Any) -> LawResult:
    f = run.trace.plugin(run.plugin.label, "force").astype(np.float64)
    x = _travel(run)
    worst = float(np.max(np.abs(f))) + float(np.max(np.abs(x - x[0])))
    # bound: rounding only (the float32 start travel -> float64 volumes leave ~1e-12 N)
    return LawResult.within(worst, 1e-9, f"closed valves, chambers at P_atm, no gravity: |force| [N] + |travel "
                            f"change| [m] max {worst:.3g}", "the force is (P_A - P_atm)·A_A - (P_B - P_atm)·A_B")


# H-5: a linear valve (G) opens from a 10 bar supply into a closed 10 mL liquid chamber: τ = κ/G = V/(K·G) = 10 ms
H5_TAU, H5_V, H5_K = 0.01, 1.0e-5, 1.5e9
H5 = {"nodes": (Node("c", medium="liquid", volume=H5_V, bulk_modulus=H5_K),),
      "edges": (Edge("valve", "linear", "c", reservoir=10.0e5, conductance=H5_V / (H5_K * H5_TAU)),),
      "actuators": (), "openings": ()}
# H-6: pump (2 mL/s) -> line (20 mL) -> valve (2 bar drop) -> cap of a vertical cylinder lifting 5 kg (K_eff 1 GPa)
H6 = {"nodes": (Node("line", medium="liquid", volume=2.0e-5, bulk_modulus=1.0e9),
                Node("cap", medium="liquid", volume=1.0e-4, bulk_modulus=1.0e9)),
      "edges": (Edge("pump", "source", "line", flow=2.0e-6), Edge("valve", "linear", "line", "cap", conductance=1.0e-11)),
      "actuators": (Cylinder("cyl", joint="cyl_slide", a="cap", bore=0.02, stroke=0.1, x0=0.02, dead_a=5.0e-5,
                             viscous=200.0),),
      "openings": ()}
# vane stall: a 0.2 m arm on a horizontal hinge (axis -y: +θ raises it), A at 6 bar, B at 3 bar, D = 6.5 cm³/rad
VANE_L, VANE_D = 0.2, 6.5e-6
VANE = {"nodes": (Node("va", pressure=SUPPLY), Node("vb", pressure=BACK)),
        "edges": (Edge("supply_a", "orifice_gas", "va", reservoir=SUPPLY, sonic_conductance=BIG_C),
                  Edge("back_b", "orifice_gas", "vb", reservoir=BACK, sonic_conductance=BIG_C)),
        "actuators": (Vane("vane", joint="vane_hinge", a="va", b="vb", displacement=VANE_D, travel=math.pi,
                           x0=0.5 * math.pi),),
        "openings": ()}


def vane_scene(plug: Any, load: float, periods: int = 12) -> TestScene:
    """The arm horizontal, its 2 cm load cube at x = VANE_L; weight torque = load·(P_A - P_B)·D."""
    m = load * (SUPPLY - BACK) * VANE_D / (G * VANE_L)

    def build(b: newton.ModelBuilder) -> None:
        hinge = body_xform(0.0, 0.0, 0.3)
        arm = b.add_link(xform=hinge, label="arm")
        b.add_shape_box(arm, xform=body_xform(VANE_L, 0.0, 0.0), hx=0.01, hy=0.01, hz=0.01,
                        cfg=newton.ModelBuilder.ShapeConfig(density=m / 8.0e-6))
        j = b.add_joint_revolute(-1, arm, parent_xform=hinge, axis=(0.0, -1.0, 0.0), label="vane_hinge")
        b.add_articulation([j], label="vane")

    return TestScene(build=build, periods=periods, ground=False)


def _vane_stall(direction: int):
    """direction +1: weight torque 0.98·τ, the arm rises at (τ - m·g·L)/I; -1: 1.02·τ, it sinks. Fitted angular
    acceleration over periods 1-11 (the arm within 2° of horizontal), 20 % of the 2 % scale."""
    def check(run: Any) -> LawResult:
        from alloy.plugins.library.fluid_network import _moving_inertia  # noqa: PLC0415

        t, th = np.asarray(run.trace.t, np.float64), _travel(run)
        b = int(run.bodies("arm")[0])
        m, lever = float(run.model.body_mass.numpy()[b]), float(run.model.body_com.numpy()[b][0])
        inertia = _moving_inertia(run.model, int(run.plugin.joint.numpy()[0, 0]), True)
        tau = run.plugin.config.actuators[0].force(SUPPLY, BACK)
        want = (tau - m * G * lever) / inertia
        alpha = 2.0 * np.polyfit(t[1:], th[1:], 2)[0]
        err = abs(alpha - want) / (0.02 * tau / inertia)
        ok = np.sign(alpha) == direction and err <= 0.2
        return LawResult(bool(ok), float(alpha - want), 0.2 * 0.02 * tau / inertia,
                         f"arm {'rises' if alpha > 0 else 'sinks'} at {alpha:.4g} rad/s² (want {want:.4g}: τ = "
                         f"(P_A - P_B)·D = {tau:.4g} N·m against m·g·L {m * G * lever:.4g} N·m, I {inertia:.3g} kg·m²)",
                         "vane torque (P_A - P_B)·D about the hinge, angle from the joint frames")
    return check


def _h5(run: Any) -> LawResult:
    e = run.plugin.config.edges[0]
    t = np.asarray(run.trace.t, np.float64)
    p = run.trace.plugin(run.plugin.label, "pressure").astype(np.float64)[:, 0, 0]
    want = e.reservoir + (p[0] - e.reservoir) * np.exp(-t / H5_TAU)
    err = float(np.max(np.abs(p - want)) / abs(e.reservoir - p[0]))
    return LawResult.within(err, 0.05, f"P(t) vs first order τ = V/(K·G) = {H5_TAU * 1e3:g} ms: worst {err:.1e} of "
                            "the step", "κ = V/K on a liquid node, linear valve g = G·u")


def _h6(run: Any) -> LawResult:
    lab, fl = run.plugin.label, run.plugin.fluid
    work = run.trace.plugin(lab, "work").astype(np.float64)[-1, 0]
    mech = float(run.trace.plugin(lab, "node_work").astype(np.float64)[-1, 0].sum())
    p, v = (run.trace.plugin(lab, n).astype(np.float64)[:, 0] for n in ("pressure", "volume"))
    stored = stored_energy(fl.net, p, v).sum(axis=1)
    pump, throttle = float(work[0]), float(work[1])
    rest = mech + throttle + float(stored[-1] - stored[0])
    err = abs(pump - rest) / pump
    x = _travel(run)
    return LawResult.within(err, 0.02, f"pump {pump:.4g} J = mechanical {mech:.4g} + throttling {throttle:.4g} + "
                            f"stored {stored[-1] - stored[0]:.3g} (sum {rest:.4g}, {err:.1e} off); rod rose "
                            f"{(x[-1] - x[0]) * 1e3:.2f} mm", "pump q·(P - P_tank), valve Q·ΔP, chamber ∫(P - P_ref)·dV")


TORRICELLI = {"nodes": (Node("tank", medium="tank", volume=1.0e-3, area=0.01),),
              "edges": (Edge("waste", "orifice", "tank", cd_area=1.78e-3, density=998.0, dp_lam=1.0),),
              "actuators": (), "openings": ()}
RINSE_Q = 0.04                          # m³/s: fresh water in and mixture out of a 20 L tank (Q·t/V = 1 in 0.5 s)
RINSE = {"nodes": (Node("wash", medium="tank", volume=0.02, area=0.1, species=(5.0,)),),
         "edges": (Edge("fresh", "source", "wash", flow=RINSE_Q), Edge("out", "source", "wash", flow=-RINSE_Q)),
         "actuators": (), "openings": ()}


def _tank_volume(run: Any) -> tuple[np.ndarray, np.ndarray]:
    p, v = (run.trace.plugin(run.plugin.label, n).astype(np.float64)[:, 0] for n in ("pressure", "volume"))
    return np.asarray(run.trace.t, np.float64), run.plugin.fluid.amount(p, v)[:, 0]


def _f14(run: Any) -> LawResult:
    """F-14: sqrt(V) = sqrt(V0) - Cd·A·sqrt(2g/A_t)·t/2 until 90 % has drained."""
    t, v = _tank_volume(run)
    tank, edge = run.plugin.config.nodes[0], run.plugin.config.edges[0]
    v0 = tank.volume
    want = np.maximum(math.sqrt(v0) - 0.5 * edge.cd_area * math.sqrt(2.0 * G / tank.area) * t, 0.0) ** 2
    use = want > 0.1 * v0
    err = float(np.max(np.abs(v[use] - want[use]))) / v0
    return LawResult.within(err, 0.02, f"V(t) vs Torricelli over {int(use.sum())} periods (to 90 % drained): worst "
                            f"{err:.1e} of V0", "tank head ρ·g·V/A through the liquid orifice law")


def _f16(run: Any) -> LawResult:
    """F-16 (network part): c = c0·exp(-Q·t/V) in a tank rinsed at Q with V kept."""
    t, v = _tank_volume(run)
    mass = run.trace.plugin(run.plugin.label, "species").astype(np.float64)[:, 0, 0, 0]
    tank = run.plugin.config.nodes[0]
    c = mass / v
    want = tank.species[0] * np.exp(-RINSE_Q * t / tank.volume)
    err = float(np.max(np.abs(c / want - 1.0)))
    return LawResult.within(err, 0.03, f"c(t) vs c0·exp(-Q·t/V) to Q·t/V = {RINSE_Q * t[-1] / tank.volume:.2f}: worst "
                            f"{err:.1e}", "implicit upwind species transport")


def fluid_network_laws(plug: Any) -> list[Law]:
    """H-1 (liquid and gas orifices), H-2, H-3 (slow n = 1, fast n = 1.4), H-4 (stall both ways), H-5, H-6, the vane
    stall both ways, balance, null; the open tank's F-14 (Torricelli) and F-16 (dilution, network part)."""
    fixed = _scene(plug, 0.5, 0.0, 0.1, 25)                 # rod on its lower stop; the network has no actuator
    f = stall_force()
    null = {**STALL, "nodes": (Node("cap"), Node("rod")), "openings": (("supply_cap", 0.0), ("back_rod", 0.0))}
    return [
        Law("H-1_orifice_liquid", _h1_liquid, fixed, "node between two orifices: steady Q = C_d·A·sqrt(2ΔP/ρ), 2 %",
            H1_LIQUID),
        Law("H-1_orifice_gas", _h1_gas, fixed, "node between a choked and a subsonic ISO 6358 valve, 2 %", H1_GAS),
        Law("H-2_laminar_line", _h2, fixed, "pump into a line: ΔP = 128·μ·L·Q/(π·d⁴), 2 %", H2),
        Law("H-3_closed_isothermal", _polytropic, closed_scene(plug), "closed chamber, n = 1: rest from P·V = const, "
            "1 %", closed_config(1.0)),
        Law("H-3_closed_adiabatic", _polytropic, closed_scene(plug), "closed chamber, n = 1.4 (a closed volume "
            "change only): rest from P·V^1.4 = const, 1 %", closed_config(1.4)),
        Law("H-4_stall_rises_below", _stall(+1), _scene(plug, 0.98 * (f - STALL_FC) / G, 0.05, 0.1, 20),
            "weight 0.98·(F - fc): the rod rises", STALL),
        Law("H-4_stall_sinks_above", _stall(-1), _scene(plug, 1.02 * (f + STALL_FC) / G, 0.05, 0.1, 20),
            "weight 1.02·(F + fc): the rod sinks", STALL),
        Law("amount_balance", _balance, stall_scene(plug, 0.9), "chamber amounts against the edge ledgers", STALL),
        Law("null_closed_at_atmosphere", _null, _scene(plug, 1.0, 0.05, 0.1, 10, gravity=(0.0, 0.0, 0.0)),
            "valves shut, chambers at P_atm: no force", null),
        Law("H-5_step_response", _h5, _scene(plug, 0.5, 0.0, 0.1, 10), "a valve opens into a closed liquid chamber: "
            "first-order τ = V/(K·G), 5 %", H5),
        Law("H-6_energy_ledger", _h6, _scene(plug, 5.0, 0.02, 0.1, 50), "pump -> valve -> cylinder lifting 5 kg: pump "
            "work = mechanical + throttling + stored, 2 %", H6),
        Law("vane_stall_rises_below", _vane_stall(+1), vane_scene(plug, 0.98), "weight torque 0.98·(P_A - P_B)·D: "
            "the arm rises", VANE),
        Law("vane_stall_sinks_above", _vane_stall(-1), vane_scene(plug, 1.02), "weight torque 1.02·(P_A - P_B)·D: "
            "the arm sinks", VANE),
        Law("F-14_tank_drain", _f14, fixed, "an open tank through an orifice: Torricelli's sqrt(V) falling linearly, "
            "2 %", TORRICELLI),
        Law("F-16_tank_dilution", _f16, fixed, "a tank rinsed with fresh water at Q, V kept: c = c0·exp(-Q·t/V), 3 %",
            RINSE),
    ]


__all__ = ["fluid_network_laws", "iso6358", "stall_force", "stall_scene", "vane_scene"]
