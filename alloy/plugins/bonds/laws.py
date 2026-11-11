# SPDX-License-Identifier: Apache-2.0
"""The ``Bonds`` plug-in's laws, on the coupled rigs in ``rig.py`` (CPU). The gantry follows ``driver.py`` scripts.

**Peel laws: 0.08 mm PET (3M 7818 face stock) on steel, 25 mm wide, peeled from a 5 mm tab.**
- **peel_90 / peel_180.** The mean steady pull per width, against the 7818 sheet's row at that angle (380 or
  470 N/m). The bound is 5 %. The 90° grip follows the front (the sliding fixture). The 180° grip folds the arm back
  3 mm above the tape. The row is reproduced by calibration (adhesives.py). What the law tests is the engine's
  front: its measured angle and width, and its steady advance. The pull's work on the grip must also equal the
  fracture energy G·b·Δa over the same window, within 2 %. That is Kendall's energy balance, so it checks the front
  kinematics: front advance = u·(grip path)/(1 - cos θ). The bonds released (booked a mesh row at a time) must match
  b·Δa within one row.
- **slack.** The grip lifts, then comes back down. Once the arm is slack the pull is 0 and nothing more is
  released.
- **stall.** The grip's z drive is limited to 4 N, below the 9.5 N peel force plus the link's weight. The front
  stops (under one row more after the first 0.1 s) and the arm holds the robot: the time-averaged pull (impulse
  rate) = drive force - weight, within 10 % (momentum balance).

**Formation laws: a 20 x 20 mm patch pressed onto the plate by a force-limited pad, then the pad lifts.**
- **forms_above_p_form.** At 2x p_form for 2x the dwell, every vertex is bonded. At 0.4x p_form nothing bonds. At
  2x p_form for 0.6x the dwell nothing bonds; a dwell clock running 2x fast would bond there (DC5 review S2).
- **null.** A pad that never touches: no bond, no pull, and the patch stays where it lay.

**Wet-out laws (wetout.py): a 10 x 10 mm patch under one pass of the standard ASTM D3330 roller at 20 mm/s.**
- **W-3_roll_dx<cell>.** The roller rides a free vertical slide (its 2.04 kg alone presses). Its line load, measured
  on the film, is m·g/width within 5 %, and the patch's mean wet-out is the Hertz contact's exact value at that load,
  1 - exp(-∫(p/P_REF)^M dt/tau_w) over the pass (wetout.exposure), within 2.5 %, at 2.5 mm and at 1.25 mm cells: so
  the two cell sizes agree within 5 % (WP13 must-fix 1).
- **W-4_roll_servo.** A servo presses the roller at 20 N plus its link's weight: the measured load reads that force
  over the width (5 %), and w the Hertz value at it (2.5 %).
- **W-5_roll_lifted.** A roller held 1 mm above the label, never touching it: no load, no bond, w = 0.
W-1 (the ODE per step) and W-2 (rises with pressure, none below p_form) are in tests/plugins/test_bonds_wetout.py.

**Curved substrate law (SCR-WP8-11 B3): a 10 x 10 mm label posed wrapped on a 25 mm radius tube (axis along the
roller), one pass of the dead-weight standard roller whose contact sweeps the label at 20 mm/s.** A roller moving along
x over a tube moves its contact (R_s + r)/(R_s + 2r + R) = 0.35x as fast, so the gantry moves at 58 mm/s over a
longer span. Off the top the load grows as 1/cos(phi) and the contact's speed too, so the exposure ∫p dt (M = 1) is the
same for every vertex while the whole Hertz profile is above p_form.
- **C-1_tube_dx<cell>.** The whole label bonds, the load reads m·g/width (5 %), and the mean wet-out is the Hertz
  pass with 1/R_e = 1/R + 1/R_s (2.5 %), at 2.5 and 1.25 mm cells. R_e = 16.4 mm here against the roller's 47.6: the
  contact is 0.6x as wide and 1.7x as high, so a flat-substrate model reads a different w (reported). The contact
  sweeps from 2 mm before the label to 2 mm past it: the roller starts just above the label's leading edge
  (``rig.LAND_GAP``), the label creeps 0.2-0.3 mm forward under it, and a far row the contact has not cleared reads
  low (a 1 mm margin left the last row at 0.37 for 0.44). Dropped 0.53 mm onto the edge (the start before
  2026-11-10), the knock crept the label 0.6-1.2 mm, and at 1.25 mm cells on main slid it around the tube and off.
- **GA_standard_flat / GA_standard_tube.** One standard application (ASTM D3330: two passes, one each way, at 10 mm/s
  at the contact) on the flat patch and on the tube: GA's wet-out criterion (``wetout.ga_wet_out``) passes on the
  label and on its 3 mm edge band, w/W_STD >= 0.90 for each, and every vertex bonds (WP13 review of B3).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from alloy.plugins.harness.scene import Law, LawResult, TestScene
from alloy.solvers.film.rigs import grid_mesh

from .adhesives import ROWS_7818, adhesive
from .driver import GantryScript
from .rig import PeelRig, PressRig, RollRig, TubeRig, simulation
from .wetout import STD_SPEED, exposure, ga_wet_out, wet_tau

STEEL = "3m_7818/stainless_steel"
PEEL = {"substrates": ("plate",), "adhesives": (STEEL,), "grip": "grip", "grip_reach": 1.5e-3, "prebond": True}
PRESS = {"substrates": ("plate",), "adhesives": (STEEL,), "grip": "", "prebond": False}
ROLL = {**PRESS, "rollers": ({"shape": "roller"},)}            # the standard roller card (wetout.STANDARD_ROLLER)
TUBE = {**ROLL, "substrates": ("tube",)}
ROLL_RIG = {"length": 0.010, "width": 0.010, "dt": 1.0e-3}
WINDOW = (0.010, 0.030)                # m of front travel averaged by the peel laws
STALL_EFFORT = 4.0                     # N
PRESS_DWELL_PERIODS = 10               # 0.1 s at the rig's 10 ms period = 2x the 0.05 s dwell
ROLL_SPEED = 0.02                      # m/s: the roller keeps a vertex above p_form for 0.13 s (> the 0.05 s dwell)


def _no_template(b: Any) -> None:
    raise RuntimeError("bonds law scenes build through TestScene.make")


def _binder(sim: Any) -> None:
    from .plugin import Bonds  # noqa: PLC0415

    plugins = list(sim.plugins.plugins.values())
    bonds = next((p for p in plugins if isinstance(p, Bonds)), None)      # None: the harness's timing run without it
    for p in plugins:
        if isinstance(p, GantryScript):
            p.bind(sim, bonds)


def scene(rig: PeelRig, mode: str, periods: int, worlds: int = 1, **script: Any) -> TestScene:
    """A harness scene of ``rig`` with a ``GantryScript`` companion in ``mode``."""
    return TestScene(build=_no_template, worlds=worlds, periods=periods, control_dt=rig.control_dt, dt=rig.dt,
                     companions=lambda: [GantryScript(mode=mode, **script)], init=_binder,
                     make=lambda plugins, device, w, graph: simulation(rig, list(plugins), w, device, graph))


def _series(run: Any, name: str) -> np.ndarray:
    return np.asarray(run.trace.plugin(run.plugin.label, name), np.float64)[:, 0]


def _peel(run: Any, row: float) -> LawResult:
    t, b, front = _series(run, "peel_force"), _series(run, "width"), _series(run, "front")
    work, rel, ang = _series(run, "work"), _series(run, "released"), _series(run, "angle")
    win = np.flatnonzero((front >= WINDOW[0]) & (front <= WINDOW[1]))
    if win.size < 3:
        return LawResult(False, float("inf"), 0.05, f"front reached {front.max() * 1e3:.1f} mm, window {WINDOW}",
                         "the grip must peel 30 mm: check the gantry script and the grip bonds")
    i0, i1 = win[0], win[-1]
    fb = float(np.mean(t[win] / b[win]))
    e_row = (fb - row) / row
    card = adhesive(STEEL)
    g = np.array([card.g(1.0 - np.cos(np.radians(a))) for a in ang])
    da = np.diff(front[i0:i1 + 1])
    fracture = float(np.sum(g[i0 + 1:i1 + 1] * b[i0 + 1:i1 + 1] * da))      # G·b·Δa over the continuous front
    e_energy = (work[i1] - work[i0]) / fracture - 1.0
    rows = abs((rel[i1] - rel[i0]) - np.mean(b[win]) * (front[i1] - front[i0])) / (2.5e-3 * np.mean(b[win]))
    value = max(abs(e_row), abs(e_energy) * 2.5, (rows - 1.0) * 0.05)    # energy 2 %; ledger within one row
    return LawResult.within(value, 0.05, f"F/b {fb:.1f} N/m vs sheet {row:.0f} ({e_row:+.2%}), angle "
                            f"{np.mean(ang[win]):.1f} deg, width {np.mean(b[win]) * 1e3:.2f} mm; work / G·b·Δa "
                            f"{1 + e_energy:.4f}; released vs b·Δa {rows:.2f} rows",
                            "the front's angle, width or advance (Kendall's balance)")


def _check_90(run: Any) -> LawResult:
    return _peel(run, ROWS_7818["stainless_steel"][0])


def _check_180(run: Any) -> LawResult:
    return _peel(run, ROWS_7818["stainless_steel"][1])


def _check_slack(run: Any) -> LawResult:
    t, rel = _series(run, "peel_force"), _series(run, "released")
    tt = np.asarray(run.trace.t)
    late = tt > 0.2 + 0.05                       # after the turn plus the time to take up the slack
    pull = float(np.abs(t[late]).max())
    more = float(rel[late].max() - rel[late].min())
    return LawResult.within(pull + more * 1e4, 1e-9, f"after the turn: max pull {pull:.3g} N, released "
                            f"{more * 1e6:.3g} mm² more", "a slack arm must pull with 0 and release nothing")


def _check_stall(run: Any) -> LawResult:
    t, rel = _series(run, "peel_force"), _series(run, "released")
    tt = np.asarray(run.trace.t)
    late = tt > 0.1
    row = 2.5e-3 * 25e-3                          # one mesh row of the 25 mm strip
    more = float(rel[late].max() - rel[late].min())
    weight = 0.2 * 9.81                           # the z link's weight (rig.link_mass)
    want = STALL_EFFORT - weight
    imp = _series(run, "impulse")
    i0, i1 = int(np.argmax(tt > 0.2)), len(tt) - 1
    pull = float((imp[i1] - imp[i0]) / (tt[i1] - tt[i0]))      # time-averaged pull
    err = max(abs(pull - want) / want, more / row - 1.0)
    return LawResult.within(err, 0.1, f"stalled pull {pull:.3f} N vs drive - weight {want:.3f} N; released "
                            f"{more * 1e6:.1f} mm² after 0.1 s (one row {row * 1e6:.1f})",
                            "a robot weaker than the peel force must stall the front")


def _bonded_fraction(run: Any) -> float:
    st = np.asarray(run.trace.plugin(run.plugin.label, "bond_state"))[-1]
    return float(np.mean(st == 1))


def _check_press(run: Any, want: float) -> LawResult:
    f = _bonded_fraction(run)
    wet = np.asarray(run.trace.plugin(run.plugin.label, "wet"))[-1]
    dry = float(wet.max()) if want == 0.0 else 0.0                # no wet-out below p_form either (W-2)
    return LawResult.within(abs(f - want) + dry, 0.0, f"bonded fraction {f:.3f} (want {want:.0f}), max w "
                            f"{float(wet.max()):.4f}", "formation, wet-out: p >= p_form for the dwell, and only then")


def _check_null(run: Any) -> LawResult:
    f = _bonded_fraction(run)
    pull = float(np.abs(_series(run, "peel_force")).max())
    return LawResult.within(f + pull, 0.0, f"bonded fraction {f}, max pull {pull}", "no pressure, no bond")


def roll_force(rig: RollRig) -> float:
    """N: what ``rig``'s roller presses with (its weight on a free slide, else the servo's effort plus the link's)."""
    return rig.link_mass * 9.81 + (0.0 if rig.dead_weight else rig.effort)


def roll_wet(rig: RollRig, q: float, r_sub: float = np.inf) -> float:
    """The exact wet-out after one pass of ``rig``'s roller at ROLL_SPEED under line load ``q`` (wetout.py), on a
    parallel tube of radius ``r_sub`` (inf: flat)."""
    p_form = adhesive(STEEL).p_form
    return 1.0 - float(np.exp(-exposure(rig.card, q, p_form, ROLL_SPEED, r_sub=r_sub) / wet_tau(p_form)))


def _check_roll(run: Any, rig: RollRig) -> LawResult:
    wet = np.asarray(run.trace.plugin(run.plugin.label, "wet"))[-1]
    st = np.asarray(run.trace.plugin(run.plugin.label, "bond_state"))[-1]
    load = np.asarray(run.trace.plugin(run.plugin.label, "roller_q"), np.float64)[:, 0]
    q_want = roll_force(rig) / rig.card.width
    q = float(np.median(load[len(load) // 4: 3 * len(load) // 4]))      # mid-pass, past the landing
    r_sub = rig.tube_radius if isinstance(rig, TubeRig) else np.inf
    want, got = roll_wet(rig, q, r_sub), float(np.mean(wet))               # the Hertz pass at the load it reads
    value = max(abs(got / want - 1.0), abs(q / q_want - 1.0) / 2.0)        # w 2.5 %, the load 5 %
    if isinstance(rig, TubeRig):                                           # C-1: the whole label bonds; R_e matters
        value = max(value, 1.0 - float(np.mean(st == 1)))
        flat = roll_wet(rig, q)
        return LawResult.within(value, 0.025, f"load {q:.1f} vs {q_want:.1f} N/m; mean w {got:.4f} vs Hertz on the "
                                f"{r_sub * 1e3:g} mm tube {want:.4f} (flat would be {flat:.4f}), bonded "
                                f"{np.mean(st == 1):.3f}, {rig.dx * 1e3:.2f} mm cells",
                                "the roller's Hertz contact on a tube: 1/R_e = 1/R + 1/R_s (wetout.effective_radius)")
    return LawResult.within(value, 0.025, f"load {q:.1f} vs {q_want:.1f} N/m; mean w {got:.4f} vs Hertz {want:.4f}, "
                            f"bonded {np.mean(st == 1):.3f}, w {wet.min():.4f}..{wet.max():.4f}, "
                            f"{rig.dx * 1e3:.2f} mm cells",
                            "the roller's load is what the film carries and its pressure the Hertz contact (mesh-free)")


def standard_scene(rig: RollRig) -> TestScene:
    """One standard application: two passes, one each way, at STD_SPEED at the contact (a tube: the gantry faster)."""
    span = rig.length + 2.0 * rig.lead
    speed = STD_SPEED / rig.nip_ratio if isinstance(rig, TubeRig) else STD_SPEED
    t_app = 2.0 * span / speed
    return scene(rig, "roll", int(np.ceil(t_app / rig.control_dt)) + 5, t_turn=t_app, span=span, speed=speed,
                 lift=5.0e-3)


def _check_standard(run: Any, rig: RollRig) -> LawResult:
    from .layout import vertex_areas  # noqa: PLC0415

    wet = np.asarray(run.trace.plugin(run.plugin.label, "wet"))[-1, 0].astype(np.float64)
    st = np.asarray(run.trace.plugin(run.plugin.label, "bond_state"))[-1, 0]
    nx, ny = int(round(rig.length / rig.dx)), int(round(rig.width / rig.dx))
    v, t = grid_mesh(nx, ny, rig.dx)
    edge = np.minimum(np.minimum(v[:, 0], rig.length - v[:, 0]), np.minimum(v[:, 1], rig.width - v[:, 1]))
    ga = ga_wet_out(wet, vertex_areas(v, t), edge <= 3.0e-3 + 1e-9)
    worst = min(ga["label"], ga["edge_band"])
    value = max(0.0, 0.90 - worst) + (1.0 - float(np.mean(st == 1)))
    return LawResult.within(value, 0.0, f"GA wet-out: label {ga['label']:.3f}, edge band {ga['edge_band']:.3f} (>= "
                            f"0.90 of W_STD), w {wet.min():.3f}..{wet.max():.3f}, bonded {np.mean(st == 1):.3f}",
                            "a standard application must meet GA's wet-out on the label and its 3 mm edge band")


def _check_lifted(run: Any) -> LawResult:
    wet = np.asarray(run.trace.plugin(run.plugin.label, "wet"))
    load = np.asarray(run.trace.plugin(run.plugin.label, "roller_q"), np.float64)
    f = _bonded_fraction(run)
    return LawResult.within(f + float(wet.max()) + float(np.abs(load).max()), 0.0, f"bonded {f}, max w "
                            f"{float(wet.max())}, max load {float(np.abs(load).max())} N/m",
                            "a roller that does not touch the film presses nothing")


def roll_scene(rig: RollRig, lift: float = 5.0e-3) -> TestScene:
    """One pass over the patch at ROLL_SPEED; z at -``lift`` (a negative ``lift`` holds the roller above)."""
    span = rig.length + 2.0 * rig.lead
    speed = ROLL_SPEED / rig.nip_ratio if isinstance(rig, TubeRig) else ROLL_SPEED     # the contact sweeps at 20 mm/s
    t_pass = span / speed
    return scene(rig, "roll", int(np.ceil(t_pass / rig.control_dt)) + 3, t_turn=t_pass + 1.0, span=span,
                 speed=speed, lift=lift)


def press_effort(rig: PressRig, ratio: float) -> float:
    """Drive force [N] giving ``ratio`` x p_form on the patch: p = F/A (plugin docstring), less the pad's weight."""
    return ratio * adhesive(STEEL).p_form * rig.length * rig.width - rig.link_mass * 9.81


def bonds_laws(film_substeps: int = 1) -> list[Law]:
    """The laws; ``film_substeps``: run the film entry at that many substeps per exchange (the API's clock picks
    1..64 at runtime; SOFT-1 review (c))."""
    n = int(film_substeps)
    peel, fast = PeelRig(film_substeps=n), 0.1
    press = PressRig(film_substeps=n)
    hold, short = PRESS_DWELL_PERIODS * press.control_dt, 0.6 * adhesive(STEEL).dwell
    high = PressRig(effort=press_effort(press, 2.0), film_substeps=n)
    low = PressRig(effort=press_effort(press, 0.4), film_substeps=n)
    lift = {"lift": 5.0e-3}
    return [
        Law("peel_90", _check_90, scene=scene(peel, "follow", 70), config=PEEL, doc="90° vs 7818 + energy"),
        Law("peel_180", _check_180, scene=scene(peel, "pull180", 70, speed=fast), config=PEEL,
            doc="180° vs 7818 + energy"),
        Law("slack", _check_slack, scene=scene(peel, "slack", 40, t_turn=0.2), config=PEEL, doc="slack arm: 0"),
        Law("stall", _check_stall, scene=scene(PeelRig(effort=STALL_EFFORT, film_substeps=n), "follow", 40),
            config=PEEL,
            doc="weak robot stalls the front"),
        Law("forms_above_p_form", lambda r: _check_press(r, 1.0), config=PRESS,
            scene=scene(high, "press", PRESS_DWELL_PERIODS + 10, t_turn=hold, **lift), doc="2x p_form, 2x dwell"),
        Law("none_below_p_form", lambda r: _check_press(r, 0.0), config=PRESS,
            scene=scene(low, "press", PRESS_DWELL_PERIODS + 10, t_turn=hold, **lift), doc="0.4x p_form"),
        Law("none_below_dwell", lambda r: _check_press(r, 0.0), config=PRESS,
            scene=scene(high, "press", 12, t_turn=short, **lift), doc="2x p_form, 0.6x dwell"),
        Law("null", _check_null, config=PRESS, scene=scene(press, "press", 10, t_turn=0.0, **lift),
            doc="untouched patch: nothing"),
    ] + roll_laws(n) + _air_laws()


def _air_laws() -> list[Law]:
    from .laws_air import air_laws  # noqa: PLC0415 - laws_air imports this module

    return air_laws()


def roll_laws(n: int = 1) -> list[Law]:
    """The wet-out laws on the roller rig (module docstring)."""
    dead = [RollRig(dx=dx, film_substeps=n, dead_weight=True, link_mass=2.04, **ROLL_RIG) for dx in (2.5e-3, 1.25e-3)]
    servo = RollRig(dx=2.5e-3, film_substeps=n, effort=20.0, **ROLL_RIG)
    return [Law(f"W-3_roll_dx{rig.dx * 1e3:g}", lambda r, rig=rig: _check_roll(r, rig), config=ROLL,
                scene=roll_scene(rig), doc="dead-weight roller: load m·g/width, Hertz wet-out, any cell size")
            for rig in dead] + [
        Law("W-4_roll_servo", lambda r: _check_roll(r, servo), config=ROLL, scene=roll_scene(servo),
            doc="servo roller: load = its force, Hertz wet-out"),
        Law("W-5_roll_lifted", _check_lifted, config=ROLL, scene=roll_scene(servo, lift=-1.0e-3),
            doc="a roller 1 mm above: nothing")] + [
        Law(f"C-1_tube_dx{rig.dx * 1e3:g}", lambda r, rig=rig: _check_roll(r, rig), config=TUBE,
            scene=roll_scene(rig), doc="a label rolled onto a 25 mm tube bonds whole at the tube's Hertz contact")
        for rig in (TubeRig(dx=dx, film_substeps=n, dead_weight=True, link_mass=2.04, **ROLL_RIG)
                    for dx in (2.5e-3, 1.25e-3))] + [
        Law(f"GA_standard_{name}", lambda r, rig=rig: _check_standard(r, rig), config=cfg, scene=standard_scene(rig),
            doc="a standard application meets GA's wet-out on the label and its edge band")
        for name, rig, cfg in (("flat", RollRig(dx=2.5e-3, film_substeps=n, dead_weight=True, link_mass=2.04,
                                                **ROLL_RIG), ROLL),
                               ("tube", TubeRig(dx=2.5e-3, film_substeps=n, dead_weight=True, link_mass=2.04,
                                                **ROLL_RIG), TUBE))]


__all__ = ["PEEL", "PRESS", "ROLL", "ROLL_RIG", "TUBE", "bonds_laws", "press_effort", "roll_force", "roll_laws",
           "roll_scene", "roll_wet", "scene", "standard_scene"]
