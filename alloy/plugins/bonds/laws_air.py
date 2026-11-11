# SPDX-License-Identifier: Apache-2.0
"""The trapped-air laws (air.py) on the pocket rig (``rig.AirRig``): a label pre-bonded on a plate around a recess,
whose floor is a piston on the gantry's z axis (scripted by ``driver.py``'s ``pull180`` mode at speed 0: a ramp up by
``lift`` over ``t_lift``, then held). No gravity; the label is 60 µm BOPP at 2 mm cells.

- **T-1_boyle.** A lid holds the label flat over a closed 14 x 12 mm recess, 0.5 mm deep. The piston rises 0.25 mm
  in 50 ms at the label task's 0.5 ms exchange step: the pocket's pressure follows Boyle against the piston's
  measured travel d, P·(depth + 1 µm roughness - d) constant within 1 %. The pocket stays one enclosed pocket, its
  books exact.
- **T-1_free_dt<step>.** No lid: the label alone holds a 10 % squeeze, at 0.5 and 1 ms. The gas spring is stiff
  (P·A²/V ~ 3e4 N/m on 0.2 mg vertices); a lagged coupling would ring or blow up. The pocket's volume settles: its
  swing over the last 60 ms is under 1e-4 of its volume, and its pressure stays within 1 % of the start.
- **T-2_vent.** The recess (3 µm deep) runs out past the label's end, so the pocket vents through the label's edge.
  The piston rises 1 µm in 50 ms, then holds (2 ms periods: the vent's time constant is about 11 ms). Over the hold
  the pocket's excess pressure follows the slot ODE dP/dt = -g·(P - P_atm)/V - P·(dV/dt)/V,
  g = (P + P_atm)/2·sum w·h³/(12·mu·s) (``pockets.throat_coefficients``; h measured on the label's edge vertices
  above the piston, plus the 1 µm roughness layer; the piston's settling compression included), within 5 %, while
  20 % of the excess is left (below that, nanometre film motion swamps it).
- **T-4 (an application can fail it; WP13 review of B2).** ``edge_application``: a 20 mm label floats 0.3 mm above
  the plate; a frame pad presses its 3 mm edge band, then a centre pad presses the rest (``rig.EdgeRig``). The band
  bonds first and encloses 197 mm² of air: the centre pad only compresses it (to 248 kPa), it springs back, and GA's
  20 mm² criterion fails. Both pads together bond the whole label, 0 mm². tests/plugins/test_bonds_air.py.
- **T-3 (books)** holds in every run above, period by period: slots + vented + vanished + untracked - formed = 0 to
  1e-12 of the formed air (``air.books_error``). Relabels (splits, merges, vanishing pockets, overflow) are in
  tests/plugins/test_bonds_air.py.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from alloy.plugins.harness.scene import Law, LawResult

from .air import AIR_ROUGHNESS, AIR_VISCOSITY, books_error
from .air_kernels import P_ATM
from .laws import STEEL, scene
from .layout import vertex_spacing
from .pockets import throat_coefficients
from .rig import AirRig

PA = float(P_ATM)
BOYLE = AirRig(dt=5.0e-4)                                  # the label task's exchange step
FREE = replace(BOYLE, lid=False)
VENT = AirRig(open_end=True, depth=3.0e-6, dt=5.0e-4, control_dt=2.0e-3)      # real air throughout (held = 1)
T_RAMP = 0.05                                              # s
READ = ("air_amount", "air_root", "air_volume", "air_enclosed", "air_formed", "air_vented", "air_vanished",
        "air_untracked", "pockets", "air_overflow", "air_unsettled", "bond_state")


def air_config(rig: AirRig) -> dict[str, Any]:
    """Bonds config of the pocket rig: every plate box and the piston are substrates; pre-bonds only on the plate's
    top (the reach stays under the recess depth); no formation."""
    subs = ("plate", "frame_s", "frame_n", "piston") + (() if rig.open_end else ("frame_e",))
    return {"substrates": subs, "adhesives": (STEEL,) * len(subs), "grip": "", "prebond": True, "form": False,
            "prebond_reach": min(1.0e-4, 0.3 * rig.depth), "trapped_air": True}


def air_scene(rig: AirRig, stroke: float, periods: int) -> Any:
    return scene(rig, "pull180", periods, speed=0.0, lift=stroke, t_lift=T_RAMP)


def _reads(run: Any) -> dict[str, np.ndarray]:
    return {k: np.asarray(run.trace.plugin(run.plugin.label, k)) for k in READ}


def _pocket(run: Any, d: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, str]:
    """World 0's one pocket per period after the first: (P, V), or a problem."""
    root = d["air_root"][1:, 0]
    if np.any((root >= 0).sum(axis=1) != 1) or np.any(d["pockets"][1:, 0] != 1):
        return np.zeros(0), np.zeros(0), f"pockets per period {d['pockets'][1:, 0].tolist()} (want 1)"
    k = np.argmax(root >= 0, axis=1)
    rows = np.arange(len(k))
    a, v = d["air_amount"][1:, 0][rows, k], d["air_volume"][1:, 0][rows, k]
    books = float(np.abs(books_error({n: d[n][:, 0] for n in d})).max())
    flags = float(d["air_overflow"][:, 0].max() + d["air_unsettled"][:, 0].max())
    if books > 1e-12 or flags > 0:
        return np.zeros(0), np.zeros(0), f"books off by {books:.2e}, overflow/unsettled {flags:g}"
    return a / v, v, ""


def _piston(run: Any) -> np.ndarray:
    """The piston's rise since the start [m], per period (world 0)."""
    z = np.asarray(run.trace.body_q)[:, int(run.bodies("gantry_z")[0]), 2]
    return z - z[0]


def _check_boyle(run: Any) -> LawResult:
    d = _reads(run)
    p, _, why = _pocket(run, d)
    if why:
        return LawResult(False, float("inf"), 0.01, why, "the pocket must stay one enclosed pocket with exact books")
    held = BOYLE.depth + AIR_ROUGHNESS - _piston(run)[1:]
    pv = p * held
    err = float(np.abs(pv / pv[0] - 1.0).max())
    enclosed = bool(np.all(d["air_enclosed"][1:, 0].max(axis=1) == 1))
    return LawResult.within(err if enclosed else float("inf"), 0.01,
                            f"P·(depth + roughness - rise) within {err:.2e}; P {p[0] / 1e3:.1f} -> {p[-1] / 1e3:.1f} "
                            f"kPa at rise {_piston(run)[-1] * 1e3:.3f} mm; enclosed {enclosed}",
                            "the pocket's volume must follow the piston (gap kernel) and keep its air")


def _check_free(run: Any) -> LawResult:
    d = _reads(run)
    p, v, why = _pocket(run, d)
    if why:
        return LawResult(False, float("inf"), 1e-4, why, "the pocket must stay one enclosed pocket with exact books")
    tail = v[-6:]
    swing = float(np.ptp(tail) / tail.mean())
    rise = float(np.abs(p / p[0] - 1.0).max())
    return LawResult.within(max(swing, rise * 1e-2), 1e-4, f"volume swing over the last 60 ms {swing:.2e} of V; "
                            f"pressure within {rise:.2%} of the start", "the implicit pocket pressure (pocket_force)")


def _throat(run: Any, rig: AirRig, k: int) -> float:
    """sum w·h³/(12·mu·s) over the label's free edge vertices, h from their height above the piston (host)."""
    m = run.model
    x0 = m.particle_q.numpy().astype(np.float64)
    w0 = m.particle_world.numpy() == 0
    tris = m.tri_indices.numpy()
    tris = tris[np.all(w0[tris], axis=1)]
    coef = throat_coefficients(x0, tris, vertex_spacing(x0, tris), AIR_VISCOSITY)
    q = np.asarray(run.trace.particle_q)[k]
    free = np.asarray(run.trace.plugin(run.plugin.label, "bond_state"))[k, 0] == 0
    top = -rig.depth + _piston(run)[k]
    lo, hi = rig.recess_x
    edge = w0 & (coef > 0) & (q[:, 0] > lo) & (q[:, 0] < hi)
    edge[np.flatnonzero(w0)] &= free
    h = np.maximum(q[:, 2] - rig.z_film - top, 0.0) + AIR_ROUGHNESS
    return float(np.sum(coef[edge] * h[edge] ** 3))


def _vent_ode(p0: float, v: np.ndarray, lam: np.ndarray, dt: float) -> np.ndarray:
    """RK4 of dP/dt = -lam·(P + P_atm)/2·(P - P_atm)/V - P·(dV/dt)/V from p0, V linear and lam fixed over each
    period (the piston settles during the hold, so the pocket still compresses a little)."""
    out, p = [p0], p0
    for v0, v1, ll in zip(v[:-1], v[1:], lam[1:]):
        rate = (v1 - v0) / dt
        f = lambda pp, tt: (-ll * 0.5 * (pp + PA) * (pp - PA) - pp * rate) / (v0 + rate * tt)  # noqa: E731
        h = dt / 100.0
        for k in range(100):
            tt = k * h
            k1 = f(p, tt)
            k2 = f(p + 0.5 * h * k1, tt + 0.5 * h)
            k3 = f(p + 0.5 * h * k2, tt + 0.5 * h)
            k4 = f(p + h * k3, tt + h)
            p = p + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(p)
    return np.asarray(out)


def _check_vent(run: Any) -> LawResult:
    d = _reads(run)
    p, v, why = _pocket(run, d)
    if why:
        return LawResult(False, float("inf"), 0.05, why, "the pocket must stay one pocket with exact books")
    t = np.asarray(run.trace.t)[1:]
    hold = np.flatnonzero(t >= T_RAMP + 0.01)
    hold = hold[: int(np.argmax(p[hold] - PA < 0.2 * (p[hold[0]] - PA)))]   # while 20 % of the excess is left
    lam = np.array([_throat(run, VENT, int(i) + 1) for i in hold])
    ode = _vent_ode(p[hold[0]], v[hold], lam, VENT.control_dt)
    err = float(np.abs((p[hold] - PA) / (ode - PA) - 1.0).max())
    decay = (p[hold[-1]] - PA) / (p[hold[0]] - PA)
    open_ = bool(np.all(d["air_enclosed"][1:, 0].max(axis=1) == 0))
    return LawResult.within(err if open_ else float("inf"), 0.05,
                            f"excess pressure vs the slot ODE within {err:.2%} over {len(hold)} periods (decayed to "
                            f"{decay:.2f}); throat g/P_mean {lam.mean():.3g} m³/(Pa·s); open {open_}",
                            "the vent: throat coefficients, the gap at the edge, the core's gas step")


def edge_application(together: bool, worlds: int = 1) -> dict[str, Any]:
    """T-4: the edges-first application (or both pads together) on ``rig.EdgeRig``; per period the pocket readings
    (trapped area, bonded area, the largest pocket pressure, the vanished air) and at the end the GA verdict."""
    from .adhesives import adhesive  # noqa: PLC0415
    from .air import ga_trapped_air  # noqa: PLC0415
    from .plugin import Bonds  # noqa: PLC0415
    from .rig import EdgeRig, simulation  # noqa: PLC0415

    rig = EdgeRig(p_form=adhesive(STEEL).p_form)
    bonds = Bonds(substrates=("plate",), adhesives=(STEEL,), grip="", prebond=False, trapped_air=True)
    sim = simulation(rig, [bonds], worlds=worlds)
    starts = sim.model.joint_q_start.numpy()
    col = {lab.split("/")[-1]: int(starts[j]) for j, lab in enumerate(sim.model.joint_label)}
    down, up = -(rig.lift0 + 6.0e-3), 2.0e-3
    rows = []
    for dur, zf, zp in ((0.25, down, down if together else 0.0), (0.25, down, down), (0.15, up, up)):
        for _ in range(int(round(dur / rig.control_dt))):
            tgt = sim.control.joint_target_q.numpy()
            tgt[col["frame_x"]] = tgt[col["pad_x"]] = 0.0
            tgt[col["frame_z"]], tgt[col["pad_z"]] = zf, zp
            sim.control.joint_target_q.assign(tgt.astype(np.float32))
            sim.step_period()
            rows.append(_edge_row(bonds.read()))
    return {"periods": rows, "trapped": rows[-1]["trapped"], "ga_pass": ga_trapped_air(rows[-1]["trapped"])}


def _edge_row(d: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    act = d["air_root"] >= 0
    p = np.where(act, d["air_amount"] / np.where(act, d["air_volume"], 1.0), 0.0).max(axis=1)
    return {"trapped": d["trapped_area"].copy(), "state": d["bond_state"].copy(), "p_max": p,
            "vanished": d["air_vanished"].copy(), "books": books_error(d)}


def air_laws() -> list[Law]:
    """T-1 and T-2 (module docstring); T-3's books are checked in each."""
    free = [replace(FREE, dt=dt) for dt in (5.0e-4, 1.0e-3)]
    return [Law("T-1_boyle", _check_boyle, config=air_config(BOYLE), scene=air_scene(BOYLE, 2.5e-4, 12),
                doc="lid-held label over a closed pocket: Boyle against the piston, 1 %")] + [
        Law(f"T-1_free_dt{rig.dt * 1e3:g}", _check_free, config=air_config(rig),
            scene=air_scene(rig, 0.1 * rig.depth, 16), doc="free label over a squeezed pocket: settles, no ringing")
        for rig in free] + [
        Law("T-2_vent", _check_vent, config=air_config(VENT), scene=air_scene(VENT, 1.0e-6, 50),
            doc="open pocket: vents on the laminar slot ODE of its edge throat, 5 %")]


__all__ = ["air_config", "air_laws", "air_scene", "edge_application"]
