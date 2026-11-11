# SPDX-License-Identifier: Apache-2.0
"""task_label_apply: a tamp-on labeller carries a 40 x 25 mm PET label (3M 7818) under its pad, lays it on a carton
top and presses it on. This is Gym-Anything's label tasks: carton top and press-down (REPORT.md §2); the carton is a
rigid box in v1.

**Script, per world (MuJoCo gantry, pad force-limited at ``PRESS``):**
1. The label starts held under the pad 20 mm above the carton and 60 mm to the side of it. The pad's vacuum is a
   ``Bonds`` grip.
2. The gantry carries it over the target.
3. It lowers until the label is 0.3 mm above the carton.
4. The vacuum lets go (``release_grip``).
5. The pad presses the label at twice the 7818 formation pressure for 3x the dwell, then lifts.
Bonds form where the contact pressure exceeds p_form for the dwell (``alloy.plugins.bonds``; p_form and the dwell
are ESTIMATES).

**Success, per world:**
- at least 95 % of the label's vertices bonded, and all four corners bonded;
- the label's centre within 3 mm of the target.

**GA's label criteria, reported beside success (SCR-WP8-11 Part B):** wet-out at least 90 % of a standard application
over the label and its 3 mm edge band (``wetout.ga_wet_out``); no enclosed pocket holding air larger than 20 mm²
(``Bonds.Config.trapped_air``, at the task's 2.5 mm cells); and the barcode's 3 mm² criterion, not scored at these
cells (``air.BARCODE_NOT_SCORED``, docs/known_gaps/label-barcode-trapped-air.md).

``python -m alloy.tasks.label_apply --worlds 4`` prints one JSON row.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from alloy._vendor import newton

from alloy.plugins.bonds import Bonds, adhesive
from alloy.plugins.bonds import rig as R
from alloy.plugins.bonds.air import BARCODE_NOT_SCORED, ga_trapped_air
from alloy.plugins.bonds.layout import vertex_areas
from alloy.plugins.bonds.wetout import ga_wet_out
from alloy.plugins.harness.scene import body_xform
from alloy.solvers.film.emit import add_film_mesh
from alloy.solvers.film.rigs import grid_mesh

CARD = "3m_7818/polypropylene"          # a printed PP-coated carton face; the sheet's PP row
LABEL = (0.040, 0.025)                  # m
CARRY_X, LIFT = 0.060, 0.020            # m: start offset from the target, height above the carton
CLEAR = 0.3e-3                          # m: label above the carton when the vacuum lets go
PAD_HALF = (0.0215, 0.0140, 0.002)      # m, a little larger than the label
PRESS_RATIO, HOLD = 2.0, 3.0            # x p_form, x dwell
PHASES = (0.30, 0.45, 0.47, 0.47 + 0.15, 0.80)   # s: carry, lower, release, press end, lift end
POS_TOL, BONDED_MIN = 3.0e-3, 0.95
EDGE_BAND = 3.0e-3                      # m: GA's edge band for wet-out
RIG = replace(R.PressRig(), length=LABEL[0], width=LABEL[1], pad_half=PAD_HALF, link_mass=2.0, dt=5.0e-4)


def press_force() -> float:
    """Drive force [N]: PRESS_RATIO x p_form over the label, less the pad link's weight."""
    return PRESS_RATIO * adhesive(CARD).p_form * LABEL[0] * LABEL[1] - RIG.link_mass * 9.81


def build_world(b: newton.ModelBuilder, rig: R.PressRig = RIG) -> None:
    """Carton (static, top at z = 0, centred on the target x = 0) and the label under the pad, CARRY_X to the -x side
    and LIFT up."""
    b.add_shape_box(-1, xform=body_xform(0.0, 0.0, -0.05), hx=0.06, hy=0.04, hz=0.05,
                    cfg=newton.ModelBuilder.ShapeConfig(mu=0.5, ke=R.CONTACT_KE), label="carton")
    nx, ny = int(round(LABEL[0] / rig.dx)), int(round(LABEL[1] / rig.dx))
    v, t = grid_mesh(nx, ny, rig.dx)
    x0 = -CARRY_X - 0.5 * LABEL[0]
    add_film_mesh(b, v, t, rig.film, pos=(x0, -0.5 * LABEL[1], LIFT + rig.z_film), label="label")
    pad = replace(rig, effort=press_force())
    R._gantry(b, pad, -CARRY_X, LIFT + rig.pad_z, PAD_HALF, "pad", True)


def model(worlds: int, device: str) -> newton.Model:
    from alloy.solvers.film.config import register_custom_attributes  # noqa: PLC0415

    tmpl, root = newton.ModelBuilder(), newton.ModelBuilder()
    for b in (tmpl, root):
        newton.solvers.SolverMuJoCo.register_custom_attributes(b)
        register_custom_attributes(b)
    build_world(tmpl)
    root.replicate(tmpl, worlds)
    root.color()
    m = root.finalize(device=device)
    m.soft_contact_ke = R.CONTACT_KE
    return m


def targets(t: float) -> tuple[float, float, float]:
    """Gantry (x, z) targets relative to the start, and release_grip (0/1), at time t."""
    blend = lambda a, b, s: a + (b - a) * 0.5 * (1.0 - math.cos(math.pi * min(max(s, 0.0), 1.0)))  # noqa: E731
    x = blend(0.0, CARRY_X, t / PHASES[0])
    z_rest = -(LIFT - CLEAR)
    z = blend(0.0, z_rest, (t - PHASES[0]) / (PHASES[1] - PHASES[0]))
    if PHASES[2] <= t < PHASES[3]:
        z = -LIFT - 0.015                                   # below the carton top: the force limit presses
    elif t >= PHASES[3]:
        z = blend(z_rest, 0.0, (t - PHASES[3]) / (PHASES[4] - PHASES[3]))
    return x, z, float(t >= PHASES[2])


def run(worlds: int, device: str, graph: bool = False) -> dict:
    from alloy.plugins.host import simulation  # noqa: PLC0415
    from alloy.scene.builtin_plugins import register_builtin_solvers  # noqa: PLC0415

    register_builtin_solvers()
    bonds = Bonds(substrates=("carton",), adhesives=(CARD,), grip="pad", grip_reach=1.5e-3, prebond=False,
                  trapped_air=True)
    m = model(worlds, device)
    kw = {} if not graph else _graph_kwargs()
    sim = simulation(R.built(m, RIG), [bonds], **kw)
    cols = R.gantry_columns(sim)
    for k in range(int(round(PHASES[4] / RIG.control_dt)) + 5):
        x, z, rel = targets((k + 1) * RIG.control_dt)
        R.set_gantry(sim, cols, x, z)
        bonds.set_input("release_grip", int(rel))
        sim.step_period()
    read = bonds.read()
    row = _score(read["bond_state"], sim.state.particle_q.numpy().reshape(worlds, -1, 3))
    return {"task": "task_label_apply", "card": CARD, "worlds": worlds, "device": device, "graph": graph,
            "press_n": press_force(), **row, **ga_criteria(read)}


def ga_criteria(read: dict[str, np.ndarray]) -> dict[str, Any]:
    """GA's label criteria per world from the bonds' readings: wet-out (label and edge band) and trapped air."""
    nx, ny = int(round(LABEL[0] / RIG.dx)), int(round(LABEL[1] / RIG.dx))
    v, t = grid_mesh(nx, ny, RIG.dx)
    area = vertex_areas(v, t)
    edge = np.minimum(np.minimum(v[:, 0], LABEL[0] - v[:, 0]), np.minimum(v[:, 1], LABEL[1] - v[:, 1]))
    band = edge <= EDGE_BAND
    wet = [ga_wet_out(w, area, band) for w in read["wet"]]
    trapped = read["trapped_area"]
    return {"ga_wet_out_pass": int(sum(w["pass"] for w in wet)), "ga_wet_out_min": min(w["label"] for w in wet),
            "ga_trapped_air_pass": int(ga_trapped_air(trapped).sum()),
            "trapped_area_mm2_max": float(trapped.max() * 1e6), "air_overflow_max": float(read["air_overflow"].max()),
            "air_vanished_max": float(read["air_vanished"].max()),       # Pa·m³ taken up at bonding (roughness air)
            "barcode": BARCODE_NOT_SCORED}


def _graph_kwargs() -> dict[str, Any]:
    from alloy.integration.simulation import IntegratedSimulation  # noqa: PLC0415

    return {"cls": IntegratedSimulation, "use_graph": True, "cpu_graph": True}


def _score(state: np.ndarray, q: np.ndarray) -> dict:
    """Per world: bonded fraction, corners bonded, the label centre's error from the target (0, 0)."""
    nx, ny = int(round(LABEL[0] / RIG.dx)) + 1, int(round(LABEL[1] / RIG.dx)) + 1
    corners = [0, ny - 1, (nx - 1) * ny, nx * ny - 1]
    frac = np.mean(state == 1, axis=1)
    corner_ok = np.all(state[:, corners] == 1, axis=1)
    err = np.hypot(q[:, :, 0].mean(axis=1), q[:, :, 1].mean(axis=1))
    ok = (frac >= BONDED_MIN) & corner_ok & (err <= POS_TOL)
    return {"success": int(ok.sum()), "bonded_min": float(frac.min()), "corners_ok": int(corner_ok.sum()),
            "center_err_mm_max": float(err.max() * 1e3)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--worlds", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    row = run(a.worlds, a.device, a.graph)
    print(json.dumps(row))
    if a.out is not None:
        with a.out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
    return 0 if row["success"] == a.worlds else 1


if __name__ == "__main__":
    raise SystemExit(main())
