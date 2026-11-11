# SPDX-License-Identifier: Apache-2.0
"""Yield-stress paste laws on the MPM paste route (docs/notes/viscoplastic-design.md §5), through the public API.

Each law writes a throw-away material and object card into ``root`` (``alloy.Scene(roots=[root])``), builds one
small scene, runs it and returns a JSON-ready row with the measured value, the reference and the inputs:

- ``spread``: L0, a Newtonian block released on the ground vs Huppert (1982), R = 0.894 (g V^3 t / 3 nu)^(1/8);
- ``slump``: L1, a cylinder H0 = 2 R0 of a Bingham paste vs Pashias et al. (1996), s' = 1 - 2 t'(1 - ln 2 t');
- ``pancake``: L2, the final radius of a thin deposit vs Roussel and Coussot (2005), R = (225 rho g V^2 /
  128 pi^2 tau_y)^(1/5);
- ``slope``: L3, a layer on a plane tilted theta: it flows above h_stop = tau_y / (rho g sin theta), stops below;
- ``film``: L4 / L5, the steady surface speed of a Bingham or Herschel-Bulkley film on an incline vs the analytic
  u_s = n / (n + 1) (rho g sin theta / K)^(1/n) (h - h_p)^(1 + 1/n), h_p = tau_y / (rho g sin theta).

The paste's MPM voxel is set through the card's ``features.passage_width`` (voxel = passage / 4, at most 5 mm). MPM
colliders meet the paste with their own Coulomb friction (the shape's mu), so every law stands on a rough surface
(``ROUGH`` = 1.0): a real paste sticks to its plate, while the default plywood ground (0.4) lets a base slide.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger("alloy.solvers.mpm_rheology")

G = 9.81
ROUGH = 1.0                 # Coulomb friction of the law's ground or board: the paste does not slide on it


@dataclass(frozen=True)
class Paste:
    """A law material: density (kg/m^3), yield stress (Pa), viscosity (Pa s; the Bingham plastic viscosity with a
    yield stress) and, for Herschel-Bulkley, consistency K (Pa s^n) and flow index n (both 0 = none)."""

    rho: float = 1100.0
    tau_y: float = 0.0
    viscosity: float = 10.0
    consistency: float = 0.0
    flow_index: float = 0.0


def _q(value: float, note: str) -> str:
    return f"{{value: {value!r}, source: model, note: '{note}'}}"


def write_cards(root: Path, name: str, paste: Paste, shape_yaml: str, voxel: float, budget: int | None = None) -> str:
    """Material ``<name>_mat`` and object ``<name>`` (``shape_yaml``: the object's shape block) under ``root``;
    ``budget`` raises the object's particle budget (default 200,000) for fine-voxel runs."""
    (root / "materials").mkdir(parents=True, exist_ok=True)
    (root / "objects").mkdir(parents=True, exist_ok=True)
    plastic = f"plastic:\n  yield_stress: {_q(paste.tau_y, 'law input')}\n" if paste.tau_y > 0 else ""
    hb = (f"  consistency: {_q(paste.consistency, 'law input')}\n  flow_index: {_q(paste.flow_index, 'law input')}\n"
          if paste.consistency > 0 else "")
    (root / "materials" / f"{name}_mat.yaml").write_text(
        f"schema: alloy.material/1\nname: {name}_mat\ndensity: {_q(paste.rho, 'law input')}\n"
        f"surface:\n  mu_static: {_q(0.0, 'paste')}\n  mu_dynamic: {_q(0.0, 'paste')}\n"
        f"  restitution: {_q(0.0, 'paste')}\nfluid:\n  viscosity: {_q(paste.viscosity, 'law input')}\n{hb}{plastic}",
        encoding="utf-8")
    (root / "objects" / f"{name}.yaml").write_text(
        f"schema: alloy.object/1\nname: {name}\nshape:\n{shape_yaml}material: {name}_mat\n"
        f"features:\n  passage_width: {_q(4.0 * voxel, 'law voxel x 4')}\n"
        + (f"budget:\n  particles: {int(budget)}\n" if budget else ""), encoding="utf-8")
    return name


def block_shape(size: tuple[float, float, float]) -> str:
    """A particle block with its base at z = 0, centred in x and y."""
    hx, hy = 0.5 * size[0], 0.5 * size[1]
    return (f"  particles:\n    region:\n    - [{-hx!r}, {-hy!r}, 0.0]\n    - [{hx!r}, {hy!r}, {size[2]!r}]\n"
            f"    spacing: 0.002\n")


def cylinder_shape(radius: float, height: float) -> str:
    """A cylinder primitive centred on its own origin (place it at z = height / 2)."""
    return f"  primitive:\n    kind: cylinder\n    size: [{radius!r}, {0.5 * height!r}]\n"


def run_points(root: Path, name: str, seconds: float, pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
               every: int = 5, device: str | None = None, extra=None,
               gravity: tuple[float, float, float] | None = None,
               profile: str = "standard") -> tuple[np.ndarray, list[np.ndarray], bool]:
    """Build a rough ground + the paste object (+ ``extra(scene)``) with ``profile``, optionally set every world's
    ``gravity`` vector (m/s^2) on the engine model, run ``seconds``; (t, frames, check ok)."""
    import alloy  # noqa: PLC0415 - the public API, imported on use

    scene = alloy.Scene(roots=[str(root)])
    scene.add_ground(friction=ROUGH)
    if extra is not None:
        extra(scene)
    scene.add_object(name, name, pos=pos)
    sim = alloy.build(scene, device=device, profile=profile)
    if gravity is not None:
        g = sim.engine.model.gravity
        g.assign(np.tile(np.asarray(gravity, np.float32), (g.shape[0], 1)))
    rec = sim.run(seconds, record=[name], every=every).world(0)
    pts = [np.asarray(p, np.float64) for p in rec[f"{name}.points"]]
    return np.asarray(rec["t"], np.float64).ravel(), pts, bool(sim.check()["ok"])


def radius95(p: np.ndarray, q: float = 95.0) -> float:
    """``q``-th percentile of the horizontal distance from the deposit's centre (m); 99.5 reads the edge."""
    c = p[:, :2].mean(axis=0)
    return float(np.percentile(np.hypot(p[:, 0] - c[0], p[:, 1] - c[1]), q))


def huppert_radius(volume: float, nu: float, t: float) -> float:
    """Axisymmetric viscous gravity current of constant volume (Huppert 1982)."""
    return 0.894 * (G * volume ** 3 * t / (3.0 * nu)) ** 0.125


def pashias_slump(tau_prime: float) -> float:
    """Dimensionless cylinder slump s' (Pashias et al. 1996, Eq. 11), tau' = tau_y / (rho g H0) < 0.5."""
    return 1.0 - 2.0 * tau_prime * (1.0 - math.log(2.0 * tau_prime))


def pancake_radius(volume: float, rho: float, tau_y: float) -> float:
    """Final radius of a thin yield-stress deposit (Roussel and Coussot 2005)."""
    return (225.0 * rho * G * volume ** 2 / (128.0 * math.pi ** 2 * tau_y)) ** 0.2


def spread(root: Path, paste: Paste, size=(0.04, 0.04, 0.03), seconds: float = 1.0, voxel: float = 0.005,
           device: str | None = None) -> dict:
    """L0: r95 / Huppert R at three times of a Newtonian block (tau_y 0) released on the ground."""
    t0 = time.time()
    name = write_cards(root, "law_spread", Paste(paste.rho, 0.0, paste.viscosity), block_shape(size), voxel)
    t, pts, ok = run_points(root, name, seconds, device=device)
    vol, nu = float(np.prod(size)), paste.viscosity / paste.rho
    rows = []
    for k in (len(t) // 4, len(t) // 2, len(t) - 1):
        ref = huppert_radius(vol, nu, float(t[k]))
        rows.append({"t": round(float(t[k]), 3), "r95_mm": round(radius95(pts[k]) * 1e3, 2),
                     "huppert_mm": round(ref * 1e3, 2), "ratio": round(radius95(pts[k]) / ref, 3)})
    return {"law": "L0 spread", "rows": rows, "check_ok": ok, "particles": len(pts[-1]),
            "wall_s": round(time.time() - t0, 1)}


def slump(root: Path, paste: Paste, height: float = 0.04, seconds: float = 1.5, voxel: float = 0.0025,
          budget: int | None = None, device: str | None = None) -> dict:
    """L1: the cylinder's final slump s' = (H0 - H_final) / H0 vs Pashias; H_final = the 99th-percentile height."""
    t0 = time.time()
    name = write_cards(root, "law_slump", paste, cylinder_shape(0.5 * height, height), voxel, budget)
    t, pts, ok = run_points(root, name, seconds, pos=(0.0, 0.0, 0.5 * height), device=device)
    tau_prime = paste.tau_y / (paste.rho * G * height)
    h_end = float(np.percentile(pts[-1][:, 2], 99))
    h_mid = float(np.percentile(pts[(3 * len(pts)) // 4][:, 2], 99))
    s = (height - h_end) / height
    return {"law": "L1 slump", "tau_prime": round(tau_prime, 4), "slump": round(s, 4),
            "pashias": round(pashias_slump(tau_prime), 4), "still_mm": round((h_mid - h_end) * 1e3, 3),
            "check_ok": ok, "particles": len(pts[-1]), "wall_s": round(time.time() - t0, 1)}


def pancake(root: Path, paste: Paste, volume: float = 50e-6, seconds: float = 2.0, voxel: float = 0.0015,
            budget: int | None = None, device: str | None = None) -> dict:
    """L2: a cylinder of ``volume`` (H0 = R0) released on the ground; final r95 vs Roussel and Coussot."""
    t0 = time.time()
    r0 = (volume / math.pi) ** (1.0 / 3.0)
    name = write_cards(root, "law_pancake", paste, cylinder_shape(r0, r0), voxel, budget)
    t, pts, ok = run_points(root, name, seconds, pos=(0.0, 0.0, 0.5 * r0), device=device)
    ref = pancake_radius(volume, paste.rho, paste.tau_y)
    r_end, r_mid = radius95(pts[-1], 99.5), radius95(pts[(3 * len(pts)) // 4], 99.5)
    return {"law": "L2 pancake", "r_edge_mm": round(r_end * 1e3, 2), "r95_mm": round(radius95(pts[-1]) * 1e3, 2),
            "ref_mm": round(ref * 1e3, 2), "ratio": round(r_end / ref, 3), "still_mm": round((r_end - r_mid) * 1e3, 3),
            "check_ok": ok, "particles": len(pts[-1]), "wall_s": round(time.time() - t0, 1)}


def slope(root: Path, paste: Paste, thickness: float, theta_deg: float = 30.0, length: float = 0.2,
          width: float = 0.03, seconds: float = 1.0, voxel: float = 0.005, device: str | None = None) -> dict:
    """L3: a layer ``thickness`` thick on the rough ground with gravity tilted ``theta_deg`` towards +x (an inclined
    plane without rotating the paste: particle objects are placed unrotated today, SCR-SOFT-2-4). Returns the middle
    third's travel along +x over the run and over its last quarter, vs h_stop = tau_y / (rho g sin theta). The voxel
    is at most a quarter of the layer's thickness (the router), so thin layers cost many particles."""
    t0 = time.time()
    th = math.radians(theta_deg)
    name = write_cards(root, "law_slope", paste, block_shape((length, width, thickness)), voxel)
    gravity = (G * math.sin(th), 0.0, -G * math.cos(th))
    t, pts, ok = run_points(root, name, seconds, device=device, gravity=gravity)
    mid = np.abs(pts[0][:, 0]) < length / 6.0
    travel = [float((p[mid, 0] - pts[0][mid, 0]).mean()) for p in pts]
    h_stop = paste.tau_y / (paste.rho * G * math.sin(th))
    k = (3 * len(pts)) // 4
    return {"law": "L3 slope", "h_mm": round(thickness * 1e3, 2), "h_stop_mm": round(h_stop * 1e3, 2),
            "h_over_hstop": round(thickness / h_stop, 3), "travel_mm": round(travel[-1] * 1e3, 3),
            "last_quarter_mm": round((travel[-1] - travel[k]) * 1e3, 3), "check_ok": ok,
            "particles": len(pts[-1]), "wall_s": round(time.time() - t0, 1)}


def film_speed(paste: Paste, thickness: float, theta_deg: float, mean: bool = False) -> float:
    """Analytic surface (plug) speed of a steady Bingham (K = viscosity, n = 1) or Herschel-Bulkley film on an
    incline, or with ``mean`` its depth-averaged speed u_p (h_s (n + 1) / (2n + 1) + h_p) / h, h_s = h - h_p."""
    k, n = (paste.consistency, paste.flow_index) if paste.consistency > 0 else (paste.viscosity, 1.0)
    drive = paste.rho * G * math.sin(math.radians(theta_deg))
    plug = min(paste.tau_y / drive, thickness)
    sheared = thickness - plug
    if sheared <= 0:
        return 0.0
    u = n / (n + 1.0) * (drive / k) ** (1.0 / n) * sheared ** (1.0 + 1.0 / n)
    return u * (sheared * (n + 1.0) / (2.0 * n + 1.0) + plug) / thickness if mean else u


def film(root: Path, paste: Paste, thickness: float, theta_deg: float = 10.0, length: float = 0.2,
         width: float = 0.03, seconds: float = 1.0, voxel: float = 0.005, window: tuple[float, float] = (0.5, 1.0),
         budget: int | None = None, device: str | None = None) -> dict:
    """L4 / L5: a layer on the rough ground between two frictionless side walls, under gravity tilted ``theta_deg``
    towards +x; the middle third's depth-averaged x speed over ``window`` (fractions of the run) vs the analytic mean
    speed (``film_speed(mean=True)``) at the middle third's own thickness over the window (a released layer
    stretches and thins as it flows; speed goes as h^(1 + 1/n)). On the dense MPM grid: the ground lies midway
    between the default sparse grid's node planes, which holds a film's bottom layers back (known gap
    mpm-sparse-floor-offset: 0.82 of analytic for a 20 Pa s film); the dense grid needs a run without CUDA graphs."""
    from alloy.integration.mpm_grid import grid_override  # noqa: PLC0415

    t0 = time.time()
    th = math.radians(theta_deg)
    name = write_cards(root, "law_film", paste, block_shape((length, width, thickness)), voxel, budget)

    def side_walls(scene) -> None:                  # frictionless: the film cannot spread sideways, nor drag on them
        for label, side in (("wall_left", 1.0), ("wall_right", -1.0)):
            scene.add_box(label, size=(length + 0.3, 0.01, 3.0 * thickness), friction=1e-3, static=True,
                          pos=(0.0, side * (0.5 * width + 0.005 + 0.25 * voxel), 1.5 * thickness))

    with grid_override("dense"):
        t, pts, ok = run_points(root, name, seconds, device=device, extra=side_walls,
                                gravity=(G * math.sin(th), 0.0, -G * math.cos(th)),
                                profile="standard" if device == "cpu" else "debug")
    p0 = pts[0]
    mid = np.abs(p0[:, 0]) < length / 6.0
    a, b = (min(len(t) - 1, int(f * (len(t) - 1))) for f in window)
    speed = float((pts[b][mid, 0] - pts[a][mid, 0]).mean() / (t[b] - t[a]))       # depth-averaged: every particle
    offset = thickness - float(np.percentile(p0[mid, 2], 99))           # the surface sits half a spacing above
    h_now = float(np.mean([np.percentile(pts[k][mid, 2], 99) for k in range(a, b + 1)])) + offset
    ref, ref0 = film_speed(paste, h_now, theta_deg, mean=True), film_speed(paste, thickness, theta_deg, mean=True)
    return {"law": "L4/L5 film", "grid": "dense", "h_mm": round(thickness * 1e3, 2),
            "h_window_mm": round(h_now * 1e3, 2),
            "theta_deg": theta_deg, "speed_mm_s": round(speed * 1e3, 3), "ref_mm_s": round(ref * 1e3, 3),
            "ratio": round(speed / ref, 3) if ref > 0 else None, "ref_initial_mm_s": round(ref0 * 1e3, 3),
            "check_ok": ok, "particles": len(pts[-1]), "wall_s": round(time.time() - t0, 1)}
