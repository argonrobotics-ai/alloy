# SPDX-License-Identifier: Apache-2.0
"""The tub fill of the pad tasks: Gym-Anything's "there is concrete in the tub for low spots" (concrete_finishing
task.py:20-22; tub layout.py:346-366).

Their margin trowel is a carrying knife (tools.py:136-146): it holds up to CARRY_CAP on its blade (params.py:160),
and lifted, only what its paste's yield holds there, 0.5·A·tau/(rho·g·6) (coupling.py:922-938, DROP_STICK); the rest
falls as a gob. Here (WP15's ruling, 2026-11-07, after its review of 2eb37860c) that is a scraping surface-flow store,
Store(scrape=True, split 1, fill "knife", capacity CARRY_CAP, stick_area = the blade's face), with the tub both a Tray
the store picks up from and a supply Pool (DC5's stage drips) that catches what the lifted blade sheds back. Each
trip: dip in the tub, lift (the blade keeps its stick limit, ~3.9e-5 m³ of 5 min concrete), fly over the low, let go
(``release``: the gob falls through the supply and lands as GA's cone, coupling.py:978-986), back for the next.

A scripted policy reads each world's seeded start, as a mason reads the slab: ``low_spot`` is the smoothed surface's
lowest paste cell, ``loads`` the trips its deficit needs, and ``drops`` puts each gob in the deepest cell near the low
that the gobs before it left. Each world gets its own program (``ToolCarriage.world_programs``).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from alloy.plugins.surface_flow.stores import Store, Tray
from alloy.plugins.surface_flow.supply import Pool
from alloy.tasks.concrete_pad_start import FORM_H
from alloy.tasks.concrete_patch import CARRY_CAP, TROWEL     # the margin trowel 52D and its carry cap
from alloy.tasks.concrete_tools import HEIGHT, Stroke, ToolProgram

G, DROP_STICK = 9.81, 6.0               # coupling.py:922-938, params.py:137
TUB_FROM_PAD = (1.40, 1.34)             # m: the tub's centre from the pad's (layout.py:27; params.py:29)
TUB_HALF, TUB_VOLUME = (0.45, 0.30), 6.0e-3   # their 0.9 x 0.6 m tub with a ~6 L heap (layout.py:346-351; P:173)
CART_TOP = 0.84                         # m (params.py:172)
TROWEL_FROM_PAD = (1.40, -0.85)         # the trowel's place on the cart from the pad's centre (layout.py:148-150)
AIR_Z = 0.20                            # m over the form plane: the trowel's flights over the pad
DIP = 0.004                             # m: the blade this far over the tub's floor (the heap stands ~11 mm)
DIP_S = 1.5                             # s in the heap (pickup 5/s: 1 - e^-7.5 of the capacity)
DROP_H = 0.04                           # m over the form plane where the trowel lets go
RELEASE_S = 0.3                         # s letting go
MAX_LOADS = 10
LOW_SIGMA = 0.04                        # m: smoothing that finds the low spot under the 60 mm ridges
LOW_RADIUS = 0.2                        # m: the deficit counted around it


def interior(f: dict[str, Any]) -> tuple[float, float]:
    """The centre (x, y) of the paste inside the forms."""
    m = np.asarray(f["paste_mask"]) > 0.5
    xs, ys = np.where(m.any(1))[0], np.where(m.any(0))[0]
    return 0.5 * (f["x"][xs[0]] + f["x"][xs[-1]]), 0.5 * (f["y"][ys[0]] + f["y"][ys[-1]])


def _smooth(a: np.ndarray, sigma_cells: float) -> np.ndarray:
    r = int(3 * sigma_cells) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_cells) ** 2)
    k /= k.sum()
    for axis in (0, 1):
        a = np.apply_along_axis(lambda v: np.convolve(np.pad(v, r, mode="edge"), k, "valid"), axis, a)
    return a


def _deficit(f: dict[str, Any]) -> np.ndarray:
    """[nx, ny] m: the smoothed surface's depth under the form plane, on paste cells."""
    m = np.asarray(f["paste_mask"]) > 0.5
    h = np.where(m, np.asarray(f["h"]) - FORM_H, 0.0)
    return np.where(m, np.maximum(-_smooth(h, LOW_SIGMA / float(f["cell"])), 0.0), 0.0)


def low_spot(f: dict[str, Any]) -> tuple[float, float, float]:
    """(x, y, deficit m³): the smoothed surface's lowest paste cell, and the volume under the form plane within
    LOW_RADIUS of it."""
    m = np.asarray(f["paste_mask"]) > 0.5
    h = np.where(m, np.asarray(f["h"]) - FORM_H, 0.0)
    d = _deficit(f)
    i, j = np.unravel_index(np.argmax(d), d.shape)
    X, Y = np.meshgrid(f["x"], f["y"], indexing="ij")
    near = m & ((X - f["x"][i]) ** 2 + (Y - f["y"][j]) ** 2 < LOW_RADIUS ** 2)
    return float(f["x"][i]), float(f["y"][j]), float(np.maximum(-h, 0.0)[near].sum() * float(f["cell"]) ** 2)


def gob(tau: float, rho: float) -> float:
    """m³ a lifted, level blade keeps (coupling.py:933-936): 0.5·A·tau/(rho·g·DROP_STICK), at most CARRY_CAP."""
    face = 4.0 * TROWEL["half"][0] * TROWEL["half"][1]
    return min(0.5 * face * tau / (rho * G * DROP_STICK), CARRY_CAP)


def loads(deficit: float, volume: float) -> int:
    """Trowel trips for a low spot's deficit (m³) at ``volume`` m³ a trip."""
    return max(1, min(MAX_LOADS, math.ceil(deficit / volume)))


def cone_radius(volume: float, cell: float) -> float:
    """m: the radius a gob of ``volume`` m³ lands over: GA's (3V/pi)^(1/3) rounded up to whole cells, at least 2
    (coupling.py:978-986 with _lay; DC5's stage drips, supply_spray)."""
    return cell * max(2, math.ceil((3.0 * volume / math.pi) ** (1.0 / 3.0) / cell))


def drops(f: dict[str, Any], trips: int, volume: float) -> list[tuple[float, float]]:
    """Where each gob goes: within LOW_RADIUS of the low spot, the deepest cell of the deficit under the form plane
    the gobs before it left (each lands as GA's cone, ``cone_radius``). Cell by cell, not smoothed: the deepest
    pockets are the ones a darby's roll may pass over unfilled."""
    m = np.asarray(f["paste_mask"]) > 0.5
    X, Y = np.meshgrid(f["x"], f["y"], indexing="ij")
    x0, y0, _ = low_spot(f)
    near = m & (np.hypot(X - x0, Y - y0) < LOW_RADIUS)
    d = np.where(near, np.maximum(FORM_H - np.asarray(f["h"]), 0.0), 0.0) * float(f["cell"]) ** 2   # m³ per cell
    rad = cone_radius(volume, float(f["cell"]))
    out = []
    for _ in range(trips):
        i, j = np.unravel_index(np.argmax(d), d.shape)
        x, y = float(f["x"][i]), float(f["y"][j])
        w = np.maximum(1.0 - np.hypot(X - x, Y - y) / rad, 0.0)
        d = np.maximum(d - volume * w / w.sum(), 0.0)
        out.append((x, y))
    return out


def tub(f: dict[str, Any]) -> Tray:
    """The tub beside the pad's north-east corner, where theirs stands, its floor on the ground."""
    ix, iy = interior(f)
    return Tray(center=(ix + TUB_FROM_PAD[0], iy + TUB_FROM_PAD[1], 0.0), half=TUB_HALF, volume=TUB_VOLUME)


def network(f: dict[str, Any]) -> dict:
    """The supply (DC5's stage drips): the tub as a tank and its pool, catching what the lifted blade sheds back."""
    t = tub(f)
    return {"nodes": [{"name": "tub", "medium": "tank", "volume": 0.0, "area": 4.0 * TUB_HALF[0] * TUB_HALF[1]}],
            "edges": [], "pools": [Pool("tub", floor=t.center, half=TUB_HALF)]}


def store() -> Store:
    """The margin trowel's blade: GA's carrying knife (cut what stands above it, fill the lows below it from the
    carry, keep its stick limit when lifted), at their carry cap."""
    face = 4.0 * TROWEL["half"][0] * TROWEL["half"][1]
    return Store(capacity=CARRY_CAP, retained=0.0, held=1.0, split=1.0, drip=0.0, scrape=True, fill="knife",
                 stick_area=face)


def home(f: dict[str, Any]) -> tuple[float, float, float]:
    """The trowel's place on the cart."""
    ix, iy = interior(f)
    return (ix + TROWEL_FROM_PAD[0], iy + TROWEL_FROM_PAD[1], CART_TOP + TROWEL["half"][2])


def trowel_program(f: dict[str, Any], points: list[tuple[float, float]], body: str = "mtrowel",
                   speed: float = 0.5) -> ToolProgram:
    """Cart -> per point: tub (dip, lift), over the point, let go; back to the cart (flights at ``speed`` m/s)."""
    tx, ty, _ = tub(f).center
    zb = TROWEL["half"][2]
    keys = [Stroke(0.0, home(f), mode=HEIGHT)]
    t = 0.0
    for x, y in points:
        t = _fly(keys, t, (tx, ty, FORM_H + AIR_Z), speed)
        for dt, z in ((0.5, DIP + zb), (DIP_S, DIP + zb), (0.5, FORM_H + AIR_Z)):
            t += dt
            keys.append(Stroke(t, (tx, ty, z), mode=HEIGHT))
        t = _fly(keys, t, (x, y, FORM_H + AIR_Z), speed)
        t = _fly(keys, t, (x, y, FORM_H + DROP_H + zb), speed, release=1.0)
        t = _fly(keys, t + RELEASE_S, (x, y, FORM_H + AIR_Z), speed)
    _fly(keys, t, home(f), speed)
    return ToolProgram(body, tuple(keys))


def _fly(keys: list[Stroke], t: float, p: tuple[float, float, float], speed: float = 0.5,
         release: float = 0.0) -> float:
    t += math.dist(keys[-1].pos, p) / speed + 0.2
    keys.append(Stroke(t, p, mode=HEIGHT, release=release))
    return t


__all__ = ["CARRY_CAP", "TROWEL", "cone_radius", "drops", "gob", "home", "interior", "loads", "low_spot", "network",
           "store", "trowel_program", "tub"]
