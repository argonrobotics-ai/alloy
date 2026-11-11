# SPDX-License-Identifier: Apache-2.0
"""behind-radiator's long-reach dog-leg brush on the surface-flow stores, the cut pail it loads from, and the tool's
clearance from the radiator and the trim.

The brush (Gym-Anything's ``longreach``, ``PT/params.py:156-170``: a ProDec PDLB50-class 2 in dog-leg radiator
brush): a 270 mm handle (r 11 mm), then the head set at 35° to it, a 25 mm ferrule (54 x 18 mm) and 60 mm bristles,
50.8 mm wide and 14 mm thick: 355 mm overall. It is held at the end of its handle, the grip centre (the TCP, the body
origin) 11 mm from the handle's end (``PT/env.py:39-60``). Body frame: x along the handle toward the head, z the
side the head bends to (toward the wall in use), y = z x x across the brush's width.

The store (``alloy.plugins.surface_flow.stores``), Gym-Anything's bristle-bundle values (``BRUSH_PAINT``,
``PT/params.py:177-188``): a residual 0.25 of the wetted pores never released (``retained``), 0.75 held after
withdrawal (``held``, ASTM D5068 practice), uptake 2.0 1/s (``pickup``) and drips 0.5 1/s (``drip``). Its capacity
is Gym-Anything's wetted pore volume (``PT/brush.py:69-76``, ``:519-522``), tuft by tuft: the 20 x 5 tufts share
0.8 of the head's envelope (a flat brush's plug), less their filaments (46.8 a tuft, the 12/8 mil taper; twice the
tip half's volume, as Gym-Anything), each wetted over the share of its length that is in the paint. The 15° bevel
leaves the tufts 46.7-59.7 mm long (``PT/tools.py:171-172``), so with the head down and the long side's tips 20 mm
into the paint (the script's dip: 10 mm off the floor of the pail's 30 mm) the short side's tips are 7 mm in:
5.6 mL. The whole head holds 22.0 mL.

It lays by Gym-Anything's metering law (``Store(meter=C)``, ``PT/brush.py:370-391``): while it slides, the film under
its tips is driven toward ``h_m = min(L_c·sqrt(C·η·U/w), 150 µm)``, laid from the free load or picked up. ``C`` is
the sash brush's 0.075, which Gym-Anything gives the long-reach brush too (``PT/params.py:158-166``); ``η`` its Cross
viscosity at ``U``/50 µm with no structure (``PT/brush.py:387``, ``PT/paint.py:98-102``): 0.1805 Pa·s at 0.15 m/s,
0.18 here. The pad has Gym-Anything's tip contact stiffness (100 N/m per tip, ``kc``, x its 20 x 5 tufts over the
tips' area); the bristles' bending (Gym-Anything's pseudo-rigid tufts) is not modelled, so the press is a depth.

The press, 2 N, is not Gym-Anything's force: ``C``, ``L_c`` and the press are one fitted number. Gym-Anything meters
tuft by tuft, ``L_c`` a tuft's strip ``b(1 + 2 sin a)`` (``b`` = sqrt(0.8 x the envelope / 100) = 2.39 mm, ``a``
its bend) and ``w`` the brush's force over the touching tufts' strips (``n·b``, 0.239 m with all 100 touching;
``PT/brush.py:72``, ``:381-385``). The store takes ``L_c`` the contact's length along the slide (14 mm) and ``w``
the force over its width (50.8 mm), so at one force its film is 1.1-1.7x Gym-Anything's (bends 0.8-0.3 rad). The
2 N press lays Gym-Anything's own calibration target, the brushed build (100 µm wet at 0.15 m/s; 80-115 µm, the
Rhoplex SG-10M data sheet, its E7); Gym-Anything's brush lays it at about 0.7-1.6 N (ESTIMATE: the bend is not
modelled here).

The underside: a ``rect`` at the bristle tips, parallel to the wall when the head bends onto it (the pressed tips),
the brush's width by its thickness.

The cut pail: Gym-Anything's 1 qt Bercom pail (r 80 mm at its floor, 140 mm tall) with 0.6 L, on the drop cloth at
x -0.57 m, 0.375 m out from the wall (``PT/params.py:190``, ``PT/layout.py:189``), as a store ``Tray`` of the floor's
area.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from alloy.tasks import paint_process as pp
from alloy.tasks import radiator_scene as rs

# the brush (PT/params.py:156-170)
WIDTH, THICK, BRISTLE = 0.0508, 0.0140, 0.060
FERRULE = (0.054, 0.018, 0.025)            # width, thickness, length [m]
HANDLE_LEN, HANDLE_R, DOGLEG = 0.27, 0.011, math.radians(35.0)
FILAMENTS, D_ROOT, D_TIP = 4680, 0.28e-3, 0.18e-3
HEAD = FERRULE[2] + BRISTLE                # m, bend to tip
BEND = HANDLE_LEN - HANDLE_R               # m, TCP to the bend along the handle
E_HEAD = np.array([math.cos(DOGLEG), 0.0, math.sin(DOGLEG)])     # the head's direction, body frame
TIP = np.array([BEND, 0.0, 0.0]) + HEAD * E_HEAD                  # the bristle tips' centre, body frame
MASS = 0.12                                # kg (a Hamilton radiator brush, PT/params.py:162)
HANDLE_BOX = (0.5 * (BEND + HANDLE_R), HANDLE_R)                   # half length, half side [m]: its mass's shape
HANDLE_COM = (0.5 * (BEND - HANDLE_R), 0.0, 0.0)

# its store (PT/params.py:177-188) and metering (PT/params.py:156-170, PT/brush.py:370-391)
RETAINED, HELD, PICKUP, DRIP = 0.25, 0.75, 2.0, 0.5
TUFT_GRID = (20, 5)                        # tufts across the width and the thickness (PT/params.py:156)
PLUG, BEVEL = 0.8, math.radians(15.0)      # a flat brush's tufts fill 0.8 of the envelope (PT/brush.py:69); its bevel
IMMERSION = 0.020                          # m: the long side's tips into the paint in a dip (the script's)
METER, METER_MAX = 0.075, 150e-6           # GA's metering constant (the sash's) and its film cap h_max [m]
KC, TUFTS = 100.0, 20 * 5                  # N/m per tip, tufts
PAD_STIFFNESS = KC * TUFTS / (WIDTH * THICK)       # Pa/m, 1.41e7: GA's line load at a given press
PRESS_FORCE = 2.0                          # N
PRESS = PRESS_FORCE / (PAD_STIFFNESS * WIDTH * THICK)    # m the tips are pressed past the wall's face (0.2 mm)
BRUSHED_WET = 100e-6                       # m, the brushed build at PRESS_FORCE and 0.15 m/s
METER_VISCOSITY = 0.18                     # Pa·s at the brush's shear, U/50 µm (GA's latex eta_inf, PT/params.py:36)

# the cut pail (PT/params.py:190, PT/layout.py:189)
PAIL_R, PAIL_WALL, PAIL_PAINT = 0.080, 0.0015, 0.6e-3
PAIL_POS = (-0.02 - rs.GA_X, 1.30 - rs.GA_Y, 0.0)
HAND_CLEAR = 0.035          # m: the TCP this far outside the radiator's box (above its top, past its ends): the
                            # hand-clear reach of Gym-Anything's phantom-held brush (down to z 0.341 m, PT/task.py:61)


def tuft_lengths() -> np.ndarray:
    """Each tuft's free length [m], width-major (Gym-Anything's ``tuft_layout``, ``PT/tools.py:149-172``): the bevel
    shortens them toward the head's short side."""
    nx, ny = TUFT_GRID
    x = (np.arange(nx) + 0.5) / nx * WIDTH - 0.5 * WIDTH
    return np.repeat(BRISTLE - (0.5 * WIDTH - x) * math.tan(BEVEL), ny)


def tuft_pores() -> np.ndarray:
    """Each tuft's pore volume over its whole length [m³], Gym-Anything's ``2 C_half`` (``PT/brush.py:69-76``): its
    share of the plug less twice its filaments' tip half (from the mean length's middle to its end; linear taper)."""
    ls = tuft_lengths()
    n, lm = ls.size, float(ls.mean())
    d_mid = 0.5 * (D_ROOT + D_TIP)
    tip_half = 0.25 * math.pi * 0.5 * lm * (d_mid * d_mid + d_mid * D_TIP + D_TIP * D_TIP) / 3.0
    return 2.0 * np.maximum(PLUG * WIDTH * THICK / n * ls / 2.0 - FILAMENTS / n * tip_half * ls / lm, 1e-12)


def capacity(immersion: float = IMMERSION) -> float:
    """The pore volume a dip wets [m³] (Gym-Anything's ``cap``, ``PT/brush.py:519-522``): each tuft's pores times
    the share of its length in the paint, the head down with the long side's tips ``immersion`` deep."""
    ls = tuft_lengths()
    depth = np.clip(immersion - (ls.max() - ls), 0.0, None)
    return float((tuft_pores() * np.minimum(depth / ls, 1.0)).sum())


def metered(speed: float, viscosity: float, length: float = THICK, force: float = PRESS_FORCE) -> float:
    """GA's metered film [m]: ``min(L_c·sqrt(C·η·U/w), h_max)``, ``w`` the press over the brush's width."""
    return min(length * math.sqrt(METER * viscosity * speed * WIDTH / force), METER_MAX)


def brush_store(paint_name: str, initial: float = 0.0):  # noqa: ANN201 - stores.Store
    """The bristles' store, laying by the metering law."""
    from alloy.plugins.surface_flow.stores import Store  # noqa: PLC0415

    kw = {"meter_viscosity": METER_VISCOSITY} if meter_viscosity_supported() else {}
    return Store(capacity=capacity(), retained=RETAINED, held=HELD, pickup=PICKUP, drip=DRIP, initial=initial,
                 species=pp.source_species(paint_name), meter=METER, meter_max=METER_MAX, **kw)  # fmt: skip


def meter_viscosity_supported() -> bool:
    """Whether this plug-in's store meters at a viscosity of its own (else the film's plastic viscosity plus
    τ_y·50 µm/U, the card's low-shear value: the metered film then sits at its cap)."""
    from dataclasses import fields  # noqa: PLC0415

    from alloy.plugins.surface_flow.stores import Store  # noqa: PLC0415

    return "meter_viscosity" in {f.name for f in fields(Store)}


def brush(paint_name: str, initial: float = 0.0, body: str = "brush"):  # noqa: ANN201
    """The brush as a store tool (a surface-flow ``Applicator``) on the wall chart: a rect at the tips, normal toward
    the wall (body +z), its long side across the width (body y)."""
    from alloy.plugins.surface_flow.applicators import Applicator  # noqa: PLC0415

    return Applicator(body=body, shape="rect", size=(0.5 * WIDTH, 0.5 * THICK), center=tuple(float(v) for v in TIP),
                      normal=(0.0, 0.0, 1.0), axis=(0.0, 1.0, 0.0), chart=rs.WALL, stiffness=PAD_STIFFNESS,
                      implicit=True,
                      store=brush_store(paint_name, initial))  # fmt: skip


def pail_tray(volume: float = PAIL_PAINT):  # noqa: ANN201 - stores.Tray
    """The pail as the store's pool: a square of its floor's area, so the paint stands at its true depth."""
    from alloy.plugins.surface_flow.stores import Tray  # noqa: PLC0415

    half = 0.5 * math.sqrt(math.pi) * PAIL_R
    return Tray(center=(PAIL_POS[0], PAIL_POS[1], PAIL_WALL), half=(half, half), volume=volume)


def pail_network() -> dict:
    """The surface-flow supply's network: the pail as a tank with a pool over its floor, so drips falling over it go
    back into it (the supply's stage drips, DC5). Drips elsewhere land on a chart (the wall, the radiator's top, the
    baseboard's ledge), on a shape, or on the floor (``lost``: the drop cloth). The pool's tank books what returns;
    it is not the store's tray the brush loads from (one paint, two books)."""
    from alloy.plugins.surface_flow.supply import Pool  # noqa: PLC0415

    half = 0.5 * math.sqrt(math.pi) * PAIL_R
    floor = (PAIL_POS[0], PAIL_POS[1], PAIL_WALL)
    return {"nodes": [{"name": "pail", "medium": "tank", "volume": 0.0, "area": 4.0 * half * half, "species": (0.0,)}],
            "edges": [], "pools": [Pool("pail", floor=floor, half=(half, half))], "species": (pp.SOLIDS,)}


# --- poses -----------------------------------------------------------------------------------------------------------

def attitude(alpha: float) -> np.ndarray:
    """The brush's rotation (columns: body x, y, z in the world) with its handle in the wall's plane at ``alpha``
    degrees from +x toward +z (0: from the radiator's left end along the gap, 180: from its right end, -90: head
    down from above), the head bending onto the wall (body z = +y). The tips' rect stays parallel to the wall."""
    a = math.radians(alpha)
    x = np.array([math.cos(a), 0.0, math.sin(a)])
    z = np.array([0.0, 1.0, 0.0])
    return np.column_stack([x, np.cross(z, x), z])


def quat_xyzw(rot: np.ndarray) -> tuple[float, float, float, float]:
    """A rotation matrix's quaternion (x, y, z, w), w >= 0."""
    m = rot
    w = 0.5 * math.sqrt(max(1.0 + m[0, 0] + m[1, 1] + m[2, 2], 0.0))
    if w > 1e-6:
        q = ((m[2, 1] - m[1, 2]) / (4 * w), (m[0, 2] - m[2, 0]) / (4 * w), (m[1, 0] - m[0, 1]) / (4 * w), w)
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(max(1.0 + m[i, i] - m[j, j] - m[k, k], 0.0)) * 2.0
        q4 = [0.0, 0.0, 0.0, (m[k, j] - m[j, k]) / s]
        q4[i], q4[j], q4[k] = 0.25 * s, (m[j, i] + m[i, j]) / s, (m[k, i] + m[i, k]) / s
        q = tuple(q4)
    q = np.asarray(q, np.float64) / np.linalg.norm(q)
    return tuple(float(v) for v in (q if q[3] >= 0 else -q))


def tcp_for_tip(tip: Iterable[float], alpha: float) -> np.ndarray:
    """The TCP that puts the bristle tips' centre at ``tip`` [world] at attitude ``alpha``."""
    return np.asarray(tip, np.float64) - attitude(alpha) @ TIP


def on_wall(x: float, z: float, lift: float = 0.0) -> np.ndarray:
    """The tips' centre at wall point (x, z), pressed ``PRESS`` in, or ``lift`` [m] off the wall when > 0."""
    return np.array([x, -lift if lift > 0.0 else PRESS, z])


# --- clearance -------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class OBB:
    centre: np.ndarray
    axes: np.ndarray            # columns: unit axes, world
    half: np.ndarray


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])     # body x onto the head's direction


def tool_boxes(tcp: np.ndarray, alpha: float) -> list[OBB]:
    """The handle, the ferrule and the bristles as boxes [world]."""
    R = attitude(alpha)
    head = R @ _rot_y(DOGLEG)
    bend = np.array([BEND, 0.0, 0.0])
    parts = [(np.array([0.5 * (BEND - HANDLE_R), 0.0, 0.0]), R, (0.5 * (BEND + HANDLE_R), HANDLE_R, HANDLE_R)),
             (bend + 0.5 * FERRULE[2] * E_HEAD, head, (0.5 * FERRULE[2], 0.5 * FERRULE[0], 0.5 * FERRULE[1])),
             (bend + (FERRULE[2] + 0.5 * BRISTLE) * E_HEAD, head, (0.5 * BRISTLE, 0.5 * WIDTH, 0.5 * THICK))]
    return [OBB(tcp + R @ c, axes, np.asarray(h, np.float64)) for c, axes, h in parts]


def box_obb(b: rs.Box) -> OBB:
    x, y, z, w = b.quat
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return OBB(np.asarray(b.pos, np.float64), R, np.asarray(b.half, np.float64))


def separations(tool: list[OBB], obstacles: list[OBB]) -> np.ndarray:
    """The largest gap between each tool box's and each obstacle's projections over their 15 separating axes [m],
    [tools, obstacles]: > 0 apart, <= 0 overlapping (the least penetration along a face or edge axis)."""
    ta = np.stack([t.axes for t in tool])                       # [T, 3, 3], columns are the axes
    oa = np.stack([o.axes for o in obstacles])                  # [O, 3, 3]
    th, oh = np.stack([t.half for t in tool]), np.stack([o.half for o in obstacles])
    d = np.stack([o.centre for o in obstacles])[None] - np.stack([t.centre for t in tool])[:, None]   # [T, O, 3]
    T, O = len(tool), len(obstacles)
    cross = np.cross(np.swapaxes(ta, 1, 2)[:, None, :, None, :], np.swapaxes(oa, 1, 2)[None, :, None, :, :])
    cross = cross.reshape(T, O, 9, 3)
    n = np.concatenate([np.broadcast_to(np.swapaxes(ta, 1, 2)[:, None], (T, O, 3, 3)),
                        np.broadcast_to(np.swapaxes(oa, 1, 2)[None], (T, O, 3, 3)), cross], axis=2)   # [T, O, 15, 3]
    ln = np.linalg.norm(n, axis=-1)
    keep = ln > 1e-9
    n = n / np.where(keep, ln, 1.0)[..., None]
    ra = (np.abs(np.einsum("tij,toaj->toai", np.swapaxes(ta, 1, 2), n)) * th[:, None, None, :]).sum(-1)
    rb = (np.abs(np.einsum("oij,toaj->toai", np.swapaxes(oa, 1, 2), n)) * oh[None, :, None, :]).sum(-1)
    gap = np.abs(np.einsum("toj,toaj->toa", d, n)) - ra - rb
    return np.where(keep, gap, -np.inf).max(axis=-1)


def separation(a: OBB, b: OBB) -> float:
    return float(separations([a], [b])[0, 0])


def hand_clear(tcp: np.ndarray) -> float:
    """How far the TCP is outside the radiator's box past ``HAND_CLEAR`` [m] (>= 0: clear): above its top, past an
    end, or in front of it."""
    x, y, z = tcp
    out = max(abs(x) - 0.5 * rs.RAD_LEN, z - rs.RAD_HEIGHT, -rs.RAD_GAP - rs.RAD_DEPTH - y)
    return out - HAND_CLEAR


def clearance(tcp: np.ndarray, alpha: float, obstacles: list[OBB] | None = None) -> float:
    """The tool's least separation from the radiator and the trim, and the hand's margin [m]: >= 0 is clear."""
    obs = obstacles if obstacles is not None else [box_obb(b) for b in rs.obstacles()]
    gap = float(separations(tool_boxes(tcp, alpha), obs).min())
    return min(gap, hand_clear(tcp))


__all__ = ["BRUSHED_WET", "DRIP", "HAND_CLEAR", "HANDLE_BOX", "HANDLE_COM", "HELD", "MASS", "METER", "METER_MAX",
           "METER_VISCOSITY", "OBB", "PAD_STIFFNESS", "PAIL_POS", "PRESS", "PRESS_FORCE", "RETAINED", "TIP", "attitude",
           "box_obb", "brush", "brush_store", "capacity", "clearance", "hand_clear", "meter_viscosity_supported",
           "metered", "on_wall", "pail_network", "pail_tray", "quat_xyzw", "separation", "separations",
           "tcp_for_tip", "tool_boxes", "tuft_lengths", "tuft_pores"]
