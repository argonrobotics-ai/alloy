# SPDX-License-Identifier: Apache-2.0
"""Gym-Anything's concrete PAD start state (pad-darby, pad-trowel) in numpy, for WP6's Alloy concrete proofs.

Port of ``concrete_finishing/layout.py`` ``pad_region`` and helpers (L = ``layout.py``, P = ``params.py``, M =
``materials.py``, all under ``[internal note]``;
spec ``[internal note]`` sections 1, 3.3, 5). The pad is drawn on their own 2 mm
grid with their seeded RNG (``default_rng(1000 + seed)``, same draw order), so seed s gives their field cell for cell;
``block`` averages it onto a coarser cell. Material curves are ``alloy.tasks.concrete_materials`` (not re-implemented).

Frame: x east, y north, origin at the pad interior's centre, z = 0 at the subgrade (their floor). Arrays are indexed
[ix, iy] along x, y. Lengths in m, times in s, E / D / S dimensionless in [0, 1].
"""

from __future__ import annotations

import math

import numpy as np

from alloy.tasks import concrete_materials as CM

PAD_TASKS = ("pad-darby", "pad-edge", "pad-float", "pad-trowel", "pad-finish")
INTERIOR = (1.219, 0.914)                  # m, 48 x 36 in inside the forms (P:30)
FORM_W, FORM_H = 0.038, 0.089              # m, 2x4 form: width, height = top of the pad (P:32)
CELL = 0.002                               # m, their pad grid (L:228)
RNG_OFFSET = 1000                          # default_rng(1000 + seed) (L:223)
# age at start at 21 C [s] (L:210); scaled by the MASCO factor of the site temperature (L:241-242)
AGES_21C = {"pad-darby": 5 * 60.0, "pad-edge": 60 * 60.0, "pad-float": 150 * 60.0, "pad-trowel": 165 * 60.0,
            "pad-finish": 5 * 60.0}
T_SURF_STRUCK, T_SURF_MAX = 0.0008, 0.003  # paste layer struck-off / after floating (P:79-80)
AGG_EXPOSED = 0.30                         # share of a struck-off surface with aggregate showing (P:86)
EDGER_RADIUS, EDGER_TRACK, EDGER_TRACK_W = 0.0095, 0.0012, 0.0762   # m (P:146 edger radius; P:167; L:309 3 in track)
AIR = CM.AIR                               # (air C, RH %, wind m/s) at 21.1 C (P:101)
BLEED_DT = 10.0                            # s, bleed-history step (L:333)


def smooth_noise(rng: np.random.Generator, shape: tuple[int, int], cells: int) -> np.ndarray:
    """Their smoothed unit noise (L:214-221): (cells)^2 passes of the 3x3 binomial kernel, unit std."""
    k = rng.normal(0, 1, shape)
    for _ in range(max(1, int(cells)) ** 2):
        k = 0.5 * k + 0.125 * (np.roll(k, 1, 0) + np.roll(k, -1, 0) + np.roll(k, 1, 1) + np.roll(k, -1, 1))
    return k / max(float(k.std()), 1e-9)


def _grid() -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Cell-centre x, y (m, from the interior's south-west corner) and the shape: origin one form width + 2 mm outside
    the interior (L:225-231); nu, nv = round((size + 2 FW + 4 mm) / cell) = 650 x 497."""
    shape = (int(round((INTERIOR[0] + 2 * FORM_W + 0.004) / CELL)),
             int(round((INTERIOR[1] + 2 * FORM_W + 0.004) / CELL)))
    o = -FORM_W - 0.002
    return (o + (np.arange(shape[0]) + 0.5) * CELL, o + (np.arange(shape[1]) + 0.5) * CELL, shape)


def _struck(rng, u, v, shape, seed, inside) -> dict[str, np.ndarray]:
    """The struck-off surface (L:243-262): ridges +-2 mm at 60 mm, a low corner, a high band, 0.5 mm roughness; the
    bulk and surface layer, exposed aggregate and texture. u, v are x, y from the south-west corner (L:244-245)."""
    # L:246
    lo_c = (0.15 + 0.10 * rng.random(), 0.15 + 0.10 * rng.random()) if seed % 2 == 0 else (INTERIOR[0] - 0.2, 0.18)
    # L:247
    hi_c = (0.55 + 0.25 * rng.random(), 0.45 + 0.25 * rng.random())
    # L:248
    ridge = 0.002 * np.sin(2 * math.pi * u / 0.060 + 0.8 * np.sin(2 * math.pi * v / 0.35) + rng.random() * 6.28)
    # L:249
    low = -0.005 * np.exp(-((u - lo_c[0]) ** 2 + (v - lo_c[1]) ** 2) / (2 * 0.08 ** 2))
    # L:250
    high = 0.004 * np.exp(-((u - hi_c[0]) ** 2 + (v - hi_c[1]) ** 2) / (2 * 0.05 ** 2))
    # L:251
    rough = 0.0005 * smooth_noise(rng, shape, 2)
    # L:253
    top = FORM_H + ridge + low + high + rough
    return {
        # L:255
        "ts": np.where(inside, T_SURF_STRUCK, 0.0),
        # L:256
        "tb": np.where(inside, np.maximum(top - T_SURF_STRUCK, 0.0), 0.0),
        # L:257
        "E": np.where(inside, np.clip(AGG_EXPOSED * (1 + 0.4 * smooth_noise(rng, shape, 3)), 0, 1), 0.0),
        # L:258
        "a": np.where(inside, 8.0e-4, 0.0)}


def _smear(f: dict, rng, X, Y, shape) -> None:
    """Concrete the board pushed over two form tops: 1.5 mm +-0.8 mm noise, tapering to the ends (L:263-270)."""
    for side in ("s", "w"):
        if side == "s":
            # L:265
            band = (Y < 0) & (Y > -FORM_W) & (X > 0.2) & (X < 0.75)
            # L:266
            along = (X - 0.475) / 0.275
        else:
            # L:265
            band = (X < 0) & (X > -FORM_W) & (Y > 0.3) & (Y < 0.6)
            # L:266
            along = (Y - 0.45) / 0.15
        # L:267
        taper = np.clip(1.0 - np.abs(along), 0.0, 1.0) ** 0.5
        # L:268
        f["tb"] = np.where(band, (0.0015 + 0.0008 * smooth_noise(rng, shape, 8).clip(-1.5, 1.5)) * taper, f["tb"])
    # L:269
    f["tb"] = np.maximum(f["tb"], 0.0)


def _edge_distance(X, Y) -> np.ndarray:
    """Distance in from the nearest form's inner face (negative on the forms) (L:299, 319)."""
    return np.minimum.reduce([X, INTERIOR[0] - X, Y, INTERIOR[1] - Y])


def _darbied(f: dict, rng, inside, shape) -> None:
    """Darbied flat to about +-1.3 mm: surface layer 2 mm, E 0.01, texture 0.4 mm (L:286-296)."""
    # L:290
    top = FORM_H + 0.0003 * smooth_noise(rng, shape, 8) + 0.0001 * smooth_noise(rng, shape, 1)
    # L:291-292
    f["ts"] = np.where(inside, 0.002, 0.0)
    # L:293
    f["tb"] = np.where(inside, np.maximum(top - 0.002, 0.0), f["tb"])
    # L:294
    f["E"] = np.where(inside, 0.01, 0.0)
    # L:295
    f["a"] = np.where(inside, 4.0e-4, 0.0)


def _edged(f: dict, inside, d) -> None:
    """The edger's work (L:299-318): a 3/8 in arris along all four forms and its 76 mm track pressed 1.2 mm below
    the field. Adds edge_drop [m] (arris removal), track (mask), compact (lip cell, 1) to the fields."""
    r = EDGER_RADIUS
    top = f["b"] + f["tb"] + f["ts"]
    # L:307
    drop = np.where((d >= 0) & (d < r), r - np.sqrt(np.maximum(r * r - (r - d) ** 2, 0.0)), 0.0)
    # L:308
    track = (d >= 0) & (d < EDGER_TRACK_W)
    # L:310
    new_top = np.where(inside, top - drop + np.where(track, -EDGER_TRACK, 0.0), top)
    # L:311
    f["tb"] = np.where(inside, np.maximum(new_top - f["b"] - f["ts"], 0.0), f["tb"])
    # L:312
    f["compact"] = np.where(inside & (d < 0.0025), 1.0, 0.0).astype(np.float32)
    # L:313
    f["a"] = np.where(track & inside, 5.0e-5, f["a"])
    f["edge_drop"], f["track"] = np.where(inside, drop, 0.0), track & inside


def _floated(f: dict, inside, d) -> None:
    """The floats' work (L:321-326): the track's step floated out, a = 0.25 mm, surface layer 3 mm, D = 0.10."""
    # L:324
    h = f["b"] + f["tb"] + f["ts"]
    # L:326
    h = np.where(inside & (d >= 0) & (d < EDGER_TRACK_W), h + EDGER_TRACK, h)
    # L:327
    f["a"] = np.where(inside, 2.5e-4, 0.0)
    # L:328
    f["ts"] = np.where(inside, T_SURF_MAX, 0.0)
    # L:329
    f["D"] = np.where(inside, 0.10, 0.0)
    # L:330
    f["tb"] = np.where(inside, np.maximum(h - f["b"] - f["ts"], 0.0), f["tb"])


def evaporation_rate(temp_c: float, air: str) -> float:
    """kg/m2/h at the site: concrete at temp_c, room air shifted with it (L:81 reg.evap, M:166-169)."""
    return CM.air_evaporation_kg_m2_h(air, temp_c)


def bleed_history(age: float, temp_c: float, air: str, bleed_factor: float = 1.0) -> float:
    """Bleed film w [m] on the paste at `age` s: their clock from strike-off in 10 s steps, rising bleed at mid-step
    age less evaporation, clipped at 0 (L:333-343). float32 arithmetic as theirs. Trapped water q stays 0: nothing
    has sealed the surface yet (L:329, no q write). Their `bleed` argument is the mean in kg/m2/h (L:232, 675;
    default 0.15, variants 0.8-1.2 x); here it is BLEED_MEAN * bleed_factor."""
    mat = CM.Concrete(bleed_mean=CM.BLEED_MEAN * bleed_factor)
    evap = np.float32(evaporation_rate(temp_c, air) / 3600.0 / 1000.0)
    w, clock = np.float32(0.0), -age
    while clock < -1e-9:
        step = min(BLEED_DT, -clock)
        rate = float(mat.bleed_rate(clock + step / 2 + age, temp_c))
        w = np.float32(w + np.float32(step * rate / 1000.0))
        w = np.float32(max(float(w - np.float32(step) * evap), 0.0))
        clock += step
    return float(w)


def pad_fields(task: str, seed: int, temp_c: float = 21.1, air: str = "still",
               bleed_factor: float = 1.0) -> dict[str, np.ndarray | float]:
    """The pad's start state for `task` (pad-darby, pad-trowel; also pad-edge / -float / -finish) on their 2 mm grid.

    Keys (arrays [650, 497]): x, y (1-D cell centres, m, origin at the interior's centre); b rigid base (m; form
    top 0.089 on the forms, 0 inside); tb bulk, ts surface layer (m); bl blister (0); h = b + tb + ts + bl surface
    height (m, z = 0 at subgrade); paste_mask (interior), form_mask; smear (m, paste on form tops = tb there); E
    aggregate exposure, a texture amplitude (m), D densification, S sealing (0), w bleed film (m), q trapped water
    (m, 0), compact, edge_drop (m), track (mask), d_edge (m, distance in from the nearest form, <0 on forms).
    Floats: age (s at start, MASCO-scaled), age_21c, cell, form_top, edger_radius; interior_half (array, m)."""
    if task not in PAD_TASKS:
        raise ValueError(f"task must be one of {PAD_TASKS}, got {task!r}")
    if air not in AIR:
        raise ValueError(f"air must be one of {tuple(AIR)}, got {air!r}")
    u, v, shape = _grid()
    X, Y = u[:, None] + 0 * v[None, :], 0 * u[:, None] + v[None, :]
    # L:236
    inside = (X > 0) & (X < INTERIOR[0]) & (Y > 0) & (Y < INTERIOR[1])
    rng = np.random.default_rng(RNG_OFFSET + seed)
    f = _struck(rng, X, Y, shape, seed, inside)
    # L:240
    f["b"] = np.where(~inside, FORM_H, 0.0)
    _smear(f, rng, X, Y, shape)
    d = _edge_distance(X, Y)
    # L:271
    if task in ("pad-edge", "pad-float", "pad-trowel"):
        _darbied(f, rng, inside, shape)
    # L:274
    if task in ("pad-float", "pad-trowel"):
        _edged(f, inside, d)
    # L:277
    if task == "pad-trowel":
        _floated(f, inside, d)
    return _assemble(f, task, temp_c, air, bleed_factor, (u, v), inside, d)


def _assemble(f, task, temp_c, air, bleed_factor, uv, inside, d) -> dict[str, np.ndarray | float]:
    """Add the bookkeeping fields, the age and the bleed film, with their dtypes (b, tb, ts, bl float64; the rest
    float32)."""
    shape = inside.shape
    # L:241-242
    age = AGES_21C[task] * CM.masco_factor(temp_c)
    out: dict[str, np.ndarray | float] = {k: f[k] for k in ("b", "tb", "ts")}
    out["bl"] = np.zeros(shape)
    for k in ("E", "a", "D", "S", "q", "compact", "edge_drop"):
        out[k] = np.asarray(f.get(k, np.zeros(shape)), np.float32)
    out["track"] = f.get("track", np.zeros(shape, bool))
    # paste.py:108-109
    out["h"] = out["b"] + out["tb"] + out["ts"] + out["bl"]
    out["smear"] = np.where(~inside, out["tb"], 0.0)
    out["paste_mask"], out["form_mask"], out["d_edge"] = inside, ~inside, d
    film = np.float32(bleed_history(age, temp_c, air, bleed_factor))
    out["w"] = np.where(inside, film, np.float32(0.0)).astype(np.float32)
    origin = np.array(INTERIOR) / 2
    out["x"], out["y"] = uv[0] - origin[0], uv[1] - origin[1]
    out.update(age=age, age_21c=AGES_21C[task], cell=CELL, form_top=FORM_H, edger_radius=EDGER_RADIUS,
               interior_half=origin)
    return out


def _block_axis(a: np.ndarray, axis: int, factor: int, boolean: bool) -> np.ndarray:
    n = a.shape[axis] // factor * factor
    a = np.take(a, np.arange(n), axis=axis)
    shape = a.shape[:axis] + (n // factor, factor) + a.shape[axis + 1:]
    m = a.reshape(shape).astype(float).mean(axis=axis + 1)
    return m >= 0.5 if boolean else m


def block(field: np.ndarray, factor: int) -> np.ndarray:
    """Block-average a pad field (1-D coordinates or [ix, iy]) onto cells `factor` times larger (2 mm -> 4 mm for 2,
    10 mm for 5). Boolean masks go by majority (>= half). A remainder of fewer than `factor` trailing cells is cut
    (497 rows -> 495 at 5, so the blocks start at the 2 mm grid's origin and the form edge stays on a block edge)."""
    if factor < 1 or int(factor) != factor:
        raise ValueError(f"factor must be a positive integer, got {factor!r}")
    a = np.asarray(field)
    boolean = a.dtype == bool
    for axis in range(a.ndim):
        a = _block_axis(a, axis, int(factor), boolean and axis == a.ndim - 1)
    return a.astype(field.dtype) if boolean or field.dtype != np.float64 else a


def blocked(fields: dict, factor: int) -> dict[str, np.ndarray | float]:
    """pad_fields output on a `factor` times coarser grid (arrays blocked, scalars kept, cell scaled)."""
    out = {k: block(v, factor) if isinstance(v, np.ndarray) and k != "interior_half" else v for k, v in fields.items()}
    out["cell"] = fields["cell"] * factor
    return out
