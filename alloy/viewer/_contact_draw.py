# SPDX-License-Identifier: Apache-2.0
"""Draw rigid contacts (:class:`alloy.viewer._contacts.ContactSet`) on a Newton viewer: a point at each contact and an
arrow for the force on ``body_b`` (on ``body_a`` when ``body_b`` is a fixed object, so the arrow is on the
body that can move).

``draw(backend, sets)`` runs between ``backend.begin_frame()`` and ``backend.end_frame()`` of any Newton ``ViewerBase``
(null, GL, Viser, Rerun, USD, file). It calls only the base logging methods ``log_points`` and ``log_arrows``; a
backend without one is skipped with one debug line. Each world's points move by the viewer's own world offset
(``world_offsets``, as ``log_state`` places that world's shapes) and the active layer's transform; worlds hidden with
``set_visible_worlds`` are not drawn. Zero contacts, or ``points=False`` / ``forces=False``, clear the batch, so no
stale drawing stays from the frame before.

**Arrow length** needs no tuning: the weight of the median moving body (its mass times gravity; the median, so a
heavy robot does not shrink the arrows on the small things it handles) is drawn ``REF_BODIES`` body sizes long, a
body size being the median collision radius of the shapes on bodies (Newton's scene scale). Length grows with the
**square root** of the force, so the small forces on light objects stay visible beside large ones: four times the
force draws twice the arrow. Read exact values from ``alloy.viewer.contacts``. One fixed scale per model, so arrows keep their
lengths from frame to frame; an arrow longer than ``MAX_BODIES`` body sizes (an impact) is cut to that length.
"""

from __future__ import annotations

import logging
import weakref
from typing import Any

import numpy as np

from alloy.viewer._contacts import ContactSet

log = logging.getLogger("alloy.viewer")

POINTS = "/alloy/contacts/points"
FORCES = "/alloy/contacts/forces"
POINT_COLOR = (1.0, 0.75, 0.0)          # amber
FORCE_COLOR = (1.0, 0.1, 0.6)           # magenta
REF_BODIES = 4.0                        # a world's whole weight is drawn this many body sizes long
MAX_BODIES = 12.0                       # longest arrow, in body sizes
POINT_BODIES = 0.08                     # point radius, in body sizes
DEFAULT_SIZE = 0.05                     # body size (m) when the model has no shapes on bodies
MIN_ARROW = 1e-6                        # m: shorter arrows (no force: a contact inside the gap) are not drawn
_SKIPPED: set[tuple[str, str]] = set()
_SCALES: weakref.WeakKeyDictionary[Any, tuple[float, float]] = weakref.WeakKeyDictionary()


def draw(backend: Any, sets: dict[int, ContactSet], *, points: bool = True, forces: bool = True) -> None:
    """Log ``sets`` (world -> contacts, from ``read``) as points and force arrows on ``backend`` (module docstring).
    Example: ``v.begin_frame(sim.t); v.log_state(state); draw(v, read(sim, [0])); v.end_frame()``."""
    shown = {w: s for w, s in sets.items() if len(s) and _world_shown(backend, w)}
    pos = _placed(backend, shown)
    if points and len(pos):
        _log(backend, "log_points", POINTS, _vec3(backend, pos), POINT_BODIES * _scales(backend)[0],
             _colors(backend, len(pos), POINT_COLOR))
    else:
        _log(backend, "log_points", POINTS, None, None, None)
    vec = _arrows(backend, np.concatenate([_drawn_force(s) for s in shown.values()])) if forces and len(pos) else pos
    keep = np.linalg.norm(vec, axis=1) > MIN_ARROW if forces else np.zeros(len(pos), bool)
    if keep.any():
        start = pos[keep]
        _log(backend, "log_arrows", FORCES, _vec3(backend, start), _vec3(backend, start + vec[keep]),
             _colors(backend, len(start), FORCE_COLOR))
    else:
        _log(backend, "log_arrows", FORCES, None, None, None)


def _drawn_force(s: ContactSet) -> np.ndarray:
    """The force an arrow shows: on ``body_b``, or on ``body_a`` when ``body_b`` is a fixed object (a ball on a
    table then gets an arrow pushing the ball up, not one buried in the table)."""
    if s.fixed_b.shape[0] != s.force.shape[0]:
        return s.force
    return np.where(s.fixed_b[:, None], -s.force, s.force)


def _log(backend: Any, method: str, name: str, *args: Any) -> None:
    fn = getattr(backend, method, None)
    if not callable(fn):
        key = (type(backend).__name__, method)
        if key not in _SKIPPED:
            _SKIPPED.add(key)
            log.debug("contacts: %s has no %s; contact %s not drawn", key[0], method, name)
        return
    qualify = getattr(backend, "_qualify", None)
    fn(qualify(name) if callable(qualify) else name, *args)


def _world_shown(backend: Any, world: int) -> bool:
    shown = getattr(backend, "_should_render_world", None)
    return bool(shown(world)) if callable(shown) else True


def _placed(backend: Any, sets: dict[int, ContactSet]) -> np.ndarray:
    """Every world's contact points, each moved by its world offset; (n, 3) float64."""
    if not sets:
        return np.zeros((0, 3))
    offsets = _offsets(backend)
    rows = []
    for w, s in sets.items():
        shift = offsets[w] if offsets is not None and 0 <= w < len(offsets) else np.zeros(3)
        rows.append(s.pos + shift)
    return np.concatenate(rows)


def _offsets(backend: Any) -> np.ndarray | None:
    off = getattr(backend, "world_offsets", None)
    if off is None:
        return None
    host = off.numpy() if hasattr(off, "numpy") else np.asarray(off)
    return np.asarray(host, np.float64).reshape(-1, 3)


def _vec3(backend: Any, pts: np.ndarray) -> Any:
    """A Warp vec3 array on the viewer's device, in the active layer's frame."""
    import warp as wp  # noqa: PLC0415 - base dependency, imported when drawing

    arr = wp.array(np.ascontiguousarray(pts, np.float32), dtype=wp.vec3, device=getattr(backend, "device", None))
    move = getattr(backend, "_apply_layer_transform_to_points", None)
    if callable(move) and getattr(backend, "layer", None) is not None:
        return move(arr)
    return arr


def _colors(backend: Any, n: int, rgb: tuple[float, float, float]) -> Any:
    """One colour per item as a Warp vec3 array (the GL viewer does not take a plain tuple here)."""
    import warp as wp  # noqa: PLC0415 - base dependency, imported when drawing

    return wp.array(np.tile(np.float32(rgb), (n, 1)), dtype=wp.vec3, device=getattr(backend, "device", None))


def _arrows(backend: Any, force: np.ndarray) -> np.ndarray:
    """Force (N) -> arrow vector (m): length by the square root of the force at the model's fixed scale, cut to
    ``MAX_BODIES`` body sizes."""
    size, ref = _scales(backend)
    mag = np.linalg.norm(force, axis=1, keepdims=True)
    length = np.minimum(REF_BODIES * size * np.sqrt(mag / ref), MAX_BODIES * size)
    return force * (length / np.maximum(mag, 1e-12))


def _scales(backend: Any) -> tuple[float, float]:
    """(body size m, reference force N) of the backend's model, computed once per model."""
    model = getattr(backend, "model", None)
    try:
        hit = _SCALES.get(model) if model is not None else None
    except TypeError:
        hit = None
    if hit is not None:
        return hit
    out = (_body_size(backend), _ref_force(model))
    try:
        if model is not None:
            _SCALES[model] = out
    except TypeError:
        pass
    return out


def _body_size(backend: Any) -> float:
    est = getattr(backend, "_estimate_scene_scale", None)
    size = float(est()) if callable(est) and getattr(backend, "model", None) is not None else 0.0
    return size if np.isfinite(size) and size > 0.0 else DEFAULT_SIZE


def _ref_force(model: Any) -> float:
    """The weight of the median moving body (N); 1 N when the model has no moving mass or gravity."""
    if model is None or not getattr(model, "body_count", 0):
        return 1.0
    mass, inv = model.body_mass.numpy(), model.body_inv_mass.numpy()
    moving = np.isfinite(mass) & (inv > 0.0)
    if not moving.any():
        return 1.0
    total = float(np.median(mass[moving]))
    g = model.gravity.numpy() if getattr(model, "gravity", None) is not None else np.zeros((1, 3))
    g_mag = float(np.linalg.norm(np.asarray(g, np.float64).reshape(-1, 3)[0])) if np.size(g) else 0.0
    weight = total * (g_mag if np.isfinite(g_mag) and g_mag > 0.0 else 9.81)
    return weight if np.isfinite(weight) and weight > 0.0 else 1.0


__all__ = ["FORCES", "POINTS", "draw"]
