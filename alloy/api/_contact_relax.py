# SPDX-License-Identifier: Apache-2.0
"""Contacts made soft on purpose in the Scene API (GAP-E1; physics ruling D, 2026-11-10 23:41Z):
``scene.relax_contact_depth``.

A robot file can make contacts soft on purpose: Menagerie ANYmal C gives its feet ``solimp="0.015 1 0.03"``, so they
rest 17.2 mm deep (plain MuJoCo C reads the same), and health's PEN rule (PEN_FAIL past 1 cm for 0.1 s) froze it
standing still. ``Scene`` records each relaxed name in ``scene.contact_relax`` (plain data, kept by ``to_dict`` /
``from_dict``, not by ``scene.save()``'s cards: it warns); ``alloy.build`` resolves the names on the built scene and
marks their shapes in the health monitor (``alloy.precision.health_pen.relax_soft_contacts``), in every world and in
every worker process.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from alloy.api.errors import SceneError, UnknownNameError

log = logging.getLogger("alloy.api")
_FIX = "pass a robot or object name (sim.names()), and depth=None or a depth in metres >= 0"


class ContactRelaxMixin:
    """``Scene`` method for contacts made soft on purpose (mixed into ``alloy.Scene``)."""

    @property
    def contact_relax(self) -> dict[str, float | None]:
        """Relaxed names -> stated depth in m (None: each contact's own solimp width)."""
        return self.__dict__.setdefault("_contact_relax", {})

    def relax_contact_depth(self, name: str, depth: float | None = None) -> None:
        """Let the contacts of ``name`` (a robot, one of its bodies ``"<robot>/<body>"``, or an object) rest as deep as
        their file makes them, without health flags. Health then judges each of its contacts by the depth past an
        allowance: by default the contact's own MJCF ``solimp`` width (its third number) when that is wider than
        MuJoCo's 1 mm default, so only contacts made soft on purpose change; or ``depth`` in metres for every contact
        of ``name``. A contact is relaxed when either of its two shapes is, so relaxing a shared object (``"ground"``,
        a table) relaxes every contact with it: name the robot or the part. PEN_WARN fires 1 mm past the allowance and
        PEN_FAIL 1 cm past it for 0.1 s (START_PEN likewise); ``max_penetration_m`` and ``peak_penetration_m`` still
        read the true depths. Rigid bodies and robots only.

        Example: Menagerie ANYmal C's feet (``solimp="0.015 1 0.03"``: 30 mm) rest 17.2 mm deep, which reads INVALID
        standing still; ``scene.add_robot("anymal", path); scene.relax_contact_depth("anymal")`` reads OK, and its
        other contacts (MuJoCo's default solimp) keep the full rule. Kept by ``scene.to_json()`` / ``Scene.from_dict``;
        ``scene.save()``'s cards do not keep it (it warns)."""
        if not isinstance(name, str) or not name:
            raise SceneError(f"name must be a robot or object name, got {name!r}", _FIX, field="name")
        if depth is not None:
            try:
                d = float(depth)
            except (TypeError, ValueError):
                d = math.nan
            if not math.isfinite(d) or d < 0.0:
                raise SceneError(f"depth must be None or a finite depth in metres >= 0, got {depth!r}", _FIX,
                                 field="depth")
            depth = d
        self.contact_relax[name] = depth


def saved(scene: Any) -> dict[str, Any]:
    """``Scene.to_dict`` entry: ``{"contact_relax": {...}}`` when a name is relaxed, else nothing."""
    record = scene.contact_relax
    return {"contact_relax": dict(record)} if record else {}


def warn_unsaved(scene: Any) -> None:
    """``scene.save()``: one warning per setting its cards do not keep (rod clamps, robot sysid, relaxed contacts).

    Example: ``warn_unsaved(scene)`` logs "scene.save(): the cards do not keep relax_contact_depth for anymal; ..."."""
    from alloy.api import _robots, _rods  # noqa: PLC0415

    _rods.warn_unsaved_clamps(scene.items)
    _robots.warn_unsaved_sysid(scene.items)
    if scene.contact_relax:
        log.warning("scene.save(): the cards do not keep relax_contact_depth for %s; a scene loaded from them judges "
                    "those contacts by the full PEN rule again. Keep it with scene.to_json() / Scene.from_dict, or "
                    "call scene.relax_contact_depth again.", ", ".join(sorted(scene.contact_relax)))


def load(scene: Any, data: dict[str, Any]) -> Any:
    """``Scene.from_dict``: the relaxed names back; returns ``scene``."""
    scene.contact_relax.update(dict(data.get("contact_relax") or {}))
    return scene


def install(engine: Any, built: Any, scene: Any) -> None:
    """Mark the shapes of every relaxed name in ``engine``'s health monitor (none relaxed: nothing)."""
    record = getattr(scene, "contact_relax", None) or {}
    if not record:
        return
    from alloy.api.names import NameIndex  # noqa: PLC0415
    from alloy.precision.health_pen import relax_soft_contacts  # noqa: PLC0415

    idx = NameIndex.from_built(built)
    for name, depth in record.items():
        relax_soft_contacts(engine, shapes_of(built, idx, name), depth)


def shapes_of(built: Any, idx: Any, name: str) -> np.ndarray:
    """Every world's shape ids of a handle (a robot, an object) or of one robot body ``"<robot>/<body>"``."""
    handle = built.handles.get(name) or built.handles.get(name.split("/", 1)[0])
    if handle is None or (name not in built.handles and name not in idx.bodies):
        raise UnknownNameError(f"relax_contact_depth: no robot, body or object named {name!r}",
                               f"use one of {sorted(built.handles)} or a robot body from sim.names()['bodies']",
                               field="name")
    t = handle.template
    shapes = np.asarray(list(t.shape), np.int64)
    if name not in built.handles:                                  # one robot body: its own shapes
        owner = built.model.shape_body.numpy()
        shapes = shapes[owner[shapes] == idx.bodies[name]]
    per = len(handle.world_stride.shape)
    if per == 0:
        return shapes
    return (shapes[None, :] + np.arange(int(built.world_count))[:, None] * per).ravel()


def soft_on_purpose(result: dict[str, Any], h: dict[str, Any], engine: Any) -> None:
    """``alloy.check``'s deepest-contact error (past 1 cm) becomes a warning when that world's contacts stayed within
    the allowances of ``scene.relax_contact_depth`` (contacts soft on purpose): the scene relaxed some, and health
    never raised PEN_WARN or PEN_FAIL in that world. ``result["ok"]`` follows. It relies on PEN_WARN: a contact not
    relaxed that ``alloy.check`` saw past 1 cm was past 1 mm at that check, so its world lists PEN_WARN in
    ``seen_flags`` and keeps the error. Example: a relaxed ANYmal C beside ``add_ground`` rests 17 mm deep: listed as
    a warning, ``ok`` True. Called by ``alloy.api.health.check``."""
    from alloy.precision.health_pen import health_monitor  # noqa: PLC0415

    mon = health_monitor(engine)
    allow = getattr(mon, "_shape_allow", None)
    if allow is None or not allow.shape[0] or not allow.numpy().any():
        return
    w = ((result.get("report") or {}).get("metrics") or {}).get("penetration", {}).get("world")
    rows = h.get("worlds") or []
    if w is None or not 0 <= int(w) < len(rows) or {"PEN_WARN", "PEN_FAIL"} & set(rows[int(w)].get("seen_flags", [])):
        return
    for p in result.get("problems", []):
        if p.get("check") == "penetration" and p.get("level") == "error":
            p["level"] = "warn"
            p["what"] += "; within the allowance of scene.relax_contact_depth (soft on purpose; health: no PEN flag)"
    result["ok"] = not any(p.get("level") == "error" for p in result.get("problems", []))
