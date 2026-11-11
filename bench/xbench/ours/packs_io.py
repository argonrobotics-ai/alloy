# SPDX-License-Identifier: Apache-2.0
"""Pack lookup shared by the Alloy-side tools: a pack directory names its scene and its Alloy bully target.

``packs/<name>/<name>.json`` (``scene_pack.py``) carries ``alloy_target``; the tools take ``--pack packs/<name>`` and
build Alloy's side from that target (``alloy.bully.targets.get_target``), so T12 and the P3-2b embodiments run
through the same code. A batch npz names its target in ``meta.alloy_target`` (default ``task_t12``).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np

log = logging.getLogger("xbench.ours")
HERE = Path(__file__).resolve().parent
T12 = HERE / "packs" / "t12"
DEFAULT_TARGET = "task_t12"


def load_pack(pack_dir: Path = T12) -> dict:
    pack_dir = Path(pack_dir)
    return json.loads((pack_dir / f"{pack_dir.name}.json").read_text())


def target_of(pack: dict) -> str:
    return str(pack.get("alloy_target") or DEFAULT_TARGET)


def set_route(det: bool | None) -> str:
    """WP5's ordered MuJoCo Warp route on (default) or off (``ALLOY_MJW_ROUTE=off``); returns the setting in force."""
    if det is not None:
        os.environ["ALLOY_MJW_ROUTE"] = "on" if det else "off"
    return os.environ.get("ALLOY_MJW_ROUTE", "on (default)")


def pack_of_target(target: str) -> dict | None:
    """The pack built for ``target`` (``alloy_target``), or None (law packs have none)."""
    for f in sorted(HERE.glob("packs/*/*.json")):
        if f.stem == f.parent.name:
            pack = json.loads(f.read_text())
            if pack.get("alloy_target") == target:
                return pack
    return None


def mujoco_shape_pairs(model, target: str):
    """The model's own shape pairs (``model.shape_contact_pairs``) kept only where the pack's ``contact_body_pairs``
    (MuJoCo's filter: excludes, contype / conaffinity, welded bodies and filterparent) lets the two bodies collide.

    Newton's importer filters less (a jointless gripper two fixed joints below the wrist, fixed bases resting on the
    table), so without this the scorer reads overlaps MuJoCo never generates contacts for (the 2F-85 four-bar on its
    mount, at rest by design). Returns None (the model's pairs) when ``target`` has no pack."""
    import warp as wp

    pack = pack_of_target(target)
    pairs = getattr(model, "shape_contact_pairs", None)
    if pack is None or pairs is None:
        return None
    allowed = {tuple(sorted((p["body1"], p["body2"]))) for p in pack["contact_body_pairs"]}
    names = [str(lbl).rsplit("/", 1)[-1] for lbl in model.body_label] + ["world"]   # shape_body -1 -> "world"
    sb = model.shape_body.numpy()
    host = pairs.numpy().reshape(-1, 2)
    keep = [tuple(sorted((names[sb[a]], names[sb[b]]))) in allowed for a, b in host]
    return wp.array(host[np.asarray(keep, dtype=bool)], dtype=wp.vec2i, device=model.device)


def runner(target: str, worlds: int, device: str):
    """Alloy's ``EpisodeRunner`` (WP7's SB-7 harness) for ``target``."""
    from alloy.bully.replay import EpisodeRunner
    from alloy.bully.targets import get_target
    from alloy.bully.types import BullyLimits

    return EpisodeRunner(get_target(target), worlds, device, BullyLimits())


def target_model(target: str, worlds: int, device: str, hulls: bool = True):
    """The Newton model Alloy builds for ``target`` with ``worlds`` worlds (its ``make_sim``). With ``hulls`` (the
    scorer's model) its colliding mesh geoms are convex hulls, as MuJoCo collides them (``hull_meshes``)."""
    from alloy.bully.targets import get_target
    from hull_meshes import HullCount, colliding_mesh_shapes, meshes_as_hulls

    tg = get_target(target)
    if not hulls:
        return tg.make_sim(worlds, device).model
    with meshes_as_hulls() as counts:
        model = tg.make_sim(worlds, device).model
    n = counts[-1] if counts else HullCount()
    log.info("scorer model %s x %d: %d colliding mesh shapes as convex hulls (%d meshes), %d SDF/hydroelastic kept, "
             "%d MESH left", target, worlds, n.shapes, n.meshes, n.kept_sdf, colliding_mesh_shapes(model).size)
    return model


__all__ = ["DEFAULT_TARGET", "T12", "load_pack", "mujoco_shape_pairs", "pack_of_target", "runner", "set_route",
           "target_model", "target_of"]
