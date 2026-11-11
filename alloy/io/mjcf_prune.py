# SPDX-License-Identifier: Apache-2.0
"""Cut the robots out of a larger MJCF file by pruning a copy of the file itself.

Everything that belongs to the kept robots stays exactly as the author wrote it: default classes, actuators of any
kind, tendons, equality couplings, contact excludes and pairs, sensors, and the keyframes (cut down to the kept
joints and actuators). Everything that belongs to a removed body goes, with every element that refers to it. Mesh and
texture paths become absolute, so the pruned file can live anywhere. The pruned file is loaded once in MuJoCo.

Example::

    from alloy.io.mjcf_prune import prune_mjcf
    rep = prune_mjcf("scene.xml", {"link0"}, "out/panda_robot.xml")   # keep the top-level body "link0"
    print(rep.kept_world, rep.removed["contact"], rep.notes)
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("alloy.io")

WORLD_TAGS = ("body", "geom", "site", "camera", "light")
# attribute -> the kind of element it names (sections after <worldbody> refer to bodies, geoms, ... by name)
REF_KIND = {"body": "body", "body1": "body", "body2": "body", "geom": "geom", "geom1": "geom", "geom2": "geom",
            "joint": "joint", "joint1": "joint", "joint2": "joint", "jointinparent": "joint", "site": "site",
            "site1": "site", "site2": "site", "refsite": "site", "cranksite": "site", "slidersite": "site",
            "sidesite": "site", "tendon": "tendon", "tendon1": "tendon", "tendon2": "tendon",
            "actuator": "actuator", "camera": "camera"}
ANY_KIND = ("objname", "refname")                       # sensors name any kind of object
SECTIONS = ("contact", "tendon", "equality", "actuator", "sensor")
QPOS_SIZE = {0: 7, 1: 4, 2: 1, 3: 1}                    # mjtJoint free, ball, slide, hinge
DOF_SIZE = {0: 6, 1: 3, 2: 1, 3: 1}
FILE_DIRS = {"mesh": ("meshdir", "assetdir"), "hfield": ("meshdir", "assetdir"), "skin": ("meshdir", "assetdir"),
             "texture": ("texturedir", "assetdir")}


class PruneError(ValueError):
    """The file cannot be pruned (the caller falls back to a generated robot file); the message says why."""


@dataclass
class PruneReport:
    """What a prune kept and removed: ``removed`` counts elements per section, ``kept_world`` says the file's
    static world geoms stayed (the file excludes robot-world contacts), ``notes`` lists what was dropped and why."""

    path: Path
    kept_world: bool = False
    removed: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class _Removed:
    names: dict[str, set[str]] = field(default_factory=lambda: {k: set() for k in
                                                               ("body", "geom", "joint", "site", "camera", "tendon",
                                                                "actuator")})

    def add_subtree(self, el: ET.Element) -> None:
        for sub in el.iter():
            kind = "joint" if sub.tag == "freejoint" else sub.tag
            if kind in self.names and sub.get("name"):
                self.names[kind].add(sub.get("name"))

    def refers(self, el: ET.Element) -> bool:
        """True if ``el`` or a child names a removed element."""
        union = set().union(*self.names.values())
        for sub in el.iter():
            for attr, value in sub.attrib.items():
                if attr in REF_KIND and value in self.names[REF_KIND[attr]]:
                    return True
                if attr in ANY_KIND and value in union:
                    return True
        return False


def _absolute_files(root: ET.Element, src_dir: Path) -> None:
    """Make every asset ``file`` attribute absolute (compiler meshdir / texturedir / assetdir applied, then dropped)."""
    dirs: dict[str, str] = {}
    strip = False
    for comp in root.iter("compiler"):
        for key in ("meshdir", "texturedir", "assetdir"):
            if comp.get(key) is not None:
                dirs[key] = comp.attrib.pop(key)
        strip = strip or comp.attrib.pop("strippath", "false") == "true"
    for asset in root.iter("asset"):
        for el in asset:
            base = next((dirs[d] for d in FILE_DIRS.get(el.tag, ()) if d in dirs), "")
            for attr, value in list(el.attrib.items()):
                if attr.startswith("file") and value:
                    name = Path(value).name if strip else value
                    el.set(attr, str((src_dir / base / name).resolve()))


def _joint_elements(body: ET.Element) -> list[ET.Element]:
    """Joint elements under ``body`` in MuJoCo's order (depth first, document order)."""
    out = []
    for el in body:
        if el.tag in ("joint", "freejoint"):
            out.append(el)
        elif el.tag == "body":
            out.extend(_joint_elements(el))
    return out


@dataclass
class _KeyMasks:
    """Which entries of a keyframe's arrays belong to kept joints, actuators and mocap bodies."""

    qpos: list[bool]
    qvel: list[bool]
    ctrl: list[bool]
    act: list[bool]
    mocap: list[bool]


def _key_masks(src: Path, root: ET.Element, keep: set[str]) -> tuple[_KeyMasks, list[int]]:
    """Masks from the compiled original model (joint types) and the document order, and each actuator's number of
    state values. The ctrl and act masks are filled in once the actuators are pruned."""
    import mujoco  # noqa: PLC0415

    try:
        m = mujoco.MjModel.from_xml_path(str(src))
    except Exception as exc:  # noqa: BLE001
        raise PruneError(f"MuJoCo cannot load the source file: {exc}") from exc
    tops = [el for el in root.find("worldbody") if el.tag == "body"]
    joints = [top.get("name") in keep for top in tops for _ in _joint_elements(top)]
    mocap = [top.get("name") in keep for top in tops if top.get("mocap") == "true"]
    nu = sum(len(sec) for sec in root.findall("actuator"))
    if len(joints) != m.njnt or nu != m.nu or len(mocap) != m.nmocap:
        raise PruneError("joint, actuator or mocap counts differ from the compiled model")
    qpos = [k for k, t in zip(joints, m.jnt_type) for _ in range(QPOS_SIZE[int(t)])]
    qvel = [k for k, t in zip(joints, m.jnt_type) for _ in range(DOF_SIZE[int(t)])]
    return _KeyMasks(qpos, qvel, [], [], mocap), [int(n) for n in m.actuator_actnum]


def _trim(text: str | None, mask: list[bool], width: int = 1) -> str | None:
    if text is None:
        return None
    vals = text.split()
    if len(vals) != len(mask) * width:
        raise PruneError(f"keyframe array has {len(vals)} values, the model needs {len(mask) * width}")
    kept = [v for i, v in enumerate(vals) if mask[i // width]]
    return " ".join(kept)


def _trim_keys(root: ET.Element, masks: _KeyMasks) -> None:
    for sec in root.findall("keyframe"):
        for key in sec:
            for attr, mask, width in (("qpos", masks.qpos, 1), ("qvel", masks.qvel, 1), ("ctrl", masks.ctrl, 1),
                                      ("act", masks.act, 1), ("mpos", masks.mocap, 3), ("mquat", masks.mocap, 4)):
                new = _trim(key.get(attr), mask, width)
                if new is None:
                    continue
                if new:
                    key.set(attr, new)
                else:
                    del key.attrib[attr]


def _world_excluded(root: ET.Element, kept_bodies: set[str]) -> bool:
    """True if the file excludes contacts between the world body and a kept body."""
    for ex in root.iter("exclude"):
        pair = {ex.get("body1"), ex.get("body2")}
        if "world" in pair and pair & kept_bodies:
            return True
    return False


def _prune_world(root: ET.Element, keep: set[str], rep: PruneReport) -> _Removed:
    world = root.find("worldbody")
    if world is None or len(root.findall("worldbody")) != 1:
        raise PruneError("the file needs exactly one <worldbody>")
    if any(el.tag not in WORLD_TAGS for el in world):
        raise PruneError(f"<worldbody> has children other than {WORLD_TAGS} (frames, replicates, includes)")
    tops = {el.get("name"): el for el in world if el.tag == "body"}
    missing = keep - set(tops)
    if missing:
        raise PruneError(f"no top-level body named {sorted(missing)}; top-level bodies: {sorted(n for n in tops if n)}")
    kept_names = {b.get("name") for name in keep for b in tops[name].iter("body")}
    rep.kept_world = _world_excluded(root, kept_names)
    removed, keep_el = _Removed(), {id(tops[name]) for name in keep}
    for el in list(world):
        gone = (el.tag == "body" and id(el) not in keep_el) or (el.tag == "geom" and not rep.kept_world)
        if gone:
            removed.add_subtree(el)
            world.remove(el)
            rep.removed["worldbody"] = rep.removed.get("worldbody", 0) + 1
    return removed


def _prune_sections(root: ET.Element, removed: _Removed, rep: PruneReport) -> list[bool]:
    """Drop elements that name a removed element; returns the kept mask of the actuators (document order)."""
    act_mask: list[bool] = []
    for tag in SECTIONS:
        for sec in root.findall(tag):
            for el in list(sec):
                gone = removed.refers(el)
                if tag == "actuator":
                    act_mask.append(not gone)
                if not gone:
                    continue
                if tag in ("tendon", "actuator") and el.get("name"):
                    removed.names[tag].add(el.get("name"))
                sec.remove(el)
                rep.removed[tag] = rep.removed.get(tag, 0) + 1
    return act_mask


def _drop_unused_meshes(root: ET.Element) -> None:
    """Remove mesh assets no remaining geom (or default class) uses, so their files are not needed."""
    used = {el.get("mesh") for el in root.iter() if el.tag != "mesh" and el.get("mesh")}
    for asset in root.findall("asset"):
        for el in asset.findall("mesh"):
            if (el.get("name") or Path(el.get("file", "")).stem) not in used:
                asset.remove(el)


def _keyframes(root: ET.Element, masks: _KeyMasks | None, actnum: list[int], act_mask: list[bool], why: str,
               rep: PruneReport) -> None:
    """Cut the keyframes down to the kept joints and actuators, or drop them when the masks are unknown."""
    if root.find("keyframe") is None:
        return
    if masks is None or len(act_mask) != len(actnum):
        for sec in root.findall("keyframe"):
            root.remove(sec)
        rep.notes.append(f"keyframes dropped ({why or 'actuator count differs from the compiled model'})")
        return
    masks.ctrl = act_mask
    masks.act = [k for k, n in zip(act_mask, actnum) for _ in range(n)]
    _trim_keys(root, masks)


def _check_loads(path: Path) -> None:
    import mujoco  # noqa: PLC0415

    try:
        mujoco.MjModel.from_xml_path(str(path))
    except Exception as exc:  # noqa: BLE001
        raise PruneError(f"the pruned file does not load in MuJoCo: {exc}") from exc


def prune_mjcf(src: str | Path, keep: set[str], dst: str | Path) -> PruneReport:
    """Write a copy of the MJCF ``src`` at ``dst`` that keeps only the top-level bodies named in ``keep`` (and the
    world's own geoms when the file excludes robot-world contacts). Raises :class:`PruneError` when the file's
    layout is not supported (includes, frames, deformables) or the result does not load in MuJoCo."""
    src, dst = Path(src).resolve(), Path(dst)
    tree = ET.parse(src)
    root = tree.getroot()
    for tag in ("include", "deformable", "replicate", "frame", "attach"):
        if root.find(f".//{tag}") is not None:
            raise PruneError(f"the file uses <{tag}>, which the pruner does not handle")
    rep = PruneReport(dst)
    masks, actnum, why = None, [], ""
    try:
        masks, actnum = _key_masks(src, root, set(keep))
    except PruneError as exc:
        why = str(exc)
    removed = _prune_world(root, set(keep), rep)
    act_mask = _prune_sections(root, removed, rep)
    _keyframes(root, masks, actnum, act_mask, why, rep)
    _absolute_files(root, src.parent)
    _drop_unused_meshes(root)
    dst.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(dst, encoding="unicode")
    _check_loads(dst)
    if rep.kept_world:
        rep.notes.append("the file excludes contacts between the world and the robot, so the world's own geoms "
                         "(floor, fixtures) stay in the robot file and are not made cards")
    log.info("pruned %s to %s: removed %s", src.name, dst, rep.removed)
    return rep
