# SPDX-License-Identifier: Apache-2.0
"""Robot-only MJCF for an arm: the embodiment pack's robot, element for element (physics text unchanged), plus
menagerie's render meshes, optionally name-prefixed so two arms sit in one world.

``robot_xml(spec, prefix, visuals)`` returns an MJCF string (absolute mesh paths) that Newton's ``add_mjcf`` and MuJoCo
both read; ``add_arm(builder, spec, base, prefix)`` places one arm at ``base`` (x y z [qx qy qz qw]); ``robot_file``
writes it for a ``RobotRef``. Built with ElementTree from the pack file, never through ``MjSpec.to_xml`` (its
shorthand vectors are misread by Newton's importer). Changes from the pack:

- the pack's world (table, box, cylinder, mug, floor), its ``init`` keyframe and the excludes / assets they used go;
- the gripper servo overrides of the spec (Franka Hand: ``specs.FRANKA_HAND_SERVO``) are applied;
- render meshes: the pack's visual layer (``library/robots/<pack>/visuals``, :mod:`alloy.scene.robot_visuals`), else
  every visual geom of the vendored menagerie source (contype = conaffinity = 0: no contact; density 0 and the
  pack's explicit inertials keep every mass), added to the body of the same name, named
  ``<body>_vis<k>_<material>`` so look rules can match on the material, coloured by the material's rgba; a geom
  already in the tree is not added twice.
"""

from __future__ import annotations

import hashlib
import logging
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from alloy.arms.specs import ArmSpec, assets_dir

log = logging.getLogger("alloy.arms")

REF_ATTRS = ("mesh", "material", "joint", "joint1", "joint2", "body1", "body2", "geom1", "geom2", "site", "target",
             "tendon", "objname", "body", "texture")  # fmt: skip
_OPTION_FIELDS = ("timestep", "impratio", "tolerance", "ls_tolerance", "noslip_tolerance", "ccd_tolerance",
                  "gravity", "integrator", "cone", "jacobian", "solver", "iterations", "ls_iterations",
                  "noslip_iterations", "ccd_iterations", "disableflags", "enableflags")  # fmt: skip


class ArmModelError(RuntimeError):
    """The pack or the vendored render meshes are missing or do not match the spec."""


def menagerie_dir() -> Path:
    return assets_dir() / "menagerie"


def _f(v) -> str:
    return " ".join(repr(float(x)) for x in v)


def pack_robot(spec: ArmSpec) -> ET.Element:
    """The pack's MJCF tree with only the robot (root body ``spec.root``), its options, actuators, equalities."""
    xml = spec.pack_xml()
    if not xml.is_file():
        raise ArmModelError(f"{spec.key}: pack {xml} not found")
    root = ET.parse(xml).getroot()
    wb = root.find("worldbody")
    if not any(b.tag == "body" and b.get("name") == spec.root for b in wb):
        raise ArmModelError(f"{spec.key}: root body {spec.root!r} not in {xml}")
    for child in [c for c in wb if not (c.tag == "body" and c.get("name") == spec.root)]:
        wb.remove(child)
    for kf in root.findall("keyframe"):
        root.remove(kf)
    bodies = {b.get("name") for b in wb.iter("body")}
    for contact in root.findall("contact"):
        gone = [e for e in contact
                if e.get("body1", spec.root) not in bodies or e.get("body2", spec.root) not in bodies]
        for e in gone:
            contact.remove(e)
    used = {g.get("mesh") for g in wb.iter("geom") if g.get("mesh")}
    meshdir = root.find("compiler").attrib.pop("meshdir", "")
    for asset in root.findall("asset"):
        for m in list(asset):
            if m.tag == "mesh" and m.get("name") not in used:
                asset.remove(m)
            elif m.tag == "mesh":
                m.set("file", str((xml.parent / meshdir / m.get("file")).resolve()))
    _servo(root, spec)
    return root


def _servo(root: ET.Element, spec: ArmSpec) -> None:
    o = spec.gripper.servo
    if not o:
        return
    act = next((a for a in root.find("actuator") if a.get("name") == spec.gripper.actuator), None)
    if act is None:
        raise ArmModelError(f"{spec.key}: gripper actuator {spec.gripper.actuator!r} missing")
    act.set("gainprm", _f([o["kp"]] + [0.0] * 9))
    act.set("biasprm", _f([0.0, -o["kp"], -o["kv"]] + [0.0] * 7))
    act.set("ctrlrange", _f(o["ctrl"]))
    act.set("forcerange", _f(o["force"]))
    act.set("ctrllimited", "true")
    act.set("forcelimited", "true")


def add_visuals(root: ET.Element, spec: ArmSpec) -> int:
    """Add the pack's visual layer, or else menagerie's visual geoms (vendored copy), to the bodies of the same name;
    geoms already in the tree are skipped. Returns how many were added."""
    import mujoco

    from alloy.scene.robot_visuals import attach_mjcf, has_visuals

    if has_visuals(spec.pack_xml()):
        return attach_mjcf(root, spec.pack_xml())
    bodies = {b.get("name"): b for b in root.find("worldbody").iter("body")}
    asset = root.find("asset")
    added = 0
    for rel, prefix in spec.visuals:
        src_xml = menagerie_dir() / rel
        if not src_xml.is_file():
            raise ArmModelError(f"{spec.key}: render meshes {src_xml} missing")
        src = mujoco.MjSpec.from_file(str(src_xml))   # parse only: resolves menagerie's default classes
        base = src_xml.parent / src.meshdir
        meshes = {m.name: m for m in src.meshes}
        rgba = {m.name: list(m.rgba) for m in src.materials}
        for body in src.bodies:
            vis = [g for g in body.geoms if g.contype == 0 and g.conaffinity == 0 and g.meshname]
            if not vis:
                continue
            target = bodies.get(prefix + body.name)
            if target is None:
                raise ArmModelError(f"{spec.key}: pack has no body {prefix + body.name!r} for {rel} visuals")
            for k, g in enumerate(vis):
                added += _visual_geom(asset, target, g, meshes[g.meshname], base, rgba, prefix + body.name, k)
    return added


def _visual_geom(asset, target, g, mesh, base: Path, rgba: dict, body: str, k: int) -> int:
    material = (g.material or "plain").replace(" ", "_")
    if target.find(f"geom[@name='{body}_vis{k}_{material}']") is not None:
        return 0
    name = f"vis_{body}_{mesh.name}"
    if asset.find(f"mesh[@name='{name}']") is None:
        attrs = {"name": name, "file": str((base / mesh.file).resolve())}
        if list(mesh.scale) != [1.0, 1.0, 1.0]:
            attrs["scale"] = _f(mesh.scale)
        ET.SubElement(asset, "mesh", attrs)
    ET.SubElement(target, "geom", {
        "name": f"{body}_vis{k}_{material}", "type": "mesh", "mesh": name, "pos": _f(g.pos), "quat": _f(g.quat),
        "contype": "0", "conaffinity": "0", "group": "2", "density": "0",
        "rgba": _f(rgba.get(g.material, list(g.rgba)))})  # fmt: skip
    return 1


def _prefix(root: ET.Element, prefix: str) -> None:
    """Prefix every name and every reference to one (the packs have no default classes)."""
    names = {el.get("name") for el in root.iter() if el.get("name") and el.tag != "mujoco"}
    for el in root.iter():
        if el.tag == "mujoco":
            continue
        if el.get("name"):
            el.set("name", prefix + el.get("name"))
        for a in REF_ATTRS:
            if el.get(a) in names:
                el.set(a, prefix + el.get(a))


def robot_xml(spec: ArmSpec, prefix: str = "", visuals: bool = True) -> str:
    """MJCF string of one arm (robot only). ``prefix`` goes in front of every name (``left_``); the model is named
    ``sc6_<prefix><key>`` so Newton labels stay unique when the same arm is added twice. Cached per process."""
    key = (spec.key, prefix, visuals)
    if key not in _XML_CACHE:
        _XML_CACHE[key] = _build_xml(spec, prefix, visuals)
    return _XML_CACHE[key]


_XML_CACHE: dict[tuple[str, str, bool], str] = {}


def _build_xml(spec: ArmSpec, prefix: str, visuals: bool) -> str:
    import mujoco

    root = pack_robot(spec)
    if visuals:
        log.debug("%s: %d render geoms", spec.key, add_visuals(root, spec))
    if prefix:
        _prefix(root, prefix)
    root.set("model", f"sc6_{prefix}{spec.key}")
    text = ET.tostring(root, encoding="unicode")
    mujoco.MjModel.from_xml_string(text)   # compiles: every mesh loads, every reference resolves
    return text


def cache_dir() -> Path:
    """Where ``robot_file`` writes (machine-specific: the XML holds this checkout's absolute mesh paths)."""
    return Path(os.environ.get("ALLOY_SC6_CACHE", Path.home() / ".cache" / "alloy" / "sc6"))


def robot_file(spec: ArmSpec, prefix: str = "", visuals: bool = True) -> Path:
    """``robot_xml`` as a file (content-addressed in ``cache_dir()``), for a ``RobotRef(path=..., kind="mjcf")``."""
    text = robot_xml(spec, prefix, visuals)
    digest = hashlib.sha256(text.encode()).hexdigest()[:12]
    path = cache_dir() / f"sc6_{prefix}{spec.key}{'' if visuals else '_novis'}-{digest}.xml"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(text)
        tmp.replace(path)
    return path


def add_arm(builder, spec: ArmSpec, base=(0.0, 0.0, 0.0), *, prefix: str = "", visuals: bool = True,
            **mjcf_kw) -> None:  # fmt: skip
    """Add one arm to a Newton ``builder`` at ``base`` ``(x, y, z)`` or ``(x, y, z, qx, qy, qz, qw)``, with its MJCF
    actuators in MJCF order (``ctrl_direct``). Extra keywords go to ``add_mjcf`` (e.g. ``parse_mujoco_options``).
    Register MuJoCo's custom attributes on the builder first (``SolverMuJoCo.register_custom_attributes``)."""
    import warp as wp

    pose = tuple(base) + ((0.0, 0.0, 0.0, 1.0) if len(base) == 3 else ())
    kw = {"ctrl_direct": True, "parse_visuals": visuals} | mjcf_kw
    builder.add_mjcf(robot_xml(spec, prefix, visuals), xform=wp.transform(pose[:3], pose[3:]), **kw)


__all__ = ["ArmModelError", "add_arm", "add_visuals", "cache_dir", "menagerie_dir", "pack_robot", "robot_file",
           "robot_xml"]
