# SPDX-License-Identifier: Apache-2.0
"""Checks and preparation of a robot file before Newton's importer reads it; the scene assembler calls it.

:func:`robot_source` gives what to hand the importer for one robot file. An unexpanded xacro URDF is refused with
the command that expands it; a URDF whose massless root link carries the robot on a ``floating`` joint becomes a
free base (any other floating layout is refused, never guessed); a USD stage in other length units is read in metres
(:mod:`alloy.io.usd_units`) and one OpenUSD cannot open is refused; URDF mesh paths (``file://``, ``package://``,
``model://``) are resolved and a missing mesh is refused; an MJCF's ``settotalmass`` is applied and
``coordinate="global"`` refused (:mod:`alloy.io.robot_mjcf`). :func:`require_joint` refuses a scene whose robot
files hold only static geometry while nothing else in it has a joint (MuJoCo's converter needs one), with what to
add.

Every refusal is a :class:`RobotFileError` whose ``fix`` says what to change; ``alloy.build`` shows it as a
``BuildError`` with that fix.

Example::

    from alloy._vendor import newton
    from alloy.io.robot_file import robot_source
    sub = newton.ModelBuilder()
    source, options = robot_source("urdf", "robot.urdf")
    sub.add_urdf(source, **options)
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from alloy.io.robot_mjcf import RobotFileError, mjcf_source

log = logging.getLogger("alloy.io")
XACRO = re.compile(r"\$\(|\$\{|<xacro:")
URI = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")       # package://, file:, model://, http: ...


def check_xacro(text: str, name: str) -> None:
    """Refuse a URDF that still has xacro macros or substitutions (``$(find pkg)``, ``${prop}``, ``<xacro:...>``)."""
    found = XACRO.search(text)
    if found is None:
        return
    line = text.count("\n", 0, found.start()) + 1
    snippet = text[found.start():found.start() + 24]
    raise RobotFileError(f"{name} is an unexpanded xacro file (line {line}: {snippet!r}): its macros and $(...) "
                         "substitutions are not numbers yet",
                         f"expand it with ROS xacro, then load the result: xacro {name} > robot.urdf")


def _is_empty(link: ET.Element) -> bool:
    mass = link.find("inertial/mass")
    heavy = mass is not None and float(mass.get("value", 0.0) or 0.0) > 0.0
    return not heavy and link.find("collision") is None and link.find("visual") is None


def floating_base(root: ET.Element, name: str) -> ET.Element | None:
    """The ``floating`` joint of a URDF when it joins a massless, shapeless root link to the rest of the robot (the
    usual way to write a free base); None when the file has no floating joint. Any other layout raises."""
    joints = root.findall("joint")
    floats = [j for j in joints if j.get("type") == "floating"]
    if not floats:
        return None
    link_of = lambda j, tag: j.find(tag).get("link") if j.find(tag) is not None else None  # noqa: E731
    children = {link_of(j, "child") for j in joints}
    links = {lk.get("name"): lk for lk in root.findall("link")}
    roots = [n for n in links if n not in children]
    j = floats[0]
    ok = (len(floats) == 1 and len(roots) == 1 and link_of(j, "parent") == roots[0] and _is_empty(links[roots[0]])
          and sum(1 for k in joints if link_of(k, "parent") == roots[0]) == 1)
    if not ok:
        raise RobotFileError(f"{name}: floating joint {j.get('name')!r} is not on a massless root link that carries "
                             "only it; a free joint must join the robot's base to the world",
                             "make the floating joint the only joint of an empty root link (<link name=\"world\"/>), "
                             "or remove it and convert the robot to MJCF with <freejoint/> on the body that floats")
    return j


def _pose_of(joint: ET.Element):
    from alloy.io.urdf_read import origin_pose  # noqa: PLC0415

    return origin_pose(joint)


def resolve_meshes(root: ET.Element, folder: Path, name: str) -> bool:
    """Make every ``<mesh filename>`` an absolute path of an existing file (``file://``, ``package://`` and
    ``model://`` resolved from the URDF's folder and its parents). Returns True if a URI was resolved; a mesh that
    is not found raises (the importer would drop the link's shape with a warning only)."""
    from alloy.io.errors import ImportFileError  # noqa: PLC0415
    from alloy.io.urdf_read import resolve_mesh  # noqa: PLC0415

    uri = False
    for mesh in root.iter("mesh"):
        ref = mesh.get("filename", "")
        if not ref:
            continue
        try:
            found = resolve_mesh(ref, folder)
        except ImportFileError as exc:
            raise RobotFileError(f"{name}: {exc.message}; the link would load without this shape",
                                 "fix the mesh path in the URDF: a path relative to the URDF, an absolute path, or "
                                 "package://<pkg>/... with the package folder beside or above the URDF") from exc
        uri = uri or bool(URI.match(ref))
        mesh.set("filename", str(found.resolve()))
    return uri


def urdf_source(path: str | Path) -> tuple[str, dict]:
    """The URDF to import and Newton ``add_urdf`` options: the path itself, or for a free base the document without
    its empty root link (mesh paths made absolute), ``floating=True`` and the joint origin as the start pose."""
    import warp as wp  # noqa: PLC0415

    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    check_xacro(text, path.name)
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise RobotFileError(f"{path.name} is not well-formed XML: {exc}", "fix the line the message names") from exc
    joint = floating_base(root, path.name)
    uri = resolve_meshes(root, path.parent, path.name)
    if joint is None:
        return (ET.tostring(root, encoding="unicode"), {}) if uri else (str(path), {})
    pose = _pose_of(joint)
    root.remove(next(lk for lk in root.findall("link") if lk.get("name") == joint.find("parent").get("link")))
    root.remove(joint)
    log.info("%s: floating joint %s on an empty root link read as a free base", path.name, joint.get("name"))
    return ET.tostring(root, encoding="unicode"), {"floating": True,
                                                    "xform": wp.transform(wp.vec3(*pose[:3]), wp.quat(*pose[3:]))}


def robot_source(kind: str, path: str | Path) -> tuple[object, dict]:
    """``(source, options)`` for Newton's importer of a robot file of ``kind`` ("mjcf", "urdf", "usd")."""
    if kind == "urdf" and Path(path).is_file():
        return urdf_source(path)
    if kind == "mjcf" and Path(path).is_file():
        return mjcf_source(path), {}
    if kind == "usd" and Path(path).is_file():
        try:
            from alloy.io.usd_units import metre_source  # noqa: PLC0415

            return metre_source(path), {}
        except ImportError:                 # no USD packages: the importer raises with the install command
            return str(path), {}
        except Exception as exc:  # noqa: BLE001 - OpenUSD's own error type; its message names the line
            raise RobotFileError(f"{Path(path).name} is not a USD file OpenUSD can open: {str(exc).strip()[-400:]}",
                                 "fix the line the message names (usdchecker or usdcat shows it), or export the "
                                 "file again from the tool that wrote it") from exc
    return str(path), {}


def require_joint(joint_count: int, robot_files: list[str]) -> None:
    """Refuse a scene whose only would-be moving parts are robot files without a single joint or free body."""
    if joint_count or not robot_files:
        return
    names = ", ".join(Path(p).name for p in robot_files)
    raise RobotFileError(f"nothing in this scene can move: {names} has only static geometry (no joint, no free body) "
                         "and the scene has no other moving object",
                         "add an object that moves (scene.add_box(...), or a body with a joint in the file); the "
                         "file's static geometry then loads beside it. To reuse the geometry as cards: "
                         "alloy.io.import_file(path, folder)")
