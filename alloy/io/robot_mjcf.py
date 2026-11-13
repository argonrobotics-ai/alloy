# SPDX-License-Identifier: Apache-2.0
"""MJCF compiler settings Newton's importer does not apply, handled before it reads the file.

- ``<compiler coordinate="global"/>``: MuJoCo 3 refuses it; so does :func:`mjcf_source`, with the fix (the
  importer read the world poses as parent-relative offsets and stacked them).
- ``<compiler settotalmass="M"/>``: MuJoCo scales every body's mass and inertia so the model weighs M (DeepMind
  Control and Gym models use it). :func:`mjcf_source` writes a copy with the scaled values as explicit
  ``<inertial>`` elements, taken from MuJoCo's own compiled model, and without the attribute.

Example::

    from alloy.io.robot_mjcf import mjcf_source
    path = mjcf_source("hopper.xml")        # the file itself, or a cached copy with MuJoCo's scaled inertials
"""

from __future__ import annotations

import hashlib
import logging
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from alloy.core.errors import AssemblyError

log = logging.getLogger("alloy.io")


class RobotFileError(AssemblyError):
    """A robot file cannot be loaded as it is. ``fix`` is what to run or change."""

    def __init__(self, message: str, fix: str) -> None:
        super().__init__(message)
        self.fix = fix


def cache_dir() -> Path:
    """Folder for rewritten robot files: ``$ALLOY_CACHE/io`` (default ``~/.cache/alloy/io``)."""
    return Path(os.environ.get("ALLOY_CACHE", Path.home() / ".cache" / "alloy")) / "io"


def compiler_settings(root: ET.Element) -> dict[str, str]:
    """Attributes of every ``<compiler>`` element merged in document order (later ones win, as in MuJoCo)."""
    out: dict[str, str] = {}
    for comp in root.iter("compiler"):
        out.update(comp.attrib)
    return out


def _with_inertials(root: ET.Element, model) -> None:
    """Give every body the compiled model's mass, centre of mass and inertia; drop ``settotalmass``."""
    from alloy.io.usd_text import num  # noqa: PLC0415

    txt = lambda v: " ".join(num(float(x)) for x in v)  # noqa: E731
    bodies = [b for w in root.findall("worldbody") for b in w.iter("body")]
    if len(bodies) != model.nbody - 1:
        raise ValueError(f"the file has {len(bodies)} <body> elements, MuJoCo compiled {model.nbody - 1}")
    for i, body in enumerate(bodies, start=1):
        for old in body.findall("inertial"):
            body.remove(old)
        if float(model.body_mass[i]) > 0.0:
            body.insert(0, ET.Element("inertial", pos=txt(model.body_ipos[i]), quat=txt(model.body_iquat[i]),
                                      mass=num(float(model.body_mass[i])), diaginertia=txt(model.body_inertia[i])))
    for comp in root.iter("compiler"):
        comp.attrib.pop("settotalmass", None)


def _scaled_copy(path: Path, root: ET.Element, total: str) -> Path:
    import mujoco  # noqa: PLC0415

    from alloy.io.mjcf_prune import _absolute_files  # noqa: PLC0415

    fix = ("scale the masses in the file yourself (multiply each mass and density so they sum to the total) and "
           "remove settotalmass")
    try:
        _with_inertials(root, mujoco.MjModel.from_xml_path(str(path)))
    except Exception as exc:  # noqa: BLE001 - MuJoCo's message plus the fix
        raise RobotFileError(f"{path.name}: <compiler settotalmass={total!r}> could not be applied ({exc})",
                             fix) from exc
    _absolute_files(root, path.parent)
    text = ET.tostring(root, encoding="unicode")
    out = cache_dir() / f"{path.stem}-{hashlib.sha256(text.encode()).hexdigest()[:12]}.xml"
    if not out.is_file():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(out)
    log.info("%s: settotalmass=%s applied (MuJoCo's scaled inertials written to %s)", path.name, total, out)
    return out


def mjcf_source(path: str | Path) -> str:
    """The MJCF file to hand Newton's importer: ``path``, or a cached copy with ``settotalmass`` applied. Raises
    :class:`RobotFileError` for ``coordinate="global"``."""
    from alloy.io.mjcf_include import read_mjcf  # noqa: PLC0415

    path = Path(path)
    try:
        root = read_mjcf(path)
    except (ValueError, ET.ParseError, OSError):
        return str(path)                            # the importer reports what is wrong with the file
    comp = compiler_settings(root)
    if comp.get("coordinate", "local") == "global":
        raise RobotFileError(f'{path.name} uses <compiler coordinate="global"/>, which MuJoCo 3 no longer reads: '
                             "its body positions are world positions, not offsets from the parent body",
                             'rewrite the body, geom and joint positions relative to their parent body and remove '
                             'coordinate="global" (MuJoCo 2.x can convert the file: load it and save the XML)')
    total = comp.get("settotalmass")
    if total is None or float(total) <= 0.0:
        return str(path)
    return str(_scaled_copy(path, root, total))
