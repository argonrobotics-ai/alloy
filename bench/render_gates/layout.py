# SPDX-License-Identifier: Apache-2.0
"""Image layout of a render-gate run: which files are references, control seeds and engines under test.

A gate root holds ``<scene>/<frame>/<engine>/<W>x<H>_<spp>spp_<on|off>.png``:

- ``cycles``: the reference engine (Cycles, seed 0); its ``<res>_4096spp_on.png`` is the reference image;
- ``cycles_s1``, ``cycles_s2``, ...: control seeds of the same Cycles setup (the yardstick for noise);
- every other directory is an engine under test.

Example:
    >>> from bench.render_gates.layout import parse_name
    >>> parse_name("128x128_16spp_on.png")
    ('128x128', 16, 'on')
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

NAME = re.compile(r"^(\d+x\d+)_(\d+)spp_(on|off|den|noden|optix|oidn|none)\.png$")
DEN_ON = frozenset({"on", "den", "optix", "oidn"})
REF_ENGINE = "cycles"
REF_SPP = 4096
S1 = "cycles_s1"
CONTROL = re.compile(r"^cycles_s\d+$")


@dataclass(frozen=True)
class Img:
    """One image file in the layout: its path, size label ``WxH``, samples per pixel and denoise state."""

    path: Path
    res: str
    spp: int
    den: str  # "on" | "off"


def parse_name(name: str) -> tuple[str, int, str] | None:
    """Parse ``<W>x<H>_<spp>spp_<state>.png`` into ``(res, spp, "on"|"off")``; None if the name does not match.

    Example:
        >>> parse_name("640x480_5spp_oidn.png")
        ('640x480', 5, 'on')
        >>> parse_name("notes.txt") is None
        True
    """
    m = NAME.match(name)
    if not m:
        return None
    return m[1], int(m[2]), "on" if m[3] in DEN_ON else "off"


def image_name(res: str, spp: int, den: str) -> str:
    """The file name the gate expects for one image.

    Example:
        >>> image_name("1280x720", 12, "on")
        '1280x720_12spp_on.png'
    """
    return f"{res}_{spp}spp_{den}.png"


def is_control(engine: str) -> bool:
    """True for a Cycles control seed directory (``cycles_s1``, ``cycles_s2``, ...).

    Example:
        >>> is_control("cycles_s3"), is_control("cycles"), is_control("ours")
        (True, False, False)
    """
    return bool(CONTROL.match(engine))


def list_images(engine_dir: Path) -> list[Img]:
    """All gate images in one engine directory, sorted by file name; other files are ignored.

    Example:
        >>> list_images(Path("/nonexistent"))
        []
    """
    if not engine_dir.is_dir():
        return []
    out = []
    for p in sorted(engine_dir.glob("*.png")):
        parsed = parse_name(p.name)
        if parsed:
            out.append(Img(p, *parsed))
    return out


def collect(root: Path, ref_engine: str = REF_ENGINE) -> dict[tuple[str, str], dict[str, list[Img]]]:
    """Map ``(scene, frame)`` to ``{engine: [Img]}`` for every frame directory that has the reference engine.

    Raises FileNotFoundError naming the root when it does not exist.

    Example:
        >>> collect(Path("."), "no_such_engine")
        {}
    """
    if not root.is_dir():
        raise FileNotFoundError(f"gate root {root} does not exist; pass the folder that holds <scene>/<frame>/<engine>/")
    frames: dict[tuple[str, str], dict[str, list[Img]]] = {}
    for scene_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for frame_dir in sorted(p for p in scene_dir.iterdir() if p.is_dir()):
            engines = {e.name: list_images(e) for e in sorted(frame_dir.iterdir()) if e.is_dir()}
            if ref_engine in engines:
                frames[(scene_dir.name, frame_dir.name)] = engines
    return frames


def reference_path(root: Path, scene: str, frame: str, res: str, ref_engine: str = REF_ENGINE) -> Path:
    """Path of the reference image for one scene, frame and size.

    Example:
        >>> reference_path(Path("g"), "c3", "420", "128x128").as_posix()
        'g/c3/420/cycles/128x128_4096spp_on.png'
    """
    return root / scene / frame / ref_engine / image_name(res, REF_SPP, "on")
