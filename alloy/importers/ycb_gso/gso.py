# SPDX-License-Identifier: Apache-2.0
"""Google Scanned Objects (Downs et al., ICRA 2022): listing, archive layout and metadata.

Every model is published by Google Research on Gazebo Fuel under CC BY 4.0 (``license_name`` in the Fuel API and the
``legal`` block of each ``metadata.pbtxt``). Archive layout: ``meshes/model.obj`` (metres, +Z up),
``meshes/model.mtl`` (``map_Kd texture.png``, resolved by Gazebo from ``materials/textures/texture.png``),
``metadata.pbtxt`` (name, description, brand, GTIN, categories). No mass is published, so every GSO card gets an
ESTIMATE density by category and a row in the sourcing queue (with the GTIN, which identifies the retail product).
"""

from __future__ import annotations

import codecs
import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

FUEL_API = "https://fuel.gazebosim.org/1.0/GoogleResearch/models"
LICENCE = "CC-BY-4.0"
LICENCE_NAME = "Creative Commons Attribution 4.0 International"
ATTRIBUTION = ("Google Scanned Objects (Downs et al., ICRA 2022), Google LLC, via Gazebo Fuel (GoogleResearch), "
               "licensed CC BY 4.0; processed by Alloy (watertight remesh, CoACD hulls)")
MESH_MEMBER = "meshes/model.obj"
METADATA = "metadata.pbtxt"


@dataclass(frozen=True)
class GsoMeta:
    """What ``metadata.pbtxt`` says about one model."""

    name: str
    description: str = ""
    brand: str = ""
    gtin: str = ""
    categories: tuple[str, ...] = ()
    license: str = ""
    extra: dict = field(default_factory=dict)


def plain_card_name(model: str) -> str:
    """``gso_`` + the model name lower-cased, non-alphanumerics folded to ``_`` (batch outputs before relabels).

    Example::

        plain_card_name("Vtech_Cruise_Learn_Car_25_Years")   # "gso_vtech_cruise_learn_car_25_years"
    """
    return "gso_" + re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")


def card_name(model: str) -> str:
    """Alloy card name for a GSO model: ``plain_card_name``, or the relabel's name when the upstream title names a
    different object than the mesh shows (``relabels.RELABELS``).

    Example::

        card_name("Ortho_Forward_Facing")   # "gso_plush_giraffe"
    """
    from alloy.importers.ycb_gso.relabels import RELABELS  # noqa: PLC0415 - relabels imports estimates

    rel = RELABELS.get(model)
    return rel.card if rel is not None else plain_card_name(model)


def archive_url(model: str) -> str:
    """Download URL of a model's zip (version 1, the only one published); the name is percent-encoded.

    Example::

        archive_url("Pokémon_X")   # ".../Pok%C3%A9mon_X/1/Pok%C3%A9mon_X.zip"
    """
    q = urllib.parse.quote(model)
    return f"{FUEL_API}/{q}/1/{q}.zip"


def fetch_listing(per_page: int = 100, max_pages: int = 50) -> list[dict]:
    """Every Google Research model on Fuel (name, licence, file size, ...), following the API pages.

    Example::

        names = [m["name"] for m in fetch_listing()]   # 1,033 models (2026-11)
    """
    out: list[dict] = []
    for page in range(1, max_pages + 1):
        with urllib.request.urlopen(f"{FUEL_API}?per_page={per_page}&page={page}", timeout=60) as resp:
            rows = json.load(resp)
        out += rows
        if len(rows) < per_page:
            break
    return out


_PAIR = re.compile(r'^\s*(\w+)\s*:\s*"((?:[^"\\]|\\.)*)"\s*$')


def _unescape(s: str) -> str:
    """Text-format protobuf string escapes (``\\n``, ``\\"``, octal ``\\303\\251`` = UTF-8 bytes of "é")."""
    return codecs.escape_decode(s.encode("utf-8"))[0].decode("utf-8", "replace")


def _blocks(text: str) -> list[tuple[str, dict[str, str]]]:
    """Flat parse of a text-format protobuf: top-level ``key: "v"`` (block "") and one level of ``name { ... }``."""
    out: list[tuple[str, dict[str, str]]] = [("", {})]
    stack: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if s.endswith("{"):
            stack.append(s[:-1].strip())
            if len(stack) == 1:
                out.append((stack[0], {}))
            continue
        if s == "}":
            stack.pop()
            continue
        m = _PAIR.match(s)
        if m and len(stack) <= 1:
            out[-1 if stack else 0][1].setdefault(m.group(1), _unescape(m.group(2)))
    return out


def parse_metadata(text: str) -> GsoMeta:
    """Parse ``metadata.pbtxt`` (name, description, brand/gtin annotations, categories, licence).

    Example::

        meta = parse_metadata(Path("metadata.pbtxt").read_text())
        meta.categories   # ("Toys",)
    """
    top: dict[str, str] = {}
    ann: dict[str, str] = {}
    cats: list[str] = []
    lic = ""
    for name, kv in _blocks(text):
        if name == "":
            top = kv
        elif name == "annotations" and "key" in kv:
            ann[kv["key"]] = kv.get("value", "")
        elif name == "categories" and kv.get("first"):
            cats.append(kv["first"])
        elif name == "legal":
            lic = kv.get("license", "")
    return GsoMeta(name=top.get("name", ""), description=top.get("description", ""), brand=ann.get("brand", ""),
                   gtin=ann.get("gtin", ""), categories=tuple(cats), license=lic,
                   extra={k: v for k, v in ann.items() if k not in ("brand", "gtin")})


def read_metadata(path: Path) -> GsoMeta:
    """``parse_metadata`` of a file.

    Example::

        read_metadata(Path("out/gso_x/metadata.pbtxt"))
    """
    return parse_metadata(Path(path).read_text(encoding="utf-8"))
