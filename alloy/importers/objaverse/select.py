# SPDX-License-Identifier: Apache-2.0
"""Metadata stage: which Objaverse objects are worth downloading.

Input rows are Objaverse 1.0 annotations joined with LVIS labels (one dict per object, see :func:`row_from_annotation`).
An object passes when its licence is accepted, an LVIS label maps to a robot-relevant category, it is not animated or
age-restricted, it has textures, a sane face count and download size, and no name/tag/description term suggests a
brand, a game rip or a non-physical item. Passing objects are ranked by :func:`meta_score` and capped per category
(``cap x oversample``, the mesh stage rejects some more).

Example:
    >>> from alloy.importers.objaverse.select import screen
    >>> row = {"uid": "u1", "license": "by", "lvis": ["mug"], "name": "Ceramic mug", "tags": ["scan"],
    ...        "faceCount": 20000, "glb_textures": 2, "glb_size": 4e6, "animationCount": 0, "likeCount": 3,
    ...        "viewCount": 200, "user_display": "Ann"}
    >>> screen(row).category
    'mug'
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from alloy.importers.objaverse.categories import Category, load_categories, lvis_index
from alloy.importers.objaverse.errors import ImportRejected
from alloy.importers.objaverse.licence import check_licence

#: Words that mark brands/IP (trademark risk), game rips or stylised, non-physical items (matched as whole words in
#: name, tags and description, case-insensitive).
BLOCK_TERMS = (
    "coca", "cola", "pepsi", "sprite", "fanta", "redbull", "red bull", "starbucks", "mcdonald", "mcdonalds",
    "heineken", "budweiser", "nike", "adidas", "apple inc", "iphone", "ipad", "macbook", "samsung", "nintendo",
    "pokemon", "minecraft", "fortnite", "roblox", "disney", "marvel", "star wars", "harry potter", "zelda", "mario",
    "skyrim", "fallout", "overwatch", "valorant", "genshin", "halo", "ripped", "rip", "fanart", "fan art",
    "cartoon", "toon", "stylized", "stylised", "anime", "chibi", "voxel", "kawaii", "emoji", "meme", "nsfw",
    "low poly", "lowpoly", "low-poly", "pixel", "papercraft", "kitbash", "set of", "collection", "pack", "bundle",
)
_BLOCK_RE = re.compile(r"\b(" + "|".join(re.escape(t) for t in BLOCK_TERMS) + r")\b", re.IGNORECASE)
_SCAN_RE = re.compile(r"\b(scan|scanned|3d ?scan|photogrammetry|realitycapture|reality capture|metashape|"
                      r"polycam|lidar)\b", re.IGNORECASE)
_PBR_RE = re.compile(r"\b(pbr|realistic|photoreal|photorealistic|substance)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Limits:
    """Metadata thresholds (defaults tuned on the LVIS subset; change them here, not in code)."""

    min_faces: int = 500
    max_faces: int = 400_000
    max_glb_bytes: int = 60_000_000
    min_textures: int = 1
    oversample: float = 2.0


@dataclass(frozen=True)
class Candidate:
    """An object that passed the metadata stage."""

    uid: str
    category: str
    group: str
    spdx: str
    score: float
    name: str
    author: str
    url: str
    flags: tuple[str, ...] = field(default_factory=tuple)

    def as_row(self) -> dict[str, Any]:
        """JSON row."""
        return asdict(self)


def _text(row: Mapping[str, Any]) -> str:
    tags = " ".join(str(t) for t in row.get("tags") or ())
    return f"{row.get('name') or ''} | {tags} | {row.get('description') or ''}"


def _category_of(row: Mapping[str, Any], idx: Mapping[str, Category]) -> Category:
    for label in row.get("lvis") or ():
        if label in idx:
            return idx[label]
    raise ImportRejected(str(row.get("uid")), "select", f"no robot-relevant category among LVIS {row.get('lvis')}")


def _check_counts(row: Mapping[str, Any], lim: Limits) -> None:
    uid = str(row.get("uid"))
    faces = int(row.get("faceCount") or row.get("glb_faces") or 0)
    if not lim.min_faces <= faces <= lim.max_faces:
        raise ImportRejected(uid, "select", f"{faces} faces outside [{lim.min_faces}, {lim.max_faces}]")
    if int(row.get("glb_textures") or 0) < lim.min_textures:
        raise ImportRejected(uid, "select", "no textures (real textures required)")
    if int(row.get("glb_size") or 0) > lim.max_glb_bytes:
        raise ImportRejected(uid, "select", f"GLB {row.get('glb_size')} B > {lim.max_glb_bytes} B")
    if int(row.get("animationCount") or 0) > 0:
        raise ImportRejected(uid, "select", "animated model")
    if row.get("isAgeRestricted"):
        raise ImportRejected(uid, "select", "age-restricted")


def meta_score(row: Mapping[str, Any]) -> tuple[float, tuple[str, ...]]:
    """Ranking score from metadata alone (higher is better) and the flags that contributed.

    likes and views (log), staff picks, scan/photogrammetry and PBR terms add; very coarse or very heavy meshes lose.

    Example:
        >>> meta_score({"likeCount": 9, "viewCount": 99, "faceCount": 20000, "name": "mug scan"})[1]
        ('scan',)
    """
    text = _text(row)
    flags: list[str] = []
    s = math.log10(1 + int(row.get("likeCount") or 0)) + 0.5 * math.log10(1 + int(row.get("viewCount") or 0))
    if row.get("staffpickedAt"):
        s += 1.0
        flags.append("staffpick")
    if _SCAN_RE.search(text):
        s += 0.8
        flags.append("scan")
    if _PBR_RE.search(text):
        s += 0.3
        flags.append("pbr")
    faces = int(row.get("faceCount") or 0)
    if faces < 2_000 or faces > 200_000:
        s -= 0.5
    return round(s, 4), tuple(flags)


def screen(row: Mapping[str, Any], cats: Mapping[str, Category] | None = None,
           lim: Limits = Limits()) -> Candidate:
    """Accept one annotation row as a :class:`Candidate`, or raise :class:`ImportRejected` saying why."""
    uid = str(row.get("uid"))
    verdict = check_licence(row.get("license"))
    if not verdict.accepted:
        raise ImportRejected(uid, "licence", verdict.reason)
    cat = _category_of(row, lvis_index(cats if cats is not None else load_categories()))
    _check_counts(row, lim)
    hit = _BLOCK_RE.search(_text(row))
    if hit:
        raise ImportRejected(uid, "select", f"blocked term {hit.group(0)!r} (brand, game rip or stylised item)")
    author = str(row.get("user_display") or row.get("user") or "").strip()
    if not author:
        raise ImportRejected(uid, "licence", "no author name; CC BY attribution needs one")
    score, flags = meta_score(row)
    url = str(row.get("viewerUrl") or row.get("uri") or "")
    return Candidate(uid, cat.name, cat.group, verdict.spdx, score, str(row.get("name") or ""), author, url, flags)


def select(rows: Iterable[Mapping[str, Any]], cats: Mapping[str, Category] | None = None,
           lim: Limits = Limits()) -> tuple[list[Candidate], list[dict[str, str]]]:
    """Screen every row, rank, and keep at most ``ceil(cap x oversample)`` per category.

    Returns (candidates sorted by category then score, reject rows). Over-cap objects are rejected with stage
    ``cap``.

    Example:
        >>> rows = [{"uid": "a", "license": "by-nc", "lvis": ["mug"]}]
        >>> select(rows)[1][0]["stage"]
        'licence'
    """
    cats = cats if cats is not None else load_categories()
    kept: dict[str, list[Candidate]] = {}
    rejects: list[dict[str, str]] = []
    for row in rows:
        try:
            c = screen(row, cats, lim)
        except ImportRejected as e:
            rejects.append(e.as_row())
            continue
        kept.setdefault(c.category, []).append(c)
    out: list[Candidate] = []
    for name in sorted(kept):
        ranked = sorted(kept[name], key=lambda c: (-c.score, c.uid))
        n = math.ceil(cats[name].cap * lim.oversample)
        out += ranked[:n]
        rejects += [ImportRejected(c.uid, "cap", f"{name}: rank {i + n + 1} > {n}").as_row()
                    for i, c in enumerate(ranked[n:])]
    return out, rejects
