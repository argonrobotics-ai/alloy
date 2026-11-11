# SPDX-License-Identifier: Apache-2.0
"""Licence filter and attribution for Objaverse 1.0 objects (Sketchfab uploads).

Objaverse 1.0 (Deitke et al. 2023) is distributed under ODC-By 1.0, and each object keeps the licence its Sketchfab
author chose. Only licences that allow commercial use and modification without share-alike are accepted:
CC0 1.0 and CC BY 4.0. Everything else (BY-SA, BY-ND, any NC, Sketchfab Standard/Editorial) is rejected.

Example:
    >>> from alloy.importers.objaverse.licence import check_licence, attribution
    >>> check_licence("by").spdx
    'CC-BY-4.0'
    >>> check_licence("by-nc").accepted
    False
    >>> attribution({"name": "Mug", "user_display": "Ann", "uri": "https://sketchfab.com/3d-models/x",
    ...              "license": "by"})["text"]
    '"Mug" by Ann (https://sketchfab.com/3d-models/x), CC-BY-4.0; rescaled, re-oriented, repaired and decimated for Alloy'
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alloy.importers.objaverse.errors import ImportRejected

#: Sketchfab licence slug (Objaverse ``license`` field) -> SPDX id, for the accepted licences only.
ACCEPTED: dict[str, str] = {
    "cc0": "CC0-1.0",
    "by": "CC-BY-4.0",
}

#: Known rejected slugs and why (anything not listed in ACCEPTED or here is rejected as "unknown licence").
REJECTED: dict[str, str] = {
    "by-sa": "share-alike (CC BY-SA) is not on the accepted list (CC0, CC BY, MIT, Apache-2.0, BSD)",
    "by-nd": "no-derivatives (CC BY-ND): Alloy rescales and repairs every mesh",
    "by-nc": "non-commercial (CC BY-NC)",
    "by-nc-sa": "non-commercial (CC BY-NC-SA)",
    "by-nc-nd": "non-commercial (CC BY-NC-ND)",
    "st": "Sketchfab Standard licence (not redistributable)",
    "ed": "Sketchfab Editorial licence (not redistributable)",
}

#: What every imported object changed relative to the upload (CC BY asks to indicate changes).
CHANGES = "rescaled, re-oriented, repaired and decimated for Alloy"

DATASET_CREDIT = ("Objaverse 1.0 (Deitke et al., CVPR 2023, https://objaverse.allenai.org), ODC-By 1.0; "
                  "LVIS category labels from the same release")


@dataclass(frozen=True)
class LicenceVerdict:
    """Result of :func:`check_licence`: ``accepted``, the SPDX id (or ``""``) and the reason when rejected."""

    slug: str
    accepted: bool
    spdx: str
    reason: str = ""


def _normalise(slug: str | None) -> str:
    s = (slug or "").strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {"cc-by": "by", "cc-by-4.0": "by", "cc0-1.0": "cc0", "cc-0": "cc0", "public-domain": "cc0"}
    return aliases.get(s, s)


def check_licence(slug: str | None) -> LicenceVerdict:
    """Classify an Objaverse/Sketchfab licence slug.

    Example:
        >>> check_licence("CC0").accepted, check_licence("by-sa").reason.split(" (")[0]
        (True, 'share-alike')
    """
    s = _normalise(slug)
    if s in ACCEPTED:
        return LicenceVerdict(s, True, ACCEPTED[s])
    reason = REJECTED.get(s, f"unknown licence {slug!r}: only {sorted(ACCEPTED)} are accepted")
    return LicenceVerdict(s, False, "", reason)


def require_licence(slug: str | None, uid: str = "?") -> str:
    """The SPDX id of an accepted licence; raises :class:`ImportRejected` (stage ``licence``) otherwise.

    Example:
        >>> require_licence("by", "abc")
        'CC-BY-4.0'
    """
    v = check_licence(slug)
    if not v.accepted:
        raise ImportRejected(uid, "licence", v.reason)
    return v.spdx


def attribution(meta: dict[str, Any]) -> dict[str, str]:
    """Attribution record for one object: title, author, source URL, SPDX licence, changes and a one-line ``text``.

    ``meta`` is an Objaverse annotation row (keys ``name``, ``user_display`` or ``user``, ``uri`` or ``viewerUrl``,
    ``license``). Raises :class:`ImportRejected` when the licence is not accepted or the author is unknown
    (CC BY cannot be honoured without an author).

    Example:
        >>> attribution({"name": "Cup", "user": "bob", "viewerUrl": "https://sketchfab.com/m/1", "license": "cc0"})["spdx"]
        'CC0-1.0'
    """
    uid = str(meta.get("uid", "?"))
    spdx = require_licence(meta.get("license"), uid)
    author = (meta.get("user_display") or meta.get("user") or "").strip()
    if not author:
        raise ImportRejected(uid, "licence", "no author name in the annotation; CC BY needs one (skip the object)")
    title = (meta.get("name") or "untitled").strip()
    url = meta.get("viewerUrl") or meta.get("uri") or f"https://sketchfab.com/3d-models/{uid}"
    text = f'"{title}" by {author} ({url}), {spdx}; {CHANGES}'
    return {"title": title, "author": author, "url": url, "spdx": spdx, "changes": CHANGES, "text": text}
