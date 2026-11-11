# SPDX-License-Identifier: Apache-2.0
"""``fetch`` for the ``alloy`` command line and the MCP tool server (an ``alloy.cli.spec.Op``; JSON in and out).

Example::

    alloy fetch ycb_025_mug objv_mug_1a2b3c4d          # two objects
    alloy fetch --library robocasa --category blenders  # a category
    alloy fetch --list --library objaverse              # what there is, nothing downloaded
"""
from __future__ import annotations

from typing import Any

from alloy.cli.spec import Op, Param


def op_fetch(session: Any, a: dict[str, Any]) -> dict[str, Any]:
    """Fetch objects into the cache, or list them (``list``); returns counts and cache roots."""
    from alloy.assets.hub.fetch import fetch, list_objects  # noqa: PLC0415

    if a.get("list"):
        return list_objects(a.get("library"), a.get("category"), a.get("repo"))
    return fetch(a.get("names") or None, a.get("library"), a.get("category"), a.get("repo"),
                 a.get("workers") or 8)


OPS: tuple[Op, ...] = (Op(
    "fetch", "Download objects of Alloy's object library (cards, meshes, textures, thumbnails) from the Hugging Face "
             "Hub into the local cache.",
    (Param("names", "strings", "object names (alloy fetch --list shows them)", positional=True),
     Param("library", "string", "a whole library: ycb_gso, robocasa, gymanything, objaverse, alloy"),
     Param("category", "string", "only this category (e.g. blenders, mug)"),
     Param("list", "boolean", "list libraries, categories and names instead of downloading"),
     Param("repo", "string", "dataset repo id (default $ALLOY_ASSETS_REPO, else Alloy's library)"),
     Param("workers", "integer", "parallel downloads", default=8, minimum=1)),
    op_fetch,
    "Files go to $ALLOY_CACHE/imports/<library>/ (default ~/.cache/alloy/imports/<library>/), where scenes, compose "
    "and the API find them; files already there with the right sha256 are not downloaded again. An object whose model "
    "uses another object's files (a composed cabinet's door panel and handle) brings that object too (`required`). "
    "Several fetches can fill one cache at the same time. The library is a "
    "public Hugging Face dataset: no login is needed (`hf auth login` or HF_TOKEN raises the Hub's rate limit).",
    {"library": "ycb_gso", "category": "ycb"}, read_only=False, engine=False),)

__all__ = ["OPS", "op_fetch"]
