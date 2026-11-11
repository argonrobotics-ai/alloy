# SPDX-License-Identifier: Apache-2.0
"""Citations: Gym-Anything file:line references -> pinned notes on Alloy quantities.

Every imported value keeps Gym-Anything's own label (datasheet, search, estimate, ...) and their primary-source text,
so an agent reading a card can tell a maker datasheet from an engineering guess.

Example::

    >>> cite_ref("wine_pouring", "params.py:139")
    'gym-anything/robotics@febb260:benchmarks/roboworld/environments/wine_pouring/params.py:139'
    >>> q_from(Cited(2500.0, "literature", "params.py:139", "soda-lime 2.5 g/cm3"), "wine_pouring").source
    'catalog'
"""

from __future__ import annotations

from alloy.core.quantity import Q
from alloy.importers.gymanything.spec import Cited

UPSTREAM = "gym-anything/robotics"
UPSTREAM_URL = "https://github.com/gym-anything/robotics"
UPSTREAM_COMMIT = "febb260"
UPSTREAM_LICENCE = "MIT, Copyright (c) 2026 cmu-l3"
ENV_DIR = "benchmarks/roboworld/environments"
_ROOTS = ("benchmarks/", "src/", "tests/", "docs/", "agents/", "requirements/", "docker/")


def cite_ref(sim: str, cite: str) -> str:
    """A pinned upstream reference for ``cite``: bare ``file:line`` is taken inside the sim's environment folder."""
    c = cite.strip()
    if c.startswith(UPSTREAM):
        return c
    if c.startswith(_ROOTS):
        path = c
    elif c.split("/", 1)[0] == sim:
        path = f"{ENV_DIR}/{c}"
    else:
        path = f"{ENV_DIR}/{sim}/{c}"
    return f"{UPSTREAM}@{UPSTREAM_COMMIT}:{path}"


def note_for(c: Cited, sim: str, what: str = "") -> str:
    """``"<what>; <their label> (<pinned ref>): <their text>"``, the note an imported ``Q`` carries."""
    head = f"{what}; " if what else ""
    tail = f": {c.quote}" if c.quote else ""
    return f"{head}Gym-Anything label '{c.basis}' ({cite_ref(sim, c.cite)}){tail}"


def q_from(c: Cited, sim: str, what: str = "", scale: float = 1.0) -> Q:
    """The cited value as an Alloy ``Q`` (value and ``unc`` multiplied by ``scale``)."""
    unc = (c.unc[0] * scale, c.unc[1] * scale) if c.unc else None
    return Q(c.value * scale, c.alloy_source, unc, note_for(c, sim, what))  # type: ignore[arg-type]


def default_q(value: float) -> Q:
    """An unsourced library default (listed in ``default_leaves``; queued for the sourcing agent)."""
    return Q(value, "default")
