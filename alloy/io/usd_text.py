# SPDX-License-Identifier: Apache-2.0
"""Tiny writer for ``.usda`` text: typed prims with API schemas, attributes and children.

Example::

    from alloy.io.usd_text import Prim
    p = Prim("Xform", "box", ["PhysicsRigidBodyAPI"])
    p.attr("float physics:mass", "0.2")
    print(p.text())
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def num(x: float) -> str:
    """Shortest text that reads back as the same float32 (the engine's precision): ``num(0.029999999329) == '0.03'``.
    """
    v = np.float32(float(x) + 0.0)
    return str(v) if np.isfinite(v) else ("inf" if v > 0 else "-inf")


def f3(v: Sequence[float]) -> str:
    """USD tuple text of three numbers: ``(x, y, z)``."""
    return "(" + ", ".join(num(x) for x in list(v)[:3]) + ")"


def quat_text(q: Sequence[float]) -> str:
    """USD quaternion text ``(w, x, y, z)`` of an (x, y, z, w) quaternion."""
    x, y, z, w = (float(c) for c in q)
    return f"({num(w)}, {num(x)}, {num(y)}, {num(z)})"


class Prim:
    """One ``def <Type> "<name>"`` block."""

    def __init__(self, typ: str, name: str, schemas: Sequence[str] = ()) -> None:
        self.typ, self.name, self.schemas = typ, name, list(schemas)
        self.attrs: list[tuple[str, str]] = []
        self.children: list[Prim] = []

    def attr(self, decl: str, value: str) -> None:
        """Add ``<decl> = <value>`` (``decl`` is the type and name, e.g. ``"float physics:mass"``)."""
        self.attrs.append((decl, value))

    def _unique_children(self) -> list[Prim]:
        seen: dict[str, int] = {}
        for c in self.children:
            k = seen.get(c.name, 0)
            seen[c.name] = k + 1
            if k:
                c.name = f"{c.name}_{k}"
        return self.children

    def text(self, indent: int = 0) -> str:
        """The prim and its children as ``.usda`` text."""
        pad = "    " * indent
        meta = ""
        if self.schemas:
            meta = " (\n%s    prepend apiSchemas = [%s]\n%s)" % (pad, ", ".join(f'"{s}"' for s in self.schemas), pad)
        lines = [f'{pad}def {self.typ} "{self.name}"{meta}', f"{pad}{{"]
        lines += [f"{pad}    {d} = {v}" for d, v in self.attrs]
        for c in self._unique_children():
            lines.append("")
            lines.append(c.text(indent + 1).rstrip("\n"))
        lines.append(f"{pad}}}")
        return "\n".join(lines) + "\n"
