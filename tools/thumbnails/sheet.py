# SPDX-License-Identifier: Apache-2.0
"""Contact sheet: thumbnails on a grid with their names under each (Pillow).

Example::

    contact_sheet([Path("a/thumb.png"), Path("b/thumb.png")], Path("sheet.png"), cols=4)
"""
from __future__ import annotations

from pathlib import Path

CELL, LABEL, PAD = 256, 22, 8


def _label(path: Path) -> str:
    name = path.parent.name if path.name == "thumb.png" else path.stem
    return name if len(name) <= 34 else name[:31] + "..."


def contact_sheet(images: list[Path], out: Path, cols: int = 4, cell: int = CELL) -> Path:
    """Write ``out``: ``images`` scaled to ``cell`` px in ``cols`` columns, each labelled with its object name."""
    from PIL import Image, ImageDraw  # noqa: PLC0415

    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (cell + PAD) + PAD, rows * (cell + LABEL + PAD) + PAD), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for i, p in enumerate(images):
        x, y = PAD + (i % cols) * (cell + PAD), PAD + (i // cols) * (cell + LABEL + PAD)
        img = Image.open(p).convert("RGB")
        img.thumbnail((cell, cell))
        sheet.paste(img, (x, y))
        draw.text((x + 2, y + cell + 4), _label(p), fill=(40, 40, 40))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out
