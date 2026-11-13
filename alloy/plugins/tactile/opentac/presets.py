# SPDX-License-Identifier: Apache-2.0
"""OpenTac boards shipped as presets: the four example designs of the OpenTac repository.

Each preset is that design's ``manifest.json``, copied unchanged into ``data/`` from github.com/opentac/opentac
``examples/manifests/`` at commit e302861 (MIT licence, Copyright (c) 2026 The OpenTac authors; ``LICENSE-OPENTAC``).
Every number below is read from that manifest by ``board.load_board``; the bump height is OpenTac-CAD's default
(2.0 mm, ``opentac/sim/mujoco/params.py``), since none of the four carries a ``bump_params`` block. Saturation is
per taxel at gain 1 with each readout profile (``readout.saturation_force_n``).

Example::

    Tactile(bodies=("left_finger", "right_finger"), opentac="opentac_finger_8x6_60pin", opentac_readout="v3")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alloy.plugins.tactile.opentac.board import Board, load_board

DATA = Path(__file__).parent / "data"


@dataclass(frozen=True)
class BoardPreset:
    name: str
    file: str
    source: str


_E = "examples/manifests/{} at e302861: "

BOARDS: dict[str, BoardPreset] = {p.name: p for p in (
    BoardPreset("opentac_finger_8x6_60pin", "finger_8x6_60pin.json",
                _E.format("finger_8x6_60pin.json") + "45 taxels of 2.8 x 2.8 mm at 3.0 mm pitch on an 8 x 6 scan grid "
                "(60-pin board), 15 x 21 mm; 12 bumps, 2 of them on blocks the outline cut short; manifest v6. "
                "Saturates at 0.35 N per taxel with v2, 9.3 N with v3."),
    BoardPreset("opentac_pinky_tip_30pin", "pinky_tip_30pin.json",
                _E.format("pinky_tip_30pin.json") + "32 taxels of 1.4 x 1.6 mm at 2.0 mm pitch on a 10 x 6 scan grid "
                "(30-pin board), 10 x 18 mm; 8 bumps, all on full 2x2 blocks; manifest v5, imported from a KiCad "
                "board (its 'pixels_true' are ignored here, as OpenTac's loader does). Saturates at 0.42 N per taxel "
                "with v2, 13.2 N with v3."),
    BoardPreset("opentac_square_9x9_30pin", "square_9x9_30pin.json",
                _E.format("square_9x9_30pin.json") + "81 taxels of 2 x 2 mm at 2.0 mm pitch on a 9 x 9 scan grid "
                "(30-pin board), 16 x 16 mm; 25 bumps, 9 on cut blocks (odd grid); manifest v3. Saturates at 0.42 N "
                "per taxel with v2, 13.2 N with v3."),
    BoardPreset("opentac_square_9x9_60pin", "square_9x9_60pin.json",
                _E.format("square_9x9_60pin.json") + "81 taxels of 4 x 4 mm at 4.2 mm pitch on a 9 x 9 scan grid "
                "(60-pin board), 33.6 x 33.6 mm; 25 bumps, 9 on cut blocks; manifest v6. Saturates at 0.31 N per "
                "taxel with v2, 7.0 N with v3."),
)}


def resolve(source: Any, bumps: bool = True) -> Board:
    """A board from a preset name (``opentac_`` prefix optional), a manifest path, or a manifest dict."""
    if isinstance(source, str):
        key = source if source.startswith("opentac_") else f"opentac_{source}"
        if key in BOARDS:
            return load_board(DATA / BOARDS[key].file, bumps=bumps, name=key)
        if not source.endswith(".json") and not Path(source).is_file():
            raise ValueError(f"no OpenTac board {source!r}: presets {sorted(BOARDS)}, or a manifest.json path")
    return load_board(source, bumps=bumps)
