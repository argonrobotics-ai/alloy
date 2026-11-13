# SPDX-License-Identifier: Apache-2.0
"""OpenTac touch-skin output for ``Tactile``: what a real OpenTac board would read, from the solver's contacts.

OpenTac (github.com/opentac/opentac; arXiv 2610.31661) is an open-source resistive touch skin from Johns Hopkins: a
flex board of taxels, a pressure-sensitive film and an optional printed TPU bump sheet, each bump on a 2x2 block of
taxels. The board reads normal load per taxel only; shear tips a bump and splits its load over its taxels, and a
decoder reads the split back. It streams 12-bit ADC counts.

This package ports OpenTac's own published models (MIT licence, Copyright (c) 2026 The OpenTac authors;
``LICENSE-OPENTAC``) and runs them batched inside Alloy. Nobody, including us, has compared the result with a real
skin, and OpenTac's authors mark their bump model "Not validated against hardware".

Use it through ``Tactile``::

    Tactile(bodies=("left_pad", "right_pad"), opentac="opentac_finger_8x6_60pin", opentac_readout="v3")
    # read()["tactile"]: taxels, pad_force (newtons, unchanged) + board_load, counts, bump_wrench

and decode the counts as the firmware would: ``decode(board, counts, "v3")``. See ``docs/notes/opentac.md``.
"""

from alloy.plugins.tactile.opentac.board import Board, ManifestError, load_board
from alloy.plugins.tactile.opentac.model import board_loads, decode, decode_shear
from alloy.plugins.tactile.opentac.presets import BOARDS, resolve
from alloy.plugins.tactile.opentac.readout import PROFILES, Profile, profile

__all__ = ["BOARDS", "PROFILES", "Board", "ManifestError", "Profile", "board_loads", "decode", "decode_shear",
           "load_board", "profile", "resolve"]
