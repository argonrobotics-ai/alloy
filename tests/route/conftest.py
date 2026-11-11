# SPDX-License-Identifier: Apache-2.0
"""tests/route: make the shared C-extension scenes (mjc_scenes.py) importable under --import-mode=importlib."""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
