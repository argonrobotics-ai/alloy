# SPDX-License-Identifier: Apache-2.0
"""tests/contact: make the shared GPU harness (peg_harness.py) importable under --import-mode=importlib."""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
