# SPDX-License-Identifier: Apache-2.0
"""Identity hashes are the same in fresh processes with different hash seeds (SPEC §7 WP1 acceptance)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = """
import json
from alloy.cards.library import CardLibrary
lib = CardLibrary(assets=None)
print(json.dumps({n: lib.resolve(lib.object(n)).card_hash for n in lib.names("objects")}, sort_keys=True))
"""


def _hashes(seed: str) -> dict[str, str]:
    env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(REPO)}
    out = subprocess.run([sys.executable, "-c", SCRIPT], env=env, cwd=REPO, capture_output=True, text=True, check=True, timeout=120)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_card_hashes_identical_across_processes() -> None:
    a, b = _hashes("1"), _hashes("4242")
    assert len(a) >= 40 and a == b
