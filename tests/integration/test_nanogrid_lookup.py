# SPDX-License-Identifier: Apache-2.0
"""warp.fem's packed-grid cell lookup in integers (``alloy.integration.nanogrid_lookup``; known gap
docs/known_gaps/mpm-moving-container-many-worlds.md). CPU laws; the GPU rows are tests/integration/test_nanogrid_lookup_gpu.py.
- the guard: the pinned Warp and the audited source of the function it replaces (a Warp bump fails here first);
- the copy is Warp's function but for the lines marked "Alloy";
- an MPM build installs it; a scene without MPM (demo_pour) never does (fresh processes).

    PYTHONPATH=. python -m pytest tests/integration/test_nanogrid_lookup.py -q
"""

from __future__ import annotations

import difflib
import inspect
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import warp as wp

from alloy.integration import nanogrid_lookup as N

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]


def test_the_guard_matches_the_pinned_warp() -> None:
    """A Warp other than the pinned one, or an edited lookup factory, must be audited again: re-copy the function,
    move the "Alloy" lines over, and record its new hash in SOURCE_SHA256."""
    assert wp.__version__ == N.WARP_VERSION
    assert N.warp_source_sha256() == N.SOURCE_SHA256


def test_the_copy_differs_from_warps_function_only_at_the_alloy_lines() -> None:
    ours = textwrap.dedent(inspect.getsource(N.make_filtered_cell_lookup)).splitlines()
    warp = textwrap.dedent(inspect.getsource(N._WARP)).splitlines()
    removed = [ln for ln in difflib.unified_diff(warp, ours, lineterm="", n=0)
               if ln.startswith("-") and not ln.startswith("---")]
    added = [ln for ln in difflib.unified_diff(warp, ours, lineterm="", n=0)
             if ln.startswith("+") and not ln.startswith("+++")]
    assert [ln[1:].strip() for ln in removed] == [
        "def make_filtered_cell_lookup(grid_geo, filter_func: wp.Function = None):",
        "suffix = f\"{grid_geo.name}{filter_func.key if filter_func is not None else ''}\"",
        "packed_uvw = uvw + vec3_type(env_offset)",
        "i, j, k = int(wp.floor(packed_uvw[0])), int(wp.floor(packed_uvw[1])), int(wp.floor(packed_uvw[2]))",
    ], removed
    marks = ("def make_filtered_cell_lookup", "Warp 1.18.0", "Alloy", "neighbouring cell", "env_offset[", "alloy_int")
    assert len(added) == 8 and all(any(m in ln for m in marks) for ln in added), added


def _fresh(code: str) -> str:
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=600,
                         env={**__import__("os").environ, "PYTHONPATH": str(REPO)})
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout.strip().splitlines()[-1]


def test_an_mpm_build_installs_it_and_a_scene_without_mpm_does_not() -> None:
    probe = ("import alloy\nfrom alloy.integration import nanogrid_lookup as N\n{build}\n"
             "print('INSTALLED' if N.installed() else 'WARP')")
    pour = 'sim = alloy.load("demo_pour", device="cpu")'
    sand = ('s = alloy.Scene(dt=0.01); s.add_ground()\n'
            's.add_box("sand", size=(0.02, 0.02, 0.01), pos=(0, 0, 0.005), material="sand_silica_dry")\n'
            'sim = alloy.build(s, device="cpu")')
    assert _fresh(probe.format(build=pour)) == "WARP"
    assert _fresh(probe.format(build=sand)) == "INSTALLED"
