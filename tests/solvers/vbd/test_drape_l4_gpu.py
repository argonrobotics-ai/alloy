# SPDX-License-Identifier: Apache-2.0
"""The cloth default on XB4's drape, one L4 row (coordinator 2026-11-08 13:21Z): the default flip's gate.

XB4-10 measured the L5 Cusick drape (the pack's 8 mm disc mesh through ``alloy.Scene``, 64 worlds, the 2-3 s window)
on main db1d4b946, before WP11's self-contact rule (e3d056ab1) and WP2's Chebyshev fix (5f88e8d9d): the shipped
default stretched 11.7 % and ``cloth_accuracy="super"`` ran at 0.26x MuJoCo Warp 4.0. The bar for the flip: the
default at <= 0.6 % window strain, "super" at >= 1.5x MuJoCo Warp 4.0's 0.79 simulated seconds per wall second at 64
worlds on the L4 (bench/xbench/xb4/mujoco4/RERUN.md). Measured on one L4 on WP2's flip 75ef2413c (merged 1de04ca01;
[internal note] notes/wp11-sc/out_gate): the default (adaptive + Chebyshev 0.9, 40 iterations) DC 0.728 at 0.36 %, 1.89
(2.39x); "super" DC 0.732 at 0.34 %, 1.31 (1.66x); both tests passed (532 s).

The harness is XB4's own (``bench/xbench/xb4/mujoco4/flex/drape_alloy.py``), so the row is the bench's. The speed
bar holds on an L4 only (other GPUs skip it); the strain bar holds on any CUDA device.

Bits (coordinator ruling 2026-11-10 01:42Z): the default drape's window frames (11 frames x 64 worlds x 1,101 nodes,
float32) hash to ``DEFAULT_BITS`` on an L4. Newton release-1.8 compiles the VBD kernels with fast math on CUDA, which
alone moved this drape (peak strain 0.364 % to 0.543 %, no world equal); vendored patch 0013 turns it off and the
drape is then the build before the update bit for bit (``alloy/_vendor/VENDORED.md``; rows: [internal note]
notes/wp11-bump/fastmath). The hash was taken on the update with the patch and equals main's before the update
(PIN_ROWS). Re-key it, with the row, when a change is meant to move GPU cloth; an unexplained move is a regression.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.nightly, pytest.mark.timeout(1800)]

HARNESS = Path(__file__).resolve().parents[3] / "bench/xbench/xb4/mujoco4/flex/drape_alloy.py"
MJW_L4 = 0.79              # MuJoCo Warp 4.0, L5 drape, 64 worlds, simulated s per wall s on the L4 (XB4-10)
SUPER_VS_MJW = 1.5
DEFAULT_STRAIN_PCT = 0.6
WORLDS = 64
DEFAULT_BITS = "d805149d4ab8043dd58caf3767d0fa1f19b84a47ef9f528cf95048185b99213f"   # sha256, the L4's window frames
PIN_ROWS = ("2026-11-10, one L4: the update with patch 0013 and main before the update (790a4eb73) both give this "
            "hash, strain 0.364 %, DC 0.7279; the update with fast math on: dea4577a..., 0.543 %")


@pytest.fixture(scope="module")
def drape():
    """XB4's drape module, loaded from its file (bench/ is not a package)."""
    spec = importlib.util.spec_from_file_location("xb4_drape_alloy", HARNESS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _row(drape, tmp_path, accuracy: str) -> dict:
    pack = json.loads((drape.PACKS / "L5.drape" / "drape.json").read_text())
    res = drape.run(pack, "h8mm", WORLDS, "cuda:0", {}, pack["duration_s"], tmp_path, accuracy)
    assert res["check_ok"], res["problems"]
    return res


@pytest.fixture(scope="module")
def default_row(drape, tmp_path_factory) -> dict:
    return _row(drape, tmp_path_factory.mktemp("drape_default"), "default")


def test_the_default_drape_stretches_at_most_0p6_percent(default_row):
    res = default_row
    assert res["strain_window_max_pct"] <= DEFAULT_STRAIN_PCT, (res["strain_window_max_pct"], res["dc_window_mean"])


def test_the_default_drape_keeps_its_bits_on_the_l4(default_row):
    """Fast math off (vendored patch 0013): the drape is bit for bit the one of the build before Newton release-1.8."""
    import numpy as np
    import warp as wp

    if "L4" not in wp.get_device("cuda:0").name:
        pytest.skip(f"the bits are the L4's ({wp.get_device('cuda:0').name})")
    frames = np.ascontiguousarray(default_row["_q"], dtype=np.float32)
    assert frames.shape[1] == WORLDS and all(np.array_equal(frames[:, 0], frames[:, w]) for w in range(1, WORLDS))
    got = hashlib.sha256(frames.tobytes()).hexdigest()
    assert got == DEFAULT_BITS, (got, default_row["strain_window_max_pct"], default_row["dc_window_mean"], PIN_ROWS)


def test_super_drapes_at_1p5x_mujoco_warp_on_the_l4(drape, tmp_path):
    import warp as wp

    if "L4" not in wp.get_device("cuda:0").name:
        pytest.skip(f"the speed bar is the L4's ({wp.get_device('cuda:0').name})")
    res = _row(drape, tmp_path, "super")
    assert res["world_sim_s_per_s"] >= SUPER_VS_MJW * MJW_L4, (res["world_sim_s_per_s"], res["dc_window_mean"])
