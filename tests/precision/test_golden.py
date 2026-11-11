# SPDX-License-Identifier: Apache-2.0
"""Determinism goldens (SPEC §6.5): this environment's digests equal the stored golden (skips when no golden exists
for this environment key, e.g. another CPU class). A CPU golden must exist for the current Warp version and Newton
pin, or the skip would hide that no golden covers what main runs (WP0 2026-11-05: none covered pin 8f3466f9)."""

from __future__ import annotations

import pytest

import warp as wp

from alloy.core.newton_compat import NEWTON_PIN
from alloy.precision.golden import GOLDEN_DIR, check_goldens, golden_key, load_golden

pytestmark = pytest.mark.det


def _check(device: str) -> None:
    key = golden_key(device)
    if load_golden(key) is None:
        pytest.skip(f"no golden for {key}")
    status = check_goldens(device)
    assert status and all(v == "equal" for v in status.values()), status


def test_cpu_golden() -> None:
    _check("cpu")


@pytest.mark.gpu
def test_gpu_golden() -> None:
    _check("cuda:0")


def test_a_cpu_golden_covers_the_current_pin() -> None:
    tail = f"-warp{wp.config.version}-newton{NEWTON_PIN[:8]}.json"
    found = sorted(p.name for p in GOLDEN_DIR.glob(f"cpu-*{tail}"))
    assert found, (f"no CPU golden for Warp {wp.config.version} / Newton {NEWTON_PIN[:8]}: on the gate host run "
                   f"python -m alloy.precision.golden update --device cpu --reason \"<pin or Warp bump>\" (6 s) "
                   f"and land it with a 'golden-update: <reason>' trailer; WP5 refreshes the GPU golden")
