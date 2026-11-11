# SPDX-License-Identifier: Apache-2.0
"""P3-1: MuJoCo Warp DeterminismType hook, stub tripwire and the honest downgrade reason (CPU)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from alloy.core.errors import PrecisionError
from alloy.core.newton_compat import MJWARP_PIN, installed_mujoco_warp_commit
from alloy.precision import mjw_det
from alloy.precision.determinism import solver_det
from alloy.precision.types import Determinism


class SolverMuJoCo:  # named like Newton's class: solver_det dispatches on the class name
    def __init__(self, bits: int = 0) -> None:
        self._deterministic = 0
        self.mjw_model = SimpleNamespace(opt=SimpleNamespace(deterministic=bits))


def test_noop_detection() -> None:
    def stub():
        """Doc."""

    def real():
        return len([1])

    assert mjw_det._is_noop(stub)
    assert not mjw_det._is_noop(real)
    assert not mjw_det._is_noop(mjw_det.stub_bits)


def test_stub_bits_at_pin() -> None:
    """Tripwire for a mujoco_warp bump: when CONTACTS / CONSTRAINT get real implementations, rerun the P3-1
    measurement (tests/precision/test_mjw_determinism_gpu.py) and lift the downgrade if R1/R2/R4 and M7 pass."""
    if not mjw_det.has_determinism_option():
        pytest.skip("installed mujoco_warp has no DeterminismType")
    if not installed_mujoco_warp_commit().startswith(MJWARP_PIN[:8]):
        pytest.skip("installed mujoco_warp is not the pinned commit")
    assert mjw_det.stub_bits() == mjw_det.CONTACTS | mjw_det.CONSTRAINT, (
        "mujoco_warp now implements a determinism sort: rerun P3-1 (docs/notes/wp5-mjw-determinism.md)")


def test_set_and_read_bits() -> None:
    s = SolverMuJoCo()
    assert mjw_det.set_mjw_determinism(s, "all") == mjw_det.ALL
    assert mjw_det.mjw_bits(s) == 7
    assert mjw_det.set_mjw_determinism(s, 4) == 4
    with pytest.raises(PrecisionError):
        mjw_det.set_mjw_determinism(s, 8)
    with pytest.raises(PrecisionError):
        mjw_det.set_mjw_determinism(SimpleNamespace(), 4)
    assert mjw_det.mjw_bits(SimpleNamespace()) == 0


def test_downgrade_reason_on_gpu() -> None:
    gpu = SimpleNamespace(is_cuda=True)
    assert mjw_det.downgrade_reason(SolverMuJoCo(0)) is None
    det = solver_det(SolverMuJoCo(mjw_det.ALL), 1024, gpu)
    assert det.mode == Determinism.NONE
    assert "DeterminismType 7" in det.reason and "wp5-mjw-determinism" in det.reason
    plain = solver_det(SolverMuJoCo(0), 1024, gpu)
    assert plain.mode == Determinism.NONE and "lowering is off" in plain.reason
