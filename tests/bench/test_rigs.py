# SPDX-License-Identifier: Apache-2.0
"""Rig datasets: sha256 manifest, refusal of edited trials, committed synthetic sets (SPEC §3.10 load_rig)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from alloy.bench import rigs
from alloy.core.errors import LawError
from alloy.testing import bench as TB

REPO_RIGS = Path(__file__).resolve().parents[2] / "bench" / "rigs"


def test_round_trip_and_immutability(tmp_path) -> None:
    data = TB.r4_dataset(seed=3)
    path = rigs.write_rig(data, tmp_path)
    back = rigs.load_rig(path)
    assert back.card == data.card and len(back.trials) == len(data.trials) and len(back.sha256) == 64
    t0, b0 = data.trials[0], back.trials[0]
    assert b0.scalars == pytest.approx(t0.scalars) and np.array_equal(b0.series["t"], t0.series["t"])
    with pytest.raises(LawError):
        rigs.write_rig(data, tmp_path)  # datasets are immutable
    assert rigs.is_synthetic(back) and rigs.datasets_for("R4.drop_bounce", tmp_path) == [path]


def test_edited_trial_is_refused(tmp_path) -> None:
    path = rigs.write_rig(TB.r3_dataset(), tmp_path)
    victim = sorted((path / "trials").glob("*.npz"))[0]
    with np.load(victim) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["scalar__slip_angle"] = np.asarray(0.9)
    np.savez(victim, **arrays)
    with pytest.raises(LawError, match="does not match"):
        rigs.load_rig(path)


def test_extra_trial_is_refused(tmp_path) -> None:
    path = rigs.write_rig(TB.r4_dataset(), tmp_path)
    (path / "trials" / "sneaky__99.npz").write_bytes(b"x")
    with pytest.raises(LawError, match="not in the manifest"):
        rigs.load_rig(path)


def test_committed_synthetic_datasets_load() -> None:
    r3 = rigs.load_rig(REPO_RIGS / "R3.incline_friction" / "synthetic-v1")
    r4 = rigs.load_rig(REPO_RIGS / "R4.drop_bounce" / "synthetic-v1")
    assert len(r3.split("calib")) == len(r3.split("valid")) == 20
    mean_slip = np.mean([t.scalars["slip_angle"] for t in r3.split("valid") if t.condition_id == "pp_on_steel"])
    assert mean_slip == pytest.approx(math.atan(0.35), abs=math.radians(0.1))
    assert {c.id for c in r4.conditions} == {"drop5cm", "drop10cm", "drop15cm"}
    assert TB.r3_dataset().trials[0].scalars == r3.trials[0].scalars  # generator is deterministic
