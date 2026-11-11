# SPDX-License-Identifier: Apache-2.0
"""Bench test helpers (WP6, SPEC §6.4): synthetic rig datasets for the R3/R4 twins and fake twins for unit tests.

Synthetic data come from the analytic law plus Gaussian noise (WP0's `synthetic_rig`), with instrument u95 values
chosen so the TUR rule (bound / U_real >= 3) can judge each tolerance. Real rig data replace them later; the
scoreboard keeps the real R3/R4 rows `not_run` while only synthetic data exist (protocol `synthetic/...`).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from alloy.bench.rigs import write_rig
from alloy.bench.types import Condition, RigDataset
from alloy.testing.fakes import synthetic_rig

R3_LAW, R4_LAW = "R3.incline_friction", "R4.drop_bounce"
DATASET_ID = "synthetic-v1"
R3_CONDITIONS = (Condition("pp_on_steel", {"mu_static": 0.35}), Condition("rubber_on_wood", {"mu_static": 0.7}))
R4_CONDITIONS = tuple(Condition(f"drop{int(h * 100)}cm", {"drop_height": h}) for h in (0.05, 0.10, 0.15))
R4_RESTITUTION = 0.7


def _r3_truth(params: Mapping[str, float], cond: Condition) -> dict[str, float]:
    """Coulomb: slip at atan(mu_s); one mu (static = kinetic, as Newton models it); no creep."""
    mu = cond.params["mu_static"]
    return {"slip_angle": math.atan(mu), "mu_kinetic": mu, "creep": 0.0}


def _r4_truth(params: Mapping[str, float], cond: Condition) -> dict[str, float]:
    e, h = params["restitution"], cond.params["drop_height"]
    return {"restitution": e, "rebound_height": e * e * h}


def r3_dataset(seed: int = 0) -> RigDataset:
    noise = {"slip_angle": math.radians(0.05), "mu_kinetic": 0.005, "creep": 1e-7}
    data = synthetic_rig(R3_LAW, {}, noise, conditions=R3_CONDITIONS, trials_per_condition=20,
                         calib_per_condition=10, seed=seed, model=_r3_truth)
    inst = {"slip_angle": {"u95": math.radians(0.05), "what": "inclinometer 0.05 deg"},
            "mu_kinetic": {"u95": 0.005, "what": "240 fps video track"}, "creep": {"u95": 1e-6, "what": "dial"}}
    return _restamp(data, inst, {"box": {"half_extent_m": 0.025, "mass_kg": 0.125}})


def r4_dataset(seed: int = 0) -> RigDataset:
    noise = {"restitution": 0.005, "rebound_height": 5e-4}
    data = synthetic_rig(R4_LAW, {"restitution": R4_RESTITUTION}, noise, conditions=R4_CONDITIONS,
                         trials_per_condition=20, calib_per_condition=10, seed=seed, model=_r4_truth)
    inst = {"restitution": {"u95": 0.005, "what": "apex height, 240 fps + scale bar"},
            "rebound_height": {"u95": 5e-4, "what": "apex height"}}
    return _restamp(data, inst, {"ball": {"radius_m": 0.02}})


def _restamp(data: RigDataset, instruments: Mapping[str, Any], objects: Mapping[str, Any]) -> RigDataset:
    card = dataclasses.replace(data.card, dataset_id=DATASET_ID, instruments=dict(instruments),
                               objects=dict(objects), protocol_version="synthetic/1")
    return dataclasses.replace(data, card=card)


def write_synthetic_rigs(root: Path = Path("bench/rigs")) -> list[Path]:
    """Write the R3 and R4 synthetic datasets (immutable: refuses to overwrite)."""
    return [write_rig(r3_dataset(), root), write_rig(r4_dataset(), root)]


def fake_rig_dataset(values: Mapping[str, float], noise: float = 0.0, n: int = 10) -> RigDataset:
    """A tiny one-condition dataset for comparator/judge unit tests."""
    return synthetic_rig("TEST.rig", dict(values), noise, trials_per_condition=n, calib_per_condition=n // 2)


def noisy(values: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    return np.asarray(values, np.float64) + np.random.default_rng(seed).normal(0.0, sigma, np.shape(values))


__all__ = ["r3_dataset", "r4_dataset", "write_synthetic_rigs", "fake_rig_dataset", "noisy", "R3_LAW", "R4_LAW"]
