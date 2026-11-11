# SPDX-License-Identifier: Apache-2.0
"""A synthetic "real robot": simulate the robot with hidden parameters, a delay and sensor noise, and record it.

Used to check a fit against a known answer (the tests and ``examples/sysid``), and to try the commands without a
robot: ``python -m alloy.sysid demo`` writes such a recording.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from alloy.sysid.excitation import Excitation
from alloy.sysid.recording import Recording, make_episode
from alloy.sysid.replay import Bench, Track


def record(robot: Any, excitation: Excitation, hidden: dict[str, Any] | None = None, *, noise: float = 0.0,
           seed: int = 0, device: str | None = None) -> Recording:
    """Play ``excitation`` on the simulated robot whose joints have the ``hidden`` parameters (field -> (J,)
    values or one value for all; ``"delay"`` in seconds; fields left out keep the robot file's values), and
    return the recording with Gaussian position noise of ``noise`` rad (deterministic in ``seed``).

    Example: ``record("panda", exc, {"kp": 3000.0, "delay": 0.012}, noise=2e-4)``."""
    dt = float(np.median(np.diff(excitation.t)))
    bench = Bench(robot, excitation.joints, worlds=1, dt=dt, device=device)
    count = len(excitation.joints)
    fields = {f: np.broadcast_to(np.asarray(v, np.float64), (count,)).copy() for f, v in (hidden or {}).items()}
    track = Track(excitation.command, np.tile(excitation.command[0], (len(excitation.t), 1)))
    sim_q = bench.rollout(fields, [track])[0][:, 0, :]
    rng = np.random.default_rng(seed)
    measured = sim_q + float(noise) * rng.standard_normal(sim_q.shape)
    rec = Recording(list(excitation.joints), [make_episode(excitation.t, excitation.command, measured)],
                    source="synthetic")
    rec.notes.append(f"synthetic: hidden parameters {sorted(fields)}, noise {noise} rad, seed {seed}")
    return rec


__all__ = ["record"]
