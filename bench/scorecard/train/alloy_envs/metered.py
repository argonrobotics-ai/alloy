# SPDX-License-Identifier: Apache-2.0
"""``MeteredEnv``: a ``VectorEnv`` that measures benchmark metrics from the state of every world each step.

The metrics are read before finished worlds reset, so the last step of an episode is the one measured. The runner
reads ``env.last_metrics`` (name -> (W,) array) after each step. Example::

    env = MeteredEnv(scene, task, metrics=lambda s: {"h": s["cube.pos"][:, 2]}, worlds=64)
    env.step(actions); env.last_metrics["h"]
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from alloy.envs import VectorEnv
from alloy.envs.state import BatchState

MetricsFn = Callable[[BatchState], dict[str, Any]]


class MeteredEnv(VectorEnv):
    """``VectorEnv`` plus ``last_metrics``: the metrics of the step just taken, from the pre-reset state."""

    def __init__(self, sim: Any, task: Any, *, metrics: MetricsFn, **kwargs: Any) -> None:
        self._metrics_fn = metrics
        self.last_metrics: dict[str, Any] = {}
        super().__init__(sim, task, **kwargs)

    def _ends(self, s: BatchState) -> tuple[Any, Any, Any, Any]:
        self.last_metrics = self._metrics_fn(s)
        return super()._ends(s)
