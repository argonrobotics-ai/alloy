# SPDX-License-Identifier: Apache-2.0
"""CUDA-only contact laws (A7.hydroelastic, A8.clearance, A8.push).

Newton bakes texture SDFs on CUDA only, so these laws cannot run on another device. :class:`CudaOnly` wraps a law
twin: on a non-CUDA device the law reports ``not_run`` with that reason (SB-12 lists it as not run and leaves it out
of the pass rate) instead of ending as ``error``; on CUDA the wrapped twin runs as before, a direct (``evaluate``) twin
directly and a stepped twin through the runner's public analytic path (``plan_worlds`` -> ``step_twin`` ->
``judge_analytic``, the same sequence ``alloy.bench.runner.run_law`` uses for analytic laws).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

CUDA_REASON = "needs a CUDA device: Newton bakes the texture SDFs these contact laws use on CUDA only"


def is_cuda(device: str) -> bool:
    return str(device).startswith("cuda")


def _finite(state: Any) -> bool:
    for name in ("body_q", "body_qd", "particle_q", "particle_qd"):
        arr = getattr(state, name, None)
        if arr is not None and arr.shape[0] > 0 and not np.all(np.isfinite(arr.numpy())):
            return False
    return True


class CudaOnly:
    """A law twin that runs on CUDA devices only and reports ``not_run`` (with :data:`CUDA_REASON`) elsewhere."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.monitors: dict[str, Any] = {}

    def settings_record(self, solver: str) -> Any:
        record = getattr(self.inner, "settings_record", None)
        return record(solver) if callable(record) else {}

    def evaluate(self, ctx: Any) -> tuple:
        from alloy.bench.compare import placeholder_verdict

        if not is_cuda(ctx.device):
            self.monitors = {"not_run_reason": CUDA_REASON}
            return tuple(placeholder_verdict(n, "1", "not_run", ctx.spec.stat.get(n, "mean")) for n in ctx.spec.metrics)
        if hasattr(self.inner, "evaluate"):
            verdicts = tuple(self.inner.evaluate(ctx))
            self.monitors = dict(getattr(self.inner, "monitors", {}))
            return verdicts
        return self._stepped(ctx)

    def _stepped(self, ctx: Any) -> tuple:
        from alloy.bench.runner import judge_analytic, plan_worlds, step_twin

        twin, spec = self.inner, ctx.spec
        worlds = plan_worlds(spec, tuple(twin.conditions(ctx.quick)), ctx, getattr(twin, "expand", None))
        run = step_twin(spec, twin, worlds, ctx)
        run.monitors["finite"] = _finite(run.state)
        values = twin.scalars(run.log)
        refs = twin.references(run.log) if hasattr(twin, "references") else {}
        verdicts = judge_analytic(spec, twin, values, refs, [w.condition for w in worlds])
        run.monitors.update(dict(getattr(twin, "monitors", {})))
        self.monitors = run.monitors
        return tuple(verdicts)


def cuda_only(factory: Callable[[], Any]) -> Callable[[], CudaOnly]:
    """Twin factory for ``register_law``: each call wraps a fresh ``factory()`` twin in :class:`CudaOnly`."""

    def make() -> CudaOnly:
        return CudaOnly(factory())

    make.inner_factory = factory  # type: ignore[attr-defined]
    return make


__all__ = ["CUDA_REASON", "CudaOnly", "cuda_only", "is_cuda"]
