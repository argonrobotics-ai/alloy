# SPDX-License-Identifier: Apache-2.0
"""Precision mode (WP5, SPEC §3.7, §4.3, D9, D11, D14): presets, determinism, frames, Fixed64, digests, health.

Types live in ``alloy.precision.types`` (WP0). What each module gives:

- ``config.preset(name, tier)``: the fast, standard, precision and reference presets. ``PrecisionConfig.preset``
  delegates to it. ``gpu_tier(device)`` gives A (sm_80/90), B (other CUDA) or CPU.
- ``determinism``:
  - ``effective_determinism(cfg, solvers, worlds)`` reports the mode that actually runs, with a reason for every
    downgrade. It reads ``SolverCapability.determinism`` through ``registry.capability_for``, then probes the live
    instance: Steel's ``deterministic``, MuJoCo's ``_deterministic``, per-instance Warp module options, and
    ``CollisionPipeline.deterministic``.
  - ``native_share``.
- ``apply.apply_precision(built, cfg)``: the pipeline and solver kwargs a config implies. It also enforces F1.
- ``frames``: ``check_worlds_at_origin`` (spaced replicas or far worlds raise ``PrecisionError``), ``rebase_worlds``
  (F4, float64 shift into ``world_origin``) and ``plan_rebase`` (auto rebasing).
- ``fixed64``: ``Fixed64`` two-pass int64 scatter sums that are order-independent and fail closed on non-finite
  terms; ``fixed64_sum`` for host code.
- ``digest``: ``world_digests`` (128 bits per world, batch-invariant, extras with item->world maps), ``run_digest``
  and ``env_tuple``.
- ``health.HealthMonitor`` (``HealthLike``): per-world ``StepHealth`` with cheap and full checks, then ``latch`` and
  ``poll``. Its parts:
  - ``ladder``: the §4.3 severity table as code (``LADDER``, ``severity_of``, ``thresholds``).
  - ``health_adapters``: MuJoCo ``solver_niter`` and overflow bits, Steel overflow, Newton contact overflow.
- ``repeat.repeat_test(make_sim, case)``: the R1-R8 bitwise-repeat protocol on any ``SimulationLike``.
- ``golden``: per-environment determinism goldens in ``bench/goldens/``.
- ``mjw_det``: MuJoCo Warp's own ``DeterminismType`` on a SolverMuJoCo (set, read, stub tripwire). At the pin no
  mode is repeatable (P3-1, ``docs/notes/wp5-mjw-determinism.md``), so it only feeds the downgrade reason.
- ``mjw_order`` / ``mjw_rows``: Alloy's bodies for the CONTACTS and CONSTRAINT stubs (route c): canonical contact
  and constraint-row order, installed by ``set_mjw_determinism``.
- ``world_groups``: per-world grouping of host item arrays (one stable sort, linear in the world count), used
  by the digest, health and frame setup tables.

The determinism lint is ``python -m alloy.tools.lint_atomics alloy/`` (no float atomics, D3), and it runs in the merge
gate. Test helpers live in ``alloy.testing.precision``. Results: ``docs/notes/wp5-results.md``.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "preset": "config", "gpu_tier": "config", "effective_determinism": "determinism", "apply_precision": "apply",
    "check_worlds_at_origin": "frames", "rebase_worlds": "frames", "Fixed64": "fixed64",
    "fixed64_mantissa_bits": "fixed64", "fixed64_add": "fixed64", "fixed64_read": "fixed64",
    "fixed64_prepare": "fixed64", "world_digests": "digest", "run_digest": "digest", "HealthMonitor": "health",
    "repeat_test": "repeat",
}
__all__ = sorted(_EXPORTS)


def _install_guards() -> None:
    """MuJoCo C's pivot clamp on MuJoCo Warp's Newton Cholesky for every SolverMuJoCo in the process (BUL-0016,
    ``mjw_chol``), route on or off, and the skip of the last iteration's unused factorization (``mjw_lazy_factor``,
    SCR-WP0-RIGIDDT-2). Skipped where mujoco_warp is not installed."""
    try:
        from alloy.precision import mjw_chol
    except ImportError:
        return
    try:
        mjw_chol.install()
    except ImportError:
        pass
    try:
        from alloy.precision import mjw_lazy_factor
    except ImportError:
        return
    mjw_lazy_factor.install()


_install_guards()


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module 'alloy.precision' has no attribute {name!r}")
    return getattr(importlib.import_module(f"alloy.precision.{_EXPORTS[name]}"), name)
