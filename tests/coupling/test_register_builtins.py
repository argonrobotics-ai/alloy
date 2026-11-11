# SPDX-License-Identifier: Apache-2.0
"""WP12's report (2026-11-04): ``register_builtins`` was idempotent per process, so a Scheduler built inside
``isolated_registry(empty=True)`` after the first import had no stability estimators or state adapters (empty
``stable_dt``, rollback snapshots without the adapter arrays). It is idempotent per registry now."""

from __future__ import annotations

import alloy._vendor.newton.solvers as ns

import alloy.coupling.scheduler  # noqa: F401  (registers the builtins into the global registry at import)
from alloy.core import registry
from alloy.core.registry import isolated_registry, register_estimator
from alloy.coupling import register_builtins


def _estimators() -> dict:
    return registry._REG.estimators                 # WP0's live table (no public listing)


def test_an_empty_isolated_registry_gets_the_builtins():
    register_builtins()                              # the global registry (whatever ran before)
    assert ns.SolverMuJoCo in _estimators()
    with isolated_registry(empty=True):
        assert ns.SolverMuJoCo not in _estimators()
        register_builtins()
        assert {ns.SolverMuJoCo, ns.SolverVBD} <= set(_estimators())
        assert registry._REG.state                   # the state adapters too
    assert ns.SolverMuJoCo in _estimators()          # the global registry is back, unchanged


def test_a_copied_registry_keeps_its_overrides():
    def custom(solver):
        return None

    with isolated_registry():
        register_estimator(ns.SolverVBD, custom)
        register_builtins()
        assert _estimators()[ns.SolverVBD] is custom
