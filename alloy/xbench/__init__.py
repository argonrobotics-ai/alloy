# SPDX-License-Identifier: Apache-2.0
"""Alloy adapter for the cross-engine bench (bench/xbench, CROSS_ENGINE.md "How --engine-under-test alloy works").

``install(protocol=...)`` replaces the stock solver classes in ``newton.solvers`` with Alloy's (``adapter``), so an
unchanged xbench scene script builds Alloy solvers with its own arguments; ``uninstall()`` puts them back.
The solvers (``solvers``) subclass the stock ones (``mjw_model``/``mjw_data`` and every other attribute kept) and
run WP5's health checks after each step, as WP12's ``IntegratedSimulation`` does.
"""

from __future__ import annotations

from alloy.xbench.adapter import HEALTH_MARKER, PROFILE_ENV, health_report, install, installed, uninstall
from alloy.xbench.solvers import HealthLedger, is_alloy_solver, ledgers

__all__ = ["HEALTH_MARKER", "PROFILE_ENV", "HealthLedger", "health_report", "install", "installed",
           "is_alloy_solver", "ledgers", "uninstall"]
