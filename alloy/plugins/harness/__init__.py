# SPDX-License-Identifier: Apache-2.0
"""The plug-in test harness: ``check(plugin) -> CheckReport`` (laws, crash, determinism, world independence, speed).

Example::

    from alloy.plugins.harness import check
    print(check("magnet").summary())
"""

from alloy.plugins.harness.check import check
from alloy.plugins.harness.report import CheckReport, CheckResult
from alloy.plugins.harness.run import Run, Trace, run_scene
from alloy.plugins.harness.scene import GRAVITY, Law, LawResult, TestScene, body_xform, build_model, build_simulation

__all__ = ["GRAVITY", "CheckReport", "CheckResult", "Law", "LawResult", "Run", "TestScene", "Trace", "body_xform",
           "build_model", "build_simulation", "check", "run_scene"]
