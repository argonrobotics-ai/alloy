# SPDX-License-Identifier: Apache-2.0
"""``alloy audit <scene>``: alloy.check over several seeds plus the bully agent over random legal episodes,
merged into findings by severity with replays and fixes (JSON).

Example::

    >>> import alloy.audit
    >>> report = alloy.audit.audit("demo_peg_d12", rounds=2, episodes=8)       # doctest: +SKIP
    >>> report.ok, report.to_dict()["counts"]                                   # doctest: +SKIP
    (True, {'error': 0, 'warn': 1, 'info': 0})

Command line: ``python -m alloy.audit my_scenes/scenes/scene.yaml --rounds 3 --episodes 32 --library my_scenes``.
"""

from __future__ import annotations

from typing import Any

from alloy.audit.report import AuditFinding, AuditReport

__all__ = ["AuditFinding", "AuditReport", "audit", "library"]


def audit(scene: Any, **kwargs: Any) -> AuditReport:
    """Audit a scene; see ``alloy.audit.run.audit``. Example: ``alloy.audit.audit("demo_peg_d12").summary()``."""
    from alloy.audit.run import audit as _audit  # noqa: PLC0415 - keeps `import alloy.audit` light

    return _audit(scene, **kwargs)


def library(roots: Any = ()) -> Any:
    """The card library an audit reads (your ``roots`` first, then the roots ``alloy.check`` uses); see
    ``alloy.audit.run.library``. Example: ``replay_finding(handle, library=alloy.audit.library(["my_scenes"]))``."""
    from alloy.audit.run import library as _library  # noqa: PLC0415

    return _library(roots)
