# SPDX-License-Identifier: Apache-2.0
"""Sandbox child: ``python -m alloy.plugins.sandbox.child <policy.json>``.

Applies resource limits, then Landlock (file allow-list, no TCP), and only then imports and checks the plug-in, so
every line of plug-in code runs confined. Writes the harness report to ``policy["out"]``.
"""

from __future__ import annotations

import json
import logging
import resource
import sys
import traceback
from pathlib import Path
from typing import Any

log = logging.getLogger("alloy.plugins")


def apply_limits(pol: dict[str, Any]) -> dict[str, int]:
    """CPU seconds, file size, open files, no core dumps and (optionally) address space."""
    limits = {"cpu_s": resource.RLIMIT_CPU, "max_file_bytes": resource.RLIMIT_FSIZE,
              "open_files": resource.RLIMIT_NOFILE, "memory_bytes": resource.RLIMIT_AS}
    applied = {"core": 0}
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    for key, which in limits.items():
        value = pol.get(key)
        if value:
            soft, hard = resource.getrlimit(which)
            cap = int(value) if hard == resource.RLIM_INFINITY else min(int(value), hard)
            resource.setrlimit(which, (cap, cap))
            applied[key] = cap
    return applied


def confine(pol: dict[str, Any]) -> dict[str, Any]:
    """Limits, then Landlock if the kernel has it (else the parent's layers only)."""
    from alloy.plugins.sandbox.landlock import abi_version, restrict_self  # noqa: PLC0415

    info: dict[str, Any] = {"limits": apply_limits(pol)}
    if abi_version() > 0:
        ll = restrict_self(pol["read"], pol["write"], deny_tcp=not pol.get("network", False))
        info["landlock"] = {"abi": ll["abi"], "tcp_denied": ll["tcp_denied"], "paths": len(ll["paths"])}
    else:
        info["landlock"] = None
    return info


def run(pol: dict[str, Any]) -> dict[str, Any]:
    """Confine this process, then load and check the plug-in; the report as a dict."""
    import warp as wp  # noqa: PLC0415 - trusted imports before confinement

    import alloy.plugins.harness.check  # noqa: F401, PLC0415
    import alloy.plugins.registry  # noqa: F401, PLC0415

    wp.init()
    info = confine(pol)
    from alloy.plugins.harness.check import check  # noqa: PLC0415
    from alloy.plugins.registry import load_target  # noqa: PLC0415

    classes = load_target(pol["target"])
    if len(classes) != 1:
        raise ValueError(f"{pol['target']} holds {len(classes)} plug-ins {[c.name for c in classes]}; name one")
    report = check(classes[0](pol.get("config") or {}), device=pol["device"], quick=pol["quick"],
                   speed=pol["speed"]).to_dict()
    report["sandbox"] = info
    return report


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    pol = json.loads(Path(argv[0]).read_text())
    out = Path(pol["out"])
    try:
        report = run(pol)
    except BaseException:  # noqa: BLE001 - everything the plug-in raises goes back to the parent
        out.write_text(json.dumps({"error": traceback.format_exc()[-3000:]}))
        return 1
    out.write_text(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
