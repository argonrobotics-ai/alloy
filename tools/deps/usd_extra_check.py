# SPDX-License-Identifier: Apache-2.0
"""Install check for the ``usd`` extra: read a USD file into cards, then load a USD scene Alloy exported.

    python tools/deps/usd_extra_check.py        # one JSON line; exit 0 when every step ran

Reading USD goes through the vendored Newton's ``add_usd``, which needs ``pxr`` (OpenUSD, from ``usd-core``) and
``newton_usd_schemas``. The core install has neither, so on it this check fails at step 1 and says what to install.
With ``pip install "alloy-engine[usd]"`` it:

1. imports ``pxr`` and ``newton_usd_schemas`` and reports their versions;
2. runs ``alloy.io.import_file`` on the ``cartpole.usda`` the vendored Newton ships, into card files;
3. exports the library scene ``demo_peg_d12`` with ``alloy.io.export_scene``, loads the ``.usda`` it wrote with
   ``alloy.load`` and runs it 0.1 s on the CPU.

Example (Python):
    >>> [name for name, _ in STEPS]
    ['packages', 'import_file', 'export_load']
"""

from __future__ import annotations

import importlib.metadata as md
import json
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

FIX = 'pip install "alloy-engine[usd]"'
PACKAGES = {"pxr": ("usd-core", "usd-exchange"), "newton_usd_schemas": ("newton-usd-schemas",)}
SCENE = "demo_peg_d12"


def version_of(dists: tuple[str, ...]) -> str | None:
    """The installed version of the first of ``dists`` that is installed, as ``"<dist>==<version>"``."""
    for dist in dists:
        try:
            return f"{dist}=={md.version(dist)}"
        except md.PackageNotFoundError:
            continue
    return None


def packages(work: Path) -> dict[str, Any]:  # noqa: ARG001 - every step takes the work folder
    """Import the two modules USD reading needs; their distributions and versions."""
    row = {}
    for module, dists in PACKAGES.items():
        __import__(module)
        row[module] = version_of(dists)
    return row


def import_file(work: Path) -> dict[str, Any]:
    """The vendored Newton's cartpole.usda into card files under ``work``."""
    import alloy
    import alloy.io

    src = Path(alloy.__file__).parent / "_vendor" / "newton" / "examples" / "assets" / "cartpole.usda"
    res = alloy.io.import_file(src, work / "cards", name="cartpole")
    doc = res.to_json()
    return {"source": src.name, "robots": len(res.robots), "objects": len(res.objects), "issues": len(res.issues),
            "scene": Path(str(doc.get("scene", ""))).name}


def export_load(work: Path) -> dict[str, Any]:
    """Export a library scene, load the USD file it wrote and run it 0.1 s on the CPU."""
    import alloy
    import alloy.io

    usd = alloy.io.export_scene(SCENE, work / "export").to_json()["files"]["usd"]
    sim = alloy.load(str(usd), device="cpu")
    sim.run(0.1)
    return {"scene": SCENE, "file": Path(str(usd)).name, "bodies": len(sim.names()["bodies"]),
            "check_ok": bool(sim.check()["ok"])}


STEPS: tuple[tuple[str, Callable[[Path], dict[str, Any]]], ...] = (
    ("packages", packages), ("import_file", import_file), ("export_load", export_load))


def run() -> dict[str, Any]:
    """Run the steps in order, stopping at the first that fails; the JSON row."""
    t0 = time.time()
    doc: dict[str, Any] = {"ok": False}
    with tempfile.TemporaryDirectory() as tmp:
        for name, step in STEPS:
            try:
                doc[name] = step(Path(tmp))
            except Exception as exc:  # noqa: BLE001 - report what broke, e.g. a missing optional package
                doc.update(failed=name, error=f"{type(exc).__name__}: {str(exc)[:400]}",
                           fix=f"{FIX} (usd-core and newton-usd-schemas at Alloy's pins), then run this check again")
                break
        else:
            doc["ok"] = bool(doc["export_load"]["check_ok"] and doc["import_file"]["robots"]
                             + doc["import_file"]["objects"] > 0)
    doc["wall_s"] = round(time.time() - t0, 1)
    return doc


def main() -> int:
    """CLI entry: one JSON line on stdout."""
    doc = run()
    sys.stdout.write(json.dumps(doc) + "\n")
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
