# SPDX-License-Identifier: Apache-2.0
"""Alloy beside a user's own ``newton`` / ``mujoco_warp`` in one process: kernels launched, Warp modules kept apart.

    python -I tools/deps/coexist_probe.py --first alloy|user --out row.json      (in a venv that has both)

Warp keeps ``module="unique"`` kernels by (qualified name, content hash) and named modules by name, not by Python
module. A user's copy of the engines can then take over a vendored engine's Warp module, or the reverse, and one copy
runs the other's kernel code (or fails on its struct types). The probe:

1. imports every submodule of both copies, ``--first`` first (``alloy``: Alloy's ``alloy._vendor.*``; ``user``: the
   top-level ``newton`` / ``mujoco_warp``);
2. steps an Alloy scene (robot, box, cloth) on the CPU's Warp route (the GPU's code path, so MuJoCo Warp's kernels
   launch too), then a cloth scene of the user's ``newton`` (in that order or the reverse);
3. checks Warp's module registry: no module that held kernels of the first copy got kernels of the second, and every
   module that holds a vendored kernel has a name private to Alloy (``alloy._vendor.*`` from the Python module, or
   ``alloy_vendor__*`` from ``alloy/_vendor/_warp.py``).

Writes one JSON row to ``--out`` (and stdout); exit 0 when everything ran and both rules held.

Example (Python):
    >>> unshared({"alloy_vendor__k_1a": ["alloy._vendor.newton.a"], "soft_face_contacts_False":
    ...           ["alloy._vendor.newton.b"], "newton._src.x": ["newton._src.x"]})
    ['soft_face_contacts_False']
    >>> taken_over({"m": {"k": "alloy"}, "n": {"k": "user"}}, {"m": {"k": "alloy"}, "n": {"k": "alloy"}})
    ['n']
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import pkgutil
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

log = logging.getLogger("alloy.deps.coexist")
VENDOR = "alloy._vendor."
PRIVATE = (VENDOR, "alloy_vendor__")
ENGINES = ("newton", "mujoco_warp")
# Optional parts (viewers, USD, tests, examples) that need packages a plain install lacks: skipped in both copies.
SKIP = (".tests", ".examples", ".viewer", ".usd", "._src.utils.selection")
ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="link1" pos="0 0 0.1">
  <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" damping="2"/>
  <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/></body></worldbody>
  <actuator><position joint="shoulder" kp="200" kv="10"/></actuator></mujoco>"""


def unshared(origins: dict[str, list[str]]) -> list[str]:
    """Warp modules (name -> the Python modules of their kernels) holding a vendored kernel under a name that is not
    private to Alloy, so a user's copy can share them (example in the module docstring)."""
    return sorted(n for n, mods in origins.items()
                  if any(m.startswith(VENDOR) for m in mods) and not n.startswith(PRIVATE))


def taken_over(before: dict[str, dict[str, str]], after: dict[str, dict[str, str]]) -> list[str]:
    """Warp modules where a kernel key's copy (key -> "alloy" / "user") changed between ``before`` and ``after``: the
    other copy registered its kernel under that key (example in the module docstring)."""
    return sorted(n for n, ks in before.items() if any(after.get(n, {}).get(k) != i for k, i in ks.items()))


def registry() -> dict[str, Any]:
    """Warp's module registry (name -> Module); raises naming the Warp version when its layout moved."""
    import warp

    for path in ("warp._src.context", "warp.context"):
        try:
            return importlib.import_module(path).user_modules
        except (ImportError, AttributeError):
            continue
    raise RuntimeError(f"warp {warp.__version__} has no user_modules registry: update registry() in "
                       "tools/deps/coexist_probe.py for this Warp")


def copy_of(python_module: str) -> str:
    """Which copy a kernel's Python module belongs to: "alloy" (vendored), "user" (top-level engines) or "other".

    >>> [copy_of(m) for m in ("alloy._vendor.newton._src.x", "newton._src.x", "alloy.solvers.x")]
    ['alloy', 'user', 'other']
    """
    if python_module.startswith(VENDOR):
        return "alloy"
    return "user" if python_module.split(".")[0] in ENGINES else "other"


def snapshot() -> tuple[dict[str, dict[str, str]], dict[str, list[str]]]:
    """(kernel key -> copy, kernel Python modules) per Warp module that has kernels."""
    copies, origins = {}, {}
    for name, module in list(registry().items()):
        if module.kernels:
            mods = {k: getattr(v.func, "__module__", "") or "" for k, v in module.kernels.items()}
            copies[name] = {k: copy_of(m) for k, m in mods.items()}
            origins[name] = sorted(set(mods.values()))
    return copies, origins


def borrowed() -> list[str]:
    """Kernels one copy's Python modules hold that the other copy defined (a reused ``module="unique"`` kernel: the
    second copy's decorator returned the first copy's kernel, bound to the first copy's struct types)."""
    import warp

    out = []
    for name, mod in list(sys.modules.items()):
        side = copy_of(name)
        if side == "other" or mod is None:
            continue
        for attr, obj in list(vars(mod).items()):
            if not isinstance(obj, warp.Kernel):
                continue
            if copy_of(getattr(obj.func, "__module__", "") or "") not in (side, "other"):
                out.append(f"{name}.{attr}")
    return sorted(out)[:25]


def walk(prefix: str) -> dict[str, int]:
    """Import every submodule of the engines under ``prefix`` ("" or VENDOR); counts imported and skipped."""
    done = failed = 0
    for top in ENGINES:
        pkg = importlib.import_module(prefix + top)
        for info in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
            if any(s in info.name for s in SKIP):
                continue
            try:
                importlib.import_module(info.name)
                done += 1
            except Exception as e:  # noqa: BLE001 - an optional dependency missing alike in both copies
                log.debug("skip %s: %s", info.name, e)
                failed += 1
    return {"imported": done, "failed": failed}


def alloy_step() -> dict[str, Any]:
    """A robot, a static box and a cloth through ``import alloy``, two control steps on the CPU's Warp route."""
    os.environ["ALLOY_CPU_ROUTE"] = "warp"  # the GPU's kernels (MuJoCo Warp too), not MuJoCo C
    import alloy

    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_robot("arm", mjcf=ARM, pos=(0, 0, 0.3))
    s.add_box("box", size=(0.1, 0.1, 0.1), pos=(0.3, 0, 0.05), static=True)
    s.add_cloth("sheet", size=(0.2, 0.2), pos=(0.15, 0, 0.5), resolution=4, mass=0.01)
    sim = alloy.build(s, device="cpu")
    sim.step(2)
    return {"ok": bool(sim.check()["ok"]), "routes": sorted({r["route"] for r in sim.routes().values()})}


def user_step() -> dict[str, Any]:
    """A cloth above a box body on the ground through the user's own ``newton``: collide + XPBD, two steps, CPU."""
    import warp as wp

    # The top-level copy by name: scripts/vendor_engines.py --alloy-only turns a bare ``import newton`` into Alloy's.
    newton = importlib.import_module(ENGINES[0])

    b = newton.ModelBuilder()
    b.add_ground_plane()
    body = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
    b.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    b.add_cloth_grid(pos=wp.vec3(-0.1, -0.1, 0.3), rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0), dim_x=4,
                     dim_y=4, cell_x=0.05, cell_y=0.05, mass=0.01)
    model = b.finalize(device="cpu")
    s0, s1, control = model.state(), model.state(), model.control()
    solver = newton.solvers.SolverXPBD(model)
    for _ in range(2):
        contacts = model.collide(s0)
        solver.step(s0, s1, control, contacts, 0.01)
        s0, s1 = s1, s0
    return {"ok": True, "newton": newton.__version__}


def timed(fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """``fn()`` with its seconds; an exception becomes ``{"ok": False, "error", "where"}``."""
    t0 = time.perf_counter()
    try:
        row = fn()
    except Exception as e:  # noqa: BLE001 - a failed step is reported in the row
        row = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:400]}", "where": traceback.format_exc()[-600:]}
    row["s"] = round(time.perf_counter() - t0, 1)
    return row


def probe(first: str) -> dict[str, Any]:
    """The row for one import order (``first``: "alloy" or "user"); see the module docstring."""
    t0 = time.perf_counter()
    prefixes = (VENDOR, "") if first == "alloy" else ("", VENDOR)
    walked = {"first": walk(prefixes[0])}
    at_first, _ = snapshot()
    walked["second"] = walk(prefixes[1])
    imported, _ = snapshot()
    steps = (("alloy", alloy_step), ("user", user_step))
    launched = {name: timed(fn) for name, fn in (steps if first == "alloy" else steps[::-1])}
    end, origins = snapshot()
    row = {"first": first, "walked": walked, "warp_modules": len(end), **launched,
           "taken_at_import": taken_over(at_first, imported), "taken_by_launch": taken_over(imported, end),
           "unshared_vendored": unshared(origins), "borrowed": borrowed()}
    rules = ("taken_at_import", "taken_by_launch", "unshared_vendored", "borrowed")
    row["ok"] = all(launched[n]["ok"] for n in launched) and not any(row[r] for r in rules)
    if not row["ok"]:
        row["fix"] = ("Alloy's vendored engines share Warp modules or kernels with the user's copy (taken_*, "
                      "unshared_vendored and borrowed name them): every vendored wp.kernel / wp.func / wp.get_module "
                      "that names a Warp module must go through alloy/_vendor/_warp.py (scripts/vendor_engines.py "
                      "--check); a failed step's error is under alloy / user")
    row["s"] = round(time.perf_counter() - t0, 1)
    return row


def main(argv: list[str] | None = None) -> int:
    """CLI: one import order, one JSON row to ``--out`` and stdout; exit 0 when it passed."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--first", choices=("alloy", "user"), required=True, help="which copy is imported first")
    ap.add_argument("--out", type=Path, help="write the JSON row here too")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    row = probe(args.first)
    text = json.dumps(row)
    if args.out:
        args.out.write_text(text + "\n")
    sys.stdout.write(text + "\n")
    return 0 if row["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
