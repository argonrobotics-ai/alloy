# SPDX-License-Identifier: Apache-2.0
"""Release regression check of an INSTALLED Alloy (the wheel in a clean venv), never of a source checkout.

    <venv>/bin/python -I tools/deps/wheel_smoke.py [--out result.json]

``-I`` keeps this script's folder, the working directory and ``PYTHONPATH`` off ``sys.path``, so ``import alloy``
finds the installed wheel even when run from a clone. Each check is one row of the JSON report:

- ``installed``: alloy imports from this interpreter's site-packages and no Alloy checkout is on ``sys.path``.
- ``package_data``: the files the code reads at run time are installed (C route sources, BSDF tables, OptiX
  sources, tactile calibrations, licence notices).
- ``quick_start``: AGENTS.md section 2 on the CPU. The card library comes from the installed package
  (``alloy/_data/library``, SCR-OSS2-2), and the block, on its default material, rests on the table top.
- ``robot_cpu_c``: AGENTS.md section 5's arm on the CPU. The C step (``alloy/route/_mjc/step.c``, compiled on
  first use) must load and the shoulder must follow its ramp to 0.8 rad.
- ``camera``: one ``alloy.camera`` render of that scene with the fast tracer and one with full materials (which
  read the Cycles BSDF tables).
- ``tactile``: a GelSight sensor on the arm reads the sim, and a 1 mm dome pressed into the calibration changes the
  image.
- ``vendored_engine`` (last): Newton and MuJoCo Warp come from ``alloy/_vendor`` with their LICENSE files and
  ``VENDORED.md``; no installed Alloy file imports a top-level ``newton`` or ``mujoco_warp`` (parsed, so docstrings
  that name them are fine), and none was imported while the checks above ran. A user's own newton or mujoco-warp in
  the same venv is reported, never used.

Prints one JSON document (also written to ``--out``); exit 0 when every check passes, 1 otherwise. A failed row
carries the error and a fix. Expect 2 to 6 min on a 4-core CPU on a cold kernel cache, under 1 min warm.

Example (Python):
    >>> sorted(PACKAGE_FILES)[:2]
    ['alloy.camera', 'alloy.camera.optix']
"""

from __future__ import annotations

import argparse
import ast
import importlib.metadata as md
import json
import logging
import sys
import sysconfig
import time
import traceback
from collections.abc import Callable
from importlib import resources
from pathlib import Path
from typing import Any

log = logging.getLogger("alloy.deps")

ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="link1" pos="0 0 0.1">
  <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" damping="2"/>
  <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/>
  </body></worldbody>
  <actuator><position joint="shoulder" kp="200" kv="10"/></actuator></mujoco>"""  # AGENTS.md section 5, unchanged
TARGET_RAD, TARGET_TOL = 0.8, 0.05
REST_Z, REST_TOL = 0.425, 0.003  # quick start: the 5 cm block's centre on the 0.4 m table top (m)
ENGINE = {"newton": "newton", "mujoco_warp": "mujoco-warp"}  # vendored module -> its upstream distribution
# Run-time files per package (a glob each must match): what F1 (2026-11-08) found missing from the release wheel.
PACKAGE_FILES = {
    "alloy.route._mjc": ["step.c", "noslip.c", "cone.c", "boxbox_line.c", "alloy_mjc.h", "mjc_internal.h"],
    "alloy.camera.pbr.bsdf": ["cycles_tables.npz"],
    "alloy.camera.optix": ["csrc/owf.cpp", "csrc/owf_programs.cu", "csrc/owf_params.h"],
    "alloy.camera.tactile": ["data/gelsight.npz", "data/digit.npz", "data/LICENSE-Taxim.txt"],
    "alloy.camera": ["LICENSING.md", "THIRD_PARTY.md"],
    "alloy.contact.vendor": ["LICENSE-mujoco_warp"],
}


def check_installed() -> dict[str, Any]:
    """alloy comes from site-packages, and no directory on sys.path holds an Alloy checkout."""
    import alloy

    site = Path(sysconfig.get_paths()["purelib"]).resolve()
    where = Path(alloy.__file__).resolve()
    checkouts = [p for p in sys.path if p and (Path(p) / "alloy" / "__init__.py").is_file()
                 and (Path(p) / "pyproject.toml").is_file()]
    ok = site in where.parents and not checkouts
    row = {"ok": ok, "alloy": str(where), "version": alloy.__version__, "checkouts_on_path": checkouts}
    if not ok:
        row["fix"] = "run with the venv's python -I from outside the clone, with alloy installed from the wheel"
    return row


def check_package_data() -> dict[str, Any]:
    """Every file in ``PACKAGE_FILES`` is installed with its package."""
    missing = [f"{pkg}/{name}" for pkg, names in PACKAGE_FILES.items() for name in names
               if not resources.files(pkg).joinpath(name).is_file()]
    row = {"ok": not missing, "files": sum(map(len, PACKAGE_FILES.values())), "missing": missing}
    if missing:
        row["fix"] = "rebuild the wheel with tools/deps/wheels.py (it ships the whole package and checks it)"
    return row


def check_quick_start() -> dict[str, Any]:
    """AGENTS.md section 2 on the CPU, with the card library the installed package carries: the block, on its
    default material, ends 1 s later at rest on the table top."""
    import alloy
    from alloy.cards.library import default_root

    root = default_root().resolve()
    materials = len(list((root / "materials").glob("*.yaml"))) if (root / "materials").is_dir() else 0
    row: dict[str, Any] = {"library": str(root), "in_package": Path(alloy.__file__).resolve().parent in root.parents,
                           "materials": materials}
    if not (row["in_package"] and materials):
        return row | {"ok": False, "fix": "the installed package carries no card library at alloy/_data/library: "
                      "build it with tools/deps/wheels.py, or with the build step that ships library/ (SCR-OSS2-2)"}
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_box("table", size=(0.6, 0.6, 0.4), pos=(0, 0, 0.2), static=True)
    scene.add_box("block", size=(0.05, 0.05, 0.05), pos=(0, 0, 0.6), mass=0.2)
    sim = alloy.build(scene, device="cpu")
    traj = sim.run(1.0, record=["block"])
    z, health = float(sim.state(world=0)["block.pos"][2]), sim.check()["ok"]
    ok = abs(z - REST_Z) < REST_TOL and health and traj["block.pos"].shape[1:] == (1, 3)
    row |= {"ok": ok, "block_z_m": round(z, 4), "rest_z_m": REST_Z, "health_ok": health}
    return row if ok else row | {"fix": f"the block ended at z = {z:.4f} m, not {REST_Z}: the quick start's physics "
                                 "changed (see sim.check())"}


def build_scene(robot: bool = True) -> Any:
    """AGENTS.md section 5's arm (or, when that fails to build, a 5 cm block), built on the CPU."""
    import alloy

    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    if robot:
        scene.add_robot("arm", mjcf=ARM, pos=(0, 0, 0.5))
    else:
        scene.add_box("block", size=(0.05, 0.05, 0.05), pos=(0, 0, 0.1), mass=0.2)
    return alloy.build(scene, device="cpu")


def check_robot(sim: Any) -> dict[str, Any]:
    """The arm follows its ramp on the CPU route, stepped by the C extension built from the installed step.c."""
    from alloy.route import _mjc
    from alloy.route._mjc.build import library_path

    sim.run(1.0, record=["arm/shoulder"], targets={"shoulder": (0.0, TARGET_RAD)})
    sim.set_targets({"shoulder": TARGET_RAD})
    sim.step(50)
    q = float(sim.state(world=0)["arm/shoulder.q"])
    lib = _mjc.load()
    health = sim.check()["ok"]
    ok = lib is not None and abs(q - TARGET_RAD) < TARGET_TOL and health
    row = {"ok": ok, "q_rad": round(q, 4), "target_rad": TARGET_RAD, "health_ok": health,
           "c_library": str(library_path()) if lib is not None else None, "c_unavailable": _mjc.reason or None,
           "route": {k: v.get("route") for k, v in sim.routes().items()}}
    if lib is None:
        row["fix"] = "the C step did not load (see c_unavailable): install a C compiler (cc) and check step.c ships"
    elif not ok:
        row["fix"] = f"the shoulder ended at {q:.3f} rad, not {TARGET_RAD}: the CPU route's physics changed"
    return row


def check_camera(sim: Any) -> dict[str, Any]:
    """One fast-tracer and one full-material render of world 0; both must show something."""
    fast = sim.render(width=64, height=48, spp=4)
    full = sim.render(width=32, height=24, spp=1, quality="full")
    stats = {name: {"shape": list(img.shape), "std": round(float(img.std()), 2)} for name, img in
             (("fast", fast), ("full", full))}
    ok = all(s["std"] > 1.0 for s in stats.values()) and fast.shape == (48, 64, 3) and full.shape == (24, 32, 3)
    return {"ok": ok, **stats} | ({} if ok else {"fix": "a render came back blank: see alloy.camera's log"})


def _dome(height: int, width: int, mm_per_px: float, depth_mm: float = 1.0, radius_mm: float = 4.0) -> Any:
    """(1, H, W) press map of a sphere cap ``depth_mm`` deep and ``radius_mm`` wide at the image centre."""
    import numpy as np

    yy, xx = np.mgrid[:height, :width]
    r = np.hypot(xx - width / 2, yy - height / 2) * mm_per_px
    return (depth_mm * np.clip(1 - (r / radius_mm) ** 2, 0, None))[None].astype(np.float32)


def check_tactile(sim: Any) -> dict[str, Any]:
    """A gel on the arm's tip (or the block) reads the sim (no touch: press 0), and a 1 mm dome visibly changes a
    GelSight image."""
    import numpy as np
    from alloy.camera.tactile import TactileRenderer, TactileSensor

    body = "arm/link1" if "arm/link1" in sim.names()["bodies"] else "block"
    sensor = TactileSensor("gel", attach_to=body, pose=((0.3 if body == "arm/link1" else 0.0, 0.0, 0.0),
                                                        (0.0, 0.0, 0.0, 1.0)), preset="gelsight", width=64)
    out = sensor.render(sim, output="all")
    read = {"shape": list(out.rgb.shape), "press_max_mm": round(float(out.press.numpy().max()), 4)}
    r = TactileRenderer("gelsight", width=64, num_worlds=1)
    flat = r.render(np.zeros((1, r.height, r.width), np.float32)).numpy().astype(np.int32)
    pressed = r.render(_dome(r.height, r.width, r.mm_per_px)).numpy().astype(np.int32)
    change = float(np.abs(pressed - flat).max())
    ok = read["shape"][1:] == [r.height, r.width, 3] and np.isfinite(read["press_max_mm"]) and change >= 10
    row = {"ok": bool(ok), "sensor": read, "dome_max_change": change}
    return row if ok else row | {"fix": "the tactile image did not respond to a 1 mm press: check data/gelsight.npz"}


def engine_imports(root: Path) -> list[str]:
    """``file:line`` of each real ``import newton`` / ``from mujoco_warp ...`` (absolute, top-level) under ``root``.

    >>> import tempfile; d = Path(tempfile.mkdtemp())
    >>> _ = (d / "a.py").write_text('s = "import newton"\\nimport newton.x\\n')
    >>> [s.split(":")[-1] for s in engine_imports(d)]
    ['2']
    """
    hits = []
    for f in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(f.read_bytes(), str(f))
        except (SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            if any(n.split(".")[0] in ENGINE for n in names):
                hits.append(f"{f.relative_to(root.parent)}:{node.lineno}")
    return hits


def _separate_install(dist: str) -> str | None:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def check_vendored() -> dict[str, Any]:
    """Alloy's engine is its vendored copy: present with licences, imported, and never a top-level newton."""
    import alloy

    root = Path(alloy.__file__).resolve().parent
    vendor = root / "_vendor"
    copies = {m: {"package": (vendor / m / "__init__.py").is_file(), "licence": any((vendor / m).glob("LICENSE*"))}
              for m in ENGINE}
    loaded = {m: Path(sys.modules[f"alloy._vendor.{m}"].__file__).resolve().is_relative_to(vendor)
              for m in ENGINE if f"alloy._vendor.{m}" in sys.modules}
    stray = engine_imports(root)
    leaked = sorted(m for m in ENGINE if m in sys.modules)
    ok = (all(c["package"] and c["licence"] for c in copies.values()) and (vendor / "VENDORED.md").is_file()
          and all(loaded.values()) and len(loaded) == len(ENGINE) and not stray and not leaked)
    row = {"ok": ok, "copies": copies, "vendored_md": (vendor / "VENDORED.md").is_file(), "imported": loaded,
           "top_level_imports": stray[:10], "top_level_loaded": leaked,
           "separate_install": {d: _separate_install(d) for d in ENGINE.values()}}
    if not ok:
        row["fix"] = ("Alloy must ship and use only alloy/_vendor/{newton,mujoco_warp} with their LICENSE files and "
                      "_vendor/VENDORED.md: rewrite the listed imports to alloy._vendor.*")
    return row


def _run(name: str, fn: Callable[..., dict[str, Any]], *args: Any) -> dict[str, Any]:
    """One check with its wall time; an exception becomes a failed row with its last traceback lines."""
    t0 = time.perf_counter()
    try:
        row = fn(*args)
    except Exception as exc:  # noqa: BLE001 - every failure is reported, the next check still runs
        row = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
               "trace": traceback.format_exc().strip().splitlines()[-6:],
               "fix": "a file the installed package needs may be missing: rebuild the wheel with tools/deps/wheels.py"}
    log.info("%s: %s (%.1f s)", name, "ok" if row["ok"] else "FAILED", time.perf_counter() - t0)
    return {"check": name, "seconds": round(time.perf_counter() - t0, 1), **row}


def main(argv: list[str] | None = None) -> int:
    """Run every check; one JSON document on stdout (and in ``--out``); exit 0 all passed, 1 otherwise."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, help="also write the JSON report here")
    args = ap.parse_args(argv)
    rows = [_run("installed", check_installed), _run("package_data", check_package_data),
            _run("quick_start", check_quick_start)]
    held: dict[str, Any] = {}
    rows.append(_run("build_arm", lambda: held.update(sim=build_scene()) or {"ok": True}))
    rows.append(_run("robot_cpu_c", check_robot, held["sim"]) if "sim" in held else
                {"check": "robot_cpu_c", "ok": False, "error": "skipped: the arm scene did not build"})
    if "sim" not in held:  # still run the camera and tactile checks, on a block
        rows.append(_run("build_block", lambda: held.update(sim=build_scene(robot=False)) or {"ok": True}))
    for name, fn in (("camera", check_camera), ("tactile", check_tactile)):
        rows.append(_run(name, fn, held["sim"]) if "sim" in held else
                    {"check": name, "ok": False, "error": "skipped: no scene built"})
    rows.append(_run("vendored_engine", check_vendored))
    doc = {"ok": all(r["ok"] for r in rows), "python": sys.version.split()[0], "checks": rows}
    text = json.dumps(doc, indent=1)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    sys.exit(main())
