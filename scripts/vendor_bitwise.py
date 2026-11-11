# SPDX-License-Identifier: Apache-2.0
"""Bitwise law for the engine vendoring (coordinator ruling 2026-11-07): the same scenes, built and stepped before
(main, engines from the venv) and after (vendored engines), give identical model digests, per-world state digests
and recorded trajectories, bit for bit.

    python scripts/vendor_bitwise.py run --out DIR [--device cpu|cuda:0] [--worlds N] [--case NAME ...]
    python scripts/vendor_bitwise.py compare DIR_BEFORE DIR_AFTER

BLOCK_ENGINES=1 makes a top-level newton / mujoco_warp import raise (use it for the vendored arm).

``run`` uses only Alloy's public API plus two digest helpers that exist on both sides, so the same file runs in either
tree (PYTHONPATH=<tree>). Each case runs in its own process; ALLOY_CPU_ROUTE picks the CPU route as usual.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("alloy.scripts.vendor_bitwise")

ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="link1" pos="0 0 0.1">
  <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" damping="2"/>
  <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/>
  </body></worldbody>
  <actuator><position joint="shoulder" kp="200" kv="10"/></actuator></mujoco>"""


def _boxes(alloy: Any, worlds: int, device: str) -> tuple[Any, float, dict]:
    s = alloy.Scene(worlds=worlds, dt=0.01)
    s.add_ground()
    s.add_box("table", size=(0.6, 0.6, 0.4), pos=(0, 0, 0.2), static=True)
    s.add_box("block", size=(0.05, 0.05, 0.05), pos=(0, 0, 0.6), mass=0.2)
    s.add_sphere("ball", 0.03, pos=(0.1, 0.05, 0.7), material="steel_carbon")
    return alloy.build(s, device=device, seed=0), 0.6, {}


def _arm(alloy: Any, worlds: int, device: str) -> tuple[Any, float, dict]:
    s = alloy.Scene(worlds=worlds, dt=0.01)
    s.add_ground()
    s.add_robot("arm", mjcf=ARM, pos=(0, 0, 0.5))
    return alloy.build(s, device=device, seed=0), 0.5, {"targets": {"shoulder": (0.0, 0.8)}}


def _drive(alloy: Any, worlds: int, device: str) -> tuple[Any, float, dict]:
    s = alloy.Scene(worlds=worlds, dt=0.01)
    s.add_ground()
    s.add_box("pusher", size=(0.05, 0.2, 0.1), pos=(-0.2, 0, 0.05), kinematic=True)
    s.add_box("crate", size=(0.1, 0.1, 0.1), pos=(0, 0, 0.05), mass=0.5)
    sim = alloy.build(s, device=device, seed=0)
    sim.drive("pusher", velocity=(0.3, 0, 0), duration=0.5)
    return sim, 0.5, {}


def _library(name: str, seconds: float, **kw: Any) -> Callable[[Any, int, str], tuple[Any, float, dict]]:
    def make(alloy: Any, worlds: int, device: str) -> tuple[Any, float, dict]:
        return alloy.load(name, worlds=worlds, device=device, **kw), seconds, {}
    return make


CASES: dict[str, Callable[[Any, int, str], tuple[Any, float, dict]]] = {
    "boxes": _boxes, "arm": _arm, "drive": _drive,
    "cloth_drop": _library("demo_cloth_drop", 0.2), "pour": _library("demo_pour", 0.2),
    "peg_d12": _library("demo_peg_d12", 0.1, profile="precision"),
    "panda": _library("task_emb_e2_panda", 0.2),
}


def _sha(arr: Any) -> str:
    import numpy as np  # noqa: PLC0415

    a = np.ascontiguousarray(arr)
    return hashlib.sha256(f"{a.dtype.str}|{a.shape}|".encode() + a.tobytes()).hexdigest()


class _BlockEngines:
    """BLOCK_ENGINES=1 (the vendored arm): importing a top-level newton / mujoco_warp raises, so a missed rewrite
    cannot quietly load the venv's copy (the same pin, bit for bit) and pass the law (WP2's review)."""

    def find_spec(self, fullname: str, path: object = None, target: object = None) -> None:
        if fullname.split(".", 1)[0] in ("newton", "mujoco_warp"):
            raise ImportError(f"BLOCK_ENGINES: top-level {fullname} imported")
        return None


def _engine_modules() -> list[str]:
    """Top-level engine packages loaded in this process (the before arm: newton, mujoco_warp; after: none)."""
    return sorted({m.split(".", 1)[0] for m in sys.modules if m.split(".", 1)[0] in ("newton", "mujoco_warp")})


def run_case(name: str, out: Path, worlds: int, device: str) -> dict[str, Any]:
    if os.environ.get("BLOCK_ENGINES") == "1":
        sys.meta_path.insert(0, _BlockEngines())
    import alloy  # noqa: PLC0415
    from alloy.scene.assembler import digest_model  # noqa: PLC0415
    from alloy.testing.integration import digests  # noqa: PLC0415

    sim, seconds, run_kw = CASES[name](alloy, worlds, device)
    names = sim.names()
    record = [n for key in ("bodies", "joints", "particles") for n in names.get(key, [])]
    model_digest = digest_model(sim.engine.model)
    traj = sim.run(seconds, record=record, **run_kw)
    arrays = {k: traj[k] for k in sorted(traj.keys())}
    row = {"case": name, "device": device, "worlds": worlds, "route": os.environ.get("ALLOY_CPU_ROUTE", ""),
           "model": model_digest, "state": digests(sim.engine).tobytes().hex(),
           "arrays": {k: _sha(v) for k, v in arrays.items()}, "steps": int(sim.physics_steps),
           "engines": _engine_modules(), "blocked": os.environ.get("BLOCK_ENGINES") == "1"}
    import numpy as np  # noqa: PLC0415

    np.savez_compressed(out / f"{name}.npz", **{k.replace("/", "__"): v for k, v in arrays.items()})
    return row


def cmd_run(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rc = 0
    for name in args.case or list(CASES):
        if args.child:
            row = run_case(name, out, args.worlds, args.device)
            (out / f"{name}.json").write_text(json.dumps(row, indent=1, sort_keys=True))
            continue
        cmd = [sys.executable, __file__, "run", "--out", str(out), "--device", args.device, "--worlds",
               str(args.worlds), "--case", name, "--child"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        (out / f"{name}.log").write_text(r.stdout + r.stderr)
        log.info("%s: exit %d", name, r.returncode)
        rc = rc or r.returncode
    return rc


def cmd_compare(args: argparse.Namespace) -> int:
    a, b = Path(args.before), Path(args.after)
    bad, same = [], 0
    for fa in sorted(a.glob("*.json")):
        fb = b / fa.name
        if not fb.exists():
            bad.append(f"{fa.stem}: missing after")
            continue
        ra, rb = json.loads(fa.read_text()), json.loads(fb.read_text())
        for key in ("model", "state", "steps"):
            if ra[key] != rb[key]:
                bad.append(f"{fa.stem}: {key} differs")
        diff = sorted(k for k in set(ra["arrays"]) | set(rb["arrays"]) if ra["arrays"].get(k) != rb["arrays"].get(k))
        bad += [f"{fa.stem}: array {k} differs" for k in diff[:10]]
        same += not diff and ra["model"] == rb["model"] and ra["state"] == rb["state"]
    engines = {f.stem: json.loads(f.read_text()).get("engines") for f in sorted(b.glob("*.json"))}
    loaded = {k: v for k, v in engines.items() if v}
    if loaded:
        print(f"after arm loaded top-level engines: {loaded}")
    print(f"bitwise identical cases: {same}; differences: {len(bad)}")
    for line in bad:
        print("  " + line)
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--device", default="cpu")
    r.add_argument("--worlds", type=int, default=2)
    r.add_argument("--case", action="append", choices=sorted(CASES))
    r.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    c = sub.add_parser("compare")
    c.add_argument("before")
    c.add_argument("after")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return cmd_run(args) if args.cmd == "run" else cmd_compare(args)


if __name__ == "__main__":
    sys.exit(main())
