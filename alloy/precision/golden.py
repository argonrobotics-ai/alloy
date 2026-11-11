# SPDX-License-Identifier: Apache-2.0
"""Determinism goldens per environment tuple (SPEC §6.5; research precision §4.6).

``bench/goldens/<key>.json`` holds, per golden case, the SHA-256 of every world digest after each control period of
a fixed scene. The key names the environment (device, arch or CPU model, Warp version, Newton pin), not the Alloy
commit, so any change that moves bits on that environment shows up as a mismatch. Updating a golden is deliberate:
``python -m alloy.precision.golden update --reason "<why>"`` rewrites the file and the commit that lands it needs a
``golden-update: <reason>`` trailer (SPEC §6.5).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

from alloy.core.errors import DeterminismError

log = logging.getLogger("alloy.precision")

GOLDEN_DIR = Path(__file__).resolve().parents[2] / "bench" / "goldens"
SCHEMA = "alloy.precision.golden/1"


@dataclass(frozen=True)
class GoldenCase:
    name: str
    make_sim: Callable[[int], object] | None
    worlds: int
    periods: int
    series: Callable[[int, int], list[np.ndarray]] | None = None   # (worlds, periods) -> digests; replaces make_sim


def _cpu_cases() -> tuple[GoldenCase, ...]:
    from alloy.testing.precision import box_sim, pendulum_sim

    return (GoldenCase("box_drop5cm_xpbd", box_sim, 4, 4), GoldenCase("pendulum_featherstone", pendulum_sim, 4, 4))


def _gpu_cases() -> tuple[GoldenCase, ...]:
    import functools

    from alloy.precision import golden_feat as F
    from alloy.testing.precision import det_sim

    return (GoldenCase("box_drop27_steel_det", functools.partial(det_sim, "steel"), 64, 8),
            GoldenCase("box_drop27_xpbd_det", functools.partial(det_sim, "xpbd"), 64, 8),
            GoldenCase("box_drop27_fake", functools.partial(det_sim, "fake"), 64, 8),
            GoldenCase("feat_robot_randomize", None, 64, 8, F.robot_randomize_series),
            GoldenCase("feat_vectorenv_reset", None, 16, 8, F.vectorenv_reset_series),
            GoldenCase("feat_camera", None, 16, 4, F.camera_series))


def golden_cases(device: wp.context.Devicelike = None) -> tuple[GoldenCase, ...]:
    return _gpu_cases() if wp.get_device(device).is_cuda else _cpu_cases()


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def golden_key(device: wp.context.Devicelike = None) -> str:
    """``cuda-sm89-warp1.18.0-newton8f3466f9`` or ``cpu-x86_64-<cpu model hash>-warp...-newton...``."""
    from alloy.core.newton_compat import NEWTON_PIN

    dev = wp.get_device(device)
    if dev.is_cuda:
        where = f"cuda-sm{dev.arch}"
    else:
        where = f"cpu-{platform.machine()}-{hashlib.sha256(_cpu_model().encode()).hexdigest()[:8]}"
    return f"{where}-warp{wp.config.version}-newton{NEWTON_PIN[:8]}"


def case_digest(case: GoldenCase) -> dict[str, object]:
    """SHA-256 over the world digests after every period of the case (host sync)."""
    from alloy.precision.repeat import run_digests

    if case.series is not None:
        series = case.series(case.worlds, case.periods)
    else:
        series = run_digests(case.make_sim(case.worlds), case.periods)
    h = hashlib.sha256(f"{SCHEMA}\0{case.name}\0{case.worlds}\0{case.periods}\0".encode())
    for d in series:
        h.update(np.ascontiguousarray(d, dtype="<u8").tobytes())
    return {"sha256": h.hexdigest(), "worlds": case.worlds, "periods": case.periods,
            "worlds_equal_at_end": bool((series[-1] == series[-1][0]).all())}


def golden_path(key: str) -> Path:
    return GOLDEN_DIR / f"{key}.json"


def load_golden(key: str) -> dict | None:
    path = golden_path(key)
    return json.loads(path.read_text()) if path.exists() else None


def check_goldens(device: wp.context.Devicelike = None) -> dict[str, str]:
    """Compare every case with the stored golden for this environment: name -> "equal" | "missing". Raises
    ``DeterminismError`` on a mismatch."""
    key = golden_key(device)
    stored = load_golden(key)
    out: dict[str, str] = {}
    for case in golden_cases(device):
        want = (stored or {}).get("cases", {}).get(case.name)
        if want is None:
            out[case.name] = "missing"
            continue
        got = case_digest(case)
        if got["sha256"] != want["sha256"]:
            raise DeterminismError(f"golden {key}/{case.name} changed: {want['sha256'][:16]} -> {got['sha256'][:16]}; "
                                   "if intended: python -m alloy.precision.golden update --reason ... and a "
                                   "'golden-update: <reason>' commit trailer")
        out[case.name] = "equal"
    return out


def update_goldens(reason: str, device: wp.context.Devicelike = None) -> Path:
    if not reason.strip():
        raise ValueError("a golden update needs a reason")
    key = golden_key(device)
    data = {"schema": SCHEMA, "key": key, "reason": reason, "cpu_model": _cpu_model(),
            "cases": {c.name: case_digest(c) for c in golden_cases(device)}}
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    path = golden_path(key)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    log.info("golden %s written (%s)", path, reason)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m alloy.precision.golden")
    sub = parser.add_subparsers(dest="cmd", required=True)
    up = sub.add_parser("update")
    up.add_argument("--reason", required=True)
    up.add_argument("--device", default=None)
    chk = sub.add_parser("check")
    chk.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    if args.cmd == "update":
        sys.stdout.write(f"{update_goldens(args.reason, args.device)}\n")
        return 0
    sys.stdout.write(json.dumps(check_goldens(args.device)) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
