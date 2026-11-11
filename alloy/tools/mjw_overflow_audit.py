# SPDX-License-Identifier: Apache-2.0
"""Run any Alloy command with a MuJoCo Warp overflow audit on (SCR-SC0-1 addendum: did a headline run overflow?).

    python -m alloy.tools.mjw_overflow_audit --out audit.json -- alloy.parallel.ladder_run --scene t12 ...
    python -m alloy.tools.mjw_overflow_audit --out audit.json -- bench/xbench/ours/alloy_t12.py --worlds 16 ...

Every ``SolverMuJoCo`` (subclasses included) built while the command runs gets a per-world int32 audit array,
allocated right after its constructor. After every ``step`` a kernel ORs ``mjw_data.overflow`` into it, on device,
so graph capture still works. MuJoCo Warp's bits are sticky until ``reset_data``; the OR also keeps what a per-world
reset clears.

At exit it writes one JSON record per solver:
- worlds, naconmax and njmax;
- the OR of all bits;
- per class (contact, rows, iterations, line search, sensor) the number of worlds that set it;
- the step count.

Exit code: the command's own.
"""

from __future__ import annotations

import argparse
import atexit
import json
import runpy
import sys
import weakref
from typing import Any

import numpy as np
import warp as wp

CLASSES = {  # mujoco_warp OverflowType bits by class (types.py at the pin)
    "contact": 4 | 8 | 16 | 32 | 256,  # BROADPHASE, NARROWPHASE, CCD, HFIELD, EPA_HORIZON: contacts dropped
    "rows": 1 | 2 | 128,               # NEFC, NJMAX_NNZ, NVMAX: constraint rows dropped
    "iterations": 512,                 # ITERATIONS
    "linesearch": 1024,                # LS_ITERATIONS
    "sensor": 64 | 2048,               # CONTACT_MATCH, TACTILE
}
_AUDITS: list[tuple[weakref.ref, wp.array, dict]] = []


@wp.kernel
def _or_bits(overflow: wp.array[wp.int32], seen: wp.array[wp.int32]):
    w = wp.tid()
    seen[w] = seen[w] | overflow[w]


def _install() -> None:
    from alloy._vendor.newton.solvers import SolverMuJoCo

    init, step = SolverMuJoCo.__init__, SolverMuJoCo.step

    def audited_init(self: Any, *args: Any, **kwargs: Any) -> None:
        init(self, *args, **kwargs)
        d = getattr(self, "mjw_data", None)
        if d is not None and getattr(d, "overflow", None) is not None:
            seen = wp.zeros(int(d.nworld), dtype=wp.int32, device=d.overflow.device)
            self._alloy_overflow_seen = seen
            _AUDITS.append((weakref.ref(self), seen, {"type": type(self).__name__, "worlds": int(d.nworld),
                                                      "naconmax": int(d.naconmax), "njmax": int(d.njmax),
                                                      "steps": 0}))  # steps: host step() calls (a graph replay does not count)

    def audited_step(self: Any, *args: Any, **kwargs: Any) -> Any:
        out = step(self, *args, **kwargs)
        seen = getattr(self, "_alloy_overflow_seen", None)
        if seen is not None:
            wp.launch(_or_bits, dim=seen.shape[0], inputs=[self.mjw_data.overflow, seen], device=seen.device)
            for ref, arr, info in _AUDITS:
                if arr is seen:
                    info["steps"] += 1
        return out

    SolverMuJoCo.__init__, SolverMuJoCo.step = audited_init, audited_step


def summary() -> list[dict]:
    out = []
    for _, seen, info in _AUDITS:
        bits = seen.numpy()
        rec = {**info, "bits_or": int(np.bitwise_or.reduce(bits)) if bits.size else 0}
        rec.update({f"worlds_{name}": int(np.count_nonzero(bits & mask)) for name, mask in CLASSES.items()})
        out.append(rec)
    return out


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    split = argv.index("--") if "--" in argv else len(argv)
    p = argparse.ArgumentParser(prog="python -m alloy.tools.mjw_overflow_audit")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv[:split])
    command = argv[split + 1:]
    if not command:
        p.error("give the command after --, e.g. -- alloy.parallel.ladder_run --scene t12")

    def write() -> None:
        wp.synchronize()
        with open(a.out, "w") as fh:
            json.dump({"command": command, "solvers": summary()}, fh, indent=1)

    _install()
    atexit.register(write)
    sys.argv = [command[0], *command[1:]]
    try:
        if command[0].endswith(".py"):  # a script path (bench/xbench/ours/*.py)
            runpy.run_path(command[0], run_name="__main__")
        else:
            runpy.run_module(command[0], run_name="__main__", alter_sys=True)
    except SystemExit as e:
        return int(e.code or 0) if not isinstance(e.code, str) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
