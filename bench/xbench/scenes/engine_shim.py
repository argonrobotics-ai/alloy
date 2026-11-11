# SPDX-License-Identifier: Apache-2.0
"""Run one xbench scene script with the engine under test selected: Newton (the baseline) or Alloy.

Every xbench scene script (scenes/*_newton.py) builds a Newton model and constructs a stock Newton solver through
`newton.solvers.<SolverX>(model, **kwargs)` at call time (grep `newton.solvers.Solver` in scenes/). This shim lets
`run_all.py --engine-under-test alloy` time Alloy with the same scene, the same settings and the same clock, without
editing the scene scripts:

  newton  run the script unchanged.
  alloy   import `alloy.xbench` and call `install(protocol=...)` BEFORE the script is executed. The adapter must
          replace the solver classes in `newton.solvers` that the scene uses (SolverMuJoCo, SolverXPBD,
          SolverFeatherstone, and whatever Newton's asv example classes construct for the `newton` protocol) with
          Alloy solvers that accept the same constructor arguments, and return a JSON-serialisable dict saying what it
          swapped (Alloy commit, solver map). Scenes that read MuJoCo Warp internals (`solver.mjw_model`,
          `solver.mjw_data`: mujoco_warp_newton.py, genesis_newton.py contact stats) need those attributes or a
          per-protocol adapter path; the adapter is told the protocol for that reason.

If `alloy.xbench` is missing the shim exits with code 3 and an `XB_ENGINE_ERROR:` line, so the Alloy column stays
"pending" instead of silently timing Newton under Alloy's name.

Usage: python engine_shim.py --engine {newton,alloy} --protocol NAME -- SCRIPT [script args...]
Prints one `XB_ENGINE:{json}` line (engine provenance) before the script's own output.

A scene that starts child Python processes (newton_newton.py runs one interpreter per benchmark, like ASV) must
start them with `child_command(...)`: the shim exports the engine and protocol (`XB_ENGINE_UNDER_TEST`,
`XB_PROTOCOL`), and `child_command` sends the child through this shim again, so an Alloy run never times a stock
Newton child under Alloy's name.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import runpy
import sys
from pathlib import Path

ENGINES = ("newton", "alloy")
EXIT_NO_ADAPTER = 3
REPO = Path(__file__).resolve().parents[3]
ENGINE_ENV, PROTOCOL_ENV = "XB_ENGINE_UNDER_TEST", "XB_PROTOCOL"


def child_command(script: str, args: list[str]) -> list[str]:
    """Command for a child interpreter running ``script``: through this shim with the parent's engine when the
    parent ran under the shim, else the plain script."""
    engine, protocol = os.environ.get(ENGINE_ENV), os.environ.get(PROTOCOL_ENV)
    if engine:
        return [sys.executable, str(Path(__file__).resolve()), "--engine", engine, "--protocol", protocol or "",
                str(script), *args]
    return [sys.executable, str(script), *args]


def install_engine(engine: str, protocol: str) -> dict:
    """Select the engine under test; return provenance for the result rows."""
    from alloy._vendor import newton  # every scene needs it; fail early with a clear error if the venv lacks it

    info: dict = {"engine_under_test": engine, "protocol": protocol, "newton_version": newton.__version__}
    os.environ[ENGINE_ENV], os.environ[PROTOCOL_ENV] = engine, protocol   # children inherit (child_command)
    if engine == "newton":
        return info
    sys.path.insert(0, str(REPO))  # the alloy package lives at the repo root (also fine if it is pip-installed)
    try:
        adapter = importlib.import_module("alloy.xbench")
    except ImportError as exc:
        print(f"XB_ENGINE_ERROR: alloy.xbench adapter not importable ({exc}); Alloy column stays pending", flush=True)
        sys.exit(EXIT_NO_ADAPTER)
    swapped = adapter.install(protocol=protocol)
    info["alloy"] = swapped if isinstance(swapped, dict) else {"install_returned": repr(swapped)}
    return info


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--engine", choices=ENGINES, required=True)
    p.add_argument("--protocol", required=True)
    p.add_argument("script", type=Path)
    p.add_argument("script_args", nargs=argparse.REMAINDER)
    args = p.parse_args()
    rest = args.script_args[1:] if args.script_args[:1] == ["--"] else args.script_args
    info = install_engine(args.engine, args.protocol)
    print("XB_ENGINE:" + json.dumps(info), flush=True)
    script = args.script.resolve()
    sys.argv = [str(script), *rest]
    sys.path.insert(0, str(script.parent))
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
