# SPDX-License-Identifier: Apache-2.0
"""Warm the Warp kernel cache before timed tests (SPEC §6.2): import every Alloy package (which registers plug-ins)
and take one step of each stock solver and Alloy core kernel on a tiny model. Not timed; run by
``scripts/ci_cpu.sh`` as ``python -m alloy.core.warm``.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import pkgutil
import time
from collections.abc import Callable

import warp as wp

import alloy

log = logging.getLogger("alloy.core")

STOCK = ("xpbd", "featherstone", "semi_implicit", "mujoco", "coupled")


def import_all() -> list[str]:
    """Import every ``alloy`` module (registration side effects); returns the modules that failed to import."""
    failed = []
    for info in pkgutil.walk_packages(alloy.__path__, prefix="alloy."):
        if ".contrib." in info.name or info.name.endswith("__main__"):
            continue
        try:
            importlib.import_module(info.name)
        except Exception as err:  # noqa: BLE001 - a broken package must not stop warming the others
            log.warning("warm: import %s failed: %s", info.name, err)
            failed.append(info.name)
    return failed


def _warm_stock(kind: str, device: str) -> None:
    from alloy.testing import scenes

    scene = scenes.box_on_cloth(1, 1.0, device=device) if kind == "coupled" else scenes.box_on_plane(1, device=device)
    runner = scene.runner(kind)
    runner.step(2)
    if kind == "mujoco":
        _warm_noslip(runner)


def _warm_noslip(runner: object) -> None:
    """MuJoCo Warp's noslip kernels (pinned mujoco-warp; skipped on a mujoco-warp without noslip)."""
    from alloy.core.newton_compat import enable_mujoco_noslip, mujoco_warp_has_noslip

    if mujoco_warp_has_noslip():
        enable_mujoco_noslip(runner.solver, 2)  # type: ignore[attr-defined]
        runner.step(2)  # type: ignore[attr-defined]


def _warm_core(device: str) -> None:
    from alloy.core.snapshot import WorldSnapshot, mask_from_worlds
    from alloy.testing import fakes, scenes

    scene = scenes.box_on_plane(2, device=device)
    state = scene.model.state()
    snap = WorldSnapshot(scene.model, state)
    snap.save(state)
    snap.restore(state, mask_from_worlds([0], 2, device))
    sim = fakes.FakeSimulation(scene.model, fakes.FakeStepper(fakes.FakeSolver(scene.model)),
                               health=fakes.FakeHealth(scene.model))
    sim.step_period()


def _warm_rodcheck(device: str) -> None:
    """The rod self-tests (SCR-OSS1-9): ``alloy.route.rodcheck.warm`` (WP2) fills ``$ALLOY_CACHE_DIR/rodcheck.json`` for
    the library's rod cards and the test scenes' cables, so no test pays a cold self-test (900 s on a fresh 4-CPU CI
    runner). Logged and skipped while rodcheck has no ``warm``."""
    from alloy.route import rodcheck

    fn = getattr(rodcheck, "warm", None)
    if fn is None:
        log.info("warm: alloy.route.rodcheck has no warm() yet (WP2, SCR-OSS1-9); rod self-tests stay lazy")
        return
    fn(device=device)


def warm(device: str = "cpu", solvers: tuple[str, ...] = STOCK) -> dict[str, float]:
    """Run every warm-up; returns seconds per item (failures are logged, never raised)."""
    times: dict[str, float] = {}
    jobs: list[tuple[str, Callable[[], object]]] = [("import", import_all), ("core", lambda: _warm_core(device))]
    jobs += [(k, (lambda k=k: _warm_stock(k, device))) for k in solvers]
    jobs.append(("rodcheck", lambda: _warm_rodcheck(device)))
    for name, job in jobs:
        t0 = time.perf_counter()
        try:
            job()
        except Exception as err:  # noqa: BLE001
            log.warning("warm: %s failed: %s", name, err)
        times[name] = time.perf_counter() - t0
        log.info("warm: %s %.1f s", name, times[name])
    return times


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--solvers", default=",".join(STOCK))
    args = ap.parse_args()
    wp.config.quiet = True
    warm(args.device, tuple(s for s in args.solvers.split(",") if s))


if __name__ == "__main__":
    main()
