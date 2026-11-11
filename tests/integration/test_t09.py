# SPDX-License-Identifier: Apache-2.0
"""WP12 acceptance: T09 (USB-A plug + 1 m cable into a port, yam bench) runs end to end on both
``native_contacts`` values (SPEC §7 WP12; D10 spike passed, SCR-WP3-2).

- CPU: the card scene assembles and runs ``PERIODS`` control periods through ``build_simulation`` with health, on
  both values, with MuJoCo / the Newton pipeline colliding every pair natively (``contact_layer="never"``): the
  precision pipeline needs texture SDFs, which need CUDA.
- GPU: the same with the contact layer (plug/receptacle precision pair appended into SolverMuJoCo under
  ``"mujoco"``, merged into the Newton buffer under ``"newton"``) and the gap monitor.

Prerequisites outside WP12 are checked first and reported as skips with the reason: the plug-cable weld
(attachments as WP10 bonds, WP2 branch) and, for the precision layer, collision filter pairs + feature SDFs for
the precision pair from the assembler (WP2/WP3).
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import warp as wp

from alloy.core.errors import AssemblyError
from alloy.core.registry import isolated_registry
from alloy.testing import integration as I

pytestmark = [pytest.mark.compile_heavy, pytest.mark.integration, pytest.mark.timeout(1200)]

PERIODS = 20
WORLDS = 2
CMD = "PYTHONPATH=. python -m pytest tests/integration/test_t09.py -q"


def _card():
    try:
        from alloy.cards.library import CardLibrary
    except ImportError:
        pytest.skip("WP1 CardLibrary not merged")
    lib = CardLibrary()
    return lib.scene("task_t09"), lib


def _built(device: str):
    card, lib = _card()
    built = I.assemble(card, lib, device, WORLDS)
    if card.attachments and built.bonds is None:
        pytest.skip("T09's plug-cable weld is not in the assembled scene (attachments as WP10 bonds: WP2 branch)")
    return built


def _run(built, **overrides):
    from alloy.integration import build_simulation

    sim = build_simulation(built, "standard", **overrides)
    for _ in range(PERIODS):
        sim.step_period()
    return sim


def _finite(sim) -> bool:
    ok = bool(np.isfinite(sim.state.body_q.numpy()).all())
    if sim.model.particle_count:
        ok = ok and bool(np.isfinite(sim.state.particle_q.numpy()).all())
    return ok


@pytest.mark.slow          # 140-200 s per value on the shared VM's CPU
@pytest.mark.parametrize("native", ["mujoco", "newton"])
def test_t09_cpu_native_collision(native):
    with isolated_registry(empty=True):
        sim = _run(_built("cpu"), native_contacts=native, contact_layer="never", graph="off")
        rep = sim.report()
    ok = _finite(sim) and not rep["health"].invalid_worlds
    I.record("WP12-T09", float(ok), "pass" if ok else "fail", CMD, device="cpu", native=native,
             precision_layer=False, periods=PERIODS, invalid=list(rep["health"].invalid_worlds))
    assert _finite(sim), "T09 went non-finite"
    assert not rep["health"].invalid_worlds, rep["health"]


@pytest.mark.gpu
@pytest.mark.parametrize("native", ["mujoco", "newton"])
def test_t09_gpu_precision_layer(native):
    if not wp.is_cuda_available():
        pytest.skip("needs CUDA")
    dev = os.environ.get("ALLOY_DEVICE", "cuda:0")
    with isolated_registry(empty=True):
        built = _built(dev)
        try:
            sim = _run(built, native_contacts=native)
        except AssemblyError as exc:
            I.record("WP12-T09", None, "not_run", CMD, device=dev, native=native, reason=str(exc)[:300])
            pytest.skip(f"precision pair not prepared by the assembler: {str(exc)[:200]}")
        rep = sim.report()
    ok = _finite(sim) and not rep["health"].invalid_worlds
    I.record("WP12-T09", float(ok), "pass" if ok else "fail", CMD, device=dev, native=native, precision_layer=True,
             periods=PERIODS, graph=sim.graph_stats(), invalid=list(rep["health"].invalid_worlds))
    assert type(sim.contact_layer).__name__ == "AlloyContactLayer" and sim.host_syncs == 0
    assert ok
