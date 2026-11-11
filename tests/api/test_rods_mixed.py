# SPDX-License-Identifier: Apache-2.0
"""A chain rod and an elastic rod in one scene (launch blocker AGENTS-5, 2026-11-09), CPU.

An unclamped rod is a MuJoCo chain and a clamped one steps on the elastic-rod solver, so the two together are a
coupled scene, and their pair runs on the ADMM coupler. An ADMM entry's model view holds a prefix of the model's
bodies, and such a view returned its first colour-group list for every later override, so the rod solver's
per-iteration colour swap was lost and the first step raised IndexError in the VBD sweep, in either order of adding
the rods (fixed in the vendored view: tests/api/test_rods_model_view_memo.py).

Law: the two rods are 0.3 m apart and never touch, so each moves as it does alone in its own scene.
- The clamped rod swings down exactly as alone (every vertex within 10 um at every 0.1 s; measured 0), its clamped
  end does not move and its length holds to 0.1 %.
- The loose rod drops 47 mm onto the ground and lies there as alone: within 0.1 mm at 0.5 s (measured 0.014 mm; it
  differs by up to 0.26 mm while landing).
"""

from __future__ import annotations

import numpy as np
import pytest

import alloy

STEPS = (10, 20, 30, 40, 50)


def _held(sc: alloy.Scene) -> None:
    sc.add_rod("held", length=0.3, radius=0.003, pos=(0, 0.3, 0.3), direction=(1, 0, 0), clamp="start")


def _loose(sc: alloy.Scene) -> None:
    sc.add_rod("loose", length=0.3, radius=0.003, pos=(0, 0, 0.05), direction=(1, 0, 0))


def _run(*adds) -> tuple[dict[str, list[np.ndarray]], dict]:
    """Each rod's centre line at the start and after every 10 control periods, and the scene's routes."""
    sc = alloy.Scene(dt=0.01)
    sc.add_ground()
    for add in adds:
        add(sc)
    sim = alloy.build(sc, device="cpu")
    routes = {k: v["route"] for k, v in sim.routes().items()}
    names = [n for n in ("held", "loose") if n in routes]
    out = {n: [sim.state([n], world=0)[f"{n}.points"].copy()] for n in names}
    for _ in STEPS:
        sim.step(10)
        for n in names:
            out[n].append(sim.state([n], world=0)[f"{n}.points"].copy())
    assert sim.check()["ok"], sim.check()["problems"]
    sim.close()
    return out, routes


@pytest.fixture(scope="module")
def alone():
    return {"held": _run(_held)[0]["held"], "loose": _run(_loose)[0]["loose"]}


@pytest.mark.law
@pytest.mark.timeout(900)
@pytest.mark.parametrize("order", [(_loose, _held), (_held, _loose)], ids=["loose_first", "held_first"])
def test_a_chain_rod_and_a_clamped_rod_step_together_as_each_does_alone(order, alone):
    mixed, routes = _run(*order)
    assert routes["loose"] == "chain" and routes["held"] == "rod_vbd"
    held, loose = mixed["held"], mixed["loose"]
    gap = max(float(np.abs(h - a).max()) for h, a in zip(held[1:], alone["held"][1:], strict=True))
    rest = float(np.linalg.norm(loose[-1] - alone["loose"][-1], axis=1).max())
    print(f"held vs alone: {gap * 1e6:.3f} um over {len(STEPS)} reads; loose vs alone at rest: {rest * 1e3:.4f} mm")
    assert gap < 1.0e-5, gap
    assert rest < 1.0e-4, rest
    for p in held[1:]:
        assert float(np.abs(p[0] - held[0][0]).max()) < 1e-6                      # the clamped end does not move
        assert float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) == pytest.approx(0.3, rel=1e-3, abs=0)
    assert held[-1][-1, 2] < held[0][-1, 2] - 0.05                                # its free end swung down
    assert float(np.abs(loose[-1][:, 2] - 0.003).max()) < 2.0e-4                  # lying on the ground (r = 3 mm)
    assert float(loose[0][:, 2].min() - loose[-1][:, 2].max()) > 0.04             # after its 47 mm drop
