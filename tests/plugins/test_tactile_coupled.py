# SPDX-License-Identifier: Apache-2.0
"""Contact-force plug-ins in coupled scenes (S1-009, Gemini audit of 0.1.2).

A scene with a robot beside liquid, cloth, a soft body or sand steps on a coupled stepper. ``Tactile`` and
``GelSensor`` there raised a BuildError naming "method" (the host took the stepper's ``solver``, a coupled stepper's
lookup method). Finding the MuJoCo entry alone was not enough: the coupled step hands that entry its own contacts, so
the host asked for no force export and every pad read 0 N with ``sim.check()`` ok. The law: a pad pressing a static
block reads the same force beside a coupled partner as in the rigid-only scene."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import alloy
from alloy.plugins.base import SensorPlugin
from alloy.plugins.errors import PluginError
from alloy.plugins.host import contact_solver
from alloy.plugins.tactile.coupled_law import BOUND, PAD, PRESS_M, PRESS_S, PROBE, press_scene
from alloy.plugins.tactile import GelSensor, Tactile



def _scene(partner: str) -> alloy.Scene:
    s = press_scene(partner == "liquid")                       # the law's scene (coupled_law.py)
    s.cloth_accuracy = "fast"
    if partner == "cloth":
        s.add_cloth("towel", size=(0.10, 0.10), pos=(0.30, 0.0, 0.02), resolution=6, mass=0.03, self_contact=False)
    elif partner == "soft":
        s.add_box("sponge", size=(0.05, 0.05, 0.04), pos=(0.30, 0.0, 0.025), material="pu_foam_sponge", rigid=False,
                  mass=0.05)
    elif partner == "sand":
        s.add_open_box("cup", inner=(0.08, 0.08, 0.05), wall=0.01, pos=(0.30, 0.0, 0.0))
        s.add_box("sand", size=(0.04, 0.04, 0.03), pos=(0.30, 0.0, 0.025), material="sand_silica_dry")
    return s


def _press(partner: str) -> tuple[float, float, object]:
    """(tactile pad normal force, gel normal force) in N after the law's press, and the stepper."""
    sim = alloy.build(_scene(partner), device="cpu", plugins=[Tactile(**PAD), GelSensor(**PAD)])
    sim.run(PRESS_S, record=[], targets={"probe/press_z": (0.0, PRESS_M)})
    assert sim.check()["ok"]
    r = sim.plugins.read()
    out = (float(np.asarray(r["tactile"]["pad_force"])[0, 0, 0]), float(np.asarray(r["gel_tactile"]["force"])[0, 0, 0]),
           sim.engine.stepper)
    sim.close()
    return out


@pytest.fixture(scope="module")
def rigid_only() -> tuple[float, float]:
    tac, gel, stepper = _press("none")
    assert contact_solver(stepper) is stepper.solver
    assert abs(tac) > 1.0 and abs(gel) > 1.0                    # the pad presses the block
    return tac, gel


def _same_force(partner: str, rigid_only: tuple[float, float]) -> None:
    tac, gel, stepper = _press(partner)
    assert not hasattr(stepper, "solver") or not isinstance(stepper.solver, type(contact_solver(stepper)))
    assert tac == pytest.approx(rigid_only[0], rel=BOUND)         # 0 N before the export fix
    assert gel == pytest.approx(rigid_only[1], rel=BOUND)


def test_beside_liquid_reads_the_rigid_force(rigid_only):
    _same_force("liquid", rigid_only)


@pytest.mark.compile_heavy
@pytest.mark.timeout(600)                                       # the soft body: 132 s on a CPU
@pytest.mark.parametrize("partner", ["cloth", "soft", "sand"])
def test_beside_cloth_soft_and_sand_reads_the_rigid_force(rigid_only, partner):
    _same_force(partner, rigid_only)


def test_contact_solver_lookup():
    class Coupled:                                              # a coupled stepper: solver(name) is a lookup method
        def solver(self, name):
            return None

    assert contact_solver(Coupled()) is None
    fake = object()
    assert contact_solver(SimpleNamespace(solver=fake)) is fake   # a stepper's own solver object (test fakes)
    assert contact_solver(SimpleNamespace(base=Coupled())) is None  # a one-way follower over one


class _NeedsAFix(SensorPlugin):
    """Fails its setup with its own fix."""

    name = "needs_a_fix_test"

    def setup(self, ctx):
        raise PluginError("needs_a_fix_test: nothing to read", fix="give it something to read")


def test_build_error_carries_the_plugins_own_fix():
    with pytest.raises(alloy.BuildError) as e:                  # the generic heat fix before (S1-009)
        alloy.build(_scene("none"), device="cpu", plugins=[_NeedsAFix()])
    assert e.value.fix == "give it something to read"


def test_newtons_contacts_raise_instead_of_reading_zero():
    """A block that collides as an SDF puts MuJoCo on Newton's contacts, whose forces it does not export: the pad read
    0 N with sim.check() ok before; now the build says so."""
    h, w = 0.01, 0.025
    v = np.array([[x, y, z] for x in (-w, w) for y in (-w, w) for z in (-h, h)])
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6],
                  [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_robot("probe", mjcf=PROBE)
    s.add_mesh("block", vertices=v, faces=f, collision="sdf", pos=(0.0, 0.0, 0.01), static=True)
    with pytest.raises(alloy.BuildError, match="Newton's contacts") as e:
        alloy.build(s, device="cpu", plugins=[Tactile(**PAD)])
    assert "convex" in e.value.fix
