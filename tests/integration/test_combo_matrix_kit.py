# SPDX-License-Identifier: Apache-2.0
"""Scenes for the material-combination sweep (user ask 2026-11-07; tests/integration/test_combo_matrix.py and the
sweep script in [internal note] notes/wp13-spikes/combo/).

Every scene is a static 400 x 400 x 20 mm plate on the ground, and the kinds asked for are stacked on it from the
bottom up in the order of ``KINDS``. Each part starts a few mm above the one below and falls onto it, so every pair
touches. The parts are the smallest sensible ones:

- ``film``: a surface film (``SurfaceFlow``, 1 mm of water) on the plate's top face;
- ``heat``: the plate is a heater at 80 °C, and the particle kinds (liquid, paste) start at 20 °C and take heat;
- ``paper``: a 300 mm kraft sheet (``paper_kraft_sheet``, 21 x 21 particles, the thin-film solver; ``paper_a4``
  has no route line and goes to the cloth solver);
- ``cloth``: a 140 mm cotton square, 10 x 10 vertices;
- ``mpm``: 150 ml of peanut butter (``peanut_butter_box_150ml``, MPM paste, 4 mm particles);
- ``liquid``: a 40 x 40 x 20 mm block of water (SPH), a puddle once it lands;
- ``soft``: a 50 x 50 x 30 mm sponge block (polyurethane foam, a VBD soft body);
- ``rigid``: a 40 mm beech cube, 0.1 kg;
- ``robot``: a one-joint pusher (a 30 mm finger on a vertical slide servo) pressed down onto the stack.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import alloy

KINDS = ("film", "heat", "paper", "cloth", "mpm", "liquid", "soft", "rigid", "robot")   # bottom to top
PLATE = (0.40, 0.40, 0.02)            # m, full edges; its top face is at z = 0.02
TOP = PLATE[2]
HEATER_K, START_K = 353.15, 293.15
FILM_DEPTH = 1.0e-3                   # m
PUSH = 0.02                           # m the pusher's target goes below the top of the stack under it
PUSHER = """<mujoco><compiler angle="radian"/><worldbody><body name="slider" pos="0 0 0">
  <joint name="lift" type="slide" axis="0 0 1" range="-0.2 0" damping="20"/>
  <geom name="finger" type="box" size="0.015 0.015 0.015" mass="0.3"/>
  </body></worldbody>
  <actuator><position joint="lift" kp="400" kv="40" forcerange="-30 30"/></actuator></mujoco>"""


@dataclass
class Combo:
    """A built-to-be scene for ``kinds``: the alloy.Scene, the plug-ins to pass to ``alloy.build``, what to do right
    after the build (``setup(sim)``: a film's start depth) and the pusher's travel (m, 0 without a robot)."""

    kinds: tuple[str, ...]
    scene: alloy.Scene
    plugins: list[Any] = field(default_factory=list)
    setup: list[Callable[[Any], None]] = field(default_factory=list)
    push: float = 0.0
    flow: Any = None
    heat_material: str | None = None


def _film(c: Combo, z: float) -> float:
    from alloy.plugins.surface_flow import PlaneChart, SurfaceFlow  # noqa: PLC0415

    c.flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(30, 30)),), material="water")
    c.plugins.append(c.flow)
    c.setup.append(lambda sim: c.flow.set_depth(FILM_DEPTH))
    return z


def _heat(c: Combo, z: float) -> float:
    c.scene.set_heater("plate", temperature=HEATER_K)
    return z


def _temperature(c: Combo) -> float | None:
    return START_K if "heat" in c.kinds else None


def _paper(c: Combo, z: float) -> float:
    c.scene.add_object("paper", "paper_kraft_sheet", pos=(0.0, 0.0, z + 0.002))
    return z + 0.003


def _cloth(c: Combo, z: float) -> float:
    c.scene.add_cloth("cloth", size=(0.14, 0.14), pos=(0.0, 0.0, z + 0.004), resolution=10, mass=0.02)
    return z + 0.006


def _mpm(c: Combo, z: float) -> float:
    hm = c.heat_material if "heat" in c.kinds else None
    c.scene.add_object("paste", "peanut_butter_box_150ml", pos=(0.0, 0.0, z), temperature=_temperature(c),
                       heat_material=hm)
    return z + 0.0538                                           # the card's region tops out 51.8 mm above pos


def _liquid(c: Combo, z: float) -> float:
    c.scene.add_liquid("water", size=(0.04, 0.04, 0.02), pos=(0.0, 0.0, z + 0.013), temperature=_temperature(c))
    return z + 0.025


def _soft(c: Combo, z: float) -> float:
    c.scene.add_box("sponge", size=(0.05, 0.05, 0.03), pos=(0.0, 0.0, z + 0.018), material="pu_foam_sponge")
    return z + 0.035


def _rigid(c: Combo, z: float) -> float:
    c.scene.add_box("block", size=(0.04, 0.04, 0.04), pos=(0.0, 0.0, z + 0.025), mass=0.1)
    return z + 0.05


def _robot(c: Combo, z: float) -> float:
    base = z + 0.035                                            # the finger's centre: 20 mm clear of the stack
    c.scene.add_robot("pusher", mjcf=PUSHER, pos=(0.0, 0.0, base))
    c.push = base - 0.015 - z + PUSH                            # travel to PUSH below the stack's top
    return base + 0.015


ADD: dict[str, Callable[[Combo, float], float]] = {
    "film": _film, "heat": _heat, "paper": _paper, "cloth": _cloth, "mpm": _mpm, "liquid": _liquid, "soft": _soft,
    "rigid": _rigid, "robot": _robot}


def combo(kinds: tuple[str, ...] | list[str], worlds: int = 2, dt: float = 0.01,
          heat_material: str | None = None) -> Combo:
    """The scene for ``kinds`` (any of ``KINDS``), stacked on the plate bottom-up in ``KINDS`` order.
    ``heat_material``: the heat data for the paste when it is heated (peanut butter has none of its own)."""
    unknown = sorted(set(kinds) - set(KINDS))
    if unknown:
        raise ValueError(f"unknown kinds {unknown}; use {KINDS}")
    ordered = tuple(sorted(set(kinds), key=KINDS.index))
    scene = alloy.Scene(worlds=worlds, dt=dt)
    scene.add_ground()
    scene.add_box("plate", size=PLATE, pos=(0.0, 0.0, 0.5 * PLATE[2]), static=True)
    c = Combo(ordered, scene, heat_material=heat_material)
    z = TOP
    for k in ordered:
        z = ADD[k](c, z)
    return c


def build(c: Combo, device: str = "cpu", **kw: Any) -> Any:
    """alloy.build with the combo's plug-ins, then its setup (a film's start depth)."""
    sim = alloy.build(c.scene, device=device, plugins=c.plugins, **kw)
    for f in c.setup:
        f(sim)
    return sim


def targets(c: Combo, seconds: float) -> dict[str, tuple[float, float]] | None:
    """The pusher's ramp: down by ``c.push`` over the first half of the run, then held."""
    return {"lift": (0.0, -c.push)} if c.push else None
