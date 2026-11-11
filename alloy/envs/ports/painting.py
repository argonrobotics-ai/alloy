# SPDX-License-Identifier: Apache-2.0
"""The painting ports (unofficial ports of Gym-Anything painting-tasks-v1) as environments: eight tasks, each on its
port's own run.

| env | Gym-Anything task | what the policy moves |
|---|---|---|
| ``PaintRollStrip-v0`` | roll-strip | a fixed FR3 with the 9 in roller: 7 joint targets, the cover's spin |
| ``PaintLoadRoller-v0`` | load-roller | the same run, judged at the end of the first load (the tray's checks) |
| ``PaintBrushStroke-v0`` | brush-stroke | a fixed FR3 with the round brush: 7 joint targets |
| ``PaintSprayStencil-v0`` | spray-stencil | a fixed FR3 with the aerosol can: 7 joint targets, the valve |
| ``PaintSwatchCard-v0`` | swatch-card | the round brush (kinematic): its pose and velocity |
| ``PaintClearCoat-v0`` | clear-coat | the aerosol can (kinematic): its pose and velocity, the valve; batched worlds |
| ``PaintCutInTrim-v0`` | cut-in-trim | the sash brushes (kinematic), one per band: pose and velocity each |
| ``PaintBehindRadiator-v0`` | behind-radiator | the long-reach brush (kinematic): pose and velocity |

Success is the port's verdict (``succeeded``: every check the port judges; ``docs/guide/envs.md`` lists them).
``size="small"`` runs the ports' small layouts (their CPU test crops) where they have one; the default is the port's
full layout.
The film (wet paint) is not drawn by the camera yet: the images show the wall, the tools and the robot.
"""

from __future__ import annotations

from typing import Any

from alloy.envs.ports.env import PortSpec
from alloy.envs.ports.movers import KinematicBody, Mover, PluginInput, ServoJoints

FR3 = tuple(f"fr3_joint{i}" for i in range(1, 8))
SPIN = 400.0                 # rad/s: the roller's axle servo range (the plan rolls at 0.25 m/s / 0.024 m)


def _fr3(*extra: Any) -> Mover:
    return Mover(ServoJoints(FR3), *extra)


def _succeeded(result: Any, w: int) -> bool:  # noqa: ARG001 - one-world ports
    return bool(result["succeeded"])


# --- the FR3 runs ----------------------------------------------------------------------------------------------------


def _wall(size: str) -> Any:
    from alloy.tasks.paint_scene import Band, WallLayout  # noqa: PLC0415

    if size == "small":   # the FR3 test's crop: 0.4 x 0.352 m at 8 mm, a 0.2 x 0.2 m strip
        return WallLayout(width=0.4, height=0.352, cell=0.008, strip=Band(-0.1, 0.1, 0.076, 0.276))
    return WallLayout()


def run_roll_strip(worlds: int, device: str, seed: int, size: str = "full", passes: int = 3,  # noqa: ARG001
                   settle: float | None = None) -> dict:
    """roll-strip and load-roller with the FR3 holding the roller (``paint_roll_strip.run_roller``)."""
    from alloy.tasks import paint_roll_strip as rs  # noqa: PLC0415
    from alloy.tasks.paint_fr3 import Fr3Roller  # noqa: PLC0415

    layout = _wall(size)
    kw = {} if settle is None else {"settle": settle}
    return rs.run_roller(layout, device=device, passes=passes, driver=Fr3Roller(layout, passes), **kw)


def run_brush_stroke(worlds: int, device: str, seed: int, loaded: bool = False,  # noqa: ARG001
                     settle: float | None = None) -> dict:
    """brush-stroke with the FR3 holding the round brush (``paint_brush_stroke.run_brush``)."""
    from alloy.tasks.paint_brush_stroke import run_brush  # noqa: PLC0415
    from alloy.tasks.paint_fr3_brush import Fr3Brush  # noqa: PLC0415

    kw = {} if settle is None else {"settle": settle}
    return run_brush(device=device, loaded=loaded, driver=Fr3Brush(loaded), **kw)


def run_spray_stencil(worlds: int, device: str, seed: int, settle: float | None = None) -> dict:  # noqa: ARG001
    """spray-stencil with the FR3 holding the aerosol can (``paint_spray_supply.run_supply``)."""
    from alloy.tasks.paint_fr3_can import Fr3Can  # noqa: PLC0415
    from alloy.tasks.paint_spray_supply import Can, SprayLayout, run_supply  # noqa: PLC0415

    layout, can = SprayLayout(), Can()
    kw = {} if settle is None else {"settle": settle}
    return run_supply(layout, can, device=device, driver=Fr3Can(layout, can), **kw)


# --- the kinematic tools ---------------------------------------------------------------------------------------------


def run_swatch_card(worlds: int, device: str, seed: int, settle: float | None = None) -> dict:  # noqa: ARG001
    from alloy.tasks import swatch_card  # noqa: PLC0415

    return swatch_card.run(device=device, **({} if settle is None else {"settle": settle}))


def run_clear_coat(worlds: int, device: str, seed: int, size: str = "full",  # noqa: ARG001
                   settle: float | None = None) -> dict:
    from alloy.tasks import paint_clear_coat as cc  # noqa: PLC0415

    kw = {} if settle is None else {"settle": settle}
    return cc.run(cc.SMALL if size == "small" else None, device=device, worlds=worlds, **kw)


def run_cut_in_trim(worlds: int, device: str, seed: int, settle: float | None = None) -> dict:  # noqa: ARG001
    from alloy.tasks import trim_cut_in  # noqa: PLC0415

    return trim_cut_in.run(device=device, **({} if settle is None else {"settle": settle}))


def run_behind_radiator(worlds: int, device: str, seed: int, settle: float | None = None) -> dict:  # noqa: ARG001
    from alloy.tasks import radiator_task  # noqa: PLC0415

    return radiator_task.run(device=device, **({} if settle is None else {"settle": settle}))


def _trim_mover() -> Mover:
    from alloy.tasks.trim_cut_in import body_name  # noqa: PLC0415
    from alloy.tasks.trim_layout import JOBS  # noqa: PLC0415

    return Mover(*(KinematicBody(body_name(kind)) for kind in JOBS))


def _clear_coat_world(result: Any, w: int) -> bool:
    return bool(result["worlds"][w]["succeeded"])



SPECS = [
    PortSpec("PaintRollStrip-v0", "painting/roll-strip", run_roll_strip,
             lambda: _fr3(ServoJoints(["axle"], low=[-SPIN], high=[SPIN], proprio=False)), _succeeded,
             summary="roll a 0.6 x 1.0 m strip between tape lines with an FR3 and a 9 in roller"),
    PortSpec("PaintLoadRoller-v0", "painting/load-roller", run_roll_strip,
             lambda: _fr3(ServoJoints(["axle"], low=[-SPIN], high=[SPIN], proprio=False)),
             lambda r, w: bool(r["load_roller"]["succeeded"]),
             summary="load the roller in the tray's well (judged at the end of the first hold)"),
    PortSpec("PaintBrushStroke-v0", "painting/brush-stroke", run_brush_stroke, _fr3, _succeeded,
             summary="one stroke of a round brush on watercolour paper, FR3"),
    PortSpec("PaintSprayStencil-v0", "painting/spray-stencil", run_spray_stencil,
             lambda: _fr3(PluginInput("valve")), _succeeded,
             summary="spray a crate's stencil with an aerosol can (FR3), two coats"),
    PortSpec("PaintSwatchCard-v0", "painting/swatch-card", run_swatch_card,
             lambda: Mover(KinematicBody("brush")), _succeeded,
             summary="brush a swatch square on a card, round brush"),
    PortSpec("PaintClearCoat-v0", "painting/clear-coat", run_clear_coat,
             lambda: Mover(KinematicBody("can"), PluginInput("valve")), _clear_coat_world, batched=True,
             summary="clear-coat a printed box lid with an aerosol can"),
    PortSpec("PaintCutInTrim-v0", "painting/cut-in-trim", run_cut_in_trim, _trim_mover, _succeeded,
             summary="cut in along a door casing and an outlet cover with a sash brush"),
    PortSpec("PaintBehindRadiator-v0", "painting/behind-radiator", run_behind_radiator,
             lambda: Mover(KinematicBody("brush")), _succeeded,
             score=lambda r, w: float(r.get("score", float(r["succeeded"]))),
             summary="paint the wall behind a radiator with a long-reach brush"),
]

__all__ = ["SPECS", "run_behind_radiator", "run_brush_stroke", "run_clear_coat", "run_cut_in_trim",
           "run_roll_strip", "run_spray_stencil", "run_swatch_card"]
