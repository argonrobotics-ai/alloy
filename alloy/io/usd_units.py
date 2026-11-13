# SPDX-License-Identifier: Apache-2.0
"""Bring a USD stage authored in other length units to metres before Newton's importer reads it.

Newton's USD importer reads lengths as metres and only warns when a stage declares ``metersPerUnit`` other than 1,
so a centimetre stage came in 100 times too large. :func:`metre_source` returns the path unchanged for a stage in
metres (or one that declares no unit), and for any other stage an in-memory stage that sublayers the file, declares
metres, scales each root prim by ``metersPerUnit`` and converts the UsdPhysics values whose units contain a length:

================================================  =====================  ==============
value                                             UsdPhysics unit        factor (u = metersPerUnit)
================================================  =====================  ==============
``physics:density``                               mass / length^3        1 / u^3
``physics:diagonalInertia``                       mass * length^2        u^2
angular drive ``stiffness``, ``damping``          mass * length^2 / ...  u^2
angular drive ``maxForce`` (a torque)             mass * length^2 / s^2  u^2
linear drive ``maxForce`` (a force)               mass * length / s^2    u
``physics:gravityMagnitude``                      length / s^2           u
prismatic ``lowerLimit``, ``upperLimit``          length                 u
================================================  =====================  ==============

Linear drive stiffness and damping (mass / s^2, mass / s) and masses do not change.

Example::

    from alloy._vendor import newton
    from alloy.io.usd_units import metre_source
    b = newton.ModelBuilder()
    b.add_usd(metre_source("arm_centimetres.usda"))       # link at z = 45 (cm) lands at 0.45 m
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

log = logging.getLogger("alloy.io")
SCALE_OP = "alloyMetres"
#: attribute name -> power of metersPerUnit its value is multiplied by
FACTORS = {"physics:density": -3, "physics:diagonalInertia": 2, "physics:gravityMagnitude": 1,
           "drive:angular:physics:stiffness": 2, "drive:angular:physics:damping": 2,
           "drive:angular:physics:maxForce": 2, "drive:linear:physics:maxForce": 1}
PRISMATIC = ("physics:lowerLimit", "physics:upperLimit")


def stage_units(path: str | Path) -> float | None:
    """``metersPerUnit`` the stage declares, or None when it declares none (USD's fallback of 0.01 is not a
    statement by the author: such files are read as metres, as before)."""
    from pxr import Usd, UsdGeom  # noqa: PLC0415

    stage = Usd.Stage.Open(str(path))
    if stage is None or not stage.HasAuthoredMetadata("metersPerUnit"):
        return None
    return float(UsdGeom.GetStageMetersPerUnit(stage))


def has_plane(path: str | Path) -> bool:
    """True if the USD stage has a ``Plane`` prim that collides (a floor). False without the USD packages or for a
    file OpenUSD cannot open. Example: an ``alloy.io.export_scene`` stage of a scene with a ground -> True."""
    try:
        from pxr import Usd, UsdGeom, UsdPhysics  # noqa: PLC0415

        stage = Usd.Stage.Open(str(path))
        return stage is not None and any(p.IsA(UsdGeom.Plane) and p.HasAPI(UsdPhysics.CollisionAPI)
                                         for p in stage.Traverse())
    except Exception:  # noqa: BLE001 - no packages, or a broken file: the loader reports it
        return False


def _scale_roots(stage, u: float) -> None:
    """Scale the top-most transformable prims by ``u`` (the scale is the outermost op of each)."""
    from pxr import Gf, UsdGeom  # noqa: PLC0415

    pending = list(stage.GetPseudoRoot().GetChildren())
    while pending:
        prim = pending.pop()
        xf = UsdGeom.Xformable(prim)
        if not prim.IsA(UsdGeom.Xformable):
            pending.extend(prim.GetChildren())
            continue
        ops = xf.GetOrderedXformOps()
        op = xf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble, SCALE_OP)
        op.Set(Gf.Vec3d(u, u, u))
        xf.SetXformOpOrder([op, *ops], xf.GetResetXformStack())


def _convert_values(stage, u: float) -> int:
    """Rewrite the authored values of :data:`FACTORS` (and prismatic limits) in metres; returns how many."""
    n = 0
    for prim in stage.Traverse():
        names = dict(FACTORS)
        if prim.GetTypeName() == "PhysicsPrismaticJoint":
            names.update(dict.fromkeys(PRISMATIC, 1))
        for name, power in names.items():
            attr = prim.GetAttribute(name)
            if not attr or not attr.HasAuthoredValue():
                continue
            value = attr.Get()
            if value is None or (isinstance(value, float) and not math.isfinite(value)):
                continue
            attr.Set(value * (u ** power))
            n += 1
    return n


def metre_source(path: str | Path):
    """What to hand Newton's ``add_usd`` for ``path``: the path itself for a stage in metres or without a declared
    unit, else an in-memory ``Usd.Stage`` of the same content in metres (module docstring)."""
    from pxr import Sdf, Usd, UsdGeom  # noqa: PLC0415

    u = stage_units(path)
    if u is None or math.isclose(u, 1.0, rel_tol=1e-9):
        return str(path)
    src = Sdf.Layer.FindOrOpen(str(Path(path).resolve()))
    layer = Sdf.Layer.CreateAnonymous(".usda")
    layer.subLayerPaths.append(src.identifier)
    for key in src.pseudoRoot.ListInfoKeys():       # stage metadata lives on the root layer only: up axis, default
        if key not in ("subLayers", "subLayerOffsets", "metersPerUnit"):    # prim, mass unit, time codes
            layer.pseudoRoot.SetInfo(key, src.pseudoRoot.GetInfo(key))
    stage = Usd.Stage.Open(layer)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    _scale_roots(stage, u)
    n = _convert_values(stage, u)
    log.info("%s declares metersPerUnit = %g: read in metres (lengths x %g, %d physics values converted)",
             Path(path).name, u, u, n)
    return stage
