# SPDX-License-Identifier: Apache-2.0
"""Synthetic RoboCasa-format assets for the importer's tests (no downloads).

``write_object_asset`` writes a robosuite-style ``model.xml`` for a closed, UV-mapped box (one visual part named
``<Asset>_Clear``), a texture and a task region; ``write_fixture_asset`` writes a cabinet-like fixture: a fixed box
body with a hinged door; ``write_pack_zip`` zips assets in a pack's layout.

Example::

    >>> from alloy.testing.imp_robocasa import write_object_asset
    >>> folder = write_object_asset(Path(tmp) / "glass_cup" / "GlassCup900")   # doctest: +SKIP
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np

from alloy.importers.robocasa.thumbs import write_png

BOX_HALF = (0.03, 0.03, 0.05)       # m: a 6 x 6 x 10 cm box


def box_obj(half: tuple[float, float, float], name: str = "box", uv_faces: int = 6) -> str:
    """OBJ text of a closed, outward-wound box; the first ``uv_faces`` faces get a UV square each (UV seams split
    corners), the rest are written without ``vt`` (0: a mesh with no UVs at all)."""
    hx, hy, hz = half
    corners = [(x, y, z) for x in (-hx, hx) for y in (-hy, hy) for z in (-hz, hz)]
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    lines = [f"o {name}"] + [f"v {x} {y} {z}" for x, y, z in corners]
    lines += ["vt 0 0", "vt 1 0", "vt 1 1", "vt 0 1"] if uv_faces else []
    for i, (a, b, c, d) in enumerate(quads):
        uv = i < uv_faces
        lines.append(" ".join(["f"] + [f"{q + 1}/{k + 1}" if uv else str(q + 1) for k, q in enumerate((a, b, c, d))]))
    return "\n".join(lines) + "\n"


def _texture(path: Path) -> None:
    img = np.zeros((8, 8, 3), np.uint8)
    img[::2, ::2] = (200, 60, 60)
    write_png(path, img)


OBJECT_XML = """<mujoco model="model">
  <compiler angle="radian"/>
  <asset>
    <mesh file="visual/{stem}.obj" name="{stem}_vis"/>
    <texture file="visual/T_BC001.png" name="tex" type="2d"/>
    <material name="mat" texture="tex" rgba="1 1 1 0.7" shininess="0.5" specular="0.3"/>
    <mesh file="collision/collision_mesh_collision_0.obj" name="col0"/>
  </asset>
  <default>
    <default class="visual"><geom conaffinity="0" contype="0" group="1" type="mesh"/></default>
    <default class="collision"><geom group="0" rgba="0.5 0 0 0.5" density="1000.0"/></default>
    <default class="region"><geom group="1" conaffinity="0" contype="0" rgba="0 1 0 0" mass="0"/></default>
  </default>
  <worldbody>
    <body>
      <body name="object">
        <geom mesh="{stem}_vis" type="mesh" class="visual" material="mat"/>
        <geom mesh="col0" type="mesh" class="collision"/>
        <geom class="region" name="reg_bbox" type="box" size="0.03 0.03 0.05"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def write_object_asset(folder: Path, half: tuple[float, float, float] = BOX_HALF, uv_faces: int = 6) -> Path:
    """A one-part object asset (closed box, textured material) in RoboCasa's layout; returns ``folder``."""
    stem = f"{folder.name}_Clear"
    (folder / "visual").mkdir(parents=True, exist_ok=True)
    (folder / "collision").mkdir(parents=True, exist_ok=True)
    (folder / "visual" / f"{stem}.obj").write_text(box_obj(half, stem, uv_faces))
    (folder / "collision" / "collision_mesh_collision_0.obj").write_text(box_obj(half, "col"))
    _texture(folder / "visual" / "T_BC001.png")
    (folder / "model.xml").write_text(OBJECT_XML.format(stem=stem))
    return folder


FIXTURE_XML = """<?xml version='1.0' encoding='UTF-8'?>
<mujoco model="model">
  <compiler angle="radian"/>
  <asset>
    <mesh file="visuals/body.obj" name="body_vis"/>
    <mesh file="visuals/door.obj" name="door_vis"/>
  </asset>
  <default>
    <default class="visual"><geom conaffinity="0" contype="0" group="1" type="mesh"/></default>
    <default class="collision"><geom group="0" rgba="0.5 0 0 0.5"/></default>
  </default>
  <worldbody>
    <body>
      <body name="object">
        <geom mesh="body_vis" class="visual"/>
        <geom class="collision" type="box" pos="0 0.02 0" size="0.25 0.18 0.15"/>
        <body name="door">
          <joint name="door_joint" type="hinge" axis="0 0 1" pos="-0.25 -0.2 0" range="-1.5707963 0" limited="true"
                 damping="1" frictionloss="1" armature=".01" stiffness="0.05"/>
          <geom mesh="door_vis" class="visual" pos="0 -0.2 0"/>
          <geom class="collision" type="box" pos="0 -0.2 0" size="0.25 0.01 0.15"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def write_fixture_asset(folder: Path) -> Path:
    """A box appliance with a hinged front door, in RoboCasa's fixture layout; returns ``folder``."""
    (folder / "visuals").mkdir(parents=True, exist_ok=True)
    (folder / "visuals" / "body.obj").write_text(box_obj((0.25, 0.18, 0.15), "body"))
    (folder / "visuals" / "door.obj").write_text(box_obj((0.25, 0.01, 0.15), "door"))
    (folder / "model.xml").write_text(FIXTURE_XML)
    return folder


PART_XML = """<mujoco model="model">
  <compiler angle="radian"/>
  <asset><mesh file="visuals/{stem}.obj" name="vis"/></asset>
  <default>
    <default class="visual"><geom conaffinity="0" contype="0" group="1" type="mesh"/></default>
    <default class="collision"><geom group="0"/></default>
  </default>
  <worldbody><body><body name="object">
    <geom mesh="vis" class="visual"/>
    {colliders}
  </body></body></worldbody>
</mujoco>
"""


def write_panel_asset(folder: Path) -> Path:
    """A Lightwheel-style door panel (0.5 x 0.028 x 0.7 m) with a box collider; returns ``folder``."""
    (folder / "visuals").mkdir(parents=True, exist_ok=True)
    (folder / "visuals" / f"{folder.name}.obj").write_text(box_obj((0.25, 0.014, 0.35), folder.name))
    col = '<geom class="collision" type="box" size="0.25 0.014 0.35"/>'
    (folder / "model.xml").write_text(PART_XML.format(stem=folder.name, colliders=col))
    return folder


def write_handle_asset(folder: Path) -> Path:
    """A bar handle (two posts and a bar along z, sticking out along +y); returns ``folder``."""
    (folder / "visuals").mkdir(parents=True, exist_ok=True)
    (folder / "visuals" / f"{folder.name}.obj").write_text(box_obj((0.006, 0.012, 0.07), folder.name))
    col = ('<geom class="collision" type="cylinder" size="0.006 0.07" pos="0 0.02 0"/>'
           '<geom class="collision" type="cylinder" size="0.005 0.01" pos="0 0.01 0.06" quat="0.7071068 0.7071068 0 0"/>'
           '<geom class="collision" type="cylinder" size="0.005 0.01" pos="0 0.01 -0.06" quat="0.7071068 0.7071068 0 0"/>')
    (folder / "model.xml").write_text(PART_XML.format(stem=folder.name, colliders=col))
    return folder


def write_pack_zip(path: Path, top: str, assets: dict[str, Path]) -> Path:
    """Zip asset folders as ``<top>/<asset id>/...`` plus a README, like a RoboCasa pack."""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("README.md", "test pack\n")
        for asset, folder in assets.items():
            for f in sorted(folder.rglob("*")):
                if f.is_file():
                    zf.write(f, f"{top}/{asset}/{f.relative_to(folder)}")
    return path


__all__ = ["BOX_HALF", "box_obj", "write_fixture_asset", "write_handle_asset", "write_object_asset",
           "write_pack_zip", "write_panel_asset"]
