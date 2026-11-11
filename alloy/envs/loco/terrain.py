# SPDX-License-Identifier: Apache-2.0
"""Rough ground for legged robots: a grid of terrain tiles in one heightfield, harder row by row.

``Terrain`` holds the heights (metres) of a field of ``levels`` x ``len(kinds)`` square tiles on a regular grid,
with a flat border around them.
Every tile has a flat platform in its middle (where robots start) and falls back to height 0 at its border, so
tiles join without cliffs. Level 0 is flat; the last level has the full size of each kind::

    from alloy.envs.loco.terrain import Terrain, TerrainSpec
    terrain = Terrain(TerrainSpec(levels=5, kinds=("rough", "waves", "blocks")))
    terrain.heights.shape              # (ny, nx) m, heights[j, i] at x = x0 + i * cell, y = y0 + j * cell
    terrain.tile_center(level=3, kind=1)   # (x, y) of that tile's platform
    terrain.height_at(xp, xy)          # ground height under points (..., 2), bilinear between grid nodes

Kinds: "rough" (random height per cell, up to ``roughness`` m), "waves" (smooth bumps, ``wave_height`` m high and
``wave_length`` m long), "blocks" (a checkerboard of square blocks of random height up to ``block_height`` m; the
grid joins neighbouring nodes with a ramp one cell wide, so a block's side is a steep ramp, not a vertical wall).
``terrain_mjcf`` writes a copy of a robot file whose floor plane is replaced by the field.
"""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from alloy.envs.errors import EnvError
from alloy.envs.mjcf import cache_dir

KINDS = ("rough", "waves", "blocks")
FLOOR_ATTRS = ("contype", "conaffinity", "condim", "priority", "friction", "solmix", "solref", "solimp", "margin",
               "gap", "group")


@dataclass(frozen=True)
class TerrainSpec:
    """Size and difficulty of a terrain field. Example: ``TerrainSpec(levels=3, tile=4.0, roughness=0.05)``."""

    levels: int = 5                                   # rows of tiles along x; level 0 is flat
    kinds: Sequence[str] = KINDS                      # columns of tiles along y
    tile: float = 6.0                                 # m, edge of a square tile
    cell: float = 0.1                                 # m between grid nodes
    platform: float = 1.0                             # m, edge of the flat start platform in each tile
    border: float = 4.0                               # m of flat ground around the tiles (robots may walk onto it)
    roughness: float = 0.06                           # m, largest random height of "rough" at the last level
    wave_height: float = 0.15                         # m, crest height of "waves" at the last level
    wave_length: float = 2.0                          # m
    block_height: float = 0.08                        # m, tallest block of "blocks" at the last level
    block_size: float = 0.5                           # m
    seed: int = 0

    def validate(self) -> None:
        """Raise ``EnvError`` on a bad value. Example: ``TerrainSpec(levels=0).validate()`` raises."""
        unknown = sorted(set(self.kinds) - set(KINDS))
        if unknown or not self.kinds:
            raise EnvError(f"unknown terrain kinds {unknown}", f"use kinds from {KINDS}", field="kinds")
        if self.levels < 1 or self.levels % 2 == 0 or len(self.kinds) % 2 == 0:
            raise EnvError(f"levels={self.levels} and {len(self.kinds)} kinds", "use odd counts of levels and kinds "
                           "(the field's middle tile is centred on the origin, where the robot file stands)",
                           field="levels")
        if self.border < 0:
            raise EnvError(f"border is {self.border}", "pass border >= 0 (m of flat ground around the tiles)",
                           field="border")
        if not 0 < self.cell <= self.platform < self.tile:
            raise EnvError("terrain needs 0 < cell <= platform < tile", "e.g. tile=6, platform=1, cell=0.1",
                           field="tile")


class Terrain:
    """The height grid of a ``TerrainSpec`` and where its tiles are. Example: see the module docstring."""

    def __init__(self, spec: TerrainSpec | None = None) -> None:
        self.spec = spec = spec or TerrainSpec()
        spec.validate()
        self.levels, self.kinds = int(spec.levels), len(spec.kinds)
        per_tile = int(round(spec.tile / spec.cell))
        self.cell = spec.tile / per_tile
        self.pad = int(round(spec.border / self.cell))                          # border width in cells
        self.nx, self.ny = self.levels * per_tile + 2 * self.pad + 1, self.kinds * per_tile + 2 * self.pad + 1
        border = self.pad * self.cell
        self.half = (0.5 * self.levels * spec.tile + border, 0.5 * self.kinds * spec.tile + border)   # m, (x, y)
        self.x0, self.y0 = -self.half[0], -self.half[1]
        self.heights = self._build(per_tile)
        self._device: dict[str, Any] = {}

    def _build(self, per_tile: int) -> np.ndarray:
        spec, rng = self.spec, np.random.default_rng(self.spec.seed)
        out = np.zeros((self.ny, self.nx), np.float32)
        u = (np.arange(per_tile + 1) - 0.5 * per_tile) * self.cell              # m from the tile centre
        ux, uy = np.meshgrid(u, u)
        reach = np.maximum(np.abs(ux), np.abs(uy))
        edge = 0.5 * spec.tile
        ramp = 0.5                                                              # m over which a tile fades in and out
        window = np.clip((reach - 0.5 * spec.platform) / ramp, 0.0, 1.0) * np.clip((edge - reach) / ramp, 0.0, 1.0)
        for level in range(self.levels):
            scale = level / max(self.levels - 1, 1)
            for k, kind in enumerate(spec.kinds):
                tile = _tile(kind, spec, scale, ux, uy, rng) * window
                i, j = level * per_tile + self.pad, k * per_tile + self.pad
                out[j:j + per_tile + 1, i:i + per_tile + 1] = np.maximum(out[j:j + per_tile + 1, i:i + per_tile + 1],
                                                                         tile)
        return out

    def tile_center(self, level: Any, kind: Any) -> tuple[Any, Any]:
        """(x, y) of the platform of tile (``level``, ``kind``); numbers or arrays.

        Example: ``terrain.tile_center(0, 1)``."""
        t = self.spec.tile
        return (level - 0.5 * (self.levels - 1)) * t, (kind - 0.5 * (self.kinds - 1)) * t

    def height_at(self, xp: Any, xy: Any) -> Any:
        """Ground height (m) under points ``xy`` (..., 2), bilinear between grid nodes; 0 outside the field.

        Example: ``terrain.height_at(env.xp, env.state["go2/base.pos"][:, 0:2])``."""
        grid = self._device.get(xp.name)
        if grid is None:
            grid = self._device[xp.name] = xp.asarray(self.heights.reshape(-1))
        fx = xp.clip((xy[..., 0] - self.x0) / self.cell, 0.0, self.nx - 1.001)
        fy = xp.clip((xy[..., 1] - self.y0) / self.cell, 0.0, self.ny - 1.001)
        ix, iy = _floor(xp, fx), _floor(xp, fy)
        tx, ty = fx - xp.float(ix), fy - xp.float(iy)
        base = iy * self.nx + ix
        h00, h10, h01, h11 = grid[base], grid[base + 1], grid[base + self.nx], grid[base + self.nx + 1]
        return (h00 * (1 - tx) + h10 * tx) * (1 - ty) + (h01 * (1 - tx) + h11 * tx) * ty

    def inside(self, xy: Any, margin: float = 0.3) -> Any:
        """True where points ``xy`` (..., 2) are on the field, ``margin`` m from its edge.

        Example: ``terrain.inside(env.state["go2/base.pos"][:, 0:2])``."""
        return ((abs(xy[..., 0]) < self.half[0] - margin) & (abs(xy[..., 1]) < self.half[1] - margin))


def _tile(kind: str, spec: TerrainSpec, scale: float, ux: np.ndarray, uy: np.ndarray,
          rng: np.random.Generator) -> np.ndarray:
    """Heights (m, >= 0) of one tile of ``kind`` at difficulty ``scale`` in [0, 1], before the edge window."""
    if kind == "rough":
        return rng.uniform(0.0, spec.roughness * scale, ux.shape)
    if kind == "waves":
        k = 2.0 * np.pi / spec.wave_length
        return 0.5 * spec.wave_height * scale * (1.0 - np.cos(k * ux) * np.cos(k * uy))
    n = max(1, int(round(spec.tile / spec.block_size)))
    blocks = rng.uniform(0.0, spec.block_height * scale, (n + 1, n + 1))
    blocks[::2, ::2] = 0.0
    blocks[1::2, 1::2] = 0.0
    ix = np.clip(((ux + 0.5 * spec.tile) / spec.block_size).astype(int), 0, n)
    iy = np.clip(((uy + 0.5 * spec.tile) / spec.block_size).astype(int), 0, n)
    return blocks[iy, ix]


def _floor(xp: Any, x: Any) -> Any:
    return x.long() if xp.name == "torch" else np.floor(x).astype(np.int64)


def terrain_mjcf(path: str | Path, terrain: Terrain, floor: str = "floor") -> Path:
    """A cached copy of the robot file ``path`` (as written by ``alloy.envs.mjcf.without_bodies``: file paths
    absolute) whose plane geom ``floor`` is replaced by ``terrain`` as a MuJoCo heightfield with the plane's contact
    settings. The field is centred on the origin with its lowest point at z = 0.

    Example: ``scene.add_robot("go2", terrain_mjcf(without_bodies(src, ["obstacle"]), Terrain()))``."""
    src = Path(path)
    text = src.read_text(encoding="utf-8")
    top = float(terrain.heights.max())
    zmax = max(top, 1e-3)
    key = hashlib.sha256(text.encode() + terrain.heights.tobytes() + repr(terrain.half).encode()).hexdigest()[:16]
    out = cache_dir() / f"{src.stem}-terrain-{key}.xml"
    if out.is_file():
        return out
    root = ET.fromstring(text)
    world = root.find("worldbody")
    plane = next((g for g in (world.findall("geom") if world is not None else []) if g.get("name") == floor), None)
    if plane is None or plane.get("type", "sphere") != "plane":
        raise EnvError(f"the robot file has no plane geom named {floor!r} under <worldbody>",
                       "name the floor plane, or pass floor=<its name>", field="floor")
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    # MuJoCo: row 0 is the +y edge, column 0 the -x edge; inline elevation is rescaled to span 0..1, so the data
    # must contain both (a field with no relief gets one node lifted by nothing: zmax is then 1 mm of range)
    elevation = np.flipud(terrain.heights / zmax).astype(np.float64)
    if top <= 0.0:
        elevation = elevation.copy()
        elevation[0, 0] = 1.0
    field = ET.SubElement(asset, "hfield", name="terrain", nrow=str(terrain.ny), ncol=str(terrain.nx),
                          size=f"{terrain.half[0]:.6g} {terrain.half[1]:.6g} {zmax:.6g} 0.1")
    field.set("elevation", " ".join(f"{v:.5g}" for v in elevation.ravel()))
    attrs = {k: plane.get(k) for k in FLOOR_ATTRS if plane.get(k) is not None}
    ground = ET.Element("geom", name="terrain", type="hfield", hfield="terrain", pos="0 0 0", **attrs)
    world.insert(list(world).index(plane), ground)
    world.remove(plane)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".tmp{hashlib.sha256(str(id(root)).encode()).hexdigest()[:8]}")
    ET.ElementTree(root).write(tmp, encoding="unicode")
    tmp.replace(out)
    return out


__all__ = ["KINDS", "Terrain", "TerrainSpec", "terrain_mjcf"]
