# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# make_filtered_cell_lookup below is warp-lang 1.18.0's NanogridBase.make_filtered_cell_lookup
# (warp/_src/fem/geometry/nanogrid.py), copied verbatim except for the lines marked "Alloy" (Apache-2.0; Alloy's
# changes, 2026-11-11: the environment lookup picks its cell in integers, and its cache keys carry a suffix).
"""Warp fem's packed-grid cell lookup in integers (known gap docs/known_gaps/mpm-moving-container-many-worlds.md;
physics ruling 2026-11-11 (b)).

Alloy steps the worlds of an MPM entry in one shared sparse grid: Warp's ``fem.Nanogrid`` with environments, each
world's cells at an integer offset (about 22 voxels per world along x for a small scene). Warp's
``cell_lookup_env`` picks a particle's cell as ``floor(float32(local index + offset))``. Near a cell face that
rounds into the neighbouring cell once the offset is large, and the error grows with the offset: 1e-4 voxel at 1,400
voxels, 0.016 at 131,072. At 64 GPU worlds grains then sank into bodies they touched (a lifted box's floor, a tool,
a press, a robot's finger, a free box resting on sand), with the count rising from world 0 to the last world.
Measured on one L4 (plate pressing dough, 64 worlds, grains more than 1 mm inside the plate): as built, 1 in world 0
to about 32 in the last 8 worlds; every offset shifted 131,072 voxels, about 633 in every world. With the lookup in
integers, ``floor(local index) + offset``, it read 0 in every world in both runs.

``install`` replaces ``NanogridBase.make_filtered_cell_lookup`` (so every grid made afterwards looks up in integers)
once per process, from ``sparse_guard.install`` at each MPM entry's build. A grid with one environment takes Warp's
other branch (``cell_lookup_default``, no offset), unchanged. A scene without MPM never installs it. The guard: Warp
must be ``WARP_VERSION`` and its function's source must hash to ``SOURCE_SHA256``; otherwise nothing is replaced and
the build logs a warning (tests/integration/test_nanogrid_lookup.py fails then, so a Warp bump re-checks this).
"""

# No ``from __future__ import annotations``: Warp evaluates the kernel functions' annotations (grid_geo.CellArg, a
# closure), as in its own nanogrid.py.
import hashlib
import inspect
import logging
from typing import Any

import warp as wp
from warp._src.fem import cache
from warp._src.fem.cache import cached_vec_type
from warp._src.fem.geometry.nanogrid import NanogridBase
from warp._src.fem.types import NULL_ELEMENT_INDEX, make_free_sample

log = logging.getLogger("alloy.integration")

__all__ = ["SOURCE_SHA256", "WARP_VERSION", "install", "installed", "make_filtered_cell_lookup"]

WARP_VERSION = "1.18.0"
SOURCE_SHA256 = "71823f8993234b595ed26a5d1e47b95791e6526f37c744e4057baceaeac09b83"   # Warp's function, as copied
_WARP = NanogridBase.make_filtered_cell_lookup


def warp_source_sha256() -> str:
    """SHA-256 of Warp's own ``make_filtered_cell_lookup`` source (``inspect.getsource``)."""
    return hashlib.sha256(inspect.getsource(_WARP).encode()).hexdigest()


def installed() -> bool:
    """Whether grids made from now on use the integer lookup. Example: ``installed()`` after an MPM build."""
    return NanogridBase.make_filtered_cell_lookup is make_filtered_cell_lookup


def install() -> bool:
    """Replace Warp's lookup factory once per process (module doc); True when this call replaced it."""
    if installed():
        return False
    if wp.__version__ != WARP_VERSION or warp_source_sha256() != SOURCE_SHA256:
        log.warning("packed-grid cell lookup not replaced: Warp %s's NanogridBase.make_filtered_cell_lookup is not "
                    "the audited %s source, so MPM at many GPU worlds keeps the float32 lookup "
                    "(docs/known_gaps/mpm-moving-container-many-worlds.md)", wp.__version__, WARP_VERSION)
        return False
    NanogridBase.make_filtered_cell_lookup = make_filtered_cell_lookup
    log.debug("packed-grid cell lookup: floor(local index) + world offset, in integers")
    return True


def make_filtered_cell_lookup(grid_geo: Any, filter_func: wp.Function = None) -> Any:
    """Warp 1.18.0's ``NanogridBase.make_filtered_cell_lookup``, the cell picked in integers (module doc)."""
    suffix = f"{grid_geo.name}{filter_func.key if filter_func is not None else ''}alloy_int"   # Alloy: own keys
    scalar = grid_geo._scalar_type
    vec3_type = cached_vec_type(3, scalar)
    CoordsType = grid_geo.coords_type

    if grid_geo.environment_count() <= 1:

        @cache.dynamic_func(suffix=suffix)
        def cell_lookup_default(
            args: grid_geo.CellArg, pos: vec3_type, max_dist: float, filter_data: Any, filter_target: Any
        ):
            grid = args.cell_grid

            uvw = wp.volume_world_to_index(grid, pos) + vec3_type(scalar(0.5))
            i, j, k = int(wp.floor(uvw[0])), int(wp.floor(uvw[1])), int(wp.floor(uvw[2]))
            cell_index = grid_geo._lookup_cell_index(args, i, j, k)

            if cell_index != -1:
                coords = grid_geo._cell_coordinates_local(args, cell_index, uvw)
                if wp.static(filter_func is None):
                    return make_free_sample(cell_index, coords)
                else:
                    if filter_func(filter_data, cell_index) == filter_target:
                        return make_free_sample(cell_index, coords)

            cell_size = vec3_type(
                wp.length(wp.volume_index_to_world_dir(grid, vec3_type(scalar(1.0), scalar(0.0), scalar(0.0)))),
                wp.length(wp.volume_index_to_world_dir(grid, vec3_type(scalar(0.0), scalar(1.0), scalar(0.0)))),
                wp.length(wp.volume_index_to_world_dir(grid, vec3_type(scalar(0.0), scalar(0.0), scalar(1.0)))),
            )

            offset = scalar(0.5)
            min_cell_size = wp.min(cell_size)
            max_offset = wp.ceil(max_dist / min_cell_size)
            scales = wp.cw_div(vec3_type(min_cell_size), vec3_type(cell_size))

            closest_cell = NULL_ELEMENT_INDEX
            closest_coords = CoordsType()

            while closest_cell == NULL_ELEMENT_INDEX:
                uvw_min = wp.vec3i(uvw - offset * scales)
                uvw_max = wp.vec3i(uvw + offset * scales) + wp.vec3i(1)

                closest_dist = min_cell_size * min_cell_size * scalar(offset * offset)

                for i in range(uvw_min[0], uvw_max[0]):
                    for j in range(uvw_min[1], uvw_max[1]):
                        for k in range(uvw_min[2], uvw_max[2]):
                            cell_index = grid_geo._lookup_cell_index(args, i, j, k)
                            if cell_index == -1:
                                continue

                            if wp.static(filter_func is not None):
                                if filter_func(filter_data, cell_index) != filter_target:
                                    continue
                            dist, coords = grid_geo._cell_closest_point_local(args, cell_index, uvw)

                            if dist <= closest_dist:
                                closest_dist = dist
                                closest_coords = coords
                                closest_cell = cell_index

                if offset >= max_offset:
                    break
                offset = wp.min(scalar(3.0) * offset, max_offset)

            return make_free_sample(closest_cell, closest_coords)

    @cache.dynamic_func(suffix=(suffix, "env"))
    def cell_lookup_env(
        args: grid_geo.CellArg,
        pos: vec3_type,
        max_dist: float,
        filter_data: Any,
        filter_target: Any,
        env_index: int,
    ):
        grid = args.cell_grid

        uvw = wp.volume_world_to_index(grid, pos) + vec3_type(scalar(0.5))
        env_offset = args.env_offsets[env_index]
        # Alloy: the cell in integers, floor(local) + offset. Warp floors float32(local + offset), whose error
        # grows with the world's offset and puts particles near a cell face in the neighbouring cell.
        i = int(wp.floor(uvw[0])) + env_offset[0]
        j = int(wp.floor(uvw[1])) + env_offset[1]
        k = int(wp.floor(uvw[2])) + env_offset[2]
        cell_index = grid_geo._lookup_cell_index(args, i, j, k)

        if cell_index != -1:
            if args.cell_env[cell_index] == env_index:
                coords = grid_geo._cell_coordinates_local(args, cell_index, uvw)
                if wp.static(filter_func is None):
                    return make_free_sample(cell_index, coords)
                else:
                    if filter_func(filter_data, cell_index) == filter_target:
                        return make_free_sample(cell_index, coords)

        cell_size = vec3_type(
            wp.length(wp.volume_index_to_world_dir(grid, vec3_type(scalar(1.0), scalar(0.0), scalar(0.0)))),
            wp.length(wp.volume_index_to_world_dir(grid, vec3_type(scalar(0.0), scalar(1.0), scalar(0.0)))),
            wp.length(wp.volume_index_to_world_dir(grid, vec3_type(scalar(0.0), scalar(0.0), scalar(1.0)))),
        )

        offset = scalar(0.5)
        min_cell_size = wp.min(cell_size)
        max_offset = wp.ceil(max_dist / min_cell_size)
        scales = wp.cw_div(vec3_type(min_cell_size), vec3_type(cell_size))

        closest_cell = NULL_ELEMENT_INDEX
        closest_coords = CoordsType()

        while closest_cell == NULL_ELEMENT_INDEX:
            uvw_min = wp.vec3i(uvw - offset * scales) + env_offset
            uvw_max = wp.vec3i(uvw + offset * scales) + wp.vec3i(1) + env_offset

            closest_dist = min_cell_size * min_cell_size * scalar(offset * offset)

            for i in range(uvw_min[0], uvw_max[0]):
                for j in range(uvw_min[1], uvw_max[1]):
                    for k in range(uvw_min[2], uvw_max[2]):
                        cell_index = grid_geo._lookup_cell_index(args, i, j, k)
                        if cell_index == -1:
                            continue

                        if args.cell_env[cell_index] != env_index:
                            continue

                        if wp.static(filter_func is not None):
                            if filter_func(filter_data, cell_index) != filter_target:
                                continue
                        dist, coords = grid_geo._cell_closest_point_local(args, cell_index, uvw)

                        if dist <= closest_dist:
                            closest_dist = dist
                            closest_coords = coords
                            closest_cell = cell_index

            if offset >= max_offset:
                break
            offset = wp.min(scalar(3.0) * offset, max_offset)

        return make_free_sample(closest_cell, closest_coords)

    if grid_geo.environment_count() <= 1:

        @cache.dynamic_func(suffix=suffix, allow_overloads=True)
        def cell_lookup(
            args: grid_geo.CellArg, pos: vec3_type, max_dist: float, filter_data: Any, filter_target: Any
        ):
            return cell_lookup_default(args, pos, max_dist, filter_data, filter_target)

    @cache.dynamic_func(suffix=suffix, allow_overloads=True)
    def cell_lookup(
        args: grid_geo.CellArg,
        pos: vec3_type,
        max_dist: float,
        filter_data: Any,
        filter_target: Any,
        env_index: int,
    ):
        return cell_lookup_env(args, pos, max_dist, filter_data, filter_target, env_index)

    return cell_lookup
