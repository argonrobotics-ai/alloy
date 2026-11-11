# SPDX-License-Identifier: Apache-2.0
"""Blender-free look materials: preset + overrides -> pack-v2 node trees for the path tracer's generator.

Example::

    from alloy.camera.looks import material_tree
    mat = material_tree("pad", "rubber_black")          # same tree Alloy's Blender build makes
    # alloy.camera.pbr.codegen.gen_material("pad", mat, mat["textures"]) -> (Warp source, meta)
"""
from .material import material_tree, rgba
from .spec import resolve_spec
from .textures import LocalAssets, image_entry, local_texture_files

__all__ = ["LocalAssets", "image_entry", "local_texture_files", "material_tree", "resolve_spec", "rgba"]
