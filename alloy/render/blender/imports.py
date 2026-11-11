# SPDX-License-Identifier: Apache-2.0
"""Model imports through Blender's own importers (glTF 2.0 .glb/.gltf, Wavefront .obj)."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import bpy
from mathutils import Matrix

log = logging.getLogger(__name__)
IMPORTERS = {".glb": "gltf", ".gltf": "gltf", ".obj": "obj"}


def run_importer(path: Path) -> list[bpy.types.Object]:
    """Import into the active collection; returns the new objects."""
    before = set(bpy.data.objects)
    if IMPORTERS[path.suffix.lower()] == "gltf":
        bpy.ops.import_scene.gltf(filepath=str(path))
    else:
        bpy.ops.wm.obj_import(filepath=str(path), forward_axis="Y", up_axis="Z")
    return [o for o in bpy.data.objects if o not in before]


def into_collection(path: Path, name: str) -> bpy.types.Collection | None:
    """Import a model into a new collection that is not linked to the scene (used only through instances)."""
    scene, layer = bpy.context.scene, bpy.context.view_layer
    coll = bpy.data.collections.new(name)
    scene.collection.children.link(coll)
    previous = layer.active_layer_collection
    layer.active_layer_collection = layer.layer_collection.children[coll.name]
    try:
        run_importer(path)
    except Exception as exc:  # noqa: BLE001 - a broken prop is skipped, never fatal
        log.warning("model %s failed to import (%s); skipping it", path, exc)
        bpy.data.collections.remove(coll)
        return None
    finally:
        layer.active_layer_collection = previous
    if coll.name in scene.collection.children:
        scene.collection.children.unlink(coll)
    return coll


def joined_mesh(path: Path, name: str) -> bpy.types.Mesh:
    """All mesh primitives of a .glb/.gltf as one mesh in world coordinates, keeping materials, UVs and normals."""
    new = run_importer(path)
    meshes = [o for o in new if o.type == "MESH"]
    if not meshes:
        raise ValueError(f"{path} has no mesh")
    for o in meshes:
        if o.data.users > 1:
            o.data = o.data.copy()
    if len(meshes) > 1:
        with bpy.context.temp_override(active_object=meshes[0], object=meshes[0], selected_objects=meshes,
                                       selected_editable_objects=meshes):
            bpy.ops.object.join()
    obj = meshes[0]
    world = obj.matrix_world.copy()
    if world != Matrix.Identity(4):
        obj.data.transform(world)
    me = obj.data
    me.name = name
    for o in new:
        if o != obj:
            bpy.data.objects.remove(o)
    bpy.data.objects.remove(obj)
    return me


def base_name(material_name: str) -> str:
    """'plastic.001' -> 'plastic' (Blender's suffix for a name clash)."""
    return re.sub(r"\.\d{3}$", "", material_name)
