# SPDX-License-Identifier: Apache-2.0
"""Shared importer pipeline, used by every library importer.

One source mesh in, per-object files out (``process``): a watertight ``physics.obj``, CoACD hulls
``physics.hulls.obj``, a textured ``visual.glb`` and a ``thumb.png``; then card, manifest and sourcing-queue writers.
Mesh steps need the optional ``import`` extra (trimesh, fast-simplification, coacd, Pillow); the card, manifest and
queue writers need only Alloy.

Example::

    from alloy.importers.common.process import process
    result = process(Path("textured.obj"), Path("out/ycb_025_mug"), "ycb_025_mug")
"""
