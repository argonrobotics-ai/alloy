# SPDX-License-Identifier: Apache-2.0
"""The two device scenes ``camera_scene`` builds for a model: the fast diffuse tracer's and the full-material one's."""
from __future__ import annotations

from .scene import SceneData
from .tracer.gpu_scene import GpuScene


def fast_scene(sd: SceneData, device: str, layouts: tuple, model) -> GpuScene:
    """The diffuse tracer's device scene, with the deformables', liquids' and films' per-world meshes (``layouts``:
    the deform, liquid, engine and film layouts of the source, each None when it has none)."""
    layout, liquid, engine, films = layouts
    if layout is None and liquid is None and films is None:
        return GpuScene(sd, device=device)
    from .deform_gpu import DeformSet
    from .films import Dynamic, FilmSet
    from .liquid_scene import LiquidSet
    ds = DeformSet(layout, len(sd.meshes), device, model) if layout is not None else None
    sd = ds.scene_data(sd) if ds is not None else sd
    ls = LiquidSet(liquid, len(sd.meshes), device, engine) if liquid is not None else None
    sd = ls.scene_data(sd) if ls is not None else sd
    fs = FilmSet(films, len(sd.meshes), device) if films is not None else None
    sd = fs.scene_data(sd) if fs is not None else sd
    pts = Dynamic(ds, fs).points() if ds is not None or fs is not None else None
    gs = GpuScene(sd, device=device, points=pts, liquid=ls.buffers() if ls else None)
    for dyn in (ds, ls, fs):
        if dyn is not None:
            dyn.attach(gs)
    return gs


def full_scene(model, look, lighting, device: str, items: dict | None = None, layouts=(None,) * 4, sim=None):
    """A newton.Model as a full-material scene (look materials and lights, item looks; see model_pack), with the
    deformables, liquids and films of ``layouts`` (``_layouts``) as per-world meshes refreshed every render."""
    from .pbr import PbrScene
    from .scan_pack import pack_with_scans
    layout, liquid, engine, films = layouts
    pk, frame, binding = pack_with_scans(model, look, lighting, items, layout, liquid, sim)
    n_liq = 0 if liquid is None else liquid.worlds * len(liquid.objects)
    ds = ls = None
    if layout is not None:
        from .deform_gpu import DeformSet
        ds = DeformSet(layout, len(frame.meshes) - n_liq - layout.worlds * len(layout.objects), device, model)
    if liquid is not None:
        from .liquid_scene import LiquidSet
        ls = LiquidSet(liquid, len(frame.meshes) - n_liq, device, engine)
    if films is not None:  # after the deformables and liquids: their first meshes are counted without the films
        from .films import Dynamic, add_to_pack
        binding, fs = add_to_pack(pk, frame, binding, films, device)
        ds = Dynamic(ds, fs)
    return PbrScene(pk, frame, binding[0].shape[1], device, binding=binding, deform=ds, liquid=ls)
