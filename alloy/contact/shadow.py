# SPDX-License-Identifier: Apache-2.0
"""Shadow shapes: per-pair contact modes without forking Newton (contact §4.4, IL RJ45 pattern).

A precision pair (a, b) gets two density-0 coincident copies (a', b') on the same bodies. The copies carry the
precision settings (gap, margin, hydroelastic flag and kh, the feature crop) and collide only with each other; the
originals keep their ordinary settings for every other partner and never meet each other. Filtering uses Newton
collision filter pairs, which SolverMuJoCo also turns into MuJoCo exclusions, so neither the bulk pipeline nor
MuJoCo's native narrow phase sees (a, b), (a, b') or (a', anything but b').

Call ``add_shadow_shapes`` after every shape of the world template exists: shapes added later are not filtered
against the shadows (``check_shadow_filters`` on the finalized model catches that).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from alloy._vendor import newton
from alloy.contact.types import PrecisionPairSpec
from alloy.core.errors import AssemblyError, PrecisionSpecError
from alloy.core.log import get_logger

log = get_logger("contact.shadow")

_MESH_TYPES = (int(newton.GeoType.MESH), int(newton.GeoType.CONVEX_MESH))


def _crop_mesh(mesh: newton.Mesh, crop: tuple) -> newton.Mesh:
    """Triangles whose centroid lies in the crop box (shape frame); an open shell unless the feature is closed."""
    v = np.asarray(mesh.vertices, dtype=np.float64)
    t = np.asarray(mesh.indices, dtype=np.int64).reshape(-1, 3)
    lo, hi = np.asarray(crop[0], float), np.asarray(crop[1], float)
    c = v[t].mean(axis=1)
    keep = np.all((c >= lo) & (c <= hi), axis=1)
    if not keep.any():
        raise PrecisionSpecError(f"crop {crop} keeps no triangle of the mesh")
    used, inverse = np.unique(t[keep], return_inverse=True)
    return newton.Mesh(v[used].astype(np.float32), inverse.reshape(-1).astype(np.int32), compute_inertia=False,
                       is_solid=False)


def _shadow_cfg(builder: newton.ModelBuilder, shape: int, spec: PrecisionPairSpec) -> newton.ModelBuilder.ShapeConfig:
    cfg = newton.ModelBuilder.ShapeConfig(
        density=0.0,
        mu=float(builder.shape_material_mu[shape]),
        restitution=float(builder.shape_material_restitution[shape]),
        mu_torsional=float(builder.shape_material_mu_torsional[shape]),
        mu_rolling=float(builder.shape_material_mu_rolling[shape]),
        ke=float(builder.shape_material_ke[shape]),
        kd=float(builder.shape_material_kd[shape]),
        gap=spec.gap,
        margin=spec.margin,
        has_particle_collision=False,
        is_visible=False,
    )
    if spec.mode == "hydroelastic":
        cfg.is_hydroelastic = True
        cfg.kh = spec.kh
    if spec.mode == "hydroelastic" and int(builder.shape_type[shape]) not in _MESH_TYPES:
        # hydroelastic primitives need Newton's texture SDF (built at finalize, CUDA only); sdf-mode primitives keep
        # the exact analytic narrow phase; meshes get their SDF from alloy.contact.sdf_cache
        cfg.sdf_target_voxel_size = spec.voxel_size
        cfg.sdf_narrow_band_range = tuple(spec.narrow_band)
        cfg.sdf_texture_format = spec.texture_format
    return cfg


def _copy_shape(builder: newton.ModelBuilder, shape: int, spec: PrecisionPairSpec, crop: tuple | None,
                label: str) -> int:
    src = builder.shape_source[shape]
    if crop is not None:
        if not isinstance(src, newton.Mesh):
            raise PrecisionSpecError(f"crop given for shape {shape}, which is not a mesh")
        src = _crop_mesh(src, crop)
    elif isinstance(src, newton.Mesh):
        # own copy: an SDF baked on the shadow (sdf_cache) must not change the original shape's narrow phase
        src = newton.Mesh(np.asarray(src.vertices, np.float32), np.asarray(src.indices, np.int32).reshape(-1),
                          compute_inertia=False, is_solid=bool(getattr(src, "is_solid", True)))
    return builder.add_shape(
        body=int(builder.shape_body[shape]),
        type=int(builder.shape_type[shape]),
        xform=builder.shape_transform[shape],
        cfg=_shadow_cfg(builder, shape, spec),
        scale=builder.shape_scale[shape],
        src=src,
        label=label,
    )


def _filter_all_but(builder: newton.ModelBuilder, shadow: int, keep: Iterable[int]) -> None:
    keep = set(keep)
    for other in range(builder.shape_count):
        if other != shadow and other not in keep:
            builder.add_shape_collision_filter_pair(shadow, other)


def set_flat_solimp(builder: newton.ModelBuilder, shape: int, imp: float) -> None:
    """Shape attribute ``mujoco:geom_solimp = (imp, imp, 0.001, 1, 0.5)`` (D22 flat solimp). Needed on the
    ``native="newton"`` path, where Newton's converter takes imp from the geom (kernels.py:663-685); the MuJoCo append
    writes the pair's imp per row and does not need it. Registers SolverMuJoCo's attributes if missing."""
    key = "mujoco:geom_solimp"
    if not builder.has_custom_attribute(key):
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    attr = builder.custom_attributes[key]
    if attr.values is None:
        attr.values = {}
    attr.values[int(shape)] = attr.dtype(imp, imp, 0.001, 1.0, 0.5)


def add_shadow_shapes(builder: newton.ModelBuilder, spec: PrecisionPairSpec) -> tuple[int, int]:
    """Add the density-0 precision copies of ``spec.shape_a``/``shape_b`` and the filter pairs; returns their ids.

    The caller builds the precision ``PrecisionPairSpec`` for the layer with the returned ids. SDFs for the copies are
    attached by ``alloy.contact.sdf_cache`` (CUDA), not here, so this runs on CPU.
    """
    a, b = int(spec.shape_a), int(spec.shape_b)
    n = builder.shape_count
    if not (0 <= a < n and 0 <= b < n) or a == b:
        raise PrecisionSpecError(f"precision pair shapes ({a}, {b}) invalid for a builder with {n} shapes")
    label_a = f"{builder.shape_label[a] or a}:precision"
    label_b = f"{builder.shape_label[b] or b}:precision"
    sa = _copy_shape(builder, a, spec, spec.crop_a, label_a)
    sb = _copy_shape(builder, b, spec, spec.crop_b, label_b)
    builder.add_shape_collision_filter_pair(a, b)
    for shadow in (sa, sb):
        set_flat_solimp(builder, shadow, spec.mj_imp)
    _filter_all_but(builder, sa, keep=(sb,))
    _filter_all_but(builder, sb, keep=(sa,))
    log.info("shadow shapes %d (%s) and %d (%s) for pair (%d, %d), mode=%s", sa, label_a, sb, label_b, a, b, spec.mode)
    return sa, sb


def check_shadow_filters(model: newton.Model, shadow_pairs: Iterable[tuple[int, int]]) -> None:
    """Raise ``AssemblyError`` if a shadow can meet any shape other than its partner (shapes added afterwards)."""
    filt = {tuple(sorted(map(int, p))) for p in model.shape_collision_filter_pairs}
    body, world = model.shape_body.numpy(), model.shape_world.numpy()
    for sa, sb in shadow_pairs:
        for shadow, partner in ((sa, sb), (sb, sa)):
            for other in range(model.shape_count):
                if other in (shadow, partner) or (body[other] == body[shadow] and body[other] >= 0):
                    continue
                if world[other] >= 0 and world[shadow] >= 0 and world[other] != world[shadow]:
                    continue
                if (min(shadow, other), max(shadow, other)) not in filt:
                    raise AssemblyError(f"shadow shape {shadow} is not filtered against shape {other}: add shadow "
                                        f"shapes after every other shape of the world template")
