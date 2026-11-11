# SPDX-License-Identifier: Apache-2.0
"""AlloyContactLayer: one contact buffer for every solver, with a private precision pipeline (SPEC §3.5, §2.3).

- ``native="mujoco"``: MJWarp collides the bulk; the layer runs only the precision pipeline and ``contacts()`` is the
  precision stream, which ``MixedContactBinding`` appends inside MJWarp's ``contactfilter`` (D10).
- ``native="newton"``: a bulk ``CollisionPipeline`` fills ``contacts()``; precision rows are appended after it.

The precision pipeline uses explicit pairs (each pair once per world), ``publish_capacity=False`` and, when
``deterministic``, Newton's sorted contacts plus ``contact_matching="latest"`` (needed for the D22 weights). Its
``max_triangle_pairs`` is sized from the meshes; when the deterministic reducer cap (2^20 - 1 triangle pairs) would
bind, worlds are split over several pipelines (spike finding 3). Per-row K for sdf-mode pairs is
``K_pair · weight`` (D22); hydroelastic rows keep Newton's ``area·p/|depth|``. The layer keeps its own copy of the
precision K (``row_stiffness``) so a later write to ``rigid_contact_stiffness`` cannot change what is appended.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.contact.buffer import append_rows
from alloy.contact.mesh_sign import ParityRecheck
from alloy.contact.precision import UNSUPPORTED_NATIVE, pair_stiffness
from alloy.contact.samples import surface_area_estimate
from alloy.contact.sidecar import PATCH_BUCKETS, StreamTables, fill_stream, sidecar_from_pipeline
from alloy.contact.types import AlloyContactSidecar, ContactClass, PrecisionPairSpec
from alloy.core.errors import AssemblyError, PrecisionSpecError
from alloy.core.log import get_logger

log = get_logger("contact.layer")

DEFAULT_DESIGN_FORCE_N = 50.0           # LoadEnvelope.design_force default (§4.2) when pair_stiffness is unset
TRIANGLE_PAIR_CAP = (1 << 20) - 1       # newton contact_reduction_global.py:376 (deterministic packing)
ISO_VOXEL_CAP = 419_430                 # newton sdf_hydroelastic.py:114 _MAX_DETERMINISTIC_ISO_VOXELS
# Newton sizes the hydroelastic refinement buffers from the SDFs' active tiles; the peg-in-hole uses ~1.1 % of that
# at every level (measured 10k/40k/157k of 0.92M/3.7M/14.8M), so they are scaled to 5 % (4.5x headroom, drops
# counted in hydro_overflow). The broad buffer holds every tile of the pair, so its multiplier undoes the fraction.
HYDRO_BUFFER_FRACTION = 0.05


def hydro_iso_estimate(model: newton.Model, a: int, b: int, voxel: float) -> float:
    """Iso voxels one hydroelastic pair needs per world: one voxel layer over the smaller part's surface, ``A/h²``.

    Newton keeps every voxel whose prune band ``gap_a + gap_b + √3·h`` spans the local clearance
    (sdf_hydroelastic.py ``count_iso_voxels_block``), so in a tight fit most of the engaged surface is kept. Measured
    for the 12 mm peg in its 0.1 mm radial-clearance bore (tilted 0.2°, 50 N): 345k at h = 50 µm with gap 5 µm
    (estimate 544k), 650k with gap 0.5 mm (see ``hydro_band_spans_clearance``)."""
    geo, scale = model.shape_type.numpy(), model.shape_scale.numpy()
    area = min(surface_area_estimate(int(geo[s]), scale[s], model.shape_source[s]) for s in (a, b))
    return area / (voxel * voxel)


def hydro_band_spans_clearance(spec: PrecisionPairSpec, gap_sum: float) -> bool:
    """True when Newton's iso prune band ``gap_sum + √3·h`` reaches across the radial clearance, which keeps the whole
    clearance shell as iso voxels (about 2x the need; over the deterministic cap for the 12 mm test peg)."""
    return spec.clearance is not None and gap_sum + math.sqrt(3.0) * spec.voxel_size >= 0.5 * spec.clearance


@wp.kernel
def _hydro_overflow(counts: wp.array[wp.int32], cap: int, out: wp.array[wp.int32]):
    if counts[0] > cap:
        wp.atomic_add(out, 0, counts[0] - cap)


def world_shape_pairs(model: newton.Model, pairs: Sequence[PrecisionPairSpec]) -> np.ndarray:
    """[P, W, 2] global shape ids, (min, max) per row: template id ``t`` in world ``w`` is
    ``shape_world_start[w] + t`` (Newton ``replicate`` layout)."""
    starts = model.shape_world_start.numpy()
    out = np.zeros((len(pairs), model.world_count, 2), np.int32)
    for p, spec in enumerate(pairs):
        for w in range(model.world_count):
            a, b = starts[w] + int(spec.shape_a), starts[w] + int(spec.shape_b)
            if max(a, b) >= starts[w + 1]:
                raise AssemblyError(f"pair {p} template shapes ({spec.shape_a}, {spec.shape_b}) outside world {w}")
            out[p, w] = (min(a, b), max(a, b))
    return out


def _tri_count(model: newton.Model, shape: int) -> int:
    src = model.shape_source[shape]
    idx = getattr(src, "indices", None)
    return 64 if idx is None else int(np.asarray(idx).size // 3)


def _chunks(model: newton.Model, wpairs: np.ndarray, deterministic: bool, iso_per_world: float = 0.0
            ) -> list[np.ndarray]:
    """World chunks so each pipeline's triangle-pair estimate (and, for hydroelastic, iso-voxel estimate) stays
    under Newton's deterministic caps."""
    per_world = sum(2 * (_tri_count(model, int(a)) + _tri_count(model, int(b))) for a, b in wpairs[:, 0])
    per_world = max(per_world, 1)
    worlds = model.world_count
    size = worlds if not deterministic else max(1, min(worlds, TRIANGLE_PAIR_CAP // per_world))
    if deterministic and iso_per_world > 0.0:
        if iso_per_world > ISO_VOXEL_CAP:
            log.warning("hydroelastic iso-voxel estimate %.0f per world > Newton's deterministic cap %d: one world per "
                        "pipeline; drops, if any, show in hydro_overflow / CONTACT_OVERFLOW", iso_per_world,
                        ISO_VOXEL_CAP)
        size = max(1, min(size, int(ISO_VOXEL_CAP // iso_per_world)))
    return [np.arange(s, min(s + size, worlds)) for s in range(0, worlds, size)]


@wp.kernel
def _precision_stiffness(start: wp.array[wp.int32], count: wp.array[wp.int32], pair: wp.array[wp.int32],
                         weight: wp.array[wp.float32], pair_k: wp.array[wp.float32], pair_hydro: wp.array[wp.int32],
                         stiffness: wp.array[wp.float32], damping: wp.array[wp.float32],
                         row_k: wp.array[wp.float32]):
    i = wp.tid()
    k = start[0] + i
    if i >= count[0] or k >= row_k.shape[0]:
        return
    p = pair[k]
    if p < 0:
        return
    kk = stiffness[k]
    if pair_hydro[p] == 0:
        kk = pair_k[p] * weight[k]
        stiffness[k] = kk
        damping[k] = 0.0            # MuJoCo: dampratio 1 (critical) when kd = 0 (kernels.py:663-685)
    row_k[k] = kk


def _check_native(pairs: Sequence[PrecisionPairSpec], native: str, measure_unsupported: bool) -> None:
    """Refuse a (mode, native) combination in ``UNSUPPORTED_NATIVE``; the CT tests pass ``measure_unsupported`` to
    keep measuring it (the evidence for the refusal), never a router."""
    for p, spec in enumerate(pairs):
        reason = UNSUPPORTED_NATIVE.get((spec.mode, native))
        if reason is None:
            continue
        if not measure_unsupported:
            raise PrecisionSpecError(f"pair {p}: {reason}")
        log.warning("pair %d: measuring an unsupported combination (%s, native=%s)", p, spec.mode, native)


class AlloyContactLayer:
    """See module docstring. Satisfies ``ContactLayerLike``."""

    def __init__(self, model: newton.Model, pairs: Sequence[PrecisionPairSpec], *,
                 native: Literal["mujoco", "newton"] = "mujoco", precision_contacts_per_world: int = 256,
                 deterministic: bool = True, sdf_cache_dir: Path | None = None,
                 hydro_iso_per_world: float | None = None, measure_unsupported: bool = False) -> None:
        if native not in ("mujoco", "newton"):
            raise ValueError(f"native must be 'mujoco' or 'newton', got {native!r}")
        _check_native(pairs, native, measure_unsupported)
        self.model, self.pairs, self.native = model, tuple(pairs), native
        self.deterministic, self.sdf_cache_dir = deterministic, sdf_cache_dir
        self.hydro_iso_per_world = hydro_iso_per_world      # override the iso-voxel estimate (measured value)
        self.device = model.device
        self.world_pairs = world_shape_pairs(model, self.pairs) if self.pairs else np.zeros((0, 0, 2), np.int32)
        self._binding = None
        self.overflow = wp.zeros(1, dtype=wp.int32, device=self.device)        # merge rows dropped
        self.hydro_overflow = wp.zeros(1, dtype=wp.int32, device=self.device)  # hydroelastic buffer entries dropped
        self._precision: list[tuple[newton.CollisionPipeline, newton.Contacts, wp.array]] = []
        if self.pairs:
            self._check_filtered()
            if native == "newton":
                self._check_newton_solimp()
            self._check_hydro_gap()
            self._build_precision(precision_contacts_per_world)
        self._build_output(precision_contacts_per_world)
        self._build_tables()
        log.info("contact layer: native=%s, %d pairs x %d worlds, %d precision pipeline(s), capacity %d",
                 native, len(self.pairs), model.world_count, len(self._precision), self._contacts.rigid_contact_max)

    # ---------------------------------------------------------------------------------------------- build ---
    def _check_filtered(self) -> None:
        """Each precision pair must be filtered from native/bulk collision, else it would be collided twice."""
        filt = {tuple(sorted((int(a), int(b)))) for a, b in self.model.shape_collision_filter_pairs}
        missing = [(int(a), int(b)) for a, b in self.world_pairs.reshape(-1, 2) if (int(a), int(b)) not in filt]
        if missing:
            raise AssemblyError(f"precision pairs not in the model's collision filter pairs (first: {missing[:3]}); "
                                f"use add_shadow_shapes or ModelBuilder.add_shape_collision_filter_pair")

    def _check_newton_solimp(self) -> None:
        """Under ``native="newton"`` the flat imp must sit on the shapes (``set_flat_solimp``), else MJWarp's
        ``τ ≥ 2·dt`` clamp caps K_real at ``m/((1-imp)·4dt²)`` with the geom's imp (D22)."""
        sol = getattr(getattr(self.model, "mujoco", None), "geom_solimp", None)
        dmax = sol.numpy()[:, 1] if sol is not None else None
        for p, spec in enumerate(self.pairs):
            for shape in self.world_pairs[p, 0]:
                have = 0.95 if dmax is None else float(dmax[shape])
                if abs(have - spec.mj_imp) > 1e-6:
                    log.warning("pair %d shape %d: mujoco:geom_solimp dmax %.4g != mj_imp %.4g on the native='newton' "
                                "path; K_real is capped lower (set_flat_solimp / add_shadow_shapes)", p, shape, have,
                                spec.mj_imp)

    def _check_hydro_gap(self) -> None:
        """Warn when a hydroelastic pair's gaps make Newton keep the whole clearance shell as iso voxels."""
        gaps = self.model.shape_gap.numpy()
        for p, spec in enumerate(self.pairs):
            a, b = self.world_pairs[p, 0]
            gap_sum = float(gaps[a] + gaps[b])
            if spec.mode == "hydroelastic" and hydro_band_spans_clearance(spec, gap_sum):
                log.warning("pair %d: hydroelastic gap sum %.3g m + sqrt(3)*voxel >= radial clearance %.3g m; Newton "
                            "keeps the whole clearance shell as iso voxels (~2x). Use gap <= %.3g m per shape", p,
                            gap_sum, 0.5 * spec.clearance,
                            max(0.0, 0.25 * spec.clearance - 0.5 * math.sqrt(3.0) * spec.voxel_size))

    def _pipeline_kwargs(self, idx: list[int]) -> dict:
        """Pipeline options for one mode group. Hydroelastic pairs get no velocity-expanded gap: Newton raises
        "Speculative contact generation does not yet support hydroelastic SDF contacts" (collide.py:1883-1890), so
        their L1 rows come from the authored ``gap`` only (``ladder_rungs`` drops L0 for them)."""
        pairs = [self.pairs[p] for p in idx]
        kw: dict = dict(broad_phase="explicit", deterministic=self.deterministic, publish_capacity=False,
                        verify_buffers=False)
        hydro = any(s.mode == "hydroelastic" for s in pairs)
        spec_gap = [s.speculative_gap_max for s in pairs if s.ccd != "off"]
        if spec_gap and not hydro:
            kw["speculative_contact_gap_max"] = max(spec_gap)
        if hydro:
            from alloy._vendor.newton.geometry import HydroelasticSDF  # noqa: PLC0415

            kw["sdf_hydroelastic_config"] = HydroelasticSDF.Config(
                reduce_contacts=any(s.reduce != "off" for s in pairs), anchor_contact=True,
                moment_matching=True, buffer_fraction=HYDRO_BUFFER_FRACTION,
                buffer_mult_broad=round(1.0 / HYDRO_BUFFER_FRACTION), buffer_mult_iso=1)
        if self.deterministic:
            kw["contact_matching"] = "latest"
            kw["contact_matching_pos_threshold"] = min(s.resolved_match_threshold() for s in pairs)
        return kw

    def _build_precision(self, per_world: int) -> None:
        """One pipeline per (mode group, world chunk); see ``_pipeline_kwargs`` and ``_chunks``."""
        groups = [[p for p, s in enumerate(self.pairs) if (s.mode == "hydroelastic") == h] for h in (False, True)]
        for idx in (g for g in groups if g):
            kw = self._pipeline_kwargs(idx)
            sub = self.world_pairs[idx]
            iso = 0.0
            if "sdf_hydroelastic_config" in kw:
                iso = self.hydro_iso_per_world or sum(hydro_iso_estimate(self.model, int(sub[k, 0, 0]),
                                                                         int(sub[k, 0, 1]), self.pairs[p].voxel_size)
                                                      for k, p in enumerate(idx))
            for worlds in _chunks(self.model, sub, self.deterministic, iso):
                flat = sub[:, worlds].reshape(-1, 2)
                arr = wp.array(flat, dtype=wp.vec2i, device=self.device)
                tri = min(TRIANGLE_PAIR_CAP, sum(2 * (_tri_count(self.model, int(a)) + _tri_count(self.model, int(b)))
                                                 for a, b in flat))
                pipe = newton.CollisionPipeline(self.model, shape_pairs_filtered=arr, rigid_contact_max=per_world *
                                                len(worlds) * len(idx), max_triangle_pairs=max(tri, 1024), **kw)
                contacts, _ = sidecar_from_pipeline(AlloyContactSidecar, pipe)
                self._precision.append((pipe, contacts, wp.zeros(1, dtype=wp.int32, device=self.device)))

    def _build_output(self, per_world: int) -> None:
        """The solver-facing buffer: bulk (newton), the single precision buffer, or a merge buffer."""
        prec_cap = sum(c.rigid_contact_max for _, c, _ in self._precision)
        self._parity = None
        if self.native == "newton":
            self._bulk = newton.CollisionPipeline(self.model, rigid_contact_max=self._bulk_capacity() + prec_cap,
                                                  deterministic=self.deterministic, publish_capacity=False)
            self._parity = ParityRecheck(self.model)          # Warp parity sign on watertight meshes (mesh_sign)
            self._contacts, self._sidecar = sidecar_from_pipeline(AlloyContactSidecar, self._bulk)
        elif len(self._precision) == 1:
            self._bulk = None
            self._contacts = self._precision[0][1]
            self._sidecar = AlloyContactSidecar.allocate(self._contacts.rigid_contact_max, device=self.device)
        else:
            self._bulk = None
            self._contacts = newton.Contacts(max(prec_cap, 1), 0, device=self.device,
                                             per_contact_shape_properties=True, requested_attributes={"force"})
            self._sidecar = AlloyContactSidecar.allocate(self._contacts.rigid_contact_max, device=self.device)
        self.row_stiffness = wp.zeros(self._contacts.rigid_contact_max, dtype=wp.float32, device=self.device)
        self._zero_start = wp.zeros(1, dtype=wp.int32, device=self.device)

    def _bulk_capacity(self) -> int:
        probe = newton.CollisionPipeline(self.model, publish_capacity=False)
        return int(probe.rigid_contact_max)

    def _build_tables(self) -> None:
        S, P, W = self.model.shape_count, len(self.pairs), self.model.world_count
        shape_pair = np.full(S, -1, np.int32)
        for p in range(P):
            for a, b in self.world_pairs[p]:
                shape_pair[a] = shape_pair[b] = p
        pair_class = [int(ContactClass.PRECISION_HYDRO if s.mode == "hydroelastic" else ContactClass.PRECISION_SDF)
                      for s in self.pairs]
        dev = self.device
        self._tables = StreamTables(
            shape_pair=wp.array(shape_pair, dtype=wp.int32, device=dev),
            pair_class=wp.array(np.asarray(pair_class or [0], np.int8), dtype=wp.int8, device=dev),
            pair_count=wp.zeros(max(P, 1) * W * PATCH_BUCKETS, dtype=wp.int32, device=dev), world_count=W)
        self._weight_prev = [wp.zeros(c.rigid_contact_max, dtype=wp.float32, device=dev)
                             for _, c, _ in self._precision]
        self._bulk_weight_prev = wp.zeros(self._contacts.rigid_contact_max, dtype=wp.float32, device=dev)
        k = [pair_stiffness(s, DEFAULT_DESIGN_FORCE_N) if s.mode == "sdf" else 0.0 for s in self.pairs]
        for s in self.pairs:
            if s.mode == "sdf" and s.pair_stiffness is None:
                log.warning("pair (%d, %d): pair_stiffness unset; K from the default design force %.0f N",
                            s.shape_a, s.shape_b, DEFAULT_DESIGN_FORCE_N)
        self.pair_k = wp.array(np.asarray(k or [0.0], np.float32), dtype=wp.float32, device=dev)
        self.pair_hydro = wp.array(np.asarray([int(s.mode == "hydroelastic") for s in self.pairs] or [0], np.int32),
                                   dtype=wp.int32, device=dev)
        self.pair_imp = wp.array(np.asarray([s.mj_imp for s in self.pairs] or [0.99], np.float32), dtype=wp.float32,
                                 device=dev)

    # ------------------------------------------------------------------------------------------------ API ---
    def contacts(self) -> newton.Contacts:
        return self._contacts

    def sidecar(self) -> AlloyContactSidecar:
        return self._sidecar

    @property
    def precision_streams(self) -> list[tuple[newton.Contacts, wp.array]]:
        """(buffer, start offset in ``contacts()``) per precision pipeline, for the MuJoCo append."""
        return [(c, start) for _, c, start in self._precision]

    def collide(self, state: newton.State, contacts: newton.Contacts, *, dt: float) -> None:
        """Run the bulk pipeline (``native="newton"``) and every precision pipeline, merge, fill the sidecar and the
        precision K, and arm the MuJoCo append guard. ``contacts`` must be ``self.contacts()``."""
        if contacts is not self._contacts:
            raise AssemblyError("AlloyContactLayer.collide must be given the layer's own contacts() buffer")
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"dt must be positive and finite, got {dt}")
        if self._bulk is not None:
            self._bulk.collide(state, contacts, dt=dt)
            self._parity.apply(self.model, state, contacts)   # soft rows come from the bulk pass only
            fill_stream(self._sidecar, self._tables, self.model, state, contacts, contacts, self._zero_start,
                        self._bulk_weight_prev)
        elif len(self._precision) > 1:
            contacts.clear()
        for k, (pipe, buf, start) in enumerate(self._precision):
            self._collide_precision(state, contacts, pipe, buf, start, k, dt)
        if self._binding is not None:
            self._binding.arm(state)

    def _collide_precision(self, state, contacts, pipe, buf, start, k: int, dt: float) -> None:
        pipe.collide(state, buf, dt=dt)
        self._check_hydro(pipe)
        if buf is not contacts:
            append_rows(contacts, buf, start, self.overflow)
        fill_stream(self._sidecar, self._tables, self.model, state, contacts, buf, start, self._weight_prev[k])
        wp.launch(_precision_stiffness, dim=buf.rigid_contact_max, device=self.device, inputs=[
            start, buf.rigid_contact_count, self._sidecar.pair, self._sidecar.weight, self.pair_k, self.pair_hydro,
            contacts.rigid_contact_stiffness, contacts.rigid_contact_damping, self.row_stiffness])

    def _check_hydro(self, pipe) -> None:
        """Count hydroelastic buffer overflow (broad blocks and the 4 refinement levels; public counters of
        newton HydroelasticSDF). Newton only prints it; here it reaches CONTACT_OVERFLOW and the report."""
        hs = getattr(pipe, "hydroelastic_sdf", None)
        if hs is None:
            return
        caps = (hs.max_num_blocks_broad, *hs.iso_max_dims)
        for counts, cap in zip(hs.iso_buffer_counts, caps):
            wp.launch(_hydro_overflow, dim=1, inputs=[counts, int(cap), self.hydro_overflow], device=self.device)

    def bind_mujoco(self, solver) -> "MixedContactBinding":  # noqa: F821
        from alloy.contact.mujoco_append import MixedContactBinding  # noqa: PLC0415

        if self.native != "mujoco":
            raise PrecisionSpecError("bind_mujoco needs native='mujoco'; with 'newton' pass contacts() to the solver")
        self._binding = MixedContactBinding(solver, self)
        return self._binding

    def monitor(self, *, sample_spacing: float = 5e-4) -> "GapMonitor":  # noqa: F821
        from alloy.contact.monitor import GapMonitor  # noqa: PLC0415

        return GapMonitor(self, sample_spacing=sample_spacing)

    def collider_field(self) -> "ColliderField":  # noqa: F821
        from alloy.contact.field import ColliderField  # noqa: PLC0415

        return ColliderField(self)

    def overflow_report(self) -> dict[str, int]:
        """Host readback (not in the hot loop): dropped rows per cause."""
        out = {"merge": int(self.overflow.numpy()[0]), "hydroelastic": int(self.hydro_overflow.numpy()[0])}
        for k, (_, buf, _) in enumerate(self._precision):
            cnt = buf.contact_counters.numpy()
            out[f"precision{k}_reduction"] = int(cnt[2])
            out[f"precision{k}_capacity"] = max(0, int(cnt[0]) - buf.rigid_contact_max)
        return out
