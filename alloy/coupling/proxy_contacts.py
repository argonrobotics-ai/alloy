# SPDX-License-Identifier: Apache-2.0
"""Proxy collision pipelines sized by what can collide (SCR-DC5-19).

Newton's CollisionPipeline builds a soft-contact candidate for every particle x every shape of a world and sizes its
soft-contact buffer to that count (newton@ac7f7830:newton/_src/sim/collide.py:1324-1339, 2189-2243). ACTIVE and
COLLIDE_PARTICLES are applied per thread, not when sizing, so a coupled VBD pad (~900 particles) in a 204-shape robot
cell needs 185 k candidates per world: 2.27 GB of contacts at 1,024 worlds.

``ProxyContactPipeline`` is a ``SolverCoupledProxy.Proxy(collision_pipeline=...)`` factory that builds the proxy's
pipeline with
- candidates only for shapes with COLLIDE_PARTICLES at construction (scenes clear it before the sim; Newton's coupler
  clears it in a view on shapes neither owned nor proxied). Newton's kernel skips those pairs, so the remaining
  candidates are Newton's list with them removed, in Newton's order, and contacts are unchanged. Inactive particles
  stay candidates: ACTIVE changes at runtime (emitters activate spares);
- ``soft_contact_max`` = the candidate count, or ``soft_contacts_per_world`` x worlds when that is smaller;
- ``full_surface`` (SCR-SC4-4, on for SolverVBD destinations): Newton's edge and triangle soft contacts besides the
  per-vertex ones, so a rigid feature narrower than the cloth's vertex gap (C7's 2-6 mm pad ridges vs 8.5 mm cloth
  edges) cannot pass between vertices and thread the cloth onto the pad. Only ``full_surface_shapes`` get them
  (make_stepper: shapes on bodies that move, which the destination's view also sees as reduced hulls,
  ``alloy.coupling.full_surface``); the others keep per-vertex contact. ``soft_contact_max`` is
  then ``soft_contacts_per_world`` x worlds when set, else ``FULL_SURFACE_RECORDS`` per soft feature; the band is
  ``FULL_SURFACE_GAP`` unless ``soft_contact_gap`` is given;
- the same capacity as SolverVBD's construction-time contact-state hint (``entry_contact_sizing``): Newton sizes it
  as every particle x shape of the view (solver_vbd.py:1474-1478, 28 B each) and only grows it later;
- rigid pairs (``rigid_shapes``, the destination's own shapes): Newton's explicit pairs of the view, which already
  hide shapes neither owned nor proxied, kept only when one shape is the destination's own. Contacts among proxies or
  between a proxy and static geometry are the source's (it resolves them; resolving them again in the destination puts
  them in the proxy feedback twice). A particle-only destination (VBD pad, SPH) gets no rigid pairs, so no narrow phase
  or contact reducer (3.76 GB at 1,024 DC5 drip worlds). ``rigid_contact_max`` = ``rigid_contacts_per_world`` x
  worlds when set, 0 with no pairs, else Newton's; ``max_triangle_pairs`` = ``triangle_pairs_per_world`` x worlds
  when set (Newton's estimate is the busiest world's mesh pairs x 128 x worlds: 12.6 GB at 1,024 DC5 worlds);
- after each collide, the rows on watertight (parity-signed) mesh shapes that claim inside asked again with the
  pseudo-normal sign, flipped where it says outside (``alloy.contact.mesh_sign``: Warp's parity says inside at
  degenerate points, WP2's trace);
- on CUDA, the soft records in a canonical order after each collide (``alloy.coupling.contact_order``, SCR-SC4-4d):
  Newton's slots come from atomic counters, so their order, and VBD's sums over them, changed from run to run;
- an int32[1] overflow counter on the device, sticky until ``reset_overflow``: contacts dropped (soft, rigid or
  triangle-pair count above capacity, or the contact reducer's loss flag) plus shapes excluded at construction whose
  COLLIDE_PARTICLES is on now (their contacts would be missing). The buffer is global, so a nonzero count cannot be
  attributed to a world and fails closed: ``Scheduler.capacity_flags()`` hands it to HealthMonitor (CONTACT_OVERFLOW
  -> INVALID on every world, SC0-1 policy).

The simulation's fallback buffer (``alloy.scene.steppers.NewtonContacts``, SCR-WP4-6) builds through the same factory
on the full model, with ``filter_candidates`` only when a budget is set (without one it is Newton's pipeline) or when
it leaves out the particles of contact-free entries (SPH, SCR-JOB2-4: ``particles``).
"""

from __future__ import annotations

import functools
import logging
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.contact import mesh_cull
from alloy.contact.mesh_sign import ParityRecheck
from alloy.core.errors import CouplingError
from alloy.coupling.contact_order import SoftContactOrder
from alloy.coupling.flat_edges import FlatEdgeMute
from alloy.coupling.full_surface import FULL_SURFACE_GAP
from alloy.coupling.pass_through import PassThroughFlags
from alloy._vendor.newton._src.sim import collide as _collide
from alloy._vendor.newton._src.solvers.vbd import solver_vbd as _vbd

log = logging.getLogger("alloy.coupling")

COLLIDE_PARTICLES = int(newton.ShapeFlags.COLLIDE_PARTICLES)
PROXY_PARTICLE = int(newton.ParticleFlags.PROXY)

# Destination capabilities whose step never reads ``contacts``: SPH (alloy/solvers/sph/solver.py SolverDFSPH.step:
# ``del control, contacts``; the hub's shapes are its sampled boundary) and MPM (SolverImplicitMPM.step: "contacts:
# unused; collisions are handled internally", its colliders are rasterized from the shapes it sees). Their proxy gets
# no pipeline (SCR-DC5-19), their pairs no contact-stiffness clock bound (SCR-JOB2-3): there is no penalty spring at
# that interface, and their particles no fallback soft candidates (SCR-JOB2-4). MPM joined for CPL-3 (engine item 2):
# the spring bound clocked a cup on sand at dt_x 3.9e-5 s, and the soft-contact PEN rule latched a 1.4 mm overlap.
CONTACT_FREE_DESTINATIONS = frozenset({"sph", "mpm"})

FULL_SURFACE_KW = "enable_rigid_soft_full_surface_contact"
# Full-surface soft records per soft feature (vertex, edge, triangle) of the view: Newton reports every vertex-face,
# face-vertex and edge-edge pair in the contact band, redundant ones included (a two-pad rig with the whole cloth
# inside both pads' 1 cm band: 10 per feature). Newton's own estimate scales with features x shapes per world (117 M
# for C7's 16 worlds, 4x the step time) and can also fall short (2,992 for 3,410 on that rig); overflow fails closed.
FULL_SURFACE_RECORDS = 16


def soft_feature_count(view: Any, particle_ok: np.ndarray | None = None) -> int:
    """Particles (``particle_ok`` ones when given) + edges + triangles of ``view``."""
    n = int(view.particle_count or 0) if particle_ok is None else int(np.count_nonzero(particle_ok))
    return n + int(getattr(view, "edge_count", 0) or 0) + int(getattr(view, "tri_count", 0) or 0)


def contact_free(solver: Any, capability: str | None = None) -> bool:
    """True when the destination's capability is in ``CONTACT_FREE_DESTINATIONS``: ``capability`` when given (the
    entry's, from make_stepper), else the registered capability nearest in ``solver``'s MRO (capabilities registered
    without ``solver_cls``, e.g. today's "sph", are only found by name)."""
    if capability is not None:
        return capability in CONTACT_FREE_DESTINATIONS
    from alloy.core.registry import capability_for  # noqa: PLC0415 - registry imports solver plug-ins lazily

    cap = capability_for(solver)
    return cap is not None and cap.name in CONTACT_FREE_DESTINATIONS


@wp.kernel(enable_backward=False)
def dropped_contacts_kernel(count: wp.array[wp.int32], cap: int, out: wp.array[wp.int32]):
    over = count[0] - cap
    if over > 0:
        wp.atomic_max(out, 0, over)


@wp.kernel(enable_backward=False)
def flag_set_kernel(flag: wp.array[wp.int32], out: wp.array[wp.int32]):
    if flag[0] != 0:
        wp.atomic_add(out, 0, 1)


@wp.kernel(enable_backward=False)
def excluded_enabled_kernel(ids: wp.array[wp.int32], flags: wp.array[wp.int32], bit: int, out: wp.array[wp.int32]):
    if (flags[ids[wp.tid()]] & bit) != 0:
        wp.atomic_add(out, 0, 1)


def _shape_mask(model: Any, shape_ok: np.ndarray | None, collide: np.ndarray | None = None) -> np.ndarray:
    """Shapes that collide with particles (``collide``, else the flags now) and pass Newton's own ``shape_ok``."""
    ok = ((model.shape_flags.numpy() & COLLIDE_PARTICLES) != 0) if collide is None else np.array(collide, dtype=bool)
    if shape_ok is not None:
        ok &= np.asarray(shape_ok, dtype=bool)
    return ok


def _particle_world(model: Any, particle_ok: np.ndarray | None) -> np.ndarray:
    """``particle_world`` with the particles outside ``particle_ok`` moved to world ``world_count``, which pairs with
    no shape (neither global nor a local world): they get no candidates, the others keep Newton's order."""
    p_world = model.particle_world.numpy()
    if particle_ok is None:
        return p_world
    ok = np.asarray(particle_ok, dtype=bool)
    if ok.shape != p_world.shape:
        raise ValueError(f"particle_ok has {ok.size} entries for {p_world.size} particles")
    p_world = p_world.copy()
    p_world[~ok] = max(int(model.world_count or 0), 0)
    return p_world


def filtered_soft_pairs(model: Any, *, shape_ok: np.ndarray | None = None, particle_ok: np.ndarray | None = None,
                        collide: np.ndarray | None = None) -> wp.array:
    """Newton's soft particle-shape candidates restricted to COLLIDE_PARTICLES shapes (``collide``, else the flags at
    call time) and to ``particle_ok`` particles (None: all), in Newton's order."""
    if not int(model.particle_count or 0) or not int(model.shape_count or 0):
        return wp.array(np.empty((0, 2), np.int32), dtype=wp.vec2i, device=model.device)
    return _collide._world_compatible_pairs(_particle_world(model, particle_ok), model.shape_world.numpy(),
                                            int(model.world_count or 0), model.device,
                                            _shape_mask(model, shape_ok, collide))


def soft_pair_count(model: Any, *, shape_ok: np.ndarray | None = None, particle_ok: np.ndarray | None = None) -> int:
    """``len(filtered_soft_pairs(model))`` on the host (the three disjoint groups of ``_world_compatible_pairs``)."""
    if not int(model.particle_count or 0) or not int(model.shape_count or 0):
        return 0
    worlds = int(model.world_count or 0)
    p_world, ok = _particle_world(model, particle_ok), _shape_mask(model, shape_ok)
    s_world = model.shape_world.numpy()[ok]
    p_local, s_local = (p_world >= 0) & (p_world < worlds), (s_world >= 0) & (s_world < worlds)
    per_world = np.bincount(p_world[p_local], minlength=worlds) if worlds else np.zeros(0, np.int64)
    return (int(np.sum(p_world < 0)) * len(s_world) + int(np.sum(p_local)) * int(np.sum(s_world < 0))
            + int(per_world[s_world[s_local]].sum()))


_HOOKS = ((_collide, "_build_soft_particle_rigid_contact_pairs"), (_collide, "_world_compatible_pairs"),
          (_vbd, "_count_soft_particle_rigid_contact_pairs"), (_collide, "_full_surface_capable_shape_mask"))


def _check_newton_hooks() -> None:
    """The newton@ac7f7830 private names this module relies on (SCR-WP4-5 asks WP0 to pin them)."""
    for module, name in _HOOKS:
        if not callable(getattr(module, name, None)):
            raise CouplingError(f"{module.__name__}.{name} is missing (Newton pin changed); see SCR-WP4-5")


@contextmanager
def _filtered_builder(particle_ok: np.ndarray | None = None, collide: np.ndarray | None = None) -> Iterator[None]:
    """Route CollisionPipeline's candidate builder through ``filtered_soft_pairs`` while one pipeline is built."""
    original = _collide._build_soft_particle_rigid_contact_pairs
    _collide._build_soft_particle_rigid_contact_pairs = functools.partial(filtered_soft_pairs, particle_ok=particle_ok,
                                                                          collide=collide)
    try:
        yield
    finally:
        _collide._build_soft_particle_rigid_contact_pairs = original


@contextmanager
def _full_surface_only(view: Any, shapes: np.ndarray | None) -> Iterator[np.ndarray]:
    """While one pipeline is built: Newton's edge/face passes only for ``shapes`` (None: every capable shape). Newton
    takes analytic shapes from ``_full_surface_capable_shape_mask`` and meshes by COLLIDE_PARTICLES at construction,
    so other meshes lose the flag until the pipeline exists (they keep per-vertex candidates: the filtered builder
    reads the yielded flags from before, and Newton applies the flag per thread at run time). Yields those flags."""
    flags = view.shape_flags.numpy()
    collide = (flags & COLLIDE_PARTICLES) != 0
    if shapes is None:
        yield collide
        return
    keep = np.zeros(flags.shape[0], dtype=bool)
    keep[shapes[(shapes >= 0) & (shapes < keep.size)]] = True
    is_mesh = np.isin(view.shape_type.numpy(), (int(newton.GeoType.MESH), int(newton.GeoType.CONVEX_MESH)))
    off = is_mesh & collide & ~keep
    original = _collide._full_surface_capable_shape_mask
    _collide._full_surface_capable_shape_mask = lambda model: original(model) & keep
    if off.any():
        view.shape_flags.assign(np.where(off, flags & ~COLLIDE_PARTICLES, flags).astype(flags.dtype))
    try:
        with warnings.catch_warnings():         # the shapes left out are not "fallbacks" (Newton's warning)
            warnings.filterwarnings("ignore", message="enable_rigid_soft_full_surface_contact=True: these")
            yield collide
    finally:
        _collide._full_surface_capable_shape_mask = original
        if off.any():
            view.shape_flags.assign(flags)


@contextmanager
def entry_contact_sizing(factories: Sequence[ProxyContactPipeline]) -> Iterator[None]:
    """While a coupler builds its entry solvers: SolverVBD's soft-contact state hint is the largest capacity
    ``factories`` would give that view, not every particle x shape. A VBD that is no proxy destination grows to its
    contacts' ``soft_contact_max`` on its first (eager) step, as Newton does for any short hint."""
    if not factories:
        yield
        return
    _check_newton_hooks()
    original = _vbd._count_soft_particle_rigid_contact_pairs

    def hint(view: Any) -> int:
        return min(int(original(view)), max(f.capacity(view) for f in factories))

    _vbd._count_soft_particle_rigid_contact_pairs = hint
    try:
        yield
    finally:
        _vbd._count_soft_particle_rigid_contact_pairs = original


def owned_rigid_pairs(view: Any, owned_shapes: Sequence[int]) -> wp.array:
    """The view's explicit shape pairs (Newton's, already limited to owned and proxied shapes) with at least one shape
    in ``owned_shapes``, in Newton's order. Shape ids are global in an entry view (Newton never compacts shapes)."""
    pairs = getattr(view, "shape_contact_pairs", None)
    host = np.empty((0, 2), np.int32) if pairs is None or pairs.shape[0] == 0 else pairs.numpy().reshape(-1, 2)
    n = int(view.shape_count or 0)
    own = np.zeros(n + 1, dtype=bool)                      # [n]: out-of-range / -1 entries never count as owned
    ids = np.asarray(list(owned_shapes), dtype=np.int64)
    own[ids[(ids >= 0) & (ids < n)]] = True
    a, b = np.where((host[:, 0] >= 0) & (host[:, 0] < n), host[:, 0], n), np.where(
        (host[:, 1] >= 0) & (host[:, 1] < n), host[:, 1], n)
    kept = host[own[a] | own[b]].astype(np.int32)
    return wp.array(kept.reshape(-1, 2), dtype=wp.vec2i, device=view.device)


def _triangle_capacity(pipeline: Any) -> int:
    """The narrow phase's triangle-pair buffer size (0 when it has no mesh-like route)."""
    narrow = getattr(pipeline, "narrow_phase", None)
    pairs = getattr(narrow, "triangle_pairs", None)
    return 0 if pairs is None else int(pairs.shape[0])


def _excluded(flags: wp.array, bit: int) -> wp.array:
    ids = np.flatnonzero((flags.numpy() & bit) == 0).astype(np.int32)
    return wp.array(ids, dtype=wp.int32, device=flags.device)


def _world_reset_mask(view: Any, worlds: np.ndarray) -> wp.array:
    """Newton's reset mask (``world_count + 1`` bools, the last for global shapes, left unset) from per-world bools;
    a mask longer than the view's worlds (Newton's own, with its global entry) is cut to them."""
    w = max(int(view.world_count or 0), 1)
    sel = np.asarray(worlds, dtype=bool).reshape(-1)
    if sel.shape[0] < w:
        raise ValueError(f"reset_matching: {sel.shape[0]} world flags for a view of {w} worlds")
    return wp.array(np.append(sel[:w], False), dtype=wp.bool, device=view.device)


class ProxyContactPipeline:
    """Callable ``view -> CollisionPipeline`` for a proxy (module docstring); keeps what it built for the checks."""

    def __init__(self, soft_contacts_per_world: int | None = None, rigid_contacts_per_world: int | None = None, *,
                 rigid_shapes: Sequence[int] | None = None, triangle_pairs_per_world: int | None = None,
                 filter_candidates: bool = True, label: str = "proxy", pass_through: bool = True,
                 particles: Sequence[int] | None = None, proxied: bool = False, full_surface: bool = False,
                 full_surface_shapes: Sequence[int] | None = None, stable_order: bool = True,
                 **pipeline_kwargs: Any) -> None:
        """``rigid_shapes``: the destination's own shapes (None: every pair of the view, Newton's default).
        ``triangle_pairs_per_world``: Newton's ``max_triangle_pairs`` = that x worlds (None: Newton's estimate).
        ``filter_candidates=False`` keeps Newton's full soft candidate list (and so no excluded-shape check).
        ``pass_through``: check each collide for particle centres inside a shape (``alloy.coupling.pass_through``).
        ``particles``: the only particles given soft candidates (None: all; needs ``filter_candidates``), e.g. the
        fallback buffer without contact-free (SPH) particles (SCR-JOB2-4).
        ``proxied``: with ``particles``, also the view's PROXY particles at build (a destination's own particles plus
        those proxied into it: the particles its solve uses; WP4's review of WP10 lever #4)
        ``full_surface``: Newton's edge/face soft contacts as well (SCR-SC4-4; only SolverVBD consumes them), for the
        shapes ``full_surface_shapes`` (None: every capable shape); other shapes keep per-vertex contact.
        ``stable_order``: on CUDA, put each collide's soft records in a canonical order
        (``alloy.coupling.contact_order``, SCR-SC4-4d) instead of the slot order Newton's atomic counters give; the
        CPU order is already reproducible."""
        for name, v in (("soft_contacts", soft_contacts_per_world), ("rigid_contacts", rigid_contacts_per_world),
                        ("triangle_pairs", triangle_pairs_per_world)):
            if v is not None and v < 1:                    # 0: Newton skips soft contacts, uncounted
                raise ValueError(f"{name}_per_world must be >= 1 or None, got {v}")
        self.soft_contacts_per_world = soft_contacts_per_world
        self.rigid_contacts_per_world = rigid_contacts_per_world
        self.triangle_pairs_per_world = triangle_pairs_per_world
        self.rigid_shapes = None if rigid_shapes is None else tuple(int(s) for s in rigid_shapes)
        self.filter_candidates, self.label, self.check_pass_through = bool(filter_candidates), label, bool(pass_through)
        self.stable_order = bool(stable_order)
        self.particles = None if particles is None else np.unique(np.asarray(particles, dtype=np.int64))
        self.proxied = bool(proxied)
        if self.particles is not None and not self.filter_candidates:
            raise ValueError("particles needs filter_candidates=True")
        self.kwargs = dict(pipeline_kwargs)
        self.full_surface = bool(full_surface or self.kwargs.get(FULL_SURFACE_KW, False))
        if self.full_surface:
            self.kwargs[FULL_SURFACE_KW] = True
            self.kwargs.setdefault("soft_contact_gap", FULL_SURFACE_GAP)
        self.full_surface_shapes = None if full_surface_shapes is None else \
            np.unique(np.asarray(full_surface_shapes, dtype=np.int64))
        self.built: list[tuple[Any, Any, Any, wp.array]] = []   # (view, pipeline, contacts, excluded shapes)
        self.flat_edges: list[FlatEdgeMute | None] = []         # per built pipeline (SCR-SC4-6), full-surface only
        self.orders: list[SoftContactOrder | None] = []         # per built pipeline (SCR-SC4-4d), CUDA only
        self.signs: list[ParityRecheck] = []                     # per built pipeline: Warp parity re-check (WP2's trace)
        self.overflow: wp.array | None = None
        self.pass_through: PassThroughFlags | None = None     # SCR-SC4-4: per-world particle-centre-inside flags

    def capacity(self, view: Any) -> int:
        """Soft contacts the pipeline on ``view`` holds: the filtered candidates, or ``soft_contacts_per_world`` x
        worlds when smaller. Full-surface: ``soft_contacts_per_world`` x worlds when set, else
        ``FULL_SURFACE_RECORDS`` per soft feature (module docstring)."""
        worlds = max(int(view.world_count or 0), 1)
        if self.full_surface:
            if self.soft_contacts_per_world is not None:
                return self.soft_contacts_per_world * worlds
            return max(FULL_SURFACE_RECORDS * soft_feature_count(view, self._particle_ok(view)), 1)
        if not self.filter_candidates:
            n = int(_collide._count_soft_particle_rigid_contact_pairs(view))
        else:
            n = soft_pair_count(view, particle_ok=self._particle_ok(view))
        return n if self.soft_contacts_per_world is None else min(n, self.soft_contacts_per_world * worlds)

    def _particle_ok(self, view: Any) -> np.ndarray | None:
        if self.particles is None:
            return None
        ok = np.zeros(int(view.particle_count or 0), dtype=bool)
        ok[self.particles[self.particles < ok.size]] = True
        if self.proxied and ok.size:
            ok |= (view.particle_flags.numpy() & PROXY_PARTICLE) != 0
        return ok

    def pipeline_kwargs(self, view: Any) -> dict[str, Any]:
        """CollisionPipeline keyword arguments for ``view`` (private pipeline: capacities are not published)."""
        kw: dict[str, Any] = {"publish_capacity": False, **self.kwargs}
        explicit = self.full_surface or self.soft_contacts_per_world is not None
        kw["soft_contact_max"] = self.capacity(view) if explicit else None
        if self.rigid_shapes is not None:
            pairs = owned_rigid_pairs(view, self.rigid_shapes)
            kw.update(broad_phase="explicit", shape_pairs_filtered=pairs)
            if pairs.shape[0] == 0:
                kw["rigid_contact_max"] = 0                # no pairs: no narrow phase, no reducer
        worlds = max(int(view.world_count or 0), 1)
        if self.rigid_contacts_per_world is not None and kw.get("rigid_contact_max") != 0:
            kw["rigid_contact_max"] = self.rigid_contacts_per_world * worlds
        if self.triangle_pairs_per_world is not None:
            kw["max_triangle_pairs"] = self.triangle_pairs_per_world * worlds
        return kw

    def __call__(self, view: Any) -> Any:
        _check_newton_hooks()
        kw = self.pipeline_kwargs(view)
        with _full_surface_only(view, self.full_surface_shapes if self.full_surface else None) as collide:
            if self.filter_candidates:
                with _filtered_builder(self._particle_ok(view), collide):
                    pipeline = newton.CollisionPipeline(view, **kw)
            else:
                pipeline = newton.CollisionPipeline(view, **kw)
        if self.filter_candidates:
            excluded = _excluded(view.shape_flags, COLLIDE_PARTICLES)
        else:
            excluded = wp.zeros(0, dtype=wp.int32, device=view.device)
        if self.overflow is None:
            self.overflow = wp.zeros(1, dtype=wp.int32, device=view.device)
        if self.pass_through is None and self.check_pass_through:
            self.pass_through = PassThroughFlags(view.world_count, view.device)
        rigid = kw.get("shape_pairs_filtered")
        log.info("%s collision pipeline: %d soft candidates (of %d particle x shape), soft_contact_max %d; rigid "
                 "pairs %s, rigid_contact_max %d; max_triangle_pairs %d", self.label,
                 int(pipeline.soft_contact_pair_count), int(view.particle_count or 0) * int(view.shape_count or 0),
                 int(pipeline.soft_contact_max), "all of the view" if rigid is None else int(rigid.shape[0]),
                 int(pipeline.rigid_contact_max), _triangle_capacity(pipeline))
        if self.stable_order:                              # WP10 lever #4: culled mesh vertex pairs on CUDA
            mesh_cull.attach(pipeline)                     # (their slots are put in canonical order after collide)
        self.built.append((view, pipeline, None, excluded))
        self.signs.append(ParityRecheck(view))
        self.flat_edges.append(FlatEdgeMute(view, pipeline) if self.full_surface else None)
        self.orders.append(None)
        return _Tracked(self, len(self.built) - 1, pipeline)

    def _bind_contacts(self, index: int, contacts: Any) -> None:
        view, pipeline, bound, ex_s = self.built[index]
        self.built[index] = (view, pipeline, contacts, ex_s)
        if contacts is bound:
            return
        # differentiable contacts keep Newton's slots: backward replay maps each thread to the slot it wrote
        grad = bool(getattr(contacts.soft_contact_body_pos, "requires_grad", False))
        use = self.stable_order and view.device.is_cuda and not grad
        self.orders[index] = SoftContactOrder(contacts, view) if use else None

    def update_overflow(self) -> None:
        """Device-only: add this exchange's dropped contacts and re-enabled exclusions to the sticky counter."""
        for view, pipeline, contacts, ex_s in self.built:
            dev = view.device
            triangles = getattr(getattr(pipeline, "narrow_phase", None), "triangle_pairs_count", None)
            if triangles is not None:                      # mesh triangle pairs over max_triangle_pairs
                wp.launch(dropped_contacts_kernel, dim=1, device=dev,
                          inputs=[triangles, _triangle_capacity(pipeline), self.overflow])
            if contacts is not None:
                wp.launch(dropped_contacts_kernel, dim=1, device=dev,
                          inputs=[contacts.soft_contact_count, int(contacts.soft_contact_max), self.overflow])
                wp.launch(dropped_contacts_kernel, dim=1, device=dev,
                          inputs=[contacts.rigid_contact_count, int(contacts.rigid_contact_max), self.overflow])
                wp.launch(flag_set_kernel, dim=1, device=dev,              # [2]: the contact reducer dropped some
                          inputs=[contacts.contact_counters[2:3], self.overflow])
            if ex_s.shape[0]:
                wp.launch(excluded_enabled_kernel, dim=ex_s.shape[0], device=dev,
                          inputs=[ex_s, view.shape_flags, COLLIDE_PARTICLES, self.overflow])

    def reset_overflow(self) -> None:
        if self.overflow is not None:
            self.overflow.zero_()

    def reset_matching(self, worlds: np.ndarray | None = None) -> int:
        """Clear the cross-step contact matching (a rod entry's "sticky", SCR-WP9-ROD F3) of every pipeline built with
        it, for ``worlds`` (bool per world; None: every world and the global shapes). ``alloy.coupling.match_reset``
        says when. Returns the pipelines cleared."""
        n = 0
        for view, pipeline, _, _ in self.built:
            if getattr(pipeline, "contact_matching", "disabled") == "disabled":
                continue
            pipeline.reset_contact_matching(None if worlds is None else _world_reset_mask(view, worlds))
            n += 1
        return n


class _Tracked:
    """The pipeline Newton stores for the proxy; records the Contacts it hands out so overflow can be checked."""

    def __init__(self, owner: ProxyContactPipeline, index: int, pipeline: Any) -> None:
        self._owner, self._index, self.pipeline = owner, index, pipeline

    def contacts(self) -> Any:
        contacts = self.pipeline.contacts()
        self._owner._bind_contacts(self._index, contacts)
        return contacts

    def collide(self, state: Any, contacts: Any, **kwargs: Any) -> Any:
        out = self.pipeline.collide(state, contacts, **kwargs)
        owner = self._owner
        owner.signs[self._index].apply(owner.built[self._index][0], state, contacts)   # parity-signed inside rows
        mute = owner.flat_edges[self._index] if self._index < len(owner.flat_edges) else None
        if mute is not None:                               # SCR-SC4-6: flat-edge edge-edge records duplicate faces
            mute.apply(contacts)
        order = owner.orders[self._index]                  # SCR-SC4-4d: slots come from atomic counters; after the
        if order is not None and order.contacts is contacts:  # mute, whose tag change enters the key
            order.apply()
        if owner.pass_through is not None:                 # SCR-SC4-4: particle centres inside a shape
            owner.pass_through.check(owner.built[self._index][0], state, contacts)
        return out

    def __getattr__(self, name: str) -> Any:
        return getattr(self.pipeline, name)
