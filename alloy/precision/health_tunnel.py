# SPDX-License-Identifier: Apache-2.0
"""TUNNEL / FAST_CONTACT: a rigid body passing through a static shape (audit ERR-2).

Penetration checks read the contacts that exist after a step, so a body that crosses a thin static plate in one step
leaves no contact and health read OK. This check follows each judged body's MATTER POINTS (its COM when that lies inside
one of its own shapes, plus up to 3 points per primitive shape: sphere / ellipsoid centre; capsule / cylinder centre and
the two axis points; box centre and the two points at +-(h_max - h_mid) on its longest axis; at most 24 per body, the
rest dropped, first shapes kept; a convex mesh gives its COM only) against the static shapes of its world, every
rigid step (it wraps the MuJoCo solver instance's ``step``: also the k rigid steps of a
coupled exchange, independent of ``CheckPolicy.every``, inside a captured period graph). Judged bodies and static
shapes and the pair exclusions: ``health_tunnel_setup``.

Rules below read "the point" = a matter point (each has its own entry tracking); FAST_CONTACT is judged on the COM only.
Per point, static shape s and step (p0 -> p1, signed distances d0, d1, outward normals n0, n1):
- FAST_CONTACT (DEGRADED), all three: (1) the COM approached s by ``d0 - d1 > TRAVEL_LIMIT_M`` in this one rigid step
  (the constant ``alloy.core.defaults.RIGID_GAP_M``, 5 mm, whatever gap or margin the shapes carry: MuJoCo applies no
  force before overlap, so the contact gap plays no part in a pass-through); (2) s is not a plane (a half-space cannot
  be passed) and its extent along the approach normal (twice its support there) is under twice the travel (a thicker
  part is met by a contact first); (3) the body's near surface ends within the limit of s: the contact was not seen
  before the surfaces met. A body passing by, sliding or rolling along s does not flag.
  It is not sticky: an event sets a per-world counter to ``HOLD_STEPS`` = 64 rigid steps, every watched step first
  counts non-zero counters down, and the flag shows while the counter is non-zero. A health check must surely see it:
  checks run every exchange on the standard and precision profiles and every 10th exchange on profile fast, an exchange
  takes k rigid steps (a coupled one a few), so 10 x k <= 64 for k <= 6; the key of the largest event of the window
  (travel, body, shape) stays readable for the text and goes with the counter.
- Entry: not yet inside s, ``d0 >= 0`` and ``d1 < 0``: remember s and n0. Both ends outside but the segment meets the
  solid (exact slab test for boxes, <= 16 samples otherwise): TUNNEL when ``n1 . n0 < -0.5`` (it came out of the
  opposite side) and, for a box, the face pair nearest the middle of the run inside is the pair of n0, or (a box) it
  came out of any other side (``< 0.5``) with the middle of the run ``SIDE_DEPTH_M`` or more inside: the in-and-out
  rules below, judged at mid-run (AUDIT-STACK 17:51Z: an edge-on rail, a pass 2 mm inside a plate's edge).
- Inside s: ``d1 >= 0`` and ``n1 . n_entry < -0.5`` is TUNNEL (it left through the opposite face) when, for a box, the
  face pair nearest the point at its deepest inside is the entry / exit pair (``in_thru``; ties count): it went through the
  part's thickness. A point that slides along a face while shallowly inside a thin box, in by its front face and out by
  its back face, never was nearest to either (DC2 eth_swap's latch tab under the port roof, 2026-11-10); so is leaving by
  another face (``n1 . n_entry < 0.5``) after the point was ``SIDE_DEPTH_M`` = 2 mm or more inside (the deepest depth is
  tracked): a pass through a corner; shallower grazes are not. Any exit clears the entry. The COM point alone: wholly
  inside (``-d1 >= r_out``, e.g. below a ground plane) is TUNNEL. A point that starts
  inside a static shape is never judged against it (START_PEN's business). No speed condition.
- Dynamic-vs-dynamic pairs, kinematic bodies and static meshes that are not box-like are not judged (a static convex
  mesh whose hull fills 98 % of its bounding box is judged as that box: ``health_tunnel_setup``).

Static shapes are judged where they are now: the watch reads the model's ``shape_transform`` array on the device every
step (``alloy.episode.layout`` moves a world's statics after the build with ``.assign``, in place). A caller that rebinds
``model.shape_transform`` to a new array leaves the watch on the old one; a static moved while a point is tracked
inside it can read as an exit, so move statics at a reset (layouts do; the reset clears the tracking).

TUNNEL is sticky per world until ``reset(mask)``; both reach health as ``nonzero`` capacity sources (FAST_CONTACT
DEGRADED S2, TUNNEL INVALID S1). The launches use device arrays fixed at build; host reads happen only in ``notes``
(``sim.health()``).

Rollback: the tracker (flags, keys, per-body entry tracking) is solver state. ``install_state_adapter`` adds its arrays
to the MuJoCo solver's state adapter (composed with the registered one), so every snapshot restore (a discarded or
re-run pass, the latch, a task's restore, ``sim.reset``) puts them back with the bodies, per world. Only the ``seen_*``
copies of the keys stay out of snapshots: a frozen world's tracker is restored every step, its text must not be.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.precision import health_tunnel_kernels as K
from alloy.precision import health_tunnel_setup as S
from alloy.precision.types import HealthFlags

log = logging.getLogger("alloy.precision")
WATCH_ATTR = "_alloy_tunnel_watch"


class TunnelWatch:
    """Per-world sticky ``fast`` / ``tunnel`` int32 [W] flags of one model's rigid bodies against its static shapes."""

    def __init__(self, model: newton.Model, bodies: list[S.JudgedBody], statics: np.ndarray) -> None:
        self.model, self.bodies, self.statics = model, bodies, statics
        self.device, self.world_count = model.device, max(int(model.world_count), 1)
        self.com = model.body_com
        self.dt = 0.0                                             # last rigid step (s)
        self.st, self._by_world = self._static_table(statics)
        self.bt, self._local = self._body_table(bodies)
        self.it, self.item_world = self._item_table(bodies)
        n, W = len(self.item_world), self.world_count
        self.tr = K.Tracker()
        self.tr.in_shape = wp.full(n, -1, dtype=wp.int32, device=self.device)
        self.tr.in_normal = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.tr.in_depth = wp.zeros(n, dtype=wp.float32, device=self.device)
        self.tr.in_thru = wp.zeros(n, dtype=wp.int32, device=self.device)
        self.tr.fast, self.tr.tunnel = (wp.zeros(W, dtype=wp.int32, device=self.device) for _ in range(2))
        self.tr.fast_key, self.tr.tunnel_key = (wp.zeros(W, dtype=wp.uint64, device=self.device) for _ in range(2))
        self.tr.seen_fast, self.tr.seen_tunnel = (wp.zeros(W, dtype=wp.uint64, device=self.device) for _ in range(2))
        self.fast, self.tunnel = self.tr.fast, self.tr.tunnel
        self._worlds = wp.array(np.arange(W, dtype=np.int32), dtype=wp.int32, device=self.device)

    def _static_table(self, statics: np.ndarray) -> tuple[K.StaticTable, dict[int, list[int]]]:
        m = self.model
        st, d = K.StaticTable(), self.device
        scale, types_ = m.shape_scale.numpy(), m.shape_type.numpy()
        st.kind = wp.array(types_[statics].astype(np.int32), dtype=wp.int32, device=d)
        st.half = wp.array(scale[statics][:, :3].astype(np.float32), dtype=wp.vec3, device=d)
        st.shape = wp.array(np.asarray(statics, np.int32), dtype=wp.int32, device=d)
        st.xform = m.shape_transform                                       # live: layouts move statics after the build
        local = np.tile(np.array([0, 0, 0, 0, 0, 0, 1], np.float32), (len(statics), 1))
        for pos, (half, centre) in S.box_overrides(m, statics).items():    # box-like convex meshes: a BOX entry
            kind, hs = st.kind.numpy(), st.half.numpy()
            kind[pos], hs[pos], local[pos, :3] = K.T_BOX, half, centre
            st.kind, st.half = wp.array(kind, dtype=wp.int32, device=d), wp.array(hs, dtype=wp.vec3, device=d)
        st.local = wp.array(local, dtype=wp.transform, device=d)
        by_world = S.statics_by_world(m, statics)
        start, flat = S.csr([by_world[w] for w in range(self.world_count)])
        st.world_start = wp.array(start, dtype=wp.int32, device=d)
        st.world_idx = wp.array(flat, dtype=wp.int32, device=d)
        return st, by_world

    def _body_table(self, bodies: list[S.JudgedBody]) -> tuple[K.BodyTable, np.ndarray]:
        bt, d = K.BodyTable(), self.device
        ex = S.excluded_pairs(self.model, bodies, self.statics, self._by_world)
        start, flat = S.csr([ex.get(j, []) for j in range(len(bodies))])
        rank: dict[int, int] = {}
        local = np.zeros(len(bodies), np.int32)
        for j, b in enumerate(bodies):
            local[j] = rank[b.world] = rank.get(b.world, -1) + 1
        shapes = [s for b in bodies for s in b.shapes]
        sh_start = np.zeros(len(bodies) + 1, np.int32)
        sh_start[1:] = np.cumsum([len(b.shapes) for b in bodies])
        bt.body = wp.array([b.body for b in bodies], dtype=wp.int32, device=d)
        bt.world = wp.array([b.world for b in bodies], dtype=wp.int32, device=d)
        bt.local = wp.array(local, dtype=wp.int32, device=d)
        bt.r_out = wp.array([b.r_out for b in bodies], dtype=wp.float32, device=d)
        bt.ex_start, bt.ex_idx = wp.array(start, dtype=wp.int32, device=d), wp.array(flat, dtype=wp.int32, device=d)
        bt.sh_start = wp.array(sh_start, dtype=wp.int32, device=d)
        bt.sh_kind = wp.array([s.gtype for s in shapes], dtype=wp.int32, device=d)
        bt.sh_half = wp.array(np.array([s.half for s in shapes], np.float32).reshape(-1, 3), dtype=wp.vec3, device=d)
        bt.sh_pose = wp.array(np.array([s.xf for s in shapes], np.float32).reshape(-1, 7), dtype=wp.transform, device=d)
        return bt, local

    def _item_table(self, bodies: list[S.JudgedBody]) -> tuple[K.ItemTable, wp.array]:
        """One row per matter point: the COM of a body solid about it, then its shapes' points (module doc)."""
        rows = [(j, p, int(k == 0 and b.has_com)) for j, b in enumerate(bodies)
                for k, p in enumerate(([np.zeros(3)] if b.has_com else []) + list(b.points))]
        it, d = K.ItemTable(), self.device
        it.body = wp.array([r[0] for r in rows], dtype=wp.int32, device=d)
        it.point = wp.array(np.array([r[1] for r in rows], np.float32).reshape(-1, 3), dtype=wp.vec3, device=d)
        it.is_com = wp.array([r[2] for r in rows], dtype=wp.int32, device=d)
        return it, wp.array([bodies[r[0]].world for r in rows], dtype=wp.int32, device=d)

    def after_step(self, state_in: newton.State, state_out: newton.State, dt: float = 0.0) -> None:
        """Judge one rigid step (state_in -> state_out): one launch, no host read."""
        self.dt = float(dt) or self.dt
        wp.launch(K.age_fast, dim=self.world_count, inputs=[self.tr], device=self.device)
        wp.launch(K.tunnel_step, dim=len(self.item_world), inputs=[state_in.body_q, state_out.body_q, self.com,
                                                                  self.bt, self.it, self.st, self.tr], device=self.device)
        wp.launch(K.remember_keys, dim=self.world_count, inputs=[self.tr], device=self.device)

    def state_arrays(self) -> dict[str, wp.array]:
        """The tracker arrays a snapshot must restore with the bodies (name -> array; maps in ``state_world_maps``)."""
        t = self.tr
        return {"tunnel.in_shape": t.in_shape, "tunnel.in_normal": t.in_normal, "tunnel.in_depth": t.in_depth,
                "tunnel.in_thru": t.in_thru, "tunnel.fast": t.fast, "tunnel.tunnel": t.tunnel, "tunnel.fast_key": t.fast_key,
                "tunnel.tunnel_key": t.tunnel_key}

    def state_world_maps(self) -> dict[str, wp.array]:
        """Item -> world int32 maps of ``state_arrays`` (per matter point: its world; per world: the world itself)."""
        return {k: (self.item_world if k in _ITEM_KEYS else self._worlds)
                for k in self.state_arrays()}

    def reset(self, world_mask: wp.array | None = None) -> None:
        """Forget the masked worlds' flags and entry tracking (all when None; int32 [>= W], nonzero = reset)."""
        mask = world_mask if world_mask is not None else self.tr.fast
        every = int(world_mask is None)
        wp.launch(K.clear_flags, dim=self.world_count, inputs=[mask, every, self.tr], device=self.device)
        wp.launch(K.clear_worlds, dim=len(self.item_world), inputs=[mask, every, self.tr, self.item_world],
                  device=self.device)

    def notes(self, dt: float) -> list[dict[str, Any]]:
        """Per world: ``{"tunnel": (body, shape) | None, "fast": (travel_m, body, shape) | None, "gap_m": g}`` names
        from the sticky keys (host read; call outside a step)."""
        names_b, names_s = self.model.body_label, self.model.shape_label
        by_body = {w: [b for b in self.bodies if b.world == w] for w in range(self.world_count)}
        fast, tunnel = self.tr.seen_fast.numpy(), self.tr.seen_tunnel.numpy()

        def who(w: int, key: int) -> tuple[str, str]:
            jl, sl = local_index((key >> 16) & 0xFFFF, len(by_body[w])), local_index(key & 0xFFFF, len(self._by_world[w]))
            body = by_body[w][jl].body if jl is not None else -1
            shape = int(self.statics[self._by_world[w][sl]]) if sl is not None else -1
            return (_leaf(names_b[body]) if body >= 0 else "a body", _leaf(names_s[shape]) if shape >= 0 else "a shape")
        out = []
        for w in range(self.world_count):
            row: dict[str, Any] = {"tunnel": None, "fast": None, "dt": dt}
            if int(tunnel[w]):
                row["tunnel"] = who(w, int(tunnel[w]))
            if int(fast[w]):
                travel = float(np.array([int(fast[w]) >> 32], np.uint32).view(np.float32)[0])
                row["fast"] = (travel, *who(w, int(fast[w])))
            out.append(row)
        return out


_ITEM_KEYS = ("tunnel.in_shape", "tunnel.in_normal", "tunnel.in_depth", "tunnel.in_thru")
KEY_LIMIT = 0xFFFF      # the key packs local indices in 16 bits; the kernel clamps larger ones to this value


def local_index(code: int, count: int) -> int | None:
    """The local body / static-shape index a key code names, None when it cannot be told: the clamp sentinel (a world
    with more than 65,535 of them) or an index past ``count``."""
    return code if code < min(count, KEY_LIMIT) else None


def _leaf(label: str) -> str:
    return str(label).rsplit("/", 1)[-1] if label else "a body"


def _wrap_step(solver: Any, tw: TunnelWatch) -> None:
    inner = solver.step

    def step(state_in, state_out, control, contacts, dt, *args, **kwargs):
        inner(state_in, state_out, control, contacts, dt, *args, **kwargs)
        tw.after_step(state_in, state_out, dt)

    solver.step = step


def watch(solver: Any) -> TunnelWatch | None:
    """The solver's watch, attached on first call (idempotent, on the instance only); None when the model has no
    judged body or no judged static shape."""
    existing = getattr(solver, WATCH_ATTR, None)
    if existing is not None:
        return existing
    model = getattr(solver, "model", None)
    if model is None or getattr(model, "shape_count", 0) == 0:
        return None
    bodies, statics = S.judged_bodies(model), S.static_shapes(model)
    if not bodies or not len(statics):
        return None
    tw = TunnelWatch(model, bodies, statics)
    setattr(solver, WATCH_ATTR, tw)
    _wrap_step(solver, tw)
    log.info("tunnel check: %d bodies against %d static shapes", len(bodies), len(statics))
    return tw


def _compose(base: Any, extra: Any) -> Any:
    def composed(solver: Any) -> dict[str, wp.array]:
        out = dict(base(solver)) if base is not None else {}
        out.update(extra(solver))
        return out
    composed._tunnel_composed = True              # type: ignore[attr-defined]
    return composed


def _own_arrays(solver: Any) -> dict[str, wp.array]:
    tw = getattr(solver, WATCH_ATTR, None)
    return tw.state_arrays() if tw is not None else {}


def _own_maps(solver: Any) -> dict[str, wp.array]:
    tw = getattr(solver, WATCH_ATTR, None)
    return tw.state_world_maps() if tw is not None else {}


def install_state_adapter() -> None:
    """Add the watch's tracker arrays to the MuJoCo solver's state adapter, keeping the registered one (composed: its
    arrays and maps first). Idempotent per registry entry; ``register_builtin_adapters`` replaces the entry at every
    build, so ``sources`` calls this again at each health monitor's construction, before the snapshots are made."""
    from alloy.core import registry  # noqa: PLC0415

    cls = newton.solvers.SolverMuJoCo
    reg = registry._REG
    base = reg.state.get(cls)
    if getattr(base, "_tunnel_composed", False):
        return
    maps = reg.world_maps.get(cls)
    registry.register_state_adapter(cls, _compose(base, _own_arrays), world_map=_compose(maps, _own_maps))


def sources(solver: Any) -> list[Any]:
    """The two capacity sources (FAST_CONTACT, TUNNEL) of the solver's watch; [] when nothing is judged."""
    from alloy.precision.health_adapters import CapacitySource  # noqa: PLC0415

    tw = watch(solver)
    if tw is None:
        return []
    install_state_adapter()
    return [CapacitySource(tw.fast, "nonzero", int(HealthFlags.FAST_CONTACT), per_world=True, label="fast contact",
                           reset=tw.reset),
            CapacitySource(tw.tunnel, "nonzero", int(HealthFlags.TUNNEL), per_world=True, label="tunnel",
                           reset=tw.reset)]


def watch_of(engine: Any) -> TunnelWatch | None:
    """The tunnel watch behind an engine's health monitor (None: none)."""
    for src in getattr(getattr(engine, "health", None), "_sources", ()):
        owner = getattr(getattr(src, "reset", None), "__self__", None)
        if isinstance(owner, TunnelWatch):
            return owner
    return None


def tunnel_texts(report: Any, engine: Any) -> Any:
    """``report`` with ``tunnel_text`` / ``fast_text`` per world (names, mm and ms), or unchanged without a watch or a
    flag. Host read of the watch's keys: call outside a step (``alloy.api.health.poll``)."""
    import dataclasses  # noqa: PLC0415

    tw = watch_of(engine)
    if tw is None or report is None or not np.any(np.asarray(report.flags) & _BITS):
        return report
    rows = tw.notes(tw.dt)
    flags = np.asarray(report.flags)
    on_t, on_f = (flags & int(HealthFlags.TUNNEL)) != 0, (flags & int(HealthFlags.FAST_CONTACT)) != 0
    return dataclasses.replace(report, tunnel_text=tuple(_tunnel(r) if on_t[w] else "" for w, r in enumerate(rows)),
                               fast_text=tuple(_fast(r, tw) if on_f[w] else "" for w, r in enumerate(rows)))


_BITS = int(HealthFlags.TUNNEL | HealthFlags.FAST_CONTACT)


def _tunnel(row: dict[str, Any]) -> str:
    return f'"{row["tunnel"][0]}" passed through "{row["tunnel"][1]}"' if row["tunnel"] else ""


def _fast(row: dict[str, Any], tw: TunnelWatch) -> str:
    if not row["fast"]:
        return ""
    travel, body, shape = row["fast"]
    return (f'"{body}" moved {travel * 1e3:.1f} mm toward "{shape}" in one {row["dt"] * 1e3:.3g} ms rigid step, more '
            f"than the {K.TRAVEL_LIMIT_M * 1e3:.3g} mm travel limit")
