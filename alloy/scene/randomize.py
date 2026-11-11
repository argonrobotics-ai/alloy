# SPDX-License-Identifier: Apache-2.0
"""Per-world randomization (SPEC §3.4 `apply_randomization`): values only, never topology or routes.

Draw u = sha256(seed | global world id | field digest)[:8] / 2^64, value = lo + u·(hi − lo): counter-based, the same
on every device and process (D9). Supported `Randomization.field` paths (object-card relative, SI):
- `material.density`: body mass/inertia and particle mass scaled by value/nominal;
- `material.surface.{mu_static,mu_dynamic,restitution,contact_stiffness}`: per-world pair-table rows of the object's
  own material key (randomized objects get one, so partners sharing the card are untouched), and for friction each
  drawn world's MuJoCo pair rows and geom friction (`alloy.scene.randomize_mujoco`: MuJoCo's native contacts do not
  read the pair table). Contacts use one friction coefficient per pair (μ_d), so a `mu_static` draw sets μ_d to the
  same value (`Scene.randomize(name, "friction", ...)` draws it) and a `mu_dynamic` draw keeps μ_static >= μ_d. Several
  surface draws of one object combine in each world;
- `material.elastic.youngs`: cloth tri/edge and tet stiffness (and their β·k damping), CHAIN/PANELS passive
  stiffness, scaled by value/nominal;
- `pose.0`, `pose.1`, `pose.2` (SCR-WP1-1 grammar; x/y/z aliases): the root free joint's coordinates.
The drawn values are kept on `built.randomized[(object, field)]` ([W] float64) for readback.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from alloy.cards.types import MaterialCard, Randomization, SceneCard
from alloy.core.errors import AssemblyError
from alloy.core.ids import digest_of
from alloy.scene import pairs as pt
from alloy.scene.types import BuiltScene, ObjectHandle

if TYPE_CHECKING:
    from alloy.scene.assembler import Plan

log = logging.getLogger("alloy.scene")
SURFACE = ("mu_static", "mu_dynamic", "restitution", "contact_stiffness")
ROUTING_ONLY = ("loads", "features", "budget")   # card fields consumed by routing/assembly at their worst case
POSE_AXES = {"pose.0": 0, "pose.1": 1, "pose.2": 2, "pose.x": 0, "pose.y": 1, "pose.z": 2}  # SCR-WP1-1


def has_randomization(scene: SceneCard) -> bool:
    return any(p.randomize for p in scene.objects)


def draw(seed: int, world: int, field: str, lo: float, hi: float) -> float:
    """Counter-based uniform draw in [lo, hi) keyed on (seed, global world id, field digest)."""
    key = f"{int(seed)}|{int(world)}|{digest_of(field)}".encode()
    u = int.from_bytes(hashlib.sha256(key).digest()[:8], "little") / 2.0**64
    return lo + u * (hi - lo)


def _gids(rng: range, stride: int, worlds: Sequence[int]) -> np.ndarray:
    idx = np.asarray(list(rng), np.int64)
    return (np.asarray(worlds, np.int64)[:, None] * stride + idx[None, :]) if idx.size else np.zeros((len(worlds), 0),
                                                                                                    np.int64)


def _scale_rows(arr: wp.array, rows: np.ndarray, scale: np.ndarray, cols: Sequence[int] | None = None) -> None:
    """Multiply rows[w] of `arr` (whole elements, or only columns `cols` of a 2-D array) by scale[w]."""
    host = arr.numpy()
    for w in range(rows.shape[0]):
        if cols is None:
            host[rows[w]] = host[rows[w]] * scale[w]
        else:
            idx = np.ix_(rows[w], np.asarray(cols))
            host[idx] = host[idx] * scale[w]
    arr.assign(host)


def _density(built: BuiltScene, h: ObjectHandle, worlds: Sequence[int], ratio: np.ndarray) -> None:
    m = built.model
    st = h.world_stride
    if h.template.body:
        rows = _gids(h.template.body, len(st.body), worlds)
        _scale_rows(m.body_mass, rows, ratio)
        _scale_rows(m.body_inertia, rows, ratio)
        _scale_rows(m.body_inv_mass, rows, 1.0 / ratio)
        _scale_rows(m.body_inv_inertia, rows, 1.0 / ratio)
    if h.template.particle:
        rows = _gids(h.template.particle, len(st.particle), worlds)
        _scale_rows(m.particle_mass, rows, ratio)
        _scale_rows(m.particle_inv_mass, rows, 1.0 / ratio)


def _youngs(built: BuiltScene, h: ObjectHandle, worlds: Sequence[int], ratio: np.ndarray) -> None:
    m, st = built.model, h.world_stride
    k = getattr(getattr(m, "alloy", None), "tri_ortho_k", None)       # a woven card's orthotropic membrane
    if h.template.tri and k is not None and np.any(k.numpy()[list(h.template.tri), :3] != 0.0):
        raise AssemblyError(f"{h.name}: youngs randomization of a woven card on CLOTH_VBD is not supported: its warp, "
                            "weft and shear stiffness (alloy.scene.ortho_membrane) would stay while the bending moved; "
                            "randomize density, or draw the card's elastic.cloth values per scene")
    if h.template.tri:
        _scale_rows(m.tri_materials, _gids(h.template.tri, len(st.tri), worlds), ratio, [0, 1, 2])   # mu, lambda, kd
    if h.template.edge:
        _scale_rows(m.edge_bending_properties, _gids(h.template.edge, len(st.edge), worlds), ratio, [0, 1])
    if h.template.tet and m.tet_materials is not None:
        _scale_rows(m.tet_materials, _gids(h.template.tet, len(st.tet), worlds), ratio, [0, 1, 2])
    if h.template.joint and getattr(m, "joint_stiffness", None) is not None:   # Newton 1.8: the core passive springs
        qd = m.joint_qd_start.numpy()
        jt = _gids(h.template.joint, len(st.joint), worlds)
        rows = np.asarray([[d for j in row for d in range(qd[j], qd[j + 1])] for row in jt], np.int64)
        _scale_rows(m.joint_stiffness, rows, ratio)


def _pose(built: BuiltScene, h: ObjectHandle, worlds: Sequence[int], axis: int, values: np.ndarray) -> None:
    m = built.model
    if not h.template.joint:
        raise AssemblyError(f"{h.name}: pose randomization needs a free root joint")
    jq = m.joint_q.numpy()
    qs = m.joint_q_start.numpy()
    for w, v in zip(worlds, values):
        j = h.template.joint.start + w * len(h.world_stride.joint)
        jq[qs[j] + axis] = v
    m.joint_q.assign(jq)


def _surface(built: BuiltScene, plan: Plan, name: str, leaf: str, worlds: Sequence[int], values: np.ndarray) -> None:
    """Rebuild the object's material row for every drawn world into the per-world pair tables, then MuJoCo's values
    (``randomize_mujoco``)."""
    from alloy.scene.randomize_mujoco import write_mujoco_surface

    table = built.pair_table
    t = next(o for o in plan.objects if o.name == name)
    i = table.material_index(t.material_key)
    ensure_world_tables(table, built.world_count)
    wt, ws = table.world_table.numpy(), table.world_stiffness.numpy()
    entries = _entries(plan)
    mu = np.empty(len(worlds))
    for k, (w, v) in enumerate(zip(worlds, values)):
        card = drawn_surface(entries[i].card, surface_draws(built, name, w) | {leaf: float(v)})
        mu[k] = float(card.surface.mu_dynamic.value)
        mine = pt.MaterialEntry(entries[i].key, card, entries[i].k_override)
        for j, other in enumerate(entries):
            other = mine if j == i else other
            pv = pt.pair_values(mine, other, plan_pair(plan, mine.card.name, other.card.name), plan.scene.combine)
            wt[w, i, j] = wt[w, j, i] = (pv.mu_s, pv.mu_d, pv.e, pv.lever)
            ws[w, i, j] = ws[w, j, i] = pv.k
    table.world_table.assign(wt)
    table.world_stiffness.assign(ws)
    write_mujoco_surface(built, plan, name, leaf, worlds, mu)


def surface_draws(built: BuiltScene, name: str, world: int) -> dict[str, float]:
    """{surface leaf: value} of the draws of object ``name`` stored for ``world``."""
    out = {}
    for leaf in SURFACE:
        prev = getattr(built, "randomized", {}).get((name, f"material.surface.{leaf}"))
        if prev is not None and np.isfinite(prev[world]):
            out[leaf] = float(prev[world])
    return out


def drawn_surface(card: MaterialCard, draws: dict[str, float]) -> MaterialCard:
    """``card`` with one world's surface draws, independent of draw order: a μ_static draw is the pair's friction
    (μ_d follows) unless μ_d is drawn too; a μ_d draw raises μ_static to it where needed."""
    for leaf, value in draws.items():
        card = _with_surface(card, leaf, value)
    if "mu_static" in draws and "mu_dynamic" not in draws:
        card = _with_surface(card, "mu_dynamic", draws["mu_static"])
    if "mu_dynamic" in draws and float(card.surface.mu_static.value) < draws["mu_dynamic"]:
        card = _with_surface(card, "mu_static", draws["mu_dynamic"])
    return card


def ensure_world_tables(table, worlds: int) -> None:
    if getattr(table, "world_table", None) is None:
        base, k = table.table.numpy(), table.stiffness.numpy()
        dev = table.table.device
        table.world_table = wp.array(np.broadcast_to(base, (worlds, *base.shape)).copy(), dtype=wp.vec4, device=dev)
        table.world_stiffness = wp.array(np.broadcast_to(k, (worlds, *k.shape)).copy(), dtype=wp.float32, device=dev)


def _with_surface(card: MaterialCard, leaf: str, value: float) -> MaterialCard:
    s = card.surface
    old = getattr(s, leaf)
    from alloy.core.quantity import Q

    q = old.with_value(value) if old is not None else Q(value, "model", note="randomized")
    return replace(card, surface=replace(s, **{leaf: q}))


def _entries(plan: Plan) -> list[pt.MaterialEntry]:
    from alloy.scene.finish import material_entries

    return material_entries(plan, plan._template)


def plan_pair(plan: Plan, a: str, b: str):
    return plan._assembler.pair_card(a, b, plan.scene)


def _nominal(plan: Plan, name: str, field: str) -> float:
    t = next(o for o in plan.objects if o.name == name)
    if field in POSE_AXES:
        return float(t.placement.pose[POSE_AXES[field]])
    node = t.obj.card if not field.startswith("material") else t.obj
    for part in field.split("."):
        node = getattr(node, part)
    return float(getattr(node, "value", node))


def apply_randomization(built: BuiltScene, seed: int, world_ids: wp.array | None = None, *,
                        plan: Plan | None = None) -> None:
    """Draw and write every placement's randomized fields for the given local worlds (all if None)."""
    plan = plan if plan is not None else getattr(built, "_plan", None)
    if plan is None:
        raise AssemblyError("apply_randomization needs the assembly plan (built by SceneAssembler)")
    built._plan = plan
    worlds = list(range(built.world_count)) if world_ids is None else [int(w) for w in world_ids.numpy()]
    if not hasattr(built, "randomized"):
        built.randomized = {}
    names = [t.name for t in plan.objects if t.obj is not None]
    for name, placement in zip(names, plan.scene.objects):
        for r in placement.randomize:
            vals = np.asarray([draw(seed, built.global_world_id(w), f"{name}.{r.field}", r.lo, r.hi)
                               for w in worlds])
            _write(built, plan, name, r, worlds, vals)
            store = built.randomized.setdefault((name, r.field), np.full(built.world_count, np.nan))
            store[worlds] = vals
    if world_ids is None:
        built.randomized_seed = int(seed)          # the seed every world was drawn at (`draw_seed`, reset)
    log.info("randomized %d fields over %d worlds (seed %d)", len(built.randomized), len(worlds), seed)


def draw_seed(built: BuiltScene) -> int:
    """The seed the scene's worlds were last drawn at as a whole: `built.randomized_seed`, else 0 (finish_scene's
    assembly draw). `Simulation.reset` redraws at it, so a reset world gets back its assembly values (SCR-DC3-8)."""
    seed = getattr(built, "randomized_seed", None)
    return 0 if seed is None else int(seed)


def _base(built: BuiltScene, plan: Plan, name: str, field: str, worlds: list[int]) -> np.ndarray:
    """What the Model holds now, per world: the previous draw where there is one, else the card's nominal value.
    Density and Young's modulus scale the current Model values by new/base, so a redraw does not compound
    (SCR-DC3-8, WP12: a second draw multiplied box mass by 1.11-1.14)."""
    nominal = _nominal(plan, name, field)
    prev = getattr(built, "randomized", {}).get((name, field))
    if prev is None:
        return np.full(len(worlds), nominal)
    p = np.asarray(prev, np.float64)[worlds]
    return np.where(np.isfinite(p), p, nominal)


def _write(built: BuiltScene, plan: Plan, name: str, r: Randomization, worlds: list[int], vals: np.ndarray) -> None:
    h = built.handles[name]
    writers: dict[str, Callable[[], None]] = {
        "material.density": lambda: _density(built, h, worlds, vals / _base(built, plan, name, r.field, worlds)),
        "material.elastic.youngs": lambda: _youngs(built, h, worlds, vals / _base(built, plan, name, r.field, worlds)),
    }
    for leaf in SURFACE:
        writers[f"material.surface.{leaf}"] = lambda leaf=leaf: _surface(built, plan, name, leaf, worlds, vals)
    for path, axis in POSE_AXES.items():
        writers[path] = lambda axis=axis: _pose(built, h, worlds, axis, vals)
    fn = writers.get(r.field)
    if fn is None and r.field.split(".")[0] in ROUTING_ONLY:
        note = (f"{name}.{r.field}", "randomized routing/sizing input: the worst end of its range is used at "
                                     "assembly (routes and dt are shared by every world), no per-world Model value")
        if note not in built.report.unsupported:
            built.report.unsupported.append(note)
        return
    if fn is None:
        raise AssemblyError(f"{name}: randomization field {r.field!r} is not supported (values only: "
                            f"{sorted(writers)})")
    fn()
