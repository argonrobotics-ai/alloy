# SPDX-License-Identifier: Apache-2.0
"""Explicit MuJoCo contact pairs for native contacts (D13): `mujoco:pair_*` rows where the pair table differs from
MuJoCo's own rule (μ = max of the two geoms' friction, newton@ac7f7830:newton/_src/solvers/mujoco/kernels.py:123-170).

Rows are added to the world template (pair_world and pair_geom* are world/shape references, so `replicate` maps them
per copy; solver_mujoco.py:2080-2180). Each row carries the pair's μ_d (torsional = the table lever, rolling = the
larger shape value) and its own solref from the series K and the reduced mass (native_gains, ζ from the pair's e), so
listed pairs get exact pair stiffness in native mode too. Shapes on static/kinematic bodies get geom_solmix ≈ 0 so
the dynamic geom's solref governs unlisted contacts (MuJoCo averages solref by solmix weight).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import NamedTuple

import numpy as np

from alloy._vendor import newton

from alloy.scene import pairs as pt
from alloy.scene.native_gains import DEFAULT_IMP, contact_zeta, native_gains, stable_ke
from alloy.scene.template import TemplateObject

log = logging.getLogger("alloy.scene")
MU_RTOL = 1e-6
ROWS_WARN = 2000
STATIC_SOLMIX = 1e-6
MJ_MINVAL = 1e-15
DEFAULT_SOLIMP = (0.9, DEFAULT_IMP, 0.001, 0.5, 2.0)   # Newton's mujoco:geom_solimp default
SOLIMP_ATOL = 1e-6                                     # float32 storage of the geom values
FIT_ATTRS = ("pair_k", "pair_zeta", "pair_kecap")       # read by alloy.contact.pair_fit (SCR-DC3-5 part 2)
DRAWN_ATTRS = ("pair_m0", "pair_tcmin")                # drawn rows: template reduced mass, 2·dt (SCR-DC3-8)
_COUNT_ONLY: ContextVar[list[int] | None] = ContextVar("alloy_pair_rows_count_only", default=None)


@contextmanager
def count_only() -> Iterator[list[int]]:
    """Inside the block ``pair_rows`` adds no row and appends the number it would add (per world) to the yielded
    list instead: ``alloy.estimate``'s assembly, which reads counts the rows do not change (16 bowls of
    27 hulls make 88,000 rows, most of an estimate's time). Example::

        with count_only() as rows:
            built = assembler.assemble(card, device, world_count=1)
        print(sum(rows))
    """
    sink: list[int] = []
    token = _COUNT_ONLY.set(sink)
    try:
        yield sink
    finally:
        _COUNT_ONLY.reset(token)




def _shape_bodies(b: newton.ModelBuilder, objs: Sequence[TemplateObject]) -> list[tuple[int, int, int]]:
    """(shape, body, object index) of every collidable template shape."""
    out = []
    for oi, t in enumerate(objs):
        for s in t.ranges.shape:
            if b.shape_flags[s] & int(newton.ShapeFlags.COLLIDE_SHAPES):
                out.append((s, b.shape_body[s], oi))
    return out


def _mass(b: newton.ModelBuilder, body: int) -> float:
    return float(b.body_mass[body]) if body >= 0 else 0.0


def pair_rows(b: newton.ModelBuilder, objs: Sequence[TemplateObject], keys: Sequence[str], table: np.ndarray,
              stiffness: np.ndarray, dt: float | None = None, integrator: str = "implicitfast") -> int:
    """Add rows for shape pairs on different bodies (at least one dynamic) whose μ_pair ≠ max(μ_a, μ_b), or where an
    object draws a surface value per world."""
    if "mujoco:pair_geom1" not in b.custom_attributes:
        return 0
    shapes = _shape_bodies(b, objs)
    index: dict[str, int] = {}                     # a key's first index, as keys.index, in O(1) per row
    for i, k in enumerate(keys):
        index.setdefault(k, i)
    pairs = _candidates(b, shapes, objs, index, table)
    sink = _COUNT_ONLY.get()
    if sink is not None:                                  # count_only(): the estimate's assembly
        sink.append(len(pairs))
        return len(pairs)
    fit, memo = FitBodies(armature_bodies(b, objs), drawn_bodies(objs)), {}
    rows = 0
    for i, j in pairs:
        rows += _maybe_row(b, shapes[i], shapes[j], objs, index, table, stiffness, (dt, integrator), fit, memo)
    if rows > ROWS_WARN:
        log.warning("%d explicit MuJoCo pairs per world (GOTCHAS:503: millions of keyed pairs slow the build)", rows)
    log.info("native MuJoCo pair rows: %d per world", rows)
    return rows


def _candidates(b: newton.ModelBuilder, shapes: Sequence[tuple[int, int, int]], objs: Sequence[TemplateObject],
                index: dict[str, int], table: np.ndarray) -> list[tuple[int, int]]:
    """(i, j) indices into ``shapes``, i < j in the double loop's order, of the pairs that get a row: ``_maybe_row``'s
    tests (different bodies, one dynamic, not ``_excluded``, μ_pair ≠ max(μ_a, μ_b) or a CCD / drawn object) in numpy,
    one shape against all later ones (the rows are the same; the per-pair Python test made the build O(shapes²) in
    Python: 16 decomposed bowls, 433 shapes, 17 s of a CPU build)."""
    from alloy.scene.entries import ccd_backed

    n = len(shapes)
    if n < 2:
        return []
    shape, body, obj = (np.array([x[k] for x in shapes], dtype=np.int64) for k in range(3))
    mass = np.array([_mass(b, int(x)) for x in body], dtype=np.float64)
    live = np.array([objs[o].obj is not None for o in obj], dtype=bool)
    mat = np.array([index[objs[o].material_key] if live[k] else 0 for k, o in enumerate(obj)], dtype=np.int64)
    mu_s = np.array([float(b.shape_material_mu[s]) for s in shape], dtype=np.float64)
    group = np.array([int(b.shape_collision_group[s]) for s in shape], dtype=np.int64)
    always = np.array([live[k] and (ccd_backed(objs[o]) or _draws_surface(objs[o])) for k, o in enumerate(obj)],
                      dtype=bool)
    mu_table = np.asarray(table[:, :, 1], dtype=np.float64)
    filtered = b._shape_collision_filter_pairs
    out = []
    for i in range(n - 1):
        if not live[i]:
            continue
        j = np.arange(i + 1, n)
        ok = live[j] & (body[j] != body[i]) & (obj[j] != obj[i]) & ((mass[j] > 0.0) | (mass[i] > 0.0))
        ok &= _groups_collide_np(group[i], group[j])
        mu_max = np.maximum(mu_s[i], mu_s[j])
        differs = np.abs(mu_table[mat[i], mat[j]] - mu_max) > MU_RTOL * np.maximum(np.abs(mu_max), 1e-12)
        ok &= differs | always[j] | always[i]
        s0 = int(shape[i])
        out.extend((i, int(k)) for k in j[ok] if (min(s0, int(shape[k])), max(s0, int(shape[k]))) not in filtered)
    return out


def _groups_collide_np(g0: int, g1: np.ndarray) -> np.ndarray:
    """``groups_collide(g0, x)`` for each x in ``g1``."""
    if g0 == 0:
        return np.zeros(g1.shape, dtype=bool)
    if g0 > 0:
        return (g1 != 0) & ((g1 == g0) | (g1 < 0))
    return (g1 != 0) & (g1 != g0)


def _excluded(b: newton.ModelBuilder, sa, sb, objs) -> bool:
    """Pairs that must never get a row: same object (robot internals, chain links: the file/self rules apply),
    robot shapes (MJCF friction stays authoritative), builder-filtered pairs (parent/child, <exclude>) and
    collision-group mismatches. An explicit MuJoCo pair bypasses contype/conaffinity/exclude, so a row on a
    filtered pair would create a contact the model disabled (T09: 7345 rows, link-vs-own-finger overlaps)."""
    (s0, _, o0), (s1, _, o1) = sa, sb
    if o0 == o1 or objs[o0].obj is None or objs[o1].obj is None:
        return True
    if (min(s0, s1), max(s0, s1)) in b._shape_collision_filter_pairs:
        return True
    return not groups_collide(b.shape_collision_group[s0], b.shape_collision_group[s1])


def groups_collide(g0: int, g1: int) -> bool:
    """Newton's collision-group rule (``geometry.broad_phase_common.test_group_pair``): 0 collides with nothing; a
    positive group with itself and any negative one; a negative group with everything but itself. Two shapes in one
    negative group never collide, so they get no row either (WP9, 2026-11-09: the old test gave them one)."""
    g0, g1 = int(g0), int(g1)
    if g0 == 0 or g1 == 0:
        return False
    if g0 > 0:
        return g0 == g1 or g1 < 0
    return g0 != g1


def _maybe_row(b, sa, sb, objs, index, table, stiffness, step=(None, "implicitfast"), fit=None, memo=None) -> int:
    """A row where μ_pair ≠ max(μ_a, μ_b), or where either object collides through MuJoCo's CCD path (so the
    resting ζ floor reaches a box resting on a static mesh, whose own solref MuJoCo would otherwise use).

    SCR-DC3-5 part 2: a flat geom solimp (d0 = dmax, D22's touched-face setting) reaches the row at ζ = 1
    (`_row_solimp`). A row touching a body in `fit` (joint armature, SCR-DC3-5; or a per-world drawn mass,
    SCR-DC3-8) records K, ζ and the ke cap so `alloy.contact.pair_fit` resizes it from MuJoCo's per-world
    body_invweight0; a drawn row also records its template reduced mass (`_row_fit_values`)."""
    from alloy.scene.native_gains import ZETA_CCD

    (s0, b0, o0), (s1, b1, o1) = sa, sb
    m0, m1 = _mass(b, b0), _mass(b, b1)
    if b0 == b1 or (m0 <= 0.0 and m1 <= 0.0) or _excluded(b, sa, sb, objs):
        return 0
    memo = {} if memo is None else memo                  # per-object values over one pair_rows call
    (i, ccd0, drawn0, floor0), (j, ccd1, drawn1, floor1) = (_obj_facts(objs, index, o, memo) for o in (o0, o1))
    mu = float(table[i, j, 1])
    mu_max = max(float(b.shape_material_mu[s0]), float(b.shape_material_mu[s1]))
    ccd = ccd0 or ccd1
    drawn = drawn0 or drawn1                                            # per-world values go into the row
    if abs(mu - mu_max) <= MU_RTOL * max(abs(mu_max), 1e-12) and not ccd and not drawn:
        return 0
    m_eff = 1.0 / sum(1.0 / m for m in (m0, m1) if m > 0.0)
    solimp, imp, flat = _row_solimp(b, s0, s1)
    # resting floors: the CCD one from either side, as before (a box resting on a static mesh); the multi-point one
    # from the dynamic sides only (a sphere bouncing on a static box table keeps its e)
    floors = [f for f, m in ((floor0, m0), (floor1, m1)) if m > 0.0]
    zeta_min = 1.0 if flat else max([ZETA_CCD if ccd else 0.0, *floors])
    K, e = float(stiffness[i, j]), float(table[i, j, 2])
    ke, kd = native_gains(K, m_eff, e, imp, dt=step[0], zeta_min=zeta_min, integrator=step[1])
    rolling = max(float(b.shape_material_mu_rolling[s0]), float(b.shape_material_mu_rolling[s1]))
    solref = (2.0 / kd, 0.5 * kd * np.sqrt(1.0 / ke))
    values = {
        "mujoco:pair_world": b.current_world, "mujoco:pair_geom1": s0, "mujoco:pair_geom2": s1,
        "mujoco:pair_condim": 3, "mujoco:pair_solref": solref,
        # friction rows stay soft (0.02 s, ζ 1): on an L4 a box peg lying on plywood walked 0.2-1.8 mm per 0.25 s with
        # friction rows at the normal row's solref (ζ(e) or ζ 1), 0.027 mm with these (CARD-1)
        "mujoco:pair_solreffriction": (0.02, 1.0), "mujoco:pair_solimp": solimp,
        "mujoco:pair_margin": max(float(b.shape_margin[s0]), float(b.shape_margin[s1])),
        "mujoco:pair_gap": max(float(b.shape_gap[s0] or 0.0), float(b.shape_gap[s1] or 0.0)),
        "mujoco:pair_friction": (mu, mu, float(table[i, j, 3]), rolling, rolling),
    }
    values.update(_row_fit_values(b, (b0, b1), fit or FitBodies(), m_eff, (K, contact_zeta(e, zeta_min), imp, step)))
    b.add_custom_values(**values)
    return 1


def _obj_facts(objs: Sequence[TemplateObject], index: dict[str, int], o: int,
               memo: dict) -> tuple[int, bool, bool, float]:
    """(material index, CCD-backed, draws a surface value, resting damping floor) of object ``o``, kept in ``memo``."""
    from alloy.scene.entries import ccd_backed
    from alloy.scene.native_gains import resting_floor

    if ("obj", o) not in memo:
        t = objs[o]
        memo[("obj", o)] = (index[t.material_key], ccd_backed(t), _draws_surface(t), resting_floor(t))
    return memo[("obj", o)]


def _attr(b: newton.ModelBuilder, key: str, s: int, fallback):
    """A per-shape custom attribute value (its default when unset, `fallback` when unregistered)."""
    a = b.custom_attributes.get(key)
    if a is None:
        return fallback
    v = a.values.get(s) if isinstance(a.values, dict) else None
    return a.default if v is None else v


def _row_solimp(b: newton.ModelBuilder, s0: int, s1: int) -> tuple[tuple[float, ...], float, bool]:
    """(solimp, imp for the gains, flat) of the row. The solimp MuJoCo would give the geom pair (mj_contactParam: the
    higher geom priority wins, else the solmix-weighted mean) is honoured when it is flat (d0 = dmax: D22's
    touched-face option, gains at that imp). Any other geom solimp keeps the default row: DC4's rack PSU faces
    (0.99, 0.999, 0.1 mm) are tuned on it, and honouring them moved its latch either way (gains at dmax: creep
    1.85 mm per 10 s, bound 0.05; at d0: unmating 9.6 N, floor 12 N). Default geoms keep DEFAULT_SOLIMP exactly."""
    a0, c0 = _attr(b, "mujoco:geom_solimp", s0, DEFAULT_SOLIMP), _attr(b, "mujoco:geom_solimp", s1, DEFAULT_SOLIMP)
    if _plain_solimp(a0) and _plain_solimp(c0):
        return DEFAULT_SOLIMP, DEFAULT_IMP, False
    a, c = np.asarray(a0, dtype=np.float64), np.asarray(c0, dtype=np.float64)
    p0, p1 = int(_attr(b, "mujoco:geom_priority", s0, 0)), int(_attr(b, "mujoco:geom_priority", s1, 0))
    w = 1.0 if p0 > p1 else 0.0 if p1 > p0 else _solmix_weight(b, s0, s1)
    mixed = tuple(float(x) for x in w * a + (1.0 - w) * c)
    if abs(mixed[0] - mixed[1]) > SOLIMP_ATOL:
        return DEFAULT_SOLIMP, DEFAULT_IMP, False
    return mixed, mixed[1], True


def _plain_solimp(v) -> bool:
    """Whether a geom solimp is the default (np.allclose(v, DEFAULT_SOLIMP, rtol=0, atol=SOLIMP_ATOL) without numpy's
    per-call cost: it runs once per shape pair)."""
    return len(v) == len(DEFAULT_SOLIMP) and all(abs(float(x) - y) <= SOLIMP_ATOL for x, y in zip(v, DEFAULT_SOLIMP))


def _solmix_weight(b: newton.ModelBuilder, s0: int, s1: int) -> float:
    """MuJoCo's solmix weight of geom 1 (engine_collision_driver.c mj_contactParam)."""
    m0, m1 = float(_attr(b, "mujoco:geom_solmix", s0, 1.0)), float(_attr(b, "mujoco:geom_solmix", s1, 1.0))
    if m0 < MJ_MINVAL and m1 < MJ_MINVAL:
        return 0.5
    if m0 < MJ_MINVAL or m1 < MJ_MINVAL:
        return 0.0 if m0 < MJ_MINVAL else 1.0
    return m0 / (m0 + m1)


def armature_bodies(b: newton.ModelBuilder, objs: Sequence[TemplateObject] = ()) -> frozenset[int]:
    """Bodies whose own joint carries armature on any dof (CON-45 regularised latch members). With `objs`, only
    bodies of non-robot, non-CHAIN objects count: the CHAIN route's armature floor (1e-6 kg·m² on every rope dof,
    emitters.chain) is a numerical floor, and resizing rope contacts would change T09's cable (not in SCR-DC3-5)."""
    from alloy.route.types import Route

    allowed = None
    if objs:
        allowed = {int(x) for t in objs if t.obj is not None and t.decision.route != Route.CHAIN for x in t.ranges.body}
    out = set()
    for j, child in enumerate(b.joint_child):
        if allowed is not None and int(child) not in allowed:
            continue
        start = b.joint_qd_start[j]
        if any(float(b.joint_armature[k]) > 0.0 for k in range(start, start + sum(b.joint_dof_dim[j]))):
            out.add(int(child))
    return frozenset(out)


class FitBodies(NamedTuple):
    """Bodies whose rows `alloy.contact.pair_fit` resizes after the entry is built."""

    armature: frozenset[int] = frozenset()   # joint armature (SCR-DC3-5 part 2)
    drawn: frozenset[int] = frozenset()      # per-world density draw (SCR-DC3-8)


def _row_fit_values(b: newton.ModelBuilder, bodies: tuple[int, int], fit: FitBodies, m_eff: float, gains) -> dict:
    """`alloy:pair_*` values of one row: K, ζ, cap (`gains` = (K, ζ, imp, (dt, integrator))) when it touches a fitted
    body, plus its template reduced mass `m_eff` and MuJoCo's timeconst clamp 2·dt when it touches a drawn one; zeros
    keep registered lists as long as the rows."""
    drawn = any(x in fit.drawn for x in bodies)
    out = {}
    if drawn or any(x in fit.armature for x in bodies):
        out.update(_fit_values(b, *gains))
    elif f"alloy:{FIT_ATTRS[0]}" in b.custom_attributes:
        out.update({f"alloy:{name}": 0.0 for name in FIT_ATTRS})
    if drawn:
        dt = gains[3][0]
        for name, v in zip(DRAWN_ATTRS, (m_eff, 2.0 * dt if dt else 0.0), strict=True):
            _register(b, name)
            out[f"alloy:{name}"] = v
    elif f"alloy:{DRAWN_ATTRS[0]}" in b.custom_attributes:
        out.update({f"alloy:{name}": 0.0 for name in DRAWN_ATTRS})
    return out


def drawn_bodies(objs: Sequence[TemplateObject]) -> frozenset[int]:
    """Bodies of objects whose placement draws `material.density` per world (SCR-DC3-8): their rows are sized from
    the drawn masses after the entry is built (alloy.contact.pair_fit), not the template mass."""
    return frozenset(int(x) for t in objs if t.obj is not None and _draws_mass(t) for x in t.ranges.body)


def _draws_mass(t: TemplateObject) -> bool:
    return any(r.field == "material.density" for r in getattr(t.placement, "randomize", ()))


def _draws_surface(t: TemplateObject) -> bool:
    """The object's placement draws a surface value per world: its rows always exist, so each world's draw has a row
    to land in (``alloy.scene.randomize_mujoco``)."""
    return any(r.field.startswith("material.surface.") for r in getattr(t.placement, "randomize", ()))


def _fit_values(b: newton.ModelBuilder, K: float, zeta: float, imp: float, step) -> dict[str, float]:
    """`alloy:pair_*` values for a row `alloy.contact.pair_fit` resizes (registered on first use, so scenes
    without armature rows keep their model digest)."""
    for name in FIT_ATTRS:
        _register(b, name)
    cap = stable_ke(zeta, step[0], imp, step[1]) if step[0] else 0.0
    return {"alloy:pair_k": K, "alloy:pair_zeta": zeta, "alloy:pair_kecap": cap}


def _register(b: newton.ModelBuilder, name: str) -> None:
    """Register `alloy:<name>` (one float per pair row) on first use, so scenes without such rows keep their digest."""
    import warp as wp

    if f"alloy:{name}" not in b.custom_attributes:
        b.add_custom_attribute(newton.ModelBuilder.CustomAttribute(
            name=name, frequency="mujoco:pair", dtype=wp.float32, default=0.0, namespace="alloy"))


def static_solmix(b: newton.ModelBuilder, objs: Sequence[TemplateObject]) -> None:
    """Near-zero solmix on shapes of static/kinematic bodies (and world shapes)."""
    key = "mujoco:geom_solmix"
    if key not in b.custom_attributes:
        return
    values = b.custom_attributes[key].values
    for t in objs:
        for s in t.ranges.shape:
            body = b.shape_body[s]
            if body < 0 or _mass(b, body) <= 0.0:
                values[s] = STATIC_SOLMIX


def add_native_pairs(b: newton.ModelBuilder, objs: Sequence[TemplateObject], entries: Sequence[pt.MaterialEntry],
                     lookup, combine: str, dt: float | None = None, integrator: str = "implicitfast") -> int:
    """Build the host pair tables for the template materials and add rows + solmix (assembler hook); row gains are
    capped for stability at `dt` under the entry's `integrator` (native_gains)."""
    built = pt.build_tables(entries, lookup, combine)
    static_solmix(b, objs)
    return pair_rows(b, objs, [e.key for e in entries], built.table, built.stiffness, dt, integrator)
