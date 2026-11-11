# SPDX-License-Identifier: Apache-2.0
"""Card validation (SPEC §3.2): material and pair rules, shared helpers and the public entry points.

Every rule names the dotted field it judges. ``error`` issues reject a card (``CardLibrary`` raises
``CardError``); ``warn`` issues are logged and kept. Object and scene rules live in ``validate_object.py``.
"""

from __future__ import annotations

import logging
import math
import re
import dataclasses
from collections.abc import Iterable
from dataclasses import dataclass, field

from alloy.cards import types as ct
from alloy.cards.paths import iter_q_leaves
from alloy.core.errors import CardError
from alloy.core.quantity import Q

log = logging.getLogger("alloy.cards")

DENSITY_RANGE = (10.0, 25_000.0)          # kg/m^3
YOUNGS_RANGE = (1e3, 1e12)                # Pa
MU_MAX_WARN = 2.0
POISSON_SNH_ONLY = 0.45                   # above: linear/corotated lock (object-model §4.4)
TORSIONAL_WARN = 1e-3                     # m (Hertz levers are ~1e-6..1e-4 m, GOTCHAS:501)
ROUND_SHAPES = frozenset({"sphere", "capsule", "cylinder", "ellipsoid"})
NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+$")     # card names: also file stems; ':' is reserved for object:feature


@dataclass
class Issues:
    """Collects ``CardIssue``s for one card; paths are prefixed with ``prefix``."""

    card: str
    prefix: str = ""
    items: list[ct.CardIssue] = field(default_factory=list)

    def _path(self, path: str) -> str:
        return f"{self.prefix}.{path}" if self.prefix and path else (self.prefix or path)

    def error(self, path: str, message: str) -> None:
        self.items.append(ct.CardIssue("error", self.card, self._path(path), message))

    def warn(self, path: str, message: str) -> None:
        self.items.append(ct.CardIssue("warn", self.card, self._path(path), message))

    def sub(self, prefix: str, card: str | None = None) -> Issues:
        """A collector that shares this list, with a longer prefix (and optionally another card name)."""
        child = Issues(card or self.card, self._path(prefix))
        child.items = self.items
        return child


def fmt(x: float) -> str:
    return f"{x:.4g}"


def check_range(iss: Issues, path: str, q: Q | float | None, lo: float | None = None, hi: float | None = None,
                *, open_lo: bool = False, open_hi: bool = False, unit: str = "") -> None:
    """Error if ``q`` (a Q or float) is outside [lo, hi] (open ends as flagged); None passes."""
    if q is None:
        return
    v = q.value if isinstance(q, Q) else float(q)
    if not math.isfinite(v):
        iss.error(path, f"non-finite value {v}")
        return
    below = lo is not None and (v <= lo if open_lo else v < lo)
    above = hi is not None and (v >= hi if open_hi else v > hi)
    if below or above:
        lb = "(" if open_lo else "["
        rb = ")" if open_hi else "]"
        rng = f"{lb}{'-inf' if lo is None else fmt(lo)}, {'inf' if hi is None else fmt(hi)}{rb}"
        iss.error(path, f"{fmt(v)}{unit} outside {rng}{unit}")


def check_positive(iss: Issues, path: str, q: Q | float | None, unit: str = "") -> None:
    check_range(iss, path, q, 0.0, None, open_lo=True, unit=unit)


def check_nonneg(iss: Issues, path: str, q: Q | float | None, unit: str = "") -> None:
    check_range(iss, path, q, 0.0, None, unit=unit)


def check_plain_floats(iss: Issues, node: object, prefix: str = "") -> None:
    """Non-``Q`` float fields (sizes, poses, crops, rates) must be finite; YAML rejects them at load, this catches
    programmatic cards."""
    if isinstance(node, float):
        if not math.isfinite(node):
            iss.error(prefix, f"non-finite value {node}")
    elif isinstance(node, Q):
        return
    elif dataclasses.is_dataclass(node):
        for f in dataclasses.fields(node):
            check_plain_floats(iss, getattr(node, f.name), f"{prefix}.{f.name}" if prefix else f.name)
    elif isinstance(node, tuple):
        for i, item in enumerate(node):
            check_plain_floats(iss, item, f"{prefix}.{i}" if prefix else str(i))


def check_leaves(iss: Issues, node: object, prefix: str = "") -> None:
    """Rules every ``Q`` obeys: finite value, a note unless the source is ``default``, a sane ``unc``."""
    for path, q in iter_q_leaves(node, prefix):
        if not math.isfinite(q.value):
            iss.error(path, f"non-finite value {q.value}")
        if q.source != "default" and not q.note.strip():
            iss.error(path, f"source {q.source!r} needs a note (citation or derivation)")
        if q.unc is None:
            continue
        lo, hi = q.unc
        if not (math.isfinite(lo) and math.isfinite(hi)) or lo > hi:
            iss.error(f"{path}.unc" if path else "unc", f"unc must be finite with lo <= hi, got ({lo}, {hi})")
        elif not lo <= q.value <= hi:
            iss.warn(path, f"value {fmt(q.value)} outside its unc [{fmt(lo)}, {fmt(hi)}]")


def check_friction(iss: Issues, mu_s: Q, mu_d: Q, prefix: str = "") -> None:
    p = f"{prefix}." if prefix else ""
    for name, q in (("mu_static", mu_s), ("mu_dynamic", mu_d)):
        check_range(iss, p + name, q, 0.0, None)
        if math.isfinite(q.value) and q.value > MU_MAX_WARN:
            iss.warn(p + name, f"{fmt(q.value)} > {MU_MAX_WARN} (unusual; adhesive contact?)")
    if mu_d.value > mu_s.value:
        iss.error(p + "mu_dynamic", f"mu_dynamic {fmt(mu_d.value)} > mu_static {fmt(mu_s.value)}")


def _check_surface(iss: Issues, s: ct.Surface) -> None:
    check_friction(iss, s.mu_static, s.mu_dynamic, "surface")
    check_range(iss, "surface.restitution", s.restitution, 0.0, 1.0)
    check_nonneg(iss, "surface.torsional_lever", s.torsional_lever, " m")
    if s.torsional_lever is not None and s.torsional_lever.value > TORSIONAL_WARN:
        iss.warn("surface.torsional_lever", f"{fmt(s.torsional_lever.value)} m > {TORSIONAL_WARN} m; Hertz levers "
                 "are ~1e-6..1e-4 m (GOTCHAS:501), a 5 mm default over-damps spin")
    check_nonneg(iss, "surface.rolling_lever", s.rolling_lever, " m")
    check_positive(iss, "surface.contact_stiffness", s.contact_stiffness, " N/m")
    check_positive(iss, "surface.hydro_modulus", s.hydro_modulus, " Pa")
    check_nonneg(iss, "surface.contact_damping_ratio", s.contact_damping_ratio)


def _check_elastic(iss: Issues, e: ct.Elastic) -> None:
    check_range(iss, "elastic.youngs", e.youngs, *YOUNGS_RANGE, unit=" Pa")
    check_range(iss, "elastic.poisson", e.poisson, -1.0, 0.5, open_lo=True, open_hi=True)
    if e.model in ("linear", "corotated") and e.poisson.value > POISSON_SNH_ONLY:
        iss.error("elastic.model", f"model {e.model!r} with poisson {fmt(e.poisson.value)} > {POISSON_SNH_ONLY} "
                  "locks volumetrically; use 'snh'")
    check_nonneg(iss, "elastic.rayleigh_beta", e.rayleigh_beta, " s")
    check_positive(iss, "elastic.bending", e.bending, " N·m")
    check_positive(iss, "elastic.bending_hysteresis", e.bending_hysteresis, " N")
    if e.bending_hysteresis is not None and e.bending is None:
        iss.warn("elastic.bending_hysteresis", "KES 2HB is read beside the card's KES B (elastic.bending): without B "
                 "the hinge's friction moment HB = 2HB/2 meets a bending rigidity derived from E")
    if e.bending is not None and e.cloth is not None:
        iss.warn("elastic.bending", "CLOTH_STYLE3D takes its bending from elastic.cloth (bend_warp / bend_weft); "
                 "elastic.bending is the CLOTH_VBD route's")
    if e.cloth is not None:
        for name in ("stretch_warp", "stretch_weft", "shear", "bend_warp", "bend_weft"):
            check_positive(iss, f"elastic.cloth.{name}", getattr(e.cloth, name))


def _check_plastic(iss: Issues, p: ct.Plastic) -> None:
    for name in ("yield_stress", "yield_pressure", "cohesion", "hardening", "hardening_modulus"):
        check_nonneg(iss, f"plastic.{name}", getattr(p, name))
    if p.hardening_modulus is not None and p.model != "j2":
        iss.error("plastic.hardening_modulus", "hardening_modulus is the j2 model's: set plastic.model: j2")
    check_range(iss, "plastic.friction_angle", p.friction_angle, 0.0, math.pi / 2, open_lo=True, open_hi=True,
                unit=" rad")
    if p.dilatancy is not None and p.dilatancy.value < 0:
        iss.warn("plastic.dilatancy", f"negative dilatancy {fmt(p.dilatancy.value)} (contractive material)")


def _check_j2(iss: Issues, card: ct.MaterialCard) -> None:
    """A j2 solid (SCR-WP7-15): elastic moduli and a positive yield stress; no granular or fluid fields."""
    p = card.plastic
    if card.elastic is None:
        iss.error("plastic.model", "a j2 solid needs elastic.youngs and elastic.poisson")
    if p.yield_stress is None or p.yield_stress.value <= 0.0:
        iss.error("plastic.yield_stress", "a j2 solid needs a positive yield_stress (uniaxial initial yield, Pa)")
    for name in ("friction_angle", "cohesion", "yield_pressure", "dilatancy", "hardening"):
        if getattr(p, name) is not None:
            iss.error(f"plastic.{name}", f"{name} is not a j2 field (granular / paste models); use hardening_modulus")
    if card.fluid is not None:
        iss.error("plastic.model", "a j2 solid has no fluid block")


def _check_fluid(iss: Issues, f: ct.Fluid, density: Q, yields: bool = False) -> None:
    check_nonneg(iss, "fluid.viscosity", f.viscosity, " Pa·s")
    check_nonneg(iss, "fluid.surface_tension", f.surface_tension, " N/m")
    check_nonneg(iss, "fluid.adhesion", f.adhesion)
    check_positive(iss, "fluid.sound_speed", f.sound_speed, " m/s")
    _check_herschel_bulkley(iss, f, yields)
    if density.value < 300:
        iss.warn("density", f"fluid density {fmt(density.value)} kg/m^3 is below any common liquid")


def _check_herschel_bulkley(iss: Issues, f: ct.Fluid, yields: bool) -> None:
    """K and n come together; K > 0, 0 < n <= 1.5; the solve needs a positive starting viscosity (fluid.viscosity, the
    apparent or Bingham plastic viscosity at the card's reference rate). ``yields``: the material has a yield stress,
    which routes it to the paste (MPM) route whatever its viscosity."""
    k, n = f.consistency, f.flow_index
    if (k is None) != (n is None):
        iss.error("fluid.consistency" if k is None else "fluid.flow_index",
                  "Herschel-Bulkley needs both fluid.consistency (K, Pa·s^n) and fluid.flow_index (n)")
        return
    if k is None:
        return
    check_positive(iss, "fluid.consistency", k, " Pa·s^n")
    check_range(iss, "fluid.flow_index", n, 0.0, 1.5, open_lo=True)
    if f.viscosity.value <= 0.0:
        iss.error("fluid.viscosity", "a Herschel-Bulkley material needs a positive starting viscosity")
    elif f.viscosity.value <= 1.0 and not yields:
        iss.warn("fluid.consistency", "Herschel-Bulkley acts on the paste (MPM) route only: with viscosity <= 1 Pa·s "
                 "and no plastic.yield_stress this material routes to SPH, which ignores K and n")


def _check_failure(iss: Issues, f: ct.Failure) -> None:
    for name in ("fracture_strain", "fracture_stress", "tear_energy", "cut_min_force", "cut_min_strain"):
        check_positive(iss, f"failure.{name}", getattr(f, name))


def _check_film(iss: Issues, f: ct.Film) -> None:
    check_range(iss, "film.dead_fold_yield", f.dead_fold_yield, 0.0, math.pi, open_hi=True, unit=" rad")
    check_positive(iss, "film.contact_radius", f.contact_radius, " m")
    check_nonneg(iss, "film.air_scale", f.air_scale)
    check_positive(iss, "film.crease_ratio", f.crease_ratio)
    check_range(iss, "film.crease_yield", f.crease_yield, 0.0, math.pi, open_hi=True, unit=" rad")


def material_issues(card: ct.MaterialCard, iss: Issues) -> None:
    """All material rules into ``iss`` (paths relative to the material)."""
    if not NAME_RE.match(card.name):
        iss.error("name", f"material names use [A-Za-z0-9_-] only (they are file names), got {card.name!r}")
    check_leaves(iss, card)
    check_range(iss, "density", card.density, *DENSITY_RANGE, unit=" kg/m^3")
    _check_surface(iss, card.surface)
    if card.elastic is not None:
        _check_elastic(iss, card.elastic)
    if card.plastic is not None:
        _check_plastic(iss, card.plastic)
        if card.plastic.model == "j2":
            _check_j2(iss, card)
    if card.fluid is not None:
        yields = card.plastic is not None and card.plastic.yield_stress is not None and card.plastic.yield_stress.value > 0
        _check_fluid(iss, card.fluid, card.density, yields)
    if card.failure is not None:
        _check_failure(iss, card.failure)
    if card.film is not None:
        _check_film(iss, card.film)


def validate_material(card: ct.MaterialCard) -> list[ct.CardIssue]:
    """Value rules for one material card (density, moduli, ν, μ, e, ...); see SPEC §3.2."""
    iss = Issues(card.name)
    material_issues(card, iss)
    return iss.items


def validate_pair(card: ct.PairCard, lib: object | None = None) -> list[ct.CardIssue]:
    """Pair rules; with a library, both material names must resolve."""
    iss = Issues("__".join(sorted((card.a, card.b))))
    pair_issues(card, iss, lib)
    return iss.items


def pair_issues(card: ct.PairCard, iss: Issues, lib: object | None) -> None:
    check_leaves(iss, card)
    check_friction(iss, card.mu_static, card.mu_dynamic)
    check_range(iss, "restitution", card.restitution, 0.0, 1.0)
    check_nonneg(iss, "torsional_lever", card.torsional_lever, " m")
    if card.torsional_lever is not None and card.torsional_lever.value > TORSIONAL_WARN:
        iss.warn("torsional_lever", f"{fmt(card.torsional_lever.value)} m > {TORSIONAL_WARN} m (GOTCHAS:501)")
    if lib is None:
        return
    for side in ("a", "b"):
        name = getattr(card, side)
        if not lib.has_material(name):  # type: ignore[attr-defined]
            iss.error(side, f"unknown material {name!r}")


def errors(issues: Iterable[ct.CardIssue]) -> list[ct.CardIssue]:
    return [i for i in issues if i.level == "error"]


DEFAULT_LEAF_WARNING = "still at source 'default' and it changes routing or contact"


def log_issues(issues: list[ct.CardIssue]) -> None:
    """Log warnings once: default-leaf warnings as one INFO line per card, the rest at WARNING."""
    defaults: dict[str, list[str]] = {}
    for i in issues:
        if i.level != "warn":
            continue
        if i.message == DEFAULT_LEAF_WARNING:
            defaults.setdefault(i.card, []).append(i.field)
        else:
            log.warning("card %s: %s: %s", i.card, i.field, i.message)
    for card, paths in defaults.items():
        log.info("card %s: %d default leaves change routing or contact: %s", card, len(paths), ", ".join(paths))


def raise_on_error(issues: list[ct.CardIssue], what: str = "") -> None:
    """Raise ``CardError`` for the first error-level issue (the message counts and names the rest)."""
    errs = errors(issues)
    if not errs:
        return
    first = errs[0]
    more = f" (+{len(errs) - 1} more: {', '.join(e.field for e in errs[1:6])})" if len(errs) > 1 else ""
    raise CardError(first.card, first.field, f"{first.message}{more}{f' [{what}]' if what else ''}")


def validate_object(card: ct.ObjectCard, lib: object, rigid_dt: float = 1.25e-3) -> list[ct.CardIssue]:
    """Object rules (SPEC §3.2); implemented in ``alloy.cards.validate_object``."""
    from alloy.cards.validate_object import validate_object as _impl  # noqa: PLC0415 - import cycle

    return _impl(card, lib, rigid_dt)  # type: ignore[arg-type]


def validate_scene(scene: ct.SceneCard, lib: object, rigid_dt: float = 1.25e-3, name: str = "scene") -> list[ct.CardIssue]:
    """Scene rules plus every placed object's rules; implemented in ``alloy.cards.validate_scene``."""
    from alloy.cards.validate_scene import validate_scene as _impl  # noqa: PLC0415 - import cycle

    return _impl(scene, lib, rigid_dt, name)  # type: ignore[arg-type]
