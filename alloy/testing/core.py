# SPDX-License-Identifier: Apache-2.0
"""``FakeSolver``, ``synthetic_rig`` and the card fakes (re-exported by ``alloy.testing.fakes``; WP0).

``FakeSolver`` is a ``SolverBase`` + ``CouplingInterface`` with scripted, deterministic dynamics: free bodies and
particles under gravity, applied forces (``state.body_f`` / ``particle_f``), linear damping, a floor, and an optional
seeded instability (worlds whose fastest body exceeds ``blowup_speed`` get their velocities multiplied by
``blowup_gain`` every step). It also implements ``StabilityEstimator`` and the ``AlloySolverPlugin`` members, and is
NATIVE: per-world loops only, no atomics.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy._vendor.newton import Contacts, Control, Model, State
from alloy._vendor.newton.solvers import SolverBase

from alloy.bench.types import Condition, RigCard, RigDataset, RigTrial
from alloy.cards.types import MaterialCard, ObjectCard, PairCard, Placement, Primitive, ResolvedObject
from alloy.core.defaults import MATERIAL_CONTACT_STIFFNESS
from alloy.core.ids import digest_of
from alloy.core.newton_compat import CouplingInterface
from alloy.core.protocols import DetCapability
from alloy.core.quantity import Q
from alloy.precision.types import PrecisionObservables
from alloy.route.types import Route, RouteDecision, SceneContext


@wp.kernel
def _integrate_bodies(q: wp.array(dtype=wp.transform), qd: wp.array(dtype=wp.spatial_vector),
                      f: wp.array(dtype=wp.spatial_vector), inv_m: wp.array(dtype=wp.float32),
                      gravity: wp.vec3, damping: wp.float32, floor_z: wp.float32, use_floor: wp.int32,
                      radius: wp.float32, dt: wp.float32,
                      q_out: wp.array(dtype=wp.transform), qd_out: wp.array(dtype=wp.spatial_vector)):
    i = wp.tid()
    v = wp.spatial_top(qd[i])
    w = wp.spatial_bottom(qd[i])
    a = gravity
    if inv_m[i] > 0.0:
        a = a + wp.spatial_top(f[i]) * inv_m[i]
    else:
        a = wp.vec3(0.0)
    v = (v + a * dt) * (1.0 - damping)
    p = wp.transform_get_translation(q[i]) + v * dt
    if use_floor != 0 and p[2] < floor_z + radius:
        p = wp.vec3(p[0], p[1], floor_z + radius)
        if v[2] < 0.0:
            v = wp.vec3(v[0], v[1], 0.0)
    rot = wp.transform_get_rotation(q[i])
    dq = wp.quat(w[0], w[1], w[2], 0.0) * rot * (0.5 * dt)
    rot = wp.normalize(rot + dq)
    q_out[i] = wp.transform(p, rot)
    qd_out[i] = wp.spatial_vector(v, w)


@wp.kernel
def _integrate_particles(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), f: wp.array(dtype=wp.vec3),
                         inv_m: wp.array(dtype=wp.float32), gravity: wp.vec3, damping: wp.float32, dt: wp.float32,
                         x_out: wp.array(dtype=wp.vec3), v_out: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    a = wp.vec3(0.0)
    if inv_m[i] > 0.0:
        a = gravity + f[i] * inv_m[i]
    vel = (v[i] + a * dt) * (1.0 - damping)
    x_out[i] = x[i] + vel * dt
    v_out[i] = vel


@wp.kernel
def _blowup(qd: wp.array(dtype=wp.spatial_vector), start: wp.array(dtype=wp.int32), threshold: wp.float32,
            gain: wp.float32):
    w = wp.tid()
    fastest = float(0.0)
    for i in range(start[w], start[w + 1]):
        fastest = wp.max(fastest, wp.length(wp.spatial_top(qd[i])))
    if fastest > threshold:
        for i in range(start[w], start[w + 1]):
            qd[i] = qd[i] * gain


@wp.kernel
def _energy(q: wp.array(dtype=wp.transform), qd: wp.array(dtype=wp.spatial_vector), mass: wp.array(dtype=wp.float32),
            start: wp.array(dtype=wp.int32), g: wp.float32, z_ref: wp.array(dtype=wp.float64),
            out: wp.array(dtype=wp.float64)):
    w = wp.tid()
    e = wp.float64(0.0)
    for i in range(start[w], start[w + 1]):
        v = wp.spatial_top(qd[i])
        m = wp.float64(mass[i])
        z = wp.float64(wp.transform_get_translation(q[i])[2])
        e = e + wp.float64(0.5) * m * wp.float64(wp.dot(v, v)) + m * wp.float64(g) * (z - z_ref[w])
    out[w] = e


@wp.kernel
def _fill_obs(residual: wp.array(dtype=wp.float32), iterations: wp.array(dtype=wp.int32), iters: wp.int32):
    w = wp.tid()
    residual[w] = 0.0
    iterations[w] = iters


@wp.kernel
def _fill_f32(out: wp.array(dtype=wp.float32), value: wp.float32):
    out[wp.tid()] = value


def world_starts(item_world: np.ndarray, world_count: int) -> np.ndarray:
    """[W+1] start offsets of the contiguous per-world item blocks (``replicate`` keeps items world-sorted; global
    items, world -1, must sit before or after all of them). Raises ``ValueError`` otherwise."""
    worlds = np.asarray(item_world, dtype=np.int64)
    idx = np.flatnonzero(worlds >= 0)
    if idx.size == 0:
        return np.zeros(world_count + 1, dtype=np.int32)
    local = worlds[idx]
    if np.any(np.diff(local) < 0) or idx[-1] - idx[0] + 1 != idx.size:
        raise ValueError("per-world items are not contiguous and world-sorted")
    return (idx[0] + np.searchsorted(local, np.arange(world_count + 1), side="left")).astype(np.int32)


class FakeSolver(SolverBase, CouplingInterface):
    """Scripted, deterministic stand-in solver (see module docstring)."""

    plugin_name: ClassVar[str] = "fake"
    routes: ClassVar[frozenset[str]] = frozenset({"rigid"})
    determinism: ClassVar[DetCapability] = DetCapability.NATIVE
    Observables = PrecisionObservables
    SUPPORTED_OBSERVABLES: ClassVar[frozenset[str]] = frozenset({"residual", "iterations"})

    @dataclass
    class Config:
        gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
        damping: float = 0.0                    # fraction of velocity removed per step
        floor_z: float | None = 0.0             # bodies are kept at z >= floor_z + body_radius; None = no floor
        body_radius: float = 0.1
        blowup_speed: float | None = None       # seeded instability threshold [m/s]; None = stable
        blowup_gain: float = 2.0
        stable_dt: float = 1.25e-3              # what alloy_stable_dt reports [s]
        iterations: int = 3                     # what PrecisionObservables.iterations reports

    def __init__(self, model: Model, config: FakeSolver.Config | None = None) -> None:
        super().__init__(model)
        self.config = config or FakeSolver.Config()
        self.steps = 0
        w = int(model.world_count)
        bw = model.body_world.numpy() if model.body_count else np.zeros(0, np.int32)
        self._body_start = wp.array(world_starts(bw, w), dtype=wp.int32, device=model.device)
        self._body_mass = (wp.array(_masses(model.body_inv_mass.numpy()), dtype=wp.float32, device=model.device)
                           if model.body_count else None)

    def step(self, state_in: State, state_out: State, control: Control | None, contacts: Contacts | None,
             dt: float, *, observables: SolverBase.Observables | None = None) -> None:
        self.validate_observables(observables, contacts)
        cfg = self.config
        g = wp.vec3(*cfg.gravity)
        if self.model.body_count:
            floor = 0.0 if cfg.floor_z is None else cfg.floor_z
            wp.launch(_integrate_bodies, dim=self.model.body_count, device=self.model.device,
                      inputs=[state_in.body_q, state_in.body_qd, state_in.body_f, self.model.body_inv_mass, g,
                              cfg.damping, floor, int(cfg.floor_z is not None), cfg.body_radius, dt],
                      outputs=[state_out.body_q, state_out.body_qd])
            if cfg.blowup_speed is not None:
                wp.launch(_blowup, dim=self.model.world_count, device=self.model.device,
                          inputs=[state_out.body_qd, self._body_start, cfg.blowup_speed, cfg.blowup_gain])
        if self.model.particle_count:
            wp.launch(_integrate_particles, dim=self.model.particle_count, device=self.model.device,
                      inputs=[state_in.particle_q, state_in.particle_qd, state_in.particle_f,
                              self.model.particle_inv_mass, g, cfg.damping, dt],
                      outputs=[state_out.particle_q, state_out.particle_qd])
        if isinstance(observables, PrecisionObservables) and observables.residual is not None:
            wp.launch(_fill_obs, dim=observables.residual.shape[0], device=self.model.device,
                      inputs=[observables.residual, observables.iterations, cfg.iterations])
        self.steps += 1

    # --- StabilityEstimator -------------------------------------------------------------------------------------

    def alloy_stable_dt(self, state: State, contacts: Contacts | None, out: wp.array) -> None:
        wp.launch(_fill_f32, dim=out.shape[0], inputs=[out, self.config.stable_dt], device=out.device)

    def alloy_energy(self, state: State, z_ref: wp.array, out: wp.array) -> None:
        if not self.model.body_count:
            out.zero_()
            return
        g = math.sqrt(sum(c * c for c in self.config.gravity))
        wp.launch(_energy, dim=out.shape[0], device=out.device,
                  inputs=[state.body_q, state.body_qd, self._body_mass, self._body_start, g, z_ref, out])

    # --- AlloySolverPlugin members ------------------------------------------------------------------------------

    @staticmethod
    def register_custom_attributes(builder: newton.ModelBuilder) -> None:
        """The fake needs no custom attributes."""

    @staticmethod
    def emit(builder: newton.ModelBuilder, obj: object, decision: object) -> None:
        """The fake emits nothing; reference scenes build their own models."""

    def alloy_state_arrays(self, state: State) -> Mapping[str, wp.array]:
        names = ("body_q", "body_qd", "particle_q", "particle_qd")
        return {n: getattr(state, n) for n in names if getattr(state, n, None) is not None}

    def alloy_capacity_report(self) -> Mapping[str, int]:
        return {}

    def alloy_state_worlds(self) -> Mapping[str, wp.array]:
        """Item->world maps for ``alloy_state_arrays`` (SCR-WP9-4)."""
        m = self.model
        maps = {"body_q": m.body_world, "body_qd": m.body_world, "particle_q": m.particle_world,
                "particle_qd": m.particle_world}
        return {k: v for k, v in maps.items() if v is not None and v.shape[0] > 0}


def _masses(inv_mass: np.ndarray) -> np.ndarray:
    inv = np.asarray(inv_mass, dtype=np.float64)
    return np.where(inv > 0.0, 1.0 / np.where(inv > 0.0, inv, 1.0), 0.0).astype(np.float32)


# --- synthetic rig data ------------------------------------------------------------------------------------------

RigModel = Callable[[Mapping[str, float], Condition], dict[str, float]]


def _incline_model(params: Mapping[str, float], cond: Condition) -> dict[str, float]:
    """R3: the box slips once tan(theta) > mu_s; onset angle atan(mu_s) [rad]."""
    return {"slip_angle": math.atan(params["mu_static"])}


def _bounce_model(params: Mapping[str, float], cond: Condition) -> dict[str, float]:
    """R4: rebound height e^2·h for a drop from h [m]."""
    h = cond.params.get("drop_height", params.get("drop_height", 0.1))
    return {"rebound_height": params["restitution"] ** 2 * h}


SYNTHETIC_MODELS: dict[str, RigModel] = {"R3": _incline_model, "R4": _bounce_model}


def synthetic_rig(law: str, params: Mapping[str, float], noise: float | Mapping[str, float] = 0.0, *,
                  conditions: Sequence[Condition] | None = None, trials_per_condition: int = 5,
                  calib_per_condition: int = 3, seed: int = 0, model: RigModel | None = None) -> RigDataset:
    """A ``RigDataset`` generated from an analytic law plus Gaussian noise (1-sigma, absolute, per scalar).

    ``model`` (or the built-in model for the law prefix: R3 incline onset, R4 drop/bounce) maps (params, condition)
    to true scalars; unknown laws echo ``params``. The first ``calib_per_condition`` trials of each condition are
    ``calib``, the rest ``valid``. Deterministic in ``seed``.
    """
    rng = np.random.default_rng(seed)
    conds = tuple(conditions or (Condition("c0", {}),))
    truth = model or SYNTHETIC_MODELS.get(law.split(".")[0], lambda p, c: dict(p))
    trials = []
    for cond in conds:
        values = truth(params, cond)
        for k in range(trials_per_condition):
            scalars = {name: float(v + _sigma(noise, name) * rng.standard_normal()) for name, v in values.items()}
            split = "calib" if k < calib_per_condition else "valid"
            trials.append(RigTrial(cond.id, k, split, {"t": np.linspace(0.0, 1.0, 11)}, scalars))
    card = RigCard(law, f"synthetic-{seed}", "synthetic/1", {}, {"synthetic": {"u95": 0.0}}, {}, {})
    sha = digest_of({"law": law, "params": dict(params), "noise": noise, "seed": seed,
                     "trials": [(t.condition_id, t.index, t.split, dict(t.scalars)) for t in trials]})
    return RigDataset(card, conds, tuple(trials), sha)


def _sigma(noise: float | Mapping[str, float], name: str) -> float:
    return float(noise.get(name, 0.0)) if isinstance(noise, Mapping) else float(noise)


# --- card fakes (``FakeCardLibrary``, ``FakeRouter``; re-exported by alloy.testing.fakes) ---------------------------
# Path grammar (SCR-WP1-1): dotted, relative to a placed object; tuple items as ``.<i>``; material leaves as
# ``material.<MaterialCard path>`` and ``part_materials.<part>.<path>``. WP1's ``alloy.cards.paths`` is the real one.

WORST_LO: tuple[str, ...] = ("elastic.youngs", "elastic.poisson", "loads.tolerance", "features.thickness",
                             "features.passage_width", "features.min_wall", "clearance")
WORST_HI: tuple[str, ...] = ("density", "loads.grip_force", "loads.bend_force", "loads.design_force",
                             "loads.contact_speed", "fluid.viscosity")


def worst_end(path: str, lo: float, hi: float) -> float:
    """Routing-worst end of a randomized range (SCR-WP1-3): ``lo`` for stiffness/tolerance/size leaves, ``hi`` for
    density/loads/viscosity; other leaves ``hi`` (the fake has no nominal value to compare against)."""
    if any(path == key or path.endswith("." + key) for key in WORST_LO):
        return lo
    return hi


def default_leaves(obj: Any, prefix: str = "") -> tuple[str, ...]:
    """Dotted paths (``.i`` for tuple items) of every ``Q`` leaf with source "default" in a card."""
    if isinstance(obj, Q):
        return (prefix,) if obj.is_default else ()
    out: list[str] = []
    if is_dataclass(obj) and not isinstance(obj, type):
        for f in fields(obj):
            out += default_leaves(getattr(obj, f.name), f"{prefix}.{f.name}" if prefix else f.name)
    elif isinstance(obj, tuple):
        for i, item in enumerate(obj):
            out += default_leaves(item, f"{prefix}.{i}")
    return tuple(out)


def fill_material_defaults(material: MaterialCard) -> MaterialCard:
    """``surface.contact_stiffness`` filled as ``Q(2e6, "default")`` when None (SCR-WP1-5)."""
    if material.surface.contact_stiffness is not None:
        return material
    surface = replace(material.surface, contact_stiffness=Q(MATERIAL_CONTACT_STIFFNESS, "default"))
    return replace(material, surface=surface)


def primitive_extents(shape: object) -> tuple[float, float, float]:
    """OBB extents l1 >= l2 >= l3 [m] of a ``Primitive`` (0.1 m cube for other sources)."""
    if not isinstance(shape, Primitive):
        return (0.1, 0.1, 0.1)
    s = tuple(float(x) for x in shape.size)
    if shape.kind == "box":
        dims = (2 * s[0], 2 * s[1], 2 * s[2])
    elif shape.kind == "sphere":
        dims = (2 * s[0],) * 3
    elif shape.kind == "capsule":
        dims = (2 * (s[1] + s[0]), 2 * s[0], 2 * s[0])
    elif shape.kind in ("cylinder", "cone"):
        dims = (2 * s[1], 2 * s[0], 2 * s[0])
    else:
        dims = tuple(2 * x for x in (s + (s[-1],) * 3)[:3])
    a, b, c = sorted(dims, reverse=True)
    return (a, b, c)


class FakeCardLibrary:
    """``CardLibraryLike`` in memory (stands in for WP1's ``CardLibrary`` before it merges). ``resolve`` follows
    the WP1 rules: defaults filled, material leaves in ``default_leaves``, worst-case ends, and ``card_hash`` over
    object + materials by value + randomization (SCR-WP1-1/2/3/5)."""

    def __init__(self, materials: Mapping[str, MaterialCard] = (), objects: Mapping[str, ObjectCard] = (),
                 pairs: Mapping[tuple[str, str], PairCard] = ()) -> None:
        self.materials, self.objects = dict(materials), dict(objects)
        self.pairs = {tuple(sorted(k)): v for k, v in dict(pairs).items()}
        self.puts: list[tuple[str, str]] = []

    def material(self, name: str) -> MaterialCard:
        return self.materials[name]

    def pair(self, a: str, b: str) -> PairCard | None:
        return self.pairs.get(tuple(sorted((a, b))))

    def object(self, name: str) -> ObjectCard:
        return self.objects[name]

    def resolve(self, card: ObjectCard, placement: Placement | None = None) -> ResolvedObject:
        randomize = placement.randomize if placement else ()
        material = fill_material_defaults(self.materials[card.material])
        parts = {k: fill_material_defaults(self.materials[v]) for k, v in card.part_materials.items()}
        leaves = default_leaves(card) + default_leaves(material, "material")
        for part, mat in sorted(parts.items()):
            leaves += default_leaves(mat, f"part_materials.{part}")
        worst = {r.field: worst_end(r.field, r.lo, r.hi) for r in randomize}
        card_hash = digest_of({"object": card, "material": material, "part_materials": parts,
                               "randomize": randomize})
        return ResolvedObject(card, material, parts, card_hash, primitive_extents(card.shape), worst, leaves)

    def put(self, card: MaterialCard | PairCard | ObjectCard, *, reason: str) -> Path:
        if isinstance(card, PairCard):
            self.pairs[card.key] = card
            name = "__".join(card.key)
        else:
            (self.materials if isinstance(card, MaterialCard) else self.objects)[card.name] = card
            name = card.name
        self.puts.append((name, reason))
        return Path("memory") / f"{name}.yaml"


class FakeRouter:
    """``RouterLike``: static/kinematic -> STATIC, everything else RIGID on "mujoco" (no physics checks)."""

    def route(self, obj: ResolvedObject, ctx: SceneContext) -> RouteDecision:
        static = obj.card.motion != "dynamic"
        return RouteDecision(Route.STATIC if static else Route.RIGID, "" if static else "mujoco", {}, math.inf,
                             False, ("fake router",))
