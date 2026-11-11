# SPDX-License-Identifier: Apache-2.0
"""Process-wide registry of solver capabilities, stability estimators, health adapters and state adapters (§3.1).

``SolverCapability.determinism`` is the single source of truth for what a solver path delivers (D9). Adapters are
looked up along the solver's MRO, so registering a base class covers its subclasses. ``isolated_registry`` gives
tests a scratch registry that is restored on exit.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

import warp as wp

from alloy._vendor import newton
from alloy._vendor.newton.solvers import SolverBase

from alloy.core.errors import RegistryError
from alloy.core.newton_compat import ModelView
from alloy.core.protocols import DetCapability, Emitter, StabilityEstimator

log = logging.getLogger("alloy.core")

FrictionMix = Literal["max", "mean", "geometric", "harmonic", "none"]
FRICTION_MIX_CODES: dict[str, int] = {"max": 0, "mean": 1, "geometric": 2, "harmonic": 3, "none": 4}


@dataclass(frozen=True)
class SolverCapability:
    """What one solver (stock Newton or Alloy plug-in) can do, and how to build it as a coupled entry."""

    name: str                                           # "mujoco", "vbd", "mpm", "sph", "kamino", ...
    factory: Callable[[ModelView, Mapping[str, Any]], SolverBase]
    routes: frozenset[str]
    owns: frozenset[str]                                # body, joint, shape, particle, tri, tet, edge, rod_joint
    coupling: bool                                      # subclasses CouplingInterface
    uniform_worlds: bool                                # SolverMuJoCo True
    per_contact_friction: bool                          # reads rigid_contact_friction (a scale, D13)
    determinism: DetCapability                          # the single source of truth (D9)
    friction_mix: FrictionMix = "max"                   # solver's own μ_mix (D13)
    restitution_path: bool = False                      # reads rigid_contact_damping (§3.4 PairTable)
    emitter: Emitter | None = None
    register_attributes: Callable[[newton.ModelBuilder], None] | None = None
    default_kwargs: Mapping[str, Any] = field(default_factory=dict)
    solver_cls: type | None = None                      # class built by ``factory``; ``capability_for`` (SCR-WP5-1)

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise RegistryError(f"SolverCapability.name must be a non-empty str, got {self.name!r}")
        if self.friction_mix not in FRICTION_MIX_CODES:
            raise RegistryError(f"{self.name}: unknown friction_mix {self.friction_mix!r}")
        object.__setattr__(self, "routes", frozenset(self.routes))
        object.__setattr__(self, "owns", frozenset(self.owns))
        object.__setattr__(self, "determinism", DetCapability(self.determinism))

    @property
    def friction_mix_code(self) -> int:
        """Integer code of ``friction_mix`` for kernels (``PairTable.mu_mix_rule``)."""
        return FRICTION_MIX_CODES[self.friction_mix]


@dataclass
class HealthSample:
    """What a health adapter writes, per world: solver iterations and residual.

    Convention (WP5): ``residual`` is normalised by the solver's own tolerance (> 1 = unconverged); NaN = the solver
    reports none (then no UNCONVERGED from the residual). MuJoCo's adapter writes iterations only.
    """

    iters: wp.array        # [W] int32
    residual: wp.array     # [W] float32


HealthAdapter = Callable[[SolverBase, HealthSample], None]
StateAdapter = Callable[[SolverBase], Mapping[str, wp.array]]
WorldMapAdapter = Callable[[SolverBase], Mapping[str, wp.array]]
EstimatorFactory = Callable[[SolverBase], StabilityEstimator]


@dataclass
class _Registry:
    solvers: dict[str, SolverCapability] = field(default_factory=dict)
    estimators: dict[type, EstimatorFactory] = field(default_factory=dict)
    health: dict[type, HealthAdapter] = field(default_factory=dict)
    state: dict[type, StateAdapter] = field(default_factory=dict)
    world_maps: dict[type, WorldMapAdapter] = field(default_factory=dict)

    def copy(self) -> _Registry:
        return _Registry(dict(self.solvers), dict(self.estimators), dict(self.health), dict(self.state),
                         dict(self.world_maps))


_REG = _Registry()
_T = TypeVar("_T")


# --- solver capabilities ---------------------------------------------------------------------------------------


def register_solver(cap: SolverCapability, *, replace: bool = False) -> None:
    """Register ``cap`` under ``cap.name``. Re-registering an equal capability is a no-op; a different one with the
    same name raises ``RegistryError`` unless ``replace``."""
    old = _REG.solvers.get(cap.name)
    if old is not None and old != cap and not replace:
        raise RegistryError(f"solver capability {cap.name!r} is already registered with different values")
    _REG.solvers[cap.name] = cap
    log.debug("registered solver capability %s (routes=%s, det=%s)", cap.name, sorted(cap.routes), cap.determinism.name)


def solver(name: str) -> SolverCapability:
    """The capability registered as ``name``; ``RegistryError`` if none."""
    try:
        return _REG.solvers[name]
    except KeyError:
        raise RegistryError(f"no solver capability {name!r}; registered: {sorted(_REG.solvers)}") from None


def solvers_for_route(route: str) -> list[SolverCapability]:
    """Capabilities that serve ``route`` (a ``Route`` value), sorted by name for determinism."""
    route = str(route)
    return sorted((c for c in _REG.solvers.values() if route in c.routes), key=lambda c: c.name)


def registered_solvers() -> tuple[str, ...]:
    """Names of all registered capabilities, sorted."""
    return tuple(sorted(_REG.solvers))


def registered_routes() -> frozenset[str]:
    """Union of the routes of every registered capability (``SceneContext.registered_routes``)."""
    return frozenset(r for c in _REG.solvers.values() for r in c.routes)


def capability_for(solver_obj: object) -> SolverCapability | None:
    """The capability whose ``solver_cls`` is the nearest class in ``solver_obj``'s MRO (None if none); how WP5
    finds ``determinism`` for a live solver instance (D9, SCR-WP5-1). Ties at one class resolve by name."""
    by_cls: dict[type, SolverCapability] = {}
    for cap in sorted(_REG.solvers.values(), key=lambda c: c.name, reverse=True):
        if cap.solver_cls is not None:
            by_cls[cap.solver_cls] = cap
    return _lookup(by_cls, solver_obj)


# --- per-solver-class adapters ---------------------------------------------------------------------------------


def _check_cls(solver_cls: type, what: str) -> None:
    if not isinstance(solver_cls, type):
        raise RegistryError(f"{what} must be registered for a class, got {solver_cls!r}")


def _lookup(table: Mapping[type, _T], solver_obj: object) -> _T | None:
    for klass in type(solver_obj).__mro__:
        if klass in table:
            return table[klass]
    return None


def register_estimator(solver_cls: type, factory: EstimatorFactory) -> None:
    """Register the stability-estimator factory for ``solver_cls`` (and its subclasses)."""
    _check_cls(solver_cls, "an estimator")
    _REG.estimators[solver_cls] = factory


def estimator_for(solver_obj: SolverBase) -> StabilityEstimator | None:
    """The solver itself if it implements ``StabilityEstimator`` (plug-ins), else the registered factory's
    estimator, else None."""
    if isinstance(solver_obj, StabilityEstimator):
        return solver_obj
    factory = _lookup(_REG.estimators, solver_obj)
    return None if factory is None else factory(solver_obj)


def register_health_adapter(solver_cls: type, fn: HealthAdapter) -> None:
    """Register the function that writes ``HealthSample`` (iterations, residual) for ``solver_cls``."""
    _check_cls(solver_cls, "a health adapter")
    _REG.health[solver_cls] = fn


def health_adapter_for(solver_obj: SolverBase) -> HealthAdapter | None:
    """The health adapter for ``solver_obj``'s class (MRO lookup), or None."""
    return _lookup(_REG.health, solver_obj)


def register_state_adapter(solver_cls: type, fn: StateAdapter, world_map: WorldMapAdapter | None = None) -> None:
    """Register the function listing ``solver_cls``'s internal arrays outside ``State`` (warm starts, histories),
    and optionally one giving the item->world int32 map of each per-item array (SCR-WP4-3; arrays without a map
    are world-major if ``shape[0] == world_count``, else whole)."""
    _check_cls(solver_cls, "a state adapter")
    _REG.state[solver_cls] = fn
    if world_map is not None:
        _REG.world_maps[solver_cls] = world_map
    else:
        _REG.world_maps.pop(solver_cls, None)


def state_arrays_for(solver_obj: SolverBase) -> Mapping[str, wp.array]:
    """Internal arrays of ``solver_obj`` from its state adapter; ``{}`` (logged) if none is registered."""
    fn = _lookup(_REG.state, solver_obj)
    if fn is None:
        log.info("no state adapter for %s: snapshots cover State only", type(solver_obj).__name__)
        return {}
    return dict(fn(solver_obj))


def state_world_maps_for(solver_obj: SolverBase) -> Mapping[str, wp.array]:
    """Item->world maps for ``state_arrays_for(solver_obj)`` (``WorldSnapshot(extra_world=)``); ``{}`` if none."""
    fn = _lookup(_REG.world_maps, solver_obj)
    return {} if fn is None else dict(fn(solver_obj))


PLUGIN_ROOT = "alloy.solvers"
CONTRIB = f"{PLUGIN_ROOT}.contrib"


def _subpackages(package: str) -> list[str]:
    try:
        root = importlib.import_module(package)
    except ImportError as err:
        log.debug("plug-in root %s not importable: %s", package, err)
        return []
    return sorted(f"{package}.{m.name}" for m in pkgutil.iter_modules(root.__path__) if m.ispkg)


def plugin_packages(*, include_contrib: bool = False) -> list[str]:
    """Plug-in packages under ``alloy.solvers`` (``alloy.solvers.<name>``; with ``include_contrib`` also the
    agent-written ``alloy.solvers.contrib.<name>``), sorted."""
    names = [n for n in _subpackages(PLUGIN_ROOT) if n != CONTRIB]
    if include_contrib:
        names += _subpackages(CONTRIB)
    return names


def load_plugins(*, include_contrib: bool = False) -> list[str]:
    """Import every plug-in package so it registers its capability, laws and bully targets (§3.8: registration
    happens on import of ``alloy.solvers.<name>``; SCR-WP9-5). Idempotent; a package that fails to import is logged
    and skipped. Returns the packages imported successfully.

    Agent-written ``contrib`` packages are imported only with ``include_contrib=True`` (factory and bully code paths).
    Registration happens once per process (the first import), so call this outside ``isolated_registry`` or
    re-register inside it.
    """
    loaded = []
    for name in plugin_packages(include_contrib=include_contrib):
        try:
            importlib.import_module(name)
        except Exception as err:  # noqa: BLE001 - one broken plug-in must not hide the others
            log.warning("plug-in %s failed to import: %s", name, err)
            continue
        loaded.append(name)
    return loaded


@contextmanager
def isolated_registry(empty: bool = False) -> Iterator[None]:
    """Run a block against a scratch copy of the registry (or an empty one); the original is restored after."""
    global _REG
    saved = _REG
    _REG = _Registry() if empty else saved.copy()
    try:
        yield
    finally:
        _REG = saved
