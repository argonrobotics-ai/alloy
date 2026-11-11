# SPDX-License-Identifier: Apache-2.0
"""Alloy's deterministic route for SolverMuJoCo (P3-1 round 5): what it installs and which paths it covers.

``enable_route(solver)`` makes a built Newton ``SolverMuJoCo`` repeat bitwise (R1, R2, R4) and stay
graph-capturable when its MuJoCo Warp path is covered. It installs, module-wide but acting only on enabled models:

- the contact and constraint-row sorts (``opt.deterministic`` = CONTACTS | CONSTRAINT, Alloy's fill of the two
  stubs, ``mjw_order`` / ``mjw_rows``);
- a Fixed64 sparse Newton Hessian (``mjw_hessian``), an ordered grad_dot and a Fixed64 qfrc_constraint
  (``mjw_sums``);
- ordered tree sums for ``smooth.com_pos``, ``crb`` and the RNE backward pass (``mjw_sums``; our pin only).

MuJoCo Warp marks every kernel whose float sums can race with a ``deterministic`` switch on its factory. Coverage
is decided in two steps:
1. static: Newton or CG solver, elliptic cones, sparse Jacobian, no flex, sleep off, a proven integrator;
2. a probe: one eager step of a 1-world scratch ``Data`` records which marked factories the model's step calls.
   Each must be replaced (``SUBSTITUTIONS``) or listed in ``SAFE`` with the reason it does not race on this path.
Anything else keeps SolverMuJoCo at NONE, and ``route_reason`` names it. Set ``ALLOY_MJW_ROUTE=off`` to never enable.
"""

import contextlib
import importlib
import inspect
import logging
import os
import pkgutil
import weakref
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import warp as wp

from alloy.core.errors import PrecisionError
from alloy.precision.owned_cache import remember

log = logging.getLogger("alloy.precision")

CONTACTS, CONSTRAINT = 1, 2
SORT_BITS = CONTACTS | CONSTRAINT
_PKG = "alloy._vendor.mujoco_warp._src"
# (module, factory) -> handler key in _handlers(); stages that must be scoped for the substitutions to act
SUBSTITUTIONS: dict[tuple[str, str], str] = {
    (f"{_PKG}.solver", "_JTDACJ_sparse"): "jtdaj",
    (f"{_PKG}.solver", "_update_gradient_grad"): "grad",
    (f"{_PKG}.solver", "_update_constraint_init_qfrc_constraint_sparse"): "qfrc",
    (f"{_PKG}.smooth", "_subtree_com_acc"): "tree_vec3",
    (f"{_PKG}.smooth", "_crb_accumulate"): "tree_vec10",
    (f"{_PKG}.smooth", "_cfrc_backward"): "tree_spatial",
    (f"{_PKG}.constraint", "_efc_contact_jac_sparse"): "jqvel",
    (f"{_PKG}.set_const", "_accumulate_subtreemass"): "tree_float",
    (f"{_PKG}.smooth", "_qLD_acc"): "qld",
    (f"{_PKG}.smooth", "_solve_LD_sparse_fused"): "qld_solve",
    (f"{_PKG}.passive", "_gravity_force"): "gravcomp",   # ordered per-dof sum (coordinator 2026-11-06 06:23Z)
}
SCOPES: tuple[tuple[str, str], ...] = ((f"{_PKG}.smooth", "com_pos"), (f"{_PKG}.smooth", "crb"),
                                       (f"{_PKG}.smooth", "_rne_cfrc_backward"), (f"{_PKG}.solver", "solve"),
                                       (f"{_PKG}.constraint", "make_constraint"), (f"{_PKG}.set_const", "set_const_fixed"),
                                       (f"{_PKG}.set_const", "set_const_0"), (f"{_PKG}.smooth", "_factor_i_sparse"),
                                       (f"{_PKG}.smooth", "_solve_LD_sparse"), (f"{_PKG}.passive", "passive"))
SOLVERS = frozenset({"NEWTON"})
OPT_IN_SOLVERS = frozenset({"CG"})  # SCR-DC5-4 (ii), the CHAIN path: proven on T09, taken only when asked (allow_cg)


def _always(mjm: Any) -> bool:
    return True


def _no_tendon_armature(mjm: Any) -> bool:
    return int(mjm.ntendon) == 0 or not np.any(mjm.tendon_armature)


def actuators_per_dof(mjm: Any) -> int:
    """Most actuators whose moment row can hold one dof (a non-joint transmission counts on every dof)."""
    count = np.zeros(int(mjm.nv), dtype=np.int64)
    width = {0: 6, 1: 3, 2: 1, 3: 1}  # free, ball, slide, hinge
    for a in range(int(mjm.nu)):
        if int(mjm.actuator_trntype[a]) in (0, 1):  # JOINT, JOINTINPARENT
            j = int(mjm.actuator_trnid[a, 0])
            adr = int(mjm.jnt_dofadr[j])
            count[adr:adr + width[int(mjm.jnt_type[j])]] += 1
        else:
            count += 1
    return int(count.max()) if count.size else 0


def _no_sensor(kind: str) -> Callable[[Any], bool]:
    """True when the model has no sensor of ``mujoco.mjtSensor.<kind>``."""
    def check(mjm: Any) -> bool:
        import mujoco

        return not np.any(np.asarray(mjm.sensor_type) == int(getattr(mujoco.mjtSensor, kind)))

    return check


# Marked factories that run on the covered path without a racing float sum, each with the model condition that
# makes it so (checked against the pin's source). A factory whose condition fails is uncovered.
SAFE: dict[str, tuple[Callable[[Any], bool], str]] = {
    "solver._solve_init_jaref_kernel": (_always, "sparse J: one thread per row writes Jaref"),
    "solver._linesearch_jv_fused_kernel": (_always, "sparse J: one thread per row writes jv"),
    "solver._update_constraint_efc": (_always, "integer counters only; no changed-row list for elliptic cones"),
    "smooth._transmission": (_always, "the slot counter only moves moment rows; one thread writes each row and "
                                      "every reader walks a row in its own order"),
    "smooth._tendon_armature": (_no_tendon_armature, "returns before its atomic when tendon armature is 0"),
    "smooth._tendon_bias_coef": (_no_tendon_armature, "returns before its atomic when tendon armature is 0"),
    "smooth._tendon_bias_qfrc": (_no_tendon_armature, "returns before its atomic when tendon armature is 0"),
    "forward._qfrc_actuator": (lambda mjm: actuators_per_dof(mjm) <= 2,
                               "qfrc_actuator starts at 0 and gets at most 2 adds per dof, which commute"),
    "derivative._qderiv_actuator_passive_actuation_sparse": (lambda mjm: actuators_per_dof(mjm) <= 1,
                                                             "at most one actuator adds into each qDeriv entry"),
    "sensor._sensor_touch": (_no_sensor("mjSENS_TOUCH"), "launched over touch sensors"),
    "sensor._tendon_actuator_force": (_no_sensor("mjSENS_TENDONACTFRC"),
                                      "launched over tendon actuator-force sensors"),
    "passive._flex_bending": (_always, "launched over flex edges; the static check requires no flex"),
    "passive._flex_elasticity": (_always, "launched over flex elements; the static check requires no flex"),
    "passive._flex_passive_bend_interp": (_always, "launched over flex bend stencils; no flex"),
    "passive._spring_damper_flexedge_passive": (_always, "launched over flex edges; no flex"),
    "passive._flex_rod_bending": (_always, "launched over flex rod hinges (MuJoCo Warp 3.17); no flex"),
}
INTEGRATORS = frozenset({"IMPLICITFAST"})
_ROUTED: dict[int, weakref.ref] = {}
_REASONS: dict[int, str] = {}


def _handlers() -> dict[str, Any]:
    from alloy._vendor.mujoco_warp._src.types import vec10

    from alloy.precision.mjw_gravcomp import gravcomp_handler
    from alloy.precision.mjw_hessian import jtdaj_handler
    from alloy.precision.mjw_qld import qld_handler, solve_handler
    from alloy.precision.mjw_sums import grad_handler, jqvel_handler, qfrc_handler, tree_handler

    return {"jtdaj": jtdaj_handler, "grad": grad_handler, "qfrc": qfrc_handler,
            "tree_vec3": tree_handler(wp.vec3, False), "tree_vec10": tree_handler(vec10, True),
            "tree_spatial": tree_handler(wp.spatial_vector, False), "jqvel": jqvel_handler,
            "tree_float": tree_handler(float, False), "qld": qld_handler, "qld_solve": solve_handler,
            "gravcomp": gravcomp_handler}


def install() -> None:
    """Install the sorts, substitutions and scopes once per process (idempotent)."""
    from alloy.precision import mjw_launch
    from alloy.precision.mjw_order import install_order_hooks

    from alloy.precision.mjw_pin import PIN, changed

    moved = changed()
    if moved:
        raise PrecisionError(f"mujoco_warp differs from {PIN[:8]}, where the route was proven, in {', '.join(moved)}: "
                             "rerun P3-1 (tests/precision/test_mjw_route_gpu.py) before re-enabling")
    install_order_hooks()
    if mjw_launch.installed():
        return
    handlers = _handlers()
    for (module, factory), key in SUBSTITUTIONS.items():
        mjw_launch.substitute(module, factory, handlers[key])
    from alloy.precision.mjw_fixed import begin_solve

    for module, stage in SCOPES:
        mjw_launch.scope(module, stage, on_enter=begin_solve if stage == "solve" else None)
    log.info("mjw_route: %d substitutions in %d stages installed", len(SUBSTITUTIONS), len(SCOPES))


def static_reason(m: Any, allow_cg: bool = False) -> str | None:
    """Why the model's solver path is outside the proven one (None when inside). CG is proven but opt-in."""
    from alloy._vendor.mujoco_warp._src import types

    opt = m.opt
    solver = types.SolverType(opt.solver).name
    allowed = SOLVERS | OPT_IN_SOLVERS if allow_cg else SOLVERS
    hint = " (CG is opt-in: PrecisionConfig.mjw_route_cg)" if solver in OPT_IN_SOLVERS else ""
    checks = (
        (solver in allowed, f"solver {solver} (proven: {sorted(allowed)}){hint}"),
        (opt.cone == types.ConeType.ELLIPTIC, f"{types.ConeType(opt.cone).name} cone (only ELLIPTIC proven)"),
        (bool(m.is_sparse), "dense Jacobian (only sparse proven)"),
        (int(m.nflex) == 0, f"{int(m.nflex)} flexes (flex rows and contacts are not sorted)"),
        (not (int(opt.enableflags) & int(types.EnableBit.SLEEP)), "sleep enabled (compact solver path)"),
        (types.IntegratorType(opt.integrator).name in INTEGRATORS,
         f"integrator {types.IntegratorType(opt.integrator).name} (proven: {sorted(INTEGRATORS)})"),
    )
    bad = [why for ok, why in checks if not ok]
    return "; ".join(bad) if bad else None


def _modules() -> list[Any]:
    """The importable mujoco_warp._src modules (tests excluded)."""
    pkg = importlib.import_module(_PKG)
    out = []
    for info in pkgutil.iter_modules(pkg.__path__):
        if info.name.endswith("_test"):
            continue
        try:
            out.append(importlib.import_module(f"{_PKG}.{info.name}"))
        except ImportError as e:  # optional dependencies (render, cli)
            log.debug("mjw_route: skip %s: %s", info.name, e)
    return out


def marked_factories() -> dict[str, Any]:
    """Every mujoco_warp factory with a ``deterministic`` switch, by ``module.name``."""
    out: dict[str, Any] = {}
    for mod in _modules():
        for name, f in vars(mod).items():
            if not callable(f) or getattr(inspect.unwrap(f), "__module__", None) != mod.__name__:
                continue
            try:
                params = inspect.signature(f).parameters
            except (TypeError, ValueError):
                continue
            if "deterministic" in params:
                out[f"{mod.__name__.rsplit('.', 1)[1]}.{name}"] = inspect.unwrap(f)
    return out


def _bindings(targets: dict[str, Any]) -> list[tuple[Any, str, str]]:
    """(module, attribute, key) for every module attribute bound to a marked factory (wrapped or not)."""
    by_func = {id(f): k for k, f in targets.items()}
    out = []
    for mod in _modules():
        for name, f in list(vars(mod).items()):
            if callable(f) and id(inspect.unwrap(f)) in by_func:
                out.append((mod, name, by_func[id(inspect.unwrap(f))]))
    return out


@contextlib.contextmanager
def recording() -> Iterator[set[str]]:
    """Collect the marked factories called inside the block (every binding of each factory is wrapped)."""
    calls: set[str] = set()
    bound = _bindings(marked_factories())
    saved = [(mod, name, getattr(mod, name)) for mod, name, _ in bound]
    for (mod, name, key), (_, _, orig) in zip(bound, saved):
        def record(*a: Any, _orig: Any = orig, _key: str = key, **k: Any) -> Any:
            calls.add(_key)
            return _orig(*a, **k)

        setattr(mod, name, record)
    try:
        yield calls
    finally:
        for mod, name, orig in reversed(saved):
            setattr(mod, name, orig)


def called_factories(solver: Any) -> set[str]:
    """Marked factories one eager step of a 1-world scratch Data of ``solver``'s model calls."""
    from alloy._vendor import mujoco_warp as mjw

    with recording() as calls:
        d = solver.mjw_data
        scratch = mjw.put_data(solver.mj_model, solver.mj_data, nworld=1, njmax=int(d.njmax),
                               nconmax=max(1, int(d.naconmax) // max(int(d.nworld), 1)))
        mjw.step(solver.mjw_model, scratch)
        wp.synchronize_device(scratch.qpos.device)
    return calls


def uncovered(solver: Any, calls: set[str]) -> str | None:
    """Why ``calls`` holds a racing factory the route neither replaces nor proves safe (None when all are)."""
    replaced = {f"{mod.rsplit('.', 1)[1]}.{fac}" for mod, fac in SUBSTITUTIONS}
    bad = []
    for key in sorted(calls - replaced):
        rule = SAFE.get(key)
        if rule is None:
            bad.append(f"{key} (racing float sums, not replaced)")
        elif not rule[0](solver.mj_model):
            bad.append(f"{key} (safe only when: {rule[1]})")
    return "uncovered racing kernels: " + "; ".join(bad) if bad else None


def coverage_reason(solver: Any, allow_cg: bool = False) -> str | None:
    """None when every racing sum on ``solver``'s step is replaced or proven safe, else why not."""
    why = static_reason(solver.mjw_model, allow_cg)
    return why if why is not None else uncovered(solver, called_factories(solver))


def _prepare(solver: Any) -> None:
    """Allocate every scratch buffer up front, so the first step can be graph-captured."""
    from alloy.precision import mjw_order, mjw_rows
    from alloy.precision.mjw_fixed import register
    from alloy.precision.mjw_qld import prepare_qld
    from alloy.precision.mjw_sums import prepare_tree

    m, d = solver.mjw_model, solver.mjw_data
    register(m, d)
    prepare_qld(m)
    prepare_tree(m, (d.subtree_com, d.crb, d.cfrc_int, m.body_subtreemass))
    if int(d.naconmax):
        mjw_order._contact_scratch(m, d)
    mjw_rows._scratch(d)
    from alloy.precision import mjw_chol

    if mjw_chol.install():
        mjw_chol.prepare(m, d)


def _recompute_constants(solver: Any) -> None:
    """Recompute the model constants Newton's build left with racing sums (``set_const_fixed`` / ``set_const_0``:
    body_subtreemass, invweight0, actuator_acc0, meaninertia) now that the model is enabled. Uses Newton's public
    ``notify_model_changed``; a solver without it (tests over pure MuJoCo Warp) calls MuJoCo Warp's set_const."""
    notify = getattr(solver, "notify_model_changed", None)
    if notify is not None:
        from alloy._vendor import newton

        notify(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
        return
    from alloy._vendor import mujoco_warp as mjw

    mjw.set_const(solver.mjw_model, solver.mjw_data)


def enable_route(solver: Any, allow_cg: bool = False) -> str | None:
    """Enable the route on a built SolverMuJoCo (before its first step or capture). None = enabled, else the
    reason it stays NONE."""
    from alloy.precision import mjw_launch

    if os.environ.get("ALLOY_MJW_ROUTE", "on").lower() in ("0", "off", "false"):
        return _refuse(solver, "ALLOY_MJW_ROUTE=off")
    if getattr(solver, "mjw_model", None) is None or not hasattr(solver.mjw_model.opt, "deterministic"):
        return _refuse(solver, "no MuJoCo Warp backend with DeterminismType")
    if getattr(solver, "use_mujoco_cpu", False):   # SCR-WP8-6's MjcCoreSolver: its MuJoCo Warp model is never stepped
        return _refuse(solver, "steps through MuJoCo C (use_mujoco_cpu): no MuJoCo Warp step to order")
    try:
        install()
        why = coverage_reason(solver, allow_cg)
        if why is not None:
            return _refuse(solver, why)
        _prepare(solver)
    except PrecisionError as e:
        return _refuse(solver, str(e))
    solver.mjw_model.opt.deterministic = SORT_BITS
    mjw_launch.enable(solver.mjw_model)
    with recording() as calls:
        _recompute_constants(solver)
    why = uncovered(solver, calls)
    if why is not None:
        mjw_launch.disable(solver.mjw_model)
        solver.mjw_model.opt.deterministic = 0
        return _refuse(solver, f"model constants (set_const): {why}")
    _ROUTED[id(solver)] = weakref.ref(solver)
    _REASONS.pop(id(solver), None)
    log.info("mjw_route: SolverMuJoCo enabled (%d worlds, nv %d)", int(solver.mjw_data.nworld),
             int(solver.mjw_model.nv))
    return None


def _refuse(solver: Any, why: str) -> str:
    remember(_REASONS, id(solver), solver, why)
    log.info("mjw_route: SolverMuJoCo stays NONE: %s", why)
    return why


def route_enabled(solver: Any) -> bool:
    """True when ``enable_route`` enabled this solver and its model still carries the sort bits."""
    ref = _ROUTED.get(id(solver))
    from alloy.precision import mjw_launch

    return (ref is not None and ref() is solver and mjw_launch.is_enabled(solver.mjw_model)
            and int(solver.mjw_model.opt.deterministic) & SORT_BITS == SORT_BITS)


def route_reason(solver: Any) -> str | None:
    """Why ``enable_route`` refused this solver (None if it was never tried or it is enabled)."""
    return _REASONS.get(id(solver))


def uninstall() -> None:
    """Remove every hook the route installed and forget enabled solvers (tests; a live solver goes back to NONE)."""
    from alloy.precision import mjw_fixed, mjw_launch, mjw_qld, mjw_sums
    from alloy.precision.mjw_order import uninstall_order_hooks

    for ref in list(_ROUTED.values()):
        solver = ref()
        if solver is not None:
            solver.mjw_model.opt.deterministic = 0
    mjw_launch.uninstall()
    uninstall_order_hooks()
    mjw_fixed.release()
    mjw_sums.release()
    mjw_qld.release()
    _ROUTED.clear()
    _REASONS.clear()
