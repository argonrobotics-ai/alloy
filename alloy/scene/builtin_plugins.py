# SPDX-License-Identifier: Apache-2.0
"""SolverCapability registrations for stock Newton solvers (SPEC §3.1 registry, D9, D13, D16).

Facts per solver (newton@ac7f7830):
- friction mix: MuJoCo max of geom_friction unless geom_priority (solvers/mujoco/kernels.py:123-170); Featherstone and
  semi-implicit arithmetic mean (semi_implicit/kernels_contact.py:430-450); XPBD mean (xpbd/kernels.py:2632-2650);
  VBD geometric (vbd/rigid_vbd_kernels.py:1063-1081); Kamino average (kamino config friction_mix_mode default).
- per-contact friction scale: read by MuJoCo (Newton-contacts mode, kernels.py:585-598), Featherstone and
  semi-implicit (kernels_contact.py:452-458); ignored by XPBD, VBD and Kamino.
- per-contact damping (restitution path): MuJoCo (kernels.py:663-685), Featherstone/semi-implicit (contact kd > 0).
- determinism: MuJoCo, VBD, XPBD, Featherstone and semi-implicit take `deterministic=wp.DeterministicMode` (Warp
  lowering, WARP_LOWERING); implicit MPM, Kamino and Style3D have no deterministic path (NONE). No stock solver is
  NATIVE (D9).
- coupling: Style3D is SolverBase only (style3d/solver_style3d.py:36), so it is not a coupled entry (D16).
"""

from __future__ import annotations

import importlib
import logging
import os
import re
from collections.abc import Mapping
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy._vendor.newton.solvers import SolverBase

from alloy.core.errors import RouteError
from alloy.core.newton_compat import ModelView
from alloy.core.protocols import DetCapability
from alloy.core.registry import SolverCapability, load_plugins, register_solver
from alloy.route.types import Route
from alloy.scene.emitters.dispatch import ROUTE_EMITTER

log = logging.getLogger("alloy.scene")

SOLREF_DAMPRATIO = ("a solref-type value with a positive time constant has dampratio 0 or below, which MuJoCo's "
                    "positive form cannot use: its stiffness goes as 1 / dampratio^2 (a one-value solref is read "
                    "here as dampratio 0, not as MuJoCo's 1)")
SOLREF_FIX = ('write a dampratio above 0, such as solref="0.02 1" (MuJoCo\'s default); for a stiff or undamped '
              'contact or limit use the direct form, solref="-k -b" (stiffness k in N/m, damping b in N s/m, '
              'both written negative; "-k 0" is undamped)')
RIGID_OWNS = frozenset({"body", "joint", "shape"})
PARTICLE_OWNS = frozenset({"particle", "tri", "tet", "edge"})


def quiet_flagged_overflow(solver: Any) -> None:
    """Turn off MuJoCo Warp's overflow prints for the overflows Alloy's health flags from the same per-world word
    (CONTACT_OVERFLOW, ROW_OVERFLOW, UNCONVERGED: ``alloy.precision.health_adapters._mjw_bits``). Each overflowing
    kernel printed per step: an outside tester's 25 kinematic boxes on the ground gave 2,464 lines in 20 steps.
    The word is still written (mujoco_warp forward.py), so ``sim.health()`` and ``sim.check()`` still report it."""
    m = getattr(solver, "mjw_model", None)
    if m is None:
        return
    from alloy.precision.health_adapters import _mjw_bits  # noqa: PLC0415 - the bits health reads

    flagged = 0
    for bits in _mjw_bits():
        flagged |= int(bits)
    m.opt.warn_overflow = int(m.opt.warn_overflow) & ~flagged


def _mujoco(view: ModelView, kw: Mapping[str, Any]) -> SolverBase:
    """SolverMuJoCo; a solref-type value Newton read as dampratio 0 raises first (`check_solref`, SCR-SC6-2). Right
    after construction, before any step or graph capture: box-box pairs go to MuJoCo Warp's
    primitive collider (`alloy.contact.mjw_boxbox`, SCR-WP3-9, BUL-0011) with the box-box line contacts
    (`alloy.contact.mjw_boxbox_line`, BUL-0014; `line_default` of the model's objects, or the entry's LINE_KEY),
    explicit pair rows are put in MuJoCo's
    compiled order (`alloy.contact.mjw_pairs`, Newton pair misorder, WP0 ruling 2026-11-03), a tree weld's parent
    bodies take the contact weight they have without the chain (`alloy.scene.tree_invweight`, SCR-DC3-14), pair
    rows touching
    armature or drawn-mass bodies are sized from the built model (`alloy.contact.pair_fit`, SCR-DC3-5 / SCR-DC3-8),
    light dynamic MJCF bodies take the native contact rule (`alloy.contact.light_body`, coordinator ruling 2026-11-05
    07:19Z; the entry's LIGHT_KEY False turns it off), then a scene's noslip request (`alloy.scene.noslip_route`,
    SCR-WP3-4) is installed, or the opt-in static hold (`alloy.contact.noslip_hold`, noslip over every contact; rk4
    entries keep the creep), which covers it. The compensated float32 joint coordinates (`alloy.scene.coordinate_carry`,
    SCR-SC10-3, default-on since the 2026-11-05 22:33Z ruling) are already on the solver from its construction; the
    entry's CARRY_KEY False takes them off (A/B runs), True asks for them explicitly (rk4 entries refuse it). The
    entry's CCD_KEY (slots per world) sizes MuJoCo Warp's CCD scratch pool right after construction
    (`alloy.precision.mjw_ccd`, SCR-WP8-7 M1); without it MuJoCo Warp's default (naconmax slots) stays.
    CPU branch (SCR-WP8-6): an entry on a CPU model whose `alloy:cpu_route` (or `ALLOY_CPU_ROUTE`) is "mjc", the
    default since SCR-WP8-8, gets the CPU route on the MuJoCo C core (`alloy.route.mjc_core`, built by
    `alloy.route.mjc_core_factory`), unless the factory refuses it (logged); "warp" keeps MuJoCo Warp on CPU (MuJoCo C
    sizes its own CCD scratch: CCD_KEY is popped before that branch and not passed on)."""
    from alloy.contact import chain_grip
    from alloy.contact.light_body import GAINS_KEY, LIGHT_KEY, install_light_body
    from alloy.contact.mjw_boxbox_line import LINE_KEY, line_default, model_objects
    from alloy.contact.mjw_boxbox_reduce import REDUCE_KEY
    from alloy.contact.mjw_friction_cap import CAP_KEY, default_cap, install_friction_cap
    from alloy.contact.noslip_hold import HOLD_KEY, hold_request, install_hold
    from alloy.precision.mjw_ccd import CCD_KEY, set_ccd_slots
    from alloy.scene.coordinate_carry import CARRY_KEY, install_coordinate_carry, uninstall
    from alloy.scene.noslip_route import NOSLIP_KEY, install_entry_noslip
    from alloy.scene.tree_invweight import install_tree_invweight
    from alloy.scene.tree_weld import rigid_armature_off

    from alloy.route.mjc_core_config import MJC, requested_route
    from alloy.route.mjc_core_factory import EntryRequests, build_cpu_route

    check_solref(view)
    rigid_armature_off(view)              # SCR-DC3-14: after any scene-level armature override, before the solver
    kw = dict(kw)
    cpu_route = requested_route(kw)       # SCR-WP8-6: "mjc" = the CPU route on the MuJoCo C core (CPU models only)
    noslip, hold = kw.pop(NOSLIP_KEY, None), hold_request(kw.pop(HOLD_KEY, None))
    carry = kw.pop(CARRY_KEY, None)
    line = kw.pop(LINE_KEY, None)
    line = line_default(model_objects(view)) if line is None else bool(line)
    reduce4 = bool(kw.pop(REDUCE_KEY, False))   # SPEED-S1: the fast rigid mode's 4-point box-box face pairs
    cap = kw.pop(CAP_KEY, None)                 # RIGID-DEFAULT (b): elliptic friction rows' pull cap, m/s2
    light, gains = kw.pop(LIGHT_KEY, None), kw.pop(GAINS_KEY, None)    # gains: F6 R1's rk4-gains flag
    if cap is None and gains and kw.get("integrator") != "rk4":   # R1 / step 2c entries on implicitfast take the cap
        cap = default_cap()                     # (ruling 16:32Z; the precision profile's rk4 switch-back keeps none)
    ccd = kw.pop(CCD_KEY, None)
    grip = kw.pop(chain_grip.KEY, None)   # SCR-WP3-CHAINGRIP: impratio 300 on MuJoCo C, 30 + a pivot floor on Warp
    if kw.get("integrator") == "rk4" and (kw.get("enable_sleeping") or noslip is not None):
        log.warning("rk4 has no sleeping or noslip: this MuJoCo entry runs implicitfast")
        kw["integrator"] = "implicitfast"
    if kw.get("integrator") == "rk4" and kw.get("use_mujoco_contacts") is False:
        # PHYSA-1: Newton's pipeline builds the contacts once per step, so rk4's four stages would see every contact
        # at the step's first distance. That integrates the contact spring explicitly and adds energy at each bounce
        # (a 10 cm cube dropped 1 cm rose to 61 mm and kept bouncing; tests/contact/test_newton_contacts_rest.py).
        log.warning("rk4 needs MuJoCo's own contacts (rebuilt at each stage): on Newton's contact pipeline this "
                    "MuJoCo entry runs implicitfast, whatever integrator its route reasons name")
        kw["integrator"] = "implicitfast"
    if cpu_route == MJC:                  # None: refused (logged), built below as before
        solver = build_cpu_route(view, chain_grip.route_kwargs(kw, grip), EntryRequests(noslip, hold, carry, line,
                                                                                        light, gains))
        if solver is not None:
            return solver
    grip = chain_grip.warp_options(grip, kw)  # MuJoCo Warp: only with mjw_chol's pivot floor
    # free bodies (rk4, and F6 R1 entries on implicitfast, step 2c's "own" ones included: `gains`): no float32 state
    # round trip per step (mujoco_sync). Without it an implicitfast cube tossed at 10 rad/s on MuJoCo Warp lost
    # |dL|/L 4.8e-4 in 2 s (rk4 here 7.9e-6, C 1.1e-6). An entry that also asks for noslip keeps SolverMuJoCo, which
    # installs it below.
    if kw.get("integrator") == "rk4" or (gains and noslip is None):
        from alloy.scene.mujoco_sync import SyncedMuJoCo

        solver = SyncedMuJoCo(view, **chain_grip.route_kwargs(kw, grip))
        quiet_flagged_overflow(solver)
        chain_grip.mark_pivot_floor(solver, grip)
        if ccd is not None:
            set_ccd_slots(solver, ccd)
        solver = _pair_fit(install_tree_invweight(_pair_order(_primitive_boxbox(solver, line, reduce4))))
        install_friction_cap(solver, cap)
        chain_grip.install_anchor(solver, grip)   # after the cap: the cap clamps the anchored rows too
        install_light_body(solver, None if light is None else bool(light), rk4_gains=gains)
        if hold is not None:
            install_hold(solver, *hold)       # logs that rk4 keeps the creep (R1 entries: no noslip either)
        if carry:
            install_coordinate_carry(solver)  # raises CarryError: no per-step round trip on SyncedMuJoCo
        return solver
    solver = newton.solvers.SolverMuJoCo(view, **chain_grip.route_kwargs(kw, grip))
    quiet_flagged_overflow(solver)
    chain_grip.mark_pivot_floor(solver, grip)
    if ccd is not None:
        set_ccd_slots(solver, ccd)
    if carry is False:
        uninstall(solver)                     # before anything else wraps the solver
    solver = _pair_fit(install_tree_invweight(_pair_order(_primitive_boxbox(solver, line, reduce4))))
    install_friction_cap(solver, cap)
    chain_grip.install_anchor(solver, grip)
    install_light_body(solver, None if light is None else bool(light), rk4_gains=gains)
    if hold is not None:
        install_hold(solver, *hold)
    elif noslip is not None:
        install_entry_noslip(solver, noslip)
    if carry:
        install_coordinate_carry(solver)
    return solver


def check_solref(model) -> None:
    """SCR-SC6-2: MuJoCo reads a one-value solref-type MJCF attribute as (t, 1); Newton (pin ac7f7830) reads the geom
    `solref` and joint `solreflimit` shorthand as (t, 0) on its `mujoco:` attribute path, an undamped constraint (a
    UR10e + 2F-85 world froze or flailed, SC6). A positive time constant with dampratio <= 0 raises RouteError naming
    each joint/geom, whether one value or two were written (the model cannot tell them apart), with its own fix
    (SAMPLE-CLASSIC C3); nothing is padded. Direct (t <= 0) and unauthored (0, 0) values pass."""
    bad = [f"{name} on {_owner(model, name, int(i))} = ({v[i, 0]:g}, {v[i, 1]:g})"
           for name, v in _solref_arrays(model) for i in np.flatnonzero((v[:, 0] > 0.0) & (v[:, 1] <= 0.0))]
    if bad:
        more = f" (+{len(bad) - 8} more)" if len(bad) > 8 else ""
        err = RouteError(f"{SOLREF_DAMPRATIO}: {'; '.join(bad[:8])}{more}")
        err.fix = SOLREF_FIX     # the API shows it as the error's fix (alloy.api.errors.wrap), not RouteError's
        raise err


def _solref_arrays(model):
    """(name, [n, 2] values) of every `mujoco:*solref*` (timeconst, dampratio) attribute of `model` (or a view)."""
    ns = getattr(model, "mujoco", None)
    if ns is None:
        return
    names = {k for src in (ns, getattr(ns, "_parent_namespace", None)) if src is not None for k in vars(src)}
    for name in sorted(k for k in names if "solref" in k and "gain" not in k):
        a = getattr(ns, name, None)
        if isinstance(a, wp.array) and a.size:
            v = a.numpy().reshape(a.shape[0], -1)
            if v.shape[1] == 2:
                yield name, v


def _owner(model, name: str, i: int) -> str:
    """The joint/geom/body (or custom row) that row `i` of `mujoco:<name>` belongs to."""
    try:
        freq = model.get_attribute_frequency(f"mujoco:{name}")
    except KeyError:
        return f"row {i}"
    F = newton.Model.AttributeFrequency
    if freq == F.SHAPE:
        return f"geom {model.shape_label[i]}"
    if freq == F.JOINT_DOF:
        j = int(np.searchsorted(model.joint_qd_start.numpy(), i, side="right")) - 1
        return f"joint {model.joint_label[j]} (dof {i})"
    if freq == F.JOINT:
        return f"joint {model.joint_label[i]}"
    if freq == F.BODY:
        return f"body {model.body_label[i]}"
    return f"{freq} row {i}"


def _primitive_boxbox(solver: SolverBase, line: bool = False, reduce4: bool = False) -> SolverBase:
    """SCR-WP3-9: MuJoCo Warp's convex box-box path flips resting contacts (BUL-0011); NATIVECCD off routes box-box
    pairs to the primitive collider (MuJoCo C's); ``line``: the box-box line contacts too (BUL-0014); ``reduce4``:
    face pairs at 4 points (the fast rigid mode, SPEED-S1). Logs why when nothing changes; returns the solver."""
    from alloy.contact.mjw_boxbox import use_primitive_boxbox

    why = use_primitive_boxbox(solver, line, reduce4)
    if why is not None:
        log.debug("primitive box-box collider not set: %s", why)
    return solver


def _pair_order(solver: SolverBase) -> SolverBase:
    """Newton copies pair row j onto MuJoCo pair j while MuJoCo reorders pairs: rows go to MuJoCo's order
    (`alloy.contact.mjw_pairs`). Logs why when nothing changes; returns the solver."""
    from alloy.contact.mjw_pairs import use_mujoco_pair_order

    why = use_mujoco_pair_order(solver)
    if why is not None:
        log.debug("pair rows not reordered: %s", why)
    return solver


def _pair_fit(solver: SolverBase) -> SolverBase:
    """Explicit pair rows touching a body with joint armature take MuJoCo's own contact weight (SCR-DC3-5 part 2);
    rows of drawn-mass bodies take each world's drawn mass, refit on body-inertia notifies (SCR-DC3-8)."""
    from alloy.contact.pair_fit import fit_pair_rows

    fit_pair_rows(solver)
    return solver


def _featherstone(view: ModelView, kw: Mapping[str, Any]) -> SolverBase:
    return newton.solvers.SolverFeatherstone(view, **dict(kw))


def _semi_implicit(view: ModelView, kw: Mapping[str, Any]) -> SolverBase:
    return newton.solvers.SolverSemiImplicit(view, **dict(kw))


def _xpbd(view: ModelView, kw: Mapping[str, Any]) -> SolverBase:
    return newton.solvers.SolverXPBD(view, **dict(kw))


def _vbd(view: ModelView, kw: Mapping[str, Any]) -> SolverBase:
    """Alloy's VBD path (`alloy.solvers.vbd`, SCR-WP11-2, owner WP11): the entry kwarg `coarse_step` picks
    SolverVBDMomentum (exact coarse translation step, momentum-exact on free components) or the stock SolverVBD."""
    from alloy.scene import ortho_membrane
    from alloy.solvers.vbd import make_vbd

    return ortho_membrane.install(make_vbd(view, kw), view)    # woven cards' orthotropic membrane, if any


def scene_worlds() -> int:
    """The whole scene's world count when this process builds one block of it (CPU mode A sets
    ``ALLOY_SCENE_WORLDS`` in its workers), else 0. A subprocess a worker starts inherits it."""
    return int(os.environ.get("ALLOY_SCENE_WORLDS", "0") or 0)


def _mpm(view: ModelView, kw: Mapping[str, Any]) -> SolverBase:
    """Implicit MPM with one grid per world (``separate_worlds``, SCR-SOFT-2-2) unless the entry sets it. Newton's
    default shares one grid among all worlds. Alloy's worlds overlap in space, so in a shared grid every world's
    particles pooled their mass and stress in the same cells and moved alike, and every world's body collider acted
    on all of them: a sand fill fell through a free box at 4 worlds and rested at 1. Newton ignores it at 1 world.

    The grid is sparse unless the entry sets ``grid_type`` or ``ALLOY_MPM_GRID=dense`` / ``grid_override`` asks for
    dense (SCR-SOFT-2-5, ``alloy.integration.mpm_grid``: each grid mishandles a different wall position). A dense grid
    over its budget at the start is built sparse; one that outgrows it mid-run switches to sparse with one world and
    raises with separate worlds. A sparse entry resting on the ground warns once."""
    from alloy.integration import mpm_grid as G  # noqa: PLC0415
    from alloy.solvers.mpm_rheology.solver import SolverRheologyMPM, hb_particles  # noqa: PLC0415
    from alloy.solvers.plastic import mpm_solid  # noqa: PLC0415

    cls = newton.solvers.SolverImplicitMPM
    from alloy.integration import mpm_worlds  # noqa: PLC0415
    mpm_worlds.install()                        # each world's solve independent of the others (SCR-WP13-MPMISO-1)
    from alloy.integration import native_guard, sparse_guard  # noqa: PLC0415
    sparse_guard.install()                      # GPU full in warp.fem's sparse build: an error, not CUDA 700 (Nov 8)
    native_guard.install()                      # ... and in its grid builds, sorts, encodes (SCR-WP12-3 follow-up 1)
    from alloy.integration import mpm_solve  # noqa: PLC0415
    cfg = {"separate_worlds": True, **mpm_solve.solve_kwargs(), **dict(kw)}  # the rheology tolerance (SCR-WP12-3 (3))
    if "tolerance" in cfg:                      # an entry's own value is held to the ruled range too
        mpm_solve.check(cfg["tolerance"], "the MPM entry's solver kwarg tolerance")
    worlds = int(view.world_count)
    # a worker of CPU mode A builds a block of a larger scene: its MPM is built as the whole scene's (dense grid on the
    # CPU, no mid-run fallback), so a world steps the same in any block (alloy.parallel.cpu_worlds)
    multi = cfg["separate_worlds"] and max(worlds, scene_worlds()) > 1
    cpu_worlds = multi and not wp.get_device(view.device).is_cuda
    if cpu_worlds:
        cfg["grid_type"] = "dense"              # Newton builds per-world sparse grids (NanoVDB) on CUDA only
    else:
        cfg.setdefault("grid_type", mpm_solid.grid_type(view, G.default_grid_type()))   # j2 solid: dense on a CPU
    config = cls.Config(**cfg)
    fallback = None if cpu_worlds else G.dense_start_fallback(view, config, worlds, "mpm")
    if fallback is not None:
        config = cls.Config(**{**cfg, "grid_type": "sparse"})
    hb = len(hb_particles(view))                # Herschel-Bulkley pastes: the subclass, else the stock solver
    solver = _new_mpm_solver(SolverRheologyMPM if hb else cls, view, config)
    # mid-run, a dense grid switches to sparse only with one world: with separate worlds Newton's next step failed
    # to rebind the collision fields on CUDA (WP13's L4 log, 64 worlds, 11-07)
    why = ("no sparse grid to fall back to (separate worlds on the CPU)" if cpu_worlds else
           "a dense grid of separate worlds does not switch to sparse mid-run (Newton's rebind fails on CUDA)")
    G.guard_dense_grid(solver, int(view.particle_count), "mpm", can_fall_back=not multi, fallback=fallback,
                       no_fallback_why=why)
    G.warn_sparse_ground(solver, view)
    mpm_solid.install_notes(solver, view, config.grid_type)
    return solver


MPM_SHAPE_FIX = ("give every object that can touch the sand, soil or paste a box, sphere, capsule, cylinder, cone or "
                 "mesh shape (add_mesh with collision='convex', 'decompose', 'sdf' or 'mesh'); heightfields, "
                 "ellipsoids and Gaussian splats are not MPM colliders")


def _new_mpm_solver(cls: type, view: ModelView, config: Any) -> SolverBase:
    """``cls(view, config)``. A collision shape Newton cannot mesh as an MPM collider (its NotImplementedError "Shape
    type N not supported") becomes a RouteError naming the shape type, with ``MPM_SHAPE_FIX`` as its fix:
    the API raises it as a BuildError. Example: ``_new_mpm_solver(SolverImplicitMPM, view, config)``."""
    try:
        return cls(view, config)
    except NotImplementedError as exc:
        found = re.search(r"Shape type (\d+)", str(exc))
        if found is None:
            raise
        kind = int(found.group(1))
        name = newton.GeoType(kind).name if kind in {int(g) for g in newton.GeoType} else str(kind)
        err = RouteError(f"sand, soil or paste (MPM) cannot collide with this scene's {name.lower()} shape")
        err.fix = MPM_SHAPE_FIX
        raise err from exc


def _mpm_attributes(builder: newton.ModelBuilder) -> None:
    """Newton's ``mpm:*`` attributes plus Alloy's ``rheo:*`` (Herschel-Bulkley K and n, default 0 = none)."""
    from alloy.solvers.mpm_rheology.config import register_rheology_attributes  # noqa: PLC0415

    newton.solvers.SolverImplicitMPM.register_custom_attributes(builder)
    register_rheology_attributes(builder)


def _kamino(view: ModelView, kw: Mapping[str, Any]) -> SolverBase:
    cls = newton.solvers.SolverKamino
    return cls(view, cls.Config(**dict(kw)) if kw else None)


def _style3d(view: ModelView, kw: Mapping[str, Any]) -> SolverBase:
    return newton.solvers.SolverStyle3D(view, **dict(kw))


def _routes(*routes: Route) -> frozenset[str]:
    return frozenset(str(r) for r in routes)


def builtin_capabilities() -> tuple[SolverCapability, ...]:
    """The stock Newton solvers Alloy routes to, with the facts in the module docstring."""
    S = newton.solvers
    rigid = (Route.RIGID, Route.RIGID_COMPLIANT)
    return (
        SolverCapability(
            name="mujoco", solver_cls=S.SolverMuJoCo, factory=_mujoco,
            routes=_routes(*rigid, Route.RIGID_PILE, Route.CHAIN, Route.PANELS, Route.GRAIN_RIGID),
            owns=RIGID_OWNS, coupling=True, uniform_worlds=True, per_contact_friction=True,
            determinism=DetCapability.WARP_LOWERING, friction_mix="max", restitution_path=True,
            emitter=ROUTE_EMITTER, register_attributes=S.SolverMuJoCo.register_custom_attributes,
            # kinematic_switchable_roots (Newton 1.8, default True) gives every movable root an inactive weld so it
            # can turn kinematic at runtime; Alloy never does (kinematic objects are so from the build, and keep
            # theirs), so the model stays free of unused welds, as on Newton 1.7
            default_kwargs={"solver": "newton", "integrator": "implicitfast", "cone": "elliptic",
                            "kinematic_switchable_roots": False}),
        SolverCapability(
            name="featherstone", solver_cls=S.SolverFeatherstone, factory=_featherstone, routes=_routes(*rigid, Route.CHAIN), owns=RIGID_OWNS,
            coupling=True, uniform_worlds=False, per_contact_friction=True, determinism=DetCapability.WARP_LOWERING,
            friction_mix="mean", restitution_path=True, emitter=ROUTE_EMITTER),
        SolverCapability(
            name="semi_implicit", solver_cls=S.SolverSemiImplicit, factory=_semi_implicit, routes=_routes(*rigid), owns=RIGID_OWNS, coupling=True,
            uniform_worlds=False, per_contact_friction=True, determinism=DetCapability.WARP_LOWERING,
            friction_mix="mean", restitution_path=True, emitter=ROUTE_EMITTER),
        SolverCapability(
            name="xpbd", solver_cls=S.SolverXPBD, factory=_xpbd, routes=_routes(*rigid), owns=RIGID_OWNS | PARTICLE_OWNS, coupling=True,
            uniform_worlds=False, per_contact_friction=False, determinism=DetCapability.WARP_LOWERING,
            friction_mix="mean", emitter=ROUTE_EMITTER),
        SolverCapability(
            name="vbd", solver_cls=S.SolverVBD, factory=_vbd, routes=_routes(Route.CLOTH_VBD, Route.SOFT_VBD, Route.ROD_VBD),
            owns=PARTICLE_OWNS | frozenset({"rod_joint"}), coupling=True, uniform_worlds=False,
            per_contact_friction=False, determinism=DetCapability.WARP_LOWERING, friction_mix="geometric",
            emitter=ROUTE_EMITTER, default_kwargs={"iterations": 10}),
        SolverCapability(
            name="mpm", solver_cls=S.SolverImplicitMPM, factory=_mpm, routes=_routes(Route.GRAIN_MPM, Route.PASTE_MPM, Route.LIQUID_MPM),
            owns=frozenset({"particle"}), coupling=True, uniform_worlds=False, per_contact_friction=False,
            determinism=DetCapability.NONE, friction_mix="none", emitter=ROUTE_EMITTER,
            register_attributes=_mpm_attributes),
        SolverCapability(
            name="kamino", solver_cls=S.SolverKamino, factory=_kamino, routes=_routes(Route.RIGID_PILE), owns=RIGID_OWNS, coupling=True,
            uniform_worlds=False, per_contact_friction=False, determinism=DetCapability.NONE, friction_mix="mean",
            emitter=ROUTE_EMITTER, register_attributes=S.SolverKamino.register_custom_attributes),
        SolverCapability(
            name="style3d", solver_cls=S.SolverStyle3D, factory=_style3d, routes=_routes(Route.CLOTH_STYLE3D), owns=PARTICLE_OWNS,
            coupling=False, uniform_worlds=False, per_contact_friction=False, determinism=DetCapability.NONE,
            friction_mix="none", emitter=ROUTE_EMITTER,
            register_attributes=S.SolverStyle3D.register_custom_attributes),
    )


def register_builtin_solvers() -> None:
    """Register every stock capability, then every `alloy.solvers.<name>` plug-in (SCR-WP9-5: `load_plugins`).
    Idempotent: re-registering an equal capability is a no-op."""
    for cap in builtin_capabilities():
        register_solver(cap)
    log.debug("registered %d builtin solver capabilities", len(builtin_capabilities()))
    for name in load_plugins():
        _reregister(name)


def _reregister(package: str) -> None:
    """A plug-in registers on first import only; inside an `isolated_registry` the import is cached, so call its
    `register()` (on the package or its `plugin` module) when it exposes one."""
    for mod_name in (package, f"{package}.plugin"):
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        fn = getattr(mod, "register", None)
        if callable(fn):
            fn()
            return
