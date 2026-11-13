# SPDX-License-Identifier: Apache-2.0
"""EntrySpec creation (SPEC §3.4, D4, D21): one entry per distinct solver capability; STATIC shapes unowned.

Global ids after `replicate` are world-major: global = template id + w·(template count of that kind).
Target dt per entry on the D4 grid (control_dt/2^j): rigid 1/600 s (→ 1.25 ms at control_dt 0.02), precision
pairs from the L2 rule (precision_pairs.py), MPM min(2 ms, 0.5·voxel/v_max), SPH 0.2·2r/v_max, VBD the rigid target.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from alloy._vendor import newton

from alloy.cards import types as ct
from alloy.contact import chain_grip
from alloy.core.errors import AssemblyError
from alloy.core.registry import solver as capability
from alloy.core.timebase import snap_to_grid
from alloy.parallel.types import PackSpec
from alloy.route.types import Route
from alloy.scene import cloth_accuracy, stiff_sheet
from alloy.scene.fast_profile import FAST_EFFECT, FAST_IMPRATIO, NEWTON_EFFECT
from alloy.scene.held_route import HELD, HELD_EFFECT, HELD_REASON
from alloy.scene.integrator_route import effect_integrator
from alloy.scene.mjcf_options import MjcfOptions
from alloy.scene.template import KINDS, TemplateObject
from alloy.scene.types import EntrySpec
from alloy.scene.vbd_contact import emitted_radii, vbd_self_contact
from alloy.solvers.plastic import mpm_solid
from alloy.solvers.vbd.cloth_profile import ACCURATE_EFFECT, ACCURATE_ITERATIONS


RIGID_TARGET = 1.0 / 600.0
MPM_DT_MAX = 2.0e-3
SPH_CFL = 0.2
SUBSTEP_LADDER_MAX = 64          # ClockPolicy.substep_ladder max (D21: adaptive entries built at the max rung)
CG_ITERATIONS = 200
# A vbd entry whose objects are all rod_vbd rods (S4, 2026-11-09): the rod solve converges each chain directly, and
# since F3's sticky contacts the R4 grip law passes at 4 (ramp end 1.0002, run-on +0.0002, torque 1.0195 against a
# 3 % bound): 1.7x beside a robot (161 -> 95 s per simulated second, CPU). Entries with cloth or soft bodies keep
# their count; in a rod-only entry this also overrides the opt-in accurate cloth profile's, which is cloth-only
ROD_ITERATIONS = 4
PRECISION_IMPRATIO = 300.0       # §3.5 creep rule (elliptic cone)
FILL_IMPRATIO = 300.0            # SCR-WP3-10: when neither a file nor the route sets one
FILL_REASON = "Coulomb sliding force independent of speed (A5.speed, SCR-WP3-10)"
SLIDING_MIN_IMPRATIO = 50.0      # A5.speed passes from 50 (30: -11 %)
ROBOT_CG_EFFECT = "mujoco:solver=cg:robot_chain"   # a robot file that is a long light chain (alloy.scene.robot_chain)
ROBOT_CG_REASON = ("a robot file that is a long chain of light bodies: MuJoCo Warp's float32 Newton solver goes "
                   "non-finite at its first contact (alloy.scene.robot_chain, SAMPLE-CLASSIC C1)")
RIGID_FAST_EFFECT = "mujoco:rigid=fast"   # the opt-in fast rigid mode (SPEED-S1; WP7's rigid_step flag)
REDUCE_REASON = ("fast rigid mode (SPEED-S1): box-box face pairs at 4 contacts, not up to 8, on the MuJoCo Warp "
                 "route (alloy.contact.mjw_boxbox_reduce; MuJoCo C keeps mjc_BoxBox's)")
CAP_REASON = ("elliptic friction rows' pull capped (RIGID-DEFAULT, alloy.contact.mjw_friction_cap): a fast slide "
              "no longer lifts and rocks a box; on the MuJoCo Warp route (MuJoCo C keeps its rows)")
CONTACTS_PER_SHAPE = 8           # default MuJoCo contact capacity per shape per world without a PackSpec
ROWS_PER_CONTACT = 6             # efc rows per contact (elliptic condim 3 → 3, pyramidal → 4) plus joint rows
OWNED = {"body": "bodies", "particle": "particles", "joint": "joints", "shape": "shapes"}


def global_ids(rng: range, stride: int, worlds: int) -> list[int]:
    return [i + w * stride for w in range(worlds) for i in rng]


def _owned(objs: Sequence[TemplateObject], strides: Mapping[str, int], worlds: int) -> dict[str, tuple[int, ...]]:
    out: dict[str, list[int]] = {v: [] for v in OWNED.values()}
    for t in objs:
        for kind, field in OWNED.items():
            out[field].extend(global_ids(getattr(t.ranges, kind), strides[kind], worlds))
    return {k: tuple(sorted(v)) for k, v in out.items()}


def target_dt(entry: str, objs: Sequence[TemplateObject], control_dt: float, precision_dt: float | None,
              rigid_dt: float | None = None) -> float:
    """D4 target for an entry, snapped down to the grid. The MuJoCo entry starts from the scene's rigid step when given
    (``BuiltScene.rigid_dt``, ``alloy.scene.rigid_step``, SCR-WP0-RIGIDDT), every other entry from RIGID_TARGET; the
    MPM, SPH and precision limits below only lower it."""
    dt = rigid_dt if entry == "mujoco" and rigid_dt is not None else RIGID_TARGET
    for t in objs:
        n, route = t.decision.numbers, t.decision.route
        v_max = float(n.get("contact_speed", 0.5))
        if route in (Route.GRAIN_MPM, Route.PASTE_MPM, Route.LIQUID_MPM) and "voxel" in n:
            dt = min(dt, MPM_DT_MAX, 0.5 * n["voxel"] / v_max)
        if route == Route.LIQUID_SPH and "particle_radius" in n:
            dt = min(dt, SPH_CFL * 2.0 * n["particle_radius"] / v_max)
    if entry in ("mujoco", "featherstone", "kamino") and precision_dt is not None:
        dt = min(dt, precision_dt)
    return snap_to_grid(dt, control_dt)


def default_capacities(shapes_per_world: int) -> tuple[int, int]:
    """(nconmax, njmax) per world without a PackSpec: 8 contacts per shape, 6 rows per contact (min 64 / 384).
    MuJoCo's own defaults (48 / 64) overflow a 30-link self-colliding chain and drop contacts (ROD-5)."""
    ncon = max(64, CONTACTS_PER_SHAPE * shapes_per_world)
    return ncon, ROWS_PER_CONTACT * ncon


CCD_PRIMITIVES = ("cylinder", "cone", "ellipsoid")


def ccd_backed(t: TemplateObject) -> bool:
    """Shapes MuJoCo collides through its general convex (CCD) path: meshes, MJCF assemblies, cylinders, cones and
    ellipsoids. Without multiccd that path gives one contact per geom pair, and a flat steel spoon, a bolt on its head
    and a glass jar rocked and walked 1-5 mm/s on a table without ever resting; with multiccd (up to 4 points,
    newton@ac7f7830 solver_mujoco.py:4409) they rest within 0.02 mm (CARD-1). Margins are unaffected: Newton already
    zeroes them whenever a box meets the default NATIVECCD path (solver_mujoco.py:4680-4686)."""
    if t.obj is None:
        return False
    s = t.obj.card.shape
    return isinstance(s, (ct.MeshSource, ct.MJCFSource)) or (isinstance(s, ct.Primitive) and s.kind in CCD_PRIMITIVES)


ELLIPTIC_ONLY = ("the ordered MuJoCo Warp route (D9 repeat identity) and Alloy's scoped noslip are proven on the "
                 "elliptic cone only (a pyramidal cone turns the ordered route off)")


CHAIN_NEWTON_HOLD = ("left at MuJoCo's 1: a CHAIN stepped by the Newton solver goes non-finite at impratio 30-300 "
                     "(float32 Newton Hessian; T09 rounds 1 and 5), so sliding friction stays rate-dependent here "
                     "(A5.speed fails; SCR-WP3-10)")


def entry_overrides(objs: Sequence[TemplateObject], effects: set[str] | frozenset[str], has_precision: bool,
                    mjcf: MjcfOptions | None, chain_newton: bool) -> dict[str, tuple[object, str]]:
    """`mujoco_overrides` plus the chain-grip rule's (`alloy.contact.chain_grip`, SCR-WP3-CHAINGRIP) for the MuJoCo
    entry's objects `objs`: what the entry runs over its MJCF options, key -> (value, reason)."""
    out = mujoco_overrides(effects, has_precision, mjcf, chain_newton)
    own = {**(mjcf.values if mjcf is not None else {}), **{k: v for k, (v, _) in out.items()}}
    newton = str(own.get("solver", "newton")) == "newton"
    return {**out, **chain_grip.overrides(objs, float(own.get("impratio", 1.0)), has_precision, newton)}


def chain_on_newton(objs: Sequence[TemplateObject], effects: set[str] | frozenset[str],
                    mjcf: MjcfOptions | None = None) -> bool:
    """A CHAIN object in the MuJoCo entry whose solver resolves to Newton (no CG effect, or the fast profile's)."""
    if not any(t.decision.route == Route.CHAIN for t in objs):
        return False
    if "mujoco:solver=cg" in effects:
        return NEWTON_EFFECT in effects
    return str((mjcf.values if mjcf is not None else {}).get("solver", "newton")) == "newton"


def mujoco_overrides(effects: set[str] | frozenset[str], has_precision: bool,
                     mjcf: MjcfOptions | None = None, chain_newton: bool = False) -> dict[str, tuple[object, str]]:
    """The MuJoCo options the route sets over the scene's MJCF `<option>` (alloy.scene.mjcf_options): key ->
    (value, reason for the route report). `chain_newton`: the entry steps a CHAIN on the Newton solver
    (`chain_on_newton`), which holds back the impratio fill-in and HELD's raise (`held_route.HELD_EFFECT`: a file's
    impratio below 3 is raised to 3)."""
    out: dict[str, tuple[object, str]] = {}
    if mjcf is not None and mjcf.values.get("cone") == "pyramidal":
        out["cone"] = ("elliptic", ELLIPTIC_ONLY)
    if any(e.startswith("mujoco:integrator=") for e in effects):
        out["integrator"] = (effect_integrator(effects), "integrator route effect "
                             + ", ".join(sorted(e for e in effects if e.startswith("mujoco:integrator="))))
    if "mujoco:solver=cg" in effects:      # equality attachments on a chain (alloy.scene.attachments)
        # T09 on CPU: 50 CG iterations hit the limit on 206/256 steps, 200 on 5, 1000 on 0 (same max |v| to 1e-4).
        # The fast profile's Newton solver keeps the 200 cap, as WP3 measured it (alloy.scene.fast_profile).
        newton_ = NEWTON_EFFECT in effects
        why = ("equality attachment on a chain: the fast profile's Newton + chain armature (SCR-WP3-8)" if newton_
               else "chain attachment: CG (the SCR-DC1-2 rule did not show Newton's float32 Hessian positive "
                     "definite; Newton went non-finite on T09)")
        out["solver"] = ("newton" if newton_ else "cg", why)
        out["iterations"] = (CG_ITERATIONS, why)
    if ROBOT_CG_EFFECT in effects:         # over the fast profile's Newton too: that one relies on the CHAIN armature
        out["solver"] = ("cg", ROBOT_CG_REASON)
        out["iterations"] = (CG_ITERATIONS, ROBOT_CG_REASON)
    if has_precision:
        fast = FAST_EFFECT in effects        # opt-in fast profile: noslip holds the pairs instead
        out["cone"] = ("elliptic", "precision pairs (§3.5 creep rule, elliptic cone)")
        out["impratio"] = ((FAST_IMPRATIO, "fast profile: impratio 2 + scoped noslip on the precision pairs "
                                           "(SCR-WP3-8)") if fast
                           else (PRECISION_IMPRATIO, "precision pairs (§3.5 creep rule)"))
    if "impratio" not in out and (mjcf is None or "impratio" not in mjcf.values) and not chain_newton:
        out["impratio"] = (FILL_IMPRATIO, FILL_REASON)       # fills in only: a file's value stands
    low = mjcf is not None and float(mjcf.values.get("impratio", HELD.impratio)) < HELD.impratio
    if HELD_EFFECT in effects and "impratio" not in out and not chain_newton and low:
        out["impratio"] = (HELD.impratio, HELD_REASON)       # a gripper on a free object (alloy.scene.held_route)
    if RIGID_FAST_EFFECT in effects:
        from alloy.contact.mjw_boxbox_reduce import REDUCE_KEY  # noqa: PLC0415 - the kernel module, fast mode only
        from alloy.contact.mjw_friction_cap import CAP_DEFAULT, CAP_KEY  # noqa: PLC0415

        out[REDUCE_KEY] = (True, REDUCE_REASON)
        out[CAP_KEY] = (CAP_DEFAULT, CAP_REASON)
    return out


def sliding_warning(mjcf: MjcfOptions | None, kw: Mapping[str, object]) -> str | None:
    """A route-report warning when a file's impratio below SLIDING_MIN_IMPRATIO stands (SCR-WP3-10)."""
    if mjcf is None or "impratio" not in mjcf.values or kw.get("impratio") != mjcf.values["impratio"]:
        return None
    value = float(mjcf.values["impratio"])
    if value >= SLIDING_MIN_IMPRATIO:
        return None
    return (f"mujoco impratio={value:g} from {mjcf.source['impratio']}: sliding friction rate-dependent below "
            f"~ mu delta / (2 tau impratio); A5.speed needs >= {SLIDING_MIN_IMPRATIO:g} (SCR-WP3-10)" + _grip_note(kw))


def hold_warning(kw: Mapping[str, object]) -> str:
    """The route-report warning of a CHAIN on Newton left at impratio 1 (CHAIN_NEWTON_HOLD)."""
    return f"mujoco impratio {CHAIN_NEWTON_HOLD}" + _grip_note(kw)


def _grip_note(kw: Mapping[str, object]) -> str:
    grip = kw.get(chain_grip.KEY)
    if not isinstance(grip, Mapping) or "impratio" not in grip:
        return ""
    return (f"; the chain grip runs impratio {float(grip['impratio']):g} on MuJoCo C, "
            f"{chain_grip.WARP_IMPRATIO:g} on MuJoCo Warp")


def solver_kwargs(entry: str, objs: Sequence[TemplateObject], capacities: PackSpec | None,
                  has_precision: bool, shapes_per_world: int = 0, ccd_scene: bool = False,
                  scene_effects: Sequence[str] = (), mjcf: MjcfOptions | None = None,
                  radii: Mapping[str, float] | None = None) -> dict[str, object]:
    """Per-scene solver kwargs on top of the capability defaults (merged by the caller); `ccd_scene`: some object in
    the scene (owned or static) collides through MuJoCo's CCD path (`ccd_backed`); `scene_effects`: scene-level
    effects beside the objects' own (the assembler's rk4 rule, `integrator_route`); `mjcf`: the scene's MJCF
    `<option>` values, which the MuJoCo entry takes over its capability defaults and below the route's overrides;
    `radii`: the soft solids' particle contact radii as emitted (`vbd_contact.emitted_radii`)."""
    kw: dict[str, object] = dict(capability(entry).default_kwargs)
    effects = {e for t in objs for e in t.decision.scene_effects} | set(scene_effects)
    if entry == "mujoco":
        kw.update(mjcf.values if mjcf is not None else {})
        chain_nw = chain_on_newton(objs, effects, mjcf)
        kw.update({k: v for k, (v, _) in entry_overrides(objs, effects, has_precision, mjcf, chain_nw).items()})
        if ccd_scene or any(ccd_backed(t) for t in objs):
            kw["enable_multiccd"] = True
        if capacities is not None:
            kw.update(njmax=capacities.njmax, nconmax=capacities.nconmax)
        else:
            ncon, nj = default_capacities(shapes_per_world)
            kw.update(nconmax=ncon, njmax=nj)
    if entry == "vbd" and ACCURATE_EFFECT in effects:  # opt-in accurate cloth profile (vbd.cloth_profile)
        kw["iterations"] = ACCURATE_ITERATIONS
    if entry == "vbd" and objs and all(t.decision.route == Route.ROD_VBD for t in objs):
        kw["iterations"] = ROD_ITERATIONS
    if entry == "vbd":
        kw = cloth_accuracy.entry_kwargs(kw, effects)  # scene.cloth_accuracy (SCR-WP2-13): cloth entries only
        kw = stiff_sheet.entry_start(kw, effects)      # a stiff sheet: the inertial start (launch item 2)
        kw.update(vbd_self_contact(objs, radii))
    if entry == "mpm":
        voxels = [t.decision.numbers["voxel"] for t in objs if "voxel" in t.decision.numbers]
        if voxels:
            kw["voxel_size"] = float(min(voxels))
        if mpm_solid.EFFECT in effects:             # a j2 solid: the tight solve its yield plateau needs (GAP #6)
            kw["tolerance"], kw["max_iterations"] = mpm_solid.TOLERANCE, mpm_solid.MAX_ITERATIONS
            kw["delassus_regularization"] = mpm_solid.DELASSUS_REGULARIZATION   # slice 2b (vendored patch 0015)
    if entry == "sph":
        kw.update(sph_kwargs(objs))
    return kw


def sph_kwargs(objs: Sequence[TemplateObject]) -> dict[str, object]:
    """The router's particle radius for the SPH entry (SCR-JOB2-2 item 3, SCR-DC5-18 item 2): the plug-in seeds the
    particles at it (``sph.build.emit_object``), so the solver must run at it too (``DFSPHConfig`` defaults to 2 mm).
    One SolverDFSPH runs one radius: liquids routed at different radii are refused, naming them. The router gives no
    escape box, so ``world_bounds`` stays the solver's (none)."""
    radii = {t.name: float(t.decision.numbers["particle_radius"]) for t in objs
             if t.decision.route == Route.LIQUID_SPH and "particle_radius" in t.decision.numbers}
    if not radii:
        return {}
    if len(set(radii.values())) > 1:
        raise AssemblyError(f"SPH entry: liquids routed at different particle radii {radii}; one SolverDFSPH runs one "
                            "radius (give the cards one passage width, or split the liquids)")
    return {"particle_radius": next(iter(radii.values()))}


def max_substeps(entry: str, coupled: bool, has_precision: bool) -> int:
    """Construction rung: the D21 ladder max for coupled entries, 1 otherwise. A coupled MuJoCo entry with precision
    pairs steps once per exchange at its target dt, the pairs' L2 dt (SCR-JOB2-1: later substeps would get precision
    rows posed at the exchange start, ``alloy.integration.precision_reach``); WP12's ``precision_pin`` caps the
    exchange interval at that target and flags worlds whose stability estimate asks for less."""
    if not coupled:
        return 1
    return 1 if entry == "mujoco" and has_precision else SUBSTEP_LADDER_MAX


def make_entries(objs: Sequence[TemplateObject], strides: Mapping[str, int], worlds: int, control_dt: float, *,
                 capacities: PackSpec | None = None, precision_dt: float | None = None,
                 has_precision: bool = False, coupled: bool = False,
                 noslip: tuple | None = None, scene_effects: Sequence[str] = (),
                 mjcf: MjcfOptions | None = None, builder: newton.ModelBuilder | None = None,
                 rigid_dt: float | None = None) -> list[EntrySpec]:
    """One EntrySpec per capability, entries sorted by name (deterministic, CARD-6); `noslip` (iterations,
    tolerance[, precision pairs as template shape ids]) rides in the MuJoCo entry's kwargs for the capability factory
    (alloy.scene.noslip_route);
    `scene_effects` reach every entry's kwargs beside its objects' own (`solver_kwargs`); `rigid_dt`: the scene's
    rigid step, the MuJoCo entry's target (`target_dt`, `alloy.scene.rigid_step`); `builder`: the template
    builder, whose particle radii set the VBD contact distance (`vbd_contact`)."""
    from alloy.coupling.unsupported import check_objects  # noqa: PLC0415

    check_objects(objs, builder)    # a liquid starting inside a free soft object: refused (WP13 F12)
    radii = emitted_radii(objs, builder.particle_radius) if builder is not None else None
    by_entry: dict[str, list[TemplateObject]] = {}
    for t in objs:
        if t.decision.entry:
            by_entry.setdefault(t.decision.entry, []).append(t)
    specs = []
    ccd = any(ccd_backed(t) for t in objs)
    for name in sorted(by_entry):
        group = by_entry[name]
        owned = _owned(group, strides, worlds)
        specs.append(EntrySpec(name=name, capability=name, substeps=1,
                               target_dt=target_dt(name, group, control_dt, precision_dt, rigid_dt),
                               max_substeps=max_substeps(name, coupled, has_precision),
                               solver_kwargs=solver_kwargs(name, group, capacities, has_precision,
                                                           strides["shape"], ccd, scene_effects, mjcf, radii),
                               **owned))
        if name == "mujoco" and noslip is not None:
            from alloy.scene.noslip_route import NOSLIP_KEY

            specs[-1].solver_kwargs[NOSLIP_KEY] = (int(noslip[0]), float(noslip[1]), *noslip[2:])
    return specs


__all__ = ["KINDS", "RIGID_FAST_EFFECT", "chain_on_newton", "entry_overrides", "global_ids", "hold_warning",
           "make_entries",
           "max_substeps", "mujoco_overrides", "sliding_warning", "solver_kwargs", "sph_kwargs", "target_dt"]
