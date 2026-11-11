# SPDX-License-Identifier: Apache-2.0
"""Scene-level MuJoCo integrator rule (SPEC §3.5 routing rule beside the integrator table, SCR-WP6-4).

A routed MuJoCo entry runs `implicitfast` by default and `discrete` for CHAIN scenes (the router's per-object
`mujoco:integrator=...` effects). It runs `rk4` when every dof of the entry is on a free joint of a dynamic object (no
robot, no articulated, damped, armature or actuated dof, no equality row, no kinematic body) and the scene has no
precision pair and no noslip. Sleeping is never on at assembly; the capability factory drops rk4 for implicitfast if
a caller adds it (rk4 has no sleeping, newton@ac7f7830 solver_mujoco.py:6881).
Why: MuJoCo's free-body rotation is first order under every other integrator; WP6's A4 torque-free tumble drifts
2.29e-3 at 1.25 ms under implicitfast, discrete and implicit and 1.07e-4 under rk4 (bound 1e-3), while collide/cradle
pass under all (wp2-results.md "A4 tumble drift vs integrator").
The rule needs the whole scene (robots, precision pairs, noslip), so the assembler applies it on the emitted template,
not the per-object router, before the native contact gains are written: under rk4 they take a quarter of the
stability cap (`native_gains.RK4_KE_FRACTION`). The result is the scene effect RK4_EFFECT.

**Fast mode** (coordinator ruling 2026-11-09, main only; ``_fast_implicit``): under the rigid step request "fast"
(``alloy.scene.rigid_step``) a free-body-only entry that R1 or R2 would have taken (more than R1_MIN_FREE bodies, or
all isotropic within OWN_SPREAD; none floored) keeps implicitfast with rk4's gains, and step 2c may follow: an opt-in
throughput cost, stated in FAST_ROCKS and ``rigid_step.FAST_COST``. Its law: tests/scene/test_rigid_fast.py's ceiling.

**R1 is on MuJoCo Warp only** (RIGID-DEFAULT, coordinator ruling 2026-11-09 16:32Z; ``r1_off``). R1 went off at
10:19Z: a cube pushed across the ground inside a 9-body scene rocked and kept growing over a 2 s push (pitch p-p 1.69
-> 36.4 deg against rk4's 0.33 -> 3.86, 30 cm cube at 5 ms, MuJoCo C). The rocking is MuJoCo's soft friction pull
(-b v_slip, coupled into the normal rows by the elliptic cone); WP3's friction-pull cap on R1 entries removes it
(alloy.contact.mjw_friction_cap; implicitfast + cap 20 m/s2: 0.48 deg over the same 2 s). The cap is a MuJoCo Warp
kernel, so on the CPU route on MuJoCo C (alloy.route.mjc_core_config, the CPU default) clutter keeps rk4, step 2c with
it. The rocking law (tests/bench/test_rocking_law.py) runs both CPU routes.

**R1, clutter** (F6, SCR-WP13-SPEED-1 part 1; coordinator ruling 2026-11-08 14:18Z). Such an entry with more than
R1_MIN_FREE free bodies per world, none of which the inertia floor would raise (`inertia_floor.floor_candidates`),
runs implicitfast with rk4's gains (`IntegratorChoice.gains`; the light-body rows too, through the model's
`_alloy_gain_integrator`). The result is the scene effect R1_EFFECT. Why ([internal note], CPU
MuJoCo C):
- in clutter, contact dominates, and implicitfast is 2.6-3.4x faster per period at 8-32 bodies, 3.5-5.8x at 50-100
  (L4: 3.5x at 50 objects, 64 worlds);
- with rk4's gains, marker-free piles keep 0 INVALID, no fling (peak 3.52 m/s against rk4's 3.24) and rk4's rest;
- the gains, not the integrator, broke implicitfast's piles: at full cap 9 of 400 bodies left the bin and the
  marker law flung 12 m/s;
- a light slender body (the marker law's) still flings 13-15 m/s under implicitfast at rk4's gains, so an entry
  with one keeps rk4;
- free flight stays rk4 up to R1_MIN_FREE bodies: A4's twins have 5, and a tumble or toss drifts 16-25x more under
  implicitfast.
**No R2.** F6 R2 (an entry of at most R1_MIN_FREE isotropic bodies on R1's path) was withdrawn on 2026-11-09 (release
red, coordinator 10:19Z): a cube pushed across the ground rocked 1.7-5x more under implicitfast than under rk4 at the
same step (pitch peak to peak, 30 cm cube: 1.69 against 0.33 deg at auto's 5 ms, 1.98 against 1.17 at 2.5 ms;
5 cm: 4.35 against 1.45 at 1.25 ms; tests/scene/test_integrator_route.py's rocking law). Few-body entries keep rk4.
Known gap (docs/known_gaps/rigid-clutter-rotation.md): in an R1 scene a body's free-flight rotation is first order
(0.6 % angular momentum over a 2 s toss at 10 rad/s). The precision profile keeps rk4 (`alloy.integration.wiring`
switches an R1 entry back: the same gains, so the same scene as before R1).

**Own gains** (SCR-WP0-RIGIDDT-2 step 2c; WP6 + WP3 14:57Z, coordinator 15:16Z; ``own_gains``). An R1 entry under the
API's rigid step "auto" whose free bodies are all isotropic (inertia spread at most OWN_SPREAD, 0.03) takes
the gains mode "own" (``native_gains.OWN_KE_FRACTION`` of implicitfast's cap) when that allows a larger rigid step
than R1's and changes no contact: every shape gain, pair row and floored inertia at that step equals today's (the
fixed step under rk4's quarter cap) bit for bit, tried on copies of the template. Only the step grows: XB4-9's 27-cube
stacks go from 2.5 to 5 ms. The precision and debug profiles never ask for "auto", so they keep R1 or rk4.
"""

from __future__ import annotations

import contextlib
import copy
import logging
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, replace

import numpy as np

from alloy._vendor import newton

from alloy.scene.template import TemplateObject

AUTO_MODES = ("auto", "fast")       # rigid step requests step 2c follows ("fast": alloy.scene.rigid_step)
FAST_ROCKS = ("with the friction pull capped on MuJoCo Warp a 30 cm cube sliding at 5 m/s rocks under 1 deg and "
              "lifts 1 mm; on the CPU's MuJoCo C route (no cap) free bodies stay on rk4, where that cube rocks 6 deg "
              "and hops 8 cm")      # as rigid_step.FAST_COST (WP7 e08eadc28)

log = logging.getLogger("alloy.scene")
RK4_EFFECT = "mujoco:integrator=rk4"
R1_EFFECT = "mujoco:integrator=implicitfast"   # F6 R1: implicitfast on a clutter entry, with rk4's gains
OWN_EFFECT = "alloy:own_gains"  # step 2c: the entry's own gains at the assembled step (precision_entries refuses it)
R1_ON = True                    # F6 R1, MuJoCo Warp only (RIGID-DEFAULT ruling 2026-11-09 16:32Z; module docstring)
R1_MIN_FREE = 8                 # F6 R1: more free bodies per world than this, none floored -> implicitfast
FREE_DOFS = 6
OWN_SPREAD = 0.03               # step 2c: every free body's (I_max - I_min) / I_max at most this (WP3's isotropy rows)
_DOF_FIELDS = ("joint_damping", "joint_armature", "joint_target_ke", "joint_target_kd", "joint_target_mode")


@dataclass(frozen=True)
class IntegratorChoice:
    rk4: bool
    reason: str                 # why rk4 was or was not chosen (logged, in the route report)
    r1: bool = False            # F6 R1: implicitfast on a free-body clutter entry, with rk4's gains
    own: bool = False           # step 2c: an R1 entry on its own gains (``own_gains``), at a larger rigid step
    free: bool = False          # a free-body-only entry none of whose bodies is floored (fast mode may take R1's path)

    @property
    def integrator(self) -> str:
        """The integrator the MuJoCo entry runs ("implicitfast" also stands for MuJoCo's default)."""
        return "rk4" if self.rk4 else "implicitfast"

    @property
    def gains(self) -> str:
        """The gains mode the native contact gains are written for: rk4's quarter cap under rk4 and R1, "own"
        under step 2c."""
        return "own" if self.own else "rk4" if self.rk4 or self.r1 else "implicitfast"

    def effects(self) -> tuple[str, ...]:
        """The scene effects of the choice (RK4_EFFECT, or R1_EFFECT and step 2c's OWN_EFFECT)."""
        return (RK4_EFFECT,) if self.rk4 else (R1_EFFECT, OWN_EFFECT) if self.own else (R1_EFFECT,) if self.r1 else ()


def effect_integrator(effects: Iterable[str]) -> str:
    """The integrator the per-object or scene effects ask for: rk4, discrete, else implicitfast."""
    fx = set(effects)
    if RK4_EFFECT in fx:
        return "rk4"
    return "discrete" if "mujoco:integrator=discrete" in fx else "implicitfast"


NEWTON_COLLISIONS = ("sdf", "mesh")   # add_mesh modes alloy.build collides on Newton's contact pipeline (PHYSA-1)


def _equality_rows(b: newton.ModelBuilder) -> int:
    attr = b.custom_attributes.get("mujoco:equality_constraint_type")
    return len(attr.values or ()) if attr is not None else 0


def _not_free(b: newton.ModelBuilder, t: TemplateObject) -> str:
    """Why `t` keeps MuJoCo's default integrator, or "" when all its dofs are plain free-joint dofs."""
    if t.obj is None:
        return f"robot {t.name}"
    if t.obj.card.motion != "dynamic":
        return f"{t.name} is {t.obj.card.motion}"
    if any(e.startswith("mujoco:") for e in t.decision.scene_effects):
        return f"{t.name} asks {', '.join(t.decision.scene_effects)}"
    if not len(t.ranges.joint):
        return f"{t.name} has no joint"
    for j in t.ranges.joint:
        if b.joint_type[j] != newton.JointType.FREE:
            return f"{t.name} joint {j} is {newton.JointType(b.joint_type[j]).name}"
        dofs = range(b.joint_qd_start[j], b.joint_qd_start[j] + FREE_DOFS)
        for field in _DOF_FIELDS:
            if any(getattr(b, field)[d] for d in dofs):
                return f"{t.name} free joint {j} has {field.removeprefix('joint_')}"
    return ""


def choose_integrator(b: newton.ModelBuilder, objs: Sequence[TemplateObject], *, precision: bool,
                      noslip: bool, dt: float | None = None, device: str | None = None,
                      newton_contacts: bool = False) -> IntegratorChoice:
    """rk4 for a free-body-only MuJoCo entry without precision pairs or noslip, or R1's implicitfast for one with
    more than R1_MIN_FREE free bodies per world and none floored at the rigid step ``dt``, unless ``device``'s
    MuJoCo route is MuJoCo C (``r1_off``; module docstring)."""
    mj = [t for t in objs if t.decision.entry == "mujoco"]
    if not mj:
        return IntegratorChoice(False, "no MuJoCo entry")
    if precision:
        return IntegratorChoice(False, "precision pairs (rk4 has no gap ladder)")
    if newton_contacts:       # PHYSA-1; a card's sdf / mesh collision alone stays on MuJoCo's own contacts (hulls)
        return IntegratorChoice(False, f"Newton's contact pipeline{concave_names(objs)} makes the contacts once per "
                                       f"step (rk4's stages would see them frozen and bounces would gain energy)")
    if noslip:
        return IntegratorChoice(False, "noslip requested (rk4 has none)")
    if _equality_rows(b):
        return IntegratorChoice(False, f"{_equality_rows(b)} equality rows")
    kinematic = [t.name for t in objs if t.obj is not None and t.obj.card.motion == "kinematic"]
    if kinematic:
        return IntegratorChoice(False, f"kinematic bodies in the MuJoCo model ({', '.join(kinematic)})")
    for t in mj:
        why = _not_free(b, t)
        if why:
            return IntegratorChoice(False, why)
    return _clutter_or_rk4(b, mj, dt, device)


def concave_names(objs: Sequence[TemplateObject]) -> str:
    """" (sdf / mesh collision: a, b)" for the objects that made ``alloy.build`` pick Newton's pipeline, else ""."""
    names = [t.name for t in objs if getattr(getattr(t.obj, "card", None), "shape", None) is not None
             and getattr(t.obj.card.shape, "collision", "") in NEWTON_COLLISIONS]
    return f" (sdf / mesh collision: {', '.join(names[:4])})" if names else ""


def r1_off(device: str | None) -> str:
    """Why F6 R1 does not apply to a MuJoCo entry on ``device``, or "" when it does: R1_ON False, or a CPU device
    whose CPU route is MuJoCo C (``ALLOY_CPU_ROUTE``, default "mjc"), where the friction-pull cap has no hook.
    ``device`` None (a plan without a build): as on MuJoCo Warp."""
    if not R1_ON:
        return "implicitfast for clutter is off: a sliding body rocks more"
    if _mujoco_c(device):
        return "the CPU route on MuJoCo C has no friction-pull cap, so a sliding body would rock more"
    return ""


def _mujoco_c(device: str | None) -> bool:
    """``device`` is a CPU whose CPU route is MuJoCo C (``ALLOY_CPU_ROUTE``, default "mjc"): no friction-pull cap."""
    if device is None or not str(device).startswith("cpu"):
        return False
    from alloy.route.mjc_core_config import MJC, requested_route  # noqa: PLC0415

    return requested_route({}) == MJC


def with_gains(specs: list, choice: IntegratorChoice) -> list:
    """The entry specs with R1's gains flag on the MuJoCo entry (``light_body.GAINS_KEY``: the factories pass it to
    the light-body rule, so its rows take rk4's quarter cap as the native gains do)."""
    if not choice.r1:
        return specs
    from alloy.contact.light_body import GAINS_KEY  # noqa: PLC0415

    return [replace(e, solver_kwargs={**e.solver_kwargs, GAINS_KEY: True}) if e.name == "mujoco" else e for e in specs]


def own_gains(choice: IntegratorChoice, b: newton.ModelBuilder, plan, request, gains: Callable) -> IntegratorChoice:
    """Step 2c (module docstring): ``choice`` with ``own`` set when an R1 entry qualifies, else ``choice``. ``plan``:
    the assembler's (objects, report); ``gains(builder, mode, request, report)`` writes the rigid step, native gains,
    floor and pair rows on a builder; ``b`` itself is only read. Premise: an R1 entry has no light-body rows (they
    are written at run time from MJCF loose bodies, and a robot keeps an entry off R1), so the guard need not see
    them; under "own" the entry keeps R1's GAINS_KEY."""
    mode = getattr(request, "mode", None)
    if mode == "fast" and choice.rk4 and choice.free:
        choice = _fast_implicit(choice, b, plan)
    if not choice.r1 or mode not in AUTO_MODES:
        return choice
    report = plan.report
    why = _not_eligible(b, plan.objects)
    if why:
        log.info("step 2c: R1 keeps rk4's gains (%s)", why)
        return choice
    from alloy.scene.rigid_step import choose_rigid_step  # noqa: PLC0415

    try:                              # WP0: a trial that cannot run (e.g. an uncopyable source) keeps R1
        with _held_logs():
            r1_dt = choose_rigid_step(b, plan.objects, plan.scene, "rk4", request).dt     # reads b only
        own, own_dt = _trial(b, "own", request, gains, report)
        bigger = own_dt > r1_dt if mode == "auto" else own_dt >= r1_dt     # "fast": the stiffer gains at R1's step
        why = "" if bigger else "no larger step"
        if not why and mode == "auto" and own != _trial(b, "rk4", replace(request, mode="fixed"), gains, report)[0]:
            why = "a contact would differ from today's"     # the fast mode takes softer contacts (rigid_step)
    except Exception as exc:  # noqa: BLE001
        log.info("step 2c: R1 keeps rk4's gains (the trial failed: %s: %s)", type(exc).__name__, exc)
        return choice
    if why:
        log.info("step 2c: R1 keeps rk4's gains (%s: own %g ms, R1 %g ms)", why, own_dt * 1e3, r1_dt * 1e3)
        return choice
    same = "every contact as today's" if mode == "auto" else "fast mode: softer contacts than today's"
    own = replace(choice, own=True, reason=f"{choice.reason}; step 2c: its own gains at {own_dt * 1e3:g} ms "
                                           f"(R1: {r1_dt * 1e3:g} ms), {same}")
    _renamed_in_routes(report, plan.objects, choice, own)
    log.info("MuJoCo integrator rule: %s", own.reason)
    return own


def _fast_implicit(choice: IntegratorChoice, b: newton.ModelBuilder, plan) -> IntegratorChoice:
    """The rigid fast mode keeps implicitfast (coordinator ruling 2026-11-09, main only; module docstring, "Fast
    mode"): a free-body-only entry on rk4 takes R1's path, with rk4's gains, when it has more than R1_MIN_FREE bodies
    or all its bodies are isotropic (spread at most OWN_SPREAD: what R2 asked), and none is floored at the fixed step;
    step 2c may then take its own gains. Else ``choice``. On the CPU's MuJoCo C route it keeps rk4 (audit AB-gaps-1,
    2026-11-09): the friction-pull cap has no hook there, and under implicitfast a 30 cm cube sliding at 5 m/s
    toppled within 2 s (pitch 90 deg; rk4 at the fast step: 6 deg, an 8 cm hop)."""
    from alloy.contact.inertia_floor import floor_candidates  # noqa: PLC0415
    from alloy.scene.native_gains import rigid_dt_max  # noqa: PLC0415

    if _mujoco_c(getattr(plan, "device", None)):
        kept = replace(choice, reason=f"{choice.reason}; fast mode keeps rk4 on the CPU's MuJoCo C route (no "
                                      "friction-pull cap there: under implicitfast a cube sliding at 5 m/s toppled)")
        _renamed_in_routes(plan.report, plan.objects, choice, kept)
        return kept
    mj = [t for t in plan.objects if t.decision.entry == "mujoco"]
    if sum(len(t.ranges.joint) for t in mj) <= R1_MIN_FREE and worst_spread(b, plan.objects) > OWN_SPREAD:
        return choice
    if floor_candidates(b, mj, rigid_dt_max(plan.scene.control_dt)):
        return choice
    fast = replace(choice, rk4=False, r1=True, reason=f"{choice.reason}; fast mode: implicitfast with rk4's gains "
                                                      f"({FAST_ROCKS})")
    _renamed_in_routes(plan.report, plan.objects, choice, fast)
    return fast


def _renamed_in_routes(report, objs: Sequence[TemplateObject], old: IntegratorChoice, new: IntegratorChoice) -> None:
    """``name_in_routes``' line for ``old`` replaced by ``new``'s in each MuJoCo object's route reasons."""
    was, line = (f"mujoco integrator {c.integrator}: {c.reason}" for c in (old, new))
    for t in objs:
        d = report.decisions.get(t.name)
        if t.decision.entry == "mujoco" and d is not None:
            report.decisions[t.name] = replace(d, reasons=tuple(line if r == was else r for r in d.reasons))


def _not_eligible(b: newton.ModelBuilder, objs: Sequence[TemplateObject]) -> str:
    """Why the entry's bodies cannot take own gains, or "": an MJCF-labelled body (light_body's scope: its rows are
    written at run time, outside the guard; WP0), or an inertia spread (I_max - I_min) / I_max over OWN_SPREAD."""
    from alloy.contact.light_body import GEOM_FILE  # noqa: PLC0415

    for body in _bodies(objs):
        if GEOM_FILE.match(str(b.body_label[body])):
            return f"an MJCF body, {b.body_label[body]}"
    worst = worst_spread(b, objs)
    return f"an inertia spread of {worst:.3g} > {OWN_SPREAD:g}" if worst > OWN_SPREAD else ""


def _bodies(objs: Sequence[TemplateObject]) -> Iterator[int]:
    return (body for t in objs if t.decision.entry == "mujoco" for body in t.ranges.body)


def worst_spread(b: newton.ModelBuilder, objs: Sequence[TemplateObject]) -> float:
    """The largest inertia spread (I_max - I_min) / I_max over the MuJoCo entry's bodies (1 for a body without
    inertia, 0 for none)."""
    worst = 0.0
    for body in _bodies(objs):
        ev = np.linalg.eigvalsh(np.asarray(b.body_inertia[body], dtype=np.float64).reshape(3, 3))
        worst = max(worst, float((ev[-1] - ev[0]) / ev[-1]) if ev[-1] > 0.0 else 1.0)
    return worst


@contextlib.contextmanager
def _held_logs() -> Iterator[None]:
    """Log lines and warnings held back (``logging.disable``: process-wide while it lasts; a build is one thread)."""
    held = logging.root.manager.disable
    logging.disable(logging.WARNING)
    try:
        yield
    finally:
        logging.disable(held)


def _trial(b: newton.ModelBuilder, mode: str, request, gains: Callable, report) -> tuple[tuple, float]:
    """(every contact value, inertia and floor armature the gains pass writes, its rigid step) on a copy of ``b``,
    its log lines held back (``_held_logs``; the real pass gives them once)."""
    bb, rep = copy.deepcopy(b), copy.deepcopy(report)
    with _held_logs():
        gains(bb, mode, request, rep)
    mj = tuple((k, _bits(a.values)) for k, a in sorted(bb.custom_attributes.items()) if k.startswith("mujoco:"))
    values = (_bits(bb.shape_material_ke), _bits(bb.shape_material_kd), _bits(bb.body_inertia),
              _bits(bb.joint_armature), mj)                          # WP3: the floor writes joint_armature
    return values, float(rep.rigid_step.dt)


def _bits(values) -> bytes | tuple:
    """Exact bits of builder values (numbers, Warp vectors / matrices, lists or {index: value} of them)."""
    if isinstance(values, dict):
        return tuple((k, _bits(v)) for k, v in sorted(values.items()))
    try:
        return np.asarray(values, dtype=np.float64).tobytes()       # float32 widens exactly
    except (TypeError, ValueError):
        return tuple(_bits(v) for v in values) if isinstance(values, (list, tuple)) else (repr(values),)


def precision_entries(built, preset: str, notes: list[str]):
    """Under the precision profile an R1 clutter entry runs rk4 again (F6 ruling 2026-11-08 14:18Z). Its gains,
    floor and light rows are rk4's already, so the scene is the one the SCR-WP6-4 rule built before R1. A step-2c
    entry is refused: its step was set at assembly for implicitfast (the API asks "auto" only outside precision)."""
    if preset != "precision" or R1_EFFECT not in getattr(built.report, "scene_effects", ()):
        return built
    if OWN_EFFECT in built.report.scene_effects:
        from alloy.core.errors import AssemblyError  # noqa: PLC0415

        raise AssemblyError("the precision profile on a step-2c entry (own gains at the assembled rigid step): "
                            "assemble it with rigid_step 'fixed' (RigidStepRequest), as alloy.build does there")
    from alloy.integration.built_copy import replace_built  # noqa: PLC0415

    notes.append("mujoco integrator rk4: the precision profile keeps rk4 on a clutter entry")
    return replace_built(built, entries=[replace(e, solver_kwargs={**e.solver_kwargs, "integrator": "rk4"})
                                         if e.capability == "mujoco" else e for e in built.entries])


def name_in_routes(report, objs: Sequence[TemplateObject], choice: IntegratorChoice) -> IntegratorChoice:
    """Log the choice; under rk4 or R1, each MuJoCo object's route reasons (``Sim.routes()``,
    ``alloy.estimate(...)["routes"]``) name the integrator and why (F6 ruling condition 4)."""
    log.info("MuJoCo integrator rule: %s", choice.reason)
    if not (choice.rk4 or choice.r1):
        return choice
    line = f"mujoco integrator {choice.integrator}: {choice.reason}"
    for t in objs:
        d = report.decisions.get(t.name)
        if t.decision.entry == "mujoco" and d is not None and line not in d.reasons:
            report.decisions[t.name] = replace(d, reasons=(*d.reasons, line))
    return choice


def _clutter_or_rk4(b: newton.ModelBuilder, mj: Sequence[TemplateObject], dt: float | None,
                    device: str | None = None) -> IntegratorChoice:
    """R1 for a free-body-only entry with more than R1_MIN_FREE bodies per world, none floored, on a route R1 applies
    to (``r1_off``), else rk4."""
    from alloy.contact.inertia_floor import floor_candidates  # noqa: PLC0415

    n = sum(len(t.ranges.joint) for t in mj)                 # free bodies per world (one free joint each)
    base = f"free-body-only MuJoCo entry ({n} free bodies per world)"
    if n <= R1_MIN_FREE:
        return IntegratorChoice(True, f"{base}: rk4, free flight stays accurate up to {R1_MIN_FREE}", free=True)
    off = r1_off(device)
    if off:
        return IntegratorChoice(True, f"{base}: rk4 ({off})", free=True)
    floored = [t.name for t, _, _ in floor_candidates(b, mj, dt)] if dt is not None else []
    if floored:
        return IntegratorChoice(True, f"{base}: rk4, the inertia floor raises {', '.join(sorted(floored)[:4])} "
                                      f"(a light slender body flings under implicitfast)")
    return IntegratorChoice(False, f"{base}, more than {R1_MIN_FREE}, none floored: implicitfast with rk4's gains "
                                   f"(free-flight rotation first order)", r1=True, free=True)


__all__ = ["OWN_EFFECT", "OWN_SPREAD", "R1_EFFECT", "R1_MIN_FREE", "RK4_EFFECT", "IntegratorChoice",
           "choose_integrator", "effect_integrator", "name_in_routes", "own_gains", "precision_entries", "r1_off",
           "with_gains", "worst_spread"]
