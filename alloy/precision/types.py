# SPDX-License-Identifier: Apache-2.0
"""Precision-mode types (SPEC §3.7; presets, determinism, health and digests are WP5)."""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import warp as wp

from alloy._vendor.newton import Model
from alloy._vendor.newton.solvers import SolverBase

from alloy.core.defaults import (
    ENERGY_JUMP_REL, ENERGY_S1_REL, FRAME_FAIL_M, FRAME_WARN_M, PEN_LIMIT_M, PEN_S1_M, SPEED_FAIL_MPS,
)
from alloy.core.ids import digest_of


class Determinism(enum.IntEnum):
    NONE = 0
    RUN_TO_RUN = 1
    GPU_TO_GPU = 2


class GpuTier(enum.Enum):
    A = "fp64_half"        # sm_80/90
    B = "fp64_slow"        # sm_86/89; unknown CUDA -> B
    CPU = "cpu"


@dataclass(frozen=True)
class FramePolicy:
    worlds_at_origin: bool = True
    r_warn_m: float = FRAME_WARN_M
    r_fail_m: float = FRAME_FAIL_M
    rebase: Literal["off", "manual", "auto"] = "off"
    rebase_radius_m: float = 2.0
    far_invalid: bool = False               # |p| > r_fail is INVALID (precision/reference presets set True, SCR-WP5-2)


@dataclass(frozen=True)
class AccumPolicy:
    hessian: Literal["f32", "f64", "f32_refine", "auto"] = "auto"   # applies to Alloy-owned solves only (D14)
    refine_iters: int = 2
    positions: Literal["f32", "f32_kahan", "f64"] = "f32"            # non-f32 needs upstream hook (D14)
    scatter: Literal["atomic", "fixed64"] = "fixed64"


SKIPPABLE_CHECKS = frozenset({"energy", "joints"})


@dataclass(frozen=True)
class CheckPolicy:
    level: Literal["off", "cheap", "full"] = "cheap"
    every: int = 1
    full_every: int = 10
    on_invalid: Literal["latch", "raise"] = "latch"
    pen_limit_m: float = PEN_LIMIT_M        # D3 (S2); precision pairs per pair from PrecisionPairSpec
    pen_s1_m: float = PEN_S1_M              # D3 (S1)
    joint_limit_m: float | None = None      # None -> max(1e-3, 3 × solref steady-state violation at design load)
    joint_limit_rad: float | None = None    # None -> max(1e-3 rad, same rule)
    speed_fail_mps: float = SPEED_FAIL_MPS  # bodies and active particles (D12)
    energy_jump_rel: float = ENERGY_JUMP_REL  # per step, of E_scale, net of W_ext (D20)
    energy_s1_rel: float = ENERGY_S1_REL    # cumulative unexplained gain, of E_scale (S1)
    # SCR-SC0-1 / GAP-E1 (audit ERR-1): penetration from MuJoCo's own contacts (native route: rigid bodies, robots)
    # is always measured into max_pen_m and raises PEN_WARN / PEN_FAIL (on by default since GAP-E1). Shapes that cut
    # or press by design (a blade edge) are held to pen_s1_m alone with HealthMonitor.relax_thin_part_rule
    native_pen_flags: bool = True
    # GAP-E1: a native contact past its PEN_FAIL limit is PEN_WARN (DEGRADED) at once and PEN_FAIL (INVALID) once it
    # has stayed past the limit for this long, or at once past native_pen_hard x pen_s1_m: an impact or a landing
    # (a 5 cm ball dropped 0.9 m dips 10.1 mm for a few ms and rests at 0.1 mm) is not a resting overlap. Callers
    # that give ``HealthMonitor.update`` no ``dt`` get PEN_FAIL at the first check. The limit is 1 cm or half the
    # thinner part along the contact normal, so "at once" is past 2 cm or through the thinner part
    native_pen_dwell_s: float = 0.1
    native_pen_hard: float = 2.0
    # SCR-SC5-3 (re-ruled 2026-11-03 19:05Z): a cut-front contact (HealthMonitor.set_cut_front) may not get deeper
    # than this past its depth at the twin's first cut-front check; the blade pushes twins out, never in
    cut_front_growth_m: float = 1.0e-4
    # coordinator ruling 2026-11-06 14:17Z: a soft tet's principal strain past this leaves the soft FEM model's range
    # (HealthMonitor.set_soft_strain, WP10's alloy.topology.soft_strain measure); STRAIN_RANGE, DEGRADED
    soft_strain_range: float = 0.6
    # SCR-DC3-4: checks a rig turns off on purpose (public; no private HealthMonitor fields): "energy" (D20, e.g. a
    # gauge rig whose stored energy is not modelled) and "joints" (joint / weld violation). Unknown names raise.
    skip: frozenset[str] = frozenset()
    # SCR-WP5-14 (ruling 2026-11-08 02:51Z): a start overlap past pen_fail is INVALID only under the precision preset;
    # elsewhere START_PEN is DEGRADED and sim.check() is not ok
    start_pen_invalid: bool = False
    # SCR-WP5-14 addendum (2026-11-08 05:51Z): SPIN (DEGRADED) past this rotation in one step, |omega| dt
    spin_rad_per_step: float = 0.5
    # SCR-WP5-17 (GO 2026-11-08 14:18Z; WP2's blown-up 5 mm drape): a cloth triangle's largest principal stretch past
    # _warn is CLOTH_STRETCH (DEGRADED); past _fail the world is INVALID (a blown sheet: every particle finite)
    cloth_stretch_warn: float = 0.5
    cloth_stretch_fail: float = 5.0
    # the pass-through meter after dropped VBD self-contact pairs (self_contact_cross.py): "off" (no meter), "report"
    # (counted in the health row, DEGRADED) or "invalid" (a world whose surfaces then crossed is INVALID)
    self_contact_cross: str = "off"

    def __post_init__(self) -> None:
        unknown = set(self.skip) - SKIPPABLE_CHECKS
        if unknown:
            raise ValueError(f"CheckPolicy.skip: unknown check(s) {sorted(unknown)}; skippable: {sorted(SKIPPABLE_CHECKS)}")


Preset = Literal["fast", "standard", "precision", "reference"]


@dataclass(frozen=True)
class PrecisionConfig:
    determinism: Determinism = Determinism.RUN_TO_RUN
    native_contacts: Literal["mujoco", "newton"] = "mujoco"         # D10; spike outcome may flip the default
    frames: FramePolicy = FramePolicy()
    accum: AccumPolicy = AccumPolicy()
    checks: CheckPolicy = CheckPolicy()
    fuse_fp: bool = True
    pair_gjk_tol_m: float = 1e-6
    pair_mpr_tol_m: float = 1e-7
    # Coordinator ruling 2026-11-03 23:15Z: Alloy's ordered SolverMuJoCo route runs only on this explicit request. A
    # preset's RUN_TO_RUN determinism alone (standard, precision, reference) no longer turns it on: with the route,
    # worlds 2 and 39-63 of a 64-world T09 fast hold went bad (WP12), a world-index bug under investigation
    mjw_route: bool = False
    # SCR-DC5-4 (ii): let Alloy's ordered SolverMuJoCo route take the CG solver (CHAIN scenes). Opt-in: on T09 it
    # repeats bitwise at 1.64x the step time, so existing CG scenes keep their default path and cost. Setting it is
    # also an explicit route request
    mjw_route_cg: bool = False

    @classmethod
    def preset(cls, name: Preset, tier: GpuTier) -> PrecisionConfig:
        """The §3.7 preset table. Implemented by WP5 (``alloy.precision.config.preset``)."""
        from alloy.precision.config import preset  # WP5

        return preset(name, tier)

    def digest(self) -> str:
        """Canonical digest of every field (enters run digests and goldens)."""
        return digest_of(self)


class HealthFlags(enum.IntFlag):
    NONFINITE_Q = 1 << 0
    NONFINITE_QD = 1 << 1
    NONFINITE_PARTICLE = 1 << 2
    CONTACT_OVERFLOW = 1 << 3
    ROW_OVERFLOW = 1 << 4
    UNCONVERGED = 1 << 5
    PEN_WARN = 1 << 6
    PEN_FAIL = 1 << 7
    PAIR_PEN_FAIL = 1 << 8
    JOINT_FAIL = 1 << 9
    ENERGY_JUMP = 1 << 10
    SPEED_FAIL = 1 << 11
    FAR_FROM_ORIGIN = 1 << 12
    FALLBACK_USED = 1 << 13
    DET_DOWNGRADED = 1 << 14
    LATCHED = 1 << 15
    PAIR_PEN_WARN = 1 << 16
    RUNG_PINNED = 1 << 17     # a pinned entry's stability estimate asks a smaller dt (SCR-JOB2-1; WP12)
    STRAIN_RANGE = 1 << 18    # soft tet principal strain beyond the FEM model's range (ruling 2026-11-06 14:17Z)
    START_PEN = 1 << 19       # bodies overlapped > 1 mm at the start (SCR-WP5-14; sticky for the run)
    SPIN = 1 << 20            # a body turned > spin_rad_per_step in one step (SCR-WP5-14 addendum; WP3's markers)
    CLOTH_STRETCH = 1 << 21   # a cloth triangle stretched > cloth_stretch_warn (INVALID past _fail; SCR-WP5-17)
    SELF_CONTACT_OVERFLOW = 1 << 22   # VBD self-contact pairs dropped over their row buffers (gap #13; self_contact.py)
    FAST_CONTACT = 1 << 23    # a body closed > 5 mm in one step on a static part thin enough to pass (GAP-E2; health_tunnel.py)
    TUNNEL = 1 << 24          # a body passed through a static shape (INVALID; GAP-E2; health_tunnel.py)
    QUASI_STATIC = 1 << 25    # a mass-scaled j2 solid moved fast enough that its scaled inertia shows (GAP #6)


class Severity(enum.IntEnum):
    OK = 0
    DEGRADED = 1
    INVALID = 2


@wp.struct
class StepHealth:
    flags: wp.uint32
    severity: wp.int32
    first_bad_step: wp.int64
    solver_iters: wp.int32
    residual: wp.float32                    # NaN = solver reports none
    max_pen_m: wp.float32                   # NaN = not measured (no contact source covered the world)
    max_pair_pen_m: wp.float32
    min_pair_gap_m: wp.float32
    max_joint_err_m: wp.float32
    max_joint_err_rad: wp.float32
    max_speed_mps: wp.float32
    max_abs_pos_m: wp.float32
    pen_deep_s: wp.float32                  # GAP-E1: time the native contact depth has stayed past its PEN_FAIL limit
                                            # (here, before energy_j, it fills the 4 bytes an 8-byte field would pad)
    energy_j: wp.float64
    argmax_body: wp.int32
    pen_measured: wp.int32                  # 1: a contact source covered the world in this check (SCR-SC0-1)
    cut_front_pen_m: wp.float32             # SCR-SC5-3: deepest exempt cut-front soft contact since the record reset
    cut_front_steps: wp.int32               # SCR-SC5-3: checks with an exempt cut-front contact past the PEN limit
    peak_pen_m: wp.float32                  # GAP-E1: largest max_pen_m since the record's reset (0: none measured)
    pen_hard_n: wp.int32                    # GAP-E1: consecutive checks a contact has stayed past its hard limit
    max_pair_id: wp.int32                   # the precision pair of max_pair_pen_m (GapMonitor; -1 when not given)
    plugin_check: wp.int32                  # SCR-SC11-1: first registered plug-in depth check over its limit (-1: none)
    plugin_depth_m: wp.float32              # its depth; when none failed, the deepest registered depth this check
    # First-failure snapshot (coordinator ruling 2026-11-04 05:40Z): the values of the check that first made the world
    # INVALID (that check is ``first_bad_step``), kept beside the latest values. ``fail_flags`` 0: never INVALID.
    fail_flags: wp.uint32
    fail_time_s: wp.float64                 # sim time of that check (NaN when the caller passes no dt)
    fail_pen_m: wp.float32                  # max_pen_m then; the deepest contact's shapes / particle (-1: none)
    fail_shape_a: wp.int32
    fail_shape_b: wp.int32
    fail_particle: wp.int32
    fail_pair_pen_m: wp.float32             # max_pair_pen_m then, and its pair (-1: not given)
    fail_pair_id: wp.int32
    fail_speed_mps: wp.float32              # max_speed_mps then, and its body (argmax_body)
    fail_body: wp.int32
    fail_joint_err_m: wp.float32
    fail_joint_err_rad: wp.float32
    fail_energy_j: wp.float64
    fail_plugin_check: wp.int32             # plugin_check / plugin_depth_m then (-1 / NaN: none)
    fail_plugin_depth_m: wp.float32
    seen_flags: wp.uint32                   # every flag any check raised since the record reset (GAP-E3 ruling)
    pad_end: wp.int32                       # always 0: rounds the record to 8 bytes with a real field. No implicit
    # padding anywhere (law: test_step_health_has_no_padding): padding bytes are never written, so they held stack
    # garbage in eager runs and broke the bitwise graph-vs-eager check (main red 16:45Z Nov 10).


FAIL_FIELDS = ("flags", "time_s", "pen_m", "shape_a", "shape_b", "particle", "pair_pen_m", "pair_id", "speed_mps",
               "body", "joint_err_m", "joint_err_rad", "energy_j", "plugin_check", "plugin_depth_m")


@dataclass(frozen=True, eq=False)
class DepthCheck:
    """A plug-in's per-world depth check (SCR-SC11-1, ruling 2026-11-04 18:25Z): ``depth`` float32 [world_count] in
    metres on the monitor's device (0 or NaN: nothing crossed); deeper than ``limit_m`` is PEN_FAIL named ``name``."""

    name: str
    depth: Any
    limit_m: float


@dataclass(frozen=True, eq=False)
class FirstFailure:
    """[W] arrays: the record's ``fail_*`` snapshot (the check that first made each world INVALID). ``flags`` 0 means
    the world never went INVALID; its other values are then NaN / -1."""

    step: np.ndarray
    flags: np.ndarray
    time_s: np.ndarray
    pen_m: np.ndarray
    shape_a: np.ndarray
    shape_b: np.ndarray
    particle: np.ndarray
    pair_pen_m: np.ndarray
    pair_id: np.ndarray
    speed_mps: np.ndarray
    body: np.ndarray
    joint_err_m: np.ndarray
    joint_err_rad: np.ndarray
    energy_j: np.ndarray
    plugin_check: np.ndarray
    plugin_depth_m: np.ndarray
    plugin_names: tuple[str, ...] = ()      # registered plug-in depth checks, by index

    def world(self, w: int) -> dict | None:
        """The snapshot of world ``w`` as plain values, or None if it never went INVALID."""
        if int(self.flags[w]) == 0:
            return None
        out = {"step": int(self.step[w]), "flags": [f.name for f in HealthFlags if int(self.flags[w]) & int(f)]}
        for name in FAIL_FIELDS[1:]:
            v = getattr(self, name)[w]
            out[name] = int(v) if np.issubdtype(np.asarray(v).dtype, np.integer) else float(v)
        return out

    def text(self, w: int) -> str:
        """One line for reports: what tripped world ``w`` and the values at that check."""
        f = self.world(w)
        if f is None:
            return "never INVALID"
        t = f"t {f['time_s']:.4g} s" if np.isfinite(f["time_s"]) else "t n/a"
        parts = [f"check {f['step']} ({t}): {'|'.join(f['flags'])}"]
        if np.isfinite(f["pen_m"]):
            who = (f"shapes {f['shape_a']}/{f['shape_b']}" if f["particle"] < 0
                   else f"particle {f['particle']} / shape {f['shape_a']}")
            parts.append(f"pen {f['pen_m']:.3g} m ({who})")
        if f["pair_pen_m"] > 0.0:
            parts.append(f"pair {f['pair_id']} pen {f['pair_pen_m']:.3g} m")
        if f["plugin_check"] >= 0:
            i = f["plugin_check"]
            name = self.plugin_names[i] if i < len(self.plugin_names) else f"#{i}"
            parts.append(f"plug-in {name} depth {f['plugin_depth_m']:.3g} m")
        parts.append(f"speed {f['speed_mps']:.3g} m/s (body {f['body']})")
        return "; ".join(parts)


@dataclass(frozen=True, eq=False)
class HealthReport:
    severity: np.ndarray                    # [W]
    flags: np.ndarray                       # [W]
    first_bad_step: np.ndarray              # [W]
    max_pen_m: np.ndarray
    max_pair_pen_m: np.ndarray
    max_speed_mps: np.ndarray
    invalid_worlds: tuple[int, ...]
    det_mode: Determinism
    det_reasons: tuple[str, ...]
    # [W] bool: max_pen_m was measured (SCR-SC0-1); where False, max_pen_m is NaN and PEN_WARN/PEN_FAIL could not
    # trip. None: the producer does not say (fakes).
    pen_measured: np.ndarray | None = None
    # [W] SCR-SC5-3: cut-front soft contacts (a Blade x a twin whose bond it is cutting, phase 1) are outside the PEN
    # rule; their deepest penetration since the record reset and the number of checks they went past the PEN limit
    cut_front_pen_m: np.ndarray | None = None
    cut_front_steps: np.ndarray | None = None
    # [W] the first-failure snapshot of each world (None: the producer keeps none, e.g. fakes)
    first_failure: FirstFailure | None = None
    # SCR-WP5-14: [W] deepest start overlap (m; NaN: none measured) and its two shapes [W, 2]; [W] the run's fastest
    # spin (rad per step) and its body; the model's shape and body names (None / empty: the producer keeps none)
    start_pen_m: np.ndarray | None = None
    start_shapes: np.ndarray | None = None
    start_level: np.ndarray | None = None           # [W] 0 none, 1 past pen_warn, 2 past pen_fail
    # {world: [(shape a, shape b, depth m), ...]} every distinct pair past pen_fail at the start, deepest first
    start_pairs: dict[int, list[tuple[int, int, float]]] | None = None
    spin_limit: float = 0.5
    spin_rad: np.ndarray | None = None
    spin_body: np.ndarray | None = None
    shape_names: tuple[str, ...] = ()
    body_names: tuple[str, ...] = ()
    # SCR-WP5-17: [W] the run's largest cloth triangle stretch (0.5 = 50 %), its triangle and that triangle's first
    # particle (-1: none); the limits; [W] the cloth's name where the API knows it
    cloth_stretch: np.ndarray | None = None
    cloth_tri: np.ndarray | None = None
    cloth_particle: np.ndarray | None = None
    cloth_stretch_warn: float = 0.5
    cloth_stretch_fail: float = 5.0
    cloth_names: tuple[str, ...] = ()
    seen_flags: np.ndarray | None = None    # [W] every flag any check raised since the reset (None: fakes)
    # PHYSA-1: [W] the run's largest energy gain over the floor, of E_scale (``health_energy_gain``; None: the energy
    # check is off or the producer keeps none), and the gain that raises ENERGY_JUMP
    energy_gain: np.ndarray | None = None
    energy_gain_limit: float = 0.10
    # GAP-E2: [W] the texts of FAST_CONTACT / TUNNEL where the API knows the names ('': none)
    tunnel_text: tuple[str, ...] = ()
    fast_text: tuple[str, ...] = ()
    peak_pen_m: np.ndarray | None = None    # [W] GAP-E1: largest max_pen_m since the record's reset (0: none measured)
    # [W] the most VBD self-contact crossings one detection counted (self_contact_cross.py); INVALID with the switch on
    self_contact_crossed: np.ndarray | None = None
    self_contact_cross_invalid: bool = False

    def failure_text(self, world: int) -> str:
        """What first made ``world`` INVALID, with the values at that check (summaries use this, not the latest
        ``max_*`` values: a latched world is held at its last good pose and reads clean afterwards)."""
        if self.first_failure is None:
            return f"INVALID at step {int(self.first_bad_step[world])}" if world in self.invalid_worlds else "OK"
        return self.first_failure.text(world)

    @classmethod
    def ok(cls, world_count: int, det_mode: Determinism = Determinism.RUN_TO_RUN) -> HealthReport:
        """All worlds OK (fakes, scenes without a monitor): nothing was measured, so max_pen_m is NaN."""
        z = np.zeros(world_count)
        return cls(np.zeros(world_count, np.int32), np.zeros(world_count, np.uint32),
                   np.full(world_count, -1, np.int64), np.full(world_count, np.nan), z.copy(), z.copy(), (), det_mode,
                   (), np.zeros(world_count, bool))

    def pen_text(self, world: int) -> str:
        """``max_pen_m`` of one world for reports: metres, or "not measured"."""
        measured = self.pen_measured is None or bool(self.pen_measured[world])
        value = float(self.max_pen_m[world])
        return f"{value:.3g} m" if measured and np.isfinite(value) else "not measured"


@dataclass(eq=False)
class PrecisionObservables(SolverBase.Observables):
    """Residual and iterations per world, for Alloy plug-ins only (stock solvers use health adapters; Newton's
    ``validate_observables`` requires the solver's own Observables class, solver.py:909)."""

    residual: wp.array | None = SolverBase.Observables.field(dtype=wp.float32, frequency=Model.AttributeFrequency.WORLD)
    iterations: wp.array | None = SolverBase.Observables.field(dtype=wp.int32, frequency=Model.AttributeFrequency.WORLD)


@dataclass(frozen=True)
class RepeatResult:
    case: str
    world_counts: tuple[int, ...]
    equal: Mapping[int, bool]                         # world count -> all digests equal to the reference
    first_divergent_step: Mapping[int, int | None]
    native_share: float                               # share of world-steps on NATIVE paths (D9)
    det_mode: Determinism
    reasons: tuple[str, ...]
    notes: Mapping[str, str] = field(default_factory=dict)
