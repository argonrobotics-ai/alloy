# SPDX-License-Identifier: Apache-2.0
"""WP3 stepped law twins A7.hydroelastic and A8.push (bench §4 A7/A8, SPEC §5.3), run by ``alloy.bench.run_law``.

Both run SolverMuJoCo with MuJoCo-native contacts plus the precision rows that ``AlloyContactLayer`` appends
(``native="mujoco"``, D10). ``make_pipeline`` builds the layer, installs the append on the solver and returns a
``LayerPipeline``, which supplies the ``dt`` that ``run_law`` does not pass to ``collide``. Every world is built in the
same template shape order (shape 0 the static partner, shape 1 the moving body), so the layer finds the pair in world
``w`` at ``shape_world_start[w] + (0, 1)``. Texture SDFs need CUDA: on CPU both laws report ``not_run`` (``law_device``).

A7.hydroelastic: a steel sphere (R = 10 mm) pressed into a plate, both hydroelastic with Newton's linear pressure law
``p = kh·depth`` (``linear_pressure``). On the iso-pressure surface ``kh_a·d_a = kh_b·d_b`` with ``d_a + d_b = h``
(the local overlap), so ``p = k_eff·h`` with ``k_eff = kh_a·kh_b/(kh_a + kh_b)`` (``get_effective_stiffness``).
Pressed δ into a flat plate, h(r) = δ − r²/(2R) for r ≤ √(2Rδ), so ``F = ∫ k_eff·h·2πr dr = π·R·k_eff·δ²``. Exact
values sit slightly lower: vertical columns give the cap volume ``π·k_eff·δ²·(R − δ/3)``; Newton evaluates p on the
iso-surface sdf_a = sdf_b (equal kh), where h = δ − r²/(2R − δ) and ``F = π·k_eff·δ²·(R − δ/2)``, at most 2.5 % below
the small-δ form over δ/R ≤ 0.05 (fitted exponent ≈ 1.99). One world per load: the sphere rides a frictionless
vertical slide (an indenter guide) with condim-1 contact rows, so the normal pressure law alone carries the load
(MuJoCo's soft friction on the dish-shaped patch's tilted rows could carry part of it); zero gravity, smoothstep
ease-in, hold; δ = R − (z_centre − z_top) from the pose. Pass: fitted exponent 2 ± 0.05, every world's prefactor
F/(π·R·k_eff·δ²) within 10 %, and δ settled (≤ 1 % change between the halves of the averaging window).

A8.push: the CON-46 peg (12.0 mm) seated in its 12.2 mm blind bore, sdf mode, pair K = 2·50 N / max_pen (D22), dt =
the pair's §3.5 L2 step; settle 0.1 s, ramp 0 → 50 N down over 0.5 s, hold 0.1 s. GapMonitor (exact pseudo-normal
geometry, never the narrow phase) measures penetration from the start of the ramp. Pass: every world's maximum ≤ the
pair's max_penetration (D2; SPEC §5.3 amendment A8).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.bench.analytic.common import BaseTwin, smoothstep
from alloy.bench.twin import DeviceTwinLog, WorldRanges
from alloy.bench.types import Condition
from alloy.contact.pipeline import AlloyContactLayer
from alloy.contact.precision import pair_stiffness, size_l2
from alloy.contact.sdf_cache import build_feature_sdf
from alloy.contact.shadow import set_flat_solimp
from alloy.contact.types import PrecisionPairSpec
from alloy.core.errors import LawError
from alloy.core.log import get_logger
from alloy.testing.contact import (HYDRO_GAP_M, PegHoleSpec, blind_hole_tris, cylinder_tris, signed_volume,
                                  to_newton_mesh)

log = get_logger("contact.laws")

DESIGN_FORCE_N = 50.0        # the A8 ramp and the pair's D22 design force
CONTROL_DT = 0.02            # SceneCard default control period (D4 grid)
CONTACT_SPEED = 0.05         # m/s, slow-approach card used to size the L2 step (as the CT tests)
A7_DT = 6.25e-4              # s, D4 grid (0.02/2^5); MJWarp clamp m_eff/((1-imp)·4dt²) = 6e6 N/m per row (slide:
                             # m_eff = 1/invweight0 = 3m) ≫ the whole patch's area·k_eff ≤ 3.2e5 N/m
A7_EASE_S = 0.15
A7_AVG_S = 0.1               # δ is averaged over the last 0.1 s of the hold
A7_DURATION_S = 2.0          # rows share one DOF, so τ_eff ≈ √N_rows·τ(K_total) ~ 0.1-0.2 s at the softest load
                             # (measured: 5 % short at 0.5 s, settled to 1e-4 from 1.0 s on)
A7_HISTORY_S = (0.5, 1.0, 1.5, 2.0)        # ends of the 0.1 s windows reported as δ(t) (settling evidence)
DELTA_OVER_R = (0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05)   # bench §4 A7 range δ/R ∈ [0.005, 0.05]


# ---------------------------------------------------------------------------------------------- closed forms ---

def series_stiffness(kh_a: float, kh_b: float) -> float:
    """Newton ``get_effective_stiffness``: two linear pressure fields in series [Pa/m]."""
    total = kh_a + kh_b
    return 0.0 if total <= 0.0 else kh_a * kh_b / total


def hydro_sphere_force(radius: float, k_eff: float, delta):
    """Small-δ closed form of a sphere pressed δ into a flat hydroelastic half-space: ``π·R·k_eff·δ²`` [N]."""
    return math.pi * radius * k_eff * np.square(delta)


def power_law_exponent(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope of log y against log x; NaN unless every value is positive and finite."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if x.size < 2 or not (np.all(np.isfinite(x)) and np.all(x > 0) and np.all(np.isfinite(y)) and np.all(y > 0)):
        return math.nan
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def finite_columns(t: np.ndarray, series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Times and ``[W, T]`` samples restricted to columns recorded in every world (a stride can leave the last
    column unwritten)."""
    s = np.asarray(series, np.float64)
    ok = np.all(np.isfinite(s), axis=0)
    return np.asarray(t, np.float64)[ok], s[:, ok]


def window_mean(t: np.ndarray, series: np.ndarray, end: float, span: float) -> np.ndarray:
    """Per-world mean of ``series`` ``[W, T]`` over the samples with ``end - span < t <= end`` (NaN if none)."""
    sel = (t > end - span - 1e-12) & (t <= end + 1e-12)
    if not sel.any():
        return np.full(series.shape[0], np.nan)
    return series[:, sel].mean(axis=1)


# ----------------------------------------------------------------------------------------- shared plumbing ---

@dataclass
class LayerPipeline:
    """``run_law`` contact source: the layer collided with a fixed step (``run_law`` calls ``collide(state, c)``)."""

    layer: AlloyContactLayer
    dt: float

    def contacts(self) -> newton.Contacts:
        return self.layer.contacts()

    def collide(self, state: newton.State, contacts: newton.Contacts) -> None:
        self.layer.collide(state, contacts, dt=self.dt)


@dataclass
class BodyPose:
    """The part of a ``State`` GapMonitor reads for the previous pose (tunnel test)."""

    body_q: wp.array


@dataclass(frozen=True)
class MjSettings:
    """SolverMuJoCo settings of the contact laws (the CT peg harness: §3.5 elliptic cone, impratio 300, euler)."""

    integrator: str = "euler"
    cone: str = "elliptic"
    impratio: float = 300.0
    iterations: int = 100
    ls_iterations: int = 50
    nconmax: int = 256
    njmax: int = 1024

    def kwargs(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@wp.kernel
def _push_down(bodies: wp.array(dtype=wp.int32), force: wp.array(dtype=float), scale: float,
               body_f: wp.array(dtype=wp.spatial_vector)):
    w = wp.tid()
    body_f[bodies[w]] = wp.spatial_vector(0.0, 0.0, -force[w] * scale, 0.0, 0.0, 0.0)  # (force, torque) at the COM


def set_condim(b: newton.ModelBuilder, shape: int, condim: int) -> None:
    """Shape attribute ``mujoco:condim`` (1 = frictionless rows); the precision append mixes it like MuJoCo (max)."""
    key = "mujoco:condim"
    if not b.has_custom_attribute(key):
        newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    attr = b.custom_attributes[key]
    if attr.values is None:
        attr.values = {}
    attr.values[int(shape)] = attr.dtype(condim)


def filter_precision_pair(b: newton.ModelBuilder, shape_a: int, shape_b: int, imp: float) -> None:
    """Keep the pair out of native collision (the layer collides it) and give both shapes the flat solimp (D22)."""
    b.add_shape_collision_filter_pair(shape_a, shape_b)
    for shape in (shape_a, shape_b):
        set_flat_solimp(b, shape, imp)


class _ContactLawTwin(BaseTwin):
    """One precision pair (template shapes 0, 1) per world; SolverMuJoCo + the precision append (D10).

    Subclasses define ``pair_spec()``, ``dt`` and the ``force`` condition parameter (N, pushed down on each world's
    first body by ``push``)."""

    dt: float = 0.0
    mj: MjSettings = MjSettings()
    contacts_per_world: int = 128

    def __init__(self) -> None:
        super().__init__()
        self.solver: newton.solvers.SolverMuJoCo | None = None
        self.layer: AlloyContactLayer | None = None
        self.binding: Any = None
        self._force: wp.array | None = None

    def pair_spec(self) -> PrecisionPairSpec:
        raise NotImplementedError

    def layer_kwargs(self) -> dict[str, Any]:
        return {}

    def make_solver(self, model: newton.Model, solver: str) -> newton.solvers.SolverMuJoCo:
        if solver != "mujoco":
            raise LawError(f"{self.law_id} runs on SolverMuJoCo with the precision append, not {solver!r}")
        self.model, self.solver_name = model, solver
        self.ranges = WorldRanges.bodies(model)
        self._first_body = self.ranges.start
        self._force = wp.array(self.param("force").astype(np.float32), dtype=float, device=model.device)
        self.on_model(model)
        self.solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=True, **self.mj.kwargs())
        return self.solver

    def make_pipeline(self, model: newton.Model, solver: str) -> LayerPipeline:
        self.layer = AlloyContactLayer(model, [self.pair_spec()], native="mujoco",
                                       precision_contacts_per_world=self.contacts_per_world, **self.layer_kwargs())
        self.binding = self.layer.bind_mujoco(self.solver)
        self.binding.install()
        log.info("%s: layer bound to SolverMuJoCo, %d worlds, dt %.4g s", self.law_id, model.world_count, self.dt)
        return LayerPipeline(self.layer, self.dt)

    def settings_record(self, solver: str) -> dict[str, Any]:
        return {"solver": solver, "kwargs": self.mj.kwargs(), "use_mujoco_contacts": True, "native": "mujoco",
                "precision_pair": dataclasses.asdict(self.pair_spec()), "dt": self.dt}

    def push(self, state: Any, scale: float) -> None:
        """Downward force ``force · scale`` on each world's first body (device only)."""
        wp.launch(_push_down, dim=len(self.worlds), inputs=[self._first_body, self._force, float(scale)],
                  outputs=[state.body_f], device=self.model.device)

    def overflow(self) -> dict[str, int]:
        """Dropped rows per cause (host readback after the run; empty before the layer exists)."""
        if self.layer is None:
            return {}
        rep = self.layer.overflow_report()
        rep["append"] = int(self.binding.overflow.numpy()[0])
        rep["mjwarp"] = int(np.asarray(self.solver.mjw_data.overflow.numpy()).max())
        return rep


# ----------------------------------------------------------------------------------------------------- A7 ---

@dataclass(frozen=True)
class SpherePlate:
    """A7 geometry and material: a steel sphere on a plate (top face at z = 0), both hydroelastic with one kh."""

    radius: float = 0.01                                    # m
    density: float = 7800.0                                 # kg/m^3, steel
    kh: float = 2.0e10                                      # Pa/m per shape -> k_eff = 1e10 Pa/m
    plate_half: tuple[float, float, float] = (6.0e-3, 6.0e-3, 1.5e-3)   # contact radius ≤ √(2Rδ) = 3.2 mm
    voxel: float = 1.0e-4                                   # m, Newton texture SDF of both primitives
    band: tuple[float, float] = (-1.0e-3, 1.0e-3)
    gap: float = HYDRO_GAP_M                                # small: Newton has no speculative hydroelastic rows
    condim: int = 1                                         # frictionless normal rows (module docstring)

    @property
    def k_eff(self) -> float:
        return series_stiffness(self.kh, self.kh)

    def iso_voxels_per_world(self, delta_max: float) -> float:
        """Iso voxels the pair needs: 8 layers over the band patch π·2R(δ + gaps + √3·h) (measured need is ~2;
        the layer's default A/h² over the whole sphere would split 8 worlds over 3 pipelines)."""
        reach = delta_max + 2.0 * self.gap + math.sqrt(3.0) * self.voxel
        return 8.0 * math.pi * 2.0 * self.radius * reach / (self.voxel * self.voxel)


class HydroelasticSphereTwin(_ContactLawTwin):
    """A7.hydroelastic (module docstring)."""

    law_id = "A7.hydroelastic"
    units: Mapping[str, str] = {"exponent_err": "1", "prefactor_rel_err": "1", "settle_rel_err": "1"}
    dt = A7_DT
    geometry = SpherePlate()

    def conditions(self, quick: bool) -> tuple[Condition, ...]:
        g = self.geometry
        return tuple(Condition(f"dR{r:g}", {"force": float(hydro_sphere_force(g.radius, g.k_eff, r * g.radius)),
                                            "delta_closed": r * g.radius}) for r in DELTA_OVER_R)

    def world_gravity(self, cond: Condition) -> tuple[float, float, float]:
        return (0.0, 0.0, 0.0)

    def pair_spec(self) -> PrecisionPairSpec:
        g = self.geometry
        return PrecisionPairSpec(0, 1, mode="hydroelastic", voxel_size=g.voxel, narrow_band=g.band, gap=g.gap,
                                 kh=g.kh, ccd="off")

    def layer_kwargs(self) -> dict[str, Any]:
        return {"hydro_iso_per_world": self.geometry.iso_voxels_per_world(max(DELTA_OVER_R) * self.geometry.radius)}

    def _shape_cfg(self) -> newton.ModelBuilder.ShapeConfig:
        g = self.geometry
        cfg = newton.ModelBuilder.ShapeConfig(density=g.density, gap=g.gap, margin=0.0, is_hydroelastic=True, kh=g.kh)
        cfg.sdf_target_voxel_size = g.voxel          # primitives: Newton bakes the texture SDF at finalize (CUDA)
        cfg.sdf_narrow_band_range = g.band
        return cfg

    def build(self, b: newton.ModelBuilder, cond: Condition, sample: Mapping[str, float]) -> None:
        g, cfg = self.geometry, self._shape_cfg()
        hx, hy, hz = g.plate_half
        plate = b.add_shape_box(-1, xform=wp.transform((0.0, 0.0, -hz), wp.quat_identity()), hx=hx, hy=hy, hz=hz,
                                cfg=cfg, label="plate")
        start = wp.transform((0.0, 0.0, g.radius), wp.quat_identity())      # touching the plate
        link = b.add_link(xform=start, label="sphere")
        sphere = b.add_shape_sphere(link, radius=g.radius, cfg=cfg, label="sphere")
        joint = b.add_joint_prismatic(-1, link, parent_xform=start, child_xform=wp.transform_identity(),
                                      axis=(0.0, 0.0, 1.0))
        b.add_articulation([joint])
        filter_precision_pair(b, plate, sphere, self.pair_spec().mj_imp)
        for shape in (plate, sphere):
            set_condim(b, shape, g.condim)

    def drive(self, step: int, t: float, state: Any, control: Any) -> None:
        self.push(state, smoothstep(t / A7_EASE_S))

    def record(self, step: int, state: Any, contacts: Any, obs: Any, log_: DeviceTwinLog) -> None:
        self.record_first_body(step, state, log_)

    def scalars(self, log_: DeviceTwinLog) -> dict[str, np.ndarray]:
        g = self.geometry
        t, z = finite_columns(log_.times(), log_.series("z"))
        depth = g.radius - z                                    # δ(t) per world; plate top face at z = 0
        delta = window_mean(t, depth, t[-1], A7_AVG_S)
        settle = np.abs(window_mean(t, depth, t[-1], 0.5 * A7_AVG_S)
                        - window_mean(t, depth, t[-1] - 0.5 * A7_AVG_S, 0.5 * A7_AVG_S)) / np.abs(delta)
        force = self.param("force")
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = force / hydro_sphere_force(g.radius, g.k_eff, delta)
        exponent = power_law_exponent(delta, force)
        self.monitors.update({
            "k_eff_pa_per_m": g.k_eff, "radius_m": g.radius, "force_n": force.tolist(), "delta_m": delta.tolist(),
            "delta_closed_m": self.param("delta_closed").tolist(), "prefactor_ratio": ratio.tolist(),
            "exponent": exponent, "settle_rel": settle.tolist(), "overflow": self.overflow(),
            "delta_history_m": {f"{end:g}s": window_mean(t, depth, end, A7_AVG_S).tolist()
                                for end in A7_HISTORY_S if end <= t[-1] + 1e-9}})
        log.info("A7.hydroelastic: exponent %.4f, prefactor ratios %s", exponent, np.round(ratio, 4).tolist())
        return {"exponent_err": np.array([abs(exponent - 2.0)]), "prefactor_rel_err": np.abs(ratio - 1.0),
                "settle_rel_err": settle}


# ----------------------------------------------------------------------------------------------------- A8 ---

@dataclass(frozen=True)
class PushPlan:
    """A8 ramp: settle, linear ramp to ``force_n`` (CT-1's 'A8 ramp'), hold; GapMonitor counts from the ramp."""

    settle_s: float = 0.1
    ramp_s: float = 0.5
    hold_s: float = 0.1
    force_n: float = DESIGN_FORCE_N
    start_gap: float = 5.0e-5          # peg bottom above the bore bottom (clears ±0.4° start tilts)

    @property
    def duration_s(self) -> float:
        return self.settle_s + self.ramp_s + self.hold_s

    def scale(self, t: float) -> float:
        """Fraction of ``force_n`` applied at time ``t``."""
        return min(1.0, max(0.0, (t - self.settle_s) / self.ramp_s))


def push_pair(geometry: PegHoleSpec | None = None) -> PrecisionPairSpec:
    """The CON-46 sdf pair (hole = template shape 0, peg = 1), K = 2·50 N/max_pen (D22; 2e6 N/m)."""
    g = geometry or PegHoleSpec()
    spec = PrecisionPairSpec(0, 1, mode="sdf", voxel_size=5.0e-5, clearance=g.bore_d - g.peg_d, gap=5.0e-4,
                             narrow_band=(-5.0e-4, 5.0e-4))
    return dataclasses.replace(spec, pair_stiffness=pair_stiffness(spec, DESIGN_FORCE_N))


def peg_mass(geometry: PegHoleSpec | None = None, density: float = 7800.0) -> float:
    """Mass of the meshed peg [kg] (Newton computes it from the same mesh)."""
    g = geometry or PegHoleSpec()
    return density * signed_volume(*cylinder_tris(0.5 * g.peg_d, 0.5 * g.peg_len, g.segments))


def push_dt(geometry: PegHoleSpec | None = None) -> float:
    """The pair's §3.5 L2 step on the D4 grid for a 50 N design force and a 0.05 m/s approach (3.125e-4 s)."""
    return size_l2(push_pair(geometry), m_eff=peg_mass(geometry), design_force=DESIGN_FORCE_N,
                   contact_speed=CONTACT_SPEED, control_dt=CONTROL_DT).dt


def peg_start_xform(g: PegHoleSpec, start_gap: float, tilt_deg: float = 0.0, lateral: float = 0.0) -> wp.transform:
    """Peg centre ``start_gap`` above the bore bottom, tilted ``tilt_deg`` about X, shifted ``lateral`` in x."""
    q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.radians(tilt_deg))
    return wp.transform((lateral, 0.0, g.bottom_z + start_gap + 0.5 * g.peg_len), q)


def add_peg_hole(b: newton.ModelBuilder, meshes: tuple[newton.Mesh, newton.Mesh], cfg: newton.ModelBuilder.ShapeConfig,
                 xform: wp.transform, imp: float) -> tuple[int, int]:
    """Static blind hole (shape 0 of the world) and the peg body (shape 1), as ``peg_hole_scene`` lays them out."""
    hole_mesh, peg_mesh = meshes
    hole = b.add_shape_mesh(-1, mesh=hole_mesh, cfg=cfg, label="hole")
    body = b.add_body(xform=xform, label="peg")
    peg = b.add_shape_mesh(body, mesh=peg_mesh, cfg=cfg, label="peg")
    filter_precision_pair(b, hole, peg, imp)
    return hole, peg


class PegPushTwin(_ContactLawTwin):
    """A8.push (module docstring). Worlds: sample 0 centred and upright, the rest perturbed by ``LawSpec.perturb``
    (``tilt_deg`` about X, ``lateral_m`` along x)."""

    law_id = "A8.push"
    units: Mapping[str, str] = {"pen_max_m": "m"}
    geometry = PegHoleSpec()
    plan = PushPlan()
    contacts_per_world = 192
    monitor_spacing = 5.0e-4

    def __init__(self) -> None:
        super().__init__()
        self.dt = push_dt(self.geometry)
        self.gap_monitor: Any = None
        self._meshes: tuple[newton.Mesh, newton.Mesh] | None = None
        self._prev: BodyPose | None = None
        self._ramp_step = int(round(self.plan.settle_s / self.dt))

    def conditions(self, quick: bool) -> tuple[Condition, ...]:
        return (Condition("push50N", {"force": self.plan.force_n}),)

    def pair_spec(self) -> PrecisionPairSpec:
        return push_pair(self.geometry)

    def meshes(self) -> tuple[newton.Mesh, newton.Mesh]:
        """(hole, peg) meshes with texture SDFs baked once on the current Warp device (CUDA only)."""
        if self._meshes is None:
            g, spec = self.geometry, self.pair_spec()
            peg = to_newton_mesh(*cylinder_tris(0.5 * g.peg_d, 0.5 * g.peg_len, g.segments))
            hole = to_newton_mesh(*blind_hole_tris(g), compute_inertia=False)
            device = str(wp.get_device())
            for mesh in (hole, peg):
                build_feature_sdf(mesh, spec, device=device)
            self._meshes = (hole, peg)
        return self._meshes

    def build(self, b: newton.ModelBuilder, cond: Condition, sample: Mapping[str, float]) -> None:
        spec = self.pair_spec()
        cfg = newton.ModelBuilder.ShapeConfig(density=7800.0, mu=0.3, gap=spec.gap, margin=0.0)
        xform = peg_start_xform(self.geometry, self.plan.start_gap, sample.get("tilt_deg", 0.0),
                                sample.get("lateral_m", 0.0))
        add_peg_hole(b, self.meshes(), cfg, xform, spec.mj_imp)

    def make_pipeline(self, model: newton.Model, solver: str) -> LayerPipeline:
        pipe = super().make_pipeline(model, solver)
        self.gap_monitor = self.layer.monitor(sample_spacing=self.monitor_spacing)
        return pipe

    def drive(self, step: int, t: float, state: Any, control: Any) -> None:
        self.push(state, self.plan.scale(t))

    def record(self, step: int, state: Any, contacts: Any, obs: Any, log_: DeviceTwinLog) -> None:
        self.record_first_body(step, state, log_)
        if step == 0:
            self._prev = BodyPose(wp.clone(state.body_q))
            return
        if step == self._ramp_step:
            self.gap_monitor.reset()                # the law judges the ramp and the hold
        self.gap_monitor.update(self._prev, state)
        wp.copy(self._prev.body_q, state.body_q)

    def scalars(self, log_: DeviceTwinLog) -> dict[str, np.ndarray]:
        rep = self.gap_monitor.report()
        n = len(self.worlds)
        final = self.gap_monitor.acc.step_max.numpy()[:n].astype(np.float64)
        _, z = finite_columns(log_.times(), log_.series("z"))
        self.monitors.update({
            "max_penetration_m": self.pair_spec().resolved_max_penetration(), "world_max_m": rep.world_max_m.tolist(),
            "final_pen_m": final.tolist(), "p99_m": float(rep.pair_p99_m[0]), "samples": int(rep.pair_samples[0]),
            "tunnels": int(rep.tunnel_events.sum()), "reference": rep.reference[0], "dt": self.dt,
            "sink_m": (z[:, 0] - z[:, -1]).tolist(), "overflow": self.overflow()})
        log.info("A8.push: max penetration %s m, p99 %.3g m", np.round(rep.world_max_m, 8).tolist(),
                 rep.pair_p99_m[0])
        return {"pen_max_m": rep.world_max_m.astype(np.float64)}


__all__ = ["HydroelasticSphereTwin", "PegPushTwin", "SpherePlate", "PushPlan", "LayerPipeline", "MjSettings",
           "series_stiffness", "hydro_sphere_force", "power_law_exponent", "finite_columns", "window_mean",
           "set_condim", "push_pair", "push_dt", "peg_mass", "add_peg_hole", "peg_start_xform",
           "filter_precision_pair", "A7_DT", "A7_DURATION_S", "A7_AVG_S", "DELTA_OVER_R", "DESIGN_FORCE_N"]
