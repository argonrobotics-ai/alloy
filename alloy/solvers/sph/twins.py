# SPDX-License-Identifier: Apache-2.0
"""SimTwins of the liquid analytic laws A13 (hydrostatics, Archimedes, volume) and A14 (dam break), SPEC §5.3.

Each twin builds its scene per world through ``run_law`` (alloy.bench.runner), runs SolverDFSPH alone (D5
monolithic for the floating box), primes the hydrostatic pressure in ``init_state`` and records on device every
sampled step (probes.py). ``scalars`` returns per-world *errors* (relative or in units of the dam width), which
``judge_analytic`` compares against the LawSpec tolerances; raw values go to ``monitors``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import Any

from alloy._vendor import newton
import numpy as np
import warp as wp

from alloy.bench.types import Condition

from . import bodies as bd
from . import probes as pb
from .build import add_box_tank, add_floating_box, add_fluid_block, add_fluid_points, lattice_points
from .config import DFSPHConfig, register_custom_attributes

log = logging.getLogger("alloy.sph")

G = 9.81  # [m/s^2] |g| of Newton's default builder gravity
MAX_SAMPLES = 2000


class SPHTwin:
    """Shared hooks: one DFSPHConfig for all worlds, SolverDFSPH as the solver, no Newton contacts."""

    law_id: str = ""
    units: Mapping[str, str] = {}
    config_overrides: Mapping[str, Any] = {}

    def __init__(self, dt: float, r: float = 1.0e-3, same_size: bool = False) -> None:
        """``r``: particle radius of the conditions (1 mm: the registered rows, 2 mm spacing, liquid_accuracy
        "super"; 2 mm: the default 4 mm spacing, SCR-WP9-LIQRES). ``same_size``: the full condition's counts halved
        at r = 2 mm, so the scene has the r = 1 mm full condition's metres (fewer spacings; full mode only)."""
        self.dt = dt
        self.r = float(r)
        self.same_size = bool(same_size)
        self.worlds: list[Condition] = []
        self.monitors: dict[str, Any] = {}
        self.solver: Any = None
        self.model: Any = None
        self.config: DFSPHConfig | None = None

    # -- run_law hooks -----------------------------------------------------------------------------------------
    def prepare_builder(self, builder: newton.ModelBuilder, solver: str) -> None:
        register_custom_attributes(builder)

    def build_world(self, b: newton.ModelBuilder, cond: Condition, sample: Mapping[str, float]) -> None:
        cfg = DFSPHConfig(particle_radius=float(cond.params["r"]), **self.config_overrides)
        cfg.validate()
        if self.config is not None and self.config.particle_radius != cfg.particle_radius:
            raise ValueError(f"{self.law_id}: every world of one run needs the same particle radius")
        self.config = cfg
        self.worlds.append(cond)
        self.build(b, cond, cfg)

    def build(self, b: newton.ModelBuilder, cond: Condition, cfg: DFSPHConfig) -> None:
        raise NotImplementedError

    def make_solver(self, model: newton.Model, solver: str) -> Any:
        from .solver import SolverDFSPH

        if solver != "sph":
            raise ValueError(f"{self.law_id} runs on the 'sph' capability, not {solver!r}")
        self.model = model
        self.solver = SolverDFSPH(model, self.config)
        self.on_model(model)
        return self.solver

    def make_pipeline(self, model: newton.Model, solver: str) -> None:
        return None  # liquid-solid coupling happens inside SolverDFSPH

    def on_model(self, model: newton.Model) -> None:
        """Hook: device arrays once the model exists."""

    def init_state(self, model: newton.Model, state: newton.State) -> None:
        self.monitors["prime_iterations"] = self.solver.prime(state, self.dt)

    def drive(self, step: int, t: float, state: Any, control: Any) -> None:
        """Passive scenes."""

    def record_stride(self, n_steps: int) -> int:
        return max(1, -(-n_steps // MAX_SAMPLES))

    def settings_record(self, solver: str) -> dict[str, Any]:
        return {"config": repr(self.config), "dt": self.dt}

    def tag(self) -> str:
        """Condition name suffix: ``_r1mm``, ``_r2mm`` or ``_r2mm_samesize``."""
        return f"_r{self.r * 1e3:g}mm" + ("_samesize" if self.same_size else "")

    def counts(self, quick: bool, *values: float) -> tuple[float, ...]:
        """The condition's counts (spacings), halved for ``same_size`` (refused in quick mode: too few spacings)."""
        if not self.same_size:
            return values
        if quick:
            raise ValueError(f"{self.law_id}: same_size conditions are full-mode only")
        return tuple(0.5 * v for v in values)

    def param(self, key: str) -> np.ndarray:
        return np.array([c.params[key] for c in self.worlds], dtype=np.float64)

    def _launch(self, kernel, dim: int, inputs: list) -> None:
        wp.launch(kernel, dim=dim, inputs=inputs, device=self.model.device)


def _second_half(t: np.ndarray) -> int:
    return int(np.searchsorted(t, 0.5 * t[-1]))


class HydrostaticTwin(SPHTwin):
    """A13 / LIQ-4: force of a resting column on one tank wall vs rho0 g w H^2 / 2, averaged over the second half of
    the run; volume change over the run (LIQ-2's SPH volume). H is the law's own column, nz layers of spacing 2r
    (audit AB-laws-2: taken as SPH volume / base area it followed a wrong particle mass, and the wall force with
    it); the SPH volume's height stays as the monitor ``height_m``."""

    law_id = "A13.hydrostatic"
    units = {"wall_force": "1", "volume_loss": "1"}

    def __init__(self, dt: float, **kw: Any) -> None:
        super().__init__(dt, **kw)
        self._walls: list[int] = []

    def conditions(self, quick: bool) -> tuple[Condition, ...]:
        n, nz = self.counts(quick, 4 if quick else 8, 50)
        n, nz = int(n), int(nz)
        p = {"nx": n, "ny": n, "nz": nz, "r": self.r, "headroom": 8}
        return (Condition(f"column_{n}x{n}x{nz}{self.tag()}", p),)

    def build(self, b: newton.ModelBuilder, cond: Condition, cfg: DFSPHConfig) -> None:
        p, s = cond.params, cfg.spacing
        nx, ny, nz = int(p["nx"]), int(p["ny"]), int(p["nz"])
        tank = add_box_tank(b, (0.0, 0.0, 0.0), (nx * s, ny * s, (nz + int(p["headroom"])) * s), wall=cfg.h)
        add_fluid_block(b, cfg, (0.0, 0.0, 0.0), (nx, ny, nz))
        self._walls.append(tank["x_lo"])

    def on_model(self, model: newton.Model) -> None:
        self._wall_arr = wp.array(self._walls, dtype=wp.int32, device=model.device)
        self._vol = wp.zeros(len(self.worlds), dtype=wp.float64, device=model.device)

    def record(self, step: int, state: Any, contacts: Any, obs: Any, log_: Any) -> None:
        col = log_.column(step)
        if col < 0:
            return
        W = len(self.worlds)
        self._launch(pb.store_meter_kernel, W, [self.solver._meter_j, self._wall_arr, 0, bd.IMP_SCALE_HOST, col,
                                                log_.channel("jx")])
        pb.liquid_volume(self.solver, self._vol)
        self._launch(pb.store_column_kernel, W, [self._vol, col, log_.channel("volume")])

    def scalars(self, log_: Any) -> dict[str, np.ndarray]:
        t = log_.times()
        jx, vol = log_.series("jx").astype(np.float64), log_.series("volume").astype(np.float64)
        i0 = _second_half(t)
        force = -(jx[:, -1] - jx[:, i0]) / (t[-1] - t[i0])  # the liquid pushes the x_lo wall towards -x
        s = 2.0 * self.param("r")
        area = self.param("nx") * self.param("ny") * s * s
        height = vol[:, 0] / area                           # the solver's own column: a monitor, not the reference
        ref = 1000.0 * G * (self.param("ny") * s) * (self.param("nz") * s) ** 2 / 2.0
        loss = np.abs(vol[:, -1] - vol[:, 0]) / vol[:, 0]
        self.monitors.update(wall_force_N=force.tolist(), wall_force_ref_N=ref.tolist(), height_m=height.tolist(),
                             volume_change=((vol[:, -1] - vol[:, 0]) / vol[:, 0]).tolist())
        return {"wall_force": np.abs(force - ref) / ref, "volume_loss": loss}


class ArchimedesTwin(SPHTwin):
    """A13 / LIQ-4: draft of a free floating box vs m / (rho0 A_box) = rho_b x box height / rho0, from the law's own
    density and box (audit AB-laws-2: with m read from ``model.body_mass`` a box built too light floated high and
    moved the reference with it). The measured draft accounts for the surface rise: d = (V_liquid / A_tank -
    z_bottom) / (1 - A_box / A_tank). Box density 800 kg/m^3 with side >= 1.28 x height keeps the metacentric
    height positive (a cube of density 0.5 floats on an edge)."""

    law_id = "A13.archimedes"
    units = {"draft": "1"}
    config_overrides = {"integrate_bodies": "dynamic"}

    def __init__(self, dt: float, **kw: Any) -> None:
        super().__init__(dt, **kw)
        self._geo: list[dict] = []

    def conditions(self, quick: bool) -> tuple[Condition, ...]:
        nx, nz, bx, bz = self.counts(quick, *((16, 10, 8, 6.25) if quick else (32, 20, 16, 12.5)))
        p = {"nx": int(nx), "ny": int(nx), "nz": int(nz), "bx": int(bx), "bz": bz, "rho_b": 800.0, "r": self.r}
        return (Condition(f"box{int(bx)}_draft{bz * 0.8:g}{self.tag()}", p),)

    def build(self, b: newton.ModelBuilder, cond: Condition, cfg: DFSPHConfig) -> None:
        p = cond.params
        fb = add_floating_box(b, cfg, (int(p["nx"]), int(p["ny"]), int(p["nz"])), (p["bx"], p["bx"], p["bz"]),
                              float(p["rho_b"]))
        self._geo.append(fb)

    def on_model(self, model: newton.Model) -> None:
        self._bodies = wp.array([g["body"] for g in self._geo], dtype=wp.int32, device=model.device)
        self._vol = wp.zeros(len(self.worlds), dtype=wp.float64, device=model.device)
        self._mass = model.body_mass.numpy()[[g["body"] for g in self._geo]].astype(np.float64)

    def record(self, step: int, state: Any, contacts: Any, obs: Any, log_: Any) -> None:
        from alloy.bench.twin import record_body_position

        col = log_.column(step)
        if col < 0:
            return
        record_body_position(log_, "", state.body_q, self._bodies, step)
        if step == 0:
            pb.liquid_volume(self.solver, self._vol)
            self._launch(pb.store_column_kernel, len(self.worlds), [self._vol, col, log_.channel("volume")])

    def scalars(self, log_: Any) -> dict[str, np.ndarray]:
        t, z = log_.times(), log_.series("z").astype(np.float64)
        i0 = _second_half(t)
        v_liq = log_.series("volume").astype(np.float64)[:, 0]
        a_tank = np.array([g["area"] for g in self._geo])
        a_box = np.array([g["box_area"] for g in self._geo])
        hz = np.array([g["half"][2] for g in self._geo])
        bottom = z[:, i0:] - hz[:, None]
        draft = (v_liq[:, None] / a_tank[:, None] - bottom) / (1.0 - a_box / a_tank)[:, None]
        ref = self.param("rho_b") * self.param("bz") * 2.0 * self.param("r") / 1000.0
        mean = draft.mean(axis=1)
        self.monitors.update(draft_m=mean.tolist(), draft_ref_m=ref.tolist(),
                             draft_built_m=(self._mass / (1000.0 * a_box)).tolist(),    # from the built box's mass
                             draft_ptp_m=np.ptp(draft, axis=1).tolist(),
                             draft_over_2r=(ref / (2.0 * self.param("r"))).tolist())
        return {"draft": np.abs(mean - ref) / ref}


class DamBreakTwin(SPHTwin):
    """A14: Martin & Moyce dam break, n^2 = 2 (a x 2a column against the x_lo wall). The front is the leading edge on
    the floor (max x of liquid below 2 spacings, plus r); Z = x / a, T = t sqrt(2 g / a). Ritter's dry-bed front
    Z = 1 + 2T bounds it from above; ``ritter_excess`` = max(0, max_T<=3 (Z - 1 - 2T)). The Martin & Moyce NRMSE
    waits for the digitised curve (SPEC §8.9); the simulated Z(T) goes to ``monitors``."""

    law_id = "A14.dam_break"
    units = {"ritter_excess": "1"}

    def conditions(self, quick: bool) -> tuple[Condition, ...]:
        a, depth = self.counts(quick, *((12, 3) if quick else (50, 8)))
        return (Condition(f"a{int(a)}{self.tag()}", {"a": int(a), "depth": int(depth), "r": self.r}),)

    def build(self, b: newton.ModelBuilder, cond: Condition, cfg: DFSPHConfig) -> None:
        a, depth, s = int(cond.params["a"]), int(cond.params["depth"]), cfg.spacing
        add_box_tank(b, (0.0, 0.0, 0.0), (5 * a * s, depth * s, (2 * a + 10) * s), wall=cfg.h)
        add_fluid_block(b, cfg, (0.0, 0.0, 0.0), (a, depth, 2 * a))

    def on_model(self, model: newton.Model) -> None:
        self._front = wp.zeros(len(self.worlds), dtype=wp.float32, device=model.device)

    def record(self, step: int, state: Any, contacts: Any, obs: Any, log_: Any) -> None:
        col = log_.column(step)
        if col < 0:
            return
        r = self.config.particle_radius
        self._front.fill_(-1.0e30)
        self._launch(pb.front_kernel, self.solver.layout.n, [self.solver._particles(state), 4.0 * r, self._front])
        self._launch(pb.store_max_kernel, len(self.worlds), [self._front, col, r, log_.channel("front")])

    def scalars(self, log_: Any) -> dict[str, np.ndarray]:
        t, x = log_.times(), log_.series("front").astype(np.float64)
        a = self.param("a") * 2.0 * self.param("r")
        T = t[None, :] * np.sqrt(2.0 * G / a)[:, None]
        Z = x / a[:, None]
        within = T <= 3.0
        excess = np.where(within, Z - (1.0 + 2.0 * T), -math.inf).max(axis=1)
        keep = slice(None, None, max(1, len(t) // 60))
        self.monitors.update(front_T=T[:, keep].tolist(), front_Z=Z[:, keep].tolist(),
                             martin_moyce="NRMSE not run: curve not digitised (SPEC §8.9)")
        return {"ritter_excess": np.maximum(excess, 0.0)}


class RestingJitterTwin(SPHTwin):
    """SCR-SC-C2 resting jitter (A13 family, SPEC §5.2 LIQ resting jitter): water primed at rest in a box glass at 32
    and 100 mm depth (r = 2 mm, the route radius for a >= 64 mm glass; dt 1.25 ms, the route's D4 step), default
    settings; full mode adds the 100 mm column as a disordered packing (lattice +-0.15 spacing, fixed seed: a settled
    pour's packing, SC1's C2). After a 2 s settle the per-particle velocity RMS over the rest of the run must stay <= 3 mm/s and must
    not grow: RMS over the last window / over the first window after the settle - 1 <= 0.1, windows of 2 s (a quarter
    of the measured span when it is shorter, quick mode). Monitors: the free surface's height noise (std over the
    glass's lattice columns of the top particle's z, mean over the samples) and the density-solve iterations."""

    law_id = "A13.resting_jitter"
    units = {"rms_speed": "m/s", "growth": "1"}
    SETTLE_S = 2.0
    WINDOW_S = 2.0
    CHANNELS = ("rms", "zstd", "it_den", "it_div", "err_den", "err_div")

    def __init__(self, dt: float, **kw: Any) -> None:
        super().__init__(dt, **kw)
        self._depths: list[float] = []

    def conditions(self, quick: bool) -> tuple[Condition, ...]:
        n = 8 if quick else 16
        rows = ((8, "32mm", 0.0),) if quick else ((8, "32mm", 0.0), (25, "100mm", 0.0), (25, "100mm_disordered", 0.15))
        return tuple(Condition(f"glass_{n}x{n}_{tag}_r2mm", {"nx": n, "ny": n, "nz": nz, "r": 2.0e-3, "headroom": 8,
                                                              "perturb": jit, "seed": 1})
                     for nz, tag, jit in rows)  # 2r = 4 mm layers

    def build(self, b: newton.ModelBuilder, cond: Condition, cfg: DFSPHConfig) -> None:
        p, s = cond.params, cfg.spacing
        nx, ny, nz = int(p["nx"]), int(p["ny"]), int(p["nz"])
        add_box_tank(b, (0.0, 0.0, 0.0), (nx * s, ny * s, (nz + int(p["headroom"])) * s), wall=cfg.h)
        pts, jit = lattice_points((0.0, 0.0, 0.0), (nx, ny, nz), s), float(p.get("perturb", 0.0))
        if jit > 0.0:  # disordered packing, kept r inside the glass
            pts = pts + np.random.default_rng(int(p["seed"])).uniform(-jit, jit, pts.shape) * s
            pts[:, 0] = np.clip(pts[:, 0], 0.5 * s, (nx - 0.5) * s)
            pts[:, 1] = np.clip(pts[:, 1], 0.5 * s, (ny - 0.5) * s)
            pts[:, 2] = np.maximum(pts[:, 2], 0.5 * s)
        add_fluid_points(b, cfg, pts)
        self._depths.append(nz * s)
        self._cols = (nx, ny)

    def on_model(self, model: newton.Model) -> None:
        W, d = len(self.worlds), model.device
        if len({(int(c.params["nx"]), int(c.params["ny"])) for c in self.worlds}) != 1:
            raise ValueError(f"{self.law_id}: every world of one run needs the same glass footprint")
        self._terms = wp.zeros((2, max(self.solver.layout.n, 1)), dtype=wp.float64, device=d)
        self._sums = wp.zeros((2, W), dtype=wp.float64, device=d)
        self._top = wp.zeros((W, self._cols[0] * self._cols[1]), dtype=wp.float32, device=d)
        self._rms, self._zstd = (wp.zeros(W, dtype=wp.float64, device=d) for _ in range(2))
        self._solve = wp.zeros((4, W), dtype=wp.float64, device=d)

    def record(self, step: int, state: Any, contacts: Any, obs: Any, log_: Any) -> None:
        col = log_.column(step)
        if col < 0:
            return
        pb.rest_stats(self.solver, state, *self._cols, self._terms, self._sums, self._top, self._rms, self._zstd,
                      self._solve)
        arrs = (self._rms, self._zstd, *(self._solve[k] for k in range(4)))
        for name, arr in zip(self.CHANNELS, arrs, strict=True):
            self._launch(pb.store_column_kernel, len(self.worlds), [arr, col, log_.channel(name)])

    def scalars(self, log_: Any) -> dict[str, np.ndarray]:
        t = log_.times()
        rms, zstd, it, it_div, err_den, err_div = (log_.series(c).astype(np.float64) for c in self.CHANNELS)
        t0 = self.SETTLE_S
        win = min(self.WINDOW_S, (t[-1] - t0) / 4.0)
        meas, first, last = t >= t0, (t >= t0) & (t < t0 + win), t >= t[-1] - win
        mean_sq = lambda m: np.sqrt((rms[:, m] ** 2).mean(axis=1))  # noqa: E731
        r_all, r_first, r_last = mean_sq(meas), mean_sq(first), mean_sq(last)
        depth, cap = np.array(self._depths), self.config.max_iterations
        tol = self.solver.report().density_tol  # effective tolerance of the last step (resting-jitter rule)
        self.monitors.update(rms_mm_s=(r_all * 1e3).tolist(), first_mm_s=(r_first * 1e3).tolist(),
                             last_mm_s=(r_last * 1e3).tolist(), surface_noise_mm=(zstd[:, meas].mean(1) * 1e3).tolist(),
                             depth_m=list(self._depths), density_tol=tol.tolist(),
                             bound_tol_H_over_dt_mm_s=(tol * depth / self.dt * 1e3).tolist(),
                             # solves at the sampled steps after the settle: iterations vs the cap, residuals
                             density_iterations=it[:, meas].mean(1).tolist(), max_iterations=cap,
                             density_at_cap=(it[:, meas] >= cap).mean(1).tolist(),
                             divergence_iterations=it_div[:, meas].mean(1).tolist(),
                             divergence_at_cap=(it_div[:, meas] >= cap).mean(1).tolist(),
                             density_residual=err_den[:, meas].mean(1).tolist(),
                             divergence_residual=err_div[:, meas].mean(1).tolist())
        return {"rms_speed": r_all, "growth": r_last / r_first - 1.0}


__all__ = ["SPHTwin", "HydrostaticTwin", "ArchimedesTwin", "DamBreakTwin", "RestingJitterTwin"]
