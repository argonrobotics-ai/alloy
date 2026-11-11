# SPDX-License-Identifier: Apache-2.0
"""A5 Coulomb cone (bench §4.5, SPEC §5.3): slip onset, creep, sliding acceleration, stop distance, isotropy.

A box (half extent HALF) rests on a plane with mu = MU on both shapes. Tilted worlds get gravity as an applied
body force m g s(t) (sin th cos ph, sin th sin ph, -cos th), eased in over T_EASE with a smoothstep (CON-51: a
stepped tilt slipped 4 deg early); the model gravity of those worlds is zero, so every solver sees the same load.
- A5.onset: tilts atan(mu) + offsets on a 0.1 deg grid; a world slips if it moves > SLIP_M along the slope during
  the hold. onset = midpoint between the last sticking and the first slipping tilt; pass |onset - atan mu| <= 0.1 deg.
  creep: travel per 10 s at atan(mu) - 1 deg over the second half of the hold; pass <= CREEP_PRECISION x 10 s.
- A5.slide: at atan(mu) + 5 deg in N_DIR = 16 azimuths (quick mode too), a = g (sin th - mu cos th) within 1 %
  (quadratic fit of the along-slope travel); flat worlds launched at V0 in N_DIR azimuths under normal gravity stop
  after v0^2 / (2 mu g) within 1 %. On MuJoCo only azimuth 0 is gated (SCR-WP3-7); the spread over the 16 azimuths
  is the known gap A5.slide_spread (defaults) / A5.slide_recipe (static-hold recipe).
- A5.isotropy: bench "slip-force spread <= 1 % over 16 directions" (quick mode too): on the flat plane a
  horizontal force at the COM in azimuth ph ramps from R_LO to R_HI x mu m g; the slip force is the ramp value when the box has moved SLIP_ISO_M
  along ph. Spread = (max - min) / mean over azimuths. The detection lag biases every direction alike (it cancels
  in the spread); a box that moves below HELD_RATIO (creep: it never held) or never moves gives no slip force
  (spread inf, fail).
MuJoCo (SPEC §5.3 A5, SCR-WP6-3 and SCR-WP3-7): onset, creep and isotropy run with the static-hold recipe
(elliptic, impratio 300, flat solimp 0.99). A5.slide (acceleration, stop distance at azimuth 0) is gated on the route
Alloy gives a non-precision rigid pair with no MJCF file (`default_route.routed_mujoco_settings`: elliptic, Newton's
solimp, impratio by the route's rule: 300 when no file sets one, SCR-WP3-10; 1 before WP2 routes it). An
off-axis multi-contact slider spins up about its vertical axis and feels less down-slope friction (WP3 FRIC-cause,
MuJoCo C too), so the spread over 16 azimuths is a known gap at every setting (`law.KNOWN_GAPS`, never gated):
A5.slide_spread at the defaults, A5.slide_recipe under the recipe.
Sleeping is off.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from alloy._vendor import newton
import numpy as np
import warp as wp

from alloy.bench.analytic.common import G, BaseTwin, mujoco_law_settings, set_body_velocities, smoothstep
from alloy.bench.solvers import RIGID_SOLVERS, SolverSettings
from alloy.bench.twin import DeviceTwinLog
from alloy.bench.types import Condition, LawSpec, Tolerance
from alloy.core.defaults import CREEP_PRECISION

MU = 0.5
HALF = 0.025  # [m] box half extent
DENSITY = 1000.0
T_EASE = 0.2  # [s]
SLIP_M = 2e-4  # [m] slip detection travel during the hold
V0 = 1.0  # [m/s] stop-distance launch speed
N_DIR = 16  # azimuths, quick mode too: 8 (multiples of 45 deg) miss MuJoCo's slide anisotropy (WP3 FRIC rows)
R_LO, R_HI = 0.9, 1.1  # A5.isotropy push ramp, x mu m g
HELD_RATIO = 0.92  # A5.isotropy: a box that slips below this never held (creep), so it has no slip force
SLIP_ISO_M = 2e-5  # [m] A5.isotropy slip detection travel
TH_C = math.atan(MU)
_SRC = "bench §4.5 A5 (+ SPEC §5.3 amendment)"

LAW_ONSET = LawSpec(
    id="A5.onset", suite="analytic",
    metrics={"onset_err_deg": Tolerance(abs=0.1, source=f"{_SRC}: |theta_s - atan mu| <= 0.1 deg"),
             "creep": Tolerance(abs=CREEP_PRECISION * 10.0, source=f"{_SRC}: creep <= CREEP_PRECISION x 10 s")},
    stat={"onset_err_deg": "mean", "creep": "mean"},
    dt=1.25e-3, substeps=1, duration_s=2.0, worlds_per_condition=1, solvers=RIGID_SOLVERS,
)
LAW_SLIDE = LawSpec(
    id="A5.slide", suite="analytic",
    metrics={"accel_rel_err": Tolerance(abs=0.01, source=f"{_SRC}: a = g(sin th - mu cos th) within 1 %"),
             "stop_rel_err": Tolerance(abs=0.01, source=f"{_SRC}: stop distance v0^2/(2 mu g) within 1 %")},
    stat={"accel_rel_err": "mean", "stop_rel_err": "mean"},
    dt=1.25e-3, substeps=1, duration_s=1.0, worlds_per_condition=1, solvers=RIGID_SOLVERS,
    quick={"worlds": 2 * N_DIR, "duration_s": 1.0},
)
def _spread_law(law_id: str, setting: str) -> LawSpec:
    src = f"SPEC §5.3 A5 (SCR-WP3-7): known gap, spread of the sliding acceleration over 16 azimuths, {setting}"
    return LawSpec(id=law_id, suite="analytic", metrics={"accel_spread": Tolerance(abs=0.01, source=src)},
                   stat={"accel_spread": "mean"}, dt=1.25e-3, substeps=1, duration_s=1.0, worlds_per_condition=1,
                   solvers=("mujoco",), quick={"worlds": N_DIR, "duration_s": 1.0})


LAW_SLIDE_SPREAD = _spread_law("A5.slide_spread", "MuJoCo defaults")
LAW_SLIDE_RECIPE = _spread_law("A5.slide_recipe", "static-hold recipe")
LAW_ISO = LawSpec(
    id="A5.isotropy", suite="analytic",
    metrics={"slip_force_spread": Tolerance(abs=0.01, source=f"{_SRC}: slip-force spread <= 1 % over 16 directions")},
    stat={"slip_force_spread": "mean"},
    dt=1.25e-3, substeps=1, duration_s=1.0, worlds_per_condition=1, solvers=RIGID_SOLVERS,
    quick={"worlds": N_DIR, "duration_s": 1.0},
)


@wp.kernel
def _apply_tilted_gravity(bodies: wp.array(dtype=wp.int32), mass: wp.array(dtype=float),
                          direction: wp.array(dtype=wp.vec3), scale: float, body_f: wp.array(dtype=wp.spatial_vector)):
    w = wp.tid()
    b = bodies[w]
    f = direction[w] * (mass[b] * scale)
    body_f[b] = wp.spatial_vector(f[0], f[1], f[2], 0.0, 0.0, 0.0)  # (force, torque) at the COM, Newton order


class _CoulombTwin(BaseTwin):
    units: Mapping[str, str] = {}
    settings = {"mujoco": mujoco_law_settings()}

    def prepare_builder(self, b: newton.ModelBuilder, solver: str) -> None:
        b.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=MU))

    def world_gravity(self, cond: Condition) -> tuple[float, float, float] | None:
        return (0.0, 0.0, 0.0) if cond.params["tilted"] else None

    def build(self, b: newton.ModelBuilder, cond: Condition, sample: Mapping[str, float]) -> None:
        body = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, HALF), wp.quat_identity()), label="box")
        b.add_shape_box(body, hx=HALF, hy=HALF, hz=HALF, cfg=newton.ModelBuilder.ShapeConfig(mu=MU, density=DENSITY))

    def on_model(self, model: newton.Model) -> None:
        th, ph = self.param("theta"), self.param("phi")
        tilted = self.param("tilted") > 0
        d = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), -np.cos(th)], 1) * tilted[:, None] * G
        self._dir = wp.array(d.astype(np.float32), dtype=wp.vec3, device=model.device)
        self._slope = np.stack([np.cos(ph), np.sin(ph)], 1)  # in-plane travel direction per world

    def drive(self, step: int, t: float, state: Any, control: Any) -> None:
        wp.launch(_apply_tilted_gravity, dim=len(self.worlds),
                  inputs=[self._first_body, self.model.body_mass, self._dir, smoothstep(t / T_EASE)],
                  outputs=[state.body_f], device=self.model.device)

    def record(self, step: int, state: Any, contacts: Any, obs: Any, log: DeviceTwinLog) -> None:
        self.record_first_body(step, state, log)

    def travel(self, log: DeviceTwinLog) -> tuple[np.ndarray, np.ndarray]:
        """(times, along-slope travel [W, T] relative to the start)."""
        xy = np.stack([log.series("x"), log.series("y")], -1).astype(np.float64)
        return log.times(), np.einsum("wtk,wk->wt", xy - xy[:, :1], self._slope)


class CoulombOnsetTwin(_CoulombTwin):
    law_id = LAW_ONSET.id
    units = {"onset_err_deg": "deg", "creep": "m"}
    offsets_deg = (-2.0, -1.0) + tuple(sorted(s * (0.05 + 0.1 * k) for k in range(7) for s in (-1.0, 1.0)))

    def conditions(self, quick: bool) -> tuple[Condition, ...]:
        return tuple(Condition(f"tilt{o:+.2f}deg", {"tilted": 1.0, "theta": TH_C + math.radians(o), "phi": 0.0,
                                                    "offset_deg": o}) for o in self.offsets_deg)

    def scalars(self, log: DeviceTwinLog) -> dict[str, np.ndarray]:
        t, s = self.travel(log)
        i0 = int(np.searchsorted(t, T_EASE))
        hold = s[:, -1] - s[:, i0]
        off = self.param("offset_deg")
        onset = onset_from_grid(off, hold > SLIP_M)
        i_half = int(np.searchsorted(t, 0.5 * (T_EASE + t[-1])))
        creep_w = int(np.argmin(np.abs(off + 1.0)))
        rate = abs(s[creep_w, -1] - s[creep_w, i_half]) / max(t[-1] - t[i_half], 1e-9)
        self.monitors.update({"onset_offset_deg": onset, "hold_travel_m": dict(zip(map(str, off), hold.tolist())),
                              "creep_m_per_10s_at_-1deg": rate * 10.0})
        err = abs(onset) if math.isfinite(onset) else math.inf
        return {"onset_err_deg": np.array([err]), "creep": np.array([rate * 10.0])}


def onset_from_grid(offsets: np.ndarray, slipped: np.ndarray) -> float:
    """Midpoint between the highest sticking offset below the first slip and that slip; NaN if none/all slip."""
    order = np.argsort(offsets)
    off, sl = offsets[order], slipped[order]
    if not sl.any() or sl.all():
        return math.nan
    first = int(np.argmax(sl))
    if first == 0:
        return math.nan
    return float(0.5 * (off[first - 1] + off[first]))


class CoulombSlideTwin(_CoulombTwin):
    law_id = LAW_SLIDE.id
    units = {"accel_rel_err": "1", "stop_rel_err": "1"}

    settings: Mapping[str, SolverSettings] = {}  # MuJoCo on the default route (BaseTwin.settings_for)

    def conditions(self, quick: bool) -> tuple[Condition, ...]:
        n_dir = N_DIR
        phis = [2 * math.pi * k / n_dir for k in range(n_dir)]
        tilt = [Condition(f"slide_ph{math.degrees(p):.1f}", {"tilted": 1.0, "theta": TH_C + math.radians(5.0),
                                                              "phi": p}) for p in phis]
        stop = [Condition(f"stop_ph{math.degrees(p):.1f}", {"tilted": 0.0, "theta": 0.0, "phi": p}) for p in phis]
        return tuple(tilt + stop)

    def init_state(self, model: newton.Model, state: Any) -> None:
        qd = np.zeros((model.body_count, 6))
        flat = self.param("tilted") == 0
        ph = self.param("phi")
        qd[flat, 0], qd[flat, 1] = V0 * np.cos(ph[flat]), V0 * np.sin(ph[flat])
        set_body_velocities(model, state, qd)

    def scalars(self, log: DeviceTwinLog) -> dict[str, np.ndarray]:
        t, s = self.travel(log)
        tilted = self.param("tilted") > 0
        th = self.param("theta")
        win = t >= T_EASE + 0.05
        a_sim = np.array([np.polyfit(t[win], s[w, win], 2)[0] * 2.0 for w in range(len(self.worlds))])
        a_law = G * (np.sin(th) - MU * np.cos(th))
        acc = np.abs(a_sim - a_law)[tilted] / a_law[tilted]
        a_t = a_sim[tilted]
        iso = (a_t.max() - a_t.min()) / abs(a_t.mean()) if a_t.size else math.inf
        d_law = V0 * V0 / (2 * MU * G)
        stop = np.abs(s[~tilted, -1] - d_law) / d_law
        gated = self._gated_azimuths()
        self.monitors.update({"accel_sim": a_t.tolist(), "accel_law": float(a_law[tilted][0]) if a_t.size else None,
                              "accel_spread": iso, "stop_sim_m": s[~tilted, -1].tolist(), "stop_law_m": d_law,
                              "gated_azimuths": "0 only (SCR-WP3-7)" if self.solver_name == "mujoco" else "all"})
        return {"accel_rel_err": acc[gated[tilted]], "stop_rel_err": stop[gated[~tilted]]}

    def _gated_azimuths(self) -> np.ndarray:
        """Per world: gated? MuJoCo gates azimuth 0 only (SCR-WP3-7); other solvers every azimuth."""
        if self.solver_name != "mujoco":
            return np.ones(len(self.worlds), bool)
        return np.isclose(self.param("phi"), 0.0)


class CoulombSlideSpreadTwin(CoulombSlideTwin):
    """Known gap: the spread of the sliding acceleration over 16 azimuths at MuJoCo's defaults (tilted worlds only)."""

    law_id = LAW_SLIDE_SPREAD.id
    units = {"accel_spread": "1"}

    def conditions(self, quick: bool) -> tuple[Condition, ...]:
        return tuple(c for c in super().conditions(quick) if c.params["tilted"])

    def _gated_azimuths(self) -> np.ndarray:
        return np.ones(len(self.worlds), bool)

    def scalars(self, log: DeviceTwinLog) -> dict[str, np.ndarray]:
        super().scalars(log)
        return {"accel_spread": np.array([self.monitors["accel_spread"]])}


class CoulombSlideRecipeTwin(CoulombSlideSpreadTwin):
    """Known gap: the same spread under the static-hold recipe."""

    law_id = LAW_SLIDE_RECIPE.id

    @property
    def settings(self) -> Mapping[str, SolverSettings]:  # type: ignore[override]
        return {"mujoco": mujoco_law_settings()}


def push_ratio(t: np.ndarray | float, t_end: float) -> np.ndarray:
    """A5.isotropy push in units of mu m g: eased to R_LO over T_EASE, then a linear ramp to R_HI at t_end."""
    t = np.asarray(t, dtype=np.float64)
    ease = R_LO * np.vectorize(smoothstep)(np.minimum(t, T_EASE) / T_EASE)
    ramp = R_LO + (R_HI - R_LO) * np.clip((t - T_EASE) / (t_end - T_EASE), 0.0, 1.0)
    return np.where(t < T_EASE, ease, ramp)


def slip_ratios(t: np.ndarray, s: np.ndarray, t_end: float) -> np.ndarray:
    """Per world, the push ratio when the box has moved SLIP_ISO_M along the push since the ramp start (NaN when it
    slipped below HELD_RATIO, i.e. it never held, or never slipped)."""
    i0 = int(np.searchsorted(t, T_EASE))
    moved = (s - s[:, i0:i0 + 1]) > SLIP_ISO_M
    moved[:, :i0 + 1] = False
    out = np.full(s.shape[0], np.nan)
    for w in range(s.shape[0]):
        hit = np.flatnonzero(moved[w])
        ratio = float(push_ratio(t[hit[0]], t_end)) if hit.size else math.nan
        if ratio >= HELD_RATIO:
            out[w] = ratio
    return out


class CoulombIsotropyTwin(_CoulombTwin):
    law_id = LAW_ISO.id
    units = {"slip_force_spread": "1"}

    def conditions(self, quick: bool) -> tuple[Condition, ...]:
        n_dir = N_DIR
        return tuple(Condition(f"push_ph{math.degrees(p):.1f}", {"tilted": 0.0, "theta": 0.0, "phi": p})
                     for p in (2 * math.pi * k / n_dir for k in range(n_dir)))

    def on_model(self, model: newton.Model) -> None:
        super().on_model(model)
        ph = self.param("phi")
        push = np.stack([np.cos(ph), np.sin(ph), np.zeros_like(ph)], 1) * MU * G
        self._push = wp.array(push.astype(np.float32), dtype=wp.vec3, device=model.device)

    def drive(self, step: int, t: float, state: Any, control: Any) -> None:
        wp.launch(_apply_tilted_gravity, dim=len(self.worlds),
                  inputs=[self._first_body, self.model.body_mass, self._push,
                          float(push_ratio(t, LAW_ISO.duration_s))],
                  outputs=[state.body_f], device=self.model.device)

    def scalars(self, log: DeviceTwinLog) -> dict[str, np.ndarray]:
        t, s = self.travel(log)
        r = slip_ratios(t, s, LAW_ISO.duration_s)
        spread = (r.max() - r.min()) / r.mean() if np.all(np.isfinite(r)) and r.size else math.inf
        self.monitors.update({"slip_ratio": dict(zip((c.id for c in self.worlds), r.tolist()))})
        return {"slip_force_spread": np.array([spread])}


__all__ = ["LAW_ISO", "LAW_ONSET", "LAW_SLIDE", "LAW_SLIDE_RECIPE", "LAW_SLIDE_SPREAD", "CoulombIsotropyTwin",
           "CoulombOnsetTwin", "CoulombSlideRecipeTwin", "CoulombSlideSpreadTwin", "CoulombSlideTwin",
           "onset_from_grid", "push_ratio", "slip_ratios"]
