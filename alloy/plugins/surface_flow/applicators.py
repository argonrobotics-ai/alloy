# SPDX-License-Identifier: Apache-2.0
"""Applicators (design note §4.1; phase 2 note): tools that work the film, host side.

``Applicator`` names a tool body and its underside: a flat ``disc`` or ``rect`` (a float, trowel, squeegee, pad) or
a ``roller``. Two couplings, each stable on its own:
- **The tool on the surface.** The tool rides on the substrate (``base``: a plastic bulk's surface where there is
  one) like removal's compliant pad, with Coulomb friction there, or the film's shear ``μU/h + τ_y`` where it glides
  on film above the substrate. Its stiffness obeys removal's explicit bound on the tool's (and a floating
  workpiece's) mobility, which ``check_tools`` enforces.
- **The film under the tool.** The underside is a ceiling. Each film substep moves film above it one cell on: ahead
  of a sliding tool, or out from under a pressing one (volume exact). Under a sliding tool the film is sheared
  toward the Couette mean ``U/2``, so a blade leaves half its gap behind.

The film does not hold a tool up: a film squeezed under a resting tool leaves at the projection's pace, not by
Reynolds' equation (no Stefan law in phase 2; a penalty stiff enough for it breaks the explicit film step).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy.plugins.errors import PluginError
from alloy.plugins.removal.stability import PadResponse, pad_response
from alloy.plugins.removal.surface import Patch, patch_of
from alloy.plugins.surface_flow.charts import PlaneChart, RevolutionChart
from alloy.plugins.surface_flow.stores import store_spec
from alloy.plugins.surface_flow.tool_geometry import ToolSet

SHAPES = {"disc": 0, "rect": 1, "roller": 2, "edger": 3}
STABLE = 1.0                      # removal's bound on ω·dt and the damping product
SLIP_EPS = 1.0e-3                 # m/s


@dataclass(frozen=True)
class Applicator:
    """A tool on ``chart``. ``size``: disc (radius, unused); rect (half extents along ``axis`` and normal x axis);
    roller (radius, half width along ``axis``) [m]. ``center``, ``normal``, ``axis`` are in the tool body's frame:
    the underside's centre and outward normal (disc, rect), or the roller's axis centre and axis. An ``"edger"`` is a
    rect with GA's edger sole (paste.py:557-566): flat, then a quarter round of ``edge_radius`` down to a lip
    ``edge_lip_w`` wide that runs ``edge_lip`` deep down the form face, on the ``edge_side`` (+1: +normal x axis)
    long side."""

    body: str
    shape: str = "disc"
    size: tuple[float, float] = (0.03, 0.03)
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    normal: tuple[float, float, float] = (0.0, 0.0, -1.0)
    axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    chart: int = 0
    stiffness: float = 1.0e6       # pad stiffness on the film's top [Pa/m]
    damping: float = 0.0           # Kelvin-Voigt time [s]
    friction: float = 0.3          # Coulomb coefficient on a dry surface
    store: Any = None              # stores.Store (or a dict): the paint it carries; it then lays by the nip split
    implicit: bool = False         # the pad acts implicitly on the tool (pad_implicit.py): any stiffness is stable
    porous: bool = False           # a sponge: with room (tool_room) it absorbs the film it would plough
    edge_radius: float = 0.0       # edger: the arris radius [m]
    edge_lip: float = 0.0          # edger: how deep the lip runs down the form face [m]
    edge_lip_w: float = 0.0015     # edger: the lip's width [m]
    edge_side: int = 1             # edger: the lip's side, +1 along +normal x axis, -1 the other
    tip_slope: float = 0.0         # disc: a pointed round tip, its underside rising tip_slope·rho from the centre
    roll_lift: float | None = None  # m: caps the roll ahead at ½(rise + lift)²·L, the rest spills at the blade's ends
                                    # (roll_cap.py; GA ROLL_LIFT: float 3 mm, trowel 1 mm). None: the roll is uncapped

    def __post_init__(self) -> None:
        object.__setattr__(self, "store", store_spec(self.store))
        if self.shape not in SHAPES:
            raise PluginError(f"applicator {self.body!r}: shape must be one of {sorted(SHAPES)}, got {self.shape!r}")
        if len(self.size) != 2 or not all(math.isfinite(v) and v > 0.0 for v in self.size):
            raise PluginError(f"applicator {self.body!r}: size must be two lengths > 0, got {self.size}")
        for name in ("normal", "axis"):
            v = np.asarray(getattr(self, name), np.float64)
            if v.shape != (3,) or not np.isfinite(v).all() or np.linalg.norm(v) < 1e-9:
                raise PluginError(f"applicator {self.body!r}: {name} must be a nonzero 3-vector, got {v}")
        if self.shape == "edger":
            self._check_edger()
        if not (math.isfinite(self.tip_slope) and self.tip_slope >= 0.0) or (self.tip_slope > 0.0 and
                                                                             self.shape != "disc"):
            raise PluginError(f"applicator {self.body!r}: tip_slope must be finite and >= 0, and only on a disc, got "
                              f"{self.tip_slope} on a {self.shape}")
        if self.store is not None and self.store.meter > 0.0 and self.shape == "roller":
            raise PluginError(f"applicator {self.body!r}: a metering store is a brush (disc or rect), not a roller")
        for name in ("stiffness", "damping", "friction"):
            v = getattr(self, name)
            if not math.isfinite(v) or v < 0.0 or (name == "stiffness" and v <= 0.0):
                raise PluginError(f"applicator {self.body!r}: {name} must be finite and >= 0 (stiffness > 0), "
                                  f"got {v}")
        if self.roll_lift is not None and (not math.isfinite(self.roll_lift) or self.roll_lift < 0.0
                                           or self.store is not None):
            raise PluginError(f"applicator {self.body!r}: roll_lift must be a finite length >= 0 [m] on a ceiling tool "
                              f"(no store), got {self.roll_lift}")

    def _check_edger(self) -> None:
        r, lip, w = self.edge_radius, self.edge_lip, self.edge_lip_w
        if not (math.isfinite(r) and 0.0 < r < 2.0 * self.size[1]):
            raise PluginError(f"applicator {self.body!r}: an edger's edge_radius must be in (0, 2·size[1]), got {r}")
        if not (math.isfinite(lip) and lip >= 0.0 and math.isfinite(w) and 0.0 <= w < r):
            raise PluginError(f"applicator {self.body!r}: an edger needs edge_lip >= 0 and edge_lip_w in "
                              f"[0, edge_radius), got {lip}, {w}")
        if self.edge_side not in (1, -1):
            raise PluginError(f"applicator {self.body!r}: edge_side must be +1 or -1, got {self.edge_side}")

    @property
    def reach(self) -> float:
        return self.size[0] if self.shape == "disc" else math.hypot(*self.size)

    @property
    def footprint(self) -> float:
        """The largest pressed area [m²]: the face (disc, rect) or the roller's projection."""
        r, s = self.size
        return math.pi * r * r if self.shape == "disc" else 4.0 * r * s


def applicator_spec(spec: Applicator | dict) -> Applicator:
    if isinstance(spec, Applicator):
        return spec
    if isinstance(spec, dict):
        d = {k: tuple(v) if isinstance(v, list) else v for k, v in spec.items()}
        return Applicator(**d)
    raise PluginError(f"tools: expected an Applicator or a dict, got {spec!r}")


@dataclass
class ToolTables:
    set: ToolSet
    count: int
    specs: tuple[Applicator, ...]
    body: wp.array                 # [W, n] int32: each world's tool body
    mobility: wp.array             # [W, n] float32: summed point mobility at the tool (removal's rule)
    response: list[PadResponse]    # per tool, the worst world (check_tools)
    wins: wp.array                 # [W, n] vec4i: the cells each tool wrote last


def _patches(ctx: Any, chart: PlaneChart) -> list[Patch]:
    return [patch_of(ctx.model, int(s), chart.shape, chart.size, chart.surface_offset)
            for s in ctx.shapes(chart.shape).numpy()]


def _revolution_patches(chart: RevolutionChart, k: int, tables: Any) -> list[Patch]:
    """A revolution chart's body as one patch per world: the chart frame on its body, sized to bound the profile."""
    prof = np.asarray(chart.profile, np.float64)
    r = float(np.abs(prof[:, 0]).max())
    size = (2.0 * r, 2.0 * max(r, float(np.ptp(prof[:, 1]))))
    body, local = tables.cbody.numpy()[:, k], tables.clocal.numpy()[:, k]
    return [Patch(shape=-1, body=int(b), local=tuple(float(v) for v in x), size=size) for b, x in zip(body, local)]


def _check_curved(s: Applicator, forming: bool) -> None:
    """Revolution charts take implicit ceiling pads (disc, rect) only (WP8's sponge on a pan)."""
    if s.shape not in ("disc", "rect") or s.store is not None or not s.implicit or forming:
        raise PluginError(f"applicator {s.body!r}: a revolution chart takes an implicit ceiling pad "
                          f"(shape 'disc' or 'rect', implicit=True, no store, no Config.bulk); rollers, stores, "
                          f"explicit pads and forming work on plane charts only")


def build_tools(ctx: Any, specs: Sequence[Applicator], charts: Sequence[Any], tables: Any = None,
                forming: bool = False) -> ToolTables:
    """Tool tables for ``ctx``'s worlds (at least one tool); ``charts`` are the plug-in's chart specs and ``tables``
    its chart tables (a revolution chart's body). A revolution chart takes implicit ceiling pads only."""
    W, n = ctx.world_count, len(specs)
    bodies, mobility, response = np.zeros((W, n), np.int32), np.zeros((W, n), np.float32), []
    for k, s in enumerate(specs):
        if not 0 <= s.chart < len(charts):
            raise PluginError(f"applicator {s.body!r}: chart {s.chart} is not one of the {len(charts)} charts")
        chart = charts[s.chart]
        if isinstance(chart, RevolutionChart):
            _check_curved(s, forming)
            if s.roll_lift is not None:
                raise PluginError(f"applicator {s.body!r}: roll_lift caps a roll on a plane chart; a revolution "
                                  f"chart's implicit pads do not plough one")
            if tables is None:
                raise PluginError(f"applicator {s.body!r}: a revolution chart's tools need the chart tables")
            patches = _revolution_patches(chart, s.chart, tables)
        else:
            patches = _patches(ctx, chart)
        b = ctx.bodies(s.body).numpy().astype(np.int32)
        bodies[:, k] = b
        normal = (0.0, 0.0, -1.0) if s.shape == "roller" else s.normal
        r = pad_response(ctx.model, b, patches, s.center, normal, s.reach)
        mobility[:, k] = r.mobility
        response.append(r)
    t = ToolSet()

    def arr(values: list, dtype: Any) -> wp.array:
        scalar = np.int32 if dtype == wp.int32 else np.float32
        return ctx.array(np.asarray(values, scalar), dtype)

    t.chart = arr([s.chart for s in specs], wp.int32)
    t.shape = arr([SHAPES[s.shape] for s in specs], wp.int32)
    t.center, t.normal, t.axis = (arr([getattr(s, k) for s in specs], wp.vec3) for k in ("center", "normal", "axis"))
    t.size = arr([s.size for s in specs], wp.vec2)
    t.edge = arr([(s.edge_radius, s.edge_lip, s.edge_lip_w, float(s.edge_side)) for s in specs], wp.vec4)
    t.slope = arr([s.tip_slope for s in specs], wp.float32)
    t.stiffness, t.damping, t.friction = (arr([getattr(s, k) for s in specs], wp.float32)
                                          for k in ("stiffness", "damping", "friction"))
    t.slip_eps = SLIP_EPS
    t.ceiling = arr([0 if s.store is not None else 1 for s in specs], wp.int32)
    t.implicit = arr([1 if s.implicit else 0 for s in specs], wp.int32)
    t.porous = arr([1 if s.porous and s.store is None else 0 for s in specs], wp.int32)
    t.roll_lift = arr([-1.0 if s.roll_lift is None else s.roll_lift for s in specs], wp.float32)
    wins = np.tile(np.asarray([0, -1, 0, -1], np.int32), (W, n, 1))
    return ToolTables(set=t, count=n, specs=tuple(specs), body=ctx.array(bodies, wp.int32),
                      mobility=ctx.array(mobility, wp.float32), response=response, wins=ctx.array(wins, wp.vec4i))


def check_tools(label: str, tools: ToolTables, dt: float) -> None:
    """Removal's explicit bound for each tool's pad, at its whole footprint pressed in."""
    for s, r in zip(tools.specs, tools.response):
        if s.implicit:                                                     # backward Euler: stable at any stiffness
            continue
        stiffness = s.stiffness * s.footprint                              # [N/m]
        tilt = stiffness * 0.25 * s.reach ** 2
        omega_sq = stiffness * r.normal_mobility + 2.0 * tilt / r.tilt_inertia
        checks = {"spring": math.sqrt(omega_sq) * dt, "damping": s.damping * omega_sq * dt}
        bad = {k: round(v, 3) for k, v in checks.items() if v >= STABLE}
        if bad:
            raise PluginError(f"{label}: applicator {s.body!r} is unstable at dt={dt:.3g} s: {bad} (each must be < "
                              f"{STABLE}); lower its stiffness or damping, shrink the exchange step, or give the tool "
                              f"more mass and inertia")


__all__ = ["Applicator", "SHAPES", "ToolTables", "applicator_spec", "build_tools", "check_tools"]
