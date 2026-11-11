# SPDX-License-Identifier: Apache-2.0
"""The tactile sensor's laws (ours: the papers define none; ruling 2026-11-06), on small pad scenes.

The pad is a free 6 x 6 x 1 cm plate on the ground. Its body origin is on its top face, so the default pad pose
(origin 0, identity) is the sensing surface. Objects rest on it under gravity, which some scenes tilt to load shear.
Noise is off in every law. The laws:
- **force_sum:** Σ taxel normal force equals the engine's captured contact force on the pad, within 1e-4.
- **force_weight:** the pad's normal force equals the object's m·g, within 0.5 %.
- **localisation:** a 2 mm sphere at an off-grid point; the force-weighted taxel centroid is within 1 pitch.
- **shear_direction:** a box loaded at azimuth 30 deg (0.3 g tangential, static); the shear sum points there within
  5 deg.
- **stick_ratio:** loaded at 0.8 μ, static; sensed shear/normal equals 0.8 μ within 1 %.
- **slip_ratio:** loaded at 1.2 μ, sliding; sensed shear/normal equals μ within 5 % (MuJoCo's soft friction cone).
- **null:** nothing on the pad; every taxel reads exactly 0.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.harness.scene import Law, LawResult, TestScene, body_xform

MU = 0.5
PAD_H = 0.005          # pad half thickness (m)
BOX = 0.006            # object half extent (m): its corners sit on taxel centres of the default 4 x 4 at 4 mm
BOX_MASS = 0.05        # kg
NOISE_OFF = {"noise": 0.0, "drift": 0.0, "tau_load": 0.0, "tau_unload": 0.0, "dead": 0.0, "crosstalk": 0.0,
             "saturation": 0.0, "lsb": 0.0}
SLIP_GRID = {**NOISE_OFF, "grid": (16, 4), "pitch": (0.004, 0.004)}
SPHERE_AT = (0.0013, -0.0047)


def _cfg(density: float) -> newton.ModelBuilder.ShapeConfig:
    return newton.ModelBuilder.ShapeConfig(density=density, mu=MU)


def _pad(b: newton.ModelBuilder) -> None:
    pad = b.add_body(xform=body_xform(0.0, 0.0, 2 * PAD_H), label="pad")
    b.add_shape_box(pad, xform=wp.transform((0.0, 0.0, -PAD_H), wp.quat_identity()), hx=0.03, hy=0.03, hz=PAD_H,
                    cfg=_cfg(2000.0))


def _box(b: newton.ModelBuilder, x: float = 0.0, z: float | None = None) -> None:
    box = b.add_body(xform=body_xform(x, 0.0, 2 * PAD_H + BOX + 5e-4 if z is None else z), label="box")
    b.add_shape_box(box, hx=BOX, hy=BOX, hz=BOX, cfg=_cfg(BOX_MASS / (2 * BOX) ** 3))


def _build_rest(b: newton.ModelBuilder) -> None:
    _pad(b)
    _box(b)


def _build_slide(b: newton.ModelBuilder) -> None:
    _pad(b)
    _box(b, x=-0.022)


def _build_sphere(b: newton.ModelBuilder) -> None:
    _pad(b)
    s = b.add_body(xform=body_xform(SPHERE_AT[0], SPHERE_AT[1], 2 * PAD_H + 0.002 + 5e-4), label="ball")
    b.add_shape_sphere(s, radius=0.002, cfg=_cfg(0.01 / (4.0 / 3.0 * math.pi * 0.002 ** 3)))


def _build_null(b: newton.ModelBuilder) -> None:
    _pad(b)
    _box(b, x=0.08, z=BOX + 5e-4)          # on the ground beside the pad


def _tilted(ratio: float, azimuth_deg: float) -> tuple[float, float, float]:
    a = math.radians(azimuth_deg)
    return (ratio * 9.81 * math.cos(a), ratio * 9.81 * math.sin(a), -9.81)


def rest_scene(gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)) -> TestScene:
    """A box resting on the pad (the plug-in's default test scene)."""
    return TestScene(build=_build_rest, periods=25, ground_mu=1.0, gravity=gravity)


def _series(run: Any, name: str) -> np.ndarray:
    return np.asarray(run.trace.plugin(run.plugin.label, name))


def _check_force_sum(run: Any) -> LawResult:
    tax, pad = _series(run, "taxels")[-1, 0, 0], _series(run, "pad_force")[-1, 0, 0]
    rel = (tax[:, 0].sum() - pad[0]) / max(abs(pad[0]), 1e-12)
    return LawResult.within(rel, 1e-4, f"taxels {tax[:, 0].sum():.6g} N vs engine {pad[0]:.6g} N",
                            "every captured contact must land on a taxel: check reach and the pad layout")


def _check_weight(run: Any) -> LawResult:
    m = float(run.model.body_mass.numpy()[run.bodies("box")[0]])
    pad = _series(run, "pad_force")[-1, 0, 0]
    rel = (pad[0] - m * 9.81) / (m * 9.81)
    return LawResult.within(rel, 5e-3, f"pad normal {pad[0]:.5g} N vs m·g {m * 9.81:.5g} N",
                            "the pad frame or capture depth misses the contact; or the box has not settled")


def _check_localisation(run: Any) -> LawResult:
    tax = _series(run, "taxels")[-1, 0, 0]
    w = np.clip(tax[:, 0], 0.0, None)
    xy = run.plugin.layout.xy
    got = (w[:, None] * xy).sum(0) / max(w.sum(), 1e-12)
    pitch = min(run.plugin.config.pitch)
    err = float(np.linalg.norm(got - np.asarray(SPHERE_AT))) / pitch
    return LawResult.within(err, 1.0, f"centroid {got * 1e3} mm vs {np.asarray(SPHERE_AT) * 1e3} mm "
                            f"({err:.2f} pitch)", "contact points land on the wrong taxels: check the pad frame")


def _shear(run: Any, periods: slice) -> np.ndarray:
    tax = _series(run, "taxels")[periods, 0, 0]          # [k, T, 3]
    return tax.sum(1)                                     # [k, 3]: normal, shear x, shear y


def _check_direction(run: Any) -> LawResult:
    s = _shear(run, slice(-1, None))[0]
    err = math.degrees(math.atan2(s[2], s[1])) - 30.0
    return LawResult.within(err, 5.0, f"shear direction {err + 30.0:.2f} deg vs 30 deg",
                            "shear channels must be the force on the pad in the pad's x, y")


def _check_ratio(want: float, bound: float, periods: slice):
    def check(run: Any) -> LawResult:
        s = _shear(run, periods)
        r = float(np.median(np.hypot(s[:, 1], s[:, 2]) / np.maximum(s[:, 0], 1e-12)))
        return LawResult.within((r - want) / want, bound, f"shear/normal {r:.4f} vs {want:.4f}",
                                "the shear channels or the normal channel are mis-scaled")
    return check


def _check_null(run: Any) -> LawResult:
    v = float(np.abs(_series(run, "taxels")).max())
    return LawResult.within(v, 0.0, f"max |taxel| {v:.3g} N with nothing on the pad",
                            "contacts outside the capture depth (the pad's bottom on the ground) must not count")


def tactile_laws() -> list[Law]:
    """The laws above, each on its own scene, with noise off."""
    shear = TestScene(build=_build_rest, periods=25, ground_mu=1.0, gravity=_tilted(0.3, 30.0))
    stick = TestScene(build=_build_slide, periods=10, ground_mu=1.0, gravity=_tilted(0.8 * MU, 0.0))
    slip = TestScene(build=_build_slide, periods=10, ground_mu=1.0, gravity=_tilted(1.2 * MU, 0.0))
    return [
        Law("force_sum", _check_force_sum, config=NOISE_OFF, doc="Σ taxels = engine contact force on the pad"),
        Law("force_weight", _check_weight, config=NOISE_OFF, doc="pad normal force = m·g at rest"),
        Law("localisation", _check_localisation, scene=TestScene(build=_build_sphere, periods=25, ground_mu=1.0),
            config=NOISE_OFF, doc="taxel centroid within one pitch of the contact"),
        Law("shear_direction", _check_direction, scene=shear, config=NOISE_OFF, doc="shear direction within 5 deg"),
        Law("stick_ratio", _check_ratio(0.8 * MU, 0.01, slice(5, None)), scene=stick, config=SLIP_GRID,
            doc="static: sensed shear/normal = applied"),
        Law("slip_ratio", _check_ratio(MU, 0.05, slice(3, None)), scene=slip, config=SLIP_GRID,
            doc="sliding: sensed shear/normal = μ"),
        Law("null", _check_null, scene=TestScene(build=_build_null, periods=10, ground_mu=1.0), config=NOISE_OFF,
            doc="no contact, no reading"),
    ]
