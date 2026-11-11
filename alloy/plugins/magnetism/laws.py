# SPDX-License-Identifier: Apache-2.0
"""``Magnetism`` laws against closed forms (``field.py``), on small scenes. Every law computes its expectation from the
textbook formula, not from the plug-in's kernels:
- **inverse_fourth:** two magnets probed coaxial and side by side at d and 2d: 3μ0m²/(2πd⁴) and 3μ0m²/(4πd⁴),
  within 1e-5.
- **energy:** two free magnets close from rest (zero g); kinetic + dipole energy stays constant within 0.5 % of
  the energy released.
- **field_oscillation:** a bar magnet in a uniform field, 0.1 rad off, oscillates at ω0 = sqrt(mB/I) within 1 %
  (torque alignment).
- **stirrer_lag:** a bar on a vertical pivot with viscous damping c, 10 mm above a drive magnet. At c·ω/(mB) = 0.5
  and 0.8 it follows at lag asin(c·ω/(mB)) within 2 % (30° and 53°). At 1.3 it slips (mean speed < 0.7 ω).
  - The bar is damped near critically (ζ ≈ 0.93). A lightly damped bar has a second, slipping state above
    c·ω/(mB) ≈ (4/π)·2ζ, the damped pendulum's retrapping limit, so started from rest it may never lock in
    (ζ = 0.17 slipped at 0.8).
- **plate_force:** a magnet above an iron plate probed at h and 2h: normal 3μ0κm²/(32πh⁴), lying flat half that,
  within 1e-5. Behind the plate: exactly 0. μr = 3 gives κ = 1/2.
- **eraser_holds / eraser_falls:** an eraser with a magnet 6 mm behind its face, on a vertical whiteboard under
  gravity (μ 0.5).
  - With μF > mg it stays (< 1 mm in 0.3 s).
  - With μF ≈ mg/2 it falls as a = g - μF(h)/m predicts, within 5 %, integrated over the recorded gap h(t). This
    runs at a 0.156 ms step: 2.5e-4 or less.
  - At the default 1.25 ms step, MuJoCo's lightly pressed sliding contact tips the eraser instead: 61° of tilt,
    the gap opens to 30 mm, and the pull's 1/h⁴ loss amplifies it.
    - WP13 reproduced it without the plug-in, with a constant press (14° and 11 mm), and in MuJoCo C too.
    - At the small step the gap holds at 6.0-6.5 mm with 1.2° of tilt, and the fall matches to 1 %.
  - The contact is stiff (ke 1e6, kd 2·sqrt(ke)): the plate pull gets stiffer as the gap closes (4F/h). Newton's
    default ke of 2.5e3 let the eraser sink 5 mm into the board, and the 1/h⁴ force ran away (2 → 43 N).
    - At a 1.25 ms step MuJoCo clamps the time constant to 2·dt (2.5 ms), so ke 1e6 runs as about 2e5 (WP13).
      ke ≥ 1e5 is what matters there.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.harness.scene import Law, LawResult, TestScene, body_xform
from alloy.plugins.magnetism.field import MU0

FREE = {"ground": False, "gravity": (0.0, 0.0, 0.0)}
BAR = (0.0125, 0.004, 0.004)                         # stir bar half extents (25 x 8 x 8 mm)
H_DRIVE, M_BAR, M_DRIVE = 0.01, 0.0675, 0.375        # drive 10 mm below the bar (ESTIMATE, N35-scale magnets)
BAR_DENSITY = 2000.0                                 # light bar: near-critical damping at the lag law's c
B_DRIVE = 1e-7 * M_DRIVE / H_DRIVE ** 3              # field at the bar, antiparallel to the drive moment
RPM_LAG = (300.0, 480.0, 780.0)                      # c·ω/(mB) = 0.5, 0.8, 1.3
C_LAG = 0.5 * M_BAR * B_DRIVE / (RPM_LAG[0] * 2 * math.pi / 60)
STIRRER = {"bodies": ("stir_bar",), "moments": (M_BAR,), "axes": ((1.0, 0.0, 0.0),), "offsets": (),
           "drives": ((0.0, 0.0, -H_DRIVE),), "drive_moments": (M_DRIVE,), "drive_rpm": (RPM_LAG[0],),
           "plates": (), "external_field": (0.0, 0.0, 0.0)}
PAIR = {"bodies": ("magnet_a", "magnet_b"), "moments": (1.0, 1.0), "axes": ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        "drives": (), "drive_moments": (), "drive_rpm": (), "plates": ()}
ERASER = (0.01, 0.06, 0.025)                         # half extents: thickness (x), width, height
ERASER_MU, ERASER_GAP = 0.5, 0.006                   # dipole 6 mm behind the eraser's face
STIFF = {"ke": 1.0e6, "kd": 2.0e3}                   # contact: kd = 2·sqrt(ke)


def _yaw(q: np.ndarray) -> np.ndarray:
    return 2.0 * np.arctan2(q[..., 2], q[..., 3])


def _box(b: newton.ModelBuilder, label: str, half: tuple, xform: Any, density: float = 7500.0,
         mu: float = 0.5, link: bool = False, **contact: float) -> int:
    cfg = newton.ModelBuilder.ShapeConfig(density=density, mu=mu, **contact)
    body = b.add_link(xform=xform, label=label) if link else b.add_body(xform=xform, label=label)
    b.add_shape_box(body, hx=half[0], hy=half[1], hz=half[2], cfg=cfg)
    return body


def _pivot_bar(b: newton.ModelBuilder) -> None:
    """The stir bar on a vertical revolute pivot at the origin, with viscous damping C_LAG. It starts at yaw π,
    along the drive's field at phase 0 (the field above a dipole is antiparallel to it), not in the unstable
    antiparallel rest."""
    xf = body_xform(0.0, 0.0, 0.0, (0.0, 0.0, 1.0, 0.0))
    bar = _box(b, "stir_bar", BAR, xf, density=BAR_DENSITY, link=True)
    j = b.add_joint_revolute(-1, bar, parent_xform=xf, axis=(0.0, 0.0, 1.0), damping=C_LAG, label="stir_pivot")
    b.add_articulation([j], label="stir_bar")


def default_scene() -> TestScene:
    """A magnetic stirrer: the bar on its pivot, the drive turning at 300 rpm 20 mm below."""
    return TestScene(build=_pivot_bar, periods=25, **FREE)


def _pair(b: newton.ModelBuilder, d: float = 0.05) -> None:
    for label, x in (("magnet_a", -0.5 * d), ("magnet_b", 0.5 * d)):
        _box(b, label, (0.005, 0.005, 0.005), body_xform(x, 0.0, 0.0))


def _rpm_per_world(sim: Any) -> None:
    sim.plugins["magnetism"].set_input("rpm", np.array(RPM_LAG, np.float32)[:, None])


def check_inverse_fourth(run: Any) -> LawResult:
    a, b = run.bodies("magnet_a")[0], run.bodies("magnet_b")[0]
    q0 = run.trace.body_q[0].copy()
    errs = []
    for d in (0.03, 0.06):
        for side, rot, want in (("coaxial", (0, 0, 0, 1), -3 * MU0 / (2 * math.pi * d ** 4)),
                                ("side", (0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4)),
                                 3 * MU0 / (4 * math.pi * d ** 4))):
            q = q0.copy()
            q[a, :3], q[b, :3] = (-0.5 * d, 0, 0), (0.5 * d, 0, 0)
            q[a, 3:], q[b, 3:] = rot, rot
            fx = run.probe(q)[b, 0]                    # + = push b away from a
            errs.append(fx / want - 1.0)
    worst = max(errs, key=abs)
    return LawResult.within(worst, 1e-5, "coaxial and side-by-side at 30 and 60 mm vs 3μ0m²/(2πd⁴), 3μ0m²/(4πd⁴): "
                            + ", ".join(f"{e:+.1e}" for e in errs), "kernels.dip_force")


def check_energy(run: Any) -> LawResult:
    mag, model, tr = run.plugin, run.model, run.trace
    mass = model.body_mass.numpy()
    ids = [run.bodies(lab)[0] for lab in ("magnet_a", "magnet_b")]
    tot = []
    for k in range(len(tr.t)):
        kin = sum(0.5 * mass[b] * np.dot(tr.body_qd[k, b, :3], tr.body_qd[k, b, :3]) for b in ids)
        tot.append(kin + mag.energy(tr.body_q[k])[0])
    released = mag.energy(tr.body_q[0])[0] - mag.energy(tr.body_q[-1])[0]
    drift = (max(tot) - min(tot)) / max(abs(released), 1e-30)
    return LawResult.within(drift, 5e-3, f"energy drift {drift:.2e} of the {released:.3g} J released",
                            "forces must derive from U = 1e-7/d³ [m_i·m_j - 3(m_i·r̂)(m_j·r̂)]")


def check_field_oscillation(run: Any) -> LawResult:
    bar = run.bodies("stir_bar")[0]
    th = _yaw(run.trace.body_q[:, bar, 3:])
    t = np.asarray(run.trace.t)
    s = np.flatnonzero(np.sign(th[:-1]) != np.sign(th[1:]))
    tz = t[s] - th[s] * (t[s + 1] - t[s]) / (th[s + 1] - th[s])
    f = 0.5 / float(np.mean(np.diff(tz)))
    i_zz = float(run.model.body_inertia.numpy()[bar][2, 2])
    c = run.plugin.config
    want = math.sqrt(c.moments[0] * c.external_field[0] / i_zz) / (2 * math.pi)
    err = f / want - 1.0
    return LawResult.within(err, 0.01, f"{f:.4f} Hz vs sqrt(mB/I)/2π = {want:.4f} Hz ({len(tz)} zero crossings)",
                            "torque m × B on the body")


def check_stirrer_lag(run: Any) -> LawResult:
    bar = run.bodies("stir_bar")
    t = np.asarray(run.trace.t)
    phase = run.trace.plugin("magnetism", "phase")[:, :, 0]
    late = t > t[-1] - 0.3
    errs, detail = [], []
    for w, rpm in enumerate(RPM_LAG):
        omega = rpm * 2 * math.pi / 60
        th = np.unwrap(_yaw(run.trace.body_q[:, bar[w], 3:]))
        if w < 2:
            lag = np.angle(np.exp(1j * (phase[late, w] + math.pi - th[late])))       # bar follows -drive
            want = math.asin(C_LAG * omega / (M_BAR * B_DRIVE))
            errs.append(float(np.mean(lag)) / want - 1.0)
            detail.append(f"{math.degrees(np.mean(lag)):.2f} vs {math.degrees(want):.2f} deg")
        else:
            speed = (th[-1] - th[np.argmax(late)]) / (t[-1] - t[np.argmax(late)])
            slip = speed / omega
            errs.append(0.0 if slip < 0.7 else 1.0)
            detail.append(f"pull-out: mean speed {slip:.2f} ω")
    worst = max(errs, key=abs)
    return LawResult.within(worst, 0.02, "; ".join(detail), "m B sin δ = c ω (drive field -1e-7 M/h³ at the bar)")


def _plate_scene() -> TestScene:
    return TestScene(build=lambda b: _box(b, "magnet_a", (0.003, 0.003, 0.003), body_xform(0, 0, 0.01)),
                     periods=1, **FREE)


def check_plate_force(run: Any) -> LawResult:
    a = run.bodies("magnet_a")[0]
    kap = (3.0 - 1.0) / (3.0 + 1.0)
    q0 = run.trace.body_q[0].copy()
    errs = []
    flat = (0.0, 0.0, 0.0, 1.0)                                 # moment along body x: lying flat
    up = (0.0, -math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4))   # body x turned to +z: normal
    for h in (0.01, 0.02):
        for rot, factor in ((up, 1.0), (flat, 0.5)):
            q = q0.copy()
            q[a, :3], q[a, 3:] = (0.0, 0.0, h), rot
            want = -factor * 3 * MU0 * kap * 0.05 ** 2 / (32 * math.pi * h ** 4)
            errs.append(run.probe(q)[a, 2] / want - 1.0)
    q = q0.copy()
    q[a, :3] = (0.0, 0.0, -0.01)                                 # behind the plate
    behind = float(np.abs(run.probe(q)[a]).max())
    worst = max(errs, key=abs)
    value = worst if behind == 0.0 else 1.0
    return LawResult.within(value, 1e-5, f"normal and flat at 10, 20 mm vs 3μ0κm²/(32πh⁴) (κ = 1/2): "
                            + ", ".join(f"{e:+.1e}" for e in errs) + f"; behind the plate {behind:.1e} N",
                            "kernels.plate_wrench: image m' = κ(2(m·n)n - m) at the mirror point")


def _eraser_scene() -> TestScene:
    def build(b: newton.ModelBuilder) -> None:
        cfg = newton.ModelBuilder.ShapeConfig(mu=ERASER_MU, **STIFF)
        b.add_shape_box(-1, xform=wp.transform((-0.01, 0.0, 0.0), wp.quat_identity()), hx=0.01, hy=0.3, hz=1.0,
                        cfg=cfg, label="board")
        _box(b, "eraser", ERASER, body_xform(ERASER[0], 0.0, 0.5), density=400.0, mu=ERASER_MU, **STIFF)
    return TestScene(build=build, periods=15, ground=False)


def _eraser_cfg(moment: float) -> dict:
    return {"bodies": ("eraser",), "moments": (moment,), "axes": ((1.0, 0.0, 0.0),),
            "offsets": ((ERASER_GAP - ERASER[0], 0.0, 0.0),), "drives": (), "drive_moments": (), "drive_rpm": (),
            "plates": ((0.0, 0.0, 0.0, 1.0, 0.0, 0.0),), "plate_mu_r": (1e4,), "min_distance": 1e-3}


def _eraser_motion(run: Any) -> tuple[np.ndarray, np.ndarray, float, float]:
    e = run.bodies("eraser")[0]
    t, z = np.asarray(run.trace.t), run.trace.body_q[:, e, 2]
    h = float(np.mean(run.trace.body_q[:, e, 0])) - ERASER[0] + ERASER_GAP
    m = run.plugin.config.moments[0]
    pull = 3 * MU0 * ((1e4 - 1) / (1e4 + 1)) * m * m / (32 * math.pi * h ** 4)
    return t, z, pull, float(run.model.body_mass.numpy()[e])


def check_eraser_holds(run: Any) -> LawResult:
    t, z, pull, mass = _eraser_motion(run)
    drop = float(z[0] - z.min())
    ok = ERASER_MU * pull > mass * 9.81
    return LawResult.within(drop if ok else 1.0, 1e-3, f"μF = {ERASER_MU * pull:.3f} N > mg = {mass * 9.81:.3f} N; "
                            f"dropped {drop * 1e3:.3f} mm", "plate attraction must hold the eraser by friction")


def check_eraser_falls(run: Any) -> LawResult:
    e = run.bodies("eraser")[0]
    t, x, z = np.asarray(run.trace.t), run.trace.body_q[:, e, 0], run.trace.body_q[:, e, 2]
    m, mass = run.plugin.config.moments[0], float(run.model.body_mass.numpy()[e])
    gap = x - ERASER[0] + ERASER_GAP
    pull = 3 * MU0 * ((1e4 - 1) / (1e4 + 1)) * m * m / (32 * math.pi * gap ** 4)          # at the recorded gap
    if ERASER_MU * pull[0] >= mass * 9.81:
        return LawResult.within(1.0, 0.05, "law setup: μF must be < mg", "lower the moment")
    acc = 9.81 - ERASER_MU * pull / mass
    v = np.concatenate([[0.0], np.cumsum(0.5 * (acc[1:] + acc[:-1]) * np.diff(t))])
    fit = float(np.sum(0.5 * (v[1:] + v[:-1]) * np.diff(t)))
    err = float(z[0] - z[-1]) / fit - 1.0
    return LawResult.within(err, 0.05, f"fell {1e3 * (z[0] - z[-1]):.1f} mm vs {1e3 * fit:.1f} mm from g - μF(h)/m; "
                            f"gap {1e3 * gap.min():.2f}-{1e3 * gap.max():.2f} mm", "the plate pull must match its "
                            "closed form; a gap that opens means the step is too coarse for the sliding contact")


def magnetism_laws() -> list[Law]:
    lag = TestScene(build=_pivot_bar, periods=75, init=_rpm_per_world, **FREE)
    return [
        Law("inverse_fourth", check_inverse_fourth, TestScene(build=_pair, periods=1, **FREE), config=PAIR),
        Law("energy", check_energy, TestScene(build=lambda b: _pair(b, 0.08), periods=13, dt=5e-4, **FREE),
            config={**PAIR, "moments": (0.3, 0.3), "min_distance": 5e-3}),
        Law("field_oscillation", check_field_oscillation,
            TestScene(build=lambda b: _box(b, "stir_bar", BAR, body_xform(0, 0, 0, (0, 0, math.sin(0.05),
                                                                                    math.cos(0.05)))),
                      periods=300, control_dt=0.005, **FREE),
            config={**STIRRER, "drives": (), "drive_moments": (), "drive_rpm": (), "external_field": (2e-3, 0.0, 0.0)}),
        Law("stirrer_lag", check_stirrer_lag, lag, config=STIRRER),
        Law("plate_force", check_plate_force, _plate_scene(),
            config={**PAIR, "bodies": ("magnet_a",), "moments": (0.05,), "axes": ((1.0, 0.0, 0.0),),
                    "plates": ((0.0, 0.0, 0.0, 0.0, 0.0, 1.0),), "plate_mu_r": (3.0,), "min_distance": 1e-3}),
        Law("eraser_holds", check_eraser_holds, _eraser_scene(), config=_eraser_cfg(0.26)),
        Law("eraser_falls", check_eraser_falls, dataclasses.replace(_eraser_scene(), dt=2.5e-4),
            config=_eraser_cfg(0.13)),
    ]


__all__ = ["default_scene", "magnetism_laws"]
