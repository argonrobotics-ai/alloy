# SPDX-License-Identifier: Apache-2.0
"""``ModalSound`` laws (ours; plan section 5 and the WP13 handoff), on a steel bar struck by steel balls.

The bar is 300 x 20 x 5 mm and the balls are 5 mm in radius. There is no gravity and no ground, so the bar and the
balls touch only each other. A "fixed" bar is held by a fixed joint, so strikes do not move it and stay
independent. The plug-in models the bar as the free-free preset either way. The default strike is at x = 0.14 m,
near the end, where every free-free mode has an antinode (at 0.12 m mode 3 sits near a node, 40 dB down). Laws:
- **S-1 frequencies:** a free bar struck once. The audio's spectral peaks match Euler-Bernoulli free-free
  f_k = (β_k L)² / (2π L²)·sqrt(E t² / (12 ρ)) for the first 5 modes within 1 % (β_k L from tables, not from the
  plug-in's root finder).
- **S-1 FEM:** the same free-bar strike with the ``fem_box`` preset (P2 tets). The audio's peaks match the tet
  mesh's own eigenvalues for every mode the strike excites (kick > 1 % of the largest) within 1 %.
- **S-2 decay:** tan δ = 0.01. Each of the first 3 modes' envelope decays at ζω = (tan δ / 2)·ω within 5 %.
- **S-3 linearity:** two strikes in world 0 give world 1 (first strike only) + world 2 (second only), within 1e-4
  of the peak (float32 rounding).
- **S-4 distance:** microphones at 0.5 and 1.0 m with delay on: the peak ratio is 1/2 within 1 %, and the lag is
  0.5 m / c within 1.5 samples.
- **S-5 contact duration:** speeds 0.3, 0.8 and 2 m/s and a hard-plastic striker (E 2 GPa, ESTIMATE). The spectral
  centroid rises with speed, and its log slope matches a NumPy synthesis from the Hertz formula within 20 %.
- **S-6 step independence:** at dt/4 the modal kicks match the dt-free prediction g_k·(1+e)·m_eff·v·H(ω_k τ)
  within 5 %.
- **impulse_once:** at the default step the kicks match that prediction within 1e-4, and the sustained path adds
  at most 1 % of it. MuJoCo's multi-step soft impact is not counted twice.
- **null:** a ball moving away. Every audio sample is exactly 0.
- **rest:** a bar on the ground under gravity with a ball dropped 5 mm onto it, tan δ = 0.05 so the drop's ringing
  dies within 0.3 s. The last 0.1 s of a 0.8 s run is quiet (RMS < 1e-3 of the drop's peak): a resting load is a
  static deflection, and MuJoCo's resting jitter must stay inaudible.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from alloy._vendor import newton

from alloy.plugins.harness.scene import Law, LawResult, TestScene, body_xform
from alloy.plugins.sound.analysis import centroid, magnitude_spectrum, mode_decay_rate, peak_near
from alloy.plugins.sound.hertz import contact_time, effective, effective_modulus, pulse_weight
from alloy.plugins.sound.presets import material

L, W_BAR, T_BAR = 0.3, 0.02, 0.005
BALL_R, STEEL, GAP = 0.005, 7850.0, 0.003
EB_FREE = (4.73004074, 7.85320462, 10.99560784, 14.13716549, 17.27875966)   # Blevins table 8-1 (free-free)
STRIKER = (2e9, 0.35, BALL_R)          # E, ν, R of a hard-plastic striker (ESTIMATE)
STEEL_BALL = (200e9, 0.29, BALL_R)
FLAT = (1.0,)                          # m: the bar's own radius at the hit (flat)


def _bar(b: newton.ModelBuilder, fixed: bool) -> None:
    xf = body_xform(0.0, 0.0, 0.0)
    cfg = newton.ModelBuilder.ShapeConfig(density=STEEL)
    if fixed:
        bar = b.add_link(xform=xf, label="bar")
        b.add_shape_box(bar, hx=0.5 * L, hy=0.5 * W_BAR, hz=0.5 * T_BAR, cfg=cfg)
        b.add_articulation([b.add_joint_fixed(-1, bar, parent_xform=xf, label="bar_mount")], label="bar")
    else:
        bar = b.add_body(xform=xf, label="bar")
        b.add_shape_box(bar, hx=0.5 * L, hy=0.5 * W_BAR, hz=0.5 * T_BAR, cfg=cfg)


def _launch(speeds: dict[str, tuple[float, ...]]):
    """init(sim): each ball label gets a downward speed per world (the last entry repeats)."""
    def init(sim: Any) -> None:
        model = sim.model
        starts, child = model.joint_qd_start.numpy(), model.joint_child.numpy()
        qd = sim.state.joint_qd.numpy()
        labels, world = list(model.body_label), model.body_world.numpy()
        for label, per_world in speeds.items():
            ids = sorted((int(world[i]), i) for i, lab in enumerate(labels) if lab == label)
            for w, b in ids:
                j = int(np.flatnonzero(child == b)[0])
                qd[starts[j] + 2] = -per_world[min(w, len(per_world) - 1)]
        sim.state.joint_qd.assign(qd)
        newton.eval_fk(model, sim.state.joint_q, sim.state.joint_qd, sim.state)
    return init


def strike_scene(fixed: bool = False, balls: tuple = (("ball", 0.14, GAP),), speeds: dict | None = None,
                 periods: int = 25, dt: float = 2.0e-3, worlds: int = 3) -> TestScene:
    """A steel bar struck from above by steel balls (label, x along the bar, start gap) in zero gravity."""
    def build(b: newton.ModelBuilder) -> None:
        _bar(b, fixed)
        cfg = newton.ModelBuilder.ShapeConfig(density=STEEL)
        for label, x, gap in balls:
            s = b.add_body(xform=body_xform(x, 0.0, 0.5 * T_BAR + BALL_R + gap), label=label)
            b.add_shape_sphere(s, radius=BALL_R, cfg=cfg)
    return TestScene(build=build, worlds=worlds, periods=periods, dt=dt, ground=False, gravity=(0.0, 0.0, 0.0),
                     init=_launch(speeds or {"ball": (1.0,)}))


def _rest_scene() -> TestScene:
    def build(b: newton.ModelBuilder) -> None:
        cfg = newton.ModelBuilder.ShapeConfig(density=STEEL)
        bar = b.add_body(xform=body_xform(0.0, 0.0, 0.5 * T_BAR + 1e-4), label="bar")
        b.add_shape_box(bar, hx=0.5 * L, hy=0.5 * W_BAR, hz=0.5 * T_BAR, cfg=cfg)
        s = b.add_body(xform=body_xform(0.05, 0.0, T_BAR + BALL_R + 5e-3), label="ball")
        b.add_shape_sphere(s, radius=BALL_R, cfg=cfg)
    return TestScene(build=build, worlds=2, periods=40)


def _audio(run: Any, world: int = 0, mic: int = 0) -> tuple[np.ndarray, int]:
    return run.plugin.audio_of(world, mic), run.plugin.config.sample_rate


def _eb(k: int) -> float:
    m = material("steel")
    return EB_FREE[k] ** 2 / (2.0 * math.pi * L ** 2) * math.sqrt(m.youngs * T_BAR ** 2 / (12.0 * m.density))


def check_frequencies(run: Any) -> LawResult:
    x, fs = _audio(run)
    freqs, mag = magnitude_spectrum(x, fs, 1 << 19)
    errs = [peak_near(freqs, mag, _eb(k)) / _eb(k) - 1.0 for k in range(5)]
    worst = max(errs, key=abs)
    return LawResult.within(worst, 0.01, "modes 1-5 vs Euler-Bernoulli: " + ", ".join(f"{e:+.2%}" for e in errs),
                            "check modes.beam_roots / beam_modes against the closed form")


def check_fem_frequencies(run: Any) -> LawResult:
    x, fs = _audio(run)
    ms, kicks = run.plugin.sets[0], np.abs(run.plugin.read()["kicks"][0])
    freqs, mag = magnitude_spectrum(x, fs, 1 << 19)
    hit = np.flatnonzero(kicks > 0.01 * kicks.max())
    errs = [peak_near(freqs, mag, ms.freq_hz[k], 0.01) / ms.freq_hz[k] - 1.0 for k in hit]
    worst = max(errs, key=abs) if errs else 1.0
    return LawResult.within(worst, 0.01, f"{len(hit)} excited FEM modes vs the tet mesh eigenvalues: worst "
                            f"{worst:+.3%}", "fem.mesh_modes: gains and frequencies must reach the synthesis unchanged")


def check_decay(run: Any) -> LawResult:
    x, fs = _audio(run)
    loss = run.plugin.config.loss
    errs = []
    for k in range(3):
        want = 0.5 * loss * 2.0 * math.pi * _eb(k)
        errs.append(mode_decay_rate(x, fs, _eb(k)) / want - 1.0)
    worst = max(errs, key=abs)
    return LawResult.within(worst, 0.05, "decay rate vs zeta*omega, modes 1-3: " + ", ".join(f"{e:+.2%}" for e in errs),
                            "oscillator(): exact damped update; zeta = tan delta / 2")


def check_linearity(run: Any) -> LawResult:
    both, a, b = (_audio(run, w)[0] for w in range(3))
    n = min(len(both), len(a), len(b))
    err = float(np.max(np.abs(both[:n] - a[:n] - b[:n])) / max(np.max(np.abs(both[:n])), 1e-30))
    return LawResult.within(err, 1e-4, f"|two strikes - (first + second)| / peak = {err:.2e}",
                            "the synthesis must be linear in the contact inputs (no state shared between modes)")


def check_distance(run: Any) -> LawResult:
    near, fs = _audio(run, 0, 0)
    far, _ = _audio(run, 0, 1)
    ratio = float(np.max(np.abs(far)) / np.max(np.abs(near)))
    xc = np.fft.irfft(np.fft.rfft(far, 2 * len(far)) * np.conj(np.fft.rfft(near, 2 * len(near))))
    lag = int(np.argmax(xc))
    want_lag = 0.5 / 343.0 * fs
    err = ratio / 0.5 - 1.0
    if abs(lag - want_lag) > 1.5:
        err = math.copysign(1.0, err) if err else 1.0
    return LawResult.within(err, 0.01, f"far/near peak ratio {ratio:.4f} (want 0.5), lag {lag} samples "
                            f"(want {want_lag:.1f})", "mix(): p = src(t - r/c) / r")


def predicted_kicks(run: Any, world: int, speed: float, x_hit: float) -> np.ndarray:
    """g_k · (1 + e)·m_eff·v · H(ω_k τ): the dt-free modal impulse of one ball hitting the bar on its centre line
    at ``x_hit`` (host NumPy, independent of the kernels). m_eff = 1 / (1/m_ball + 1/m_bar + (r×n)ᵀ I⁻¹ (r×n)), with
    the bar's free-body inertia: the plug-in ignores joints when it predicts the impulse."""
    snd, model = run.plugin, run.model
    c, ms = snd.config, snd.sets[0]
    inv_m, inv_i = model.body_inv_mass.numpy(), model.body_inv_inertia.numpy()
    bar = run.bodies("bar")[world]
    rn = np.cross([x_hit, 0.0, 0.5 * T_BAR], [0.0, 0.0, 1.0])     # the bar starts unrotated; the ball's r×n = 0
    m_eff = 1.0 / (inv_m[run.bodies("ball")[world]] + inv_m[bar] + rn @ inv_i[bar] @ rn)
    mat = material(c.materials[0])
    e_eff = effective_modulus(mat.youngs, mat.poisson, c.partner[0], c.partner[1])
    tau = contact_time(m_eff, effective(c.contact_radius[0], c.partner[2]), e_eff, speed)
    p = int(np.argmin(np.linalg.norm(ms.points - np.array([x_hit, 0.0, 0.0]), axis=1)))
    j = (1.0 + c.restitution[0]) * m_eff * speed
    return -ms.gains[:, p, 2] * j * pulse_weight(2.0 * math.pi * ms.freq_hz, tau)   # the ball pushes the bar down


def _kick_error(run: Any, speed: float, x_hit: float) -> tuple[float, float]:
    got = run.plugin.read()["kicks"][0]
    want = predicted_kicks(run, 0, speed, x_hit)
    scale = float(np.max(np.abs(want)))
    sus = float(np.max(np.abs(run.plugin.read()["sustained"][0])))
    return float(np.max(np.abs(got - want)) / scale), sus / scale


def check_impulse_once(run: Any) -> LawResult:
    err, sus = _kick_error(run, 1.0, 0.0)
    value = max(err / 1e-4, sus / 0.01)
    return LawResult.within(value, 1.0, f"kicks vs prediction {err:.2e} (bound 1e-4); sustained path {sus:.2e} of "
                            f"the kick (bound 1e-2)", "impacts(): one kick per new pair; classify(): fast contacts "
                            "stay out of the sustained path")


def check_step_independence(run: Any) -> LawResult:
    err, _ = _kick_error(run, 1.0, 0.0)
    return LawResult.within(err, 0.05, f"kicks at dt/4 vs the dt-free prediction: {err:.2e}",
                            "the impact kick must use the pre-contact speed, not the step's impulse")


def _synth_centroid(run: Any, kicks: np.ndarray, fs: int, n: int) -> float:
    ms, c = run.plugin.sets[0], run.plugin.config
    t = np.arange(n) / fs
    w = 2.0 * math.pi * ms.freq_hz
    sig, wd = ms.zeta * w, w * np.sqrt(1.0 - ms.zeta ** 2)
    v0 = kicks / ms.mass
    acc = v0[:, None] * np.exp(-sig[:, None] * t) * ((sig[:, None] ** 2 - wd[:, None] ** 2)
                                                     * np.sin(wd[:, None] * t) / wd[:, None]
                                                     - 2.0 * sig[:, None] * np.cos(wd[:, None] * t))
    del c
    return centroid((acc * ms.radiation[:, None]).sum(axis=0), fs, fmin=100.0)


def check_contact_duration(run: Any) -> LawResult:
    speeds = (0.3, 0.8, 2.0)
    got, want = [], []
    for w, v in enumerate(speeds):
        x, fs = _audio(run, w)
        start = int(np.argmax(np.abs(x) > 0))
        got.append(centroid(x[start:], fs, fmin=100.0))
        want.append(_synth_centroid(run, predicted_kicks(run, w, v, 0.0), fs, len(x) - start))
    lv = math.log(speeds[-1] / speeds[0])
    s_got, s_want = math.log(got[-1] / got[0]) / lv, math.log(want[-1] / want[0]) / lv
    rising = got[0] < got[1] < got[2]
    err = s_got / s_want - 1.0 if (rising and s_want > 0) else 1.0
    return LawResult.within(err, 0.2, f"centroid {', '.join(f'{g:.0f}' for g in got)} Hz at {speeds} m/s; log "
                            f"slope {s_got:.3f} vs Hertz {s_want:.3f}", "impacts(): tau = 2.87 (m*^2/(R* E*^2 v))^0.2")


def check_null(run: Any) -> LawResult:
    peak = float(np.max(np.abs(run.plugin.read()["audio"])))
    return LawResult.within(peak, 0.0, f"max |audio| = {peak:.3g} with no contact", "no contact must give silence")


def check_rest(run: Any) -> LawResult:
    x, fs = _audio(run)
    tail = x[-int(0.1 * fs):]
    ratio = float(np.sqrt(np.mean(tail ** 2)) / max(np.max(np.abs(x)), 1e-30))
    return LawResult.within(ratio, 1e-3, f"RMS of the last 0.1 s / peak = {ratio:.2e}",
                            "sustained forces must be smooth over the step (linear interpolation)")


def sound_laws() -> list[Law]:
    fixed_hit = {"fixed": True, "balls": (("ball", 0.0, GAP),)}
    hard = {"contact_radius": FLAT, "partner": STEEL_BALL}
    return [
        Law("s1_frequencies", check_frequencies, scene=strike_scene(periods=40), config=hard),
        Law("s1_fem_frequencies", check_fem_frequencies, scene=strike_scene(periods=40),
            config={**hard, "presets": ("fem_box",), "modes": 12}),
        Law("s2_decay", check_decay, scene=strike_scene(fixed=True, periods=40), config={**hard, "loss": 0.01}),
        Law("s3_linearity", check_linearity, config=hard, scene=strike_scene(
            fixed=True, balls=(("ball_a", 0.1, GAP), ("ball_b", -0.08, 0.03)),
            speeds={"ball_a": (1.0, 1.0, 0.0), "ball_b": (1.0, 0.0, 1.0)})),
        Law("s4_distance", check_distance, scene=strike_scene(fixed=True, worlds=2),
            config={**hard, "mics": ((0.0, 0.0, 0.5), (0.0, 0.0, 1.0)), "delay": True}),
        Law("s5_contact_duration", check_contact_duration, config={"contact_radius": FLAT, "partner": STRIKER},
            scene=strike_scene(**fixed_hit, speeds={"ball": (0.3, 0.8, 2.0)}, dt=1e-3, periods=15)),
        Law("s6_step_independence", check_step_independence, config=hard,
            scene=strike_scene(**fixed_hit, dt=5e-4, periods=10)),
        Law("impulse_once", check_impulse_once, config=hard, scene=strike_scene(**fixed_hit, periods=10)),
        Law("null", check_null, scene=strike_scene(speeds={"ball": (-1.0,)}, periods=5)),
        Law("rest", check_rest, scene=_rest_scene(), config={"loss": 0.05}),
    ]


__all__ = ["predicted_kicks", "sound_laws", "strike_scene"]
