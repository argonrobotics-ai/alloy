# SPDX-License-Identifier: Apache-2.0
"""``ModalSound``: audio per world and microphone, by modal synthesis from the engine's contacts (engine item 10).

Each sounding object is a body label with a mode table (``presets.py``: beams, plates, DC6's tab and housing, or a
``modes.npz``). Every solver step, the plug-in reads the solver's contacts and forces (``needs_contacts``: the
MuJoCo route, on MuJoCo Warp, or on MuJoCo C (``use_mujoco_cpu``) in one world: Newton's C backend steps only a
multi-world model's template world) and drives the modes in two ways:
- **Impacts.** When a body pair touches for the first time at approach speed v_n > ``rest_speed``, the pair gets
  the impulse J = (1 + e)·m_eff·v_n along the normal, split over that pair's new contacts. m_eff includes
  rotation, and e is ``restitution``, a sound-side parameter (MuJoCo's own bounce comes from solref). Each mode
  receives a velocity kick g_k·J·H(ω_k τ), where τ is the Hertz contact time from E, ν and R of both sides. So a
  harder or faster impact excites higher modes, and the result does not depend on the sim step or on how many
  steps MuJoCo's soft contact spreads the impact over. That pair's contact forces are left out of the sustained
  path until it separates or stays at |v_n| <= ``rest_speed`` for ``settle`` steps (the impact's own v_n = 0
  turning point does not end it), so no impulse is counted twice. A pair that is already touching makes no new
  impact: a hit on a resting contact enters as sustained force (band-limited by the step).
- **Sustained forces.** Contacts with |v_n| <= ``rest_speed`` add g_k·F, interpolated linearly over the step. A
  resting load is a static deflection, so it is silent once the start transient decays.

Each mode is an exact damped oscillator stepped at ``sample_rate``. Modes at or above 0.45·fs are dropped. The
radiation is a monopole proxy, p = ρ_air·S·a/(4π·r): a is the modal surface acceleration, S the object's
radiating area and r its distance to each microphone (world frame). With ``delay`` the signal also arrives r/c
late. Compare spectra and decay, not absolute SPL.

Readings (``read()["<label>"]``):
- ``audio`` [W, M, N] is a ring buffer of the last ``duration`` seconds; ``samples`` [W] is the total written;
- ``kicks`` and ``sustained`` [W, K] are the cumulative modal impulse of each path (N·s);
- ``audio_of(world, mic)`` and ``write_wav(path, world, mic)`` return or save the time-ordered signal.

Example::

    s = ModalSound(bodies=("bar",), presets=("free_beam",), dims=((0.3, 0.02, 0.005),), materials=("steel",),
                   mics=((0.0, 0.0, 0.5),))
    sim = alloy.plugins.simulation(built, [s]); ...; sim.plugins["modal_sound"].write_wav("bar.wav")
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.base import SensorPlugin
from alloy.plugins.errors import PluginError
from alloy.plugins.sound import kernels as K
from alloy.plugins.sound.analysis import write_wav
from alloy.plugins.sound.modes import ModeSet
from alloy.plugins.sound.presets import material, preset

log = logging.getLogger("alloy.plugins.sound")

RHO_AIR = 1.204          # kg/m³, air at 20 °C
C_AIR = 343.0            # m/s


def _each(values: tuple, n: int, name: str) -> list:
    if len(values) not in (1, n):
        raise ValueError(f"{name}: one entry, or one per body ({n}), got {len(values)}")
    return [values[min(i, len(values) - 1)] for i in range(n)]


def oscillator(freq_hz: np.ndarray, zeta: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact one-sample update of q'' + 2ζω q' + ω² q = u (u held over the sample): Φ (K, 4), Γ (K, 2) and the
    damping terms (2ζω, ω²) (K, 2)."""
    w = 2.0 * np.pi * np.asarray(freq_hz, float)
    z = np.asarray(zeta, float)
    wd = w * np.sqrt(1.0 - z ** 2)
    e, c, s = np.exp(-z * w * h), np.cos(wd * h), np.sin(wd * h)
    phi = np.stack([e * (c + z * w / wd * s), e * s / wd, -e * w ** 2 / wd * s, e * (c - z * w / wd * s)], axis=1)
    gamma = np.stack([(1.0 - phi[:, 0]) / w ** 2, -phi[:, 2] / w ** 2], axis=1)
    return phi, gamma, np.stack([2.0 * z * w, w ** 2], axis=1)


class ModalSound(SensorPlugin):
    """Audio from contacts by modal synthesis: impacts with Hertz-duration pulses, sustained forces, exact modal
    oscillators at the audio rate and a monopole proxy to each microphone. See the module docstring."""

    name = "modal_sound"
    summary = "audio per world and microphone: modal synthesis from contact impacts and forces (Hertz pulses)"
    needs_contacts: ClassVar[bool] = True

    @dataclass
    class Config:
        bodies: tuple[str, ...] = ("bar",)                                  # sounding body labels
        presets: tuple[str, ...] = ("free_beam",)                           # one, or one per body (presets.PRESETS)
        dims: tuple[tuple[float, float, float], ...] = ((0.3, 0.02, 0.005),)   # m, per preset (L, w, t / a, b, h)
        materials: tuple[str, ...] = ("steel",)                             # presets.MATERIALS names
        loss: float = 0.0                        # > 0: constant tan δ for every object, overriding the material
        modes: int = 16                          # modes per preset (beams: per bending plane)
        contact_radius: tuple[float, ...] = (0.005,)     # m, Hertz radius of each object where it is hit
        restitution: tuple[float, ...] = (0.5,)          # e of each object's impacts (sound side, ESTIMATE)
        partner: tuple[float, float, float] = (200e9, 0.29, 1e9)   # E [Pa], ν, R [m] of non-sounding bodies
        area: tuple[float, ...] = ()                     # m², radiating area per object; empty = the preset's
        mics: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 0.5),)  # world frame, m
        sample_rate: int = 44100                 # Hz
        duration: float = 1.0                    # s of audio kept per microphone (ring buffer)
        delay: bool = False                      # propagation delay r/c
        max_distance: float = 5.0                # m: the longest delay held
        rest_speed: float = 0.01                 # m/s: slower contacts are sustained forces, faster ones impacts
        settle: int = 3                          # steps a pair must stay slow before an impact episode ends

        def __post_init__(self) -> None:
            n = len(self.bodies)
            if n == 0 or len(set(self.bodies)) != n:
                raise ValueError("bodies: one or more distinct body labels")
            for name in ("presets", "dims", "materials", "contact_radius", "restitution"):
                _each(getattr(self, name), n, name)
            if self.area:
                _each(self.area, n, "area")
            if not self.mics or self.sample_rate < 8000 or self.duration <= 0.0 or self.modes < 1:
                raise ValueError("needs a microphone, sample_rate >= 8000, duration > 0 and modes >= 1")
            if min(self.contact_radius) <= 0.0 or not all(0.0 <= e <= 1.0 for e in self.restitution):
                raise ValueError("contact_radius must be > 0 and restitution in [0, 1]")
            if self.settle < 1:
                raise ValueError("settle: at least 1 step")
            if self.rest_speed <= 0.0 or self.max_distance <= 0.0 or self.partner[0] <= 0.0 or self.partner[2] <= 0:
                raise ValueError("rest_speed, max_distance, partner E and partner R must be > 0")

    # --- setup -------------------------------------------------------------------------------------------------

    def mode_sets(self) -> list[ModeSet]:
        """Each object's modes below 0.45·fs (host; the same in every world)."""
        c, n = self.config, len(self.config.bodies)
        out = []
        for kind, dims, mat, label in zip(_each(c.presets, n, "presets"), _each(c.dims, n, "dims"),
                                          _each(c.materials, n, "materials"), c.bodies):
            ms = preset(kind, tuple(dims), material(mat, c.loss or None), c.modes, name=label)
            out.append(ms.below(0.45 * c.sample_rate))
        return out

    def setup(self, ctx: Any) -> None:
        c, W = self.config, ctx.world_count
        self.sets = self.mode_sets()
        self.O, self.K, self.M = len(self.sets), sum(s.count for s in self.sets), len(c.mics)
        if self.K == 0:
            raise PluginError(f"{self.label}: no mode below 0.45 x sample_rate; raise sample_rate or check dims")
        self._tables(ctx)
        self.ns_max = int(math.ceil(ctx.control_dt * c.sample_rate)) + 2
        self.max_delay = int(math.ceil(c.max_distance / C_AIR * c.sample_rate)) if c.delay else 0
        self.ring = self.ns_max + self.max_delay + 1
        self.length = int(round(c.duration * c.sample_rate))
        K_, O_, S_ = self.K, self.O, self.slots
        for name in ("q", "v", "f_prev", "kicks", "sustained"):
            setattr(self, name, self.add_state(name, ctx.zeros((W, K_))))
        self.pairs = self.add_state("pairs", ctx.zeros((W, O_, S_), wp.int32))
        self.samples = self.add_state("samples", ctx.zeros(W, wp.int64))
        self.frac = self.add_state("frac", ctx.zeros(W, wp.float64))
        self.src = self.add_state("source", ctx.zeros((W, O_, self.ring)))
        self.audio = self.add_state("audio", ctx.zeros((W, self.M, self.length)))
        self._active = ctx.zeros(W * O_ * S_, wp.int32)
        self._fast = ctx.zeros(W * O_ * S_, wp.int32)
        self._new = ctx.zeros(W * O_ * S_, wp.int32)
        self._acc_f = ctx.zeros(W * K_, wp.int64)
        self._acc_j = ctx.zeros(W * K_, wp.int64)
        self._mics = ctx.array(np.asarray(c.mics, np.float32), wp.vec3)
        self._dt: float | None = None

    def _tables(self, ctx: Any) -> None:
        c, model, W = self.config, ctx.model, ctx.world_count
        owner = np.full(model.body_count, -1, np.int32)
        obj_body = np.zeros(W * self.O, np.int32)
        for o, label in enumerate(c.bodies):
            for w, b in enumerate(ctx.bodies(label).numpy()):
                if owner[b] != -1:
                    raise PluginError(f"{self.label}: body {label!r} listed twice")
                owner[b], obj_body[w * self.O + o] = w * self.O + o, b
        bw = model.body_world.numpy()
        counts = np.bincount(bw[bw >= 0], minlength=W)
        self.slots = int(counts.max()) + 1 if counts.size else 1
        mode_start, point_start, gain_start = [0], [0], []
        pts, gains, freq, zeta, mass, rad = [], [], [], [], [], []
        areas = _each(c.area, self.O, "area") if c.area else [s.area for s in self.sets]
        for s, area in zip(self.sets, areas):
            for k in range(s.count):
                gain_start.append(sum(len(g) for g in gains) + k * len(s.points))
            gains.append(s.gains.reshape(-1, 3))
            pts.append(s.points)
            freq.append(s.freq_hz), zeta.append(s.zeta), mass.append(s.mass)
            rad.append(RHO_AIR * area * s.radiation / (4.0 * math.pi))
            mode_start.append(mode_start[-1] + s.count)
            point_start.append(point_start[-1] + len(s.points))
        freq, zeta = np.concatenate(freq), np.concatenate(zeta)
        phi, gamma, damp = oscillator(freq, zeta, 1.0 / c.sample_rate)
        slow = 2.0 * zeta * 2.0 * np.pi * freq / c.sample_rate < 1e-5
        if np.any(slow):
            log.warning("%s: %d mode(s) decay by < 1e-5 per sample (2ζω/fs, e.g. %.0f Hz at ζ %.1e); the float32 "
                        "oscillator then misreads their decay by up to ~5 %% (WP13); raise the loss or check it",
                        self.label, int(slow.sum()), float(freq[slow][0]), float(zeta[slow][0]))
        f32 = lambda a: np.ascontiguousarray(a, np.float32)  # noqa: E731
        ob = K.Objects()
        ob.sound_of_body = ctx.array(owner, wp.int32)
        ob.mode_start = ctx.array(np.asarray(mode_start, np.int32), wp.int32)
        ob.point_start = ctx.array(np.asarray(point_start, np.int32), wp.int32)
        ob.gain_start = ctx.array(np.asarray(gain_start, np.int32), wp.int32)
        ob.points = ctx.array(f32(np.concatenate(pts)), wp.vec3)
        ob.gains = ctx.array(f32(np.concatenate(gains)), wp.vec3)
        ob.omega = ctx.array(f32(2.0 * np.pi * freq), wp.float32)
        mats = [material(m) for m in _each(c.materials, self.O, "materials")]
        hz = [(m.youngs, m.poisson, r, e) for m, r, e in zip(mats, _each(c.contact_radius, self.O, "contact_radius"),
                                                            _each(c.restitution, self.O, "restitution"))]
        ob.hertz = ctx.array(f32(hz), wp.vec4)
        ob.partner = wp.vec3(*c.partner)
        ob.body_world = model.body_world
        ob.world_start = model.body_world_start
        ob.objects, ob.modes, ob.slots = self.O, self.K, self.slots
        self._ob = ob
        self._obj_body = ctx.array(obj_body, wp.int32)
        self._mode_start = ob.mode_start
        self._inv_mass = ctx.array(f32(1.0 / np.concatenate(mass)), wp.float32)
        self._trans = ctx.array(f32(phi), wp.vec4)
        self._drive = ctx.array(f32(gamma), wp.vec2)
        self._damp = ctx.array(f32(damp), wp.vec2)
        self._rad = ctx.array(f32(np.concatenate(rad)), wp.float32)

    def check_dt(self, dt: float) -> None:
        ns = int(math.floor(dt * self.config.sample_rate)) + 2
        if ns > self.ns_max:
            raise PluginError(f"{self.label}: step {dt} s exceeds the control period the buffers were sized for")
        if self._dt is not None and dt != self._dt:
            raise PluginError(f"{self.label}: one step size per run (had {self._dt} s, now {dt} s): the audio clock "
                              f"counts steps of a fixed length")
        if self._dt is None:
            self.ns = ns
            self._out = self.ctx.zeros(self.ctx.world_count * self.K * ns)
            self._dt = dt

    # --- step --------------------------------------------------------------------------------------------------

    def measure(self, state_in: newton.State, state_out: newton.State, dt: float) -> None:
        ctx, c = self.ctx, self.config
        con, force = ctx.contacts, ctx.contact_force
        if con is None or force is None:
            raise PluginError(f"{self.label} needs contact forces: run it on the MuJoCo route (MuJoCo Warp, or "
                              f"MuJoCo C in one world), where the plug-in host exports them")
        if self._dt != dt:
            raise PluginError(f"{self.label}: check_dt({dt}) was not called before stepping")
        model, dev, W = ctx.model, ctx.device, ctx.world_count
        spf = wp.float64(dt * c.sample_rate)
        geo = [con.rigid_contact_count, con.rigid_contact_shape0, con.rigid_contact_shape1, con.rigid_contact_point0,
               con.rigid_contact_point1, con.rigid_contact_normal, force, model.shape_body, state_in.body_q,
               state_in.body_qd, model.body_com]
        n_con = con.rigid_contact_max
        flat_pairs = self.pairs.reshape((-1,))
        if n_con > 0:
            wp.launch(K.classify, dim=n_con, device=dev,
                      inputs=[*geo, self._ob, float(c.rest_speed), flat_pairs],
                      outputs=[self._active, self._fast, self._new, self._acc_f])
            wp.launch(K.impacts, dim=n_con, device=dev,
                      inputs=[*geo, model.body_inv_mass, model.body_inv_inertia, self._ob, float(c.rest_speed),
                              flat_pairs, self._new],
                      outputs=[self._acc_j])
        flat = {k: getattr(self, k).reshape((-1,)) for k in ("q", "v", "f_prev", "kicks", "sustained")}
        wp.launch(K.synth, dim=W * self.K, device=dev,
                  inputs=[self._acc_f, self._acc_j, self.frac, spf, float(dt), self.K, self.ns, self._inv_mass,
                          self._trans, self._drive, self._damp, self._rad],
                  outputs=[flat["q"], flat["v"], flat["f_prev"], flat["kicks"], flat["sustained"], self._out])
        src = self.src.reshape((-1,))
        wp.launch(K.source, dim=W * self.O * self.ns, device=dev,
                  inputs=[self._out, self.frac, self.samples, spf, self._mode_start, self.O, self.K, self.ns,
                          self.ring], outputs=[src])
        delay_per_m = c.sample_rate / C_AIR if c.delay else 0.0
        wp.launch(K.mix, dim=W * self.M * self.ns, device=dev,
                  inputs=[src, self.frac, self.samples, spf, self._obj_body, state_out.body_q, model.body_com,
                          self._mics, self.O, self.ns, self.ring, self.length, float(delay_per_m), self.max_delay,
                          1e-3], outputs=[self.audio.reshape((-1,))])
        wp.launch(K.advance_pairs, dim=W * self.O * self.slots, device=dev,
                  inputs=[self._active, self._fast, int(c.settle), flat_pairs, self._new])
        wp.launch(K.advance_clock, dim=W, device=dev, inputs=[self.frac, self.samples, spf])

    # --- host helpers ------------------------------------------------------------------------------------------

    def audio_of(self, world: int = 0, mic: int = 0) -> np.ndarray:
        """The time-ordered samples of one world and microphone (the last ``duration`` seconds at most)."""
        n = int(self.samples.numpy()[world])
        ring = self.audio.numpy()[world, mic]
        if n <= self.length:
            return ring[:n].astype(np.float64)
        k = n % self.length
        return np.concatenate([ring[k:], ring[:k]]).astype(np.float64)

    def write_wav(self, path: str, world: int = 0, mic: int = 0, normalise: bool = True) -> float:
        """Save one world's microphone as 16-bit WAV; returns the scale applied (Pa per full scale = 1/scale)."""
        return write_wav(path, self.audio_of(world, mic), self.config.sample_rate, normalise)

    def test_scene(self):
        from alloy.plugins.sound.laws import strike_scene  # noqa: PLC0415 - laws import this module

        return strike_scene()

    def laws(self):
        from alloy.plugins.sound.laws import sound_laws  # noqa: PLC0415

        return sound_laws()


__all__ = ["C_AIR", "RHO_AIR", "ModalSound", "oscillator"]
