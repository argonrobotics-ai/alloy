# SPDX-License-Identifier: Apache-2.0
"""Sim-to-real layers for a ``VectorEnv``: observation noise, observation delay, observation history, action delay.

Each layer is opt-in (``Task(layers=[...])``), works per world on the env's array backend (NumPy, or torch on the
simulation device) and draws its random numbers from the backend's seeded generator, so ``env.reset(seed=...)``
repeats on the CPU. The simulation itself is never changed: the layers change what the policy sees and what the
writer receives. Layers run in the order given; the usual order is noise, then delay, then history.

Example::

    import dataclasses
    from alloy.envs import VectorEnv
    from alloy.envs.realism import ActionDelay, ObsDelay, ObsHistory, ObsNoise
    from alloy.envs.tasks import gripper_lift_scene, gripper_lift_task

    layers = [ObsNoise(0.01), ObsDelay(1), ObsHistory(3), ActionDelay((0, 2))]
    task = dataclasses.replace(gripper_lift_task(), layers=layers)
    env = VectorEnv(gripper_lift_scene(8), task, worlds=8)
    obs, info = env.reset(seed=0)                       # obs is (8, 3 x the plain width)
    names = env.obs_names()                             # the plain names; realism.obs_names(env) has the history's
"""

from __future__ import annotations

import fnmatch
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from alloy.envs.backend import pick_backend
from alloy.envs.errors import EnvError

POLICY = "policy"
STATE_KEY = "state"        # the flat observation's key when cameras add images beside it


def _fail(message: str, fix: str) -> EnvError:
    return EnvError(message, fix, field="layers")


# --- backend helpers (the backend has no gather, roll or normal draw) ---------------------------------------------

def _mod(xp: Any) -> Any:
    """The array module behind a backend: torch for the torch backend, else NumPy."""
    return getattr(xp, "torch", np)


def _bcast(mask: Any, ndim: int) -> Any:
    return mask.reshape((-1,) + (1,) * (ndim - 1))


def _gather(xp: Any, buf: Any, idx: Any) -> Any:
    """``buf[w, idx[w]]`` for every world ``w``: (W, K, ...) with an int (W,) index -> (W, ...)."""
    rows = xp.asarray(np.arange(buf.shape[0]), "int64")
    return buf[rows, idx]


class _Stream:
    """A random stream of a layer's own, so a layer's draws never move the env's stream (the starts the randomizer
    draws at a reset stay those of an env without the layer). It is seeded from the env's stream at the first draw and
    again after the env's stream is seeded anew (``env.reset(seed=...)``), so a seeded reset repeats it; any other
    reset, an episode end of every world at once included, keeps drawing from it, so each episode gets fresh draws."""

    def __init__(self, xp: Any) -> None:
        self.xp, self.own, self.seeded, self.source = xp, pick_backend(xp.name, xp.device), False, None

    def _env_stream(self) -> Any:
        """The env's generator object; ``xp.seed`` replaces it, so a new object means the env was seeded again."""
        return getattr(self.xp, "_gen", None) or getattr(self.xp, "_rng", None)

    def reseed(self) -> None:
        """Seed from the env's stream without advancing it. Example: ``stream.reseed()``."""
        gen = getattr(self.xp, "_gen", None)
        rng = getattr(self.xp, "_rng", None)
        raw = gen.get_state().cpu().numpy().tobytes() if gen is not None else repr(
            rng.bit_generator.state).encode() if rng is not None else b""
        self.own.seed(int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") >> 1)
        self.seeded, self.source = True, self._env_stream()

    def uniform(self, shape: Sequence[int], lo: Any, hi: Any) -> Any:
        """Uniform draws in [lo, hi). Example: ``stream.uniform((4,), 0, 1)``."""
        if not self.seeded:
            self.reseed()
        return self.own.uniform(shape, lo, hi)

    def normal(self, shape: Sequence[int]) -> Any:
        """Standard normal draws (Box-Muller). Example: ``stream.normal((4, 3))``."""
        m = _mod(self.xp)
        u1 = self.xp.clip(self.uniform(shape, 0.0, 1.0), 1e-7, 1.0)
        return m.sqrt(-2.0 * m.log(u1)) * m.cos(2.0 * np.pi * self.uniform(shape, 0.0, 1.0))

    def steps(self, lo: int, hi: int, n: int) -> Any:
        """An int64 (n,) draw of whole steps in [lo, hi]."""
        if lo == hi:
            return self.xp.asarray(np.full(n, lo), "int64")
        floored = _mod(self.xp).floor(self.uniform((n,), lo, hi + 1))
        return self.xp.asarray(self.xp.clip(floored, lo, hi), "int64")

    def on_reset(self, mask: Any) -> None:
        """Reseed only when the env's stream was seeded since the last seeding (``env.reset(seed=...)``); a reset
        without a new seed, of some worlds or all of them, draws on. Example: ``stream.on_reset(mask)``."""
        if self.seeded and self._env_stream() is not self.source:
            self.reseed()


def _push(xp: Any, buf: Any, x: Any, only: Any) -> Any:
    """The buffer (W, K, ...; newest last) after ``x`` arrives. With ``only`` (a (W,) bool mask) those worlds start
    afresh with every slot holding ``x`` and the others keep their buffer."""
    depth = buf.shape[1]
    if only is None:
        return xp.cat([buf[:, 1:], x[:, None]], axis=1)
    filled = xp.stack([x] * depth, axis=1)
    return xp.where(_bcast(only, buf.ndim), filled, buf)


def _check_steps(steps: Any, what: str) -> tuple[int, int]:
    """``steps`` (an int or ``(lo, hi)``) as ``(lo, hi)`` whole control steps."""
    lo, hi = (steps, steps) if np.isscalar(steps) else tuple(steps)
    ok = float(lo).is_integer() and float(hi).is_integer() and 0 <= lo <= hi
    if not ok:
        raise _fail(f"{what} steps is {steps!r}", "pass a whole number >= 0, or (lo, hi) with 0 <= lo <= hi")
    return int(lo), int(hi)


# --- observation access ------------------------------------------------------------------------------------------

def _get(obs: Any, group: str) -> Any:
    """The array of one group inside an observation (None when it is not there)."""
    if not isinstance(obs, Mapping):
        return obs if group == POLICY else None
    if group in obs:
        return obs[group]
    return obs.get(STATE_KEY) if group == POLICY else None


def _put(obs: Any, group: str, value: Any) -> Any:
    if not isinstance(obs, Mapping):
        return value
    key = group if group in obs else STATE_KEY
    return {**obs, key: value}


class Layer:
    """The base of every layer: do-nothing hooks the env calls (``attach`` once, ``on_action`` before the writer,
    ``on_obs`` on every observation, ``on_reset`` after worlds reset). ``names`` maps a group's column labels.

    Example: ``class Clip(Layer): on_action = lambda self, a: a.clip(-1, 1)``."""

    def attach(self, env: Any) -> None:
        """Called once, before the observation space is built. Example: ``layer.attach(env)``."""

    def on_action(self, actions: Any) -> Any:
        """The actions the writer will apply. Example: ``layer.on_action(actions)``."""
        return actions

    def on_obs(self, obs: Any, only: Any) -> Any:
        """The observation the policy sees. ``only``: None, or the (W,) bool mask of worlds to start afresh."""
        return obs

    def on_reset(self, mask: Any) -> None:
        """Worlds in the (W,) bool ``mask`` were reset. Example: ``layer.on_reset(mask)``."""

    def names(self, group: str, names: list[str]) -> list[str]:
        """The labels of ``group`` after this layer, given those before it. Example: ``l.names("policy", n)``."""
        return names


def obs_names(env: Any, group: str | None = None) -> list[str]:
    """Column labels of an observation group after every layer (``env.obs_names`` gives the labels before layers,
    which differ only with ``ObsHistory``).

    Example: ``obs_names(env, "policy")[:3]``."""
    group = group or POLICY
    names = env.obs_names(group if env.obs_groups else None)       # an env without groups has one flat vector
    for layer in env.layers:
        names = layer.names(group, names)
    return list(names)


class _GroupLayer(Layer):
    """A layer that rewrites chosen observation groups and remembers its last output per group."""

    def __init__(self, groups: Sequence[str]) -> None:
        self.groups = (groups,) if isinstance(groups, str) else tuple(groups)
        self.env: Any = None
        self._last: dict[str, Any] = {}
        if not self.groups:
            raise _fail("groups is empty", "name at least one group, e.g. groups=('policy',)")

    def attach(self, env: Any) -> None:
        valid = set(env.obs_groups) | {POLICY}
        for g in self.groups:
            if g not in valid:
                raise _fail(f"no observation group named {g!r}", f"use one of {sorted(valid)}")
        self.env, self.xp, self.stream = env, env.xp, _Stream(env.xp)

    def on_reset(self, mask: Any) -> None:
        self.stream.on_reset(mask)

    def _process(self, group: str, x: Any, only: Any) -> Any:
        raise NotImplementedError

    def on_obs(self, obs: Any, only: Any) -> Any:
        for g in self.groups:
            x = _get(obs, g)
            if x is None:
                continue
            y, last = self._process(g, x, only), self._last.get(g)
            if only is not None and last is not None:
                y = self.xp.where(_bcast(only, y.ndim), y, last)
            self._last[g] = y
            obs = _put(obs, g, y)
        return obs


# --- observation noise -------------------------------------------------------------------------------------------

def _column_scale(names: list[str], spec: float | Mapping[str, float], what: str) -> np.ndarray:
    """A (D,) array: the noise size of every column from a number or a ``{label pattern: size}`` dict."""
    if not isinstance(spec, Mapping):
        return np.full(len(names), float(spec), np.float32)
    out = np.zeros(len(names), np.float32)
    for pattern, size in spec.items():
        hit = [i for i, n in enumerate(names)
               if n == pattern or n.startswith(pattern + "[") or fnmatch.fnmatchcase(n, pattern)]
        if not hit:
            raise _fail(f"{what} pattern {pattern!r} matches no observation column",
                        f"use labels from env.obs_names(), e.g. {names[:2]}")
        out[hit] = float(size)
    return out


class ObsNoise(_GroupLayer):
    """Additive noise on observation columns, drawn fresh per world and step (white, no bias or drift).

    ``std``: a number for every column of the groups, or ``{label: std}`` (a label or a prefix such as ``"cube.pos"``
    for ``cube.pos[0..2]``, or an ``fnmatch`` pattern, from ``env.obs_names(group)``). ``uniform``: the same but a
    half-width of uniform noise in ``[-u, u]``, used instead of ``std``. Columns not named stay exact, and so do
    groups not in ``groups`` (the critic by default).

    Example: ``ObsNoise({"gripper/lift.q": 0.002, "cube.pos": 0.005})``."""

    def __init__(self, std: float | Mapping[str, float] = 0.0, uniform: float | Mapping[str, float] | None = None,
                 groups: Sequence[str] = (POLICY,)) -> None:
        super().__init__(groups)
        self.std, self.uniform = std, uniform
        values = list(uniform.values() if isinstance(uniform, Mapping) else [uniform]) if uniform is not None else []
        values += list(std.values()) if isinstance(std, Mapping) else [std]
        if any(v < 0 for v in values):
            raise _fail("a noise size is negative", "pass std >= 0 and uniform >= 0")
        self._scale: dict[str, Any] = {}

    def _scale_of(self, group: str) -> Any:
        """The (D,) noise sizes of a group, resolved at the first observation (the labels exist from then on)."""
        if group not in self._scale:
            spec, self.kind = (self.std, "std") if self.uniform is None else (self.uniform, "uniform")
            names = self.env.obs_names(group if self.env.obs_groups else None)   # a flat list is the policy's
            for layer in self.env.layers[:self.env.layers.index(self)]:
                names = layer.names(group, names)
            scale = _column_scale(names, spec, self.kind)
            self._scale[group] = scale if scale.any() else None
        return self._scale[group]

    def _process(self, group: str, x: Any, only: Any) -> Any:
        scale = self._scale_of(group)
        if scale is None:
            return x
        xp = self.xp
        draw = self.stream.normal(x.shape) if self.kind == "std" else self.stream.uniform(x.shape, -1.0, 1.0)
        return x + draw * xp.asarray(scale)


# --- buffered observation layers ---------------------------------------------------------------------------------

class _Buffered(_GroupLayer):
    """A layer that keeps the last ``depth`` inputs of a group (newest last) and reads its output from them."""

    depth = 1

    def __init__(self, groups: Sequence[str]) -> None:
        super().__init__(groups)
        self._buf: dict[str, Any] = {}

    def _read(self, buf: Any) -> Any:
        raise NotImplementedError

    def _process(self, group: str, x: Any, only: Any) -> Any:
        buf = self._buf.get(group)
        if buf is None:
            buf, only = self.xp.stack([x] * self.depth, axis=1), None
        buf = _push(self.xp, buf, x, only)
        self._buf[group] = buf
        return self._read(buf)


class ObsDelay(_Buffered):
    """Each world sees the observation from ``k`` control steps ago. ``steps`` is ``k`` or ``(lo, hi)``, drawn per world
    at every reset. After a reset the first ``k`` observations repeat the first one.

    Example: ``ObsDelay((0, 2))``."""

    def __init__(self, steps: int | tuple[int, int] = 1, groups: Sequence[str] = (POLICY,)) -> None:
        super().__init__(groups)
        self.lo, self.hi = _check_steps(steps, "ObsDelay")
        self.depth = self.hi + 1

    def attach(self, env: Any) -> None:
        super().attach(env)
        self.k = self.stream.steps(self.lo, self.hi, env.num_envs)

    def on_reset(self, mask: Any) -> None:
        self.stream.on_reset(mask)
        new = self.stream.steps(self.lo, self.hi, mask.shape[0])
        self.k = self.xp.where(mask, new, self.k)

    def _read(self, buf: Any) -> Any:
        return _gather(self.xp, buf, self.depth - 1 - self.k)


class ObsHistory(_Buffered):
    """The last ``length`` observations side by side, oldest to newest (width ``D * length``). After a reset every
    slot holds the first observation. ``flatten=False`` returns (W, length, D). Labels: ``layer.names(group, names)``
    or ``realism.obs_names(env)`` (``name@t-2``, ``name@t-1``, ``name@t``).

    Example: ``ObsHistory(3)``."""

    def __init__(self, length: int = 3, groups: Sequence[str] = (POLICY,), flatten: bool = True) -> None:
        super().__init__(groups)
        if int(length) != length or length < 1:
            raise _fail(f"ObsHistory length is {length!r}", "pass a whole number >= 1")
        self.length, self.flatten, self.depth = int(length), flatten, int(length)

    def _read(self, buf: Any) -> Any:
        return buf.reshape(buf.shape[0], -1) if self.flatten else buf

    def names(self, group: str, names: list[str]) -> list[str]:
        if group not in self.groups or not self.flatten:
            return names
        ages = [f"@t-{a}" if a else "@t" for a in range(self.length - 1, -1, -1)]
        return [f"{n}{age}" for age in ages for n in names]


# --- action delay ------------------------------------------------------------------------------------------------

class ActionDelay(Layer):
    """The writer gets the action from ``k`` control steps ago (``k`` or ``(lo, hi)``, drawn per world at every reset).
    Before ``k`` actions exist a world gets zeros ("delta" and "normalized" writers) or its first action (an
    "absolute" writer). A reset clears the world's buffer. ``env.state["action"]`` keeps the action the policy sent.

    Example: ``ActionDelay((0, 2))``."""

    def __init__(self, steps: int | tuple[int, int] = 1) -> None:
        self.lo, self.hi = _check_steps(steps, "ActionDelay")

    def attach(self, env: Any) -> None:
        self.xp, self.stream = env.xp, _Stream(env.xp)
        n, size = env.num_envs, env.writer.size
        self.absolute = getattr(env.writer.spec, "mode", "delta") == "absolute"
        self.k = self.stream.steps(self.lo, self.hi, n)
        self.buf = self.xp.zeros((n, self.hi, size))            # the last hi actions, newest last
        self.first = self.xp.zeros((n, size))
        self.seen = self.xp.zeros(n, "int64")

    def on_action(self, actions: Any) -> Any:
        if self.hi == 0:
            return actions
        xp = self.xp
        a = xp.asarray(actions, "float32")
        self.first = xp.where((self.seen == 0)[:, None], a, self.first)
        full = xp.cat([self.buf, a[:, None]], axis=1)
        out = _gather(xp, full, self.hi - self.k)
        fill = self.first if self.absolute else xp.zeros(out.shape)
        out = xp.where((self.k > self.seen)[:, None], fill, out)
        self.buf, self.seen = full[:, 1:], self.seen + 1
        return out

    def on_reset(self, mask: Any) -> None:
        xp = self.xp
        self.stream.on_reset(mask)
        new = self.stream.steps(self.lo, self.hi, mask.shape[0])
        self.k = xp.where(mask, new, self.k)
        self.buf = xp.where(_bcast(mask, 3), xp.zeros(self.buf.shape), self.buf)
        self.seen = xp.where(mask, xp.zeros(mask.shape[0], "int64"), self.seen)
