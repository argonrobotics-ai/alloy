# SPDX-License-Identifier: Apache-2.0
"""Array backends for ``alloy.envs``: PyTorch tensors or NumPy arrays over the engine's Warp arrays.

``torch``: zero-copy views of the engine's arrays (``wp.to_torch``) on the simulation device; observations, rewards
and actions stay on the GPU. ``numpy``: views on CPU, host copies on a GPU (slower; for machines without PyTorch).
Both expose the same small set of operations, so task checkers are written once. Example::

    xp = pick_backend("auto", "cuda:0")          # torch when it is installed
    body_q = xp.view(sim.engine.state.body_q)    # (bodies, 7) tensor, shares memory with the engine
    d = xp.norm(body_q[:, :3], axis=-1)
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import warp as wp

from alloy.envs.errors import EnvError, install_fix

BACKENDS = ("auto", "torch", "numpy")


def has_torch() -> bool:
    """True when PyTorch can be imported. Example: ``has_torch()``."""
    return importlib.util.find_spec("torch") is not None


def pick_backend(name: str, device: str) -> NumpyBackend | TorchBackend:
    """The backend for ``name`` ("auto" = torch when installed, else numpy) on ``device``.

    Example: ``pick_backend("numpy", "cpu").zeros((4,), "float32")``."""
    if name not in BACKENDS:
        raise EnvError(f"unknown backend {name!r}", f"pass one of {BACKENDS}", field="backend")
    if name == "auto":
        name = "torch" if has_torch() else "numpy"
    if name == "torch":
        if not has_torch():
            raise EnvError("backend='torch' but PyTorch is not installed",
                           f"{install_fix('torch')}, or pass backend='numpy'", field="backend")
        return TorchBackend(device)
    return NumpyBackend(device)


class NumpyBackend:
    """NumPy arrays. On CPU the views share memory with the engine; on a GPU every view is a host copy."""

    name = "numpy"

    def __init__(self, device: str) -> None:
        self.device = device
        self.shared = device == "cpu"

    # --- engine arrays ---
    def view(self, arr: wp.array) -> np.ndarray:
        """The Warp array as NumPy (vector dtypes get a trailing axis). Example: ``xp.view(state.body_q)``."""
        return arr.numpy()

    def update(self, arr: wp.array, fn: Callable[[np.ndarray], None]) -> None:
        """Change ``arr`` in place through ``fn(view)``. Example: ``xp.update(a, lambda v: v.fill(0))``."""
        host = arr.numpy()
        fn(host)
        if not self.shared:
            arr.assign(host)

    def to_warp_mask(self, mask: np.ndarray) -> wp.array:
        """A bool (W,) mask as the engine's int32 world mask. Example: ``xp.to_warp_mask(done)``."""
        return wp.array(np.asarray(mask, np.int32), dtype=wp.int32, device=self.device)

    # --- construction ---
    def asarray(self, values: Any, dtype: str = "float32") -> np.ndarray:
        """Values as an array of ``dtype`` ("float32", "int64", "bool"). Example: ``xp.asarray([1, 2], "int64")``."""
        return np.asarray(values, dtype=np.dtype(dtype))

    def zeros(self, shape: Sequence[int] | int, dtype: str = "float32") -> np.ndarray:
        """Zeros of ``dtype``. Example: ``xp.zeros((4, 3))``."""
        return np.zeros(shape, dtype=np.dtype(dtype))

    def to_numpy(self, x: Any) -> np.ndarray:
        """A host NumPy copy. Example: ``xp.to_numpy(reward)``."""
        return np.array(x)

    def seed(self, seed: int | None) -> None:
        """Seed ``uniform``. Example: ``xp.seed(0)``."""
        self._rng = np.random.default_rng(seed)

    def uniform(self, shape: Sequence[int], lo: Any, hi: Any) -> np.ndarray:
        """float32 uniform samples in [lo, hi) (bounds broadcast). Example: ``xp.uniform((4, 3), -1, 1)``."""
        rng = getattr(self, "_rng", None) or np.random.default_rng()
        self._rng = rng
        return (lo + (np.asarray(hi) - lo) * rng.random(tuple(shape))).astype(np.float32)

    def to_warp_bool(self, mask: Any) -> wp.array:
        """A bool array as a Warp bool array on the device. Example: ``xp.to_warp_bool(m)``."""
        return wp.array(np.asarray(mask, bool), dtype=wp.bool, device=self.device)

    # --- math ---
    def float(self, x: Any) -> np.ndarray:
        """As float32. Example: ``xp.float(done)``."""
        return np.asarray(x, dtype=np.float32)

    def cat(self, xs: Sequence[Any], axis: int = -1) -> np.ndarray:
        """Concatenate along ``axis``. Example: ``xp.cat([a, b])``."""
        return np.concatenate(list(xs), axis=axis)

    def stack(self, xs: Sequence[Any], axis: int = 0) -> np.ndarray:
        """Stack along a new ``axis``. Example: ``xp.stack([a, b], -1)``."""
        return np.stack(list(xs), axis=axis)

    def norm(self, x: Any, axis: int = -1) -> np.ndarray:
        """Euclidean norm over ``axis``. Example: ``xp.norm(p - q)``."""
        return np.linalg.norm(x, axis=axis)

    def where(self, cond: Any, a: Any, b: Any) -> np.ndarray:
        """``a`` where ``cond`` else ``b``. Example: ``xp.where(done, 0.0, r)``."""
        return np.where(cond, a, b)

    def clip(self, x: Any, lo: Any, hi: Any) -> np.ndarray:
        """``x`` clipped to [lo, hi] (arrays broadcast). Example: ``xp.clip(a, -1, 1)``."""
        return np.minimum(np.maximum(x, lo), hi)

    def amin(self, x: Any, axis: int = -1) -> np.ndarray:
        """Minimum over ``axis``. Example: ``xp.amin(z, -1)``."""
        return np.min(x, axis=axis)

    def tanh(self, x: Any) -> np.ndarray:
        """Hyperbolic tangent. Example: ``1 - xp.tanh(d / 0.1)``."""
        return np.tanh(x)

    def amax(self, x: Any, axis: int = -1) -> np.ndarray:
        """Maximum over ``axis``. Example: ``xp.amax(z, -1)``."""
        return np.max(x, axis=axis)

    def any(self, x: Any) -> bool:
        """True if any element is true (a host sync on a GPU). Example: ``xp.any(done)``."""
        return bool(np.any(x))

    def copy(self, x: Any) -> np.ndarray:
        """A copy that later engine steps do not change. Example: ``xp.copy(obs)``."""
        return np.array(x, copy=True)


class TorchBackend:
    """PyTorch tensors on the simulation device, sharing memory with the engine's Warp arrays."""

    name = "torch"
    shared = True

    def __init__(self, device: str) -> None:
        import torch  # noqa: PLC0415 - optional dependency

        self.torch, self.device = torch, device
        self.tdevice = torch.device("cuda", wp.get_device(device).ordinal) if device.startswith("cuda") else "cpu"
        self._dtypes = {"float32": torch.float32, "int64": torch.int64, "int32": torch.int32, "bool": torch.bool}

    # --- engine arrays ---
    def view(self, arr: wp.array) -> Any:
        """The Warp array as a tensor sharing its memory. Example: ``xp.view(state.body_q)`` -> (bodies, 7)."""
        return wp.to_torch(arr, requires_grad=False)

    def update(self, arr: wp.array, fn: Callable[[Any], None]) -> None:
        """Change ``arr`` in place through ``fn(view)``. Example: ``xp.update(a, lambda v: v.zero_())``."""
        fn(self.view(arr))

    def to_warp_mask(self, mask: Any) -> wp.array:
        """A bool (W,) tensor as the engine's int32 world mask. Example: ``xp.to_warp_mask(done)``."""
        return wp.from_torch(mask.to(self.torch.int32).contiguous(), dtype=wp.int32)

    # --- construction ---
    def asarray(self, values: Any, dtype: str = "float32") -> Any:
        """Values as a tensor of ``dtype`` on the device. Example: ``xp.asarray([1, 2], "int64")``."""
        t = self.torch
        if isinstance(values, t.Tensor):
            return values.to(device=self.tdevice, dtype=self._dtypes[dtype])
        return t.as_tensor(np.asarray(values), device=self.tdevice).to(self._dtypes[dtype])

    def zeros(self, shape: Sequence[int] | int, dtype: str = "float32") -> Any:
        """Zeros of ``dtype``. Example: ``xp.zeros((4, 3))``."""
        return self.torch.zeros(shape, dtype=self._dtypes[dtype], device=self.tdevice)

    def to_numpy(self, x: Any) -> np.ndarray:
        """A host NumPy copy. Example: ``xp.to_numpy(reward)``."""
        return x.detach().cpu().numpy() if isinstance(x, self.torch.Tensor) else np.array(x)

    def seed(self, seed: int | None) -> None:
        """Seed ``uniform``. Example: ``xp.seed(0)``."""
        self._gen = self.torch.Generator(device=self.tdevice)
        if seed is None:
            self._gen.seed()
        else:
            self._gen.manual_seed(int(seed))

    def uniform(self, shape: Sequence[int], lo: Any, hi: Any) -> Any:
        """float32 uniform samples in [lo, hi) on the device (bounds broadcast). Example: ``xp.uniform((4,), 0, 1)``."""
        if getattr(self, "_gen", None) is None:
            self.seed(None)
        u = self.torch.rand(tuple(shape), generator=self._gen, device=self.tdevice)
        return lo + (hi - lo) * u

    def to_warp_bool(self, mask: Any) -> wp.array:
        """A bool tensor as a Warp bool array (zero-copy). Example: ``xp.to_warp_bool(m)``."""
        return wp.from_torch(mask.to(self.torch.bool).contiguous(), dtype=wp.bool)

    # --- math ---
    def float(self, x: Any) -> Any:
        """As float32. Example: ``xp.float(done)``."""
        return x.to(self.torch.float32)

    def cat(self, xs: Sequence[Any], axis: int = -1) -> Any:
        """Concatenate along ``axis``. Example: ``xp.cat([a, b])``."""
        return self.torch.cat(list(xs), dim=axis)

    def stack(self, xs: Sequence[Any], axis: int = 0) -> Any:
        """Stack along a new ``axis``. Example: ``xp.stack([a, b], -1)``."""
        return self.torch.stack(list(xs), dim=axis)

    def norm(self, x: Any, axis: int = -1) -> Any:
        """Euclidean norm over ``axis``. Example: ``xp.norm(p - q)``."""
        return self.torch.linalg.vector_norm(x, dim=axis)

    def where(self, cond: Any, a: Any, b: Any) -> Any:
        """``a`` where ``cond`` else ``b``. Example: ``xp.where(done, 0.0, r)``."""
        t = self.torch
        a = a if isinstance(a, t.Tensor) else t.as_tensor(a, device=self.tdevice)
        b = b if isinstance(b, t.Tensor) else t.as_tensor(b, device=self.tdevice)
        return t.where(cond, a, b)

    def clip(self, x: Any, lo: Any, hi: Any) -> Any:
        """``x`` clipped to [lo, hi] (tensors broadcast). Example: ``xp.clip(a, -1, 1)``."""
        t = self.torch
        lo = lo if isinstance(lo, t.Tensor) else t.as_tensor(lo, dtype=x.dtype, device=x.device)
        hi = hi if isinstance(hi, t.Tensor) else t.as_tensor(hi, dtype=x.dtype, device=x.device)
        return t.minimum(t.maximum(x, lo), hi)

    def amin(self, x: Any, axis: int = -1) -> Any:
        """Minimum over ``axis``. Example: ``xp.amin(z, -1)``."""
        return self.torch.amin(x, dim=axis)

    def tanh(self, x: Any) -> Any:
        """Hyperbolic tangent. Example: ``1 - xp.tanh(d / 0.1)``."""
        return self.torch.tanh(x)

    def amax(self, x: Any, axis: int = -1) -> Any:
        """Maximum over ``axis``. Example: ``xp.amax(z, -1)``."""
        return self.torch.amax(x, dim=axis)

    def any(self, x: Any) -> bool:
        """True if any element is true (a host sync). Example: ``xp.any(done)``."""
        return bool(x.any().item())

    def copy(self, x: Any) -> Any:
        """A copy that later engine steps do not change. Example: ``xp.copy(obs)``."""
        return x.clone()


__all__ = ["BACKENDS", "NumpyBackend", "TorchBackend", "has_torch", "pick_backend"]
