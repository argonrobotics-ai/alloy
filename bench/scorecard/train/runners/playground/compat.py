# SPDX-License-Identifier: Apache-2.0
"""Process-local shims for installed packages that call APIs the pinned JAX removed (nothing installed is edited).

- playground 0.2.0 LeapCubeReorient calls ``jnp.clip(x, a_max=...)``; jax 0.12 only knows ``min`` / ``max``.
- brax 0.14.2 calls ``jax.device_put_replicated`` (removed in jax 0.12).
"""

import jax
import jax.numpy as jnp

_orig_clip = jnp.clip


def _clip(a, *args, a_min=None, a_max=None, **kw):
    if a_min is not None:
        kw["min"] = a_min
    if a_max is not None:
        kw["max"] = a_max
    return _orig_clip(a, *args, **kw)


def device_put_replicated(tree, devices):
    return jax.tree.map(lambda a: jax.device_put(jnp.stack([jnp.asarray(a)] * len(devices)), devices[0]), tree)


def apply() -> None:
    jnp.clip = _clip
    if not hasattr(jax, "device_put_replicated") or True:
        jax.device_put_replicated = device_put_replicated
