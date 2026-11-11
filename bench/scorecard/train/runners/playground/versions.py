# SPDX-License-Identifier: Apache-2.0
"""Versions of the stack that produced a run (for the RunHeader)."""

from importlib import metadata

PACKAGES = ("playground", "mujoco", "mujoco-mjx", "warp-lang", "jax", "jaxlib", "brax", "rsl-rl-lib", "torch", "numpy")


def collect() -> dict[str, str]:
    out = {}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = "missing"
    return out
