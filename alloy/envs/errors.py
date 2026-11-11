# SPDX-License-Identifier: Apache-2.0
"""Errors of ``alloy.envs``. Like every Alloy API error, each says what went wrong and what to change.

Example::

    from alloy.envs import EnvError
    try:
        env.step(actions_of_the_wrong_shape)
    except EnvError as e:
        print(e.message, "->", e.fix)
"""

from __future__ import annotations

from alloy.api.errors import AlloyAPIError


class EnvError(AlloyAPIError, ValueError):
    """A Gym environment was set up or used wrongly (unknown name, action shape, missing package).

    Example: ``EnvError("actions have shape (3,)", "pass (num_envs, 3)", field="actions").to_dict()``."""


NEEDS = {"torch": "pip install torch", "huggingface_hub": "pip install 'alloy-engine[import]'"}  # pip only


def has_extra(name: str) -> bool:
    """True when the installed alloy-engine declares the pip extra ``name`` (``alloy-engine[name]``)."""
    from importlib import metadata  # noqa: PLC0415

    try:
        return name in (metadata.metadata("alloy-engine").get_all("Provides-Extra") or [])
    except metadata.PackageNotFoundError:
        return False


def install_fix(package: str) -> str:
    """How to install ``package`` for the environments ("torch" or "huggingface_hub"): the ``envs`` extra, which has
    both, when this install declares it, else the package's own command.

    Example: ``install_fix("torch")`` -> "pip install 'alloy-engine[envs]'"."""
    return "pip install 'alloy-engine[envs]'" if has_extra("envs") else NEEDS[package]


__all__ = ["EnvError", "has_extra", "install_fix"]
