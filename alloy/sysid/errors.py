# SPDX-License-Identifier: Apache-2.0
"""Errors of ``alloy.sysid``: each says what went wrong and what to change, like every Alloy API error."""

from __future__ import annotations

from alloy.api.errors import FeatureUnavailableError, SceneError


class SysidError(SceneError):
    """A recording, a parameter choice or a sysid file is not usable.

    Example: ``SysidError("the recording has no joint 'elbow'", "name the columns cmd.elbow and pos.elbow")``."""


def need(module: str, command: str, what: str) -> FeatureUnavailableError:
    """The error for an optional package that is not installed.

    Example: ``raise need("h5py", "pip install h5py", "reading an HDF5 recording")``."""
    return FeatureUnavailableError(f"{what} needs the `{module}` package, which is not installed",
                                   f"run `{command}`", field=module)


__all__ = ["SysidError", "need"]
