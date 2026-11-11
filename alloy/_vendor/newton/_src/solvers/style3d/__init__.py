# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# Vendored in Alloy: alloy/_vendor/VENDORED.md lists the changes made to upstream, imports included.

"""
Style3D solver module.

This module provides helper functions for setting up Style3D cloth assets.
Use :class:`~newton.solvers.SolverStyle3D` as the canonical public solver class.
"""

from .cloth import (
    add_cloth_grid,
    add_cloth_mesh,
)

__all__ = [
    "add_cloth_grid",
    "add_cloth_mesh",
]
