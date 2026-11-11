# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# Vendored in Alloy: alloy/_vendor/VENDORED.md lists the changes made to upstream, imports included.

"""Experimental coupled-solver framework.

.. experimental::
"""

from .interface import CouplingInterface
from .model_view import ModelView
from .solver_coupled import SolverCoupled
from .solver_coupled_admm import SolverCoupledADMM
from .solver_coupled_proxy import SolverCoupledProxy

__all__ = [
    "CouplingInterface",
    "ModelView",
    "SolverCoupled",
    "SolverCoupledADMM",
    "SolverCoupledProxy",
]
