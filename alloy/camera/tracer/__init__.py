# SPDX-License-Identifier: Apache-2.0
"""Warp path-tracing kernels (Cycles 4.5 conventions): sampling, geometry, lights, integrators, renderers.

The megakernel modules compile with fast_math (about 8 % faster; images change by about 1e-6 relative).
"""
import warp as _wp

from . import integrator as _integrator

_wp.set_module_options({"fast_math": True}, _integrator)
