# SPDX-License-Identifier: Apache-2.0
"""DFSPH liquid plug-in (WP9): ``SolverDFSPH`` (divergence-free SPH, SDF tube-map walls, embedded proxy bodies).

Importing this package registers the ``sph`` solver capability (route ``liquid_sph``) and the bully B0 target
``sph_pour`` (``pour_target.py``). Types live in
``alloy.solvers.sph.types`` (WP0); scene helpers in ``alloy.solvers.sph.build`` and ``alloy.testing.sph``;
laws A13/A14 in ``alloy.solvers.sph.laws``. Design and measured numbers: docs/notes/wp9-*.md.
"""

from alloy.solvers.sph.config import DFSPHConfig
from alloy.solvers.sph.solver import SolverDFSPH
from alloy.solvers.sph import plugin as _plugin  # noqa: F401  (registers the capability)
from alloy.solvers.sph import pour_target as _pour_target

_pour_target.register()

__all__ = ["DFSPHConfig", "SolverDFSPH"]
