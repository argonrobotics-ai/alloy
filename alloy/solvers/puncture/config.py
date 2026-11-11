# SPDX-License-Identifier: Apache-2.0
"""Puncture configuration: needle specs, model parameters and the per-tet ``puncture:*`` attributes (SC11).

Per-tet material (custom attributes, namespace ``puncture``, frequency TETRAHEDRON; default 0 = off):
- ``puncture:fp``    puncture threshold of a surface face owned by this tet [N] (> 0 makes the tet's faces puncturable);
- ``puncture:fcut``  cutting force at the tip while the tip advances through this tet [N];
- ``puncture:flen``  needle-tissue friction per unit length [N/m];
- ``puncture:eta``   viscous needle-tissue friction per unit length [N·s/m²].
A model where every tet has fp = 0 and no needle is given steps exactly as SolverVBDMomentum (default off).

Needle centreline (body frame), tip at s = 0, swage at s = length:
- arc (radius > 0):  P(s) = origin + R (cos(s/R) e1 + sin(s/R) e2); e1 points from the arc centre to the tip;
- line (radius = 0): P(s) = origin + s e1; origin is the tip, e1 points to the swage.
The tip advances along -dP/ds at s = 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from alloy._vendor import newton

NAMESPACE = "puncture"
FP, FCUT, FLEN, ETA = (f"{NAMESPACE}:{n}" for n in ("fp", "fcut", "flen", "eta"))


@dataclass(frozen=True)
class NeedleSpec:
    """One rigid needle (module doc for the centreline). ``body`` is the Newton body that carries it (its pose comes
    from body_q; -1 = fixed in the world frame). The thread, if any, is particles [thread_start, thread_start +
    thread_count) with thread_start at the swage (tied to it by the solver), evenly spaced at rest, with no Newton
    springs between them: stretch (thread_ea) and bending (thread_ei) are the chain solve's (chain.py)."""

    body: int
    length: float
    wire_radius: float
    origin: tuple[float, float, float]
    e1: tuple[float, float, float]
    e2: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 0.0
    world: int = 0
    thread_start: int = -1
    thread_count: int = 0
    stopper: bool = False
    thread_flen: float = 0.0          # thread-tissue friction per unit length [N/m]
    thread_eta: float = 0.0           # viscous thread-tissue friction per unit length [N·s/m²]
    thread_ei: float = 0.0            # thread bending stiffness EI [N·m²] (chain.py)
    thread_ea: float = 0.0            # thread stretch stiffness EA [N] (chain.py; no Newton springs on the thread)

    def __post_init__(self) -> None:
        if self.length <= 0.0 or self.wire_radius <= 0.0:
            raise ValueError(f"NeedleSpec: length and wire_radius must be > 0, got {self.length}, {self.wire_radius}")
        if self.radius < 0.0:
            raise ValueError(f"NeedleSpec.radius must be >= 0 (0 = straight), got {self.radius}")
        if self.thread_count == 1:
            raise ValueError("NeedleSpec: a thread needs at least 2 particles")


@dataclass(frozen=True)
class PunctureConfig:
    """Model parameters (not material claims; design note §2, §7)."""

    k_n: float = 2.0e3               # lateral anchor stiffness [N/m]
    k_tip: float = 2.0e4             # tip stick stiffness before puncture [N/m]
    h_a: float = 0.5e-3              # anchor spacing along the track [m]
    eps_v: float = 1.0e-4            # friction smoothing speed (IPC eps_v) [m/s]
    lookahead: float = 0.5e-3        # surface faces this far ahead of the tip get a (not yet pressing) TIP contact [m]
    mu: float = 0.0                  # extra Coulomb friction on the lateral anchor force
    anchors_per_world: int = 256
    ledger: bool = True

    def __post_init__(self) -> None:
        if min(self.k_n, self.k_tip, self.h_a, self.eps_v, self.lookahead) <= 0.0:
            raise ValueError("PunctureConfig: k_n, k_tip, h_a, eps_v and lookahead must be > 0")
        if self.mu < 0.0 or self.anchors_per_world < 1:
            raise ValueError("PunctureConfig: mu must be >= 0 and anchors_per_world >= 1")


def _attr(name: str):
    return newton.ModelBuilder.CustomAttribute(name=name, dtype=wp.float32,
                                              frequency=newton.Model.AttributeFrequency.TETRAHEDRON,
                                              namespace=NAMESPACE, default=0.0)


def register_custom_attributes(builder: newton.ModelBuilder) -> None:
    """``model.puncture.{fp, fcut, flen, eta}`` per tet (module doc). Idempotent."""
    for n in ("fp", "fcut", "flen", "eta"):
        builder.add_custom_attribute(_attr(n))


@dataclass(frozen=True)
class TissueMaterial:
    """The four per-tet values for one material layer (module doc)."""

    fp: float
    fcut: float
    flen: float
    eta: float = 0.0

    def attributes(self) -> dict[str, float]:
        return {FP: self.fp, FCUT: self.fcut, FLEN: self.flen, ETA: self.eta}


def set_tet_material(builder: newton.ModelBuilder, tets: list[int], mat: TissueMaterial) -> None:
    """Assign ``mat`` to builder tets ``tets`` (e.g. those an ``add_soft_mesh`` call just added)."""
    register_custom_attributes(builder)
    vals = mat.attributes()
    builder._process_custom_attributes(list(tets), {k: [v] * len(tets) for k, v in vals.items()},
                                       newton.Model.AttributeFrequency.TETRAHEDRON)
