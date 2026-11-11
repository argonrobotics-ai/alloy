# SPDX-License-Identifier: Apache-2.0
"""How many worlds a check can run on its device: a per-world memory estimate of the assembled scene against the
device's free memory (SCR-SOFT-3-2: an MPM paste's 64-world drop test ran an L4 out of memory).

Per world: the rigid terms of ``alloy.parallel.costmodel.estimate_bytes_per_world`` (bodies, contacts, constraint
rows, the dense solver block) plus, for each solver entry, its particles times that route's bytes per particle
(``BYTES_PER_PARTICLE``, peak device memory of drop tests measured on an L4, with headroom). The worlds that fit are
``floor((USABLE x free - RESERVE) / per_world)``, at least 1 and at most the worlds asked. CPU devices are not sized.

Example (dough_ball's drop test on an empty L4)::

    >>> fit = fit_worlds(built, 64, "cuda:0", free=23.5e9)         # doctest: +SKIP
    >>> fit.worlds, fit.why                                          # doctest: +SKIP
    (26, 'memory: 26 of the 64 worlds asked fit (about 694 MB per world: mpm 17,256 particles; 24 GB free on cuda:0)')
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("alloy.check")

#: peak device bytes per particle and world by solver capability, from check_object drop tests at 1, 4 and 8 worlds
#: on an L4 (SCR-SOFT-3-2): MPM 39 kB (dough_ball, 17,256 particles: 1.15 / 3.10 / 5.85 GB), 36 kB (honey_box_150ml),
#: 30 kB (sand_box_200ml); the implicit MPM solver's sparse grid, spaces and matrices hold one cell per particle.
#: SPH (water_tumbler_200ml) and VBD (foam_pad_pu, cloth_cotton_sheet) stayed at their 0.1-0.2 GB fixed cost.
BYTES_PER_PARTICLE: dict[str, int] = {"mpm": 40_000, "sph": 1_000, "vbd": 1_000}
DEFAULT_BYTES_PER_PARTICLE = 2_000
RIGID_SCRATCH_BYTES = 4 * 2**20      # MuJoCo Warp in-step scratch per world under a CUDA graph (costmodel note)
USABLE = 0.8                         # share of the free memory a check plans to fill
RESERVE = 512 * 2**20                # shared by all worlds: modules, graphs, buffers (MPM rows: 0.5-0.6 GB at 0 worlds)


@dataclass(frozen=True)
class WorldFit:
    """The world count a check runs with and why: ``asked``, ``worlds`` (what fits, at least 1), the per-world
    estimate and its parts (bytes by capability), the free device memory (None on CPU: not sized).

    Example::

        >>> WorldFit(64, 64, 3_000_000, 2e10, "cuda:0").why
        'asked: 64 worlds fit (about 3 MB per world; 20 GB free on cuda:0)'
    """

    asked: int
    worlds: int
    per_world_bytes: int = 0
    free_bytes: float | None = None
    device: str = "cpu"
    parts: dict[str, int] = field(default_factory=dict)
    particles: dict[str, int] = field(default_factory=dict)

    @property
    def capped(self) -> bool:
        """True when fewer worlds run than were asked.

        Example::

            >>> WorldFit(64, 8, 2e9, 2e10, "cuda:0").capped
            True
        """
        return self.worlds < self.asked

    @property
    def why(self) -> str:
        """One line: how many worlds run and why.

        Example::

            >>> WorldFit(2, 2).why
            'asked: 2 worlds (CPU memory is not sized)'
        """
        if self.free_bytes is None:
            return f"asked: {_worlds(self.asked)} (CPU memory is not sized)"
        if self.capped:
            return f"memory: {self.worlds} of the {self.asked} worlds asked fit ({self.basis})"
        return f"asked: {_worlds(self.asked)} fit ({self.basis})"

    @property
    def basis(self) -> str:
        """The estimate behind the count: bytes per world, its particles, the free memory.

        Example::

            >>> WorldFit(64, 26, 694_000_000, 23.5e9, "cuda:0", particles={"mpm": 17256}).basis
            'about 694 MB per world: mpm 17,256 particles; 24 GB free on cuda:0'
        """
        per = f"about {_size(self.per_world_bytes)} per world{_detail(self.particles)}"
        return f"{per}; {_size(self.free_bytes or 0.0)} free on {self.device}"

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready: asked, worlds, why, per_world_bytes, free_bytes, parts, particles.

        Example::

            >>> WorldFit(2, 2).to_dict()["why"]
            'asked: 2 worlds (CPU memory is not sized)'
        """
        return {"asked": self.asked, "worlds": self.worlds, "why": self.why, "per_world_bytes": self.per_world_bytes,
                "free_bytes": self.free_bytes, "parts": dict(self.parts), "particles": dict(self.particles)}


def free_bytes(device: str) -> float | None:
    """Free memory of a CUDA device in bytes; None for the CPU.

    Example::

        >>> free_bytes("cpu") is None                                  # doctest: +SKIP
        True
    """
    import warp as wp  # noqa: PLC0415

    dev = wp.get_device(device)
    return float(dev.free_memory) if dev.is_cuda else None


def per_world(built: Any) -> tuple[dict[str, int], dict[str, int]]:
    """(bytes by part, particles by capability) of one world of ``built`` (a ``BuiltScene``).

    Example::

        >>> parts, particles = per_world(built)                         # doctest: +SKIP
        >>> sorted(parts), particles                                    # doctest: +SKIP
        (['mpm', 'rigid'], {'mpm': 17256})
    """
    from alloy.parallel.costmodel import MemoryTerms, estimate_bytes_per_world  # noqa: PLC0415

    m, w = built.model, max(1, int(built.world_count))
    shapes = math.ceil(int(m.shape_count) / w)
    contacts = 8 * shapes
    terms = MemoryTerms(nv=math.ceil(int(m.joint_dof_count) / w), rows_max=4 * contacts, contacts_max=contacts,
                        pairs_max=shapes**2, n_bodies=math.ceil(int(m.body_count) / w))
    parts: dict[str, int] = {"rigid": estimate_bytes_per_world(terms) + RIGID_SCRATCH_BYTES}
    particles: dict[str, int] = defaultdict(int)
    for e in built.entries:
        n = len(getattr(e, "particles", None) or ())
        if n:
            particles[e.capability] += math.ceil(n / w)
    for cap, n in particles.items():
        parts[cap] = n * BYTES_PER_PARTICLE.get(cap, DEFAULT_BYTES_PER_PARTICLE)
    return parts, dict(particles)


def fit_worlds(built: Any, asked: int, device: str, free: float | None = None) -> WorldFit:
    """The worlds of ``built`` that fit ``device`` (``free``: its free bytes, read now when None), at most ``asked``;
    26 for dough_ball's drop test on an empty L4.

    Example::

        >>> fit_worlds(built, 64, "cuda:0", free=23.5e9).worlds            # doctest: +SKIP
        26
    """
    free = free_bytes(device) if free is None else free
    if free is None:
        return WorldFit(asked, asked)
    parts, particles = per_world(built)
    need = max(1, sum(parts.values()))
    fits = int(math.floor((USABLE * free - RESERVE) / need))
    fit = WorldFit(asked, max(1, min(asked, fits)), need, float(free), device, parts, particles)
    (log.warning if fit.capped else log.info)("check: %s", fit.why)
    return fit


def _worlds(n: int) -> str:
    return f"{n} world" if n == 1 else f"{n} worlds"


def _size(b: float) -> str:
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if b >= scale:
            return f"{b / scale:.2g} {unit}" if b / scale < 10 else f"{b / scale:.0f} {unit}"
    return f"{b:.0f} B"


def _detail(particles: dict[str, int]) -> str:
    if not particles:
        return ""
    return ": " + ", ".join(f"{cap} {n:,} particles" for cap, n in sorted(particles.items()))


__all__ = ["BYTES_PER_PARTICLE", "WorldFit", "fit_worlds", "free_bytes", "per_world"]
