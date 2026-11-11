# SPDX-License-Identifier: Apache-2.0
"""Fit parameters: leaf names, the normalised search box, priors, and card leaf access (SPEC §3.12).

A leaf is ``"<kind>:<name>.<path>"``: ``material:pp.surface.mu_static`` (a MaterialCard path in the WP1 grammar,
`alloy.cards.paths`) or ``pair:pp__steel.mu_dynamic`` (a PairCard field; names sorted, `library.pair_key`). The
twin realises a leaf by its last path segment (`LeafRef.field`). The search runs in the unit box: each parameter
maps [lo, hi] (linear) or [ln lo, ln hi] (log) onto [0, 1]. A prior (mean, sigma) is given in scale space (x or
ln x) and adds z^2 = ((s - mean) / sigma)^2 to the objective.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from alloy.cards import paths
from alloy.cards import types as ct
from alloy.cards.library import pair_key
from alloy.core.quantity import Q
from alloy.fit.errors import FitError
from alloy.fit.types import FitParam

LeafKindName = Literal["material", "pair"]
_PAIR_FIELDS = frozenset({"mu_static", "mu_dynamic", "restitution", "torsional_lever"})
INIT_SIGMA = 0.3          # unit-box step size without a prior or start (CMA-ES sigma0, fraction of the range)
START_SIGMA = 0.05        # unit-box step size around a closed-form start estimate (CSA widens it if needed)
MIN_INIT_SIGMA = 0.02


@dataclass(frozen=True)
class LeafRef:
    """A parsed leaf: which card (kind, name) and which numeric leaf on it (path, field = last segment)."""

    leaf: str
    kind: LeafKindName
    name: str
    path: str

    @property
    def field(self) -> str:
        return self.path.rsplit(".", 1)[-1]

    @property
    def materials(self) -> tuple[str, ...]:
        return tuple(self.name.split("__")) if self.kind == "pair" else (self.name,)


def parse_leaf(leaf: str) -> LeafRef:
    """Parse and check a leaf against the card types; raises FitError naming the leaf."""
    kind, sep, rest = leaf.partition(":")
    name, dot, path = rest.partition(".")
    if not sep or not dot or not name or not path:
        raise FitError(f"leaf {leaf!r}: expected '<material|pair>:<name>.<path>'")
    if kind == "material":
        try:
            paths.kind_for_path(f"material.{path}")
        except paths.PathError as exc:
            raise FitError(f"leaf {leaf!r}: {exc.message}") from exc
        return LeafRef(leaf, "material", name, path)
    if kind == "pair":
        a, sep2, b = name.partition("__")
        if not sep2 or not a or not b or "__" in b:
            raise FitError(f"leaf {leaf!r}: pair name must be '<a>__<b>'")
        if name != pair_key(a, b):
            raise FitError(f"leaf {leaf!r}: pair names are sorted, write {pair_key(a, b)!r}")
        if path not in _PAIR_FIELDS:
            raise FitError(f"leaf {leaf!r}: PairCard has no numeric field {path!r} (fields: {sorted(_PAIR_FIELDS)})")
        return LeafRef(leaf, "pair", name, path)
    raise FitError(f"leaf {leaf!r}: unknown kind {kind!r} (material | pair)")


def _check_param(p: FitParam) -> None:
    if not (math.isfinite(p.lo) and math.isfinite(p.hi)) or p.lo >= p.hi:
        raise FitError(f"{p.leaf}: bounds must be finite with lo < hi, got ({p.lo}, {p.hi})")
    if p.scale == "log" and p.lo <= 0.0:
        raise FitError(f"{p.leaf}: log scale needs lo > 0, got {p.lo}")
    if p.scale not in ("linear", "log"):
        raise FitError(f"{p.leaf}: unknown scale {p.scale!r}")
    if p.prior is not None and not (p.prior[1] > 0.0 and math.isfinite(p.prior[0])):
        raise FitError(f"{p.leaf}: prior must be (mean, sigma > 0) in scale space, got {p.prior}")


class ParamSpace:
    """The normalised search box over a tuple of FitParams (rows = candidates, columns = parameters)."""

    def __init__(self, params: Sequence[FitParam]) -> None:
        if not params:
            raise FitError("a fit needs at least one FitParam")
        for p in params:
            _check_param(p)
        leaves = [p.leaf for p in params]
        if len(set(leaves)) != len(leaves):
            raise FitError(f"duplicate leaves in {leaves}")
        self.params = tuple(params)
        self.refs = tuple(parse_leaf(p.leaf) for p in params)
        self._log = np.array([p.scale == "log" for p in params])
        self._lo = np.array([_to_scale(p.lo, p.scale) for p in params])
        self._hi = np.array([_to_scale(p.hi, p.scale) for p in params])

    @property
    def n(self) -> int:
        return len(self.params)

    @property
    def leaves(self) -> tuple[str, ...]:
        return tuple(p.leaf for p in self.params)

    def scale_of(self, x: np.ndarray) -> np.ndarray:
        """SI values -> scale space (x or ln x per column)."""
        x = np.asarray(x, np.float64)
        return np.where(self._log, np.log(np.maximum(x, 1e-300)), x)

    def to_unit(self, x: np.ndarray) -> np.ndarray:
        return (self.scale_of(x) - self._lo) / (self._hi - self._lo)

    def from_unit(self, u: np.ndarray) -> np.ndarray:
        """Unit box -> SI (clipped to the box first, so a value is always inside [lo, hi])."""
        s = self._lo + np.clip(np.asarray(u, np.float64), 0.0, 1.0) * (self._hi - self._lo)
        x = np.where(self._log, np.exp(s), s)
        lo = np.array([p.lo for p in self.params])
        hi = np.array([p.hi for p in self.params])
        return np.clip(x, lo, hi)

    def prior_penalty(self, x: np.ndarray) -> np.ndarray:
        """Sum of prior z^2 per candidate row (0 for parameters without a prior)."""
        s = self.scale_of(np.atleast_2d(x))
        out = np.zeros(s.shape[0])
        for j, p in enumerate(self.params):
            if p.prior is not None:
                out += ((s[:, j] - p.prior[0]) / p.prior[1]) ** 2
        return out

    def initial(self, start: Mapping[str, float] | None = None) -> tuple[np.ndarray, float]:
        """CMA-ES start in the unit box and one step size.

        Per parameter: the prior mean if given, else `start[leaf]` (an SI estimate, e.g. a twin's closed form from
        calib means), else the box centre. sigma: the smallest prior sigma (unit box) if any prior, else
        START_SIGMA when any start value was used, else INIT_SIGMA."""
        start = start or {}
        mean = np.full(self.n, 0.5)
        sig, used_start = [], False
        for j, p in enumerate(self.params):
            span = self._hi[j] - self._lo[j]
            if p.prior is not None:
                mean[j] = float(np.clip((p.prior[0] - self._lo[j]) / span, 0.0, 1.0))
                sig.append(p.prior[1] / span)
            elif p.leaf in start and math.isfinite(start[p.leaf]):
                x = float(np.clip(start[p.leaf], p.lo, p.hi))
                mean[j] = float((_to_scale(x, p.scale) - self._lo[j]) / span)
                used_start = True
        if sig:
            return mean, float(np.clip(min(sig), MIN_INIT_SIGMA, INIT_SIGMA))
        return mean, START_SIGMA if used_start else INIT_SIGMA


def _to_scale(v: float, scale: str) -> float:
    return math.log(v) if scale == "log" else v


# -- card leaves -----------------------------------------------------------------------------------------------------
def get_leaf(card: Any, path: str) -> Q | None:
    """The Q at `path` on a MaterialCard/PairCard (None for an unset optional leaf)."""
    node = card
    for seg in paths.split(path):
        if node is None:
            return None
        if not dataclasses.is_dataclass(node) or seg not in {f.name for f in dataclasses.fields(node)}:
            raise FitError(f"{type(card).__name__}: no leaf {path!r}")
        node = getattr(node, seg)
    if node is not None and not isinstance(node, Q):
        raise FitError(f"{type(card).__name__}.{path} is not a quantity leaf ({type(node).__name__})")
    return node


def set_leaf(card: Any, path: str, q: Q) -> Any:
    """A copy of `card` with the Q leaf at `path` replaced by `q` (an unset optional leaf is created)."""
    seg, _, rest = path.partition(".")
    if not dataclasses.is_dataclass(card) or seg not in {f.name for f in dataclasses.fields(card)}:
        raise FitError(f"{type(card).__name__}: no leaf {path!r}")
    if not rest:
        return dataclasses.replace(card, **{seg: q})
    child = getattr(card, seg)
    if child is None:
        raise FitError(f"{type(card).__name__}.{seg} is unset; cannot set {path!r} below it")
    return dataclasses.replace(card, **{seg: set_leaf(child, rest, q)})


def new_pair_card(a: ct.MaterialCard, b: ct.MaterialCard) -> ct.PairCard:
    """A PairCard seeded from two materials with Drake's harmonic rule (D13), source "model", for fitted leaves."""
    names = tuple(sorted((a.name, b.name)))
    note = f"harmonic 2ab/(a+b) of {names[0]} and {names[1]} surface values (D13), seeded for a fit"

    def harmonic(x: Q, y: Q) -> Q:
        s = x.value + y.value
        return Q(2.0 * x.value * y.value / s if s > 0 else 0.0, "model", note=note)

    sa, sb = a.surface, b.surface
    return ct.PairCard(names[0], names[1], harmonic(sa.mu_static, sb.mu_static),
                       harmonic(sa.mu_dynamic, sb.mu_dynamic), restitution=harmonic(sa.restitution, sb.restitution))


__all__ = ["LeafRef", "parse_leaf", "ParamSpace", "get_leaf", "set_leaf", "new_pair_card"]
