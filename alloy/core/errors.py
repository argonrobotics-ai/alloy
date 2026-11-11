# SPDX-License-Identifier: Apache-2.0
"""Alloy error classes (SPEC §3.1). Every library error is an ``AlloyError``.

Errors that replace a builtin a caller may already catch also derive from it (``UnitError`` is a ``ValueError``,
``RegistryError`` a ``LookupError``), so ``except ValueError`` keeps working.
"""

from __future__ import annotations


class AlloyError(Exception):
    """Base class of every Alloy error."""


class UnitError(AlloyError, ValueError):
    """A unit string is unknown, malformed, non-finite or of the wrong kind (``parse_si``)."""


class QuantityError(AlloyError, ValueError):
    """A ``Q`` breaks its own rules (non-finite value, missing note, bad uncertainty interval)."""


class CardError(AlloyError):
    """A card value is invalid. ``.card`` names the card, ``.field`` the dotted path of the offending leaf."""

    def __init__(self, card: str, field: str, message: str) -> None:
        super().__init__(f"{card}: {field}: {message}")
        self.card = card
        self.field = field
        self.message = message


class RouteError(AlloyError):
    """No valid route for an object (e.g. ROD_VBD before its self-test passes, D8)."""


class AssemblyError(AlloyError):
    """The scene assembler cannot build the Model."""


class ReadbackError(AlloyError):
    """Compiled Model values differ from the cards. ``.mismatches`` = [(card field, card value, model value)]."""

    def __init__(self, message: str, mismatches: list[tuple[str, float, float]] | None = None) -> None:
        super().__init__(message)
        self.mismatches: list[tuple[str, float, float]] = list(mismatches or [])


class PrecisionSpecError(AlloyError):
    """A precision pair cannot meet its gates (validity rules, K_real, L2 dt below the grid)."""


class CouplingError(AlloyError):
    """The coupling scheduler cannot build or run an exchange."""


class CapacityError(AlloyError):
    """A fixed capacity (particles, contacts, rows, spares) is too small."""


class DeterminismError(AlloyError):
    """A determinism contract is broken (e.g. a NATIVE path diverged)."""


class PrecisionError(AlloyError):
    """Raised by ``HealthMonitor.poll`` under ``on_invalid="raise"`` and by frame checks."""


class LawError(AlloyError):
    """A bench law is malformed or cannot run."""


class NewtonPinError(AlloyError):
    """The installed Newton (or mujoco-warp) is not the pinned commit, or a pinned private hook or feature is
    missing."""


class SolverOptionError(AlloyError, ValueError):
    """A solver option cannot be applied (e.g. MuJoCo noslip under the ``discrete`` integrator)."""


class ClockError(AlloyError, ValueError):
    """A step size is not on the D4 grid ``control_dt / 2**j``."""


class RegistryError(AlloyError, LookupError):
    """Unknown or conflicting registry entry (solver capability, estimator, adapter)."""


class SnapshotError(AlloyError):
    """A ``WorldSnapshot`` cannot save or restore what was asked (e.g. per-world restore of a whole-only array)."""


class DigestError(AlloyError, TypeError):
    """``canonical_json`` met a value it cannot encode canonically."""


class TraceError(AlloyError):
    """A recorded scene or trace cannot be converted faithfully, or a ``Trace`` does not fit the model it is replayed
    on (missing column, shape mismatch, unsupported input) (SCR-WP13-2)."""


class BullyError(AlloyError):
    """Bully agent misuse: illegal trace shape, unbindable action kind, unknown monitor (SCR-WP7-3)."""


class FactoryError(AlloyError):
    """Solver-factory failure (SCR-WP7-3)."""


class PluginContractError(FactoryError):
    """A plug-in breaks the §3.8 contract. ``.failures`` lists each failed check."""

    def __init__(self, message: str, failures: list[str] | None = None) -> None:
        super().__init__(message)
        self.failures: list[str] = list(failures or [])


class SandboxViolation(FactoryError):
    """An agent wrote outside ``alloy/solvers/contrib/<name>/``. ``.paths`` lists the offending paths."""

    def __init__(self, message: str, paths: list[str] | None = None) -> None:
        super().__init__(message)
        self.paths: list[str] = list(paths or [])


class GraphCaptureError(AlloyError):
    """A control period cannot be captured into a CUDA graph: a host sync inside the period, a capture-unsafe call,
    or a step_fn that leaves its buffers swapped (replays would read stale state) (SCR-WP8-1, §3.9)."""


class LaunchError(AlloyError):
    """A multi-GPU mode-A child process failed (non-zero exit or timeout) (SCR-WP8-1)."""


class FitError(AlloyError):
    """A FitProblem the fitter cannot run: unknown leaf, a leaf the twin cannot realise, no calib trials, bad bounds
    (SCR-WP11-1)."""


class SplitError(FitError):
    """``valid`` trials were touched before the final validation call (§3.12)."""


class HeldOutGapError(FitError):
    """``write_back`` refused: held-out gap > 0.5 x bound. ``.gaps`` maps metric -> (gap, bound)."""

    def __init__(self, message: str, gaps: dict[str, tuple[float, float]] | None = None) -> None:
        super().__init__(message)
        self.gaps: dict[str, tuple[float, float]] = dict(gaps or {})
