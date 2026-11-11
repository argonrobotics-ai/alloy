# SPDX-License-Identifier: Apache-2.0
"""Records of the ported forklift criteria: one world's episode in, a verdict of criteria out.

``Episode`` is the minimal state record GA's evaluator reads (its per-tick latches, ``events.py``, and the final
physical state), as numpy arrays over control periods. All positions are in GA's facility frame
(``forklift_site`` docstring). ``Criterion`` mirrors GA's ``C(name, value, ok, unit, threshold, source)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# corner order of every box in ``Episode.loads``: (sx, sy, sz) over (-1, 1), sz fastest (F/evaluator.py:665-671)
BOX_SIGNS = np.array([(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


@dataclass(frozen=True)
class ContactEpisode:
    """One contact episode of a mover with a watched structure class (F/events.py:285-296).

    mover: ``"truck"``, ``"robot"`` or ``"load:<name>"``; structure: GA's class (``"rack"``, ``"wall"``,
    ``"trailer_wall"``,
    ``"trailer_floor"``, ...). onset_speed: the mover body's speed at the first contact (m/s); onset_normal: the
    approach
    speed toward the structure along the contact normal, relative to it (None: reads onset_speed). peak_n: largest
    single
    contact force; impulse_ns: the force integral over the episode. parts: (mover geom label, structure geom label)
    at the
    peak; a rack beam label contains ``_beam``, a trailer nose wall reads ``"<door>:nose"``.
    """

    mover: str
    structure: str
    t: float
    onset_speed: float
    peak_n: float
    impulse_ns: float
    onset_normal: float | None = None
    t_end: float | None = None
    parts: tuple[str, ...] = ()
    open: bool = False


@dataclass(frozen=True)
class LoadFloorTrack:
    """A load's contact with a floor-class surface, per control period (F/events.py:235-241).

    touching: any contact with the ground, a trailer floor, a leveler deck or lip, or a ramp lip plate; xy: the contact
    point's plan position (rows where touching is False are ignored); onset_speed: the approach speed along the contact
    normal at a sample where a contact begins (zeros elsewhere).
    """

    touching: np.ndarray
    xy: np.ndarray
    onset_speed: np.ndarray


@dataclass(frozen=True)
class TrailerTrack:
    """Pose of a trailer body (origin at its rear sill centre on the floor top) per control period."""

    pos: np.ndarray
    quat_xyzw: np.ndarray


@dataclass(frozen=True)
class Episode:
    """One world's run. T samples; t absolute simulated time (s), strictly increasing.

    truck_pos / truck_quat_xyzw: the chassis frame (x forward). speed: signed forward chassis speed (m/s).
    wheel_fz: vertical tyre force per wheel (N). fork_tips: the two blade tips (left, right) as in
    ``forklift_site.BLADE_*``; blade_pitch: blade axis pitch (rad, positive = tips up). loads: name -> eight box
    corners per
    sample in ``BOX_SIGNS`` order (a pallet: its own 48 x 40 x 4.76 in box, bottom face at its body origin; a bale or
    dunnage strip: its box); the carried pallet is ``"carried"``, stored ones ``"stored_<loc>"``, nose pallets
    ``"trailer_1L"``/``"trailer_1R"``, bales ``"bale_<k>"``. load_masses: name -> total mass (kg, pallet included).
    floor: name -> LoadFloorTrack (loads absent are never on a floor). contacts: the contact episodes with watched
    structures (any order). signals: arrays [T] named ``park`` (1.0 applied), ``direction`` (-1/0/1),
    ``engine_running``,
    ``key_on``, ``seat_switch_closed``; GA reads the first four at the end. head_contact_n: the robot's largest force on
    the overhead guard per sample. humanoid: a seated operator is simulated (else operator_seated and
    head_clear_of_guard
    are not judged). trailer_pos / trailer_quat_xyzw: the target trailer (None when there is none);
    other_trailers: further docked trailers (load-bales judges against both). beam_contacts: rack beam geom labels the
    carried pallet touches at the last sample (put-away). fault: a non-finite or unphysical world ends without success.
    """

    t: np.ndarray
    truck_pos: np.ndarray
    truck_quat_xyzw: np.ndarray
    speed: np.ndarray
    wheel_fz: np.ndarray
    fork_tips: np.ndarray
    blade_pitch: np.ndarray
    loads: dict[str, np.ndarray] = field(default_factory=dict)
    load_masses: dict[str, float] = field(default_factory=dict)
    floor: dict[str, LoadFloorTrack] = field(default_factory=dict)
    contacts: tuple[ContactEpisode, ...] = ()
    signals: dict[str, np.ndarray] = field(default_factory=dict)
    head_contact_n: np.ndarray | None = None
    humanoid: bool = False
    trailer_pos: np.ndarray | None = None
    trailer_quat_xyzw: np.ndarray | None = None
    other_trailers: dict[str, TrailerTrack] = field(default_factory=dict)
    beam_contacts: tuple[str, ...] = ()
    fault: str | None = None
    plan_done_t: float | None = None
    plan_known: bool = False

    @property
    def n(self) -> int:
        return len(self.t)


@dataclass(frozen=True)
class Criterion:
    """One criterion: GA's name, measured value, verdict, unit, threshold text and source cite.

    ``passed`` is True or False when the port judges it from simulation state, None when it does not (``note`` then
    says why). ``by_construction`` is kept for readers of the old record; nothing sets it now, because a criterion
    that holds by construction is not judged (``passed`` None).
    """

    name: str
    value: Any
    passed: bool | None
    unit: str
    threshold: str
    cite: str
    by_construction: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        if self.passed is not None and type(self.passed) is not bool:    # a numpy bool would fail ``is False``
            object.__setattr__(self, "passed", bool(self.passed))

    @property
    def judged(self) -> bool:
        return self.passed is not None


def partly(*parts: bool | None) -> bool | None:
    """A criterion from its parts: False if a judged part fails, None if a part is not judged, else True."""
    if any(p is not None and not p for p in parts):
        return False
    return None if any(p is None for p in parts) else True


def not_judged(name: str, value: Any, unit: str, threshold: str, cite: str, why: str) -> Criterion:
    """A criterion this port cannot judge from its simulation (``why`` says what is missing)."""
    return Criterion(name, value, None, unit, threshold, cite, False, why)


@dataclass(frozen=True)
class Verdict:
    """All criteria of a task, GA's success rule and its score, over the criteria the port judges.

    success: every judged criterion passes, at least one is judged, and there is no fault. A criterion that is not
    judged (``passed`` None) is never counted as passed.
    """

    task: str
    criteria: tuple[Criterion, ...]
    success: bool
    score: float
    details: dict[str, Any] = field(default_factory=dict)

    def by_name(self) -> dict[str, Criterion]:
        return {c.name: c for c in self.criteria}

    def failed(self) -> list[str]:
        return [c.name for c in self.criteria if c.passed is False]

    def not_judged(self) -> list[str]:
        """Gym-Anything criteria this port does not judge."""
        return [c.name for c in self.criteria if c.passed is None]

    @property
    def judged(self) -> tuple[int, int]:
        """(judged, all) over Gym-Anything's criteria."""
        return sum(c.passed is not None for c in self.criteria), len(self.criteria)

    @property
    def passed_count(self) -> tuple[int, int, int]:
        """(passed, judged, all)."""
        return sum(c.passed is True for c in self.criteria), *self.judged

    def summary(self) -> str:
        j, n = self.judged
        text = f"{self.passed_count[0]} passed of {j} judged; {j} of {n} Gym-Anything criteria judged"
        return text + (f"; not judged: {', '.join(self.not_judged())}" if j < n else "")


def judged_success(criteria: list[Criterion], fault: str | None) -> bool:
    """Every judged criterion passes (at least one is judged) and there is no fault."""
    judged = [c.passed for c in criteria if c.passed is not None]
    return bool(judged) and all(judged) and fault is None


def make_verdict(task: str, criteria: list[Criterion], fault: str | None, details: dict | None = None,
                 score: float | None = None) -> Verdict:
    """GA's evaluate() over the judged criteria: success = all of them met, none failed, no fault.

    The score is the fraction of judged criteria met unless a task has its own.
    """
    judged = [c for c in criteria if c.passed is not None]
    frac = sum(c.passed for c in judged) / max(len(judged), 1) if score is None else score
    return Verdict(task, tuple(criteria), judged_success(criteria, fault), float(frac), details or {})


def contact_episodes_from_series(mover: str, structure: str, t, force, speed, normal_speed=None,
                                 parts: tuple[str, ...] = ()) -> tuple[ContactEpisode, ...]:
    """Group a per-sample largest-contact-force series (N, 0 = no contact) into episodes: a run of samples with force
    > 0.

    The impulse is force x sample period summed over the run; the onset speeds are read at its first sample. A run that
    reaches the last sample is returned open.
    """
    t, force = np.asarray(t, float), np.asarray(force, float)
    dt = np.diff(t, append=t[-1] + (t[-1] - t[-2] if len(t) > 1 else 0.0))
    on = force > 0.0
    out, i, n = [], 0, len(t)
    while i < n:
        if not on[i]:
            i += 1
            continue
        j = i
        while j < n and on[j]:
            j += 1
        nv = None if normal_speed is None else float(normal_speed[i])
        out.append(ContactEpisode(mover, structure, float(t[i]), float(speed[i]), float(force[i:j].max()),
                                  float((force[i:j] * dt[i:j]).sum()), nv,
                                  None if j >= n else float(t[j]), parts, j >= n))
        i = j
    return tuple(out)
