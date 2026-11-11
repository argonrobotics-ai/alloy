# SPDX-License-Identifier: Apache-2.0
"""The reference benchmark's hands-off window, judged from the recorded window (numpy only; docs/notes/paper-pack.md).

The reference (Gym-Anything, "GA", gift_bags/task.py) fails ``hands_off_window`` if, at any of the 3 s of control
steps before the evaluation, the reaction force on any robot link is above 1e-3 N (``_robot_contact``, task.py:246-249;
``evaluate``, task.py:297-299: the window must also be complete). The public API reads no contact forces, so this
judges the same window by geometry: at every recorded control period (0.01 s, every step), is a collision shape of the
tool (the gripper's two fingers or the hook's shaft and tine; the palm, wrist and carriages have no collision) within
``margin`` (5 mm, the engine's robot contact gap) of anything it could touch?

- boxes (a carton, its flaps and walls, the contents, the mat): exact oriented-box separation (SAT) of the tool's box
  grown by the margin against the box;
- points (a film's vertices: the bag, its board): inside the tool's box grown by the margin. The film's cells are
  about 12 mm, so a 20 mm wide finger edge can pass between vertices of a sheet it touches: the point test can miss
  a graze, as GA's own plane test does between vertices;
- the floor: the tool's lowest corner within the margin of the counter plane.

Differences from GA: a distance, not a force (a shape within 5 mm that bears no load counts, so this can fail a window
GA passes, never the other way round, except for the cases below); the jar is taken as the box around it; a film's
contact through a piece it does not have a vertex in is missed as above; contacts with objects not listed as targets
(the tape strip of open-carton, lying far from the hook) are not seen. GA's bag-purchase rule (only the job's items)
does not apply to these three tasks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NOTE = ("hands_off_window is a distance test, not GA's reaction force above 1e-3 N on a robot link: a tool box within "
        "5 mm of a target at any control period counts, loaded or not (it can fail a window GA passes), and a graze "
        "of a film between two of its vertices can be missed (gift_hands_off.py)")
MARGIN = 0.005                          # m: the engine's contact distance for robot shapes (contact_gap)


def rot_from_quat(q: np.ndarray) -> np.ndarray:
    """[..., 3, 3] rotation matrices from [..., 4] (w, x, y, z) quaternions (normalised here)."""
    q = np.asarray(q, float)
    w, x, y, z = np.moveaxis(q / np.linalg.norm(q, axis=-1, keepdims=True), -1, 0)
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


@dataclass(frozen=True)
class WorldBox:
    """Oriented boxes, one per leading index (frame, world): centre [..., 3], rotation [..., 3, 3], half sizes [3]."""
    centre: np.ndarray
    rot: np.ndarray
    half: tuple

    @classmethod
    def of_body(cls, pos, quat_wxyz, centre, half) -> "WorldBox":
        """A box fixed in a body: ``centre`` in the body's frame; ``pos``, ``quat_wxyz`` the body's pose."""
        R = rot_from_quat(quat_wxyz)
        return cls(np.asarray(pos, float) + R @ np.asarray(centre, float), R, tuple(half))


def boxes_overlap(a: WorldBox, b: WorldBox, margin: float = 0.0) -> np.ndarray:
    """Do ``a`` grown by ``margin`` and ``b`` overlap? Separating-axis test on the 15 axes, over the leading dims."""
    ha, hb = np.asarray(a.half, float) + margin, np.asarray(b.half, float)
    R = np.einsum("...ji,...jk->...ik", a.rot, b.rot)
    A = np.abs(R) + 1e-9
    t = np.einsum("...ji,...j->...i", a.rot, np.asarray(b.centre, float) - a.centre)
    sep = np.zeros(t.shape[:-1], bool)
    for i in range(3):
        sep |= np.abs(t[..., i]) > ha[i] + A[..., i, :] @ hb
        sep |= np.abs(np.einsum("...i,...i->...", t, R[..., :, i])) > A[..., :, i] @ ha + hb[i]
    for i in range(3):
        i1, i2 = (i + 1) % 3, (i + 2) % 3
        for j in range(3):
            j1, j2 = (j + 1) % 3, (j + 2) % 3
            ra = ha[i1] * A[..., i2, j] + ha[i2] * A[..., i1, j]
            rb = hb[j1] * A[..., i, j2] + hb[j2] * A[..., i, j1]
            sep |= np.abs(t[..., i2] * R[..., i1, j] - t[..., i1] * R[..., i2, j]) > ra + rb
    return ~sep


def points_in_box(points: np.ndarray, box: WorldBox, margin: float = 0.0) -> np.ndarray:
    """Is any of the [..., n, 3] points inside ``box`` grown by ``margin``? Over the leading dims of ``box``."""
    local = np.einsum("...ji,...nj->...ni", box.rot, points - box.centre[..., None, :])
    return np.all(np.abs(local) <= np.asarray(box.half, float) + margin, axis=-1).any(axis=-1)


def floor_reach(box: WorldBox, floor_z: float, margin: float = 0.0) -> np.ndarray:
    """Is the box's lowest point within ``margin`` of the plane z = ``floor_z``?"""
    low = box.centre[..., 2] - np.abs(box.rot[..., 2, :]) @ np.asarray(box.half, float)
    return low <= floor_z + margin


def contact_track(tool: list[WorldBox], boxes: list[WorldBox], points: list[np.ndarray],
                  floor_z: float | None = 0.0, margin: float = MARGIN) -> np.ndarray:
    """[frames, worlds] True where any tool box is within ``margin`` of a target box or point set or the floor."""
    hit = np.zeros(tool[0].centre.shape[:-1], bool)
    for t in tool:
        for b in boxes:
            hit |= boxes_overlap(t, b, margin)
        for p in points:
            hit |= points_in_box(p, t, margin)
        if floor_z is not None:
            hit |= floor_reach(t, floor_z, margin)
    return hit


def tool_world_boxes(frames: dict, robot: str, tool_boxes: list[tuple[str, tuple, tuple]]) -> list[WorldBox]:
    """The tool's boxes per recorded frame from a trajectory's ``"<robot>/<body>.pos"`` / ``.quat_wxyz``."""
    return [WorldBox.of_body(frames[f"{robot}/{b}.pos"], frames[f"{robot}/{b}.quat_wxyz"], c, h)
            for b, c, h in tool_boxes]


def window_verdict(contact: np.ndarray, steps: int, dt: float) -> list[dict]:
    """Per world: GA's ``hands_off_window`` = the window is complete (``steps`` control periods after the first frame)
    and no frame touches. ``contact``: [frames, worlds]. Also the first touching time [s] into the window, or None."""
    complete = contact.shape[0] - 1 >= steps
    out = []
    for w in range(contact.shape[1]):
        hits = np.flatnonzero(contact[:, w])
        out.append({"hands_off_window": bool(complete and not len(hits)),
                    "hands_off_first_contact_s": round(float(hits[0]) * dt, 3) if len(hits) else None,
                    "hands_off_contact_frames": int(len(hits))})
    return out


def join_frames(parts: list[dict]) -> dict:
    """One trajectory from consecutive runs' frames (each run's first frame repeats the last of the one before)."""
    keys = parts[0].keys()
    return {k: np.concatenate([parts[0][k]] + [p[k][1:] for p in parts[1:]]) for k in keys}
