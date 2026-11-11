# SPDX-License-Identifier: Apache-2.0
"""Points along a Franka Panda arm from its seven joint angles (numpy only), for checkers that need "how close is the
arm to X" when a probe gives only the grasp point.

Franka's published (Craig) parameters: a = [0, 0, 0, .0825, -.0825, 0, .088], d = [.333, 0, .316, 0, .384, 0, 0],
alpha = [0, -pi/2, pi/2, pi/2, -pi/2, pi/2, pi/2]; flange 0.107 beyond link 7, grasp point (TCP) 0.1034 beyond the
hand frame, which is turned -45 degrees about z. The hand is about 0.2 m wide across the finger axis (hand y).
The base is placed by anchoring the chain's grasp point on the ``ee_pos`` probe, with the base frame axes parallel
to the world axes (the robot stands upright, facing +x, as the request's "in front of it" says).
"""
from __future__ import annotations

import numpy as np

A = (0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088)
D = (0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.0)
ALPHA = (0.0, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2)
FLANGE, TCP, HAND_HALF_WIDTH, SPACING = 0.107, 0.1034, 0.10, 0.02


def _link(a: float, d: float, alpha: float, theta: np.ndarray) -> np.ndarray:
    """(N, 4, 4) Craig transform for angles theta (N,)."""
    ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    m = np.zeros((theta.shape[0], 4, 4))
    m[:, 0, :3] = np.stack([ct, -st, np.zeros_like(ct)], -1)
    m[:, 1, :3] = np.stack([st * ca, ct * ca, -sa * np.ones_like(ct)], -1)
    m[:, 2, :3] = np.stack([st * sa, ct * sa, ca * np.ones_like(ct)], -1)
    m[:, :3, 3] = np.stack([a * np.ones_like(ct), -sa * d * np.ones_like(ct), ca * d * np.ones_like(ct)], -1)
    m[:, 3, 3] = 1.0
    return m


def chain(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(points (N, K, 3), grasp (N, 3)) relative to the base: points along the arm's centre line (joint frames, flange,
    grasp point, every 2 cm) and the hand's width bar (0.2 m across hand y, 6 cm beyond the flange)."""
    q = np.asarray(q, np.float64).reshape(-1, 7)
    t = np.broadcast_to(np.eye(4), (q.shape[0], 4, 4)).copy()
    nodes = [t[:, :3, 3].copy()]
    for k in range(7):
        t = t @ _link(A[k], D[k], ALPHA[k], q[:, k])
        nodes.append(t[:, :3, 3].copy())
    z = t[:, :3, 2]
    y = t[:, :3, 1]
    flange = nodes[-1] + z * FLANGE
    grasp = flange + z * TCP
    nodes += [flange, grasp]
    path = _dense(nodes)
    bar_centre = flange + z * 0.06
    bar = [bar_centre + y * s for s in np.linspace(-HAND_HALF_WIDTH, HAND_HALF_WIDTH, 9)]
    return np.concatenate([path, np.stack(bar, 1)], axis=1), grasp


def _dense(nodes: list[np.ndarray]) -> np.ndarray:
    """Points along the polyline through ``nodes`` (each (N, 3)), at most SPACING apart."""
    pts = [nodes[0][:, None]]
    for a, b in zip(nodes[:-1], nodes[1:]):
        n = max(1, int(np.ceil(np.max(np.linalg.norm(b - a, axis=-1)) / SPACING)))
        pts.append(a[:, None] + (b - a)[:, None] * (np.arange(1, n + 1) / n)[None, :, None])
    return np.concatenate(pts, axis=1)


def arm_points(q: np.ndarray, ee_pos: np.ndarray) -> np.ndarray:
    """World-frame arm points for joint angles q (..., 7) and grasp points ee_pos (..., 3): the chain is placed so its
    grasp point lands on ee_pos. Shape (..., K, 3)."""
    lead = q.shape[:-1]
    pts, grasp = chain(q.reshape(-1, 7))
    world = pts + (np.asarray(ee_pos, np.float64).reshape(-1, 3) - grasp)[:, None, :]
    return world.reshape(*lead, pts.shape[1], 3)


def distance_to_arm(q: np.ndarray, ee_pos: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Distance (...,) from ``point`` (..., 3) to the nearest arm point."""
    pts = arm_points(q, ee_pos)
    return np.linalg.norm(pts - np.asarray(point, np.float64)[..., None, :], axis=-1).min(axis=-1)
