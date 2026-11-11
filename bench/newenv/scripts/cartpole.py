# SPDX-License-Identifier: Apache-2.0
"""Harness action laws for E06 (cart-pole): zero force at rest, then LQR state feedback on the probe values.

The gains come from the linearized model of the request's numbers (cart 1 kg, uniform pole 0.1 kg x 0.5 m on a free
hinge at the cart, force 10 N x action, g = 9.81) and a continuous-time LQR; the law is the harness's, not the agent's.
Probe values used: cart_x, pole_angle (from upright, + toward +x), cart_v, pole_rate.
"""
from __future__ import annotations

import numpy as np

PHASES = [{"name": "act_hold", "seconds": 5.0, "reset": True}]

CART_KG, POLE_KG, POLE_M, GRAVITY, FORCE_N = 1.0, 0.1, 0.5, 9.81, 10.0


def linear_model() -> tuple[np.ndarray, np.ndarray]:
    """(A, B) of d/dt [x, theta, v, w] with force input, for a pole hinged at its base on the cart."""
    m, big_m, l = POLE_KG, CART_KG, POLE_M / 2.0
    inertia = m * POLE_M ** 2 / 3.0                         # about the hinge
    den = inertia - (m * l) ** 2 / (big_m + m)
    a_theta = m * GRAVITY * l / den                         # theta'' = a_theta * theta + b_theta * F
    b_theta = -(m * l / (big_m + m)) / den
    a_x, b_x = -m * l * a_theta / (big_m + m), 1.0 / (big_m + m) - m * l * b_theta / (big_m + m)
    a = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, a_x, 0, 0], [0, a_theta, 0, 0]], float)
    return a, np.array([[0.0], [0.0], [b_x], [b_theta]])


def lqr_gain(q_diag=(1.0, 20.0, 0.5, 1.0), r: float = 0.05) -> np.ndarray:
    """Continuous-time LQR gain K (1, 4) from the stable invariant subspace of the Hamiltonian matrix."""
    a, b = linear_model()
    q = np.diag(q_diag)
    ham = np.block([[a, -b @ b.T / r], [-q, -a.T]])
    vals, vecs = np.linalg.eig(ham)
    stable = vecs[:, vals.real < 0]
    p = np.real(stable[4:] @ np.linalg.inv(stable[:4]))
    return (b.T @ p) / r


GAIN = lqr_gain()


def action(phase: str, t: float, probe: dict[str, np.ndarray], act_dim: int) -> np.ndarray:
    """-K x / 10 N, clipped, during "act_hold"; zero otherwise."""
    if phase != "act_hold":
        return np.zeros((1, act_dim))
    state = np.stack([probe["cart_x"], probe["pole_angle"], probe["cart_v"], probe["pole_rate"]], axis=-1)
    force = -(state @ GAIN.T)[:, 0]
    return np.clip(force / FORCE_N, -1.0, 1.0).reshape(-1, 1)
