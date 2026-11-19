# SPDX-License-Identifier: Apache-2.0
"""Surface tension for small liquid pieces under ``surface_tension="csf"``.

CSF reads a curvature from the neighbourhood. On a piece smaller than the kernel (a stream one or two particles wide,
a droplet of a few dozen particles) that curvature is ill-defined, and the force it gives does net work: a free
7-particle cluster at rest reached 10.2 J/kg in 1500 steps, 11x the surface energy it holds, and droplets were flung
0.4 m off a pour at 1.5 mm particles. There the tension is Akinci et al. 2013's pair force instead (tension.py, the
same terms and calibration, without adhesion):
    a_i += sum_j c_ij K_ij gamma_ij [ m_j C(|x_ij|) (x_j - x_i) / |x_ij| - s(|x_ij|) (n_i - n_j) ]
Each pair term is equal and opposite, so momentum is conserved. A piece is small when no particle in a neighbourhood
has a full neighbourhood: the band reads N_i, the largest neighbour count over i and its neighbours (at most 18 in a
19-particle cluster however it packs, about 26 next to a resolved liquid's interior; a particle's own count, 9-18
against 13-18 at a resolved surface, does not tell them apart). CSF is weighted by
w_i = smoothstep((N_i - CSF_BAND[0]) / (CSF_BAND[1] - CSF_BAND[0])) and each pair by c_ij = 1 - (w_i + w_j) / 2, so a
particle crossing the band is not kicked. Resolved liquid sees CSF alone, as before; the adhesion and contact angle
stay CSF's.

Measured (wp9-results.md; free clusters at rest in zero gravity, 1500 steps, kinetic energy against the cluster's
surface energy): 7 / 19 / 33 particles 0.1 / 0.5 / 0.6 % (under 2.5 % with gamma x0.9-1.1); CSF alone 1,120 / 59 /
0.7 %. 123 and 257 particles (R/h 1.5 and 2, below the validated range) 15 and 10 % (CSF alone 23 % at 123), the
free-drop gap CSF has.
Cohesion alone (a central force with a potential) collapsed a 7-particle cluster to 0.12 spacings in two steps at its
own Kirkwood-Buff gamma and left 19 particles at 77 % at Akinci's; a band on a particle's own count (8-12) left 19
particles at 93 %.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from .pressure import Particles, is_fluid, nbr_base, nbr_j
from .tension import RAYLEIGH_FACTOR, cohesion_c, curvature_weight, lattice_sigma_per_gamma

CSF_BAND = (18.0, 24.0)  # largest neighbour count nearby over which CSF takes over from the pair force


@wp.func
def csf_weight(count: float, lo: float, hi: float) -> float:
    """0 under ``lo`` neighbours, 1 from ``hi``, smoothstep between."""
    x = wp.clamp((count - lo) / (hi - lo), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


@wp.kernel(enable_backward=False)
def max_count_kernel(p: Particles, out: wp.array[wp.float32]):
    """The largest neighbour count over i and its neighbours: at most 18 in a free 19-particle cluster however it
    packs (it holds 19), about 26 next to a resolved liquid's interior, where a particle's own count (9-18 against
    13-18 at a surface) does not tell them apart."""
    i = wp.tid()
    s = p.count[i]
    b_ = nbr_base(p, i)
    for k in range(p.count[i]):
        s = wp.max(s, p.count[nbr_j(p, b_, i, k)])
    out[i] = float(s)


@wp.kernel(enable_backward=False)
def sparse_pair_kernel(p: Particles, ratio: wp.array[wp.float32], normals: wp.array[wp.vec3],
                       gamma: wp.array[wp.float32], mcount: wp.array[wp.float32], h: float, lo: float, hi: float,
                       dt: float):
    """v_i += dt sum_j c_ij K_ij gamma_ij [cohesion - curvature] (module doc); reads positions, normals and counts
    only, so the in-place velocity update is race-free."""
    i = wp.tid()
    if not is_fluid(p, i) or gamma[i] == 0.0:
        return
    xi = p.q[i]
    ni = normals[i]
    rho_i = ratio[i] * p.rho0[i]
    wi = csf_weight(mcount[i], lo, hi)
    a = wp.vec3(0.0, 0.0, 0.0)
    b_ = nbr_base(p, i)
    for k in range(p.count[i]):
        j = nbr_j(p, b_, i, k)
        c = 1.0 - 0.5 * (wi + csf_weight(mcount[j], lo, hi))
        g = 0.5 * (gamma[i] + gamma[j])
        rho_j = ratio[j] * p.rho0[j]
        if c <= 0.0 or g == 0.0 or rho_i + rho_j <= 0.0:
            continue
        rv = xi - p.q[j]
        r = wp.length(rv)
        acc = curvature_weight(r, h) * (ni - normals[j])
        if r > 1.0e-6 * h:
            acc += (p.mass[j] * cohesion_c(r, h) / r) * rv
        a -= (c * 2.0 * p.rho0[i] / (rho_i + rho_j) * g) * acc
    p.qd[i] = p.qd[i] + dt * a


def pair_gamma(sigma: np.ndarray, rho0: np.ndarray, mass: np.ndarray, spacing: float, h: float,
               kind: str) -> np.ndarray:
    """Per-particle gamma = sigma / (RAYLEIGH_FACTOR c_lattice(rho0, mass)), tension.py's calibration; 0 where sigma
    is 0."""
    out = np.zeros(max(len(sigma), 1), np.float32)
    for r0, m in {(float(a), float(b)) for a, b, s in zip(rho0, mass, sigma, strict=True) if s > 0.0}:
        sel = (rho0 == r0) & (mass == m) & (sigma > 0.0)
        out[:len(sigma)][sel] = sigma[sel] / (RAYLEIGH_FACTOR * lattice_sigma_per_gamma(spacing, h, kind, r0, m))
    return out


__all__ = ["CSF_BAND", "csf_weight", "max_count_kernel", "pair_gamma", "sparse_pair_kernel"]
