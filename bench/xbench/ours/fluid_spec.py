# SPDX-License-Identifier: Apache-2.0
"""What the P3-2 SPH law packs say (``fluid_pack.py``): the scenes, the npz export per law and Alloy's SPH parameters
with their meaning. Text and records only; building and scoring live in ``fluid_pack.py``."""

from __future__ import annotations

SCENE = {
    "A13.hydrostatic": "a liquid column of nx x ny x nz lattice cells resting in an open box tank of the same "
                       "footprint (headroom cells above), held for the whole run; the force is scored on the x_lo wall",
    "A13.archimedes": "a pool nx x ny x nz cells deep in an open tank, a free box (bx x bx x bz cells, density rho_b) "
                      "floating in the middle at its expected draft m / (rho0 A_box); cells under it are left out",
    "A14.dam_break": "Martin & Moyce: an a x depth x 2a cell column against the x_lo wall of a 5a x depth x (2a + 10) "
                     "cell open tank, released from rest at t = 0 (no gate)",
}

EXPORT = {
    "A14.dam_break": {
        "particle_q": "[T, W, P, 3] float32: liquid particle positions [m], world frame, every frame (frame k = "
                      "physics step k x record_every; frame 0 = the packed state)",
        "metric": "ritter_excess (twin front_kernel): front x = max x of the liquid with z < 4 r, plus r; Z = x / a, "
                  "T = t sqrt(2 g / a); error = max(0, max over T <= 3 of Z - (1 + 2 T))"},
    "A13.archimedes": {
        "body_q": "[T, W, 1, 7] float32: box frame origin (= its COM) x y z and quaternion x y z w, world frame, every "
                  "frame (record_every 1)",
        "body_qd": "[T, W, 1, 6] float32: COM linear and angular velocity, world frame (contract; not scored)",
        "body_names": "['body_0'] (bodies[].name)",
        "particle_q": "float32 [Tp, W, P, 3] holding frame 0 (particle_frame = [0]), or every frame",
        "metric": "draft = (V_liquid / A_tank - z_bottom) / (1 - A_box / A_tank), z_bottom = box z - half height, mean "
                  "over the second half of the run; V_liquid = Alloy's LIQ-2 volume of frame 0 (the pack's "
                  "particles); error = |draft - m / (rho0 A_box)| / (m / (rho0 A_box))"},
    "A13.hydrostatic": {
        "wall_impulse": "[T, W, 3] float64, every frame (record_every 1): impulse [N s] the liquid has applied to the "
                        "x_lo wall since frame 0, world frame (the liquid pushes it towards -x, so x falls). From the "
                        "engine's fluid-solid coupling: the time integral of the net liquid force on that wall; for a "
                        "wall handled by position / velocity correction, minus the momentum that wall's correction "
                        "gives the liquid, summed over every physics step",
        "particle_q": "float32 [Tp, W, P, 3] holding particle_frames (listed in particle_frame) or every frame: 0 and "
                      "L - 1, L = the twin's last column = stride x floor(steps / stride), stride = ceil(steps / 2000) "
                      "(recompute with your steps if your dt differs)",
        "metric": "wall_force = -(J_x(end) - J_x(t_mid)) / (t_end - t_mid), t_mid = half the run, vs rho0 g w H^2 / 2 "
                  "with w = the wall's inner width and H = V(0) / base area; volume_loss = |V(end) - V(0)| / V(0); V = "
                  "Alloy's LIQ-2 volume sum m / (rho0 max(rho / rho0, 1)) (wall SDF volumes in rho) of the engine's "
                  "particles, column k measured on frame k - 1"},
}


def entry(value, unit: str, meaning: str) -> dict:
    return {"value": value, "unit": unit, "meaning": meaning}


def sph_params(cfg, solver, model, dt: float) -> dict:
    """Alloy's SPH parameters (``DFSPHConfig`` as the twin builds it, the solver's resolved viscosity, the model's
    particle mass and rest density), each with what it means."""
    from alloy.solvers.sph.config import full_neighbor_count
    from alloy.solvers.sph.kernels import lattice_sum

    r, s, h = cfg.particle_radius, cfg.spacing, cfg.h
    rho0 = float(model.sph.rest_density.numpy()[0])
    m, S = float(model.particle_mass.numpy()[0]), lattice_sum(s, h, cfg.kernel)
    e = entry
    return {
        "particle_radius": e(r, "m", "r; particles start on a cubic lattice of spacing 2r, one per (2r)^3 cell"),
        "spacing": e(s, "m", "2r: rest particle distance (lattice spacing), the 'particle size' of most engines"),
        "support_radius": e(h, "m", f"h = {cfg.support_factor:g} r = 2 x spacing: kernel cutoff, W = 0 at distance "
                            ">= h (SPlisHSPlasH convention; Monaghan's smoothing length of this spline is h / 2)"),
        "kernel": e(cfg.kernel, "-", "cubic spline W = 8 / (pi h^3) f(q), q = |x| / h, f = 6q^3 - 6q^2 + 1 "
                    "(q <= 1/2), 2 (1 - q)^3 (q <= 1)"),
        "rest_density": e(rho0, "kg/m^3", "rho0 of the liquid"),
        "particle_mass": e(m, "kg", f"m = rho0 / S, S = {S:.9g} m^-3 the kernel sum over the rest lattice (self "
                           "included): an interior lattice particle has rho = rho0 exactly. rho0 (2r)^3 = "
                           f"{rho0 * s ** 3:.9g} kg is {m / (rho0 * s ** 3) - 1:.2e} lighter (relative)"),
        "lattice_neighbours": e(full_neighbor_count(cfg), "-", "lattice neighbours within h (self excluded)"),
        "viscosity_model": e(solver.viscosity, "-", "Alloy's choice for mu <= 1e-2 Pa s: XSPH velocity smoothing "
                             "v_i += c sum_j (m_j / rho_j) (v_j - v_i) W_ij after gravity; no other viscous force"),
        "xsph_c": e(cfg.xsph_c, "-", "XSPH coefficient c"),
        "pressure_solver": e("DFSPH", "-", "divergence-free SPH (Bender & Koschier 2017): divergence solve, then "
                             f"constant-density solve every step; unilateral (no tension); {cfg.pressure_solver} "
                             "iterations (projected preconditioned CG)"),
        "max_density_error": e(cfg.max_density_error, "-", "mean max(rho - rho0, 0) / rho0 per world after the "
                               "density solve (and the per-particle max <= 10 x this)"),
        "max_divergence_error": e(cfg.max_divergence_error, "-", "mean max(Drho/Dt, 0) dt / rho0 per world"),
        "iterations": e([cfg.min_iterations, cfg.max_iterations], "-", "min / max pressure iterations per solve"),
        "boundary": e("sdf_tube", "-", "walls and bodies enter the density as SDF half-space volumes (tube map with "
                      "order-1 curvature, Koschier & Bender 2017 family); free-slip walls, no wall viscosity"),
        "surface_tension": e(cfg.surface_tension, "-", "none"),
        "time_step": e(dt, "s", "fixed law dt, 1 substep (no CFL adaptation in law runs)"),
        "initial_pressure": e("primed", "-", "SolverDFSPH.prime solves the hydrostatic pressure as the solver's warm "
                              "start before step 1; positions and velocities stay as packed (an engine starts from "
                              "rest; its own pressure initialisation, if any, is a setting to record)"),
        "neighbor_capacity": e(cfg.neighbor_capacity, "-", "max neighbours per particle"),
    }
