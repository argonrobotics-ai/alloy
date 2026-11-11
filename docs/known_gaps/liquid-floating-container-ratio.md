# A floating container much lighter than the liquid filling its outline is refused

## What
- A rigid container flagged by its card (`features.passage_width`: a cup, bowl, tray or boat) floating in SPH liquid
  routes like any floating body. The liquid's solver integrates it inside every SPH step at its effective mass
  (SCR-WP9-2), as a strong pair (staggered, never ADMM).
- That holds up to D5's ratio m_a/m_s = ρ_liquid·V_outline/m of 50 (`coupler_select.FLOATING_CONTAINER_RATIO_MAX`).
  V_outline is the container's outline volume (its box for a mesh), so the air inside counts.
- Above 50 the build raises `FloatingContainerError` (a `RouteError`). The message names the ratio and the mass
  that would route.
- Light bodies that are not flagged containers (a foam block, an unflagged mesh cup) are not bounded. They route as
  before, beyond what was measured.

## Number
- What was measured: open cups 60 x 60 x 50 mm (an `add_mesh` open box, collision "sdf"), released just over still
  water in 140 and 400 mm tanks. Particle radius 2 mm (h = 8 mm), default dt, 2 s per case. GPU (L4) through
  `alloy.Scene` and the MuJoCo proxy.
- D5 ratios 3, 6, 10, 30 and 50 all held. Walls were 8, 4 and 2 mm (h, h/2, h/4). Every world was healthy, no
  water entered a cup, and the settled bottom matched the tank's rise minus the draft m/(ρ·L²) within 0.27 mm.
- Nothing above 50 was run. The laws and their logs are in the development notes (not shipped).
- On the CPU, one ratio-3 release with 4 mm walls went INVALID at 0.55 s: a particle went 4.1 mm into the wall,
  and the health limit is half its thickness. The same case was healthy on the GPU.

## Why
- A coupled step feeds the body forces from a liquid that moves with it. When the liquid it must push aside
  outweighs the body many times over, the correction can overshoot (the added-mass instability).
- Integrating the body inside SPH's step removes most of it, and the probes show it holding to 50. Above that,
  stability is not shown.
- The complete fix interlinks the body's motion with the pressure solve (Gissler et al. 2019), which is not built
  (the engine specification's section on this coupling, not shipped).

## Workaround
- Make the container heavier: the message gives the mass that routes. Most real ones are inside the bound already:
  a 2 mm-walled 80 mm plastic cup is at about 10-20, a ceramic mug at about 2.
- Or use a smaller outline, or make it static (`static=True`) or kinematic when it need not float.
- A container on a robot or in a hand is not floating, so the bound does not apply.

## Planned fix
Gissler interlinked pressure (the engine specification's section on this coupling, and option (a2) in WP9's
liquid-coupling scope note; neither shipped) if a task needs floating containers lighter than 1/50 of their outline's
liquid. Not scheduled.
