# Steel in the bundled Newton 1.8 lets a robot arm's contacts through 54 times more often than the copy before it

## What is wrong
Steel (`SolverSteel`, Newton's reduced-coordinate rigid solver) comes inside Alloy's bundled Newton. No default route
uses it: robots and rigid objects step on MuJoCo. It is reached only by opting in, in one of two ways:
- `RouterConfig(pile_backend="steel")` (engine layers) sends a pile of more than 50 free bodies per world to a solver
  entry named `"steel"`. Alloy registers no such entry, so the build raises `BuildError: ... no solver capability
  'steel'` until you register one (a solver plug-in around `SolverSteel`).
- Engine-layer code that steps `alloy._vendor.newton.solvers.SolverSteel` itself.

Since Alloy's Newton moved to release 1.8 (`db915e1c`), Steel on a robot arm that picks and places blocks lets parts
pass through each other in about a third of the episodes. The Steel of the Newton copy Alloy bundled before
(`ac7f7830` plus our fixes) did that in under 1 % of them at the same settings. Stacks of free bodies, which is what
`pile_backend` routes, are not affected.

## Number
The scene is Alloy's pick-and-place benchmark (T12: a Franka arm and 8 blocks), in 2 s episodes on one L4. An episode
counts as a glitch when the benchmark's watch flags it; nearly every one here is a penetration over 1 cm. The
benchmark's bar is under 0.1 % of episodes. Glitched episodes / episodes (upper 95 % bound):

| Steel's settings | Newton 1.8 (`db915e1c`) | the copy before (`ac7f7830` + ours) |
|---|---|---|
| 0.625 ms step, baumgarte 0.2 (its default), 16 iterations | 162 / 512 = 31.6 % (35.2 %) | 3 / 512 = 0.59 % (1.5 %) |
| the same with 2,048 articulated rows instead of 1,024 | 163 / 512 = 31.8 % (35.4 %) | - |
| 0.625 ms step, baumgarte 0.05, 8 iterations | 338 / 1,024 = 33.0 % (35.5 %) | 66 / 1,024 = 6.4 % (7.85 %) |

- 54 times is 31.8 % against 0.59 %.
- Alloy's default route (MuJoCo Warp) on the same scene, with the same watch: 0 / 3,072 (upper 95 % bound 0.098 %),
  on Newton 1.8 as before it.
- Free bodies: 27 one-metre cubes in 9 stacks, 3,072 worlds, 10 ms step. Both Steel copies: 0 / 3,072 stacks fall or
  keep moving, and the cubes rest 0.31 mm (Newton 1.8) and 0.32 mm (before) from their ideal height (median).
- Both Steel copies also fail two of Alloy's friction laws: a box loaded below its friction limit creeps 5.5 mm
  where the law allows 0.01 mm (A5.onset), so it has no clean slip force either (A5.isotropy). The sliding law
  passes (stop distance within 0.61 %, bound 1 %).

## Why
Not found. The change is inside Steel's 46 commits between the two copies and has not been bisected.
- It is not the contacts: fed the same contacts, the two copies still differ, and both import the same model (the
  same margins and gaps).
- It is not a full buffer: the newer Steel builds up to about 2,000 articulated constraint rows per world where the
  older one stays under 1,024, but 2,048 rows give the same rate (31.8 %).

## Workaround
- Keep the default, `pile_backend="mujoco"`. Robots always step on MuJoCo. `"kamino"` is the other pile entry that
  ships registered.
- If you register Steel yourself, use it for free bodies only, and read `sim.check()` after every run: a part that
  passed through another shows as PEN_FAIL, a full Steel buffer as ROW_OVERFLOW.
- Tuning does not recover the newer copy: Steel's own baumgarte and twice the iterations took the older copy from
  6.4 % to 0.59 %, and left the newer one at 32 %.

## Planned fix
None on our side: no Alloy route steps Steel, and the cause is in the solver. We measure the same rows again at the
next Newton update. Steel becomes a registered entry only when it keeps this scene under 0.1 % and passes the three
friction laws.
