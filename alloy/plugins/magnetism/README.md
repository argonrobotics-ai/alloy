# Magnetism: the `Magnetism` plug-in, and `task_magnet_stirrer`

This closes the magnetism gap in the Gym-Anything comparison (`[internal note]` section 2:
"magnetic forces (stir bar, eraser)", 5 tasks). Every source is a point dipole, and every closed form is in
`field.py`.

| Source | What it does |
|---|---|
| body magnets (`bodies`, `moments`, `axes`, `offsets`) | dipole pairs push, pull and twist each other, as equal and opposite wrenches |
| drives (`drives`, `drive_moments`, `drive_axes`, `drive_bodies`, input `rpm`) | a motor magnet turning at `rpm`, per world; the phase is device state, safe under graph capture; the reaction goes to the support body or the world |
| plates (`plates`, `plate_half`, `plate_mu_r`) | a static iron half-space pulls a magnet through its image dipole, m' = κ(2(m·n)n - m) with κ = (μr - 1)/(μr + 1); one-sided and bounded |
| `external_field` | a uniform B: torque m × B |

```python
from alloy.plugins.magnetism import Magnetism
mag = Magnetism(bodies=("stir_bar",), moments=(0.0675,), axes=((1, 0, 0),),
                drives=((0, 0, -0.015),), drive_moments=(0.375,))
sim = alloy.plugins.simulation(built, [mag]); mag.set_input("rpm", 600.0)
```

The library `magnet` plug-in is the body-to-body part alone. Point dipoles hold at about one magnet length or more.
Closer, forces are evaluated at `min_distance` (an ESTIMATE). Spread a flat magnet over several `offsets`. Thin-sheet
saturation and eddy currents are not modelled.

## Laws (`laws.py`; `python -m alloy.plugins check magnetism`, CPU, 2026-11-06)

| Law | Closed form | Bound | Result |
|---|---|---|---|
| inverse_fourth | coaxial 3μ0m²/(2πd⁴) and side by side 3μ0m²/(4πd⁴), at 30 and 60 mm | 1e-5 | 1.2e-7 |
| energy | kinetic + dipole energy while two free magnets close | 0.5 % of the energy released | 0.2 % |
| field_oscillation | a bar in a uniform B oscillates at sqrt(mB/I) | 1 % | -0.06 % (2.2264 vs 2.2278 Hz) |
| stirrer_lag | steady lag asin(cω/(mB)) at ratios 0.5 and 0.8; slips at 1.3 | 2 % | 30.00 / 53.13 deg exact; slip at 0.26 ω |
| plate_force | normal 3μ0κm²/(32πh⁴), flat half that, at h and 2h (κ = 1/2); 0 behind | 1e-5 | 1.8e-7; 0 N |
| eraser_holds | μF > mg on a vertical board: the eraser stays | 1 mm in 0.3 s | 0.11 mm |
| eraser_falls | μF ≈ mg/2: the fall matches g - μF(h)/m over the recorded gap (0.156 ms step) | 5 % | 230 vs 228 mm (+1.0 %) |
| harness | crash, hook and run determinism, identical and isolated worlds, graph replay | | all pass |
| speed | CPU, 3 worlds | 2x | 1.10x |

## task_magnet_stirrer (`alloy/tasks/magnet_stirrer.py`)

The task matches Gym-Anything's titration "start-stirring".

**Scene:**
- a 25 mm PTFE bar on its pivot ring in a glass dish, 15 mm above a 0.375 A·m² drive magnet;
- water drag on the bar, k·ω² with k = ρ·C_d·w·L⁴ / 64 (an ESTIMATE that ignores the swirl).

**Script:** the drive ramps to a target rpm over 1 s and holds for 1.5 s.

**Success:** the bar spins within 5 % of the drive, within 5 mm of the axis, and on the floor.

**Predicted decoupling:** m·B = k·ω², which gives **1,081 rpm**.

Results on CPU, 2026-11-06:

| Target rpm | 200 | 400 | 600 | 800 | 1000 | 1200 | 1400 | 1600 |
|---|---|---|---|---|---|---|---|---|
| Success | yes | yes | yes | yes | yes | no | no | no |
| Spin / drive | 1.000 | 1.000 | 1.000 | 1.000 | 0.9995 | 0.002 | 0.002 | 0.002 |
| Off axis, max (mm) | 0.36 | 0.35 | 0.32 | 0.58 | 0.62 | 22.5 | 23.4 | 23.2 |

## Findings while building it

- **At the default step, MuJoCo tips a lightly pressed sliding eraser off the board.**
  - Corrected in review: the cause is the step size.
  - At a 1.25 ms step, an eraser held to a vertical whiteboard by its magnet tips while it slides (61°) and its gap
    opens to 30 mm. The pull's 1/h⁴ loss amplifies that.
  - It reproduces without the plug-in, under a constant press (14°, 11 mm off the board), and in MuJoCo C,
    so it is MuJoCo's model.
    - Statically it should not tip: the friction moment needs the normal-force centroid only 5 mm below centre on
      a 25 mm half-height face.
  - At a step ≤ 2.5e-4 (0.156 ms) the gap holds at 6.0-6.5 mm, the tilt is 1.2°, and the fall matches the closed
    form to 1 %. eraser_falls runs there.
  - With μ = 0 it holds at any step.
  - Wiping tasks need a small step, or a firm press.
- **A stiff contact is needed under a plate pull.**
  - The pull gets stiffer as the gap closes (4F/h). With Newton's default ke of 2.5e3 the eraser sank 5 mm into
    the board, and the force ran away (2 → 43 N).
  - Use ke ≥ 1e5 with kd = 2·sqrt(ke).
  - At a 1.25 ms step MuJoCo clamps the time constant to 2·dt, so ke 1e6 runs as about 2e5 there.
- **A lightly damped stir bar has two states.**
  - Like a damped pendulum under constant torque, it has a second, slipping state above cω/(mB) ≈ (4/π)·2ζ. Started
    from rest at full speed it may never lock in: ζ = 0.17 slipped at a ratio of 0.8.
  - Ramp the drive, as real stirrers do.
- **Stir bars need their pivot ring.**
  - A plain box bar spins on its corners. That contact chatters, and it threw a 600 rpm bar out of step while 800
    and 1,000 rpm ran.

GPU: not run yet. The task's CLI writes the cost row (`--device cuda:0 --graph`).
