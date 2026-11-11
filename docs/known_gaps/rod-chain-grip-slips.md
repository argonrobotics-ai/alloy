# A cable on the chain route creeps back in a held twisting grip on the CPU

## What is wrong
A short elastic rod with no clamp routes to the chain: a string of capsules in MuJoCo, stepped with the robot. Since
the chain-grip rule (`alloy.contact.chain_grip`), a chain beside a moving body (a robot, or a dynamic or kinematic
object) follows a gripper that squeezes and twists it, on the CPU and on the GPU. A 4 N squeeze sinks the pads 0.1 mm
into a 3 mm cable, not 1.5 mm. Before the rule it followed half the twist and lost most of that within a second.
- On the CPU (MuJoCo C), while the gripper holds still, the cable slowly turns back against its own torque: about
  0.14 % per second.
- On a GPU (MuJoCo Warp) the held cable stays with the grip since 2026-11-10 (the stick anchor,
  `alloy.contact.mjw_stick_anchor`; before it, it crept 2 % per second, and 10 % in 0.7 s in a rigid grip). What is
  left there is under "What the anchor does not hold".
`sim.routes()["<rod>"]["reasons"]` says when the rule covers a cable ("chain grip beside ...").

## Number
The rig of `tests/api/test_rods_grip.py` on the chain:
- a 0.4 m USB cable (r 3 mm, 20 segments of about 1 g), its start welded;
- its end between two 0.2 kg sliding pads on a wrist, squeezing 4.2 N;
- the wrist rolls 0.3 rad about the cable over 0.5 s, then holds.

The table gives the cable end / wrist. Measured with development scripts (a probe that applies the candidate rule as a
patch, its GPU runs and a chain-clamp script; not shipped).

| route | ramp's end | after 0.7 s held | after 5 s held | pad depth |
|---|---|---|---|---|
| CPU, MuJoCo C | 1.0000 | 0.9997 | 0.9931 | 0.10 mm |
| GPU, MuJoCo Warp (an L4) | 1.0000 | 1.0000 | 1.0000 | 0.10 mm |
| GPU, MuJoCo Warp before the stick anchor | 0.9976 | 0.9828 | 0.8651 | 0.10 mm |
| either route before the rule | 0.50 | 0.08 | - | 1.54 mm |
| the same rig on `rod_vbd` | 1.01 | 1.02 | - | - |

- The held cable creeps about 0.14 % per second on MuJoCo C (8 % over a minute). On MuJoCo Warp it does not move in
  the fourth decimal over 5 s, the same in every world at 64 and 1,024 worlds (one L4, 2026-11-10); without the
  anchor (`ALLOY_STICK_ANCHOR=0`) it creeps about 2 % per second (0.9976 -> 0.9828 over 0.7 s, 0.8651 after 5 s).
- The creep falls faster than 1 / impratio on MuJoCo C: 0.69 / 0.08 / 0.01 % over 5 s at impratio 300 / 1,000 /
  3,000. MuJoCo Warp runs the rule at impratio 30, not 300 (Why).
- A rigid grip (pads welded to the wrist, 0.1 mm into the cable; WP9's GPU chain rig, not shipped) on
  MuJoCo Warp: 0.9974 at the ramp's end and 0.9976 after 0.7 s, 2 s and 5 s, the same at 1, 64 and 1,024 worlds.
  Without the anchor it followed 0.978 and ran on to 0.899 / 0.762 / 0.523 after 0.7 / 2 / 5 s. On MuJoCo C the
  welded rig holds: 1.0000, -0.03 %. Fingers held by position servos, between the two, are not measured.
- The anchor adds no torque of its own. The wrist's torque over the cable's G J theta / L (the law of
  `tests/api/test_rods_chain_clamp.py`, 3 % on MuJoCo C, where it reads 0.997) on the CPU's MuJoCo Warp route, held
  2 s: welded pads 1.04 with the anchor (1.044 after 0.7 s, 1.039 after 2 s), against 1.14 rising to 1.30 without
  it as the cable ran on; sliding pads 0.90 with it and 0.92-0.94 without (MuJoCo C reads 0.89 there: the pads also
  carry the sagging end's weight). The welded rig is 4 % over on MuJoCo Warp, outside the 3 % of the MuJoCo C law.
- The anchor's own laws (`tests/contact/test_mjw_stick_anchor.py`): a cube pushed at 0.3 of its friction limit
  moves 3.02 um in 2 s without the anchor and 0.0001 um with it; pushed past the limit it slides and coasts the
  same with and without (0.2723 m in 0.6 s, deceleration 0.96 mu g).

## What the anchor does not hold (MuJoCo Warp)
- A pair lets go when the pull it would need passes its friction limit (the contact's cone, or 20 m/s2 if that is
  lower). It sticks again only once the rows' own demand has fallen under half of that limit. So a contact that has
  slid and still carries more than half of its friction limit keeps MuJoCo's rows and creeps as before.
- A pair that has no contact for one step loses its anchor and starts afresh when it touches again.
- A cable link anchors up to 4 partners at a time; a fifth keeps MuJoCo's rows.
- Contacts between two links of the cable are not anchored, and neither are contacts of any other body: the anchor
  is on the chain-grip links only. A scene without one is bit for bit unchanged.
- Elliptic friction cones only (what the chain-grip rule runs); MuJoCo C and rk4 entries keep MuJoCo's rows.
- A gripped cable's links are anchored against whatever they touch, a table or the floor too: that is static
  friction, and it is new for a draped cable under a small side load, which used to creep along the table.
- The anchor's pull is not booked separately in the health monitor's energy check. It is bounded by the friction
  cone times the slip a stuck pair holds (under a micrometre in the rows above).
- `ALLOY_STICK_ANCHOR=0` in the environment turns the anchor off (the rows of 2026-11-09).
- `sim.reset()` and a re-run of a period take each world's anchors back with the world.

## Why
MuJoCo makes every constraint soft, friction too. With soft friction rows, a steady load far below the friction limit
still creeps.
- On MuJoCo Warp the chain-grip links' contacts get a stick anchor: their two sliding rows also pull back on the slip
  the pair has built up since it last stuck, limited to the friction cone. Past the cone the pair lets go and its
  rows are MuJoCo's own again, bit for bit, until its sliding has died down (above). It is a MuJoCo Warp kernel:
  MuJoCo C keeps its rows, and so does rk4.
- On MuJoCo C the rule runs Newton at impratio 300, which makes the friction rows 300 times stiffer than the normal
  row. They are still not rigid.
- MuJoCo's no-slip pass, which would hold the cable exactly, does not run with the discrete integrator the chain needs
  for its stiff joints.
- The rule runs impratio 30 on MuJoCo Warp, which builds the Newton Hessian in float32. At 300 a gripped chain's
  comes out indefinite by rounding alone, and float32 Newton does not converge on a cable resting beside a robot: it
  ran at its 100-iteration cap in 97 % of periods, 18-25x the step on an L4, even with the Hessian damped. If the
  float32 factorization still fails at 30, `alloy.precision.mjw_chol` redoes it in float64 with its pivots floored at
  the Hessian's float32 rounding.

## Workaround
- On a GPU nothing is needed for a held grip. On the CPU, a long hold that must not creep: put the cable on the
  elastic rod. Any `clamp=` (`"start"` or `"end"`) routes it to
  `rod_vbd`, and so does a material that yields (`plastic.model: j2`); `sim.routes()` confirms the route. `segments=`
  does not change the route. On `rod_vbd` it follows the gripper within 0.05 % and holds after the gripper stops.
  Beside a robot it costs much more ([rod-vbd-beside-robot-cost.md](rod-vbd-beside-robot-cost.md)).
  `add_rod(..., clamp="start", route="chain")` is the exception: it keeps the cable on the chain with its first link
  welded, which is cheap beside a robot, and keeps the CPU's creep (the table's rows are that rig).
- Read the cable's own twist (`"<rod>.twist"`, its segments' quaternions), not the gripper's angle.

## Cost
- CPU, MuJoCo C: +33 % per simulated second with a gripped cable (WP9's api_speed robot scene, 1 world: 0.947 ->
  1.257 s). The Panda scene with a cable lying on its table showed no measurable cost (1.53 / 1.47 s).
- GPU, MuJoCo Warp, an L4, the same scene: 1.62x / 1.14x / 1.10x at 64 / 1,024 / 4,096 worlds (1.17 -> 1.89,
  5.32 -> 6.05, 21.2 -> 23.4 s per simulated second). The stick anchor adds 4 % / 3 % / 0 % at 1 / 64 / 1,024
  worlds on the gripper rig.

## Planned fix
None scheduled. Candidates:
- a contact that slid and is still loaded past half its friction limit on MuJoCo Warp: the half is what keeps a
  sliding contact Coulomb's; sticking again right up to the cone needs a test on the rows' own state;
- the MuJoCo C creep: a higher impratio (1,000 creeps ten times less at about the same cost).
