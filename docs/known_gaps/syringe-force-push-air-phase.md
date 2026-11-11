# Known gap: a syringe pushed by a constant force rebounds after its air leaves

## What
- The `syringe` plug-in (`alloy.plugins.library.syringe`) is stable only because the plunger's joint carries the
  needle's liquid column as armature: the column's added mass ρ·L·A²/a, about 1 kg for a 3 mL barrel on a 25 G x 16 mm
  needle (`syringe.inertance`).
- That mass is real while liquid flows through the needle. It is not real while only air leaves, or while the
  syringe is capped, but the plunger carries it then too.
- A plunger pushed by a constant force (the `push` input with nothing else on it) therefore gathers momentum while the
  air leaves almost unresisted. It rams the liquid, rebounds below atmospheric pressure and draws air back in through
  the needle.
- A plunger moved by a finger at a set speed (a robot's position or velocity drive), as in the syringe tasks, is not
  affected: laws G-1..G-6 run that way.

## Number
3 mL barrel (8.59 mm bore), 2.8 mL of water and 0.3 mL of air, nozzle up, 25 G x 16 mm needle open to the air,
armature 1.01 kg, a 2 N push, CPU, 1.25 ms step:
- the plunger reaches 0.11 m/s in 60 ms. A 2 g plunger alone would settle at about 0.23 m/s within milliseconds,
  braked by the air through the needle (estimated: 2 N over the air's A²/G ≈ 8.7 N·s/m), and stop on the liquid;
- after the air is out, the barrel pressure swings between 50 and 205 kPa (absolute), and the needle carries air
  back in while it is below 101 kPa.

## Why
- A plunger force computed from the fluid is explicit. Without the armature a 2 g plunger behind a 25 G needle is
  unstable above about 8 µs, and a capped all-liquid barrel above about 56 µs.
- The armature is set once, in the model, so it cannot follow the phase in the needle. Changing the body's mass at
  run time instead would keep the velocity across a 500x jump and inject energy.

## Workaround
- Drive the plunger by its joint: a position-and-speed drive on the plunger's prismatic joint
  (`JointTargetMode.POSITION_VELOCITY`, as `alloy/tasks/syringe_rig.py` does), not by a bare force.
- Not by a finger or gripper pushing the flange through contact with clearance either: the column's armature
  (1 kg on a 3 mL barrel behind 25 G) rattles in the play and on the contact spring, and the barrel pressure swings
  about ±50 % at 1 mm/s (the syringe tasks' first rig). A robot hand on the plunger needs a grip with no play, or the
  joint drive above.
- With both gains set, Newton's default drive mode is POSITION, which damps toward zero speed and ignores
  `joint_target_qd`: pass `actuator_mode=newton.JointTargetMode.POSITION_VELOCITY` to make the drive track a speed.
- If a force push is needed, push slowly or add viscous stopper friction (`Syringe.viscous`) so the plunger stays near
  its terminal speed through the air phase.

## Planned fix
An implicit joint damper set per world by the plug-in (the needle's A²/G only while liquid flows), so the armature can
drop to the plunger's own mass. That needs a plug-in hook to the solvers' per-world joint damping on both CPU and GPU
routes (WP8; the syringe tasks are ported, merged 43f77f638).
