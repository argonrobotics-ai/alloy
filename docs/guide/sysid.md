# System identification: make the simulated robot move like yours

`alloy.sysid` (optional) fits the joint parameters of a simulated robot to a recording of your real one. You play a
trajectory on the robot, log the commanded joint targets and the measured joint positions, and run one command; the
fit finds the servo stiffness, damping, dry friction and action delay that make the simulated joints track the
way yours did. `import alloy` does not load it, and a scene without `sysid=` builds and steps exactly as before.

```bash
python -m alloy.sysid excite --robot panda --out excite.csv        # 1. a trajectory to play on your robot
python -m alloy.sysid fit --robot panda --data take.csv --out my_panda.sysid.json --report sysid_report   # 2. fit
python -m alloy.sysid report my_panda.sysid.json                   # 3. read it again later
# 4. scene.add_robot("panda", path, sysid="my_panda.sysid.json")   # the simulated robot now uses the fit
```

## Try it without a robot

`demo` simulates the robot with hidden parameters (servo stiffness x 0.7, kd and joint damping x 1.4, dry friction of
2 % of the joint's torque, a 12 ms action delay) and sensor noise, and writes that as a recording. The fit should find
the hidden values again. `--robot` takes a shipped robot's name or a robot file (MJCF, URDF, USD).

```bash
python -m alloy.sysid demo --robot panda --out take.csv            # prints the hidden values the fit should find
python -m alloy.sysid fit --robot panda --data take.csv --out demo_panda.sysid.json
python -m alloy.sysid report demo_panda.sysid.json
```

The runnable version, on a small 2-joint arm defined in the script, is `examples/sysid/01_fit_synthetic.py`. This
excerpt is from a short run of that arm (16 worlds, 6 generations, CPU; the default settings fit closer):

```text
Tracking error (RMS; degrees for hinges, metres for slides) and fitted values (file's value in brackets):
joint      before   after          kp         kd    damping   friction   delay ms
shoulder    1.381  0.1628  158.6 [200]  15.69 [10]  3.139 [2]  0.992 [0.1]      13.41
elbow       1.739  0.1592  73.18 [80]  7.182 [4]  1.796 [1]  0.369 [0]        15.14
Every fitted parameter is determined by this recording (1-sigma under 8 % of its search range).
```

## 1. Record

`excite` writes the commanded targets to play on your robot. Alloy never talks to hardware: you play the file with your
own controller at the file's rate and log `t`, the commanded targets and the measured joint positions.

- One joint moves at a time by default (`--mode together` moves all at once, each on its own frequency band, for a
  shorter recording). Each joint gets a chirp from 0.2 to 2 Hz of +-0.15 rad (`--amplitude`) around a centre pose
  inside its limits, then two quick small moves. 8 s per joint (`--seconds`) at 100 Hz (`--rate`); the first and the
  last second are at rest. `--joints a,b` limits it to some joints.
- Bring the robot to the first row's pose before you play the file.
- **Safety is yours.** Alloy knows the joint limits in the robot file, not your cell, your payload or your cables. Play
  the trajectory at low speed first and keep the emergency stop in reach.
- The fit simulates the robot alone on a stand. Loose objects in the robot file are left out, and a free-base robot
  (the Go2) is welded 0.5 m above its place in the file: record it hung with its feet off the ground and nothing in
  reach of the robot.

Input formats (joint names are the robot file's, without the robot's name):

| Format | Layout | Needs |
|---|---|---|
| CSV | columns `t`, `cmd.<joint>`, `pos.<joint>`; optional `vel.<joint>`, `tau.<joint>`, `episode` | nothing |
| HDF5 (`.h5`) | datasets `t`, `command`, `position`, attribute `joints` | `pip install h5py` |
| LeRobot folder | `action` = absolute joint targets, `observation.state` = joint positions | `pip install pyarrow` |

- HDF5: the datasets sit at the top of the file or one group per episode. LeRobot: `--column-names` names the action
  columns and `--state-columns` picks the joint positions out of a longer state.
- Units are radians (metres for a slide) and seconds. A file in degrees or milliseconds is refused with a message that
  names `--angle-unit deg` or `--time-unit ms`. Limit: degrees that happen to fit inside the joint's range read as
  radians (very small motions) cannot be detected.
- The sample period is the replay's control period, held within 2 to 20 ms: record at 50 to 500 Hz.
- Velocity and torque columns are read but not used by the fit yet.

## 2. Fit

```bash
python -m alloy.sysid fit --robot panda --data take.csv --out my_panda.sysid.json --report sysid_report
```

```py
import alloy.sysid

result = alloy.sysid.fit("panda", "take.csv", worlds=64, generations=30)       # also a Recording
result.save("my_panda.sysid.json")
print(alloy.sysid.report(result, folder="sysid_report"))      # text; plots too (needs matplotlib)
```

| Option (Python name) | Default | Meaning |
|---|---|---|
| `--robot` (`robot`) | required | shipped robot's name or a robot file |
| `--data` (`data`) | required | CSV, HDF5 or LeRobot dataset folder (or a `Recording`) |
| `--out` | `robot.sysid.json` | the result file |
| `--fit` (`parameters`) | `kp,kd,damping,friction,delay` | add `armature`, `torque_limit`; or `gains`, `all` |
| `--joints` (`joints`) | every servoed joint | comma-separated joint names |
| `--worlds` (`worlds`) | 64 | candidates per generation, at least 16 |
| `--generations` (`generations`) | 30 | search generations |
| `--seed` (`seed`) | 0 | the same seed gives the same file on the CPU |
| `--angle-unit`, `--time-unit` | `rad`, `s` | units of the recording (`deg`, `ms`) |
| `--max-delay-ms` | 50 | upper bound of the delay search |
| `--device` (`device`) | a GPU if present | `cpu` or `cuda:0` |
| `--report DIR` | off | also write `report.txt` and one overlay plot per joint |

What is fitted, per joint:

- Servo stiffness `kp`; `kd` and joint damping as one sum; dry friction (MJCF `frictionloss`); action delay. Both
  `kd` and damping are a torque against joint velocity on a position servo, so a recording determines only their sum.
  It is split in the robot file's ratio (all to damping when the file has neither).
- With `--fit`: `armature` and `torque_limit`.

How: the recorded commands are replayed in `worlds` parallel worlds, each holding one candidate set of parameters per
joint. Each joint has its own CMA-ES search, scored by its own tracking error, and all searches share the same
worlds. The result is deterministic for a given seed on the CPU.

Cost: one generation replays the whole recording once in every world, so the time grows with the recording's length
times `worlds` times `generations`. Use a GPU for a real robot.

Every accuracy number on this page is from a synthetic recording made by the same model that fits it (the simulated
robot with hidden parameters, a 12 ms delay and 0.2 mrad sensor noise). None is measured on a real robot: they show
that the search finds the answer when the model is right, not how well the model matches your hardware. Each row was
measured once:

| Robot | Recording | Settings, machine, time | Result |
|---|---|---|---|
| Panda, 8 servoed joints | 8 s, joints together | 256 worlds x 30 generations, one L4, 7.1 min | Synthetic recording made by the same model, not measured on a real robot. 7 arm joints: error 1.3-1.5 deg -> 0.011-0.012 deg RMS; stiffness and kd + damping within 9 %, dry friction within 23 % (0.06 N·m), delay within 0.1 ms. The finger's delay came back 6.6 ms and was listed as poorly determined |
| Go2, 12 joints, hung | 6 s, joints together | 256 x 30, one L4, 2.7 min | Synthetic recording made by the same model, not measured on a real robot. Stiffness, kd + damping and dry friction within 0.3 %, delay within 0.1 ms, nothing listed as poorly determined |
| Panda, 7 arm joints | 6 s, joints together | 256 x 30, one L4, 4.2 min | Synthetic recording made by the same model, not measured on a real robot. Stiffness and kd + damping within 7 %, dry friction within 10 %, delay within 0.1 ms |
| 2-joint test arm | 6 s, joints together | 32 x 25, 4-core CPU, 6 min | Synthetic recording made by the same model, not measured on a real robot. Error 0.74 deg -> 0.011 deg RMS; stiffness, kd + damping, dry friction, delay (12 and 20 ms) within 4 % |

What the tests hold is looser than these single runs: on such a synthetic recording (6 s, 256 x 30 on a GPU for the
Panda's 7 arm joints and the Go2's 12; 32 x 30 on the CPU for the 2-joint arm) each recovered stiffness is within 12 %
of the hidden value, kd + damping within 15 %, dry friction within 30 %, and the delay within 3 ms. Count on those
bounds, not on the rows above, and on neither for a real robot.

A robot built from a file that holds the true values reproduces the recording's positions to 1e-6 rad (a test holds
this on both CPU routes). On the CPU the fit runs in one process on the default route; the Panda there is far slower
than on a GPU (not measured to the end).

## 3. Read the report

`report` prints, per joint, the RMS tracking error before (robot file) and after (fit) in degrees (metres for a
slide), and the fitted values beside the robot file's.

- **POORLY DETERMINED** lists the parameters the recording does not pin down: a 1-sigma interval wider than 8 % of the
  parameter's search range, or a joint that never moved more than 0.02 rad. The value is kept in the file, but do not
  trust it. Record more motion of that joint (faster and slower moves, both directions) and fit again.
- If the search finds nothing better than the robot file's values for a joint, they are kept and a note says so; a
  note also names a value that ended at a bound of its search, where the true value may lie outside.
- The flags do not catch a search that stopped too early. With too few worlds or generations a fit can return wrong
  values with no flag: a Panda fit of 3 joints at 32 worlds x 20 generations (synthetic recordings) came back 5-70 %
  off on stiffness and several times off on dry friction, one joint 46 % off on stiffness with nothing listed. Use
  `--worlds 256` with 30 generations for an arm (the rows in section 2; the default 64 worlds is not checked on a
  robot of that size), look at the error after the fit (it should be near your
  encoder noise) and at the overlay plots, and fit again with another `--seed`: values that move between seeds are
  not converged.
- Small dry-friction values can be 30-40 % off without a flag even at 256 worlds x 30 generations: in one run on a
  synthetic Panda recording with other hidden values, stiffness and kd + damping came back within 3.6 % and the delay
  within 0.1 ms, but three joints' dry friction was 30-39 % off and none was listed. Check friction against the
  report's tracking error, not the flags.
- Overlay plots, one PNG per joint (measured, simulated with the robot file's values, simulated with the fit), are
  written only by `fit --report DIR` or `report(result, folder=...)` on the result of `fit` (the curves are not kept
  in the file), and need `pip install matplotlib`. Without it the text report still prints.
- The report ends with `scene.randomize` lines to paste (next sections).

## 4. Apply

```py
import alloy

scene = alloy.Scene(dt=0.01)
scene.add_robot("panda", "panda.xml", sysid="my_panda.sysid.json")    # a path, a dict or the result object
sim = alloy.build(scene)
```

- The values are written into every world at build, before `scene.randomize` rows are drawn.
- The delay holds each joint's commanded target back by the fitted time, whatever wrote the target:
  `sim.set_targets` and `sim.run(targets=...)` (both tested), and by construction anything else that writes the
  control arrays before a step, such as a Gym env's actions (not tested there yet). It costs two host array copies
  per control period for a robot with a delay; `sim.reset()` clears the delayed history.
- `scene.to_json()` keeps the values, so no file is needed later. `scene.save()` (cards) does not keep them and
  warns: pass `sysid=` again on a scene loaded from cards. A file can be written by hand: it needs
  `"schema": "alloy.sysid/1"` and a `"joints"` block of `{joint: {kp, kd, damping, friction, armature, torque_limit,
  delay}}` in SI units (delay in seconds), and only the values you list change.
- `alloy.build(..., rerun_worlds=n)` refuses a scene with `sysid`. Not checked with `processes=N`.

## 5. Randomize around the fit

Real robots differ from unit to unit, so train on a spread around the fit. The report suggests lines such as:

```py
scene.randomize("panda", "kp_scale", 0.88, 1.12)
scene.randomize("panda", "kd_scale", 0.87, 1.13)
```

The spread is 3 sigma of the least certain joint among those the recording determines, held within +-5 % and
+-50 %. Because the fitted values go in
first, these rows scale the fitted values. Joint damping, dry friction and delay cannot be randomized per world yet:
see [Randomizing robots](randomize_robots.md).

## What is NOT covered

- Not fitted in this version: a mass scale per link, backlash, joint flexibility, contact or payload parameters,
  sensor offsets (an offset between your zero angles and the file's must be removed before the fit).
- The fit has not been validated on hardware. No real-robot data was used: every check is against a synthetic
  recording made by the same model that fits it, which tests the search, not how well the model matches a robot.
- The report has no check that the search converged (section 3).
- Only joint tracking is fitted: not a gripper's grasp, not forces, not what a robot does in contact. Velocity and
  torque columns are not used yet.
- Delta or end-effector actions cannot be used (a LeRobot dataset must hold absolute joint targets), and `alloy.il`
  recordings hold delta actions unless the env's action mode is `"absolute"`.
- A model error that is not noise (backlash, a flexible link) makes the report's intervals too narrow.
- Degrees that fit inside the joint's range in radians cannot be detected (see Record).

## Errors

| Message | Fix |
|---|---|
| `joint 'shoulder': measured positions span [-6.148, 9.497], outside its range [-2, 2] rad; ...` | `--angle-unit deg` |
| `the median sample period is 10 s, too slow for a joint recording in seconds` | the time is in ms: `--time-unit ms` |
| `the recording has no joint 'wrist'` | the message lists the recording's joints; use those names in `--joints` |
| `no joint moves in this recording (every command and position spans under 0.02 rad)` | play the `excite` file |
| `'mass' is not a parameter the fit knows` | the message lists the names (the `--fit` row above) |
| `fit needs worlds >= 16 and generations >= 1` | use at least 16 worlds |
| `robot 'x' is neither a file nor one shipped robot` | pass a robot file or a name the message lists |
| `reading an HDF5 recording needs the h5py package, which is not installed` | `pip install h5py` |
| `reading a LeRobot dataset needs the pyarrow package, which is not installed` | `pip install pyarrow` |
| `overlay plots needs the matplotlib package, which is not installed` (a note) | `pip install matplotlib` |
| `BuildError: ... its sysid file names joint 'wrist', which is not a position-servoed joint ...` | fit with it |
| `BuildError: rerun_worlds cannot be used with add_robot(sysid=) ...` | build with `rerun_worlds=0` |
| `sysid file 'x.json' does not exist` | pass the file written by `fit --out` |

## See also

- [Randomizing robots](randomize_robots.md): `kp_scale`, `kd_scale` and the other per-world robot fields.
- [Robots](robots.md): loading robots, joint names, targets and servos.
- [Imitation learning](imitation.md): recording demonstrations (a dataset of absolute joint targets can be fitted).
- `examples/sysid/01_fit_synthetic.py`: the runnable fit on a small arm.
