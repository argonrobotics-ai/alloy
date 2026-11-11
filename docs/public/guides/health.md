# World health

Check that a run is physically sound, and see what a failed world looks like.

## 1. A healthy run

```python
import alloy

def drop_block(z):
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_box("block", size=(0.1, 0.1, 0.1), pos=(0, 0, z), mass=0.2)
    sim = alloy.build(scene)
    sim.run(0.5)
    return sim

def report(sim):
    world = sim.health()["worlds"][0]
    flags = [f for f in world["flags"] if f != "DET_DOWNGRADED"]     # a notice every GPU run lists
    print(sim.check()["ok"], world["severity"], flags)

sim = drop_block(0.05)               # a 10 cm cube resting on the ground
report(sim)
```

```text
True OK []
```

`sim.check()["ok"]` is the one value to read after every run. `sim.health()` has one row per world with its
`severity` (`OK`, `DEGRADED` or `INVALID`) and the `flags` that fired.

## 2. A run that goes wrong

```python
bad = drop_block(0.0)                # the same cube, centred on the ground: half of it starts inside
report(bad)
print(bad.check()["problems"][0]["what"])
```

```text
False INVALID ['PEN_WARN', 'PEN_FAIL', 'ENERGY_JUMP', 'LATCHED', 'START_PEN']
ground and block overlapped by 47.91 mm at the start
```

Each entry in `sim.check()["problems"]` names the cause and ends in a `fix`. Here the fix is to start the cube above
the ground.

## 3. An INVALID world looks calm

```python
bad.step(20)
print(bad.state(["block"])["block.pos"].round(2), bad.check()["ok"])
```

```text
[[0. 0. 0.]] False
```

An INVALID world is frozen at its last good state and the other worlds go on. Its position is perfectly still and
shows no sign of trouble, so always read `sim.check()["ok"]` before you trust a result.

## Flags

Read `sim.health()["worlds"][w]["flags"]`; `fixes` in the same row lists the change for each flag that fired.

| Flag | Level | Meaning | What to change |
|---|---|---|---|
| NONFINITE_* | INVALID | a position, velocity or particle is NaN or inf | lower `dt`; realistic masses; no overlaps |
| PEN_WARN | DEGRADED | a contact over 1 mm deep, or past half a thin part | lower `dt`, or `profile="precision"` |
| PEN_FAIL | INVALID | past 1 cm or half a thin part for 0.1 s, or twice that | start apart, lower `dt` |
| PAIR_PEN_WARN | DEGRADED | a precision fit is near its clearance limit | lower `dt`, slow the insertion |
| PAIR_PEN_FAIL | INVALID | a precision fit is past its clearance limit | lower `dt`, slow the insertion |
| SPEED_FAIL | INVALID | something moved faster than 20 m/s | no overlaps or huge forces; lower gains |
| ENERGY_JUMP | DEGRADED, INVALID | energy rose from nothing (INVALID past 50 %) | lower `dt`; no initial overlaps |
| JOINT_FAIL | DEGRADED | a joint or weld is over 1 mm or 1 mrad apart | lower load, gains or `dt`; check limits |
| CONTACT_OVERFLOW | DEGRADED | more contacts than the buffer holds | `alloy.build(scene, contact_capacity=n)` |
| ROW_OVERFLOW | DEGRADED | more constraint rows than the buffer holds | larger `contact_capacity`; fewer contacts |
| UNCONVERGED | DEGRADED | the solver did not converge | lower `dt`, or `profile="precision"` |
| START_PEN | check fails | objects overlap by over 1 cm at the start | move apart; `sim.settle` does not help |
| SPIN | DEGRADED | a body turned over 0.5 rad in one step | heavier or thicker part; `scene.rigid_step = 0.000625` |
| CLOTH_STRETCH | DEGRADED, INVALID | cloth 50 % past rest (INVALID: 500 %) | lower `dt`; more `mass=`, `thickness=` |
| SELF_CONTACT_OVERFLOW | DEGRADED | cloth self-contacts exceed the buffer | fewer cloth layers |
| FAST_CONTACT | DEGRADED | a fast body may pass a thin static part | finer `scene.rigid_step` or `dt`; thicker part |
| TUNNEL | INVALID | a body passed through a static part | a finer `scene.rigid_step` or `dt`; slower body |
| QUASI_STATIC | DEGRADED | a mass-scaled solid moved too fast | press slower; declare the real `tool_speed` |
| STRAIN_RANGE | DEGRADED | a soft body strained over 60 % | press with an edge, or less force |
| RUNG_PINNED | DEGRADED | a solver wanted a smaller step than pinned | lower `dt` |
| FAR_FROM_ORIGIN | notice | an object is over 4 m out (INVALID at 16 m, precision) | check it is supported |
| FALLBACK_USED | notice | the engine used a fallback path | nothing, unless results look wrong |
| DET_DOWNGRADED | notice | GPU runs do not repeat bit for bit | nothing; use the CPU for the same bits |
| LATCHED | INVALID | the world was frozen at its last good state | fix the first problem listed |

## Limits

- Health does not catch every overlap between rigid bodies and robot parts, and it freezes a fallen legged robot
  that lies deeper than 1 cm ([what it misses](../../known_gaps/health-mujoco-penetration.md)).
- A fast body can pass through a thin static part before any flag fires
  ([tunnelling](../../known_gaps/rigid-tunnelling.md)).
- A GPU run does not repeat bit for bit once bodies touch in more than one place
  ([GPU determinism](../../known_gaps/determinism-gpu-rigid.md)).

## Next

- Add a robot: [Robots](robots.md).
- Known limits: [Limits](../limits.md).
