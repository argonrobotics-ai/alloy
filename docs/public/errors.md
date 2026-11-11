# Errors and fixes

Every error Alloy raises says what went wrong and what to change. Read `e.message` and `e.fix`, apply the fix, run
again.

## Read an error

```python
import alloy

scene = alloy.Scene(dt=0.01)
scene.add_ground()
scene.add_box("block", size=(0.05, 0.05, 0.05), pos=(0, 0, 0.1))
try:
    scene.add_box("thin", size=(0.05, 0.05))
except alloy.SceneError as e:
    print(e.message, "|", e.fix, "|", e.field)
```

```text
size needs 3 finite numbers, got (0.05, 0.05) | pass size=(..., ..., ...) in m | size
```

`e.field` names the argument at fault. `e.to_dict()` gives the same as a dict, ready for JSON.

## A wrong name lists the right ones

```python
sim = alloy.build(scene)
try:
    sim.state(["blok"])
except alloy.UnknownNameError as e:
    print(e.message)
    print(e.fix)
    print(sorted(e.to_dict()))
```

```text
no body, joint or particle object named 'blok'
use a name from sim.names() (bodies: block; joints: -; particles: -; static: ground)
['error', 'field', 'fix', 'message']
```

## The error classes

All of them derive from `alloy.AlloyAPIError`, so one `except` catches every one.

| Class | When |
|---|---|
| `alloy.SceneError` | a bad argument, name or card while you build the scene |
| `alloy.BuildError` | the scene cannot be built: a device that is not there, a combination Alloy cannot run |
| `alloy.SimulationError` | stepping failed, or a world went INVALID under `profile="precision"` |
| `alloy.UnknownNameError` | a name that is not in the scene; the fix lists the valid names |
| `alloy.FeatureUnavailableError` | an optional package is missing; the fix is the `pip install` line |

A run can also go wrong without raising: a world that fails its checks is frozen and the others go on. Read
`sim.check()` after every run ([World health](guides/health.md)).

## Symptom and fix

| Symptom | Fix |
|---|---|
| `UnknownNameError` on a state key | print `sim.names()`; a robot's parts are `"<robot>/<body or joint>"` |
| An object sinks into another or jumps off it | a smaller `dt`; sizes are full edge lengths, not half |
| A fast small object passes through a thin wall | a smaller `dt`, a slower object or a thicker wall |
| An object sits on top of a cup made from a mesh | a mesh collides as its outer hull: use `add_open_box` |
| A run never finishes on the CPU | `sim.routes()`: a soft material made a soft body; pass `rigid=True` |
| A joint does not move | the target is outside its range, or the joint has no actuator |
| "more contacts than its buffer holds" | `alloy.build(scene, contact_capacity=n)` with the n it names |
| Liquid splashes over a container's rim | pour it with `add_pour` instead of dropping an `add_liquid` block |
| A grasped object slips | ramp the targets, raise the fingers' `kp`, raise the pair's friction |
| `sim.check()["ok"]` is False | read each problem's `"fix"`; most often a smaller `dt` |
| The first run takes minutes | a one-time compile, cached afterwards; time only later runs |

## Next

- [World health](guides/health.md): every health flag and its fix.
- [Known limits](limits.md): what does not work yet.
