# How can I tell whether a physics simulation blew up during RL training?

The Alloy physics engine (`pip install alloy-engine`) grades every world as OK, DEGRADED or INVALID, and freezes an
INVALID world at its last good state while the others go on. Read `sim.check()["ok"]` after a run and
`sim.health()` for the flags per world. In a Gym environment the same grade ends the episode and counts it.

```python
import numpy as np
import alloy

z = np.array([0.3, 0.0, 0.6])        # world 1 starts with its cube half inside the ground; the others fall
pos = np.stack([np.zeros(3), np.zeros(3), z], axis=1)
scene = alloy.Scene(worlds=3, dt=0.01)
scene.add_ground()
scene.add_box("block", size=(0.1, 0.1, 0.1), pos=pos, mass=0.2)
sim = alloy.build(scene)
sim.run(0.8)
for w, row in enumerate(sim.health()["worlds"]):
    flags = [f for f in row["flags"] if f != "DET_DOWNGRADED"]    # a notice every GPU run lists
    print(w, row["severity"], flags)
check = sim.check()
print(check["ok"])
print(sim.state(["block"])["block.pos"][:, 2].round(2))
for p in check["problems"]:
    print(p["world"], p["level"], p["check"])
```

```text
0 OK []
1 INVALID ['PEN_WARN', 'PEN_FAIL', 'ENERGY_JUMP', 'LATCHED', 'START_PEN']
2 OK []
False
[0.05 0.   0.05]
1 fail start_overlap
1 error health
1 error penetration
0 warn health
2 warn health
```

World 1 is INVALID and frozen at z = 0, which looks calm: `check()["ok"]` is the value to trust, and each problem
ends in a `fix`. Worlds 0 and 2 fell and came to rest at z = 0.05 m. Their `warn` rows record a flag that fired
during the landing and went away; those worlds stay OK.

In a training environment you do not reset by hand. A frozen world ends its episode, reads zero reward and
observation, and starts again; the environment counts it:

```python
import numpy as np
import alloy.envs

env = alloy.envs.make("GripperLift-v0", num_envs=4)
obs, info = env.reset(seed=0)
for _ in range(20):
    obs, reward, terminated, truncated, info = env.step(env.xp.asarray(np.zeros((4, 3), np.float32)))
print(np.asarray(info["invalid"]).tolist(), np.asarray(env.invalid_episodes).tolist())
```

```text
[False, False, False, False] [0, 0, 0, 0]
```

`info["invalid"]` marks the worlds frozen on this step and `env.invalid_episodes` counts frozen episodes per world
since the env was built. Watch that count during training ([Train with RL](../guides/rl.md)).

One measured number: a ten-block tower in 1,024 worlds on one NVIDIA L4 ran 5 simulated seconds with 0 of 1,024
worlds with an error (a block moved or sank more than 1 cm, or a non-finite value) at each of three accuracy
settings, in two runs (pre-release builds).

## Keep in mind

- Health does not catch every overlap between rigid bodies and robot parts
  ([what it misses](../../known_gaps/health-mujoco-penetration.md)).
- A fast body can pass through a thin static part before any flag fires
  ([tunnelling](../../known_gaps/rigid-tunnelling.md)).
- A GPU run does not repeat bit for bit once bodies touch in more than one place
  ([GPU determinism](../../known_gaps/determinism-gpu-rigid.md)).
- Full list: [Keep in mind](../keep-in-mind.md).

## Next

- Connect an agent to the engine: [An MCP server for agents](mcp-server-physics-engine.md)
- Many worlds on a GPU: [GPU-parallel engines for RL](gpu-parallel-engines-for-rl.md)
- Every flag and its fix: [World health](../guides/health.md).
