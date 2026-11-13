# How do I let Claude Code, Codex or Cursor build a robot simulation?

Give your coding agent the prompt on the [agents page](../agents.md), which tells it to install `alloy-engine`, read
the docs and start from a ready-made environment. The Alloy physics engine installs with `pip install alloy-engine`,
and its docs, examples and error messages are written so an agent can read and act on them.

What the agent can run to find the docs:

```bash
python -m alloy.docs          # prints three paths: the docs, the examples and AGENTS.md
python -m alloy.docs agents   # prints AGENTS.md, a quick start written for agents
```

To give the agent Alloy's operations as tools instead, register the MCP server (shown for Claude Code):

```bash
claude mcp add alloy -- alloy mcp
```

The loop an agent relies on is to read an error's `message` and `fix`, apply the fix, and read `sim.check()` after
every run.

```python
import alloy

scene = alloy.Scene(dt=0.01)
scene.add_ground()
scene.add_box("block", size=(0.05, 0.05, 0.05), pos=(0, 0, 0.1))
try:
    scene.add_box("thin", size=(0.05, 0.05))
except alloy.AlloyAPIError as e:
    print(e.message)
    print(e.fix)
sim = alloy.build(scene)
sim.run(0.5)
print(sim.check()["ok"], round(float(sim.state(world=0)["block.pos"][2]), 3))
```

```text
size needs 3 finite numbers, got (0.05, 0.05)
pass size=(..., ..., ...) in m
True 0.025
```

The mistake raised `alloy.SceneError`, a kind of `alloy.AlloyAPIError`, and the fix names what to pass. After the
fix the block fell to the ground and rests with its centre at 0.025 m, and `sim.check()` returned `True`.

The whole program ran in about 11 s on an AMD EPYC 7B13 CPU with no GPU, including imports and the build, after
the first compile.

## Keep in mind

- A frozen world looks calm. Tell the agent to print `sim.check()["ok"]`, or it can report a failed run as a success
  ([World health](../guides/health.md)).
- The first run of each kind of scene compiles for minutes. Tell the agent not to stop it.
- No shipped trainer learns from camera images: on a camera environment the command trains a state-based policy
  ([Train with RL](../guides/rl.md)).
- Full list: [Keep in mind](../keep-in-mind.md).

## Next

- [Errors and fixes](../errors.md)
- [Quickstart](../quickstart.md)
- [Train with RL](../guides/rl.md)
- Related: [the MCP server](../answers/mcp-server-physics-engine.md),
  [error messages and fixes](../answers/error-messages.md).
