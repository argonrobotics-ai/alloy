# Is there a physics engine with an MCP server for AI agents?

Yes. The Alloy physics engine, installed with `pip install alloy-engine`, serves its tools over the Model Context
Protocol: start `alloy mcp` and an agent can build scenes, step them and read state as JSON without writing Python.
This page connects Claude Code and lists the 25 tools from the installed package.

```bash
pip install alloy-engine
claude mcp add alloy -- alloy mcp
```

Any other MCP client starts the same command, `alloy mcp`, and talks to it over stdio. To see what the server offers
without a client, read the tool list from Python:

```python
import textwrap
from alloy.mcp.tools import tool_list

names = [tool["name"] for tool in tool_list()]
print(len(names))
print(textwrap.fill(", ".join(names), width=100))
```

```text
25
list, scene, build, step, run, state, targets, reset, drive, pose, wrench, settle, heater, controls,
close, check, render, estimate, import, export, vary, compose, create, audit, fetch
```

Each tool takes and returns JSON. `scene` checks a scene description without building it, `build` makes the
simulation, `step` and `run` advance it, `state` reads names and values back, and `check` returns the health result
that you should read after every run ([World health](../guides/health.md)). Ask the agent to report what `check`
said, not only what it ran.

One measured number: on an x86_64 CPU, two builds with the same seed give the same trajectory bit for bit, and a world
gives the same result alone or among 15 others (one contact scene, pre-release build); GPU runs with contact do not
repeat bit for bit.

## Keep in mind

- GPU runs with contact do not repeat bit for bit, so an agent should judge a GPU result by a rate over many worlds
  ([GPU determinism](../../known_gaps/determinism-gpu-rigid.md)).
- Health does not catch every overlap between rigid bodies and robot parts
  ([what it misses](../../known_gaps/health-mujoco-penetration.md)).
- Full list: [Keep in mind](../keep-in-mind.md).

## Next

- Read a failed run: [Did my simulation blow up?](did-my-simulation-blow-up.md)
- Mixed scenes from one description: [Robots, cloth and liquids in one scene](robots-cloth-liquids-one-scene.md)
- Every health flag and its fix: [World health](../guides/health.md).
