# Use Alloy with your coding agent

Alloy is written to be driven by a coding agent: short pages, code that runs as shown, and errors that say what to
change. Paste this prompt into your agent and replace `<your task>`.

```text
Install alloy-engine with pip (Python 3.11 or 3.12, in a virtual environment). Read its docs: run python -m alloy.docs, or fetch https://alloy.argonrobotics.ai/llms.txt. Then build <your task> in many parallel worlds: start from the closest ready-made environment in the docs and change it step by step. Train a policy until it succeeds, check sim.health() after every run, and report the success rate.
```

## What the agent finds

- `python -m alloy.docs` prints three paths on your machine, all installed with the package: the docs (these short
  pages are in its `public` folder), the runnable examples, and `AGENTS.md`, a longer quick start written for
  agents (`python -m alloy.docs agents` prints it). No network is needed to read them.
- `llms.txt` at the address above lists these pages, one line each; `llms-full.txt` beside it has their full text.
- Every page is short, and its code runs as written and prints the output shown under it.

## Connect over MCP

`alloy mcp` serves the engine as Model Context Protocol tools over stdio, so an agent can build and step scenes
without writing Python. It comes with the package. For Claude Code:

```bash
claude mcp add alloy -- alloy mcp
```

Any other MCP client starts the same command, `alloy mcp`. The tools are `list`, `scene`, `build`, `step`, `run`,
`state`, `targets`, `reset`, `close`, `check`, `render`, `estimate`, `import`, `export`, `vary`, `compose`,
`create`, `audit` and `fetch`; each takes and returns JSON. The server's code is in
[alloy/mcp](https://github.com/argonrobotics-ai/alloy/tree/main/alloy/mcp).

## What to ask your agent to do

1. Start from the [Quickstart](quickstart.md), [What ships](what-ships.md) (the environments, robots and objects
   that already exist) and the guide for the task, not from memory.
2. Use `import alloy` and the calls on these pages only.
3. After every run, read `sim.check()`: when `"ok"` is False, each entry of `"problems"` has a `"fix"`. Apply it and
   run again ([World health](guides/health.md)).
4. On an exception, read `e.message` and `e.fix` ([Errors and fixes](errors.md)).
5. Report what it ran, what that printed, and any line of [Keep in mind](keep-in-mind.md) that touches the result.

## Good first tasks

- "Drop 64 boxes into a bin in 16 worlds and tell me how many end inside."
- "Add a robot arm from an MJCF file, move one joint to 0.5 rad and plot the joint angle."
- "Pour 50 ml of water into a cup and report how much lands in it."

## Keep in mind

- A frozen world looks calm. An agent that reads only positions will report a failed run as a success, so make it
  print `sim.check()["ok"]`.
- The first run of each kind of scene compiles for minutes. Tell the agent not to stop it.
- Results on a GPU differ a little from run to run once things touch: compare with a tolerance and by success rate.
