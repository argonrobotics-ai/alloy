# Command line and tool server (no Python needed)

`alloy` is a command-line tool with JSON in and JSON out, and `alloy mcp` serves the same operations as an MCP tool
server, so an agent can build, step, check and render scenes without writing a script. Every reply is one JSON
object on stdout (logs go to stderr): `{"ok": true, "op": ..., "result": {...}}` or
`{"ok": false, "op": ..., "error": {"type", "message", "fix", "field"}}`.

```bash
alloy list scenes   # also: objects, materials, robots, adds (scene recipe kinds), ops
alloy ops step                                       # one operation's JSON schema, with an example
alloy check demo_peg_d12 --quick                     # cards and start overlaps, no simulation
alloy step demo_peg_d12 --steps 50 --names peg_d12   # build, step, print state and health
alloy run '{"objects": [{"add": "ground"}, {"add": "box", "name": "cube", "size": [0.1, 0.1, 0.1],
           "pos": [0, 0, 0.5], "mass": 1.0}]}' --seconds 1 --out cube.npz
alloy batch requests.json                            # many calls in one process: build once, step many times
alloy vary scene.json --n 4 --pos 0.02 --yaw 15 --out variants   # scene variants (variants.md)
alloy fetch ycb_025_mug                              # download a library object (fetch.md)
```

- The scene JSON uses the same names and keywords as `alloy.Scene.add_*` (`"add": "box"` is `add_box`; every
  adder, including pour and heater: `alloy list adds`) and `Scene.set_*`
  (`"set": [{"set": "heater", "name": "pan_floor", "temperature": 353.15}]`). A robot's `path` also takes a library
  robot name from `alloy list robots`, e.g. `{"add": "robot", "name": "arm", "path": "emb_e2_panda"}`.
- While a simulation runs: `drive` (kinematic objects; `stop`), `pose`, `wrench`, `settle`, `heater` and `controls`
  call the `alloy.Sim` methods of the same purpose (`examples/cli/06_conveyor.sh`).
- A list flag can repeat: `--library my_objects --library my_scenes` searches both (the same as `a,b`).
- `alloy check --contacts newton` runs the check with that contact pipeline, as `alloy build --contacts` does
  (`check.md`).
- Without the console script (a checkout that is not installed): `python -m alloy.cli ...`.
- `--liquid-accuracy super` (MCP: `"liquid_accuracy": "super"`) on build, run, step and estimate sets the scene's
  liquid accuracy (`deformables.md`, Accuracy); `estimate` shows both settings. `alloy check` refuses it: a check
  reads saved cards, which keep no setting.
- Tool server: `alloy mcp` (stdio). Claude Code: `claude mcp add alloy -- alloy mcp`; other clients:
  `{"mcpServers": {"alloy": {"command": "alloy", "args": ["mcp"]}}}`. Tools: list, scene, build, step, run, state,
  targets, reset, drive, pose, wrench, settle, heater, controls, close, check, render (returns the PNG too),
  estimate, import, export, vary (returns its contact sheet too), compose, create, audit, fetch. Simulations stay open between calls, by id.

Runnable: `examples/cli/` (shell scripts, a batch session and an MCP client in Python).
