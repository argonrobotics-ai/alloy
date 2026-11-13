# Installing Alloy's MCP server

Alloy is a physics engine for robot learning. `alloy mcp`, from the `alloy-engine` package on PyPI, serves it as
Model Context Protocol (MCP) tools over stdio: build a scene, step it, read its state, check its health, render it.
It needs no API key and no configuration.

## Requirements

- Python 3.11 or 3.12. On an older Python, pip answers "from versions: none".
- Linux x86_64, on the CPU or on an NVIDIA GPU, or macOS on Apple silicon (CPU). A GPU is optional: without one
  Alloy runs on the CPU. The GPU needs an R580-series or newer NVIDIA driver and a Turing (`sm_75`) or newer GPU.

## Install

In a Python 3.11 or 3.12 virtual environment:

```bash
python3 -m venv .venv && source .venv/bin/activate      # or: uv venv -p 3.12 --seed && source .venv/bin/activate
pip install alloy-engine                                # the import name is alloy
python -c "import alloy; print(alloy.__version__)"
which alloy                                             # the server command's full path, for the client below
```

A stock Ubuntu 24.04 refuses pip outside a virtual environment (PEP 668, "externally-managed-environment"), and its
`python3 -m venv` needs `sudo apt install python3-venv` first.

## Configure the MCP client

The server is the `alloy` command with one argument, `mcp`, over stdio.

Claude Code, with the virtual environment active:

```bash
claude mcp add alloy -- alloy mcp
```

Other clients (Cline, Claude Desktop, Cursor and others that read an `mcpServers` block): give the full path that
`which alloy` printed, since the client does not run inside the virtual environment.

```json
{
  "mcpServers": {
    "alloy": {
      "command": "/full/path/to/.venv/bin/alloy",
      "args": ["mcp"]
    }
  }
}
```

## Check that it works

Call the `list` tool with `{"kind": "materials", "match": "water"}`: the reply lists one material, `water`. With
`{"kind": "scenes"}` it lists the library scenes, such as `demo_peg_d12`.

The first `build` or `run` of a new kind of scene compiles kernels once: for a few boxes, about 45 s on a CPU and 3 to
4 minutes on an NVIDIA GPU, and longer for robots, cloth and liquid (up to about 10 minutes on a GPU). The result is
cached, so later runs start in seconds; do not time out or give up on the first one.

`fetch`, `create` and `compose`'s objects from the Hub library need the `import` extra
(`pip install "alloy-engine[import]"`). Docs: https://alloy.argonrobotics.ai/docs/.
