# SPDX-License-Identifier: Apache-2.0
"""``python -m alloy.mcp``: the Alloy MCP tool server on stdio (same as ``alloy mcp``)."""

from alloy.mcp import main

if __name__ == "__main__":          # importing this module must not start the server
    raise SystemExit(main())
