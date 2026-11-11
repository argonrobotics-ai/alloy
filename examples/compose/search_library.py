# SPDX-License-Identifier: Apache-2.0
"""What can the scene builder use? Search the card library (Alloy's own cards plus every imported library) by noun.

Each match says where it comes from, whether the builder can place it (rigid and static objects can; cloth, soft
bodies, liquids and cables cannot yet) and its size as it would rest on a table.

Run:

    python examples/compose/search_library.py mug
    python -m alloy.compose search mug --hint ceramic -k 5       # the same as JSON
"""

from __future__ import annotations

import sys

import alloy.compose


def main() -> int:
    query = sys.argv[1] if len(sys.argv) > 1 else "cube"
    for row in alloy.compose.search(query, k=8):
        size = "x".join(f"{s * 1000:.0f}" for s in row["size_m"]) + " mm" if row["size_m"] else "-"
        state = "placeable" if row["placeable"] else f"not placeable: {row['reason']}"
        print(f"{row['name']:40s} {row['source']:10s} {size:>16s}  {state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
