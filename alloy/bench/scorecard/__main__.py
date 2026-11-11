# SPDX-License-Identifier: Apache-2.0
"""Entry point: `python -m alloy.bench.scorecard <row>` (see alloy.bench.scorecard.cli)."""

from alloy.bench.scorecard.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
