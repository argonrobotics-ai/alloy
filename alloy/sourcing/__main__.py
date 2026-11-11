# SPDX-License-Identifier: Apache-2.0
"""Entry point: ``python -m alloy.sourcing <cmd>`` (see ``alloy.sourcing.cli``)."""

from __future__ import annotations

import sys

from alloy.sourcing.cli import main

if __name__ == "__main__":
    sys.exit(main())
