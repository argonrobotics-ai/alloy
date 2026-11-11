# SPDX-License-Identifier: Apache-2.0
"""``python -m alloy.plugins``: see ``alloy.plugins.cli``."""

import sys

from alloy.plugins.cli import main

if __name__ == "__main__":
    sys.exit(main())
