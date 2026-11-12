# SPDX-License-Identifier: Apache-2.0
"""``python -m alloy ...`` is the ``alloy`` command; with no arguments it prints the command's help (GEO P3).
Example: ``python -m alloy list scenes``."""

import sys

from alloy.cli import main

if __name__ == "__main__":          # importing this module (tools, import checks) must not run the command
    raise SystemExit(main(sys.argv[1:] or ["--help"]))
