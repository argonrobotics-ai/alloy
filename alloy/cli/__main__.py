# SPDX-License-Identifier: Apache-2.0
"""``python -m alloy.cli ...`` is the ``alloy`` command. Example: ``python -m alloy.cli list scenes``."""

from alloy.cli import main

if __name__ == "__main__":          # importing this module (tools, import checks) must not run the command
    raise SystemExit(main())
