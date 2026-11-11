# SPDX-License-Identifier: Apache-2.0
"""``python -m alloy.create``: see ``alloy.create.cli`` (``python -m alloy.create.serve`` runs the 3D generator)."""

if __name__ == "__main__":
    from alloy.create.cli import main

    raise SystemExit(main())
