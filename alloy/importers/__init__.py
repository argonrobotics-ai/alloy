# SPDX-License-Identifier: Apache-2.0
"""Library importers: open asset libraries in, Alloy object cards out.

- ``alloy.importers.common``: the shared per-object pipeline (watertight physics mesh, convex hulls, glTF visual,
  thumbnail, card/manifest/queue writers, batch runner).
- ``alloy.importers.ycb_gso``: the YCB Object and Model Set and Google Scanned Objects (CC BY 4.0).

Example::

    python -m alloy.importers.ycb_gso list --lib ycb
"""
