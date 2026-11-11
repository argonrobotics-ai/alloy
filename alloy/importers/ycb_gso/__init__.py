# SPDX-License-Identifier: Apache-2.0
"""YCB Object and Model Set + Google Scanned Objects importer.

Imports every YCB scan (measured masses, CC BY 4.0) and every Google Scanned Object (CC BY 4.0) as Alloy object
cards under ``alloy-assets/imports/ycb_gso/``; binaries live in the asset bucket and are fetched by hash.

Example::

    python -m alloy.importers.ycb_gso list --json
"""
