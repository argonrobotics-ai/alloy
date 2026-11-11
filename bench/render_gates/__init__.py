# SPDX-License-Identifier: Apache-2.0
"""Render quality gates: does the fast camera match Blender Cycles, per scene, image size and GPU?

The gates are calibrated on Cycles' own noise: Cycles seed 1 at 128 spp + denoise against the 4,096-spp seed-0
reference sets the limit (G1 = 2x that gap, FLIP mean and LPIPS), and Cycles control seeds 1-4 set the equal-spp
yardstick. Modules:

- :mod:`.layout`: the ``<scene>/<frame>/<engine>/<W>x<H>_<spp>spp_<on|off>.png`` layout;
- :mod:`.metrics`: FLIP, LPIPS and a cached scorer;
- :mod:`.checks`: parity, end-to-end G1, equal-spp rows and the statistical verdict, speed (pure functions);
- :mod:`.score`: one GPU's report (per frame and per cell = scene x size);
- :mod:`.matrix`: the scene x size x GPU verdicts and a markdown table;
- :mod:`.refs`: hashed reference tars (``refs_manifest.json``), fetch and pack;
- :mod:`.nightly`: fetch, render, score and matrix in one call.

Example:
    >>> from bench.render_gates import matrix
    >>> list(matrix.load_spec()["scenes"])
    ['c2', 'c3', 'c4', 'dc5', 't12']
"""
