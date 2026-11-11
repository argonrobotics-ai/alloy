# SPDX-License-Identifier: Apache-2.0
"""Developer tools (WP5). ``python -m alloy.tools.lint_atomics alloy/``: the determinism lint of SPEC §3.7 (D3).

It flags float ``atomic_add``/``atomic_sub``, ``tile_atomic_add`` and array-slot ``+=``/``-=`` in Warp kernels and
functions. Slots a thread owns (``a[tid] += x``), integer, max/min and bitwise atomics pass. It also flags
order-dependent atomics (``atomic_exch``, ``tile_scatter_add``), native snippets with ``atomicAdd``, and two-term
sums (``# alloy: det-2term``) whose target is not zero-filled before every launch. Escape hatch:
``# alloy: det-ok <reason>``. Dtypes are resolved from annotations, module aliases and ``@wp.struct`` fields.

``python -m alloy.tools.mjw_det_probe --bits <0-7> --worlds N --out run.npz``: per-world digests, step time and memory
of a WP13 task scene (T12 by default) under one MuJoCo Warp ``DeterminismType`` mode (P3-1). One process per run.
"""
