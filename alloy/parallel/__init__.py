# SPDX-License-Identifier: Apache-2.0
"""Parallel speed (WP8): cost model, world packing, structure report, PeriodGraph, speed ladder (SPEC §3.9).

Types live in ``alloy.parallel.types`` (WP0). Modules:

- ``costmodel``: ``fit_cost_model`` (T(N) = t_floor + N c_w), ``n_knee``, ``pack_size``, MJW dense-H and memory
  estimates, buffer capacities.
- ``probe``: timing protocol and ``WorldProfile`` fit for a pack (``probe_targets``, ``probe_scene``);
  ``counters``: solver counters (MJW iterations, rows, contacts, overflow) and the graph-safe ``TraceReplayer``.
- ``packer``: ``WorldPacker`` (``probe``, ``plan``, ``build`` through a SceneAssembler, ``repack``);
  ``planning`` and ``repack`` hold its pure steps.
- ``structure``: ``classify_structure`` (islands, structure class, parallel unit, recommended solver).
- ``graphs``: ``PeriodGraph`` / ``GraphCache``, the capture utility ``Simulation`` calls (D18; host-sync counter).
- ``multigpu``: ``launch_mode_a`` (one process per device, global world ids).
- ``rungs``, ``rung_scenes``, ``rung_arm``: the L0-L7 ladder table and raw-Newton rung templates (PAR-1 baseline);
  the bench family is ``bench/scenarios/ladder.py``.
"""
