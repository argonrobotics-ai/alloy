# SPDX-License-Identifier: Apache-2.0
"""Solver factory (WP7, SPEC §3.11): agents write solver plug-ins; the bench decides whether they ship.

- ``template/``: the smallest package that passes the §3.8 contract (particles on a ground plane, NATIVE).
- ``evaluate.evaluate_plugin``: contract checks (``contract.py``), the plug-in's laws through the bench, B0 + B1
  attack on its B0 scenes, and the ship rule (contract ok, lint clean if NATIVE, every law ``pass``, no S1/S2).
- ``runner.ShellAgentRunner``: the agent runs in a scratch git worktree with the hidden valid set scrubbed from its
  environment; writes outside ``alloy/solvers/contrib/<name>/`` are reverted and reported.
- ``loop.factory_loop``: create -> evaluate (in a subprocess) -> feedback, up to ``rounds``.

CLI: ``python -m alloy.factory evaluate --manifest PATH`` and ``python -m alloy.factory loop --brief SPEC --name N
(--runner-cmd "..." | --copy-template)``. Types: ``alloy.factory.types``.
"""
