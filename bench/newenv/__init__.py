# SPDX-License-Identifier: Apache-2.0
"""Scorecard row 7, the new-environment test: cold agents build a requested environment in several engines.

The public part is the protocol (requests.json, the contract in preamble_contract.md), the driver every environment
is run through (drive.py), the checkers, and the reference environments; ``python -m bench.newenv`` builds a task
directory and scores a finished run. The cloud harness that runs cold agents (launch.py, harness.py) needs the
cold-agent kit in bench/agents; see README.md.

Example:
    >>> from bench.newenv import ENGINES
    >>> ENGINES
    ('alloy', 'maniskill', 'playground', 'genesis')
"""

ENGINES: tuple[str, ...] = ("alloy", "maniskill", "playground", "genesis")
