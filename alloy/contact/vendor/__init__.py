# SPDX-License-Identifier: Apache-2.0
"""Third-party code Alloy carries until it lands in a pinned dependency. Each file names its source, its license and
when to drop it.

- ``mjw_pr1845_polytope`` and ``mjw_pr1845_epa``: the first six EPA commits of google-deepmind/mujoco_warp PR #1845
  on our MuJoCo Warp pin, only the definitions they change and those that use them (Apache-2.0,
  ``LICENSE-mujoco_warp``); installed by :mod:`alloy.contact.mjw_epa`.
"""
