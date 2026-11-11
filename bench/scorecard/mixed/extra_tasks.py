# SPDX-License-Identifier: Apache-2.0
"""Four more hold tasks for the mixed-batch benchmark: three Franka FR3 embodiments and the H1-2 humanoid.

They are Alloy library embodiments (``library/robots/emb_*``), added to the single-task harness's ``TASKS`` at run
time so that file stays untouched. No Isaac Lab task carries these names; ``isaac_task`` is a label only.
"""

from __future__ import annotations

from typing import Any

PROPS = ("box", "cylinder", "mug")
NO_ISAAC = "Alloy library embodiment; no Isaac Lab task of this name"


def extra_tasks(task_cls: Any) -> tuple[Any, ...]:
    """The four ``HoldTask`` entries, built with the harness's own ``HoldTask`` class."""
    return (
        task_cls("fr3_reach", "Mixed-Reach-FR3", "emb_e1_fr3", "free", (("ee", "hand"),), -0.75, PROPS,
                 note=f"Menagerie FR3 on a table, no props. {NO_ISAAC}"),
        task_cls("fr3_bimanual_reach", "Mixed-Reach-FR3-Bimanual", "emb_e4_fr3_bimanual", "free",
                 (("ee", "left_hand"),), -0.75, PROPS,
                 note=f"Two FR3 arms on a table, left hand tracked, no props. {NO_ISAAC}"),
        task_cls("fr3_duo_reach", "Mixed-Reach-FR3-Duo", "emb_e5_fr3_duo", "free", (("ee", "left_fr3_link7"),), 0.0,
                 PROPS, note=f"Mobile base (free joint) with two FR3 arms on a floor, left wrist tracked, no props. {NO_ISAAC}"),
        task_cls("h1_2_flat", "Mixed-Flat-H1-2", "emb_e7_h1_2", "stand", (("base", "pelvis"),), 0.0,
                 note=f"Unitree H1-2 humanoid standing on a floor, all joints held at the init keyframe. {NO_ISAAC}"),
    )


def register(T: Any) -> None:
    """Add the four tasks to ``T.TASKS`` (``T`` is the imported single-task ``tasks`` module)."""
    for task in extra_tasks(T.HoldTask):
        T.TASKS[task.name] = task
