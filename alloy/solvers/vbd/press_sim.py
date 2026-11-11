# SPDX-License-Identifier: Apache-2.0
"""``PressSim``: the DirectSim of the ``cloth_box_press`` bully target (``press_target``), with WP7's S1 rows.

Imported by ``press_target.make_press_sim`` only when a sim is built: ``alloy.bully`` loads the solver plug-ins, so the
plug-in module itself must not import ``alloy.bully`` at load time.
"""

from __future__ import annotations

from alloy._vendor import newton
import numpy as np
import warp as wp

from alloy.bully.direct import DirectSim

from .press_target import count_crossings, count_pass_through, sheet_pairs


class PressSim(DirectSim):
    """DirectSim that checks the S1 rows once per control period and keeps each world's episode peak."""

    def __init__(self, model: newton.Model, solver: newton.solvers.SolverBase, boxes: np.ndarray, half: float,
                 **kw) -> None:
        super().__init__(model, solver, **kw)
        dev, w = model.device, int(model.world_count)
        self.half = float(half)
        self.box_of_world = wp.array(np.asarray(boxes, np.int32), dtype=wp.int32, device=dev)
        start, edges, tris, pe, pt = sheet_pairs(model)
        self._start, self._edges, self._tris = (wp.array(a, device=dev) for a in (start, edges, tris))
        self._pe, self._pt = wp.array(pe, device=dev), wp.array(pt, device=dev)
        self._pass, self._cross = wp.zeros(w, dtype=wp.int32, device=dev), wp.zeros(w, dtype=wp.int32, device=dev)
        self.peak_pass, self.peak_cross = np.zeros(w, np.int64), np.zeros(w, np.int64)

    def counts(self) -> tuple[np.ndarray, np.ndarray]:
        """Per world now: (particles through the box, crossing edge-triangle pairs)."""
        m, s = self.model, self.state
        self._pass.zero_()
        self._cross.zero_()
        wp.launch(count_pass_through, dim=m.particle_count, device=m.device,
                  inputs=[s.particle_q, m.particle_world, m.particle_radius, s.body_q, self.box_of_world, self.half],
                  outputs=[self._pass])
        wp.launch(count_crossings, dim=(int(m.world_count), self._pe.shape[0]), device=m.device,
                  inputs=[s.particle_q, self._start, self._edges, self._tris, self._pe, self._pt],
                  outputs=[self._cross])
        return self._pass.numpy().astype(np.int64), self._cross.numpy().astype(np.int64)

    def step_period(self) -> None:
        super().step_period()
        through, crossings = self.counts()
        np.maximum(self.peak_pass, through, out=self.peak_pass)
        np.maximum(self.peak_cross, crossings, out=self.peak_cross)

    def reset(self, world_mask: wp.array | None = None) -> None:
        super().reset(world_mask)
        sel = slice(None)
        if world_mask is not None:
            sel = np.asarray(world_mask.numpy() if hasattr(world_mask, "numpy") else world_mask).astype(bool)
        self.peak_pass[sel] = 0
        self.peak_cross[sel] = 0

    def bully_monitors(self) -> dict[str, np.ndarray]:
        """WP7's S1 rows for the episode so far (per world peak counts)."""
        return {"cloth_pass_through": self.peak_pass.copy(), "cloth_crossings": self.peak_cross.copy()}


__all__ = ["PressSim"]
