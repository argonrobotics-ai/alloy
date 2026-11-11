# SPDX-License-Identifier: Apache-2.0
"""The film step's table structs (``kernels.py`` re-exports them): cells, faces and the tools' field over the cells."""

from __future__ import annotations

import warp as wp


@wp.struct
class Cells:
    pos: wp.array[wp.vec3]
    nrm: wp.array[wp.vec3]
    area: wp.array[wp.float64]
    curv: wp.array[wp.float32]
    chart: wp.array[wp.int32]
    face: wp.array[wp.vec4i]
    tile: wp.array[wp.int32]          # the dry-tile skip (tiles.py)
    drain: wp.array[wp.int32]         # the drain a cell feeds (-1: none): a hole, never a contact line's dry side


@wp.struct
class Faces:
    a: wp.array[wp.int32]
    b: wp.array[wp.int32]
    kind: wp.array[wp.int32]
    chart: wp.array[wp.int32]
    line: wp.array[wp.vec2i]
    par: wp.array[wp.vec2i]
    perp: wp.array[wp.vec4i]
    pos: wp.array[wp.vec3]
    dir: wp.array[wp.vec3]                              # a -> b; on a chart link the chord over dist (gravity's drop)
    dira: wp.array[wp.vec3]                             # the a side's tangent, a's chart (dir off links; links.py)
    dirb: wp.array[wp.vec3]                             # the b side's tangent, b's chart
    length: wp.array[wp.float32]
    dist: wp.array[wp.float32]
    tile: wp.array[wp.int32]


@wp.struct
class ToolField:
    """Applicators over the cells (``tool_kernels.tools_kernel``): the film's ceiling and the tools' sliding."""

    ceil: wp.array2d[wp.float32]      # [W, C] the underside's height above the chart surface [m]
    cover: wp.array2d[wp.float32]     # [W, C] share of the cell under a tool
    vel: wp.array2d[wp.vec3]          # [W, C] the tool's sliding velocity, chart frame [m/s]
    push: wp.array2d[wp.vec2]         # [W, C] where film above the ceiling goes (unit, chart frame; 0: nowhere)
    press: wp.array2d[wp.float32]     # [W, C] the tool's pad pressure on the substrate [Pa] (work fields for layers)
    wceil: wp.array2d[wp.float32]     # [W, C] work fields for layers: every tool's lowest underside, store tools too
    wcover: wp.array2d[wp.float32]    # [W, C] its share of the cell (ceil/cover/vel: the film's ceilings only)
    wvel: wp.array2d[wp.vec3]         # [W, C] its sliding velocity, chart frame [m/s]
    sp_ex: wp.array3d[wp.float64]     # [W, C, K] the species leaving with a cell's excess (scratch; (1, 1, 1): none)
    pore: wp.array2d[wp.int32]        # [W, C] 1 + the porous tool with room whose ceiling the cell is under (0: none)
    room: wp.array2d[wp.float64]      # [W, tools] a porous tool's room this exchange [m³] (the lumped stage writes it)
    absorb: wp.array2d[wp.float64]    # [W, C] film a porous tool absorbed, per area [m], until the stage takes it
    absorb_sp: wp.array3d[wp.float64]  # [W, C, K] its species [kg/m²]
    on: wp.int32                      # 0: no applicators


__all__ = ["Cells", "Faces", "ToolField"]
