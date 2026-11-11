# SPDX-License-Identifier: Apache-2.0
"""The applicators' steps on the film (design note §4.1; phase 2 note): their tables, and the launches ``plugin.py``
calls once per exchange (``tools_exchange``: ceilings, stores' nips, forming), once per film substep
(``tools_substep``: the plough, its roll and the roll's cap, roll_cap.py) and in ``apply`` (``tools_apply``: the pads
and the film's drag on the tools). A porous tool (``Applicator(porous=True)``, a sponge) with room (``tool_room``,
written each exchange by the lumped stage) absorbs what it would plough: ``tf.absorb``/``absorb_sp`` per cell until
the stage takes them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from alloy.plugins.surface_flow import applicators as AP
from alloy.plugins.surface_flow import forming as FM
from alloy.plugins.surface_flow import kernels as K
from alloy.plugins.surface_flow import roll_cap as RC
from alloy.plugins.surface_flow import stores as ST
from alloy.plugins.surface_flow import stores_species as SS
from alloy.plugins.surface_flow import tool_kernels as TK
from alloy.plugins.surface_flow import tool_passes as TP
from alloy.plugins.surface_flow import tool_passes_force as TPF
from alloy.plugins.surface_flow.host_api import H_DRY, H_WET

PARALLEL = True     # the tools' ceilings and pads in tool_passes (False: tool_kernels' one thread per world)


def setup_tools(flow: Any, ctx: Any, W: int, C: int) -> None:
    """Applicators (applicators.py): their tables, and the ceiling, cover and sliding fields the film reads."""
    tf = flow.tf = K.ToolField()
    tf.on = 1 if flow.config.tools else 0
    flow.tools = AP.build_tools(ctx, flow.config.tools, flow.config.charts, flow.tables,
                                forming=flow.config.bulk is not None) if flow.config.tools else None
    flow.tool_passes = TP.ToolPasses(ctx, flow.tools, flow.tables, W) if flow.tools is not None else None
    # only ceiling tools plough: with stores alone the substep's plough and roll passes are no-ops (DC6's brush: 45 %)
    flow.ploughs = any(t.store is None for t in flow.config.tools)
    n = (W, C) if flow.tools is not None else (1, 1)
    tf.ceil = ctx.full(n, float(TK.FREE_GAP), wp.float32)
    tf.cover, tf.vel, tf.push = ctx.zeros(n, wp.float32), ctx.zeros(n, wp.vec3), ctx.zeros(n, wp.vec2)
    tf.press = ctx.zeros(n, wp.float32)
    # every tool's underside, store tools too, for process layers (WP6's paste work); the film reads only ceilings
    tf.wceil, tf.wcover, tf.wvel = ctx.full(n, float(TK.FREE_GAP), wp.float32), ctx.zeros(n, wp.float32), ctx.zeros(
        n, wp.vec3)
    tf.pore, tf.absorb = ctx.zeros(n, wp.int32), ctx.zeros(n, wp.float64)     # porous pads (a sponge's absorption)
    tf.room = flow.tool_room = ctx.zeros((W, max(len(flow.config.tools), 1)), wp.float64)
    flow.excess = ctx.zeros(n, wp.float64)              # film above a tool's underside, moved each substep
    flow.tool_sink = ctx.zeros((W, max(len(flow.config.tools), 1)), wp.float32)   # forming: surface speed
    nt = max(len(flow.config.tools), 1)               # the drag cap's readout (WP2 review: say when it binds)
    flow.tool_drag_scale = flow.add_state("tool_drag_scale", ctx.full((W, nt), 1.0, wp.float32))
    flow.tool_drag_capped = flow.add_state("tool_drag_capped", ctx.zeros((W, nt), wp.int32))
    # an implicit pad's pressure over its explicit one (pad_implicit.py), for forming: written by apply
    flow.tool_pad_ratio = flow.add_state("tool_pad_ratio", ctx.full((W, nt), 1.0, wp.float32))
    flow.tool_normal = ctx.zeros((W, nt), wp.float32)   # each tool's pad normal force [N], written by apply
    setup_stores(flow, ctx, W, C)
    # roll_lift: the roll's cap and spill; scraping stores: their overflow in the same end bands (roll_cap.py)
    flow.roll_cap = RC.RollCap.build(ctx, flow.tools, W, flow.add_state, flow.tables.shapes,
                                     max(len(flow.config.species), 1))
    if flow.roll_cap is not None and flow.roll_cap.carry and flow.stores is not None:
        flow.overflow_on = 1                            # a full blade cuts on; the bands take its overflow
    flow.bulk = None
    if flow.config.bulk is not None:                    # a paste's plastic surface (float64; base is its copy)
        flow.bulk = flow.add_state("bulk", ctx.zeros((W, C), wp.float64))
        flow.bulk_yield = flow.add_input("bulk_yield", ctx.full((W, C), flow.config.bulk.yield_stress, wp.float32))

def setup_stores(flow: Any, ctx: Any, W: int, C: int) -> None:
    """The tools' stores and the trays they pick up from (stores.py): loads, trays and their ledgers."""
    c = flow.config
    flow.stores = None
    if not any(t.store is not None for t in c.tools):
        return
    st = flow.stores = ST.build_stores(ctx, c.tools, c.trays, tuple(c.species), flow.mat)
    n, T = len(c.tools), max(len(c.trays), 1)
    start = np.zeros((W, st.bin_count))
    for k, t in enumerate(c.tools):
        if t.store is not None:
            start[:, st.start[k]:st.start[k] + st.bins[k]] = t.store.initial * t.store.capacity / st.bins[k]
    flow.tool_load = flow.add_state("tool_load", ctx.array(start, wp.float64))
    trays = np.tile(np.asarray([t.volume for t in c.trays] or [0.0], np.float64), (W, 1))
    flow.tray_volume = flow.add_state("tray_volume", ctx.array(trays.reshape(W, T), wp.float64))
    flow.tool_dripped = flow.add_state("tool_dripped", ctx.zeros((W, n), wp.float64))
    flow.tool_picked = flow.add_state("tool_picked", ctx.zeros((W, n), wp.float64))
    flow.to_tools = flow.add_state("to_tools", ctx.zeros((W, C), wp.float64))      # film ledger column
    flow.tool_nip = flow.add_state("tool_nip", ctx.zeros((W, n), wp.vec3))         # the centre line, last step
    flow.tool_nip_bin = flow.add_state("tool_nip_bin", ctx.zeros((W, n), wp.vec2))  # roller: (bin + 1, its nap)
    shape = (W, C) if any(t.store is not None and t.store.fill == "profile" for t in c.tools) else (W, 1)
    flow.nose_low = flow.add_state("nose_low", ctx.zeros(shape, wp.float32))   # a nose's lowest this pass [m]
    flow.nose_seen = flow.add_state("nose_seen", ctx.full(shape, -2, wp.int32))  # the exchange it last covered
    flow.tool_release = ctx.zeros((W, n), wp.float64)     # the owner's share of each free load to let go, per exchange
    flow.tool_released = flow.add_state("tool_released", ctx.zeros((W, n), wp.float64))
    flow.tool_hm = flow.add_state("tool_hm", ctx.zeros((W, n), wp.float64))   # a metering brush's last h_m [m]
    # a scraping store's cut beyond capacity, pending for WP13's end bands (stores_lift.py), and its lifted gobs
    K = max(len(c.species), 1)
    flow.overflow_on = 0                                # WP13's end bands set it; 0: a full blade cuts what fits
    flow.tool_overflow = flow.add_state("tool_overflow", ctx.zeros((W, n), wp.float64))
    flow.tool_overflow_sp = flow.add_state("tool_overflow_sp", ctx.zeros((W, n, K), wp.float64))
    flow.tool_dropped = flow.add_state("tool_dropped", ctx.zeros((W, n), wp.float64))
    setup_store_species(flow, ctx, start, trays.reshape(W, T), K)
    # where a scraping store may fill (GA's paste mask: the joint's cells, not the faces beside it); 1: everywhere
    flow.paste_mask = flow.add_input("paste_mask", ctx.full((W, C), 1.0, wp.float32))
    # the rigid support a scraping store bears on (GA's support_height: a mesh's fibre tops); below -1e29: the base
    flow.support = flow.add_input("support", ctx.full((W, C), -1.0e30, wp.float32))
    # drips and releases leave through DC5's stage-drips channel when there is a supply to cast them
    flow.drip_on = int(flow.has_supply)
    drips = {"src_drip_v": ctx.zeros((W, n), wp.float64), "src_drip_p": ctx.zeros((W, n), wp.vec3),
             "src_drip_u": ctx.zeros((W, n), wp.vec3), "src_drip_sp": ctx.zeros((W, n, K), wp.float64)}
    flow.drip_arrays = {k: flow.add_state(k, a) if flow.drip_on else a for k, a in drips.items()}
    if flow.drip_on:
        for name, a in flow.drip_arrays.items():
            setattr(flow, name, a)                      # the supply finds them by name (setup order: tools first)
        flow.src_drip_body = flow.tools.body            # its ray passes the tool's own shapes


def setup_store_species(flow: Any, ctx: Any, start: np.ndarray, trays: np.ndarray, K: int) -> None:
    """Each bin's and tray's species [kg] and the stores' species books (stores_species.py)."""
    c, st = flow.config, flow.stores
    names = tuple(c.species)
    conc = ST.store_species(c.tools, names)[:, :K]                  # [n, K] kg/m³, each store's paint
    per_bin = np.zeros((st.bin_count, K))
    for k, t in enumerate(c.tools):
        if t.store is not None:
            per_bin[st.start[k]:st.start[k] + st.bins[k]] = conc[k]
    rows = ST.store_species(c.tools, names)[[k for k, t in enumerate(c.tools) if t.store is not None]]
    tray = SS.tray_conc(c.trays, names, rows)[:trays.shape[1], :K]
    W, n = start.shape[0], len(c.tools)
    s = flow.store_sp = SS.StoreSpecies()
    s.load = flow.tool_load_species = flow.add_state("tool_load_species", ctx.array(start[..., None] * per_bin[None],
                                                                                    wp.float64))
    s.tray = flow.tray_species = flow.add_state("tray_species", ctx.array(trays[..., None] * tray[None], wp.float64))
    for name in ("taken", "picked", "given"):
        a = flow.add_state(f"tool_species_{name}", ctx.zeros((W, n, K), wp.float64))
        setattr(s, name, a)


def tools_apply(flow: Any, state: Any, forces: Any, dt: float) -> None:
    """In ``apply``: each tool's pad pressure, Coulomb friction and the film's drag on it (tool_passes.pads)."""
    if flow.tools is None:
        return
    if PARALLEL:
        TPF.pads(flow, state, forces, dt, H_WET)
    else:
        serial_pads(flow, state, forces, dt)


def serial_pads(flow: Any, state: Any, forces: Any, dt: float) -> None:
    """``tool_kernels.tool_force_kernel``: one thread per world (the passes' reference)."""
    if flow.tools is not None:
        t = flow.tables
        wp.launch(TK.tool_force_kernel, dim=flow.ctx.world_count,
                  inputs=[flow.tools.set, flow.tools.count, t.charts, t.prof, flow.cells, flow.tools.body,
                          t.cbody, t.clocal,
                          state.body_q, state.body_qd, flow.ctx.model.body_com, flow.tools.mobility,
                          flow.tool_sink, flow.h, flow.base, t.area, flow.fluid, H_WET,
                          float(dt), flow.ctx.model.body_mass, flow.ctx.model.body_inertia,
                          flow.gravity],
                  outputs=[forces.body_f, flow.tool_drag_scale, flow.tool_pad_ratio, flow.tool_normal],
                  device=flow.ctx.device)


def tools_exchange(flow: Any, state_in: Any, state_out: Any, dt: float, due: list) -> None:
    """Once per exchange, at the new poses: the drag-cap count, the ceilings (tools_kernel), the stores' nips and
    trays, and forming."""
    t, dev, W = flow.tables, flow.ctx.device, flow.ctx.world_count
    if flow.tools is not None:
        wp.launch(TK.count_capped_kernel, dim=flow.tool_drag_scale.shape, inputs=[flow.tool_drag_scale],
                  outputs=[flow.tool_drag_capped], device=dev)
        if PARALLEL:                                # the tools' undersides over the cells, at the new poses
            TP.ceilings(flow, state_out)
        else:
            serial_ceilings(flow, state_out)
        if flow.roll_cap is not None:               # each capped tool's sole, cap and window at these poses
            flow.roll_cap.exchange(flow, state_out)
        if flow.stores is not None:                 # the stores: the nip split, tray pick-up, drips
            st = flow.stores
            wp.launch(ST.stores_kernel, dim=W,
                      inputs=[flow.tools.set, st.stores, st.trays, flow.tools.count, t.charts, flow.tools.body,
                              t.cbody, t.clocal, state_out.body_q, state_out.body_qd, flow.ctx.model.body_com,
                              flow.base, t.area, float(dt)],
                      outputs=[flow.tool_load, flow.tray_volume, flow.h, flow.to_tools, flow.cwet,
                               flow.tool_dripped, flow.tool_picked, flow.tool_nip, flow.tool_nip_bin, flow.sp,
                               flow.fluid.thixo, flow.fluid.lam, flow.tool_release, flow.drip_on,
                               flow.tool_released, *flow.drip_arrays.values(), flow.tool_normal, flow.tool_hm,
                               flow.paste_mask, flow.tool_overflow, flow.tool_overflow_sp, int(flow.overflow_on),
                               flow.tool_dropped, flow.store_sp, flow.support, flow.tick, flow.nose_low,
                               flow.nose_seen],
                      device=dev)
            if flow.roll_cap is not None and flow.roll_cap.carry:   # a full blade's overflow in its end bands
                flow.roll_cap.overflow(flow, state_out)
        if flow.bulk is not None:                   # tools press the paste's bulk above its bearing capacity
            b = flow.config.bulk
            # at the pre-solve poses, as the pad's force in apply: the paste yields to the pressure it carried
            wp.launch(FM.forming_kernel, dim=W,
                      inputs=[flow.tools.set, flow.tools.count, t.charts, flow.tools.body, t.cbody, t.clocal,
                              state_in.body_q, state_in.body_qd, flow.ctx.model.body_com, flow.bulk_yield,
                              b.resistance, float(dt), t.area, flow.bulk, flow.tool_pad_ratio],
                      outputs=[flow.tool_sink, flow.base], device=dev)


def tools_substep(flow: Any, due: list) -> None:
    """Each film substep: the film above a ceiling moves one cell on, then its roll relaxes (ROLL_PASSES)."""
    t, dev = flow.tables, flow.ctx.device
    W, C = flow.ctx.world_count, t.cell_count
    if flow.tools is not None and flow.ploughs:  # film above a ceiling tool's underside moves one cell on
        for roll in range(1 + TK.ROLL_PASSES):         # the plough, then its roll relaxing to 45°
            wp.launch(TK.excess_kernel, dim=(W, C), inputs=[flow.cells, flow.faces, *due, flow.tf, flow.base,
                      flow.h, int(roll > 0), flow.sp], outputs=[flow.excess], device=dev)
            wp.launch(TK.displace_kernel, dim=(W, C), inputs=[flow.cells, flow.faces, *due, flow.tf, H_DRY,
                      flow.excess, flow.sp], outputs=[flow.h, flow.cwet], device=dev)
        if flow.roll_cap is not None:               # a roll past its cap spills at the blade's ends
            flow.roll_cap.substep(flow, due[0], due[1])


__all__ = ["PARALLEL", "serial_ceilings", "serial_pads", "setup_stores", "setup_tools", "tools_apply",
           "tools_exchange", "tools_substep"]


def serial_ceilings(flow: Any, state_out: Any) -> None:
    """``tool_kernels.tools_kernel``: one thread per world (the passes' reference)."""
    t = flow.tables
    wp.launch(TK.tools_kernel, dim=flow.ctx.world_count,
              inputs=[flow.tools.set, flow.tools.count, t.charts, t.prof, flow.cells, flow.tools.body,
                      t.cbody, t.clocal,
                      state_out.body_q, state_out.body_qd, flow.ctx.model.body_com, flow.tools.wins,
                      flow.base, flow.h, flow.tf],
              outputs=[flow.tf.ceil, flow.tf.cover, flow.tf.vel, flow.tf.push, flow.tf.press, flow.cwet,
                       flow.tool_pad_ratio], device=flow.ctx.device)
