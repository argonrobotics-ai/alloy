# SPDX-License-Identifier: Apache-2.0
"""Canonical constraint-row order for MuJoCo Warp (P3-1 route c), the body of ``constraint._sort_constraints``.

MuJoCo Warp calls the function at the end of ``make_constraint`` when ``opt.deterministic & CONSTRAINT``. Rows are
then in counter order (``atomic_add(nefc_out, worldid, k)``). Per world, one stable radix pass orders them by
(type, id); ``mjw_order`` explains why that order is canonical. After the pass:
- every row field ``make_constraint`` writes is gathered into the new order;
- the sparse J storage is rebuilt in the new row order (an exclusive scan of ``J_rownnz``);
- the JTDAJ block list (first row, row count) is remapped and re-sorted by first row;
- ``contact.efc_address`` is remapped through the inverse permutation.
Scratch is allocated per Data on the first call (eager); later calls are graph safe.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import warp as wp

from alloy.precision.mjw_jmove import (_cycle_leaders, _dense_permute, _live_nnz, _nnz_overflow, rebuild_sparse)
from alloy.precision.owned_cache import remember

log = logging.getLogger("alloy.precision")

_ROW_INVALID = wp.constant(wp.uint64(0xFFFFFFFFFF))  # low 40 bits all ones: after every valid row of its world
_ROW_FIELDS = ("type", "id", "pos", "margin", "D", "vel", "aref", "frictionloss", "Jqvel")


_EQUALITY = wp.constant(0)  # ConstraintType.EQUALITY
_CONTACT_FIRST = wp.constant(5)  # ConstraintType.CONTACT_FRICTIONLESS; pyramidal 6, elliptic 7


@wp.kernel
def _row_keys(nefc: wp.array[int], njmax: int, eq_type: wp.array[int], rtype: wp.array2d[int], rid: wp.array2d[int],
              keys: wp.array[wp.uint64], vals: wp.array[wp.int32]):
    w, r = wp.tid()
    s = w * njmax + r
    vals[s] = s
    base = wp.uint64(w) << wp.uint64(40)
    if r >= wp.min(nefc[w], njmax):
        keys[s] = base | _ROW_INVALID
        return
    t = rtype[w, r]
    low = wp.uint64(wp.uint32(rid[w, r]))
    if t == _EQUALITY:
        # MuJoCo Warp emits connect, weld, joint, tendon, flex rows in that kernel order (mjtEq order); equality-force
        # sensors read that layout, so the subtype ranks before the equality id
        low = (wp.uint64(eq_type[rid[w, r]] & 0xFF) << wp.uint64(24)) | (low & wp.uint64(0xFFFFFF))
    keys[s] = base | (wp.uint64(t & 0xFF) << wp.uint64(32)) | low


@wp.kernel
def _remap_contact_ids(nefc: wp.array[int], njmax: int, inv: wp.array[int], rtype: wp.array2d[int],
                       rid: wp.array2d[int]):
    w, r = wp.tid()
    if r < wp.min(nefc[w], njmax) and rtype[w, r] >= _CONTACT_FIRST:
        rid[w, r] = inv[rid[w, r]]


@wp.kernel
def _row_perm(nefc: wp.array[int], njmax: int, vals: wp.array[wp.int32], perm: wp.array2d[int],
              inv: wp.array2d[int]):
    w, r = wp.tid()
    if r >= wp.min(nefc[w], njmax):
        return
    old = int(vals[w * njmax + r]) - w * njmax
    perm[w, r] = old
    inv[w, old] = r


@wp.kernel
def _row_gather(nefc: wp.array[int], njmax: int, perm: wp.array2d[int], src: wp.array2d(dtype=Any),
                dst: wp.array2d(dtype=Any)):
    w, r = wp.tid()
    if r < wp.min(nefc[w], njmax):
        dst[w, r] = src[w, perm[w, r]]


@wp.kernel
def _row_copy(nefc: wp.array[int], njmax: int, src: wp.array2d(dtype=Any), dst: wp.array2d(dtype=Any)):
    w, r = wp.tid()
    if r < wp.min(nefc[w], njmax):
        dst[w, r] = src[w, r]


@wp.struct
class _RowCols:
    """The row fields of a sparse model, for one fused gather (fewer graph nodes at small N)."""
    type: wp.array2d(dtype=int)
    id: wp.array2d(dtype=int)
    pos: wp.array2d(dtype=float)
    margin: wp.array2d(dtype=float)
    D: wp.array2d(dtype=float)
    vel: wp.array2d(dtype=float)
    aref: wp.array2d(dtype=float)
    frictionloss: wp.array2d(dtype=float)
    Jqvel: wp.array2d(dtype=float)
    J_rownnz: wp.array2d(dtype=int)
    J_rowadr: wp.array2d(dtype=int)


@wp.func
def _move_row(src: _RowCols, w: int, i: int, dst: _RowCols, j: int, with_adr: bool):
    dst.type[w, j] = src.type[w, i]
    dst.id[w, j] = src.id[w, i]
    dst.pos[w, j] = src.pos[w, i]
    dst.margin[w, j] = src.margin[w, i]
    dst.D[w, j] = src.D[w, i]
    dst.vel[w, j] = src.vel[w, i]
    dst.aref[w, j] = src.aref[w, i]
    dst.frictionloss[w, j] = src.frictionloss[w, i]
    dst.Jqvel[w, j] = src.Jqvel[w, i]
    dst.J_rownnz[w, j] = src.J_rownnz[w, i]
    if with_adr:
        dst.J_rowadr[w, j] = src.J_rowadr[w, i]


@wp.kernel
def _rows_gather(nefc: wp.array[int], njmax: int, perm: wp.array2d[int], src: _RowCols, dst: _RowCols):
    w, r = wp.tid()
    if r < wp.min(nefc[w], njmax):
        _move_row(src, w, perm[w, r], dst, r, True)


@wp.kernel
def _rows_copy(nefc: wp.array[int], njmax: int, with_adr: int, src: _RowCols, dst: _RowCols):
    w, r = wp.tid()
    if r < wp.min(nefc[w], njmax):
        _move_row(src, w, r, dst, r, with_adr != 0)


def _row_cols(arrays: dict) -> _RowCols:
    c = _RowCols()
    for f in _RowCols.vars:
        setattr(c, f, arrays[f])
    return c


SMALL_WORLDS = 16  # at most this many worlds: one block per world ranks its rows (fewer graph nodes than radix)
RANK_LANES = 256


@wp.func
def _row_key(w: int, r: int, eq_type: wp.array[int], rtype: wp.array2d[int], rid: wp.array2d[int]) -> wp.uint64:
    """``_row_keys``' key of a live row within its world (no world bits)."""
    t = rtype[w, r]
    low = wp.uint64(wp.uint32(rid[w, r]))
    if t == _EQUALITY:
        low = (wp.uint64(eq_type[rid[w, r]] & 0xFF) << wp.uint64(24)) | (low & wp.uint64(0xFFFFFF))
    return (wp.uint64(t & 0xFF) << wp.uint64(32)) | low


@wp.func
def block_barrier() -> int:
    """A block-wide barrier: ``tile_reduce`` synchronizes the block (WP_TILE_SYNC) around its shared reduction."""
    t = wp.tile_reduce(wp.add, wp.tile(int(0), preserve_type=True))
    return t[0]


@wp.kernel
def _sort_rows_small(
    nefc: wp.array[int], njmax: int, eq_type: wp.array[int], keys: wp.array[wp.uint64], src: _RowCols,
    tmp: _RowCols, with_adr: int, check_nnz: int, njmax_nnz: int, jtdaj_remap: int, nblock: wp.array[int],
    nacon: wp.array[int], naconmax: int, nslot: int, contact_worldid: wp.array[int], remap_ids: int,
    contact_inv: wp.array[int], perm: wp.array2d[int], inv: wp.array2d[int], overflow: wp.array[int],
    jtdaj_adr: wp.array2d[int], efc_address: wp.array2d[int],
):
    """One block per world, the whole row sort in one launch (few worlds; graph nodes dominate at N = 1).
    keys -> rank (stable (key, index) order, as the radix path) -> gather -> copy back, plus the JTDAJ head remap
    (route), the contact efc_address remap and the route's NJMAX_NNZ check. ``block_barrier`` separates the phases.
    With ``remap_ids`` (imported contacts, just sorted) contact rows first take their contact's new slot."""
    w, lane = wp.tid()
    n = wp.min(nefc[w], njmax)
    base = w * njmax
    lanes = wp.block_dim()
    for r in range(lane, n, lanes):
        if remap_ids != 0 and src.type[w, r] >= _CONTACT_FIRST:
            src.id[w, r] = contact_inv[src.id[w, r]]
        keys[base + r] = _row_key(w, r, eq_type, src.type, src.id)
    block_barrier()
    nnz = int(0)
    for r in range(lane, n, lanes):
        kr = keys[base + r]
        rank = int(0)
        for q in range(n):
            kq = keys[base + q]
            if kq < kr or (kq == kr and q < r):
                rank += 1
        perm[w, rank] = r
        inv[w, r] = rank
        nnz += src.J_rownnz[w, r]
    total = wp.tile_reduce(wp.add, wp.tile(nnz, preserve_type=True))
    if check_nnz != 0 and lane == 0 and total[0] > njmax_nnz:
        overflow[w] = overflow[w] | 2  # OverflowType.NJMAX_NNZ
    for j in range(lane, n, lanes):
        _move_row(src, w, perm[w, j], tmp, j, True)
    if jtdaj_remap != 0:
        for g in range(lane, nblock[w], lanes):
            jtdaj_adr[w, g] = inv[w, jtdaj_adr[w, g]]
    for c in range(lane, wp.min(nacon[0], naconmax), lanes):
        if contact_worldid[c] == w:
            for i in range(nslot):
                a = efc_address[c, i]
                if a >= 0 and a < njmax:
                    efc_address[c, i] = inv[w, a]
    block_barrier()
    for j in range(lane, n, lanes):
        _move_row(tmp, w, j, src, j, with_adr != 0)


@wp.kernel
def _rank_rows(nefc: wp.array[int], njmax: int, keys: wp.array[wp.uint64], rownnz: wp.array2d[int],
               check_nnz: int, njmax_nnz: int, perm: wp.array2d[int], inv: wp.array2d[int], overflow: wp.array[int]):
    """One block per world: a row's new index is the number of rows with a smaller (key, index), the same stable
    order the radix path gives. Also flags NJMAX_NNZ from the live nnz total (route, storage kept in place)."""
    w, lane = wp.tid()
    n = wp.min(nefc[w], njmax)
    base = w * njmax
    nnz = int(0)
    for r in range(lane, n, wp.block_dim()):
        kr = keys[base + r]
        rank = int(0)
        for q in range(n):
            kq = keys[base + q]
            if kq < kr or (kq == kr and q < r):
                rank += 1
        perm[w, rank] = r
        inv[w, r] = rank
        nnz += rownnz[w, r]
    total = wp.tile_reduce(wp.add, wp.tile(nnz, preserve_type=True))
    if check_nnz != 0 and lane == 0 and total[0] > njmax_nnz:
        overflow[w] = overflow[w] | 2  # OverflowType.NJMAX_NNZ


@wp.kernel
def _jtdaj_remap(nblock: wp.array[int], inv: wp.array2d[int], adr: wp.array2d[int]):
    w, j = wp.tid()  # route only: blocks keep their order, heads follow their rows (the Fixed64 H is order-free)
    if j < nblock[w]:
        adr[w, j] = inv[w, adr[w, j]]


@wp.kernel
def _jtdaj_mark(nblock: wp.array[int], inv: wp.array2d[int], adr: wp.array2d[int], nrow: wp.array2d[int],
                mark: wp.array2d[int]):
    w, j = wp.tid()
    if j < nblock[w]:
        mark[w, inv[w, adr[w, j]]] = nrow[w, j]


@wp.kernel
def _jtdaj_emit(nefc: wp.array[int], njmax: int, mark: wp.array2d[int], adr: wp.array2d[int],
                nrow: wp.array2d[int]):
    w = wp.tid()
    j = int(0)
    for r in range(wp.min(nefc[w], njmax)):
        k = mark[w, r]
        if k > 0:
            adr[w, j] = r
            nrow[w, j] = k
            j += 1
        mark[w, r] = 0


@wp.kernel
def _remap_contacts(nacon: wp.array[int], naconmax: int, njmax: int, worldid: wp.array[int],
                    inv: wp.array2d[int], efc_address: wp.array2d[int]):
    c, i = wp.tid()
    if c >= wp.min(nacon[0], naconmax):
        return
    a = efc_address[c, i]
    if a >= 0 and a < njmax:
        efc_address[c, i] = inv[worldid[c], a]


@dataclass
class _RowScratch:
    keys: wp.array
    vals: wp.array
    perm: wp.array
    inv: wp.array
    copies: dict[str, wp.array]
    j_copy: wp.array | None   # sparse J relaid without the route only (rebuild_sparse), allocated on first use
    c_copy: wp.array | None
    lead: wp.array | None     # dense J: cycle leaders of the in-place row permutation
    mark: wp.array | None
    end_bit: int
    nnz_flat: wp.array | None = None  # live rownnz, flattened (sparse J), and its exclusive scan
    nnz_scan: wp.array | None = None


_SCRATCH: dict[int, _RowScratch] = {}


def _scratch(d: Any) -> _RowScratch:
    s = _SCRATCH.get(id(d))
    if s is not None:
        return s
    e, dev = d.efc, d.nefc.device
    nw, nj = int(d.nworld), int(d.njmax)
    sparse = e.J_rownnz.shape[1] > 0
    blocks = e.jtdaj_adr.shape[1] > 0
    copies = {f: wp.empty_like(getattr(e, f)) for f in _fields(sparse)}
    s = _RowScratch(keys=wp.empty(2 * nw * nj, dtype=wp.uint64, device=dev),
                    vals=wp.empty(2 * nw * nj, dtype=wp.int32, device=dev),
                    perm=wp.zeros((nw, nj), dtype=int, device=dev), inv=wp.zeros((nw, nj), dtype=int, device=dev),
                    copies=copies, j_copy=None, c_copy=None,
                    lead=None if sparse else wp.zeros((nw, nj), dtype=wp.int8, device=dev),
                    mark=wp.zeros((nw, nj), dtype=int, device=dev) if blocks else None,
                    end_bit=40 + max(int(nw - 1).bit_length(), 1),
                    nnz_flat=wp.zeros(nw * nj, dtype=int, device=dev) if sparse else None,
                    nnz_scan=wp.zeros(nw * nj, dtype=int, device=dev) if sparse else None)
    return remember(_SCRATCH, id(d), d, s)  # dropped when d is freed: a reused id never finds it


def _fields(sparse: bool) -> tuple[str, ...]:
    return (*_ROW_FIELDS, "J_rownnz", "J_rowadr") if sparse else _ROW_FIELDS


def _permute_rows(d: Any, s: _RowScratch, nj: int, sparse: bool, keep_storage: bool = False) -> None:
    """Gather every row field into the canonical order. ``keep_storage``: J entries stay where they are and each row
    keeps its storage address (``J_rowadr`` moves with the row); otherwise ``mjw_jmove.rebuild_sparse`` relays them out."""
    e, nw = d.efc, int(d.nworld)
    if sparse and set(_fields(sparse)) == set(_RowCols.vars):  # two launches instead of two per field
        src, tmp = _row_cols({f: getattr(e, f) for f in _RowCols.vars}), _row_cols(s.copies)
        wp.launch(_rows_gather, dim=(nw, nj), inputs=[d.nefc, nj, s.perm, src], outputs=[tmp])
        wp.launch(_rows_copy, dim=(nw, nj), inputs=[d.nefc, nj, int(keep_storage), tmp], outputs=[src])
        return
    for f in _fields(sparse):
        src = getattr(e, f)
        wp.launch(_row_gather, dim=(nw, nj), inputs=[d.nefc, nj, s.perm, src],
                  outputs=[s.copies[f]])
    for f in _fields(sparse):
        if f == "J_rowadr" and not keep_storage:
            continue
        dst = getattr(e, f)
        wp.launch(_row_copy, dim=(nw, nj), inputs=[d.nefc, nj, s.copies[f], dst])


def sort_constraints(m: Any, d: Any) -> None:
    """Canonical row order per world (MuJoCo Warp calls this when ``opt.deterministic & CONSTRAINT``)."""
    nw, nj = int(d.nworld), int(d.njmax)
    if nw == 0 or nj == 0:
        return
    s, e = _scratch(d), d.efc
    sparse = e.J_rownnz.shape[1] > 0
    route = _route_on(m)
    small = nw <= SMALL_WORLDS
    fused = small and sparse and set(_fields(sparse)) == set(_RowCols.vars)
    cinv = None
    if not bool(getattr(m.opt, "run_collision_detection", True)) and int(d.naconmax) > 0:
        from alloy.precision.mjw_order import sort_imported_contacts

        cinv = sort_imported_contacts(m, d)  # contact rows then refer to canonical contact slots
        if not fused:  # (the fused sort remaps them in its key phase)
            wp.launch(_remap_contact_ids, dim=(nw, nj), inputs=[d.nefc, nj, cinv, e.type], outputs=[e.id])
    if fused:
        _sort_small(d, s, nj, route, _eq_type(m), cinv)
        return
    wp.launch(_row_keys, dim=(nw, nj), inputs=[d.nefc, nj, _eq_type(m), e.type, e.id], outputs=[s.keys, s.vals])
    if small:  # one launch instead of radix + perm (+ the nnz scan on the route): fewer graph nodes at N = 1
        rownnz = e.J_rownnz if sparse and route else s.copies["type"]  # (unread unless the nnz check is on)
        wp.launch_tiled(_rank_rows, dim=nw, inputs=[d.nefc, nj, s.keys, rownnz, int(sparse and route),
                                                    int(d.njmax_nnz)],
                        outputs=[s.perm, s.inv, d.overflow], block_dim=RANK_LANES)
    else:
        wp.utils.radix_sort_pairs(s.keys, s.vals, nw * nj, end_bit=s.end_bit)
        wp.launch(_row_perm, dim=(nw, nj), inputs=[d.nefc, nj, s.vals], outputs=[s.perm, s.inv])
    _permute_rows(d, s, nj, sparse, keep_storage=sparse and route)
    if sparse and route:  # storage stays in place; NJMAX_NNZ from the live total (in _rank_rows when small)
        if not small:
            wp.launch(_live_nnz, dim=(nw, nj), inputs=[d.nefc, nj, e.J_rownnz], outputs=[s.nnz_flat])
            wp.utils.array_scan(s.nnz_flat, s.nnz_scan, inclusive=False)
            wp.launch(_nnz_overflow, dim=nw, inputs=[nj, s.nnz_flat, s.nnz_scan, int(d.njmax_nnz)],
                      outputs=[d.overflow])
    elif sparse:
        rebuild_sparse(d, s, nj)
    else:
        wp.launch(_cycle_leaders, dim=nw, inputs=[d.nefc, nj, s.perm], outputs=[s.lead])
        wp.launch(_dense_permute, dim=(nw, e.J.shape[2]), inputs=[d.nefc, nj, s.perm, s.lead], outputs=[e.J])
    if s.mark is not None and route:
        wp.launch(_jtdaj_remap, dim=(nw, nj), inputs=[e.jtdaj_nblock, s.inv], outputs=[e.jtdaj_adr])
    elif s.mark is not None:
        wp.launch(_jtdaj_mark, dim=(nw, nj), inputs=[e.jtdaj_nblock, s.inv, e.jtdaj_adr, e.jtdaj_nrow],
                  outputs=[s.mark])
        wp.launch(_jtdaj_emit, dim=nw, inputs=[d.nefc, nj, s.mark], outputs=[e.jtdaj_adr, e.jtdaj_nrow])
    if int(d.naconmax) > 0 and d.contact.efc_address.shape[1] > 0:
        wp.launch(_remap_contacts, dim=(int(d.naconmax), d.contact.efc_address.shape[1]),
                  inputs=[d.nacon, int(d.naconmax), nj, d.contact.worldid, s.inv], outputs=[d.contact.efc_address])


def _sort_small(d: Any, s: _RowScratch, nj: int, route: bool, eq_type: wp.array, cinv: wp.array | None) -> None:
    """Few worlds, sparse J: the whole row sort in one block-per-world launch (``_sort_rows_small``)."""
    e, nw = d.efc, int(d.nworld)
    src, tmp = _row_cols({f: getattr(e, f) for f in _RowCols.vars}), _row_cols(s.copies)
    n_c = int(d.naconmax)
    remap_c = n_c > 0 and d.contact.efc_address.shape[1] > 0
    addr = d.contact.efc_address if remap_c else s.perm  # (unread when there are no contact slots)
    wp.launch_tiled(_sort_rows_small, dim=nw,
                    inputs=[d.nefc, nj, eq_type, s.keys, src, tmp, int(route), int(route), int(d.njmax_nnz),
                            int(route and s.mark is not None), e.jtdaj_nblock, d.nacon, n_c if remap_c else 0,
                            int(d.contact.efc_address.shape[1]) if remap_c else 0, d.contact.worldid,
                            int(cinv is not None), cinv if cinv is not None else d.nefc],
                    outputs=[s.perm, s.inv, d.overflow, e.jtdaj_adr, addr], block_dim=RANK_LANES)
    if not route:
        rebuild_sparse(d, s, nj)
        if s.mark is not None:
            wp.launch(_jtdaj_mark, dim=(nw, nj), inputs=[e.jtdaj_nblock, s.inv, e.jtdaj_adr, e.jtdaj_nrow],
                      outputs=[s.mark])
            wp.launch(_jtdaj_emit, dim=nw, inputs=[d.nefc, nj, s.mark], outputs=[e.jtdaj_adr, e.jtdaj_nrow])


def _route_on(m: Any) -> bool:
    """Alloy's ordered route is on for ``m``: its Hessian is Fixed64, so the JTDAJ block order does not matter."""
    from alloy.precision.mjw_launch import is_enabled

    return is_enabled(m)


def _eq_type(m: Any) -> wp.array:
    eq = getattr(m, "eq_type", None)
    return eq if eq is not None and eq.shape[0] > 0 else wp.zeros(1, dtype=int, device=m.opt.timestep.device)


def release(d: Any) -> None:
    _SCRATCH.pop(id(d), None)
