# SPDX-License-Identifier: Apache-2.0
"""Canonical contact and constraint-row order for MuJoCo Warp (P3-1 route c): Alloy bodies for the two stubs.

At mujoco_warp@86e2fae0 the CONTACTS and CONSTRAINT bits of ``DeterminismType`` call stubs:
``collision_driver._sort_contacts`` and ``constraint._sort_constraints``. ``install_order_hooks`` replaces them,
module-wide and only while they are still stubs (private names, SCR-WP5-8). The replacements run only for models
whose ``opt.deterministic`` has the bit. MuJoCo Warp still hands out contact slots and constraint rows with
``atomic_add`` counters; these functions reorder the result afterwards, at the points MuJoCo Warp calls them.

- **Contacts** (end of ``collision``): stable radix passes over every contact slot.
  - Key: (world, geom pair, ``geomcollisionid``, the emitting thread's index within the pair).
  - Tie-break: a 64-bit hash of position, distance, normal and flex ids.
  - Then every ``d.contact`` field is gathered into that order.
  - Contacts a ``contactfilter`` callback appended stay where they are: after every native contact, in slot order.
    Alloy's MixedContactBinding (``alloy/contact/mujoco_append.py``) appends the precision rows at
    ``[nacon, nacon + count)`` just before this sort and finds them later by that range (L1 speculative targets,
    precision forces). It writes ``geomcollisionid`` -1; MuJoCo Warp's collision writes >= 0. Sorting them with the
    native contacts put other worlds' contacts in that range (T09's world-index bug, coordinator ruling
    2026-11-03 ~23:15Z).
- **Constraint rows** (end of ``make_constraint``): one stable radix pass per world by (type, id).
  - Each block (one equality, dof, joint, tendon or contact) comes from one thread as consecutive rows, so its
    sub-row order survives the stable sort.
  - The type order equals MuJoCo's equality / friction / limit / contact order, so the ``ne``/``nf``/``nl`` ranges
    stay valid.
  - The row fields ``make_constraint`` writes are permuted, the sparse J storage is rebuilt in the new row order,
    the JTDAJ block list is re-sorted, and every contact's ``efc_address`` is remapped.
- **Contacts MuJoCo Warp did not produce** (``run_collision_detection`` off, as on Newton's
  ``use_mujoco_contacts=False`` path) are put in canonical order inside the constraint sort
  (``sort_imported_contacts``). The key ignores ``geomcollisionid``, which Newton never writes. ``efc_address``
  moves with its contact and contact row ids are remapped.
- **Equality rows** keep MuJoCo Warp's layout: connect, then weld, joint and tendon rows, each by equality id. The
  equality-force sensors read that layout.
- Flex rows and contacts are not covered: one flex equality emits rows with a shared (type, id) from many threads.
  A model with flexes logs a warning.

All scratch is allocated on the first call per Data (eager), so later calls are CUDA-graph safe.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Any

import warp as wp

from alloy.precision.mjw_rows import block_barrier
from alloy.precision.owned_cache import remember

log = logging.getLogger("alloy.precision")

CONTACTS, CONSTRAINT = 1, 2
_STUBS = {CONTACTS: ("alloy._vendor.mujoco_warp._src.collision_driver", "_sort_contacts"),
          CONSTRAINT: ("alloy._vendor.mujoco_warp._src.constraint", "_sort_constraints")}
_INSTALLED: dict[int, Any] = {}  # bit -> the mujoco_warp function Alloy replaced
_CONTACT_INVALID = wp.constant(wp.uint64(0xFFFFFFFFFFFFFFFF))
# Contacts a ``contactfilter`` callback appended (Alloy's MixedContactBinding writes geomcollisionid -1; MuJoCo Warp's
# collision writes >= 0): after every native contact, before the unused slots, in slot order
_CONTACT_APPENDED = wp.constant(wp.uint64(0xFFFFFFFFFFFFFFFE))


@wp.func
def _mix(h: wp.uint64, x: wp.uint64) -> wp.uint64:
    z = h ^ (x + wp.uint64(0x9E3779B97F4A7C15) + (h << wp.uint64(6)) + (h >> wp.uint64(2)))
    z = (z ^ (z >> wp.uint64(30))) * wp.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> wp.uint64(27))) * wp.uint64(0x94D049BB133111EB)
    return z ^ (z >> wp.uint64(31))


@wp.func
def _fbits(x: float) -> wp.uint64:
    return wp.uint64(wp.cast(x, wp.uint32))


@wp.func
def _ibits(x: int) -> wp.uint64:
    return wp.uint64(wp.uint32(x))


@wp.kernel
def _iota(perm: wp.array[wp.int32]):
    i = wp.tid()
    perm[i] = i


@wp.func
def _tie_base(p: wp.vec3, dist: float, f: wp.mat33) -> wp.uint64:
    """Hash of position, distance and normal (the tie key without flex ids)."""
    h = _mix(_mix(_mix(wp.uint64(0), _fbits(p[0])), _fbits(p[1])), _fbits(p[2]))
    return _mix(_mix(_mix(_mix(h, _fbits(dist)), _fbits(f[0, 0])), _fbits(f[0, 1])), _fbits(f[0, 2]))


@wp.func
def _appended(c: int, gcid: wp.array[int], use_gcid: int) -> bool:
    """Contact ``c`` was appended by a ``contactfilter`` callback, not produced by MuJoCo Warp's collision."""
    return use_gcid != 0 and gcid[c] < 0


@wp.func
def _main_key(c: int, worldid: wp.array[int], geom: wp.array[wp.vec2i], gcid: wp.array[int], use_gcid: int,
              geom_bits: int, cid_bits: int) -> wp.uint64:
    """(world, geom pair, geomcollisionid) of live contact ``c``; appended contacts all share _CONTACT_APPENDED."""
    if _appended(c, gcid, use_gcid):
        return _CONTACT_APPENDED
    g = geom[c]
    k = wp.uint64(worldid[c])
    k = (k << wp.uint64(geom_bits)) | wp.uint64(g[0] + 1)
    k = (k << wp.uint64(geom_bits)) | wp.uint64(g[1] + 1)
    mask = (wp.uint64(1) << wp.uint64(cid_bits)) - wp.uint64(1)
    cid = wp.uint64(0)
    if use_gcid != 0:  # imported (Newton) contacts never set geomcollisionid; put_data leaves it uninitialized
        cid = wp.uint64(wp.max(gcid[c], 0)) & mask
    return (k << wp.uint64(cid_bits)) | cid


@wp.kernel
def _contact_tie_keys(nacon: wp.array[int], naconmax: int, perm: wp.array[wp.int32], pos: wp.array[wp.vec3],
                      dist: wp.array[float], frame: wp.array[wp.mat33], has_flex: int, flex: wp.array[wp.vec2i],
                      elem: wp.array[wp.vec2i], vert: wp.array[wp.vec2i], gcid: wp.array[int], use_gcid: int,
                      keys: wp.array[wp.uint64]):
    i = wp.tid()
    c = perm[i]
    if c >= wp.min(nacon[0], naconmax):
        keys[i] = _CONTACT_INVALID
        return
    if _appended(c, gcid, use_gcid):
        keys[i] = wp.uint64(c)  # appended contacts keep their slot order
        return
    h = _tie_base(pos[c], dist[c], frame[c])
    if has_flex != 0:  # flex/elem/vert are allocated only for models with flexes
        h = _mix(_mix(_mix(h, _ibits(flex[c][0])), _ibits(flex[c][1])), _ibits(elem[c][0]))
        h = _mix(_mix(_mix(h, _ibits(elem[c][1])), _ibits(vert[c][0])), _ibits(vert[c][1]))
    keys[i] = h >> wp.uint64(1)  # below _CONTACT_INVALID


@wp.kernel
def _contact_main_keys(nacon: wp.array[int], naconmax: int, perm: wp.array[wp.int32], worldid: wp.array[int],
                       geom: wp.array[wp.vec2i], gcid: wp.array[int], use_gcid: int, geom_bits: int, cid_bits: int,
                       keys: wp.array[wp.uint64]):
    i = wp.tid()
    c = perm[i]
    if c >= wp.min(nacon[0], naconmax):
        keys[i] = _CONTACT_INVALID
        return
    keys[i] = _main_key(c, worldid, geom, gcid, use_gcid, geom_bits, cid_bits)


@wp.kernel
def _gather2d(nacon: wp.array[int], naconmax: int, perm: wp.array[wp.int32], src: wp.array2d[int],
              dst: wp.array2d[int]):
    i, j = wp.tid()
    if i < wp.min(nacon[0], naconmax):
        dst[i, j] = src[perm[i], j]
    else:
        dst[i, j] = src[i, j]


@wp.kernel
def _invert(nacon: wp.array[int], naconmax: int, perm: wp.array[wp.int32], inv: wp.array[int]):
    i = wp.tid()
    if i < wp.min(nacon[0], naconmax):
        inv[perm[i]] = i


def _bitlen(n: int) -> int:
    return max(int(n).bit_length(), 1)


@dataclass
class _ContactScratch:
    keys: wp.array
    perm: wp.array
    copies: dict[str, wp.array]
    address: wp.array  # efc_address copy (imported contacts are sorted after make_constraint)
    inv: wp.array  # old slot -> new slot
    geom_bits: int
    cid_bits: int
    end_bit: int


_CONTACT_FIELDS = ("dist", "pos", "frame", "includemargin", "friction", "solref", "solreffriction", "solimp", "dim",
                   "geom", "flex", "elem", "vert", "worldid", "type", "geomcollisionid", "adhesion")
_SCRATCH: dict[tuple[int, str], Any] = {}


def _contact_fields(d: Any) -> tuple[str, ...]:
    """Contact fields allocated per contact slot (flex/elem/vert are empty without flexes)."""
    n = int(d.naconmax)
    return tuple(f for f in _CONTACT_FIELDS if getattr(d.contact, f).shape[0] == n)


def _contact_scratch(m: Any, d: Any) -> _ContactScratch:
    key = (id(d), "contacts")
    s = _SCRATCH.get(key)
    if s is not None:
        return s
    n = int(d.naconmax)
    geom_bits = _bitlen(int(m.ngeom) + 1)
    world_bits = _bitlen(int(d.nworld))
    cid_bits = 64 - world_bits - 2 * geom_bits
    if cid_bits < 8:
        raise ValueError(f"contact key does not fit 64 bits ({d.nworld} worlds, {m.ngeom} geoms)")
    cid_bits = min(cid_bits, 16)
    copies = {f: wp.empty_like(getattr(d.contact, f)) for f in _contact_fields(d)}
    s = _ContactScratch(wp.empty(2 * n, dtype=wp.uint64, device=d.nacon.device),
                        wp.empty(2 * n, dtype=wp.int32, device=d.nacon.device), copies,
                        wp.empty_like(d.contact.efc_address), wp.zeros(max(n, 1), dtype=int, device=d.nacon.device),
                        geom_bits, cid_bits,
                        world_bits + 2 * geom_bits + cid_bits)
    if int(m.nflex) > 0:
        log.warning("mjw_order: model has flexes; flex contacts and rows are not guaranteed canonical")
    return remember(_SCRATCH, key, d, s)  # dropped when d is freed: a reused id never finds it


@wp.kernel
def _gather(nacon: wp.array[int], naconmax: int, perm: wp.array[wp.int32], src: wp.array(dtype=Any),
            dst: wp.array(dtype=Any)):
    i = wp.tid()
    if i < wp.min(nacon[0], naconmax):  # slots past nacon are neither read nor written: they may never have been
        dst[i] = src[perm[i]]           # written, and they keep their contents


@wp.kernel
def _copy_used(nacon: wp.array[int], naconmax: int, src: wp.array(dtype=Any), dst: wp.array(dtype=Any)):
    i = wp.tid()
    if i < wp.min(nacon[0], naconmax):
        dst[i] = src[i]


def sort_contacts(m: Any, d: Any) -> None:
    """Canonical order of ``d.contact[:nacon]`` (MuJoCo Warp calls this when ``opt.deterministic & CONTACTS``)."""
    if not int(d.naconmax):
        return
    s = _contact_scratch(m, d)
    if _fused_small(d, s):
        _sort_small(d, s, use_gcid=True, move_addr=False)
        return
    _order_contacts(m, d, use_gcid=True)
    _gather_contacts(d, s)


def sort_imported_contacts(m: Any, d: Any) -> wp.array:
    """Canonical order of contacts that MuJoCo Warp's collision did not produce (``run_collision_detection`` off,
    e.g. Newton's ``use_mujoco_contacts=False``). Runs inside the constraint sort, after make_constraint, so
    ``efc_address`` moves with its contact. Returns old slot -> new slot for remapping contact row ids."""
    s = _contact_scratch(m, d)
    n = int(d.naconmax)
    if n == 0:
        return s.inv
    if _fused_small(d, s):
        _sort_small(d, s, use_gcid=False, move_addr=True)
        return s.inv
    _order_contacts(m, d, use_gcid=False)
    _gather_contacts(d, s)
    wp.launch(_gather2d, dim=s.address.shape, inputs=[d.nacon, n, s.perm, d.contact.efc_address], outputs=[s.address])
    wp.copy(d.contact.efc_address, s.address)
    wp.launch(_invert, dim=n, inputs=[d.nacon, n, s.perm], outputs=[s.inv])
    return s.inv


SMALL_CONTACTS = 4096  # at most this many contact slots: one block ranks them (fewer graph nodes than radix)
RANK_LANES = 256


@wp.func
def _contact_rank(main: wp.array[wp.uint64], tie: wp.array[wp.uint64], n: int, i: int) -> int:
    """Number of live contacts with a smaller (main key, tie key, slot) than slot ``i``."""
    mi, ti = main[i], tie[i]
    rank = int(0)
    for j in range(n):
        mj = main[j]
        if mj < mi or (mj == mi and (tie[j] < ti or (tie[j] == ti and j < i))):
            rank += 1
    return rank


@wp.kernel
def _rank_contacts(nacon: wp.array[int], naconmax: int, main: wp.array[wp.uint64], tie: wp.array[wp.uint64],
                   perm: wp.array[wp.int32]):
    """One block: a contact's new slot is the number of contacts with a smaller (main key, tie key, slot), the
    order the two stable radix passes give. Unused slots keep their place."""
    _block, lane = wp.tid()  # a tiled launch: (block, lane), one block here
    n = wp.min(nacon[0], naconmax)
    for i in range(lane, naconmax, wp.block_dim()):
        if i >= n:
            perm[i] = wp.int32(i)
            continue
        perm[_contact_rank(main, tie, n, i)] = wp.int32(i)



def _order_contacts(m: Any, d: Any, use_gcid: bool) -> None:
    """Leaves the canonical permutation (new slot -> old slot) in the scratch ``perm``."""
    n = int(d.naconmax)
    s = _contact_scratch(m, d)
    c = d.contact
    wp.launch(_iota, dim=n, inputs=[s.perm])
    has_flex = int(c.flex.shape[0] == n and c.elem.shape[0] == n and c.vert.shape[0] == n)
    if n <= SMALL_CONTACTS:  # keys per slot (identity perm), then one ranking launch instead of two radix sorts
        tie, main = s.keys[:n], s.keys[n:2 * n]
        wp.launch(_contact_tie_keys, dim=n, inputs=[d.nacon, n, s.perm, c.pos, c.dist, c.frame, has_flex, c.flex,
                                                    c.elem, c.vert, c.geomcollisionid, int(use_gcid)], outputs=[tie])
        wp.launch(_contact_main_keys, dim=n, inputs=[d.nacon, n, s.perm, c.worldid, c.geom, c.geomcollisionid,
                                                     int(use_gcid), s.geom_bits, s.cid_bits], outputs=[main])
        wp.launch_tiled(_rank_contacts, dim=1, inputs=[d.nacon, n, main, tie], outputs=[s.perm],
                        block_dim=RANK_LANES)
        return
    wp.launch(_contact_tie_keys, dim=n, inputs=[d.nacon, n, s.perm, c.pos, c.dist, c.frame, has_flex, c.flex, c.elem,
                                                c.vert, c.geomcollisionid, int(use_gcid)], outputs=[s.keys])
    wp.utils.radix_sort_pairs(s.keys, s.perm, n)
    wp.launch(_contact_main_keys, dim=n, inputs=[d.nacon, n, s.perm, c.worldid, c.geom, c.geomcollisionid,
                                                 int(use_gcid), s.geom_bits, s.cid_bits], outputs=[s.keys])
    wp.utils.radix_sort_pairs(s.keys, s.perm, n)


_VEC5 = wp.types.vector(length=5, dtype=wp.float32)


@wp.struct
class _ContactCols:
    """The contact fields of a model without flexes, for one fused gather (fewer graph nodes at small N)."""
    dist: wp.array(dtype=float)
    pos: wp.array(dtype=wp.vec3)
    frame: wp.array(dtype=wp.mat33)
    includemargin: wp.array(dtype=float)
    friction: wp.array(dtype=_VEC5)
    solref: wp.array(dtype=wp.vec2)
    solreffriction: wp.array(dtype=wp.vec2)
    solimp: wp.array(dtype=_VEC5)
    dim: wp.array(dtype=int)
    geom: wp.array(dtype=wp.vec2i)
    worldid: wp.array(dtype=int)
    type: wp.array(dtype=int)
    geomcollisionid: wp.array(dtype=int)
    adhesion: wp.array(dtype=float)


_FUSED = tuple(_ContactCols.vars)


@wp.func
def _move_contact(src: _ContactCols, i: int, dst: _ContactCols, j: int):
    dst.dist[j] = src.dist[i]
    dst.pos[j] = src.pos[i]
    dst.frame[j] = src.frame[i]
    dst.includemargin[j] = src.includemargin[i]
    dst.friction[j] = src.friction[i]
    dst.solref[j] = src.solref[i]
    dst.solreffriction[j] = src.solreffriction[i]
    dst.solimp[j] = src.solimp[i]
    dst.dim[j] = src.dim[i]
    dst.geom[j] = src.geom[i]
    dst.worldid[j] = src.worldid[i]
    dst.type[j] = src.type[i]
    dst.geomcollisionid[j] = src.geomcollisionid[i]
    dst.adhesion[j] = src.adhesion[i]


@wp.kernel
def _gather_fused(nacon: wp.array[int], naconmax: int, perm: wp.array[wp.int32], src: _ContactCols,
                  dst: _ContactCols):
    i = wp.tid()
    if i < wp.min(nacon[0], naconmax):  # as in _gather: slots past nacon are not touched
        _move_contact(src, int(perm[i]), dst, i)


@wp.kernel
def _copy_fused(nacon: wp.array[int], naconmax: int, src: _ContactCols, dst: _ContactCols):
    i = wp.tid()
    if i < wp.min(nacon[0], naconmax):
        _move_contact(src, i, dst, i)


def _cols(arrays: dict) -> _ContactCols:
    c = _ContactCols()
    for f in _FUSED:
        setattr(c, f, arrays[f])
    return c


@wp.kernel
def _sort_contacts_small(nacon: wp.array[int], naconmax: int, use_gcid: int, geom_bits: int, cid_bits: int,
                         src: _ContactCols, tmp: _ContactCols, nslot: int, addr: wp.array2d[int],
                         addr_tmp: wp.array2d[int], tie: wp.array[wp.uint64], main: wp.array[wp.uint64],
                         perm: wp.array[wp.int32], inv: wp.array[int]):
    """One block, the whole contact sort in one launch (few slots, no flex; graph nodes dominate at N = 1):
    keys -> rank (the order of the two stable radix passes) -> gather into the scratch -> copy back. With ``nslot``
    the efc_address rows move with their contact (imported contacts). ``inv``: old slot -> new slot."""
    _block, lane = wp.tid()
    lanes = wp.block_dim()
    n = wp.min(nacon[0], naconmax)
    for i in range(lane, n, lanes):
        tie[i] = _tie_base(src.pos[i], src.dist[i], src.frame[i]) >> wp.uint64(1)  # below _CONTACT_INVALID
        if _appended(i, src.geomcollisionid, use_gcid):
            tie[i] = wp.uint64(i)  # appended contacts keep their slot order
        main[i] = _main_key(i, src.worldid, src.geom, src.geomcollisionid, use_gcid, geom_bits, cid_bits)
    block_barrier()
    for i in range(lane, naconmax, lanes):
        if i < n:
            rank = _contact_rank(main, tie, n, i)
            perm[rank] = wp.int32(i)
            inv[i] = rank
        else:
            perm[i] = wp.int32(i)  # unused slots keep their place
    block_barrier()
    for i in range(lane, n, lanes):
        j = int(perm[i])
        _move_contact(src, j, tmp, i)
        for k in range(nslot):
            addr_tmp[i, k] = addr[j, k]
    block_barrier()
    for i in range(lane, n, lanes):
        _move_contact(tmp, i, src, i)
        for k in range(nslot):
            addr[i, k] = addr_tmp[i, k]


def _fused_small(d: Any, s: _ContactScratch) -> bool:
    return int(d.naconmax) <= SMALL_CONTACTS and set(s.copies) == set(_FUSED)


def _sort_small(d: Any, s: _ContactScratch, use_gcid: bool, move_addr: bool) -> None:
    n, c = int(d.naconmax), d.contact
    nslot = int(c.efc_address.shape[1]) if move_addr else 0
    wp.launch_tiled(_sort_contacts_small, dim=1,
                    inputs=[d.nacon, n, int(use_gcid), s.geom_bits, s.cid_bits,
                            _cols({f: getattr(c, f) for f in _FUSED}), _cols(s.copies), nslot, c.efc_address,
                            s.address],
                    outputs=[s.keys[:n], s.keys[n:2 * n], s.perm, s.inv], block_dim=RANK_LANES)


def _gather_contacts(d: Any, s: _ContactScratch) -> None:
    n, c = int(d.naconmax), d.contact
    if set(s.copies) == set(_FUSED):  # no flex fields: two launches instead of two per field
        src, tmp = _cols({f: getattr(c, f) for f in _FUSED}), _cols(s.copies)
        wp.launch(_gather_fused, dim=n, inputs=[d.nacon, n, s.perm, src], outputs=[tmp])
        wp.launch(_copy_fused, dim=n, inputs=[d.nacon, n, tmp], outputs=[src])
        return
    for f, dst in s.copies.items():
        wp.launch(_gather, dim=n, inputs=[d.nacon, n, s.perm, getattr(c, f)], outputs=[dst])
    for f, src in s.copies.items():
        wp.launch(_copy_used, dim=n, inputs=[d.nacon, n, src], outputs=[getattr(c, f)])


def install_order_hooks() -> int:
    """Replace the MuJoCo Warp sort stubs with Alloy's (idempotent); returns the bits Alloy now implements."""
    from alloy.precision.mjw_det import _is_noop
    from alloy.precision.mjw_rows import sort_constraints

    impl = {CONTACTS: sort_contacts, CONSTRAINT: sort_constraints}
    done = 0
    for bit, (module, name) in _STUBS.items():
        mod = importlib.import_module(module)
        current = getattr(mod, name, None)
        if current is impl[bit]:
            done |= bit
            continue
        if current is None or not _is_noop(current):
            log.info("mjw_order: mujoco_warp implements %s.%s; Alloy's is not installed", module, name)
            continue
        _INSTALLED[bit] = current
        setattr(mod, name, impl[bit])
        done |= bit
    if done:
        log.info("mjw_order: Alloy contact/constraint ordering installed for DeterminismType bits %d", done)
    return done


def uninstall_order_hooks() -> None:
    """Put the original MuJoCo Warp functions back (tests)."""
    for bit, original in list(_INSTALLED.items()):
        module, name = _STUBS[bit]
        setattr(importlib.import_module(module), name, original)
        del _INSTALLED[bit]


def release(d: Any) -> None:
    """Drop the scratch held for one MuJoCo Warp Data."""
    from alloy.precision import mjw_rows

    for key in [k for k in _SCRATCH if k[0] == id(d)]:
        del _SCRATCH[key]
    mjw_rows.release(d)


def original(bit: int) -> Any:
    """The mujoco_warp function behind ``bit``, also when Alloy's replacement is installed."""
    if bit in _INSTALLED:
        return _INSTALLED[bit]
    module, name = _STUBS[bit]
    return getattr(importlib.import_module(module), name, None)


__all__ = ["install_order_hooks", "original", "release", "sort_contacts", "sort_imported_contacts",
           "uninstall_order_hooks"]
