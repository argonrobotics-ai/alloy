# SPDX-License-Identifier: Apache-2.0
"""Materials and named mode presets for ``ModalSound``.

Material constants are typical handbook values, marked ESTIMATE, except polycarbonate. Its values are DC6's
sourced set (``alloy/tasks/dc_click_sound.py``):
- Lexan 9034 tensile modulus 345,000 psi, Poisson 0.37;
- Makrolon density 1200 kg/m³;
- tan δ from Eastman/K&S bar tests: 0.007 at 201.5 Hz ... 0.019 at 2723 Hz.
Loss factors set the decay. They vary with temperature, frequency and mounting, so treat the ESTIMATE ones as
placeholders. Compare spectra and decay, not absolute levels.

Presets (``preset(kind, dims, material, n)``; ``dims`` in m):
- ``free_beam`` (L, w, t): free-free beam along body x, bending along z (across t);
- ``free_beam_2`` (L, w, t): the same, also bending along y (across w);
- ``cantilever`` (L, w, t): clamped at x = -L/2, free at +L/2, bending along z;
- ``plate`` (a, b, h): simply supported plate in the x-y plane;
- ``dc6_tab`` and ``dc6_housing``: DC6's RJ45 latch tab (cantilever 12 x 6.6 x 1.0 mm) and plug housing (free-free
  22 x 11.7 x 6 mm, both planes), polycarbonate, ESTIMATE geometry (the stand-in of DC6's click sound);
- ``fem_box`` (L, w, t): a free box by P2 tet FEM (``fem.box_modes``), cached per (dims, material, n);
- ``npz:<path>``: a ``modes.npz`` (``ModeSet.save``; ``python -m alloy.plugins.sound.fem`` makes one from an OBJ).
"""

from __future__ import annotations

from functools import lru_cache

from alloy.plugins.sound.modes import Material, ModeSet, beam_modes, concat, plate_modes

PSI = 6894.757

MATERIALS: dict[str, Material] = {
    "steel": Material("steel", 200e9, 0.29, 7850.0, (1000.0,), (1e-3,), "ESTIMATE: typical carbon steel"),
    "aluminium": Material("aluminium", 69e9, 0.33, 2700.0, (1000.0,), (1e-3,), "ESTIMATE: typical 6061"),
    "glass": Material("glass", 72e9, 0.22, 2500.0, (1000.0,), (2e-3,), "ESTIMATE: soda-lime"),
    "abs": Material("abs", 2.2e9, 0.35, 1050.0, (1000.0,), (2e-2,), "ESTIMATE: typical ABS"),
    "pc": Material("pc", 345000 * PSI, 0.37, 1200.0, (201.5, 566.0, 1109.0, 1831.0, 2723.0),
                   (0.007, 0.007, 0.009, 0.035, 0.019),
                   "DC6 sourced: Lexan 9034 modulus/Poisson, Makrolon density, Eastman/K&S tan δ"),
}

DC6_TAB = (0.012, 0.0066, 0.001)            # ESTIMATE: DC6 stand-in tab, L x w x t (m)
DC6_HOUSING = (0.022, 0.0117, 0.006)        # ESTIMATE: DC6 stand-in housing, L x w x h (m)


def material(name: str, loss: float | None = None) -> Material:
    """A named material; ``loss`` > 0 overrides tan δ with a constant."""
    if name not in MATERIALS:
        raise ValueError(f"material {name!r}: known {sorted(MATERIALS)}")
    m = MATERIALS[name]
    if loss is not None and loss > 0.0:
        m = Material(m.name, m.youngs, m.poisson, m.density, (1000.0,), (float(loss),), m.source + "; loss override")
    return m


def preset(kind: str, dims: tuple[float, ...], mat: Material, n: int = 16, name: str = "") -> ModeSet:
    """The modes of a named preset (see the module docstring)."""
    if kind.startswith("npz:"):
        return ModeSet.load(kind[4:])
    if kind in ("dc6_tab", "dc6_housing"):
        pc = MATERIALS["pc"]
        if kind == "dc6_tab":
            return beam_modes(*DC6_TAB, pc, n, clamped=True, name=name or kind)
        L, w, h = DC6_HOUSING
        return concat([beam_modes(L, w, h, pc, n, bend="z"), beam_modes(L, h, w, pc, n, bend="y")], name or kind)
    if len(dims) != 3 or min(dims) <= 0.0:
        raise ValueError(f"preset {kind!r}: dims are 3 positive lengths in m, got {dims}")
    if kind == "free_beam":
        return beam_modes(*dims, mat, n, name=name or kind)
    if kind == "free_beam_2":
        L, w, t = dims
        return concat([beam_modes(L, w, t, mat, n, bend="z"), beam_modes(L, t, w, mat, n, bend="y")], name or kind)
    if kind == "cantilever":
        return beam_modes(*dims, mat, n, clamped=True, name=name or kind)
    if kind == "plate":
        return plate_modes(*dims, mat, n, name=name or kind)
    if kind == "fem_box":
        return _fem_box(tuple(float(d) for d in dims), mat, n)
    raise ValueError(f"preset {kind!r}: known {', '.join(PRESETS)}")


@lru_cache(maxsize=8)
def _fem_box(dims: tuple[float, float, float], mat: Material, n: int) -> ModeSet:
    from alloy.plugins.sound.fem import box_modes  # noqa: PLC0415 - scipy eigsh, seconds

    return box_modes(dims, mat, n)


PRESETS = ("free_beam", "free_beam_2", "cantilever", "plate", "fem_box", "dc6_tab", "dc6_housing", "npz:<path>")

__all__ = ["DC6_HOUSING", "DC6_TAB", "MATERIALS", "PRESETS", "material", "preset"]
